// Copyright (c) 2026 Nicolas Innocenti
// SPDX-License-Identifier: MIT
// mawmetal: minimal absent words of DNA sequences, GPU-accelerated on Apple Silicon.
//
// Tiers (W is the middle of a MAW aWb, kRep the repeat split, 31 by default):
//   census  |W| <= 6          which words of length <= 8 occur (GPU), then bitmaps (CPU)
//   tier 1  6 < |W| < kRep    GPU radix sort of 64-bit suffix keys, CPU stack scan of the sorted keys
//   deep    kRep <= |W| < D   groups sharing kRep bases, sorted further from the text (CPU)
//   tier 3  |W| >= D          every occurrence of such a W lies inside a run of repeated D-mers
//                            (marked by the deep stage), so a suffix array of those runs suffices
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <deque>
#include <memory>
#include <stdexcept>
#include <thread>

#include "gpu.h"
#include "maw.h"

namespace {

// ---- per-text pipeline ------------------------------------------------------------------------------
struct Timings {
    double init = 0, census = 0, extract = 0, sort = 0, tier1 = 0, firstBatch = 0, reserve = 0, scan = 0, busy = 0;
    double cpu1 = 0, cpuDeep = 0;  // thread CPU time in tier 1 and in the deep stage
    double residue = 0, sais = 0, scan3 = 0;
    uint64_t bases = 0, residueBases = 0, batches = 0, chunks = 0;
    int prefix = 0;  // largest region prefix length used
};

// Debug aid (MAWMETAL_CHECK_SORT=1): verifies a sorted batch against the text.
void checkSorted(const Text& t, const uint64_t* keys, const uint32_t* vals, const uint8_t* hi, const uint64_t* counts,
                 uint32_t nr, uint32_t lo, int m, int S) {
    const int low = 64 - 2 * S;
    uint64_t off = 0, bad = 0;
    for (uint32_t r = 0; r < nr; ++r) {
        for (uint64_t i = off; i < off + counts[r]; ++i) {
            uint64_t k = keys[i];
            if (vals) {  // the key must describe its own position
                uint64_t p = vals[i] | (hi ? (uint64_t)hi[i] << 32 : 0);
                uint32_t prefix = 0;
                for (int j = 0; j < m; ++j) prefix = prefix * 4 + t.base(p + j);
                uint64_t len = 0;
                while (len < (uint64_t)m + 28 && p + len < t.n && (len == 0 || !t.segStart(p + len))) ++len;
                uint64_t chars = 0;
                for (int j = 0; j < 28; ++j) chars = (chars << 2) | (m + j < (int)len ? t.base(p + m + j) : 0);
                uint64_t want = (chars << 8) | ((len - m) << 3) | (t.segStart(p) ? 4 : t.base(p - 1));
                if (prefix != lo + r || want != k) {
                    if (bad++ < 5) fprintf(stderr, "check: region %u (want %u) entry %llu pos %llu key %016llx want %016llx\n", prefix,
                                           lo + r, (unsigned long long)(i - off), (unsigned long long)p,
                                           (unsigned long long)k, (unsigned long long)want);
                }
            }
            if (i > off) {
                uint64_t a = keys[i - 1];
                bool ok = (a >> low) < (k >> low) ||
                          ((a >> low) == (k >> low) && std::min<uint64_t>((a >> 3) & 31, S) <= std::min<uint64_t>((k >> 3) & 31, S));
                if (!ok && bad++ < 5)
                    fprintf(stderr, "check: region %u order broken at %llu: %016llx then %016llx\n", lo + r,
                            (unsigned long long)(i - off), (unsigned long long)a, (unsigned long long)k);
            }
        }
        off += counts[r];
    }
    if (bad) fprintf(stderr, "check: %llu problems\n", (unsigned long long)bad);
}

void processText(Text& t, const Options& o, Gpu& gpu, WorkPool& pool, Sink& sink, Timings& tm) {
    if (t.n == 0) return;
    if (t.n >= (1ull << 36) - (1ull << 20)) throw std::runtime_error("text too long (limit ~6.8e10 bases per unit)");
    const bool wide = o.wide || t.n >= (1ull << 32) - 64;  // positions need bits 32..39 as well
    tm.bases += t.n;
    const int threads = pool.size();
    // Regions of 5 bases (1024 of them) on texts of a gigabase or more: the radix sort scatters within
    // a region, and regions of 4 bases there outgrow the GPU's caches (GRCh38 on one strand: sort 7.8 s
    // with 4 bases, 4.1 s with 5). Below that it makes no difference. m may also become 5 after the
    // census, see below.
    int m = o.m ? o.m : t.n >= (1ull << 30) ? 5 : 4;
    const int kRep = o.kRep ? o.kRep : std::min(31, m + 28);
    if (m < 1 || m > 5) throw std::runtime_error("--prefix must be in 1..5");
    if (kRep < m + 1 || kRep > m + 28) throw std::runtime_error("--split must be in [prefix+1, prefix+28]");
    const LenRange rg{std::max(2, o.kmin), o.Kmax};

    // Depth ranges (|W| of a MAW aWb): census [0, dLo1), tier 1 [dLo1, kRep), deep stage [kRep, D),
    // libsais tier [D, ...).
    const int dHiB = std::min(kCensusLen - 2, kRep - 1);
    const int dLo1 = std::max(dHiB + 1, rg.kmin - 2);
    const int dHi1 = (int)std::min<int64_t>(kRep - 1, rg.Kmax - 2);
    const bool need1 = dLo1 <= dHi1;
    const bool needDeep = rg.Kmax - 2 >= kRep;
    const int64_t D = needDeep ? std::min<int64_t>(std::max<int64_t>(o.deep, kRep), rg.Kmax - 1) : 0;
    const bool need3 = needDeep && rg.Kmax - 2 >= D;
    // Tier 1 can run on the GPU when only counts are wanted (the CPU spells the MAWs for --list).
    const bool gpuT1 = need1 && sink.countOnly() && o.tier1 != 1 && dHi1 - dLo1 + 1 <= 26;
    // Bases sorted after the prefix: enough to separate groups at kRep, or for tier 1 alone.
    auto sortedBases = [&] { return needDeep ? kRep - m : need1 ? dHi1 + 1 - m : 1; };
    int S = sortedBases();
    // The slots' keys (+ positions) plus shared temporaries must fit in a batch of memCap entries.
    const uint64_t perEntry = (Gpu::kSlots + 1) * (needDeep ? (wide ? 13 : 12) : 8) + 1;
    const uint64_t memCap = std::min<uint64_t>({gpu.workingSetBytes() * 2 / 5 / perEntry, gpu.maxBufferBytes() / 8, 3u << 30});

    gpu.setText(t);
    std::vector<uint32_t> counts, shortCounts, presence;
    double gs;
    gpu.census(m, S, counts, shortCounts, presence, gs);
    tm.census += gs;
    // Region sizes: positions with at least m valid bases, by their first m bases.
    auto regionSizes = [&] {
        const size_t nReg = 1ull << (2 * m), offM = ((1ull << (2 * m)) - 4) / 3, offM1 = ((1ull << (2 * (m + 1))) - 4) / 3;
        std::vector<uint64_t> h(nReg);
        for (size_t u = 0; u < nReg; ++u) {
            h[u] = counts[offM + u];
            for (int c = 0; c < 4; ++c) h[u] += counts[offM1 + 4 * u + c];
        }
        return h;
    };
    // A region is never split across batches, so every batch slot is as large as the largest region.
    // When that exceeds the memory budget (or --batch), as AAAA does on a 24 Gbp text, one more prefix
    // base splits each region in four; the census is then run again for it.
    if (!o.m && m < 5 && kRep >= m + 2 && (need1 || needDeep)) {
        const std::vector<uint64_t> h = regionSizes();
        if (*std::max_element(h.begin(), h.end()) > (o.batchCap ? o.batchCap : memCap)) {
            ++m;
            S = sortedBases();
            gpu.census(m, S, counts, shortCounts, presence, gs);
            tm.census += gs;
        }
    }
    tm.prefix = std::max(tm.prefix, m);
    {
        ChunkOut co;
        bitmapTier(presence, LenRange{rg.kmin, std::min<int64_t>(rg.Kmax, dHiB + 2)}, sink.countOnly(), co);
        sink.put(co);
    }
    if (!need1 && !needDeep) return gpu.releaseText();

    const std::vector<uint64_t> hist = regionSizes();
    const std::vector<uint64_t> shortHist(shortCounts.begin(), shortCounts.end());
    const size_t nReg = hist.size();
    uint64_t totalEntries = 0;
    for (uint64_t h : hist) totalEntries += h;
    uint64_t cap = o.batchCap;
    if (!cap) {
        // Within the memory budget, about 16 batches: the GPU sorts one while the CPU scans another,
        // and small buffers fault in fast. --list keeps the CPU busier than the GPU, so it takes about
        // 64 smaller batches: the extra passes over the text cost no time, and the buffers shrink by as much.
        cap = std::min<uint64_t>(memCap, sink.countOnly() ? std::max<uint64_t>(totalEntries / 16, 16u << 20)
                                                          : std::max<uint64_t>(totalEntries / 64, 4u << 20));
    }
    // The first batches ramp up from a quarter of the cap, doubling, so the CPU starts early and
    // the GPU (whose cost per batch includes a pass over the whole text) gets ahead of it.
    const uint64_t firstCap = o.firstCap ? o.firstCap : std::min<uint64_t>(cap, std::max<uint64_t>(cap / 4, 4u << 20));

    struct Batch { uint32_t lo, hi; uint64_t entries; };
    std::vector<Batch> batches;
    uint64_t maxEntries = 0, maxTiles = 0;
    for (uint32_t u = 0; u < nReg;) {
        const uint64_t c = std::min<uint64_t>(cap, firstCap << std::min<size_t>(batches.size(), 20));
        Batch b{u, u, 0};
        // A region is never split across batches.
        while (b.hi < nReg && b.hi - b.lo < 1024 && (b.hi == b.lo || b.entries + hist[b.hi] <= c)) b.entries += hist[b.hi++];
        maxEntries = std::max(maxEntries, b.entries);
        maxTiles = std::max(maxTiles, Gpu::tilesFor(&hist[b.lo], b.hi - b.lo));
        if (b.entries) batches.push_back(b);
        u = b.hi;
    }
    double r0 = nowSec();
    gpu.reserveBatches(maxEntries, maxTiles, needDeep, wide, gpuT1);
    tm.reserve += nowSec() - r0;
    // calloc maps zero pages lazily: marks are sparse.
    std::unique_ptr<uint32_t, decltype(&free)> marks(needDeep ? (uint32_t*)calloc(t.n / 32 + 8, 4) : nullptr, &free);
    if (needDeep && !marks) throw std::bad_alloc();

    std::vector<char> t1Gpu(batches.size(), 0);  // per batch: tier 1 on the GPU
    auto spec = [&](size_t bi) {
        const Batch& b = batches[bi];
        BatchSpec s;
        s.m = m, s.S = S, s.lo = b.lo, s.hi = b.hi;
        s.counts = &hist[b.lo], s.shortCounts = &shortHist[b.lo];
        s.withVals = needDeep;
        s.wide = wide;
        s.tier1 = t1Gpu[bi], s.dLo = dLo1, s.dHi = dHi1, s.kRep = kRep;
        return s;
    };
    const DeepParams dp{m, kRep, D, std::max<int64_t>(kRep, rg.kmin - 2), std::min<int64_t>(D - 1, rg.Kmax - 2),
                        sink.countOnly(), marks.get()};
    // Chunks may only be cut where no reported node straddles the cut.
    auto cutBelow = [&](bool t1OnGpu) { return std::min(need1 && !t1OnGpu ? dLo1 : INT32_MAX, needDeep ? kRep : INT32_MAX); };
    // --list: smaller chunks, since a finished chunk's output waits in memory for its turn.
    const size_t chunkTarget = o.chunk ? o.chunk : sink.countOnly() ? (1u << 18) : (1u << 16);

    // The pipeline. Batches rotate through the GPU slots. Workers take chunks in order and move
    // from one batch into the next without waiting; batch b + kSlots is submitted into b's slot
    // once every chunk of b is done. The main thread submits and plans batches and, for --list, writes
    // the chunks' outputs in order. The chunk next in line writes its own output as it goes, so a
    // long one (a frequent 7-mer cannot be cut) does not pile up.
    struct Work {
        T1Chunk c;
        size_t batch;
        const uint32_t* gbits;  // tier 1 on the GPU: the batch's group bits, from entry off on (else null)
        size_t off;
    };
    const size_t B = batches.size();
    const bool ordered = !sink.countOnly();
    // Workers run ahead of the writer while the finished chunks waiting to be written hold less
    // than pendingCap bytes, and always within threads chunks of it. A chunk that takes long (a large
    // repeat group) holds back the ones after it, the longer on larger genomes, so the budget grows
    // with the batches: one batch slot (at least 128 MB), which adds at most a quarter to their memory
    // (GRCh38 with both strands: 49 s instead of 60 s with 128 MB). A written buffer is kept for reuse
    // unless it grew past spareCap.
    const uint64_t slotBytes = maxEntries * (needDeep ? (wide ? 13 : 12) : 8);
    const uint64_t pendingCap = o.pending ? o.pending : std::max<uint64_t>(128u << 20, slotBytes), spareCap = 4u << 20;
    // Metal may run a completion handler after waitUntilCompleted returns, so what the handlers
    // touch is shared with them rather than living in this frame.
    struct Sync {
        std::mutex mu;
        std::condition_variable cvWork, cvMain;
        std::vector<char> gpuDone;
    };
    auto sy = std::make_shared<Sync>();
    sy->gpuDone.assign(B, 0);
    std::mutex& mu = sy->mu;
    std::condition_variable &cvWork = sy->cvWork, &cvMain = sy->cvMain;
    std::vector<char>& gpuDone = sy->gpuDone;
    std::deque<Work> work;
    std::deque<ChunkOut> outs;  // --list: per chunk, written in order
    std::deque<char> outDone;
    std::vector<OutBuf> spare;  // written buffers, reused
    std::vector<ChunkOut> acc(threads);  // counts only: per worker
    LenCounts gpuCounts;                 // tier 1 on the GPU
    std::vector<size_t> left(B, 0);

    size_t claimed = 0, finished = 0, written = 0;
    uint64_t pending = 0;  // capacity of the finished chunks' buffers not yet written
    bool closed = false, failed = false;
    std::exception_ptr failure;
    std::atomic<uint64_t> busyNs{0}, cpu1Ns{0}, cpuDeepNs{0};
    auto cpuNs = [] { return clock_gettime_nsec_np(CLOCK_THREAD_CPUTIME_ID); };

    auto fail = [&](std::exception_ptr e) {  // with mu held
        if (!failure) failure = e;
        failed = true;
        cvWork.notify_all();
        cvMain.notify_all();
    };
    auto worker = [&](int w) {
        DeepWork deep(t, dp);
        Groups groups;
        size_t idx = 0;
        // Once every earlier chunk is written, and until this one is finished, nobody else writes.
        const OutBuf::Spill spill = [&](const char* s, size_t n) {
            {
                std::lock_guard<std::mutex> l(mu);
                if (written != idx) return false;
            }
            sink.raw(s, n);
            return true;
        };
        for (;;) {
            Work job;
            ChunkOut* out = &acc[w];
            {
                std::unique_lock<std::mutex> l(mu);
                cvWork.wait(l, [&] {
                    return failed ||
                           (claimed < work.size() && (!ordered || claimed < written + threads || pending < pendingCap)) ||
                           (closed && claimed == work.size());
                });
                if (failed || claimed == work.size()) return;
                idx = claimed++;
                job = work[idx];
                if (ordered) {
                    out = &outs[idx];
                    if (!spare.empty()) out->text = std::move(spare.back()), spare.pop_back();
                    out->text.spill(&spill, sink.spillBytes);
                }
            }
            const T1Chunk& c = job.c;
            const double t0 = nowSec();
            try {
                groups.clear();
                const uint64_t c0 = cpuNs();
                if (job.gbits) groupsFromBits(job.gbits, job.off, c.n, groups);
                else if (need1) scanTier1(c, m, dLo1, dHi1, kRep, sink.countOnly(), *out, needDeep ? &groups : nullptr);
                else findGroups(c, m, kRep, groups);
                const uint64_t c1 = cpuNs();
                if (needDeep) deep.run(c, groups, *out);
                cpu1Ns += c1 - c0, cpuDeepNs += cpuNs() - c1;
            } catch (...) {
                out->text.spill(nullptr);
                std::lock_guard<std::mutex> l(mu);
                fail(std::current_exception());
                return;
            }
            out->text.spill(nullptr);
            busyNs += (uint64_t)((nowSec() - t0) * 1e9);
            {
                std::lock_guard<std::mutex> l(mu);
                --left[job.batch];
                ++finished;
                if (ordered) outDone[idx] = 1, pending += out->text.capacity();
            }
            cvMain.notify_one();
        }
    };
    // Waits until pred holds (or a worker failed), writing finished chunks in order meanwhile.
    auto pump = [&](auto pred) {
        std::unique_lock<std::mutex> l(mu);
        for (;;) {
            while (!failed && ordered && written < outDone.size() && outDone[written]) {
                ChunkOut& c = outs[written];
                l.unlock();
                try {
                    sink.put(c);
                } catch (...) {
                    l.lock();
                    fail(std::current_exception());
                    break;
                }
                l.lock();
                pending -= c.text.capacity();
                c.counts = LenCounts();
                c.text.clear();
                if (c.text.capacity() <= spareCap && spare.size() < (size_t)threads) spare.push_back(std::move(c.text));
                else c.text = OutBuf();
                ++written;
                cvWork.notify_all();
            }
            if (failed) return false;
            if (pred()) return true;
            cvMain.wait(l);
        }
    };
    auto submit = [&](size_t b) {
        // Tier 1 goes to the GPU while the CPU has a backlog, and to the CPU when it waits for the
        // GPU; the first batch's to the CPU, which waits for it anyway.
        if (gpuT1) {
            std::lock_guard<std::mutex> l(mu);
            t1Gpu[b] = o.tier1 == 2 || (b > 0 && claimed < work.size());
        }
        gpu.submitBatch(b % Gpu::kSlots, spec(b), [sy, b] {
            std::lock_guard<std::mutex> l(sy->mu);
            sy->gpuDone[b] = 1;
            sy->cvMain.notify_all();
        });
    };
    // After a failure, no batch may still be sorting into buffers that are about to be reused.
    auto drainGpu = [&] {
        for (int s = 0; s < Gpu::kSlots; ++s) {
            GpuBatchTimes bt;
            try {
                gpu.waitBatch(s, bt);
            } catch (...) {
            }
        }
    };

    const double p0 = nowSec();
    pool.start(worker);
    try {
        // Event loop: a batch goes into a slot as soon as every chunk of the batch the slot held is
        // done (so the GPU always has the next batch queued), and a sorted batch is published as
        // soon as it is ready.
        std::vector<char> published(B, 0);
        size_t nextSubmit = std::min<size_t>(B, Gpu::kSlots), nextPublish = 0;
        for (size_t b = 0; b < nextSubmit; ++b) submit(b);
        auto canSubmit = [&] {
            const size_t prev = nextSubmit - Gpu::kSlots;  // the batch whose slot it takes
            return nextSubmit < B && published[prev] && left[prev] == 0;
        };
        const double w0 = nowSec();
        while (nextPublish < B) {
            if (!pump([&] { return gpuDone[nextPublish] != 0 || canSubmit(); })) break;
            bool sub;
            {
                std::lock_guard<std::mutex> l(mu);
                sub = canSubmit();
            }
            if (sub) {
                submit(nextSubmit++);
                continue;
            }
            const size_t b = nextPublish++;
            if (b == 0) tm.firstBatch += nowSec() - w0;
            GpuBatchTimes bt;
            gpu.waitBatch(b % Gpu::kSlots, bt);
            tm.extract += bt.extract, tm.sort += bt.sort, tm.tier1 += bt.tier1;
            ++tm.batches;
            if (t1Gpu[b]) {
                const uint32_t* c = gpu.tier1Counts(b % Gpu::kSlots);
                for (int j = 0; j <= dHi1 - dLo1; ++j) gpuCounts.add(dLo1 + j + 2, c[j]);
            }

            const Batch& bb = batches[b];
            const uint64_t* keys = gpu.keys(b % Gpu::kSlots);
            const uint32_t* vals = gpu.vals(b % Gpu::kSlots);
            const uint8_t* hi = gpu.hi(b % Gpu::kSlots);
            if (getenv("MAWMETAL_CHECK_SORT")) checkSorted(t, keys, vals, hi, &hist[bb.lo], bb.hi - bb.lo, bb.lo, m, S);
            std::vector<T1Chunk> chunks;
            uint64_t off = 0;
            for (uint32_t u = bb.lo; u < bb.hi && (!t1Gpu[b] || needDeep); ++u) {
                if (hist[u])
                    planTier1Chunks(T1Chunk{keys + off, vals ? vals + off : nullptr, hi ? hi + off : nullptr, hist[u], u}, m,
                                    cutBelow(t1Gpu[b]), chunkTarget, chunks);
                off += hist[u];
            }
            const uint32_t* gbits = t1Gpu[b] ? gpu.groupBits(b % Gpu::kSlots) : nullptr;
            {
                std::lock_guard<std::mutex> l(mu);
                for (const T1Chunk& c : chunks) {
                    work.push_back({c, b, gbits, (size_t)(c.keys - keys)});
                    if (ordered) outs.emplace_back(), outDone.push_back(0);
                }
                left[b] = chunks.size();
                published[b] = 1;
                tm.chunks += chunks.size();
            }
            cvWork.notify_all();
        }
        {
            std::lock_guard<std::mutex> l(mu);
            closed = true;
        }
        cvWork.notify_all();
        pump([&] { return finished == work.size() && (!ordered || written == work.size()); });
    } catch (...) {
        std::lock_guard<std::mutex> l(mu);
        fail(std::current_exception());
    }
    pool.wait();
    drainGpu();
    if (failure) std::rethrow_exception(failure);
    for (ChunkOut& a : acc) sink.put(a);
    {
        ChunkOut g;
        g.counts = std::move(gpuCounts);
        sink.put(g);
    }
    tm.scan += nowSec() - p0;
    tm.busy += busyNs * 1e-9;
    tm.cpu1 += cpu1Ns * 1e-9, tm.cpuDeep += cpuDeepNs * 1e-9;
    gpu.releaseText();

    if (need3) {
        double t0 = nowSec();
        std::vector<uint8_t> R;
        buildResidue(t, marks.get(), D, R);
        marks.reset();
        tm.residue += nowSec() - t0;
        tm.residueBases += R.size();
        // Tier 3 needs about 9 bytes per residue base: if that is a lot, give it the batch buffers and
        // the text, which tier 3 does not read (a next record grows it again).
        if (R.size() > (16u << 20)) {
            gpu.releaseBatches();
            PageVec().swap(t.packed), PageVec().swap(t.segBits);
        }
        Tier3Stats st;
        tier3(R, D, rg, pool, sink, st);
        tm.sais += st.sais, tm.scan3 += st.scan;
    }
}

void writeCounts(FILE* f, const std::string& title, const LenCounts& c) {
    fprintf(f, ">%s\n", title.c_str());
    for (size_t l = 0; l < c.v.size(); ++l)
        if (c.v[l]) fprintf(f, "%zu\t%llu\n", l, (unsigned long long)c.v[l]);
    fprintf(f, "\n");
}

void usage() {
    fprintf(stderr,
            "Usage: mawmetal -i <in.fa[.gz]> [options]\n"
            "  -i, --input-file <path>   (Multi)FASTA input, optionally gzipped\n"
            "  -o, --output-file <path>  output file (default: stdout)\n"
            "  -k, --min-length <int>    minimum MAW length (default 2)\n"
            "  -K, --max-length <int>    maximum MAW length (default: unbounded)\n"
            "  -r, --reverse <0|1>       also consider the reverse complement (as MAW -r 1)\n"
            "  -a, --alphabet DNA        accepted for MAW compatibility\n"
            "      --joint               MAWs of all records together (default: per record, like MAW)\n"
            "  -l, --list                write the MAWs themselves (default: counts per MAW length)\n"
            "  -t, --threads <int>       CPU threads (default: all cores)\n"
            "  -v, --verbose             timings on stderr\n"
            " Tuning / testing:\n"
            "      --prefix <1..5>       region prefix length m (default 4)\n"
            "      --split <int>         |W| from which the deep stage takes over (default 31)\n"
            "      --deep <int>          depth the deep stage refines to; longer ones go to libsais (default 1024)\n"
            "      --batch <int>         max positions per GPU batch (default: from GPU memory)\n"
            "      --first <int>         max positions in the first GPU batch (default: batch / 4)\n"
            "      --chunk <int>         tier-1 chunk size in suffixes (default 262144, 65536 with --list)\n"
            "      --piece <int>         FASTA bytes per parallel packing piece (default 1048576)\n"
            "      --pending <int>       --list: output bytes workers may run ahead of the writer by (default: a batch slot, >= 134217728)\n"
            "      --spill <int>         --list: the awaited chunk writes its output every <int> bytes (default 1048576)\n"
            "      --wide                carry position bits 32..39 even for short texts (testing)\n"
            "      --cpu-tier1, --gpu-tier1  tier 1 (for counts) always on the CPU, or the GPU (testing)\n"
            "Non-ACGT characters (N, IUPAC codes) separate the sequence: no reported word spans one.\n");
}

Options parseArgs(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto val = [&]() -> const char* {
            if (i + 1 >= argc) {
                fprintf(stderr, "missing value for %s\n", a.c_str());
                exit(1);
            }
            return argv[++i];
        };
        if (a == "-i" || a == "--input-file") o.input = val();
        else if (a == "-o" || a == "--output-file") o.output = val();
        else if (a == "-k" || a == "--min-length") o.kmin = atoi(val());
        else if (a == "-K" || a == "--max-length") o.Kmax = atoll(val());
        else if (a == "-r" || a == "--reverse") o.revcomp = atoi(val()) != 0;
        else if (a == "-a" || a == "--alphabet") {
            if (strcmp(val(), "DNA") != 0) {
                fprintf(stderr, "only the DNA alphabet is supported\n");
                exit(1);
            }
        } else if (a == "--joint") o.joint = true;
        else if (a == "-l" || a == "--list") o.list = true;
        else if (a == "--count") o.list = false;
        else if (a == "--deep") o.deep = atoll(val());
        else if (a == "--wide") o.wide = true;
        else if (a == "--prefix") o.m = atoi(val());
        else if (a == "--split") o.kRep = atoi(val());
        else if (a == "--batch") o.batchCap = strtoull(val(), nullptr, 10);
        else if (a == "--first") o.firstCap = strtoull(val(), nullptr, 10);
        else if (a == "--chunk") o.chunk = strtoull(val(), nullptr, 10);
        else if (a == "--piece") o.piece = strtoull(val(), nullptr, 10);
        else if (a == "--pending") o.pending = strtoull(val(), nullptr, 10);
        else if (a == "--spill") o.spill = strtoull(val(), nullptr, 10);
        else if (a == "--cpu-tier1") o.tier1 = 1;
        else if (a == "--gpu-tier1") o.tier1 = 2;
        else if (a == "-t" || a == "--threads") o.threads = atoi(val());
        else if (a == "-v" || a == "--verbose") o.verbose = true;
        else if (a == "-h" || a == "--help") {
            usage();
            exit(0);
        } else {
            fprintf(stderr, "unknown option %s\n", a.c_str());
            usage();
            exit(1);
        }
    }
    if (o.input.empty()) {
        usage();
        exit(1);
    }
    return o;
}

}  // namespace

int main(int argc, char** argv) {
    const double start = nowSec();
    Options o = parseArgs(argc, argv);
    try {
        Gpu gpu;
        Timings tm;
        // Shader compilation overlaps reading the input.
        std::string err;
        bool gpuOk = false;
        std::thread initThread([&] {
            double t0 = nowSec();
            gpuOk = gpu.init(err);
            tm.init = nowSec() - t0;
        });
        struct Joiner {
            std::thread& t;
            ~Joiner() {
                if (t.joinable()) t.join();
            }
        } joiner{initThread};
        auto gpuReady = [&] {
            if (initThread.joinable()) initThread.join();
            if (!gpuOk) throw std::runtime_error("GPU init failed: " + err);
        };
        WorkPool pool(o.threads > 0 ? o.threads : (int)std::thread::hardware_concurrency());
        const double open0 = nowSec();
        FastaInput in(o.input, &pool);  // decompresses a single-member gzip file right away
        if (!in.ok()) {
            fprintf(stderr, "cannot open %s\n", o.input.c_str());
            return 1;
        }
        FILE* out = stdout;
        if (!o.output.empty() && !(out = fopen(o.output.c_str(), "w"))) {
            fprintf(stderr, "cannot open %s\n", o.output.c_str());
            return 1;
        }
        static char obuf[1 << 22];
        setvbuf(out, obuf, _IOFBF, sizeof obuf);
        Sink sink(out, !o.list);
        if (o.spill) sink.spillBytes = o.spill;
        double readTime = nowSec() - open0;
        Text t;
        TextBuilder tb(t, &pool);
        if (o.piece) tb.pieceBytes = o.piece;
        if (in.sizeHint()) tb.reserveBases(in.sizeHint() * (o.revcomp ? 2 : 1));

        auto runUnit = [&](const std::string& title, Text& t) {
            gpuReady();
            sink.raw(">" + title + "\n");
            processText(t, o, gpu, pool, sink, tm);
            sink.raw("\n");
            LenCounts rc = sink.takeRecord();
            if (!o.list) writeCounts(out, title, rc);
            if (o.verbose)
                fprintf(stderr, "%s: %llu bases, %llu MAWs\n", title.c_str(), (unsigned long long)t.n,
                        (unsigned long long)rc.total());
        };

        std::string id;
        size_t records = 0;
        if (o.joint) {
            double t0 = nowSec();
            records = in.readAll(tb);
            // The reverse complement of the whole text: every record's, after all of them (segments are
            // independent, so their order does not matter).
            if (o.revcomp) tb.addRevComp();
            tb.finish();
            readTime += nowSec() - t0;
            runUnit("joint (" + std::to_string(records) + " records)", t);
        } else {
            double t0 = nowSec();
            for (;;) {
                tb.reset();  // keeps the storage for the next record
                if (!in.next(id, tb)) break;
                if (tb.recordChars() == 0) continue;  // MAW skips empty records too
                ++records;
                if (o.revcomp) tb.addRevComp();
                tb.finish();
                readTime += nowSec() - t0;
                runUnit(id, t);
                t0 = nowSec();
            }
            readTime += nowSec() - t0;
        }
        if (fflush(out) != 0 || ferror(out) || (out != stdout && fclose(out) != 0))
            throw std::runtime_error(std::string("write failed: ") + strerror(errno));

        if (o.verbose) {
            gpuReady();
            double total = nowSec() - start;
            fprintf(stderr,
                    "device: %s, %d CPU threads\n"
                    "records %zu, bases %llu, MAWs %llu, prefix %d, GPU batches %llu, chunks %llu, residue %llu bases (%.1f%%)\n"
                    "time: total %.3fs | GPU init %.3f (overlaps reading) | read+pack %.3f"
                    " | GPU census %.3f extract %.3f sort %.3f tier 1 %.3f, first batch %.3f, buffers %.3f"
                    " | CPU scan pipeline %.3f, workers %.0f%% busy (CPU: tier 1 %.2fs, deep %.2fs)"
                    " | residue %.3f sais %.3f tier3 scan %.3f\n",
                    gpu.name().c_str(), pool.size(), records, (unsigned long long)tm.bases,
                    (unsigned long long)sink.total.total(), tm.prefix, (unsigned long long)tm.batches,
                    (unsigned long long)tm.chunks, (unsigned long long)tm.residueBases,
                    tm.bases ? 100.0 * tm.residueBases / tm.bases : 0.0, total, tm.init, readTime, tm.census,
                    tm.extract, tm.sort, tm.tier1, tm.firstBatch, tm.reserve, tm.scan,
                    tm.scan > 0 ? 100.0 * tm.busy / (tm.scan * pool.size()) : 0.0, tm.cpu1, tm.cpuDeep, tm.residue, tm.sais, tm.scan3);
        }
    } catch (const std::exception& e) {
        fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
