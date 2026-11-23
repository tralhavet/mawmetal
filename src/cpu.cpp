// Copyright (c) 2026 Nicolas Innocenti
// SPDX-License-Identifier: MIT
#include <pthread.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cerrno>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <thread>

#include <sys/sysctl.h>
#if defined(LIBSAIS_OPENMP)
#include <omp.h>
#endif

#include "libsais.h"
#include "libsais64.h"
#include "maw.h"

namespace {
const char kACGT[5] = {'A', 'C', 'G', 'T', 'N'};

// The four bases of a byte (first in bits 7..6) as four chars, first char in the low byte.
struct Dec4 {
    uint32_t v[256];
    Dec4() {
        for (int b = 0; b < 256; ++b) {
            uint32_t x = 0;
            for (int i = 0; i < 4; ++i) x |= (uint32_t)(uint8_t)kACGT[(b >> (6 - 2 * i)) & 3] << (8 * i);
            v[b] = x;
        }
    }
};
const Dec4 kDec4;

// Writes the 4 * nbytes bases held in the top bytes of x (first base in bits 63..62) as chars.
inline void decode64(uint64_t x, char* d, int nbytes) {
    for (int k = 0; k < nbytes; ++k) {
        uint32_t four = kDec4.v[(x >> (56 - 8 * k)) & 0xFF];
        memcpy(d + 4 * k, &four, 4);
    }
}

// A node of the (truncated) suffix tree during a bottom-up LCP-interval traversal keeps P, a 5x5
// bit matrix: bit 5*b + a set when some occurrence of the node's word W is preceded by a and
// followed by b (index 4 = no base: segment start or end).
inline uint32_t leftSet(uint32_t P) { return (P | (P >> 5) | (P >> 10) | (P >> 15) | (P >> 20)) & 31u; }

// For a finished node, the MAWs aWb are the pairs with aW and Wb present but aWb absent. Returns
// them as bit 5*b + a (a, b < 4), computed without branches.
inline uint32_t missMask(uint32_t P) {
    const uint32_t L = leftSet(P) & 15u;
    const uint32_t rows = P & 0xFFFFFu;  // b = 0..3
    const uint32_t nz = (rows | rows >> 1 | rows >> 2 | rows >> 3 | rows >> 4) & 0x08421u;  // Wb occurs
    return (L * 0x08421u) & ~rows & (nz * 15u);
}

// Appends the lines aWb for the pairs in miss (see missMask) to out. W must have 48 readable bytes.
void emitLines(OutBuf& out, const char* W, size_t wl, uint32_t miss) {
    const size_t L = wl + 3;
    char* d = out.reserve(__builtin_popcount(miss) * L);
    out.commit(__builtin_popcount(miss) * L);
    if (L <= 48) {  // short lines: build once, then patch both ends and store 48 bytes per line
        char line[48];
        memcpy(line + 1, W, 47);  // fixed sizes compile to a few vector moves
        line[wl + 2] = '\n';
        for (uint32_t bits = miss; bits; bits &= bits - 1) {
            const int i = __builtin_ctz(bits), b = i / 5;
            line[0] = kACGT[i - 5 * b];
            line[wl + 1] = kACGT[b];
            memcpy(d, line, 48);
            d += L;
        }
        return;
    }
    for (uint32_t bits = miss; bits; bits &= bits - 1) {
        const int i = __builtin_ctz(bits), b = i / 5;
        d[0] = kACGT[i - 5 * b];
        memcpy(d + 1, W, wl);
        d[wl + 1] = kACGT[b];
        d[wl + 2] = '\n';
        d += L;
    }
}

inline int lcpKeys(uint64_t a, uint64_t b, int m) {
    int common = std::min(__builtin_clzll((a ^ b) | 1) >> 1, 28);
    common = std::min(common, (int)std::min((a >> 3) & 31, (b >> 3) & 31));
    return m + common;
}

// 32 bases starting at p, first base in bits 63..62 (the packed text is padded).
inline uint64_t window64(const uint32_t* T, uint64_t p) {
    uint64_t w = p >> 4;
    unsigned o = (unsigned)(p & 15) * 2;
    uint64_t v = ((uint64_t)T[w] << 32) | T[w + 1];
    if (o) v = (v << o) | (T[w + 2] >> (32 - o));
    return v;
}

// Bases from x to the end of the segment that contains x-1, capped at 32 (0 if one starts at x).
inline int validChars(const Text& t, uint64_t x) {
    if (!t.coarse(x) && !t.coarse(x + 31)) return 32;
    const uint32_t* SB = t.segBits.data();
    uint64_t w = x >> 5;
    uint64_t bits = ((uint64_t)SB[w] | ((uint64_t)SB[w + 1] << 32)) >> (x & 31);
    return bits ? std::min(__builtin_ctzll(bits), 32) : 32;
}

// Bases from x before the next segment start (or n), capped at L: validChars over a longer span.
inline uint64_t segRoom(const Text& t, uint64_t x, uint64_t L) {
    const uint64_t* C = t.segCoarse.data();
    const uint32_t* SB = t.segBits.data();
    const uint64_t end = x + L;
    for (uint64_t b = x >> 6; b << 6 < end;) {  // b: block of 64 bases, one coarse bit
        const uint64_t c = C[b >> 6] >> (b & 63);
        if (!c) {
            b = (b | 63) + 1;
            continue;
        }
        b += __builtin_ctzll(c);
        for (uint64_t w = std::max(x, b << 6) >> 5; w < (b + 1) << 1 && w << 5 < end; ++w) {
            const uint32_t bits = w == x >> 5 ? SB[w] & (~0u << (x & 31)) : SB[w];
            if (bits) return std::min((w << 5) + __builtin_ctz(bits), end) - x;
        }
        ++b;  // the start in this block lies before x
    }
    return L;
}

// Leading 32-base windows on which the texts at p and q agree, at most s; one branch per 4 windows.
inline int equalWindows(const uint32_t* T, uint64_t p, uint64_t q, int s) {
    int k = 0;
    for (; k + 4 <= s; k += 4) {
        const uint64_t x0 = window64(T, p + 32 * k) ^ window64(T, q + 32 * k);
        const uint64_t x1 = window64(T, p + 32 * k + 32) ^ window64(T, q + 32 * k + 32);
        const uint64_t x2 = window64(T, p + 32 * k + 64) ^ window64(T, q + 32 * k + 64);
        const uint64_t x3 = window64(T, p + 32 * k + 96) ^ window64(T, q + 32 * k + 96);
        if (x0 | x1 | x2 | x3) return k + (x0 ? 0 : x1 ? 1 : x2 ? 2 : 3);
    }
    for (; k < s; ++k)
        if (window64(T, p + 32 * k) != window64(T, q + 32 * k)) return k;
    return s;
}
}  // namespace

double nowSec() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

void Text::buildCoarse() {
    segCoarse.assign(n / 4096 + 2, 0);
    for (uint64_t s : segStarts) segCoarse[s >> 12] |= 1ull << ((s >> 6) & 63);
}

// ---- Sink / pool / ordered runner -------------------------------------------------------------------
void Sink::put(ChunkOut& c) {
    raw(c.text.data(), c.text.size());
    record.merge(c.counts);
    total.merge(c.counts);
}

void Sink::raw(const char* s, size_t n) {
    if (countOnly_ || n == 0) return;
    if (fwrite(s, 1, n, f_) != n) throw std::runtime_error(std::string("write failed: ") + strerror(errno));
}

LenCounts Sink::takeRecord() {
    LenCounts r;
    std::swap(r, record);
    return r;
}

WorkPool::WorkPool(int threads) {
    for (int i = 0; i < std::max(1, threads); ++i) threads_.emplace_back([this, i] { loop(i); });
}

WorkPool::~WorkPool() {
    {
        std::lock_guard<std::mutex> l(mu_);
        quit_ = true;
    }
    cvJob_.notify_all();
    for (auto& t : threads_) t.join();
}

void WorkPool::start(std::function<void(int)> job) {
    std::lock_guard<std::mutex> l(mu_);
    job_ = std::move(job);
    active_ = size();
    ++gen_;
    cvJob_.notify_all();
}

void WorkPool::wait() {
    std::unique_lock<std::mutex> l(mu_);
    cvDone_.wait(l, [&] { return active_ == 0; });
    job_ = nullptr;
}

void WorkPool::loop(int id) {
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    uint64_t seen = 0;
    for (;;) {
        {
            std::unique_lock<std::mutex> l(mu_);
            cvJob_.wait(l, [&] { return quit_ || gen_ != seen; });
            if (quit_) return;
            seen = gen_;
        }
        job_(id);  // jobs catch their own exceptions; job_ stays put until wait() returns
        std::lock_guard<std::mutex> l(mu_);
        if (--active_ == 0) cvDone_.notify_all();
    }
}

void parallelFor(WorkPool& pool, size_t n, const std::function<void(size_t)>& fn) {
    std::atomic<size_t> next{0};
    std::mutex mu;
    std::exception_ptr failure;
    pool.run([&](int) {
        for (size_t i; (i = next++) < n;) {
            try {
                fn(i);
            } catch (...) {
                std::lock_guard<std::mutex> l(mu);
                if (!failure) failure = std::current_exception();
                next = n;
            }
        }
    });
    if (failure) std::rethrow_exception(failure);
}

void runOrdered(WorkPool& pool, size_t n, const std::function<void(size_t, ChunkOut&)>& work, Sink& sink) {
    if (n == 0) return;
    // Counts only: no order to keep, each worker accumulates on its own.
    const bool ordered = !sink.countOnly();
    const size_t window = 4 * pool.size() + 2;  // bounds how many finished chunks wait in memory
    std::vector<ChunkOut> slots(ordered ? n : pool.size());
    std::vector<char> done(n, 0);
    std::atomic<size_t> next{0};
    std::mutex mu;
    std::condition_variable cvDone, cvWindow;
    size_t written = 0;
    bool stop = false;
    std::exception_ptr failure;

    pool.start([&](int w) {
        size_t i = 0;
        // The chunk next in line writes its output as it goes (see processText).
        const OutBuf::Spill spill = [&](const char* s, size_t k) {
            {
                std::lock_guard<std::mutex> l(mu);
                if (written != i) return false;
            }
            sink.raw(s, k);
            return true;
        };
        for (;;) {
            i = next++;
            if (i >= n) return;
            if (ordered) {
                std::unique_lock<std::mutex> l(mu);
                cvWindow.wait(l, [&] { return stop || i < written + window; });
                if (stop) return;
                slots[i].text.spill(&spill, sink.spillBytes);
            }
            try {
                work(i, slots[ordered ? i : w]);
            } catch (...) {
                std::lock_guard<std::mutex> l(mu);
                if (!failure) failure = std::current_exception();
                stop = true;
            }
            if (ordered) {
                slots[i].text.spill(nullptr);
                std::lock_guard<std::mutex> l(mu);
                done[i] = 1;
                cvDone.notify_all();
            }
        }
    });
    for (size_t i = 0; ordered && i < n; ++i) {
        {
            std::unique_lock<std::mutex> l(mu);
            cvDone.wait(l, [&] { return done[i] != 0 || failure; });
            if (failure) break;
        }
        try {
            sink.put(slots[i]);
        } catch (...) {
            std::lock_guard<std::mutex> l(mu);
            failure = std::current_exception();
            break;
        }
        slots[i].text = OutBuf();  // frees it
        {
            std::lock_guard<std::mutex> l(mu);
            written = i + 1;
        }
        cvWindow.notify_all();
    }
    {
        // On failure, release workers blocked on the window so they can finish.
        std::lock_guard<std::mutex> l(mu);
        if (failure) stop = true;
    }
    cvWindow.notify_all();
    pool.wait();
    if (failure) std::rethrow_exception(failure);
    if (!ordered)
        for (auto& s : slots) sink.put(s);
}

// ---- short MAWs from the census --------------------------------------------------------------------
void bitmapTier(const std::vector<uint32_t>& pres, const LenRange& r, bool countOnly, ChunkOut& out) {
    const int Lmax = kCensusLen;
    std::vector<std::vector<uint8_t>> present(Lmax + 1);
    // A word of length l occurs iff it was recorded at length l (the position's segment ends right
    // after it) or it is a prefix of an occurring word of length l+1.
    for (int l = Lmax; l >= 1; --l) {
        size_t sz = 1ull << (2 * l), off = ((1ull << (2 * l)) - 4) / 3;
        present[l].assign(sz, 0);
        for (size_t w = 0; w < sz; ++w) {
            size_t b = off + w;
            uint8_t p = (pres[b >> 5] >> (b & 31)) & 1;
            if (l < Lmax) {
                const uint8_t* q = &present[l + 1][w * 4];
                p |= q[0] | q[1] | q[2] | q[3];
            }
            present[l][w] = p;
        }
    }
    char buf[16];
    for (int l = std::max(2, r.kmin); l <= std::min<int64_t>(Lmax, r.Kmax); ++l) {
        size_t sz = 1ull << (2 * l), sufMask = (1ull << (2 * (l - 1))) - 1;
        for (size_t w = 0; w < sz; ++w) {
            if (present[l][w] || !present[l - 1][w >> 2] || !present[l - 1][w & sufMask]) continue;
            out.counts.add(l, 1);
            if (countOnly) continue;
            for (int i = 0; i < l; ++i) buf[i] = kACGT[(w >> (2 * (l - 1 - i))) & 3];
            buf[l] = '\n';
            out.text.append(buf, l + 1);
        }
    }
}

// ---- tier 1 ------------------------------------------------------------------------------------------
void planTier1Chunks(const T1Chunk& region, int m, int cutBelow, size_t target, std::vector<T1Chunk>& out) {
    // Nodes deeper than cutBelow-1 never straddle an index j with lcp(j-1, j) < cutBelow, so the
    // region can be cut there. With cutBelow == m no such j exists and the region stays whole.
    const size_t n = region.n;
    const uint64_t* keys = region.keys;
    size_t s = 0;
    auto piece = [&](size_t a, size_t b) {
        T1Chunk c = region;
        c.keys += a;
        if (c.vals) c.vals += a;
        if (c.hi) c.hi += a;
        c.n = b - a;
        out.push_back(c);
    };
    if (cutBelow > m && n > target + target / 2) {
        for (size_t cut = target; cut < n; cut = std::max(cut + target, s + target)) {
            size_t j = cut;
            while (j < n && lcpKeys(keys[j - 1], keys[j], m) >= cutBelow) ++j;
            if (j >= n) break;
            piece(s, j);
            s = j;
        }
    }
    if (s < n) piece(s, n);
}

namespace {
template <bool kGroups, bool kCountOnly>
void scanTier1Impl(const T1Chunk& c, int m, int dLo, int dHi, int kRep, ChunkOut& out, Groups* groups) {
    const uint64_t* __restrict keys = c.keys;
    const size_t n = c.n;
    // Frames live at depths (F, C]: F is a floor that absorbs everything shallower and is never
    // reported, C collapses the deeper structure, which is irrelevant here.
    const int F = dLo - 1, C = dHi + 1;
    struct Fr {
        uint64_t key;  // key of any member, which spells W and gives the base after it
        uint32_t P;
        int32_t depth;
    };
    // The top frame lives in registers, the ones below it in st[0, sp); the floor is at the bottom.
    Fr st[64];
    int sp = 0;
    uint64_t topKey = keys[0];
    uint32_t topP = 0;
    int topDepth = F;
    uint64_t cnt[64] = {};  // by MAW length; index C + 2 collects the collapsed frames
    // --list: nodes with MAWs are recorded here (one per frame at most) and spelled after every
    // kBlock suffixes, which keeps the scan loop free of calls. Each suffix pushes at most one frame,
    // so a block records fewer than kBlock + 64 nodes.
    struct Rec {
        uint64_t key;
        uint32_t dm;  // depth << 24 | miss
    };
    constexpr size_t kBlock = 1 << 13;
    thread_local std::vector<Rec> recs;
    if (!kCountOnly && recs.size() < kBlock + 64) recs.resize(kBlock + 64);
    Rec* rp = recs.data();
    size_t nrec = 0;
    // Takes the count rather than capturing nrec, which would keep it out of a register.
    auto spell = [rp, m, prefix = c.prefix, &out](size_t nr) {
        char line[64];
        for (int i = 0; i < m; ++i) line[1 + i] = kACGT[(prefix >> (2 * (m - 1 - i))) & 3];
        for (size_t r = 0; r < nr; ++r) {
            const int depth = (int)(rp[r].dm >> 24);
            const uint32_t miss = rp[r].dm & 0xFFFFFu;
            const size_t L = depth + 3;
            char* d = out.text.reserve(__builtin_popcount(miss) * L);
            out.text.commit(__builtin_popcount(miss) * L);
            decode64(rp[r].key, line + 1 + m, 7);
            line[depth + 2] = '\n';
            for (uint32_t bits = miss; bits; bits &= bits - 1) {
                const int i = __builtin_ctz(bits), b = i / 5;
                line[0] = kACGT[i - 5 * b];
                line[depth + 1] = kACGT[b];
                memcpy(d, line, 48);  // lines here are at most 33 bytes
                d += L;
            }
        }
    };
    // Base at depth d (m <= d <= m + 28) of the suffix with key k, 4 past its end.
    auto right = [m](uint64_t k, int d) -> uint32_t {
        const int dd = d - m;
        const uint32_t b = (uint32_t)(k >> (62 - 2 * dd)) & 3;
        return (int)((k >> 3) & 31) > dd ? b : 4u;
    };
    // Bit j set when keys j and j+1 share kRep bases; turned into groups after the scan.
    std::vector<uint64_t> gbits(kGroups ? n / 64 + 1 : 0, 0);
    uint64_t gword = 0;  // the current word of gbits, kept in a register
    for (size_t j0 = 0; j0 < n; j0 += kBlock) {
        for (size_t j = j0; j < std::min(n, j0 + kBlock); ++j) {
            const uint64_t k = keys[j];
            uint32_t L = 1u << (k & 7);
            uint64_t rep = k;
            int h = F;
            if (j + 1 < n) {
                const int raw = lcpKeys(k, keys[j + 1], m);
                if (kGroups) {
                    gword |= (uint64_t)(raw >= kRep) << (j & 63);
                    if ((j & 63) == 63) gbits[j >> 6] = gword, gword = 0;
                }
                h = std::min(std::max(raw, F), C);
            }
            while (topDepth > h) {  // the floor is never popped
                const uint32_t P = topP | L << (5 * right(rep, topDepth));
                const uint32_t miss = missMask(P);
                cnt[topDepth + 2] += __builtin_popcount(miss);
                if (!kCountOnly) {
                    rp[nrec] = {topKey, (uint32_t)topDepth << 24 | miss};
                    nrec += miss && topDepth <= dHi;
                }
                L = leftSet(P);
                rep = topKey;
                const Fr& f = st[--sp];
                topKey = f.key, topP = f.P, topDepth = f.depth;
            }
            // Push a frame at depth h unless the top is already there, without a branch.
            const bool push = topDepth != h;
            st[sp] = {topKey, topP, topDepth};
            sp += push;
            topKey = push ? rep : topKey;
            topP = (push ? 0 : topP) | L << (5 * right(rep, h));  // the floor absorbs it harmlessly
            topDepth = h;
        }
        if (!kCountOnly) spell(nrec), nrec = 0;
    }
    if (kGroups) {
        gbits[(n - 1) >> 6] |= gword;
        // A run of set bits j0 .. j1-1 is the group of keys j0 .. j1. Bit n-1 is never set, so every
        // run ends inside gbits.
        const size_t nw = gbits.size();
        for (size_t j = 0;;) {
            size_t w = j >> 6;
            uint64_t x = w < nw ? gbits[w] & (~0ull << (j & 63)) : 0;
            while (!x && ++w < nw) x = gbits[w];
            if (!x) break;
            const size_t j0 = w * 64 + __builtin_ctzll(x);
            w = j0 >> 6;
            uint64_t y = ~gbits[w] & (~0ull << (j0 & 63));
            while (!y) y = ~gbits[++w];
            const size_t j1 = w * 64 + __builtin_ctzll(y);
            groups->push_back((uint32_t)j0), groups->push_back((uint32_t)(j1 + 1 - j0));
            j = j1;
        }
    }
    out.counts.addRange(cnt, dLo + 2, dHi + 3);
}
}  // namespace

void scanTier1(const T1Chunk& c, int m, int dLo, int dHi, int kRep, bool countOnly, ChunkOut& out, Groups* groups) {
    if (c.n == 0) return;
    if (groups) {
        if (countOnly) scanTier1Impl<true, true>(c, m, dLo, dHi, kRep, out, groups);
        else scanTier1Impl<true, false>(c, m, dLo, dHi, kRep, out, groups);
    } else {
        if (countOnly) scanTier1Impl<false, true>(c, m, dLo, dHi, kRep, out, groups);
        else scanTier1Impl<false, false>(c, m, dLo, dHi, kRep, out, groups);
    }
}

void findGroups(const T1Chunk& c, int m, int kRep, Groups& groups) {
    for (size_t i = 0; i < c.n;) {
        size_t j = i + 1;
        while (j < c.n && lcpKeys(c.keys[j - 1], c.keys[j], m) >= kRep) ++j;
        if (j - i > 1) groups.push_back((uint32_t)i), groups.push_back((uint32_t)(j - i));
        i = j;
    }
}

void groupsFromBits(const uint32_t* bits, size_t off, size_t n, Groups& groups) {
    const size_t end = off + n;
    for (size_t j = off; j < end;) {
        const uint32_t x = bits[j >> 5] >> (j & 31);
        if (!x) {
            j = (j | 31) + 1;
            continue;
        }
        j += __builtin_ctz(x);
        if (j >= end) break;
        const size_t j0 = j;  // the run of set bits j0 .. j-1 is the group j0 .. j
        while (j < end) {
            const uint32_t y = ~bits[j >> 5] >> (j & 31);
            if (y) {
                j += __builtin_ctz(y);
                break;
            }
            j = (j | 31) + 1;
        }
        j = std::min(j, end - 1);
        groups.push_back((uint32_t)(j0 - off)), groups.push_back((uint32_t)(j + 1 - j0));
        ++j;
    }
}

// ---- deep stage --------------------------------------------------------------------------------------
namespace {
struct DeepItem {
    uint64_t w;   // the 32 bases at the current depth, zero past the segment end
    uint64_t lo;  // n << 58 | charC << 44 | left << 40 | pos, so (w, lo) orders by (w, n)
};
constexpr int kNShift = 58, kCShift = 44, kLShift = 40;
constexpr uint64_t kPosMask = (1ull << 40) - 1;
inline uint64_t posOf(const DeepItem& x) { return x.lo & kPosMask; }
inline int nOf(const DeepItem& x) { return (int)(x.lo >> kNShift); }
inline uint32_t leftOf(const DeepItem& x) { return (x.lo >> kLShift) & 7; }
inline uint32_t charCOf(const DeepItem& x) { return (x.lo >> kCShift) & 7; }
inline bool sameWN(const DeepItem& x, const DeepItem& y) { return x.w == y.w && (x.lo >> kNShift) == (y.lo >> kNShift); }
// Items are ordered by (w, n) only: items equal on both are identical down to their segment end, or
// are ordered further at the next depth, so their relative order does not matter.
inline bool lessWN(const DeepItem& x, const DeepItem& y) {
    return x.w < y.w || (x.w == y.w && nOf(x) < nOf(y));
}

// Puts x and y in order with selects rather than a branch, whose outcome is a coin toss.
inline void orderPair(DeepItem& x, DeepItem& y) {
    const uint64_t xw = x.w, xl = x.lo, yw = y.w, yl = y.lo;
    const bool s = yw < xw || (yw == xw && (yl >> kNShift) < (xl >> kNShift));
    x.w = s ? yw : xw, x.lo = s ? yl : xl;
    y.w = s ? xw : yw, y.lo = s ? xl : yl;
}

// LSD radix sort by (w, n) through tmp: a counting pass over n, then one per byte of w from the
// lowest. A pass whose digit is the same for every item (common inside repeats) is skipped.
void radixItems(DeepItem* a, size_t cnt, DeepItem* tmp) {
    uint32_t h[9][256] = {};
    for (size_t i = 0; i < cnt; ++i) {
        const uint64_t w = a[i].w;
        ++h[0][nOf(a[i])];
        for (int b = 0; b < 8; ++b) ++h[b + 1][(w >> (8 * b)) & 255];
    }
    auto digit = [](const DeepItem& x, int d) -> unsigned { return d ? (x.w >> (8 * (d - 1))) & 255 : nOf(x); };
    DeepItem *src = a, *dst = tmp;
    for (int d = 0; d < 9; ++d) {
        uint32_t* hd = h[d];
        if (hd[digit(src[0], d)] == cnt) continue;
        for (uint32_t v = 0, sum = 0; v < 256; ++v) {
            const uint32_t c = hd[v];
            hd[v] = sum, sum += c;
        }
        for (size_t i = 0; i < cnt; ++i) dst[hd[digit(src[i], d)]++] = src[i];
        std::swap(src, dst);
    }
    if (src != a) std::copy(src, src + cnt, a);
}

// Groups average about 5 members: pairs and small networks first, insertion up to 16, radix for
// large groups.
constexpr size_t kRadixMin = 256;
void sortItems(DeepItem* a, size_t cnt, std::vector<DeepItem>& tmp) {
    switch (cnt) {
        case 2:
            orderPair(a[0], a[1]);
            return;
        case 3:
            orderPair(a[0], a[1]), orderPair(a[1], a[2]), orderPair(a[0], a[1]);
            return;
        case 4:
            orderPair(a[0], a[1]), orderPair(a[2], a[3]), orderPair(a[0], a[2]), orderPair(a[1], a[3]);
            orderPair(a[1], a[2]);
            return;
    }
    if (cnt <= 16) {
        for (size_t i = 1; i < cnt; ++i) {
            DeepItem x = a[i];
            size_t j = i;
            for (; j > 0 && lessWN(x, a[j - 1]); --j) a[j] = a[j - 1];
            a[j] = x;
        }
    } else if (cnt < kRadixMin) {
        std::sort(a, a + cnt, lessWN);
    } else {
        if (tmp.size() < cnt) tmp.resize(cnt);
        radixItems(a, cnt, tmp.data());
    }
}
}  // namespace

// Items ahead whose text the gathers prefetch: enough to keep many misses in flight.
constexpr size_t kPrefetch = 48;
// Windows of 32 bases that skipTies compares at most (1024 bases).
constexpr int kSkip = 32;

class DeepScanner {
public:
    // cnt_ is indexed by depth + 2 for depths up to C (whose entry collects the collapsed frames).
    DeepScanner(const Text& t, const DeepParams& p) : t_(t), p_(p), C_(p.dHi + 1), cnt_(std::max<int64_t>(p.dHi + 4, 1), 0) {}

    void begin(ChunkOut& out) { out_ = &out; }
    void flush() {
        if (p_.dLo <= p_.dHi) out_->counts.addRange(cnt_.data(), p_.dLo + 2, p_.dHi + 3);
        std::fill(cnt_.begin(), cnt_.end(), 0);
    }

    // The groups of a chunk ((first, count) pairs, see Groups). Groups are small (about 5 members on
    // human chromosomes), so the text each one reads first is prefetched kPrefetch members ahead
    // across group boundaries; a larger group prefetches its own members past the first kPrefetch.
    void run(const T1Chunk& c, const Groups& groups) {
        size_t pg = 0, pi = 0, issued = 0, start = 0;  // next to prefetch: member pi of group pg / 2, the issued-th
        for (size_t g = 0; g < groups.size(); g += 2) {
            const size_t cnt = groups[g + 1];
            if (issued < start) pg = g, pi = 0, issued = start;  // past a large group
            const size_t target = start + (cnt <= kPrefetch ? cnt + kPrefetch : kPrefetch);
            for (; issued < target && pg < groups.size(); ++issued) {
                prefetchText(c.pos(groups[pg] + pi) + p_.kRep);
                if (++pi == groups[pg + 1]) pi = 0, pg += 2;
            }
            group(c, groups[g], cnt);
            start += cnt;
        }
    }

private:
    // The text of a member from q = pos + kRep down to depth D, which skipTies reads for most of them
    // (128-byte lines of 512 bases), and its coarse segment bits, which outgrow the caches on large
    // texts (47 MB for 24 Gbp).
    void prefetchText(uint64_t q) const {
        const uint32_t* T = t_.packed.data();
        const uint64_t span = (uint64_t)std::max<int64_t>(p_.D - p_.kRep, 1);
        for (uint64_t o = 0; o < span; o += 512) __builtin_prefetch(&T[(q + o) >> 4]);
        __builtin_prefetch(&T[(q + span - 1) >> 4]);
        __builtin_prefetch(&t_.segCoarse.data()[q >> 12]);
    }

    // Members of one group (entries [first, first + cnt) of the chunk): suffixes sharing their
    // first kRep bases. Their first kPrefetch text reads were prefetched by run().
    void group(const T1Chunk& c, size_t first, size_t cnt) {
        if (cnt == 2) return pair(c.pos(first), c.pos(first + 1));
        items_.resize(cnt);
        lcp_.resize(cnt);
        cA_.resize(cnt);
        cB_.resize(cnt);
        DeepItem* a = items_.data();
        const int64_t depth = p_.kRep;
        // The first gather also reads the base before each suffix (usually the same cache line).
        constexpr size_t PD = kPrefetch;
        for (size_t i = 0; i < cnt; ++i) {
            if (i + PD < cnt) prefetchText(c.pos(first + i + PD) + depth);
            const uint64_t pos = c.pos(first + i);
            a[i].lo = pos | (uint64_t)left(pos) << kLShift;
        }
        gather(a, cnt, depth, false);
        split(a, cnt, depth, lcp_.data(), cA_.data(), cB_.data());
        if (p_.dLo <= p_.dHi) scan(cnt);
        for (size_t i = 1; i < cnt; ++i)
            if (lcp_[i] >= p_.D) mark(posOf(a[i - 1])), mark(posOf(a[i]));
    }

    // The base before the suffix at p, 4 at a segment start.
    uint32_t left(uint64_t p) const { return t_.segStart(p) ? 4u : (uint32_t)t_.base(p - 1); }

    // A group of two, the commonest (60% of them on GRCh38): its one node straight from the text,
    // without the items, sort, recursion and stack of larger groups. The same result: a node at
    // depth min(LCP, C) made of both members' bases there, and both marked at an LCP of D.
    void pair(uint64_t p0, uint64_t p1) {
        const uint32_t* T = t_.packed.data();
        const int64_t D = p_.D;
        int64_t d = p_.kRep;  // they share their first kRep bases
        int64_t lcp = D;
        uint32_t c0 = 4, c1 = 4;  // their bases at depth lcp (4: past the segment end)
        // Where they part in the window at d; false when they agree on all of it.
        auto part = [&] {
            const uint64_t w0 = window64(T, p0 + d), w1 = window64(T, p1 + d), x = w0 ^ w1;
            const int n0 = validChars(t_, p0 + d), n1 = validChars(t_, p1 + d);
            const int common = std::min({x ? __builtin_clzll(x) >> 1 : 32, n0, n1});
            if (common == 32) return false;
            lcp = std::min<int64_t>(d + common, D);
            if (common < n0) c0 = (w0 >> (62 - 2 * common)) & 3;
            if (common < n1) c1 = (w1 >> (62 - 2 * common)) & 3;
            return true;
        };
        // Most pairs part in their first window; the others skip the whole windows they agree on
        // inside both segments.
        if (!part() && (d += 32) < D) {
            const uint64_t room = std::min(segRoom(t_, p0 + d, D - d), segRoom(t_, p1 + d, D - d));
            d += 32 * equalWindows(T, p0 + d, p1 + d, (int)(room / 32));
            if (d < D) part();
        }
        if (lcp >= D) mark(p0), mark(p1);
        if (p_.dLo > p_.dHi || lcp >= C_) return;  // beyond C the structure is irrelevant
        const uint32_t miss = missMask((1u << left(p0)) << (5 * c0) | (1u << left(p1)) << (5 * c1));
        cnt_[lcp + 2] += __builtin_popcount(miss);
        if (!p_.countOnly && miss && lcp >= p_.dLo) emit(lcp, p0, miss);
    }

    // Loads the 32 bases at depth of every item, and its base at the collapse depth C when that
    // lies in this window.
    void gather(DeepItem* a, size_t cnt, int64_t depth, bool prefetch) {
        const uint32_t* T = t_.packed.data();
        const int64_t offC = C_ - depth;
        const bool capC = offC >= 0 && offC < 32;
        constexpr size_t PD = kPrefetch;
        if (prefetch)
            for (size_t i = 0; i < std::min(cnt, PD); ++i) __builtin_prefetch(&T[(posOf(a[i]) + depth) >> 4]);
        for (size_t i = 0; i < cnt; ++i) {
            if (prefetch && i + PD < cnt) __builtin_prefetch(&T[(posOf(a[i + PD]) + depth) >> 4]);
            const uint64_t x = posOf(a[i]) + depth;
            const int n = validChars(t_, x);
            const uint64_t w = n ? window64(T, x) & (~0ull << (64 - 2 * n)) : 0;
            uint64_t lo = (a[i].lo & ~(63ull << kNShift)) | (uint64_t)n << kNShift;
            if (capC) {
                const uint64_t ch = offC < n ? (w >> (62 - 2 * offC)) & 3 : 4;
                lo = (lo & ~(7ull << kCShift)) | ch << kCShift;
            }
            a[i].w = w;
            a[i].lo = lo;
        }
    }

    // Base of item x at depth e, where x was gathered at depth (e < depth means e == C, captured
    // earlier).
    uint8_t charAt(const DeepItem& x, int64_t e, int64_t depth) const {
        if (e < depth) return (uint8_t)charCOf(x);
        const int64_t off = e - depth;
        return off < nOf(x) ? (uint8_t)((x.w >> (62 - 2 * off)) & 3) : 4;
    }

    // a[0, cnt) share their first depth bases and hold their windows at depth: sorts them by the
    // following bases (down to depth D). lcp[i] receives the LCP of a[i-1] and a[i], capped at D;
    // cA[i] and cB[i] the bases of a[i-1] and a[i] at depth min(lcp[i], C), which is all the scan
    // needs from the text.
    void split(DeepItem* a, size_t cnt, int64_t depth, int32_t* lcp, uint8_t* cA, uint8_t* cB) {
        bool uniform = true;  // common inside long exact repeats: nothing to sort
        for (size_t i = 1; i < cnt && uniform; ++i) uniform = sameWN(a[i], a[0]);
        if (!uniform) sortItems(a, cnt, sortTmp_);
        const int64_t D = p_.D, C = C_;
        for (size_t i = 0; i < cnt;) {
            size_t j = i + 1;
            while (j < cnt && sameWN(a[j], a[i])) ++j;
            if (j < cnt) {  // before descending, which overwrites the windows in [i, j)
                const uint64_t x = a[j - 1].w ^ a[j].w;
                const int common = std::min({x ? __builtin_clzll(x) >> 1 : 32, nOf(a[j - 1]), nOf(a[j])});
                const int64_t l = std::min<int64_t>(depth + common, D), e = std::min(l, C);
                lcp[j] = (int32_t)l;
                cA[j] = charAt(a[j - 1], e, depth);
                cB[j] = charAt(a[j], e, depth);
            }
            if (j - i > 1) {
                const int n = nOf(a[i]);
                if (n == 32) {
                    descend(a + i, j - i, skipTies(a + i, j - i, depth + 32), lcp + i, cA + i, cB + i);
                } else {  // identical up to the end of their segments
                    const int64_t l = std::min<int64_t>(depth + n, D), e = std::min(l, C);
                    for (size_t k = i + 1; k < j; ++k) {
                        lcp[k] = (int32_t)l;
                        cA[k] = charAt(a[k - 1], e, depth);
                        cB[k] = charAt(a[k], e, depth);
                    }
                }
            }
            i = j;
        }
    }

    // a[0, cnt) (cnt >= 2) share their first d bases. Inside long exact repeats they keep agreeing
    // for many windows: rather than one gather and split per 32 bases, compare each item's
    // following windows (the same cache lines) with the first item's, and return the depth
    // d + 32 * s of the first window on which they may differ (s <= kSkip, never past D), with the
    // bases at C captured if C was skipped over.
    int64_t skipTies(DeepItem* a, size_t cnt, int64_t d) {
        int smax = (int)std::min<int64_t>(kSkip, (p_.D - d) / 32);  // only windows ending by D
        if (smax <= 0) return d;
        const uint32_t* T = t_.packed.data();
        // The spans of the first kSpanAhead items, then one more per item: 128-byte lines of 512 bases,
        // of which the gather at d loaded only the first.
        constexpr size_t kSpanAhead = 8;
        auto prefetchSpan = [&](size_t i) {
            const uint64_t q = posOf(a[i]) + d;
            for (uint64_t o = 512; o < 32u * smax + 512; o += 512) __builtin_prefetch(&T[(q + o) >> 4]);
        };
        for (size_t i = 0; i < std::min(cnt, kSpanAhead); ++i) prefetchSpan(i);
        const uint64_t p0 = posOf(a[0]) + d;
        smax = (int)std::min<uint64_t>(smax, segRoom(t_, p0, 32 * smax) / 32);  // whole windows of a[0]
        for (size_t i = 1; i < cnt && smax > 0; ++i) {
            if (i + kSpanAhead < cnt) prefetchSpan(i + kSpanAhead);
            const uint64_t p = posOf(a[i]) + d;
            smax = equalWindows(T, p0, p, (int)(segRoom(t_, p, 32 * smax) / 32));
        }
        if (smax == 0) return d;
        const int64_t offC = C_ - d;
        if (offC >= 0 && offC < 32 * smax) {
            const uint64_t ch = window64(T, p0 + offC) >> 62;
            for (size_t i = 0; i < cnt; ++i) a[i].lo = (a[i].lo & ~(7ull << kCShift)) | ch << kCShift;
        }
        return d + 32 * smax;
    }

    // a[0, cnt) (cnt >= 2) share their first depth bases.
    void descend(DeepItem* a, size_t cnt, int64_t depth, int32_t* lcp, uint8_t* cA, uint8_t* cB) {
        const int64_t D = p_.D;
        if (depth >= D && depth > C_) {  // tied down to D; their bases at C were captured already
            for (size_t k = 1; k < cnt; ++k) {
                lcp[k] = (int32_t)D;
                cA[k] = (uint8_t)charCOf(a[k - 1]);
                cB[k] = (uint8_t)charCOf(a[k]);
            }
            return;
        }
        gather(a, cnt, depth, true);
        if (depth >= D) {  // depth == C == D: this gather only captured the bases at C
            for (size_t k = 1; k < cnt; ++k) {
                lcp[k] = (int32_t)D;
                cA[k] = (uint8_t)charCOf(a[k - 1]);
                cB[k] = (uint8_t)charCOf(a[k]);
            }
            return;
        }
        split(a, cnt, depth, lcp, cA, cB);
    }

    // Bottom-up LCP-interval traversal of the refined group; all its LCPs are >= kRep. A frame's
    // first item is where its interval starts; the base of a child at its parent's depth is cB at
    // the child's first item (when the parent closes) or cA after its last item (otherwise).
    void scan(size_t cnt) {
        const int64_t F = p_.kRep - 1, C = C_;
        stack_.clear();
        stack_.push_back({F, 0, 0});
        const DeepItem* a = items_.data();
        const int32_t* lcp = lcp_.data();
        const uint8_t *cA = cA_.data(), *cB = cB_.data();
        for (size_t i = 0; i < cnt; ++i) {
            uint32_t L = 1u << leftOf(a[i]);
            uint32_t first = (uint32_t)i;
            const int64_t h = i + 1 < cnt ? std::min<int64_t>(lcp[i + 1], C) : F;
            while (stack_.back().depth > h) {
                Frame f = stack_.back();
                stack_.pop_back();
                f.P |= L << (5 * cB[first]);
                const uint32_t miss = missMask(f.P);
                cnt_[f.depth + 2] += __builtin_popcount(miss);
                if (!p_.countOnly && miss && f.depth >= p_.dLo && f.depth <= p_.dHi) emit(f.depth, posOf(a[f.first]), miss);
                L = leftSet(f.P);
                first = f.first;
            }
            if (stack_.back().depth != h) stack_.push_back({h, 0, first});
            if (stack_.back().depth > F) stack_.back().P |= L << (5 * cA[i + 1]);
        }
    }

    void emit(int64_t depth, uint64_t pos, uint32_t miss) {
        W_.resize(depth + 64);
        const uint32_t* T = t_.packed.data();
        for (int64_t d = 0; d < depth; d += 32) decode64(window64(T, pos + d), W_.data() + d, 8);
        emitLines(out_->text, W_.data(), depth, miss);
    }

    void mark(uint64_t p) { __atomic_fetch_or(&p_.marks[p >> 5], 1u << (p & 31), __ATOMIC_RELAXED); }

    const Text& t_;
    const DeepParams& p_;
    const int64_t C_;  // the collapse depth: deeper structure is irrelevant
    ChunkOut* out_ = nullptr;
    std::vector<DeepItem> items_, sortTmp_;
    std::vector<int32_t> lcp_;
    std::vector<uint8_t> cA_, cB_;
    struct Frame {
        int64_t depth;
        uint32_t P, first;
    };
    std::vector<Frame> stack_;
    std::vector<uint64_t> cnt_;  // by MAW length
    std::vector<char> W_;
};

DeepWork::DeepWork(const Text& t, const DeepParams& p) : s_(new DeepScanner(t, p)) {}
DeepWork::~DeepWork() { delete s_; }
void DeepWork::run(const T1Chunk& c, const Groups& groups, ChunkOut& out) {
    s_->begin(out);
    s_->run(c, groups);
    s_->flush();
}

// ---- residue -----------------------------------------------------------------------------------------
void buildResidue(const Text& t, const uint32_t* rep, int64_t k, std::vector<uint8_t>& R) {
    R.clear();
    const uint64_t n = t.n;
    const uint32_t* T = t.packed.data();
    size_t seg = 0;
    int64_t cs = -1, ce = -1;  // current run covered by overlapping repeated k-mers
    size_t cseg = 0;
    int64_t ps = -1, pe = -1;  // pending extended run, merged with touching neighbours
    size_t pseg = 0;
    auto append = [&](int64_t s, int64_t e) {
        size_t o = R.size();
        R.resize(o + (e - s) + 1);
        uint8_t* d = R.data() + o;
        for (int64_t p = s; p < e; p += 32) {
            const uint64_t w = window64(T, p);
            const int c = (int)std::min<int64_t>(32, e - p);
            for (int i = 0; i < c; ++i) d[i] = (uint8_t)(1 + ((w >> (62 - 2 * i)) & 3));
            d += c;
        }
        *d = 0;
    };
    auto closeRun = [&] {
        int64_t s = cs, e = ce;
        if (s > (int64_t)t.segStarts[cseg]) --s;
        if (e < (int64_t)t.segStarts[cseg + 1]) ++e;
        if (ps >= 0 && pseg == cseg && s <= pe) {
            pe = std::max(pe, e);
            return;
        }
        if (ps >= 0) append(ps, pe);
        ps = s, pe = e, pseg = cseg;
    };
    for (uint64_t w = 0; w * 32 < n; ++w) {
        for (uint32_t bits = rep[w]; bits; bits &= bits - 1) {
            int64_t p = (int64_t)(w * 32 + __builtin_ctz(bits));
            while ((int64_t)t.segStarts[seg + 1] <= p) ++seg;
            if (cs >= 0 && p < ce) {
                ce = std::max<int64_t>(ce, p + k);
            } else {
                if (cs >= 0) closeRun();
                cs = p, ce = p + k, cseg = seg;
            }
        }
    }
    if (cs >= 0) closeRun();
    if (ps >= 0) append(ps, pe);
}

// ---- tier 3 ------------------------------------------------------------------------------------------
namespace {
// Generalized suffix array and PLCP of R (0-separated runs), with 32-bit indices when they fit. The
// scan reads LCP[i] as PLCP[SA[i]], which saves the LCP array.
bool buildSaPlcp(const uint8_t* T, int32_t n, std::vector<int32_t>& SA, std::vector<int32_t>& PLCP, int threads) {
    SA.resize(n), PLCP.resize(n);
#if defined(LIBSAIS_OPENMP)
    return libsais_gsa_omp(T, SA.data(), n, 0, nullptr, threads) == 0 &&
           libsais_plcp_gsa_omp(T, SA.data(), PLCP.data(), n, threads) == 0;
#else
    return libsais_gsa(T, SA.data(), n, 0, nullptr) == 0 && libsais_plcp_gsa(T, SA.data(), PLCP.data(), n) == 0;
#endif
}

bool buildSaPlcp(const uint8_t* T, int64_t n, std::vector<int64_t>& SA, std::vector<int64_t>& PLCP, int threads) {
    SA.resize(n), PLCP.resize(n);
#if defined(LIBSAIS_OPENMP)
    return libsais64_gsa_omp(T, SA.data(), n, 0, nullptr, threads) == 0 &&
           libsais64_plcp_gsa_omp(T, SA.data(), PLCP.data(), n, threads) == 0;
#else
    return libsais64_gsa(T, SA.data(), n, 0, nullptr) == 0 && libsais64_plcp_gsa(T, SA.data(), PLCP.data(), n) == 0;
#endif
}

// Threads for libsais: about 3/4 of the performance cores. An efficiency core holds up each of its
// parallel steps, and its spinning threads need some room: on an M1 Max, 6 threads beat 8 and 10.
int saisThreads(int poolThreads) {
    int perf = 0;
    size_t len = sizeof perf;
    if (sysctlbyname("hw.perflevel0.physicalcpu", &perf, &len, nullptr, 0) != 0 || perf <= 0) return poolThreads;
    return std::max(1, std::min(poolThreads, perf * 3 / 4));
}

template <class I>
void tier3Impl(const std::vector<uint8_t>& R, int64_t kRep, const LenRange& r, WorkPool& pool, Sink& sink, Tier3Stats& st) {
    const I n = (I)R.size();
    const uint8_t* T = R.data();
    double t0 = nowSec();
    std::vector<I> SA, PLCP;
    if (!buildSaPlcp(T, n, SA, PLCP, saisThreads(pool.size()))) throw std::runtime_error("libsais failed");
    st.sais += nowSec() - t0;
    t0 = nowSec();

    const int64_t dLo = std::max<int64_t>(kRep, r.kmin - 2);
    int64_t maxLcp = 0;
    for (I i = 0; i < n; ++i) maxLcp = std::max<int64_t>(maxLcp, PLCP[i]);  // the values of LCP, permuted
    auto lcp = [&](I i) -> int64_t { return PLCP[SA[i]]; };
    const int64_t dHi = std::min<int64_t>(r.Kmax - 2, maxLcp);
    if (dLo > dHi) {
        st.scan += nowSec() - t0;
        return;
    }
    const int64_t F = dLo - 1, C = dHi + 1;  // depths are clamped into [F, C]

    // Chunks never cut inside a node of depth >= dLo.
    std::vector<I> cuts{0};
    const I target = 1 << 20;
    for (I i = target; i < n;) {
        while (i < n && lcp(i) >= dLo) ++i;
        if (i >= n) break;
        cuts.push_back(i);
        i += target;
    }
    cuts.push_back(n);

    struct Frame {
        int64_t depth;
        uint32_t P;
        uint64_t rep;  // any member, used to spell W and to read the base after it
    };
    auto code = [&](int64_t pos) -> uint32_t { return T[pos] ? T[pos] - 1u : 4u; };
    // Frames at the clamped floor are never reported, and a suffix shorter than F may attach to one.
    auto attach = [&](Frame& f, uint32_t L, uint64_t rep) {
        if (f.depth > F) f.P |= L << (5 * code(rep + f.depth));
    };
    runOrdered(pool, cuts.size() - 1, [&](size_t ci, ChunkOut& out) {
        const I i0 = cuts[ci], i1 = cuts[ci + 1];
        std::vector<Frame> stack;
        stack.reserve(256);
        stack.push_back({F - 1, 0, 0});
        std::vector<char> W;
        auto emit = [&](const Frame& f) {
            const uint32_t miss = missMask(f.P);
            if (!miss) return;
            out.counts.add(f.depth + 2, __builtin_popcount(miss));
            if (sink.countOnly()) return;
            W.resize(f.depth + 48);
            for (int64_t d = 0; d < f.depth; ++d) W[d] = kACGT[T[f.rep + d] - 1];
            emitLines(out.text, W.data(), f.depth, miss);
        };
        constexpr I kAhead = 32;  // PLCP[SA[i]] and the base before each suffix are random reads
        for (I i = i0; i < i1; ++i) {
            if (i + kAhead < i1) {
                const I q = SA[i + kAhead];
                __builtin_prefetch(&PLCP[q]), __builtin_prefetch(&T[q ? q - 1 : 0]);
            }
            int64_t pos = SA[i];
            uint32_t L = 1u << (pos ? code(pos - 1) : 4u);
            uint64_t rep = pos;
            int64_t h = i + 1 < i1 ? std::clamp<int64_t>(lcp(i + 1), F, C) : F - 1;
            while (stack.back().depth > h) {
                Frame f = stack.back();
                stack.pop_back();
                attach(f, L, rep);
                if (f.depth >= dLo && f.depth <= dHi) emit(f);
                L = leftSet(f.P);
                rep = f.rep;
            }
            if (stack.back().depth != h) stack.push_back({h, 0, rep});
            attach(stack.back(), L, rep);
        }
    }, sink);
    st.scan += nowSec() - t0;
}
}  // namespace

void tier3(const std::vector<uint8_t>& R, int64_t kRep, const LenRange& r, WorkPool& pool, Sink& sink, Tier3Stats& st) {
    if (R.empty()) return;
#if defined(LIBSAIS_OPENMP)
    // libsais runs many short parallel steps. This libomp puts its threads to sleep as soon as a step
    // ends, and waking them costs more than most steps take. Spinning for 1 ms bridges the gaps between
    // steps (1.7x faster on chr21) and, unlike longer spins, stays out of the scan that follows.
    kmp_set_blocktime(1);
#endif
    // MAWMETAL_SAIS64=1 forces the 64-bit path, to test it on small inputs.
    if (R.size() < (size_t)INT32_MAX && !getenv("MAWMETAL_SAIS64"))
        tier3Impl<int32_t>(R, kRep, r, pool, sink, st);
    else
        tier3Impl<int64_t>(R, kRep, r, pool, sink, st);
}
