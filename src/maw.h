// Copyright (c) 2026 Nicolas Innocenti
// SPDX-License-Identifier: MIT
#pragma once
#include <algorithm>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>
#include <new>
#include <string>
#include <thread>
#include <vector>

// Allocates whole 16 KB pages, so Metal can wrap the memory without copying it.
template <class T>
struct PageAlloc {
    using value_type = T;
    static constexpr size_t kPage = 16384;
    PageAlloc() = default;
    template <class U>
    PageAlloc(const PageAlloc<U>&) {}
    static size_t bytesFor(size_t n) { return (n * sizeof(T) + kPage - 1) / kPage * kPage; }
    T* allocate(size_t n) {
        void* p = nullptr;
        if (posix_memalign(&p, kPage, bytesFor(n)) != 0) throw std::bad_alloc();
        return static_cast<T*>(p);
    }
    void deallocate(T* p, size_t) { free(p); }
    template <class U>
    bool operator==(const PageAlloc<U>&) const { return true; }
    template <class U>
    bool operator!=(const PageAlloc<U>&) const { return false; }
};
using PageVec = std::vector<uint32_t, PageAlloc<uint32_t>>;

// The input as a concatenation of segments: maximal runs of ACGT. Anything else (N, IUPAC codes,
// record boundaries, the junction with a reverse complement) only separates segments, so no word
// spanning it is considered present.
struct Text {
    uint64_t n = 0;                   // number of bases
    PageVec packed;                   // 16 bases per word, first base in bits 31..30; A=0 C=1 G=2 T=3
    PageVec segBits;                  // bit p set when a segment starts at p; bit n set too
    std::vector<uint64_t> segCoarse;  // bit b set when a segment starts in [64b, 64b + 64) (or n does)
    std::vector<uint64_t> segStarts;  // segment starts, followed by n
    int base(uint64_t p) const { return (packed[p >> 4] >> (30 - 2 * (p & 15))) & 3; }
    // segCoarse mostly stays in cache (12 MB for GRCh38 with both strands, though 47 MB for 24 Gbp), so
    // most lookups never touch segBits.
    bool coarse(uint64_t p) const { return (segCoarse[p >> 12] >> ((p >> 6) & 63)) & 1; }
    bool segStart(uint64_t p) const { return coarse(p) && ((segBits[p >> 5] >> (p & 31)) & 1); }
    void buildCoarse();
};

class WorkPool;

// Packs FASTA sequence bytes into a Text, in parallel on the pool for large inputs.
class TextBuilder {
public:
    TextBuilder(Text& t, WorkPool* pool) : t_(t), pool_(pool) {}
    void reserveBases(uint64_t bases);
    // A record starts: its first base starts a segment.
    void beginRecord();
    // Raw sequence bytes of the current record: bases (ACGT, either case) are packed, line breaks
    // (LF, CR before LF) skipped, and anything else separates segments.
    void addBytes(const char* s, size_t len);
    // Whole FASTA records, for --joint: as addBytes, except that header lines (starting with '>')
    // only separate segments. Packs on all cores however small the records are; returns the number
    // of headers.
    uint64_t addRecords(const char* s, size_t len);
    // Appends the reverse complement of the current record's bases, as a new segment.
    void addRevComp();
    uint64_t recordChars() const { return recChars_; }  // sequence characters in the current record
    void finish();
    void reset();  // empties the text, keeping its storage
    uint64_t pieceBytes = 1 << 20;  // inputs shorter than this are packed on the calling thread

private:
    template <bool kHeaders>
    uint64_t add(const char* s, size_t len);
    Text& t_;
    WorkPool* pool_;
    bool inSeg_ = false, pendingCR_ = false;
    bool inHeader_ = false, atLineStart_ = true;  // addRecords: where the next span starts
    uint64_t recStart_ = 0, recChars_ = 0;
    size_t recSeg_ = 0;  // index in segStarts of the current record's first segment
};

// FASTA records from a file: mapped when it is a plain regular file; gzip files decompressed with
// libdeflate (BGZF blocks in parallel, in windows); anything else (pipes, multi-member gzip) read
// through zlib on a separate thread.
class FastaInput {
public:
    FastaInput(const std::string& path, WorkPool* pool);
    ~FastaInput();
    bool ok() const { return ok_; }
    uint64_t sizeHint() const;  // uncompressed size when known in advance, else 0
    // Reads the next record: sets id and feeds its sequence to tb (after tb.beginRecord()).
    bool next(std::string& id, TextBuilder& tb);
    // Reads every record into tb as one (--joint), headers separating segments; returns how many.
    uint64_t readAll(TextBuilder& tb);

private:
    bool fill();
    bool readLine(std::string& out);
    bool openGzip(const unsigned char* data, size_t len);
    void release(bool all);  // gives back the input before p_, or all of it
    struct Stream;
    struct Bgzf;
    bool ok_ = false;
    WorkPool* pool_;
    const char* map_ = nullptr;  // the mapped file
    uint64_t mapLen_ = 0;
    char* whole_ = nullptr;      // or the whole decompressed file, mapped anonymously (wholeCap_ bytes)
    uint64_t wholeLen_ = 0, wholeCap_ = 0;
    uint64_t released_ = 0;      // bytes at the start of whole_ (or of a plain map) given back
    bool wholeDone_ = false;
    Stream* stream_ = nullptr;
    Bgzf* bgzf_ = nullptr;
    const char *p_ = nullptr, *end_ = nullptr;
    bool atLineStart_ = true;
};

struct Options {
    std::string input, output;
    int kmin = 2;
    int64_t Kmax = INT32_MAX;
    bool revcomp = false, joint = false, list = false, verbose = false, wide = false;
    int tier1 = 0;  // counts mode: tier 1 of each batch on 0 whichever is less busy, 1 the CPU, 2 the GPU
    int m = 0;              // region prefix length (0 = auto)
    int kRep = 0;           // groups of suffixes sharing kRep bases go to the deep stage (0 = auto)
    int64_t deep = 1024;    // depth the deep stage refines groups to; longer repeats go to libsais
    uint64_t batchCap = 0;  // max entries per GPU batch (0 = auto)
    uint64_t firstCap = 0;  // max entries in the first GPU batch (0 = auto)
    uint64_t chunk = 0;     // tier-1 chunk target in keys (0 = auto)
    uint64_t piece = 0;     // FASTA bytes per packing piece (0 = auto)
    uint64_t pending = 0;   // --list: output bytes workers may run ahead of the writer by (0 = auto)
    uint64_t spill = 0;     // --list: the awaited chunk writes its output every so many bytes (0 = auto)
    int threads = 0;
};

// ---- output -------------------------------------------------------------------------------------
struct LenCounts {
    std::vector<uint64_t> v;
    void add(size_t len, uint64_t c) {
        if (len >= v.size()) v.resize(len + 1, 0);
        v[len] += c;
    }
    void addRange(const uint64_t* c, size_t lo, size_t hi) {  // v[l] += c[l] for l in [lo, hi)
        if (hi > v.size()) v.resize(hi, 0);
        for (size_t l = lo; l < hi; ++l) v[l] += c[l];
    }
    void merge(const LenCounts& o) {
        if (o.v.size() > v.size()) v.resize(o.v.size(), 0);
        for (size_t i = 0; i < o.v.size(); ++i) v[i] += o.v[i];
    }
    uint64_t total() const {
        uint64_t s = 0;
        for (uint64_t c : v) s += c;
        return s;
    }
};

// Growable byte buffer that, unlike std::string, never zero-fills what it is about to overwrite.
// reserve() keeps kSlack writable bytes past the requested size for fixed-width stores.
// With a spill function set, reserve() offers it the content every so many bytes; when it returns
// true it has written the content and the buffer starts over, so callers must not keep pointers into
// the buffer across reserve() calls.
class OutBuf {
public:
    static constexpr size_t kSlack = 64;
    using Spill = std::function<bool(const char*, size_t)>;
    OutBuf() = default;
    OutBuf(const OutBuf&) = delete;
    OutBuf& operator=(const OutBuf&) = delete;
    OutBuf(OutBuf&& o) noexcept { *this = std::move(o); }
    OutBuf& operator=(OutBuf&& o) noexcept {
        std::swap(p_, o.p_), std::swap(n_, o.n_), std::swap(cap_, o.cap_), std::swap(lim_, o.lim_);
        std::swap(spillAt_, o.spillAt_), std::swap(every_, o.every_), std::swap(spill_, o.spill_);
        return *this;
    }
    ~OutBuf() { free(p_); }
    char* reserve(size_t extra) {
        if (n_ + extra + kSlack > lim_) slow(n_ + extra + kSlack);
        return p_ + n_;
    }
    void spill(const Spill* s, size_t every = 0) {  // nullptr: none
        spill_ = s, every_ = every, spillAt_ = s ? n_ + every : SIZE_MAX;
        lim_ = std::min(cap_, spillAt_);
    }
    size_t capacity() const { return cap_; }
    void commit(size_t k) { n_ += k; }
    void append(const char* s, size_t k) {
        memcpy(reserve(k), s, k);
        n_ += k;
    }
    const char* data() const { return p_; }
    size_t size() const { return n_; }
    bool empty() const { return n_ == 0; }
    void clear() { n_ = 0; }

private:
    // need passed the spill point or the capacity, the lower of which is lim_.
    void slow(size_t need) {
        if (need > spillAt_) {
            if ((*spill_)(p_, n_)) need -= n_, n_ = 0;
            spillAt_ = n_ + every_;
        }
        if (need > cap_) grow(need);
        lim_ = std::min(cap_, spillAt_);
    }
    void grow(size_t need) {
        size_t c = std::max<size_t>({need, cap_ * 2, 1 << 16});
        char* q = static_cast<char*>(realloc(p_, c));
        if (!q) throw std::bad_alloc();
        p_ = q, cap_ = c;
    }
    char* p_ = nullptr;
    size_t n_ = 0, cap_ = 0, lim_ = 0, spillAt_ = SIZE_MAX, every_ = 0;
    const Spill* spill_ = nullptr;
};

struct ChunkOut {
    OutBuf text;
    LenCounts counts;
};

class Sink {
public:
    Sink(FILE* f, bool countOnly) : f_(f), countOnly_(countOnly) {}
    void put(ChunkOut& c);
    void raw(const char* s, size_t n);
    void raw(const std::string& s) { raw(s.data(), s.size()); }
    bool countOnly() const { return countOnly_; }
    size_t spillBytes = 1 << 20;  // the chunk next in line writes its output every this many bytes
    LenCounts record;  // counts since the last takeRecord()
    LenCounts total;
    LenCounts takeRecord();

private:
    FILE* f_;
    bool countOnly_;
};

// Persistent worker threads (at the highest QoS, so they prefer performance cores). start() runs
// job(worker) once on every worker; wait() returns when all of them have returned.
class WorkPool {
public:
    explicit WorkPool(int threads);
    ~WorkPool();
    int size() const { return (int)threads_.size(); }
    void start(std::function<void(int)> job);
    void wait();
    void run(std::function<void(int)> job) {
        start(std::move(job));
        wait();
    }

private:
    void loop(int id);
    std::vector<std::thread> threads_;
    std::mutex mu_;
    std::condition_variable cvJob_, cvDone_;
    std::function<void(int)> job_;
    uint64_t gen_ = 0;
    int active_ = 0;
    bool quit_ = false;
};

// Runs fn(i) for i in [0, n) on the pool.
void parallelFor(WorkPool& pool, size_t n, const std::function<void(size_t)>& fn);

// Runs work(i, out) for i in [0, n) on the pool; outputs reach the sink in index order.
void runOrdered(WorkPool& pool, size_t n, const std::function<void(size_t, ChunkOut&)>& work, Sink& sink);

// ---- algorithms (cpu.cpp) -------------------------------------------------------------------------
struct LenRange {
    int kmin;
    int64_t Kmax;
};

// The census records which words of length <= kCensusLen occur.
constexpr int kCensusLen = 8;
constexpr uint32_t kCensusPresWords = (((1u << (2 * (kCensusLen + 1))) - 4) / 3 + 31) / 32;

// MAWs of length 2..kCensusLen (within r) from the census presence bits: bit (4^l - 4)/3 + w is set
// when w of length l is the longest prefix, up to kCensusLen, of some position's valid suffix.
void bitmapTier(const std::vector<uint32_t>& presence, const LenRange& r, bool countOnly, ChunkOut& out);

// A piece of one sorted region: keys, and (when carried) the positions in the same order: low 32
// bits in vals, bits 32..39 in hi for texts of 2^32 bases or more (hi is null otherwise).
struct T1Chunk {
    const uint64_t* keys;
    const uint32_t* vals;
    const uint8_t* hi;
    size_t n;
    uint32_t prefix;
    uint64_t pos(size_t i) const { return vals[i] | (hi ? (uint64_t)hi[i] << 32 : 0); }
};
// Splits a region where adjacent suffixes share fewer than cutBelow bases, into pieces of about
// target keys.
void planTier1Chunks(const T1Chunk& region, int m, int cutBelow, size_t target, std::vector<T1Chunk>& out);

// Groups of suffixes sharing kRep bases: (first index, count) pairs, count >= 2.
using Groups = std::vector<uint32_t>;

// Tier 1: MAWs aWb with dLo <= |W| <= dHi (< kRep) from the sorted keys alone. With groups non-null,
// also collects the groups of suffixes sharing kRep bases.
void scanTier1(const T1Chunk& c, int m, int dLo, int dHi, int kRep, bool countOnly, ChunkOut& out, Groups* groups);
void findGroups(const T1Chunk& c, int m, int kRep, Groups& groups);
// The same from bits (bit i: entries i and i+1 share kRep bases), for entries [off, off + n).
void groupsFromBits(const uint32_t* bits, size_t off, size_t n, Groups& groups);

// Deep stage: each group of suffixes sharing kRep bases is sorted further from the text, down to
// depth D, giving the MAWs with dLo <= |W| <= dHi (kRep <= dLo, dHi < D). Positions whose suffix
// still ties another at depth D (they start a repeated D-mer) are set in marks.
struct DeepParams {
    int m, kRep;
    int64_t D, dLo, dHi;
    bool countOnly;
    uint32_t* marks;
};
class DeepScanner;
// Per-worker scratch space for the deep stage, reused across chunks.
class DeepWork {
public:
    DeepWork(const Text& t, const DeepParams& p);
    ~DeepWork();
    void run(const T1Chunk& c, const Groups& groups, ChunkOut& out);

private:
    DeepScanner* s_;
};

// Residue: the runs covered by marked k-mers, extended by one base, as bytes 1..4 (ACGT), each
// run ended by 0.
void buildResidue(const Text& t, const uint32_t* marks, int64_t k, std::vector<uint8_t>& R);

// Tier 3: MAWs aWb with |W| >= kRep from the residue.
struct Tier3Stats {
    double sais = 0, scan = 0;
};
void tier3(const std::vector<uint8_t>& R, int64_t kRep, const LenRange& r, WorkPool& pool, Sink& sink, Tier3Stats& st);

double nowSec();
