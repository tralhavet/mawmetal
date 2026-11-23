// Copyright (c) 2026 Nicolas Innocenti
// SPDX-License-Identifier: MIT
// FASTA input: reading records and packing their bases.
#include <arm_neon.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <zlib.h>
#if defined(HAVE_LIBDEFLATE)
#include "libdeflate.h"
#endif

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <condition_variable>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>

#include "maw.h"

namespace {
// Byte classes: 0..3 a base (A C G T, either case), 4 a separator, 5 LF, 6 CR.
struct ClassTable {
    uint8_t v[256];
    ClassTable() {
        memset(v, 4, sizeof v);
        v['A'] = v['a'] = 0, v['C'] = v['c'] = 1, v['G'] = v['g'] = 2, v['T'] = v['t'] = 3;
        v['\n'] = 5, v['\r'] = 6;
    }
};
const ClassTable kClass;

// Mask of the bytes of q[0, 16) that are bases.
inline uint8x16_t baseMask(uint8x16_t c) {
    const uint8x16_t u = vandq_u8(c, vdupq_n_u8(0xDF));
    return vorrq_u8(vorrq_u8(vceqq_u8(u, vdupq_n_u8('A')), vceqq_u8(u, vdupq_n_u8('C'))),
                    vorrq_u8(vceqq_u8(u, vdupq_n_u8('G')), vceqq_u8(u, vdupq_n_u8('T'))));
}

// 16 bases (all of q[0, 16) are bases) packed into a word, first base in bits 31..30.
inline uint32_t pack16(uint8x16_t c) {
    // A C G T map to 0 1 2 3 in either case.
    const uint8x16_t code = vandq_u8(veorq_u8(vshrq_n_u8(c, 1), vshrq_n_u8(c, 2)), vdupq_n_u8(3));
    static const int8_t kShift[16] = {6, 4, 2, 0, 6, 4, 2, 0, 6, 4, 2, 0, 6, 4, 2, 0};
    const uint8x16_t sh = vshlq_u8(code, vld1q_s8(kShift));
    const uint32x4_t quads = vpaddlq_u16(vpaddlq_u8(sh));  // each lane: one byte of 4 bases
    const uint16x4_t n16 = vmovn_u32(quads);
    const uint8x8_t n8 = vmovn_u16(vcombine_u16(n16, n16));
    return __builtin_bswap32(vget_lane_u32(vreinterpret_u32_u8(n8), 0));
}

inline void atomicOr(uint32_t* w, uint32_t v) { __atomic_fetch_or(w, v, __ATOMIC_RELAXED); }

// Offset of the last LF in s[a, b), or -1 (macOS has no memrchr).
int64_t lastLf(const uint8_t* s, size_t a, size_t b) {
    size_t i = b;
    for (; i >= a + 16; i -= 16)
        if (vmaxvq_u8(vceqq_u8(vld1q_u8(s + i - 16), vdupq_n_u8('\n'))))
            for (size_t j = i; j-- > i - 16;)
                if (s[j] == '\n') return (int64_t)j;
    while (i-- > a)
        if (s[i] == '\n') return (int64_t)i;
    return -1;
}

struct PieceResult {
    uint64_t bases = 0, chars = 0, records = 0;
    int last = 0;  // the last character that is not a line break: 0 none, 1 a base, 2 a separator
    bool endInHeader = false;  // the piece ends inside a header line
};

// Scans s[a, b) of a span of len bytes. With P set, packs its bases from position pos on (and
// sets segment starts in SB and starts): words wholly inside the piece are stored, the partial
// ones at either end ORed in, since neighbouring pieces share them. A CR at the end of the span
// counts as a line break here; the caller settles it with the next span.
// With kHeaders the span holds whole records: a line that starts with '>' is a header, skipped as a
// separator and counted. inHeader and atLineStart then say where s[a] lies.
template <bool kPack, bool kHeaders>
PieceResult scanPiece(const uint8_t* s, size_t a, size_t b, size_t len, uint64_t pos, bool inSeg, bool inHeader,
                      bool atLineStart, uint32_t* P, uint32_t* SB, std::vector<uint64_t>* starts) {
    PieceResult r;
    const uint64_t firstWord = pos >> 4;
    const bool sharedFirst = (pos & 15) != 0;
    uint32_t cur = 0;
    auto store = [&](uint64_t w, uint32_t x) {
        if (w == firstWord && sharedFirst) atomicOr(&P[w], x);
        else P[w] = x;
    };
    auto put = [&](uint32_t v, int k) {  // k bases in the top 2k bits of v
        if (kPack) {
            const int sl = (int)(pos & 15);
            cur |= v >> (2 * sl);
            if (sl + k >= 16) {
                store(pos >> 4, cur);
                cur = sl ? v << (32 - 2 * sl) : 0;
            }
        }
        pos += k;
    };
    auto segStart = [&] {
        if (kPack) {
            starts->push_back(pos);
            atomicOr(&SB[pos >> 5], 1u << (pos & 31));
        }
    };
    size_t i = a;
    // Moves i past the end of the header line it is in; false when the line goes on past b.
    auto skipLine = [&] {
        const void* nl = memchr(s + i, '\n', b - i);
        if (!nl) return i = b, r.endInHeader = true, false;
        i = (size_t)((const uint8_t*)nl - s) + 1, atLineStart = true;
        return true;
    };
    if (kHeaders && inHeader && !skipLine()) return r;
    while (i < b) {
        if (kHeaders && atLineStart && s[i] == '>') {
            inSeg = false;
            ++r.records, r.last = 2;
            if (!skipLine()) break;
            continue;
        }
        if (i + 16 <= b) {
            const uint8x16_t c = vld1q_u8(s + i);
            if (vminvq_u8(baseMask(c)) == 0xFF) {
                if (!inSeg) segStart(), inSeg = true;
                if (kPack) put(pack16(c), 16);
                else pos += 16;
                r.bases += 16, r.chars += 16, r.last = 1;
                i += 16;
                atLineStart = false;
                continue;
            }
        }
        // A block with line breaks or separators, or the tail: byte by byte.
        const size_t e = std::min(b, i + 16);
        for (; i < e; ++i) {
            if (kHeaders && atLineStart && s[i] == '>') break;  // a header: the loop above takes it
            const uint8_t c = kClass.v[s[i]];
            atLineStart = c == 5;
            if (c < 4) {
                if (!inSeg) segStart(), inSeg = true;
                put((uint32_t)c << 30, 1);
                ++r.bases, ++r.chars, r.last = 1;
            } else if (c == 5 || (c == 6 && (i + 1 == len || s[i + 1] == '\n'))) {
                // line break
            } else {
                inSeg = false;
                ++r.chars, r.last = 2;
            }
        }
    }
    if (kPack && (pos & 15)) atomicOr(&P[pos >> 4], cur);
    return r;
}

// Reverses the order of the 16 two-bit fields of x.
inline uint32_t rev2(uint32_t x) {
    x = __builtin_bitreverse32(x);
    return ((x >> 1) & 0x55555555u) | ((x & 0x55555555u) << 1);
}

// 16 bases starting at p, first base in bits 31..30.
inline uint32_t window16(const uint32_t* T, uint64_t p) {
    const uint64_t w = p >> 4;
    const unsigned o = (unsigned)(p & 15) * 2;
    const uint64_t v = ((uint64_t)T[w] << 32) | T[w + 1];
    return (uint32_t)(v >> (32 - o));
}

template <class V>
void growZeroed(V& v, size_t n) {
    if (v.size() < n) {
        if (v.capacity() < n) v.reserve(std::max(n, v.capacity() + v.capacity() / 2));
        v.resize(n, 0);
    }
}
}  // namespace

// ---- TextBuilder ------------------------------------------------------------------------------------
void TextBuilder::reserveBases(uint64_t bases) {
    t_.packed.reserve(bases / 16 + 16);
    t_.segBits.reserve(bases / 32 + 16);
}

void TextBuilder::beginRecord() {
    inSeg_ = pendingCR_ = inHeader_ = false;
    atLineStart_ = true;
    recStart_ = t_.n;
    recSeg_ = t_.segStarts.size();
    recChars_ = 0;
}

void TextBuilder::addBytes(const char* s, size_t len) { add<false>(s, len); }

uint64_t TextBuilder::addRecords(const char* s, size_t len) { return add<true>(s, len); }

template <bool kHeaders>
uint64_t TextBuilder::add(const char* sc, size_t len) {
    if (!len) return 0;
    const uint8_t* s = (const uint8_t*)sc;
    if (pendingCR_) {  // a CR ended the previous span: it was a separator unless an LF follows
        pendingCR_ = false;
        if (s[0] != '\n') inSeg_ = false, ++recChars_;
    }
    const uint64_t n0 = t_.n;
    const size_t P = pool_ && len >= 2 * pieceBytes ? std::min<size_t>(4 * pool_->size(), len / pieceBytes) : 1;
    std::vector<size_t> cut(P + 1);
    for (size_t k = 0; k <= P; ++k) cut[k] = len * k / P;
    // With headers: where each piece starts, inside a header line or at a line start. A piece starts
    // inside a header when the line in progress there (after the last LF of an earlier piece, or the
    // span's first line) starts with '>'.
    std::vector<char> hdr(P, 0), ls(P, 0);
    hdr[0] = inHeader_, ls[0] = atLineStart_;
    if (kHeaders && P > 1) {
        std::vector<int64_t> lf(P);
        parallelFor(*pool_, P, [&](size_t k) { lf[k] = lastLf(s, cut[k], cut[k + 1]); });
        const bool firstHeader = inHeader_ || (atLineStart_ && s[0] == '>');
        int64_t line = -1;  // start of the line in progress, -1 for the span's first one
        for (size_t k = 1; k < P; ++k) {
            if (lf[k - 1] >= 0) line = lf[k - 1] + 1;
            ls[k] = s[cut[k] - 1] == '\n';
            hdr[k] = !ls[k] && (line < 0 ? firstHeader : s[line] == '>');
        }
    }
    std::vector<uint64_t> pos(P);
    std::vector<char> seg(P);
    uint64_t bound = len;  // bases, at most
    if (P > 1) {
        // Pass 1 counts each piece's bases, so each knows where its own go.
        std::vector<PieceResult> cnt(P);
        parallelFor(*pool_, P, [&](size_t k) {
            cnt[k] = scanPiece<false, kHeaders>(s, cut[k], cut[k + 1], len, 0, false, hdr[k], ls[k], nullptr, nullptr, nullptr);
        });
        uint64_t at = n0;
        bool in = inSeg_;
        for (size_t k = 0; k < P; ++k) {
            pos[k] = at, seg[k] = in;
            at += cnt[k].bases;
            if (cnt[k].last) in = cnt[k].last == 1;
        }
        bound = at - n0;
    } else {
        pos[0] = n0, seg[0] = inSeg_;
    }
    growZeroed(t_.packed, (n0 + bound) / 16 + 2);
    growZeroed(t_.segBits, (n0 + bound) / 32 + 2);
    uint32_t* Pk = t_.packed.data();
    uint32_t* SB = t_.segBits.data();
    std::vector<std::vector<uint64_t>> starts(P);
    std::vector<PieceResult> res(P);
    auto pack = [&](size_t k) {
        res[k] = scanPiece<true, kHeaders>(s, cut[k], cut[k + 1], len, pos[k], seg[k], hdr[k], ls[k], Pk, SB, &starts[k]);
    };
    if (P > 1) parallelFor(*pool_, P, pack);
    else pack(0);
    uint64_t total = 0, records = 0;
    for (size_t k = 0; k < P; ++k) {
        total += res[k].bases, recChars_ += res[k].chars, records += res[k].records;
        if (res[k].last) inSeg_ = res[k].last == 1;
        t_.segStarts.insert(t_.segStarts.end(), starts[k].begin(), starts[k].end());
    }
    t_.n = n0 + total;
    if (s[len - 1] == '\r') pendingCR_ = true;
    inHeader_ = res[P - 1].endInHeader, atLineStart_ = s[len - 1] == '\n';
    return records;
}

void TextBuilder::addRevComp() {
    pendingCR_ = false;
    const uint64_t e = t_.n, len = e - recStart_;
    if (!len) return;
    growZeroed(t_.packed, (e + len) / 16 + 2);
    growZeroed(t_.segBits, (e + len) / 32 + 2);
    uint32_t* P = t_.packed.data();
    // Base q of the reverse complement (q in [e, e + len)) is the complement of base 2e - 1 - q.
    auto one = [&](uint64_t q) {
        const uint64_t f = 2 * e - 1 - q;
        const uint32_t c = 3 - ((P[f >> 4] >> (30 - 2 * (f & 15))) & 3);
        P[q >> 4] |= c << (30 - 2 * (q & 15));
    };
    const uint64_t qa = std::min((e + 15) / 16 * 16, e + len), qb = std::max(qa, (e + len) / 16 * 16);
    for (uint64_t q = e; q < qa; ++q) one(q);
    // Whole words: 16 source bases, reversed and complemented.
    auto words = [&](uint64_t w0, uint64_t w1) {
        for (uint64_t w = w0; w < w1; ++w) P[w] = ~rev2(window16(P, 2 * e - 16 - 16 * w));
    };
    const uint64_t wa = qa / 16, wb = qb / 16;
    if (pool_ && wb - wa > (1u << 18)) {
        const size_t k = 4 * pool_->size();
        parallelFor(*pool_, k, [&](size_t i) { words(wa + (wb - wa) * i / k, wa + (wb - wa) * (i + 1) / k); });
    } else {
        words(wa, wb);
    }
    for (uint64_t q = qb; q < e + len; ++q) one(q);
    // The record's segments, mirrored: [a, b) becomes [2e - b, 2e - a).
    const size_t last = t_.segStarts.size();
    for (size_t k = last; k-- > recSeg_;) {
        const uint64_t end = k + 1 < last ? t_.segStarts[k + 1] : e, st = 2 * e - end;
        t_.segStarts.push_back(st);
        t_.segBits[st >> 5] |= 1u << (st & 31);
    }
    t_.n = e + len;
    inSeg_ = true;
}

void TextBuilder::finish() {
    // Words past the last base are zero (padding reads rely on it).
    t_.packed.resize(t_.n / 16 + 8, 0);
    t_.segBits.resize(t_.n / 32 + 8, 0);
    t_.segBits[t_.n >> 5] |= 1u << (t_.n & 31);
    t_.segStarts.push_back(t_.n);
    t_.buildCoarse();
}

void TextBuilder::reset() {
    t_.n = 0;
    t_.packed.clear(), t_.segBits.clear(), t_.segStarts.clear(), t_.segCoarse.clear();
    inSeg_ = pendingCR_ = inHeader_ = false;
    atLineStart_ = true;
    recStart_ = recChars_ = 0;
    recSeg_ = 0;
}

// ---- FastaInput -------------------------------------------------------------------------------------
// Reads through zlib (gzip or plain) on its own thread into two alternating buffers.
struct FastaInput::Stream {
    static constexpr size_t kBuf = 32 << 20;
    gzFile f = nullptr;
    std::vector<char> buf[2];
    size_t len[2] = {0, 0};
    bool full[2] = {false, false};
    bool eof = false, quit = false;
    std::string error;
    int next = 0;     // buffer the consumer takes next
    int held = -1;    // buffer the consumer is reading
    std::mutex mu;
    std::condition_variable cv;
    std::thread th;

    explicit Stream(gzFile g) : f(g) {
        buf[0].resize(kBuf), buf[1].resize(kBuf);
        th = std::thread([this] { run(); });
    }
    ~Stream() {
        {
            std::lock_guard<std::mutex> l(mu);
            quit = true;
        }
        cv.notify_all();
        th.join();
        gzclose(f);
    }
    void run() {
        for (int b = 0;; b ^= 1) {
            {
                std::unique_lock<std::mutex> l(mu);
                cv.wait(l, [&] { return quit || (!full[b] && held != b); });
                if (quit) return;
            }
            size_t n = 0;
            bool end = false;
            std::string err;
            while (n < kBuf) {
                int r = gzread(f, buf[b].data() + n, (unsigned)std::min<size_t>(kBuf - n, 1u << 30));
                if (r < 0) {
                    int code;
                    err = gzerror(f, &code);
                    end = true;
                    break;
                }
                if (r == 0) {
                    end = true;
                    break;
                }
                n += (size_t)r;
            }
            std::lock_guard<std::mutex> l(mu);
            len[b] = n, full[b] = true;
            if (end) eof = true, error = err;
            cv.notify_all();
            if (end) return;
        }
    }
    // Hands over the next buffer that is not empty (releasing the previous one); false at the end.
    bool get(const char*& p, size_t& n) {
        std::unique_lock<std::mutex> l(mu);
        for (;;) {
            if (held >= 0) full[held] = false, held = -1, cv.notify_all();
            cv.wait(l, [&] { return full[next] || eof; });
            if (!full[next]) {
                if (!error.empty()) throw std::runtime_error("read failed: " + error);
                return false;
            }
            held = next, next ^= 1;
            p = buf[held].data(), n = len[held];
            if (n) return true;  // an empty one ends the input (empty file, or a multiple of kBuf)
        }
    }
};

#if defined(HAVE_LIBDEFLATE)
namespace {
uint32_t le32(const unsigned char* p) { return p[0] | p[1] << 8 | p[2] << 16 | (uint32_t)p[3] << 24; }

libdeflate_decompressor* decompressor() {
    thread_local std::unique_ptr<libdeflate_decompressor, void (*)(libdeflate_decompressor*)> d(
        libdeflate_alloc_decompressor(), libdeflate_free_decompressor);
    if (!d) throw std::bad_alloc();
    return d.get();
}
}  // namespace

// A BGZF file (gzip members of at most 64 KB, each with its size in a header field): blocks are
// independent, so a window of them is decompressed in parallel.
struct FastaInput::Bgzf {
    static constexpr uint64_t kWindow = 256 << 20;
    struct Block {
        uint64_t off;
        uint32_t size, isize;
    };
    std::vector<Block> blocks;
    size_t next = 0;
    std::vector<char> buf;
    const unsigned char* data;

    // False unless every member of data[0, len) is a BGZF block.
    bool parse(const unsigned char* d, size_t len) {
        data = d;
        for (uint64_t off = 0; off < len;) {
            const unsigned char* h = d + off;
            if (len - off < 28 || h[0] != 0x1f || h[1] != 0x8b || h[2] != 8 || !(h[3] & 4)) return false;
            const uint32_t xlen = h[10] | h[11] << 8;
            uint32_t bsize = 0;
            for (uint32_t x = 12; x + 4 <= 12 + xlen && x + 4 <= len - off;) {
                const uint32_t slen = h[x + 2] | h[x + 3] << 8;
                if (h[x] == 'B' && h[x + 1] == 'C' && slen == 2) bsize = (h[x + 4] | h[x + 5] << 8) + 1;
                x += 4 + slen;
            }
            if (bsize < 28 || bsize > len - off) return false;
            blocks.push_back({off, bsize, le32(h + bsize - 4)});
            off += bsize;
        }
        return !blocks.empty();
    }

    bool get(WorkPool* pool, const char*& p, size_t& n) {
        if (next == blocks.size()) return false;
        size_t end = next;
        uint64_t total = 0;
        std::vector<uint64_t> at;
        while (end < blocks.size() && (end == next || total + blocks[end].isize <= kWindow)) {
            at.push_back(total);
            total += blocks[end++].isize;
        }
        buf.resize(total);
        const size_t first = next, cnt = end - next;
        auto run = [&](size_t i0, size_t i1) {
            for (size_t i = i0; i < i1; ++i) {
                const Block& b = blocks[first + i];
                size_t got = 0;
                if (libdeflate_gzip_decompress(decompressor(), data + b.off, b.size, buf.data() + at[i], b.isize, &got) !=
                        LIBDEFLATE_SUCCESS ||
                    got != b.isize)
                    throw std::runtime_error("corrupt BGZF block");
            }
        };
        const size_t tasks = pool ? std::min<size_t>(cnt, 8 * pool->size()) : 1;
        if (tasks > 1) parallelFor(*pool, tasks, [&](size_t t) { run(cnt * t / tasks, cnt * (t + 1) / tasks); });
        else run(0, cnt);
        next = end;
        p = buf.data(), n = total;
        return true;
    }
};

// gzip input: BGZF in parallel windows, a single member in one call; false for anything else.
bool FastaInput::openGzip(const unsigned char* data, size_t len) {
    Bgzf* b = new Bgzf;
    if (b->parse(data, len)) {
        bgzf_ = b;
        return true;
    }
    delete b;
    const uint32_t isize = le32(data + len - 4);  // uncompressed size mod 2^32 of the last member
    // Mapped rather than allocated, so that release() can give it back as it is consumed.
    const size_t cap = (size_t)isize + 1;
    void* m = mmap(nullptr, cap, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);
    if (m == MAP_FAILED) return false;
    char* out = (char*)m;
    size_t in = 0, got = 0;
    if (libdeflate_gzip_decompress_ex(decompressor(), data, len, out, isize, &in, &got) == LIBDEFLATE_SUCCESS && in == len &&
        got == isize) {
        whole_ = out, wholeLen_ = got, wholeCap_ = cap;
        return true;
    }
    munmap(out, cap);  // several members, or 4 GB or more: zlib streams it
    return false;
}
#else
struct FastaInput::Bgzf {
    bool get(WorkPool*, const char*&, size_t&) { return false; }
};
bool FastaInput::openGzip(const unsigned char*, size_t) { return false; }
#endif

FastaInput::FastaInput(const std::string& path, WorkPool* pool) : pool_(pool) {
    // A regular file is mapped: plain FASTA is parsed in place, gzip decompressed from the map.
    // Anything else (pipes, /dev/stdin) goes through zlib.
    int fd = open(path.c_str(), O_RDONLY);
    if (fd >= 0) {
        struct stat st;
        if (fstat(fd, &st) == 0 && S_ISREG(st.st_mode) && st.st_size >= 28) {
            void* m = mmap(nullptr, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
            if (m != MAP_FAILED) {
                const unsigned char* d = (const unsigned char*)m;
                const size_t len = (size_t)st.st_size;
                madvise(m, len, MADV_SEQUENTIAL);
                madvise(m, len, MADV_WILLNEED);
                if (!(d[0] == 0x1f && d[1] == 0x8b)) {
                    map_ = (const char*)m, mapLen_ = len;
                    close(fd);
                    ok_ = true;
                    return;
                }
                if (openGzip(d, len)) {
                    if (bgzf_) map_ = (const char*)m, mapLen_ = len;  // kept mapped for the BGZF windows
                    else munmap(m, len);                             // decompressed whole already
                    close(fd);
                    ok_ = true;
                    return;
                }
                munmap(m, len);
            }
        }
        close(fd);
    }
    gzFile f = gzopen(path.c_str(), "rb");
    if (!f) return;
    gzbuffer(f, 1 << 20);
    stream_ = new Stream(f);
    ok_ = true;
}

FastaInput::~FastaInput() { release(true); }

// The bases before p_ are packed by now. A plain map and the whole decompressed file are given back
// page by page, the rest (BGZF window, zlib buffers, compressed map) at the end of the input.
void FastaInput::release(bool all) {
    char* base = whole_ ? whole_ : bgzf_ || stream_ ? nullptr : (char*)map_;
    if (base) {
        const uint64_t page = (uint64_t)getpagesize(), size = whole_ ? wholeCap_ : mapLen_;
        const uint64_t upto = all ? (size + page - 1) / page * page : (uint64_t)(p_ - base) / page * page;
        if (upto > released_) munmap(base + released_, upto - released_), released_ = upto;
    }
    if (!all) return;
    if (map_ && base != map_) munmap((void*)map_, mapLen_);
    map_ = nullptr, whole_ = nullptr;
    delete stream_;
    delete bgzf_;
    stream_ = nullptr, bgzf_ = nullptr;
}

uint64_t FastaInput::sizeHint() const {
#if defined(HAVE_LIBDEFLATE)
    if (bgzf_) {
        uint64_t n = 0;
        for (const auto& b : bgzf_->blocks) n += b.isize;
        return n;
    }
#endif
    if (whole_) return wholeLen_;
    return stream_ ? 0 : mapLen_;
}

bool FastaInput::fill() {
    if (bgzf_) {
        size_t n;
        if (!bgzf_->get(pool_, p_, n)) return false;
        end_ = p_ + n;
        return true;
    }
    if (whole_ || (map_ && !stream_)) {
        if (wholeDone_) return false;
        wholeDone_ = true;
        p_ = whole_ ? whole_ : map_;
        end_ = p_ + (whole_ ? wholeLen_ : mapLen_);
        return true;
    }
    if (!stream_) return false;
    size_t n;
    if (!stream_->get(p_, n)) return false;
    end_ = p_ + n;
    return true;
}

bool FastaInput::readLine(std::string& out) {
    out.clear();
    bool any = false;
    for (;;) {
        if (p_ == end_ && !fill()) break;
        any = true;
        const char* nl = (const char*)memchr(p_, '\n', end_ - p_);
        if (nl) {
            out.append(p_, nl - p_);
            p_ = nl + 1;
            atLineStart_ = true;
            break;
        }
        out.append(p_, end_ - p_);
        atLineStart_ = false;
        p_ = end_;
    }
    if (!out.empty() && out.back() == '\r') out.pop_back();
    return any;
}

bool FastaInput::next(std::string& id, TextBuilder& tb) {
    std::string line;
    do {
        if (!readLine(line)) return false;
    } while (line.empty());
    if (line[0] != '>') throw std::runtime_error("input is not in FASTA format");
    id = line.substr(1);
    tb.beginRecord();
    // The sequence runs up to the next '>' at the start of a line.
    for (;;) {
        if (p_ == end_ && !fill()) {
            release(true);  // that was the last record
            return true;
        }
        const char* hit = nullptr;
        for (const char* q = p_; (q = (const char*)memchr(q, '>', end_ - q)); ++q)
            if (q == p_ ? atLineStart_ : q[-1] == '\n') {
                hit = q;
                break;
            }
        if (hit) {
            tb.addBytes(p_, hit - p_);
            p_ = hit;
            atLineStart_ = true;
            release(false);
            return true;
        }
        tb.addBytes(p_, end_ - p_);
        atLineStart_ = end_[-1] == '\n';
        p_ = end_;
    }
}

uint64_t FastaInput::readAll(TextBuilder& tb) {
    for (;;) {  // the first line that is not empty must be a header
        if (p_ == end_ && !fill()) return 0;
        while (p_ < end_ && (*p_ == '\n' || *p_ == '\r')) ++p_;
        if (p_ < end_) break;
    }
    if (*p_ != '>') throw std::runtime_error("input is not in FASTA format");
    tb.beginRecord();
    uint64_t records = 0;
    do {
        records += tb.addRecords(p_, end_ - p_);
        p_ = end_;
        release(false);
    } while (fill());
    release(true);
    return records;
}
