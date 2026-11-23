// Copyright (c) 2026 Nicolas Innocenti
// SPDX-License-Identifier: MIT
#include <metal_stdlib>
using namespace metal;

// Layout conventions shared with the host (gpu.mm):
//  * Packed text T: 16 bases per uint, first base in bits 31..30, A=0 C=1 G=2 T=3.
//    Padded with zero words so reads a few words past the last base are safe.
//  * Segment bitmap S: bit p (word p>>5, bit p&31) is set when a segment (maximal ACGT run)
//    starts at p. Bit n is set as well. Padded with zero words.
//  * Sort key: bits 63..8 hold bases m..m+27 of the suffix (first one in bits 63..62, zero past
//    the segment end), bits 7..3 hold min(valid length, m+28) - m, bits 2..0 the preceding base
//    (4 = none). Only the top 2*S bits (S sorted bases) are radix-sorted. Suffixes with fewer than
//    S valid bases ("short") are first ordered by length at the front of their region, so the
//    stable sort places a suffix that ends before its zero padding first, as with a sentinel.

#define TG 256
#define NSG (TG / 32)
#define RB 9
#define RADIX (1 << RB)
#define KPT 8                 // keys per thread in a sort tile
#define TILE (TG * KPT)
#define SG_KEYS (TILE / NSG)  // contiguous keys ranked by one simdgroup
#define SCAN_IPT 16
#define SCAN_BLOCK (TG * SCAN_IPT)
#define MAX_REGIONS 1024
#define EX_WPT 2              // text words (16 positions each) per thread in extract
#define CENSUS_LEN 8          // the census records which words up to this length occur
#define CENSUS_PRES_WORDS 2731
#define KEY_BASES 28
// Positions are carried as a uint (low 32 bits) plus, for texts of 2^32 bases or more, a uchar
// with bits 32..39 in a separate array (withHi).

// dstOff shifts scatter destinations: the scan packs the tiles' regions back to back from 0.
struct Tile { uint start; uint end; uint histBase; uint stride; uint regionStart; uint dstOff; };
struct CensusParams { ulong n; uint m; uint S; uint nbins; };
struct ExtractParams { ulong n; uint m; uint S; uint lo; uint hi; uint withVals; uint withHi; };
struct SortParams { uint shift; uint mask; uint withVals; uint withHi; };

inline uint ctz64(ulong x) { uint lo = uint(x); return lo ? ctz(lo) : 32 + ctz(uint(x >> 32)); }

// 32 bases starting at p, first base in bits 63..62. Word and bitmap indices fit in 32 bits for
// texts below 2^36 bases.
inline ulong window32(device const uint* T, ulong p) {
    uint w = uint(p >> 4), o = uint(p & 15) << 1;
    ulong v = (ulong(T[w]) << 32) | ulong(T[w + 1]);
    if (o) v = (v << o) | ulong(T[w + 2] >> (32 - o));
    return v;
}

// Number of bases from p to the end of its segment, capped at cap (cap <= 33).
inline uint valid_len(device const uint* S, ulong p, ulong n, uint cap) {
    ulong q = p + 1;
    uint w = uint(q >> 5), o = uint(q & 31);
    ulong bits = (ulong(S[w]) | (ulong(S[w + 1]) << 32)) >> o;
    uint d = bits ? ctz64(bits) + 1 : 64;
    return uint(min(ulong(min(d, cap)), n - p));
}

inline uint base_at(device const uint* T, ulong p) { return (T[uint(p >> 4)] >> (30 - 2 * uint(p & 15))) & 3; }
inline bool seg_start(device const uint* S, ulong p) { return (S[uint(p >> 5)] >> uint(p & 31)) & 1; }

// Lanes of the simdgroup holding the same RB-bit digit (restricted to valid lanes).
inline uint peer_mask(uint d, bool valid) {
    uint peers = uint(static_cast<ulong>(simd_ballot(valid)));
    for (uint b = 0; b < RB; ++b) {
        bool bit = (d >> b) & 1;
        uint v = uint(static_cast<ulong>(simd_ballot(bit)));
        peers &= bit ? v : ~v;
    }
    return peers;
}

// Exclusive scan of one value per thread over a TG-sized threadgroup. tmp needs NSG + 1 slots.
inline uint tg_exclusive_scan(uint v, uint lane, uint sg, threadgroup uint* tmp, thread uint& total) {
    uint incl = simd_prefix_inclusive_sum(v);
    if (lane == 31) tmp[sg] = incl;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        uint s = lane < NSG ? tmp[lane] : 0;
        uint si = simd_prefix_inclusive_sum(s);
        if (lane < NSG) tmp[lane] = si - s;
        if (lane == NSG - 1) tmp[NSG] = si;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    total = tmp[NSG];
    uint r = tmp[sg] + incl - v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return r;
}

// ---------------------------------------------------------------------------------------------
// Census: counts, for every position p, the word of length min(valid length, m+1) starting at p
// (bin offset for length l is (4^l - 4) / 3), and per m-prefix the short positions, whose valid
// length is in [m, m+S). It also sets the presence bit of the word of length
// min(valid length, CENSUS_LEN) at p (same offsets, as bits). h holds P.nbins counters followed by
// CENSUS_PRES_WORDS presence words.
kernel void census(device const uint* T [[buffer(0)]],
                   device const uint* S [[buffer(1)]],
                   device atomic_uint* counts [[buffer(2)]],
                   device atomic_uint* shortCounts [[buffer(3)]],
                   constant CensusParams& P [[buffer(4)]],
                   device atomic_uint* presence [[buffer(5)]],
                   threadgroup atomic_uint* h [[threadgroup(0)]],
                   uint lid [[thread_position_in_threadgroup]],
                   uint tg [[threadgroup_position_in_grid]],
                   uint ntg [[threadgroups_per_grid]])
{
    threadgroup atomic_uint* pres = h + P.nbins;
    const uint L1 = P.m + 1, total = P.nbins + CENSUS_PRES_WORDS;
    for (uint i = lid; i < total; i += TG) atomic_store_explicit(&h[i], 0, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint cap = max(max(L1, P.m + P.S), uint(CENSUS_LEN));
    for (ulong p = ulong(tg) * TG + lid; p < P.n; p += ulong(ntg) * TG) {
        uint L = valid_len(S, p, P.n, cap);
        ulong win = window32(T, p);
        uint Lw = min(L, L1);
        atomic_fetch_add_explicit(&h[((1u << (2 * Lw)) - 4) / 3 + uint(win >> (64 - 2 * Lw))], 1,
                                  memory_order_relaxed);
        uint Lp = min(L, uint(CENSUS_LEN));
        uint b = ((1u << (2 * Lp)) - 4) / 3 + uint(win >> (64 - 2 * Lp));
        atomic_fetch_or_explicit(&pres[b >> 5], 1u << (b & 31), memory_order_relaxed);
        if (L >= P.m && L < P.m + P.S)
            atomic_fetch_add_explicit(&shortCounts[uint(win >> (64 - 2 * P.m))], 1, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = lid; i < P.nbins; i += TG) {
        uint v = atomic_load_explicit(&h[i], memory_order_relaxed);
        if (v) atomic_fetch_add_explicit(&counts[i], v, memory_order_relaxed);
    }
    for (uint i = lid; i < CENSUS_PRES_WORDS; i += TG) {
        uint v = atomic_load_explicit(&pres[i], memory_order_relaxed);
        if (v) atomic_fetch_or_explicit(&presence[i], v, memory_order_relaxed);
    }
}

// ---------------------------------------------------------------------------------------------
// Extract: builds the key of every position whose m-prefix lies in [lo, hi). Region r occupies
// [regionStart[r], regionStart[r] + size): short keys (fewer than S valid bases after the prefix)
// first, then the others, each part unordered. A batch holds a small share of the positions, so
// each thread tests 16 consecutive ones against the prefix range from a single 64-bit window, in
// one pass that counts the matches per region and a second that writes them. Both read the
// segment bitmap and the bases once per 16 positions; segment starts are rare, so a single test
// usually settles every valid length in the word.
// m-prefix of position first + i, from the two words holding bases first .. first + 31.
inline uint prefix_at(uint hi, uint lo, uint i, uint m) {
    uint x = i ? (hi << (2 * i)) | (lo >> (32 - 2 * i)) : hi;
    return x >> (32 - 2 * m);
}

// Segment starts at first .. first + 63 (bit k: one starts at first + k), first a multiple of 16.
inline ulong seg_window(device const uint* S, ulong first) {
    uint w = uint(first >> 5), o = uint(first & 31);
    ulong v = ulong(S[w]) | (ulong(S[w + 1]) << 32);
    return o ? (v >> o) | (ulong(S[w + 2]) << (64 - o)) : v;
}

kernel void extract(device const uint* T [[buffer(0)]],
                    device const uint* S [[buffer(1)]],
                    device const uint* regionStart [[buffer(2)]],
                    device const uint* shortCount [[buffer(3)]],
                    device atomic_uint* cursor [[buffer(4)]],  // [0, MAX_REGIONS) full, then short
                    device ulong* keys [[buffer(5)]],
                    device uint* vals [[buffer(6)]],
                    device uchar* vhi [[buffer(8)]],
                    constant ExtractParams& P [[buffer(7)]],
                    uint lid [[thread_position_in_threadgroup]],
                    uint tg [[threadgroup_position_in_grid]])
{
    threadgroup atomic_uint cnt[MAX_REGIONS];
    threadgroup uint base[MAX_REGIONS];
    const uint nr = P.hi - P.lo;
    for (uint r = lid; r < nr; r += TG) atomic_store_explicit(&cnt[r], 0, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // With seg = seg_window(first), s = seg >> (i + 1) has bit k set when a segment starts at
    // p + 1 + k (p = first + i). p is skipped when one starts in (p, p + m), short when one starts
    // in [p + m, p + m + S), full otherwise. near: some position of the word may not be full.
    const ulong skipMask = (1ul << (P.m - 1)) - 1, fullMask = (1ul << (P.m + P.S - 1)) - 1;
    const ulong near = ((1ul << (15 + P.m + P.S)) - 1) & ~1ul;
    // Thread lid handles text words tg * TG * EX_WPT + j * TG + lid (16 positions each).
    uint hits[EX_WPT];  // bit i: position 16 * word + i is a full entry, bit 16 + i: a short one
#pragma unroll
    for (uint j = 0; j < EX_WPT; ++j) {
        hits[j] = 0;
        const ulong first = (ulong(tg) * (TG * EX_WPT) + j * TG + lid) * 16;
        if (first >= P.n) continue;
        const uint count = uint(min(ulong(16), P.n - first));
        const uint whi = T[uint(first >> 4)], wlo = T[uint(first >> 4) + 1];
        uint in = 0;
        if (P.m == 4) {
            // Shifted by 2i bits, the word's bytes are the prefixes of positions i, i + 4, i + 8, i + 12.
#pragma unroll
            for (uint i = 0; i < 4; ++i) {
                const uint x = i ? (whi << (2 * i)) | (wlo >> (32 - 2 * i)) : whi;
                const bool4 ok = ushort4(as_type<uchar4>(x)) - ushort(P.lo) < ushort(nr);
                in |= (uint(ok.w) << i) | (uint(ok.z) << (i + 4)) | (uint(ok.y) << (i + 8)) | (uint(ok.x) << (i + 12));
            }
            in &= (1u << count) - 1;
        } else {
#pragma unroll
            for (uint i = 0; i < 16; ++i)
                if (i < count) in |= uint(prefix_at(whi, wlo, i, P.m) - P.lo < nr) << i;
        }
        if (!in) continue;
        uint full = in, shrt = 0;
        const ulong seg = seg_window(S, first);
        if (seg & near) {
            full = 0;
            for (uint bits = in; bits; bits &= bits - 1) {
                const uint i = ctz(bits);
                const ulong s = seg >> (i + 1);
                if (s & skipMask) continue;
                if (s & fullMask) shrt |= 1u << i;
                else full |= 1u << i;
            }
        }
        hits[j] = full | (shrt << 16);
        for (uint bits = full; bits; bits &= bits - 1)
            atomic_fetch_add_explicit(&cnt[prefix_at(whi, wlo, ctz(bits), P.m) - P.lo], 1, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint r = lid; r < nr; r += TG) {
        uint c = atomic_load_explicit(&cnt[r], memory_order_relaxed);
        base[r] = c ? regionStart[r] + shortCount[r] + atomic_fetch_add_explicit(&cursor[r], c, memory_order_relaxed) : 0;
        atomic_store_explicit(&cnt[r], 0, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Every valid length in the word is m + 28 unless a segment starts in (first, first + 15 + m + 28).
    const ulong nearLp = ((1ul << (15 + P.m + KEY_BASES)) - 1) & ~1ul;
    for (uint j = 0; j < EX_WPT; ++j) {
        if (!hits[j]) continue;
        const ulong first = (ulong(tg) * (TG * EX_WPT) + j * TG + lid) * 16;
        const uint w0 = uint(first >> 4);
        // Bases first - 16 .. first + 63.
        const uint wp = w0 ? T[w0 - 1] : 0, whi = T[w0], wlo = T[w0 + 1];
        const ulong t01 = (ulong(whi) << 32) | wlo, t23 = (ulong(T[w0 + 2]) << 32) | T[w0 + 3];
        const ulong seg = seg_window(S, first);
        const bool plain = (seg & nearLp) == 0;
        for (uint bits = hits[j]; bits; bits &= bits - 1) {
            const uint b = ctz(bits), i = b & 15;
            const ulong p = first + i;
            const uint r = prefix_at(whi, wlo, i, P.m) - P.lo;
            const uint lp = plain ? KEY_BASES : min(ctz64(seg >> (i + 1)) + 1, P.m + KEY_BASES) - P.m;
            const uint o = 2 * (i + P.m);  // bit offset of base p + m in t01:t23, in [2, 40]
            const ulong chars = lp ? (((t01 << o) | (t23 >> (64 - o))) & (~0ul << (64 - 2 * lp))) : 0ul;
            const uint left = (seg >> i) & 1 ? 4u : i ? (whi >> (32 - 2 * i)) & 3 : wp & 3;
            const ulong k = chars | (ulong(lp) << 3) | ulong(left);
            const uint dst = b < 16 ? base[r] + atomic_fetch_add_explicit(&cnt[r], 1, memory_order_relaxed)
                                    : regionStart[r] + atomic_fetch_add_explicit(&cursor[MAX_REGIONS + r], 1, memory_order_relaxed);
            keys[dst] = k;
            if (P.withVals) vals[dst] = uint(p);
            if (P.withHi) vhi[dst] = uchar(p >> 32);
        }
    }
}

// ---------------------------------------------------------------------------------------------
// LSD radix sort, RB bits per pass, segmented by region. Each tile lies inside one region; the
// histogram is laid out [region][digit][tile] so one exclusive scan over it yields the global
// destination of every (region, digit, tile) triple.
kernel void radix_hist(device const ulong* keys [[buffer(0)]],
                       device uint* hist [[buffer(1)]],
                       device const Tile* tiles [[buffer(2)]],
                       constant SortParams& P [[buffer(3)]],
                       uint lid [[thread_position_in_threadgroup]],
                       uint tg [[threadgroup_position_in_grid]],
                       uint lane [[thread_index_in_simdgroup]])
{
    threadgroup atomic_uint h[RADIX];
    for (uint i = lid; i < RADIX; i += TG) atomic_store_explicit(&h[i], 0, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    Tile t = tiles[tg];
    for (uint i = 0; i < KPT; ++i) {
        uint idx = t.start + i * TG + lid;
        bool valid = idx < t.end;
        uint d = valid ? uint(keys[idx] >> P.shift) & P.mask : 0;
        uint peers = peer_mask(d, valid);
        if (valid && (peers & ((1u << lane) - 1)) == 0)
            atomic_fetch_add_explicit(&h[d], popcount(peers), memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = lid; i < RADIX; i += TG)
        hist[t.histBase + i * t.stride] = atomic_load_explicit(&h[i], memory_order_relaxed);
}

// Ranks a tile stably, reorders it in threadgroup memory, then writes each digit's run to its
// global destination with consecutive threads on consecutive addresses.
kernel void radix_scatter(device const ulong* kin [[buffer(0)]],
                          device const uint* vin [[buffer(1)]],
                          device ulong* kout [[buffer(2)]],
                          device uint* vout [[buffer(3)]],
                          device const uint* hist [[buffer(4)]],
                          device const Tile* tiles [[buffer(5)]],
                          constant SortParams& P [[buffer(6)]],
                          device const uchar* hin [[buffer(7)]],
                          device uchar* hout [[buffer(8)]],
                          uint lid [[thread_position_in_threadgroup]],
                          uint tg [[threadgroup_position_in_grid]],
                          uint lane [[thread_index_in_simdgroup]],
                          uint sg [[simdgroup_index_in_threadgroup]])
{
    threadgroup ulong sk[TILE];
    threadgroup ushort sgh[NSG * RADIX];  // per-simdgroup digit counts, then their prefixes
    threadgroup uint dstart[RADIX];       // first local position of each digit
    threadgroup uint gbase[RADIX];        // global destination of each digit's first key
    threadgroup uint tmp[NSG + 1];

    Tile t = tiles[tg];
    for (uint i = lid; i < NSG * RADIX; i += TG) sgh[i] = 0;
    for (uint i = lid; i < RADIX; i += TG) gbase[i] = hist[t.histBase + i * t.stride];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Simdgroup sg ranks keys [sg * SG_KEYS, (sg + 1) * SG_KEYS) of the tile, 32 per round.
    const uint first = t.start + sg * SG_KEYS;
    ulong k[KPT];
    ushort rk[KPT];
    for (uint r = 0; r < KPT; ++r) {
        uint idx = first + r * 32 + lane;
        bool valid = idx < t.end;
        k[r] = valid ? kin[idx] : 0;
        uint d = uint(k[r] >> P.shift) & P.mask;
        uint peers = peer_mask(d, valid);
        uint below = popcount(peers & ((1u << lane) - 1));
        uint old = valid ? uint(sgh[sg * RADIX + d]) : 0;
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (valid && below == 0) sgh[sg * RADIX + d] = ushort(old + popcount(peers));
        simdgroup_barrier(mem_flags::mem_threadgroup);
        rk[r] = ushort(old + below);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Per digit: prefix over simdgroups (in order, which keeps the sort stable) and tile total.
    const uint DPT = RADIX / TG;
    uint tot[DPT], sum = 0;
    for (uint j = 0; j < DPT; ++j) {
        uint d = lid * DPT + j, run = 0;
        for (uint s = 0; s < NSG; ++s) {
            uint c = sgh[s * RADIX + d];
            sgh[s * RADIX + d] = ushort(run);
            run += c;
        }
        tot[j] = run;
        sum += run;
    }
    uint total;
    uint pre = tg_exclusive_scan(sum, lane, sg, tmp, total);
    for (uint j = 0; j < DPT; ++j) {
        dstart[lid * DPT + j] = pre;
        pre += tot[j];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    ushort lpos[KPT];
    for (uint r = 0; r < KPT; ++r) {
        if (first + r * 32 + lane >= t.end) continue;
        uint d = uint(k[r] >> P.shift) & P.mask;
        lpos[r] = ushort(dstart[d] + sgh[sg * RADIX + d] + rk[r]);
        sk[lpos[r]] = k[r];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint n = t.end - t.start;
    uint dst[KPT];
    for (uint r = 0; r < KPT; ++r) {
        uint i = r * TG + lid;
        if (i >= n) continue;
        ulong key = sk[i];
        uint d = uint(key >> P.shift) & P.mask;
        dst[r] = gbase[d] + (i - dstart[d]) + t.dstOff;
        kout[dst[r]] = key;
    }
    if (!P.withVals) return;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup uint* sv = (threadgroup uint*)sk;
    for (uint r = 0; r < KPT; ++r)
        if (first + r * 32 + lane < t.end) sv[lpos[r]] = vin[first + r * 32 + lane];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint r = 0; r < KPT; ++r) {
        uint i = r * TG + lid;
        if (i < n) vout[dst[r]] = sv[i];
    }
    if (!P.withHi) return;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup uchar* sh = (threadgroup uchar*)sk;
    for (uint r = 0; r < KPT; ++r)
        if (first + r * 32 + lane < t.end) sh[lpos[r]] = hin[first + r * 32 + lane];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint r = 0; r < KPT; ++r) {
        uint i = r * TG + lid;
        if (i < n) hout[dst[r]] = sh[i];
    }
}

// Copies the tiles' ranges from one buffer pair to the other (puts sorted short blocks back).
kernel void copy_tiles(device const ulong* kin [[buffer(0)]],
                       device const uint* vin [[buffer(1)]],
                       device ulong* kout [[buffer(2)]],
                       device uint* vout [[buffer(3)]],
                       device const Tile* tiles [[buffer(4)]],
                       constant SortParams& P [[buffer(5)]],
                       device const uchar* hin [[buffer(6)]],
                       device uchar* hout [[buffer(7)]],
                       uint lid [[thread_position_in_threadgroup]],
                       uint tg [[threadgroup_position_in_grid]])
{
    Tile t = tiles[tg];
    for (uint i = t.start + lid; i < t.end; i += TG) {
        kout[i] = kin[i];
        if (P.withVals) vout[i] = vin[i];
        if (P.withHi) hout[i] = hin[i];
    }
}

// ---------------------------------------------------------------------------------------------
// Exclusive prefix sum (in place), three kernels: per-block sums, single-threadgroup scan for
// small arrays, and block-wise apply.
kernel void scan_reduce(device const uint* data [[buffer(0)]],
                        device uint* sums [[buffer(1)]],
                        constant uint& n [[buffer(2)]],
                        uint lid [[thread_position_in_threadgroup]],
                        uint tg [[threadgroup_position_in_grid]],
                        uint lane [[thread_index_in_simdgroup]],
                        uint sg [[simdgroup_index_in_threadgroup]])
{
    threadgroup uint tmp[NSG];
    uint base = tg * SCAN_BLOCK, s = 0;
    for (uint i = 0; i < SCAN_IPT; ++i) {
        uint idx = base + i * TG + lid;
        if (idx < n) s += data[idx];
    }
    s = simd_sum(s);
    if (lane == 0) tmp[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0) {
        uint t = 0;
        for (uint i = 0; i < NSG; ++i) t += tmp[i];
        sums[tg] = t;
    }
}

inline uint scan_block(device uint* data, uint base, uint n, uint carry,
                       uint lid, uint lane, uint sg, threadgroup uint* tmp)
{
    uint v[SCAN_IPT], s = 0;
    for (uint i = 0; i < SCAN_IPT; ++i) {
        uint idx = base + lid * SCAN_IPT + i;
        v[i] = idx < n ? data[idx] : 0;
        s += v[i];
    }
    uint total;
    uint pre = carry + tg_exclusive_scan(s, lane, sg, tmp, total);
    for (uint i = 0; i < SCAN_IPT; ++i) {
        uint idx = base + lid * SCAN_IPT + i;
        if (idx < n) data[idx] = pre;
        pre += v[i];
    }
    return total;
}

kernel void scan_apply(device uint* data [[buffer(0)]],
                       device const uint* offsets [[buffer(1)]],
                       constant uint& n [[buffer(2)]],
                       uint lid [[thread_position_in_threadgroup]],
                       uint tg [[threadgroup_position_in_grid]],
                       uint lane [[thread_index_in_simdgroup]],
                       uint sg [[simdgroup_index_in_threadgroup]])
{
    threadgroup uint tmp[NSG + 1];
    scan_block(data, tg * SCAN_BLOCK, n, offsets[tg], lid, lane, sg, tmp);
}

kernel void scan_single(device uint* data [[buffer(0)]],
                        constant uint& n [[buffer(1)]],
                        uint lid [[thread_position_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]],
                        uint sg [[simdgroup_index_in_threadgroup]])
{
    threadgroup uint tmp[NSG + 1];
    uint carry = 0;
    for (uint base = 0; base < n; base += SCAN_BLOCK)
        carry += scan_block(data, base, n, carry, lid, lane, sg, tmp);
}

// ---------------------------------------------------------------------------------------------
// Tier 1 on the GPU (counts only). For each depth d in [dLo, dLo + nd), the sorted suffixes that
// share their first d bases form runs (cut where the LCP of neighbours is below d, and at region
// starts). A run's pair matrix P is the OR, over its suffixes, of bit 5 * (base at depth d) +
// (base before), and its MAWs are missMask(P): this is the CPU stack scan's result, since runs
// that do not branch at depth d have none. The runs of all depths are found with one segmented
// OR scan: t1_summary reduces each tile of T1_TILE keys, t1_carry scans the tiles, and t1_main
// rescans each tile from its carry, closing runs. It also writes bit i of gbits when suffixes
// i and i+1 share kRep bases (the deep stage's groups).
#define T1_K 32                 // keys per thread
#define T1_TILE (TG * T1_K)
#define T1_MAXD 26
#define T1_STRIDE (1 + T1_MAXD) // f mask, then one OR per depth

struct T1Params { uint n; uint m; uint dLo; uint nd; uint kRep; uint nr; uint nTiles; uint pad; };

// A segment of a scan: f has bit j set when a run starts in it at depth dLo + j; v[j] is the OR
// since the last start (or over all of it).
struct Seg { uint f; uint v[T1_MAXD]; };

inline void seg_zero(thread Seg& a) {
    a.f = 0;
    for (uint j = 0; j < T1_MAXD; ++j) a.v[j] = 0;
}
// a = a then b.
inline void seg_then(thread Seg& a, thread const Seg& b) {
    for (uint j = 0; j < T1_MAXD; ++j) a.v[j] = ((b.f >> j) & 1) ? b.v[j] : (a.v[j] | b.v[j]);
    a.f |= b.f;
}
inline void seg_load(thread Seg& a, device const uint* p) {
    a.f = p[0];
    for (uint j = 0; j < T1_MAXD; ++j) a.v[j] = p[1 + j];
}
inline void seg_store(thread const Seg& a, device uint* p) {
    p[0] = a.f;
    for (uint j = 0; j < T1_MAXD; ++j) p[1 + j] = a.v[j];
}

inline uint t1_bit(ulong k, uint dd) {  // 1 << 5 * (base at depth m + dd, 4 past the end) + base before
    uint lp = uint(k >> 3) & 31u;
    uint b = lp > dd ? uint(k >> (62 - 2 * dd)) & 3u : 4u;
    return 1u << (5u * b + (uint(k) & 7u));
}
inline uint t1_miss(uint P) {
    uint L = ((P | (P >> 5) | (P >> 10) | (P >> 15) | (P >> 20)) & 15u), rows = P & 0xFFFFFu;
    uint nz = (rows | rows >> 1 | rows >> 2 | rows >> 3 | rows >> 4) & 0x08421u;
    return (L * 0x08421u) & ~rows & (nz * 15u);
}
inline int t1_lcp(ulong a, ulong b, uint m) {
    uint c = min(uint(clz((a ^ b) | 1ul)) >> 1, 28u);
    c = min(c, uint(min((a >> 3) & 31ul, (b >> 3) & 31ul)));
    return int(m + c);
}

// Steps through the region starts rs[0, nr) (offsets in the batch, nondecreasing).
struct RegionWalk { uint r, next; };
inline RegionWalk rw_init(device const uint* rs, uint nr, uint i, uint n) {
    uint lo = 0, hi = nr;
    while (lo < hi) {
        uint mid = (lo + hi) / 2;
        if (rs[mid] < i) lo = mid + 1;
        else hi = mid;
    }
    return {lo, lo < nr ? rs[lo] : n};
}
// LCP of suffixes i-1 and i (i visited in increasing order), -1 when i starts a region.
inline int rw_lcp(thread RegionWalk& w, device const uint* rs, uint nr, uint n, device const ulong* K, uint i, uint m) {
    bool start = i == 0 || i == w.next;
    while (w.next <= i && w.r < nr) {
        w.r++;
        w.next = w.r < nr ? rs[w.r] : n;
    }
    return start ? -1 : t1_lcp(K[i - 1], K[i], m);
}

inline void t1_summarize(device const ulong* K, device const uint* rs, constant T1Params& P, uint s, thread Seg& r) {
    seg_zero(r);
    if (s >= P.n) return;
    RegionWalk w = rw_init(rs, P.nr, s, P.n);
    const uint e = min(s + T1_K, P.n);
    for (uint i = s; i < e; ++i) {
        int L = rw_lcp(w, rs, P.nr, P.n, K, i, P.m);
        ulong k = K[i];
        for (uint j = 0; j < T1_MAXD; ++j) {
            if (j >= P.nd) break;
            uint d = P.dLo + j;
            uint x = t1_bit(k, d - P.m);
            bool b = L < int(d);
            r.v[j] = b ? x : (r.v[j] | x);
            r.f |= uint(b) << j;
        }
    }
}

// Exclusive scan of the threads' segments over the threadgroup; also returns the total.
inline void t1_tg_scan(thread const Seg& me, uint lane, uint sg, threadgroup uint* tmp, thread Seg& pre, thread Seg& total) {
    Seg inc = me;
    for (uint off = 1; off < 32; off <<= 1) {
        Seg up;
        up.f = simd_shuffle_up(inc.f, off);
        for (uint j = 0; j < T1_MAXD; ++j) up.v[j] = simd_shuffle_up(inc.v[j], off);
        if (lane >= off) {
            seg_then(up, inc);
            inc = up;
        }
    }
    Seg ex;
    ex.f = simd_shuffle_up(inc.f, 1);
    for (uint j = 0; j < T1_MAXD; ++j) ex.v[j] = simd_shuffle_up(inc.v[j], 1);
    if (lane == 0) seg_zero(ex);
    if (lane == 31) {
        tmp[sg * T1_STRIDE] = inc.f;
        for (uint j = 0; j < T1_MAXD; ++j) tmp[sg * T1_STRIDE + 1 + j] = inc.v[j];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    seg_zero(pre);
    seg_zero(total);
    for (uint g = 0; g < NSG; ++g) {
        Seg t;
        t.f = tmp[g * T1_STRIDE];
        for (uint j = 0; j < T1_MAXD; ++j) t.v[j] = tmp[g * T1_STRIDE + 1 + j];
        if (g < sg) seg_then(pre, t);
        seg_then(total, t);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    seg_then(pre, ex);
}

kernel void t1_summary(device const ulong* K [[buffer(0)]],
                       device const uint* rs [[buffer(1)]],
                       constant T1Params& P [[buffer(2)]],
                       device uint* sums [[buffer(3)]],
                       uint lid [[thread_position_in_threadgroup]],
                       uint tg [[threadgroup_position_in_grid]],
                       uint lane [[thread_index_in_simdgroup]],
                       uint sg [[simdgroup_index_in_threadgroup]])
{
    threadgroup uint tmp[NSG * T1_STRIDE];
    Seg me, pre, total;
    t1_summarize(K, rs, P, (tg * TG + lid) * T1_K, me);
    t1_tg_scan(me, lane, sg, tmp, pre, total);
    if (lid == 0) seg_store(total, sums + tg * T1_STRIDE);
}

// One threadgroup: carries[t] = the scan of sums[0, t).
kernel void t1_carry(device const uint* sums [[buffer(0)]],
                     device uint* carries [[buffer(1)]],
                     constant T1Params& P [[buffer(2)]],
                     uint lid [[thread_position_in_threadgroup]],
                     uint lane [[thread_index_in_simdgroup]],
                     uint sg [[simdgroup_index_in_threadgroup]])
{
    threadgroup uint tmp[NSG * T1_STRIDE];
    const uint per = (P.nTiles + TG - 1) / TG, t0 = min(lid * per, P.nTiles), t1 = min(t0 + per, P.nTiles);
    Seg loc, x, pre, total;
    seg_zero(loc);
    for (uint t = t0; t < t1; ++t) {
        seg_load(x, sums + t * T1_STRIDE);
        seg_then(loc, x);
    }
    t1_tg_scan(loc, lane, sg, tmp, pre, total);
    for (uint t = t0; t < t1; ++t) {
        seg_store(pre, carries + t * T1_STRIDE);
        seg_load(x, sums + t * T1_STRIDE);
        seg_then(pre, x);
    }
}

kernel void t1_main(device const ulong* K [[buffer(0)]],
                    device const uint* rs [[buffer(1)]],
                    constant T1Params& P [[buffer(2)]],
                    device const uint* carries [[buffer(3)]],
                    device atomic_uint* counts [[buffer(4)]],  // MAWs by depth - dLo
                    device uint* gbits [[buffer(5)]],
                    uint lid [[thread_position_in_threadgroup]],
                    uint tg [[threadgroup_position_in_grid]],
                    uint lane [[thread_index_in_simdgroup]],
                    uint sg [[simdgroup_index_in_threadgroup]])
{
    threadgroup uint tmp[NSG * T1_STRIDE];
    threadgroup atomic_uint tcnt[T1_MAXD];
    for (uint j = lid; j < T1_MAXD; j += TG) atomic_store_explicit(&tcnt[j], 0, memory_order_relaxed);
    const uint s = (tg * TG + lid) * T1_K;
    Seg me, pre, total, acc;
    t1_summarize(K, rs, P, s, me);
    t1_tg_scan(me, lane, sg, tmp, pre, total);
    seg_load(acc, carries + tg * T1_STRIDE);
    seg_then(acc, pre);  // the runs open where this thread's keys begin
    uint cnt[T1_MAXD];
    for (uint j = 0; j < T1_MAXD; ++j) cnt[j] = 0;
    if (s < P.n) {
        RegionWalk w = rw_init(rs, P.nr, s, P.n);
        const uint e = min(s + T1_K, P.n);
        uint gw = 0;
        for (uint i = s; i < e; ++i) {
            int L = rw_lcp(w, rs, P.nr, P.n, K, i, P.m);
            if (i > s) gw |= uint(L >= int(P.kRep)) << (i - s - 1);
            ulong k = K[i];
            for (uint j = 0; j < T1_MAXD; ++j) {
                if (j >= P.nd) break;
                uint d = P.dLo + j;
                uint x = t1_bit(k, d - P.m);
                bool b = L < int(d);  // a run at depth d ends before i
                cnt[j] += b ? popcount(t1_miss(acc.v[j])) : 0;
                acc.v[j] = b ? x : (acc.v[j] | x);
            }
        }
        if (e < P.n) {
            int L = rw_lcp(w, rs, P.nr, P.n, K, e, P.m);
            gw |= uint(L >= int(P.kRep)) << (T1_K - 1);
        } else {  // the batch's last key closes every run
            for (uint j = 0; j < T1_MAXD; ++j)
                if (j < P.nd) cnt[j] += popcount(t1_miss(acc.v[j]));
        }
        gbits[s / T1_K] = gw;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint j = 0; j < T1_MAXD; ++j) {
        if (j >= P.nd) break;
        uint c = simd_sum(cnt[j]);
        if (lane == 0 && c) atomic_fetch_add_explicit(&tcnt[j], c, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < P.nd) {
        uint c = atomic_load_explicit(&tcnt[lid], memory_order_relaxed);
        if (c) atomic_fetch_add_explicit(&counts[lid], c, memory_order_relaxed);
    }
}
