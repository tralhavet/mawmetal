// Copyright (c) 2026 Nicolas Innocenti
// SPDX-License-Identifier: MIT
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "gpu.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

#include "kernels_src.h"  // generated from kernels.metal: kKernelSource

namespace {
// Must match kernels.metal.
constexpr uint32_t kTG = 256;
constexpr uint32_t kRB = 9;
constexpr uint32_t kRadix = 1u << kRB;
constexpr uint32_t kSortTile = kTG * 8;
constexpr uint32_t kScanBlock = kTG * 16;
constexpr uint32_t kExtractTile = kTG * 16 * 2;  // positions per extract threadgroup
constexpr uint32_t kScanSingleMax = 1u << 16;
constexpr uint32_t kMaxRegions = 1024;
constexpr uint32_t kT1Tile = kTG * 32;  // keys per tier-1 threadgroup
constexpr uint32_t kT1Stride = 27;      // uints per tier-1 scan segment
struct T1Params { uint32_t n, m, dLo, nd, kRep, nr, nTiles, pad; };

struct TileDesc { uint32_t start, end, histBase, stride, regionStart, dstOff; };
struct CensusParams { uint64_t n; uint32_t m, S, nbins; };
struct ExtractParams { uint64_t n; uint32_t m, S, lo, hi, withVals, withHi; };
struct SortParams { uint32_t shift, mask, withVals, withHi; };

uint64_t ceilDiv(uint64_t a, uint64_t b) { return (a + b - 1) / b; }

// Appends tiles covering [start, start + size) for each region, with histogram slots laid out
// [region][digit][tile]. The scan of that histogram places the ranges back to back from 0, so
// each tile records how far its range really starts beyond that. Returns the histogram size.
uint64_t appendTiles(std::vector<TileDesc>& td, const std::vector<uint64_t>& starts, const std::vector<uint64_t>& sizes) {
    uint64_t histOff = 0, packed = 0;
    for (size_t r = 0; r < starts.size(); ++r) {
        uint64_t s = starts[r], c = sizes[r], nt = ceilDiv(c, kSortTile);
        for (uint64_t i = 0; i < nt; ++i)
            td.push_back({(uint32_t)(s + i * kSortTile), (uint32_t)std::min(s + (i + 1) * kSortTile, s + c),
                          (uint32_t)(histOff + i), (uint32_t)nt, (uint32_t)s, (uint32_t)(s - packed)});
        histOff += kRadix * nt;
        packed += c;
    }
    return histOff;
}
}  // namespace

struct Gpu::Impl {
    id<MTLDevice> dev;
    id<MTLCommandQueue> queue;
    id<MTLComputePipelineState> census, extract, hist, scatter, copyTiles, scanReduce, scanApply, scanSingle;
    id<MTLComputePipelineState> t1Summary, t1Carry, t1Main;

    id<MTLBuffer> T, S;
    uint64_t n = 0;

    // Buffers the CPU writes are per slot, so a batch can be encoded while the other one runs; the
    // GPU-only temporaries are shared, and Metal orders the two batches' uses of them.
    struct Slot {
        id<MTLBuffer> keys, vals, hi;  // hi: bits 32..39 of the positions, for long texts
        id<MTLBuffer> tiles, regionStart, shortCount, cursor;
        id<MTLBuffer> t1Sums, t1Carries, t1Counts, gbits;  // tier 1 on the GPU
        id<MTLCommandBuffer> cbExtract, cbSort, cbTier1;
    } slot[Gpu::kSlots];
    id<MTLBuffer> keysTmp, valsTmp, hiTmp, histBuf, scanTmp, dummy;
    id<MTLBuffer> censusCounts, censusShort, censusPres;  // kept across texts
    uint64_t capEntries = 0, capTiles = 0;
    bool capVals = false, capWide = false, capT1 = false;

    id<MTLBuffer> buffer(uint64_t bytes) {
        id<MTLBuffer> b = [dev newBufferWithLength:std::max<uint64_t>(bytes, 16) options:MTLResourceStorageModeShared];
        if (!b) throw std::runtime_error("Metal buffer allocation failed (" + std::to_string(bytes) + " bytes)");
        return b;
    }

    id<MTLComputePipelineState> pipeline(id<MTLLibrary> lib, const char* name) {
        NSError* e = nil;
        id<MTLFunction> f = [lib newFunctionWithName:[NSString stringWithUTF8String:name]];
        if (!f) throw std::runtime_error(std::string("missing kernel ") + name);
        id<MTLComputePipelineState> ps = [dev newComputePipelineStateWithFunction:f error:&e];
        if (!ps) throw std::runtime_error(std::string("pipeline ") + name + ": " + e.localizedDescription.UTF8String);
        if (ps.maxTotalThreadsPerThreadgroup < kTG)
            throw std::runtime_error(std::string("kernel ") + name + " cannot run 256 threads per threadgroup");
        return ps;
    }

    static void dispatch(id<MTLComputeCommandEncoder> enc, uint64_t groups) {
        [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(kTG, 1, 1)];
    }

    // In-place exclusive scan of count uints at data+dataOff. Temporaries live in scanTmp from tmpOff.
    void encodeScan(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> data, uint64_t dataOff, uint32_t count,
                    uint64_t tmpOff) {
        if (count <= kScanSingleMax) {
            [enc setComputePipelineState:scanSingle];
            [enc setBuffer:data offset:dataOff atIndex:0];
            [enc setBytes:&count length:4 atIndex:1];
            dispatch(enc, 1);
            return;
        }
        uint32_t nb = (uint32_t)ceilDiv(count, kScanBlock);
        [enc setComputePipelineState:scanReduce];
        [enc setBuffer:data offset:dataOff atIndex:0];
        [enc setBuffer:scanTmp offset:tmpOff atIndex:1];
        [enc setBytes:&count length:4 atIndex:2];
        dispatch(enc, nb);
        encodeScan(enc, scanTmp, tmpOff, nb, tmpOff + ceilDiv(nb * 4ull, 256) * 256);
        [enc setComputePipelineState:scanApply];
        [enc setBuffer:data offset:dataOff atIndex:0];
        [enc setBuffer:scanTmp offset:tmpOff atIndex:1];
        [enc setBytes:&count length:4 atIndex:2];
        dispatch(enc, nb);
    }

    // One stable pass over the tiles [tileOff, tileOff + numTiles): histogram, scan, scatter.
    void encodePass(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> tiles, id<MTLBuffer> kin, id<MTLBuffer> vin, id<MTLBuffer> hin,
                    id<MTLBuffer> kout, id<MTLBuffer> vout, id<MTLBuffer> hout, uint64_t tileOff, uint64_t numTiles,
                    uint32_t histCount, SortParams sp) {
        [enc setComputePipelineState:hist];
        [enc setBuffer:kin offset:0 atIndex:0];
        [enc setBuffer:histBuf offset:0 atIndex:1];
        [enc setBuffer:tiles offset:tileOff * sizeof(TileDesc) atIndex:2];
        [enc setBytes:&sp length:sizeof(sp) atIndex:3];
        dispatch(enc, numTiles);
        encodeScan(enc, histBuf, 0, histCount, 0);
        [enc setComputePipelineState:scatter];
        [enc setBuffer:kin offset:0 atIndex:0];
        [enc setBuffer:vin offset:0 atIndex:1];
        [enc setBuffer:kout offset:0 atIndex:2];
        [enc setBuffer:vout offset:0 atIndex:3];
        [enc setBuffer:histBuf offset:0 atIndex:4];
        [enc setBuffer:tiles offset:tileOff * sizeof(TileDesc) atIndex:5];
        [enc setBytes:&sp length:sizeof(sp) atIndex:6];
        [enc setBuffer:hin offset:0 atIndex:7];
        [enc setBuffer:hout offset:0 atIndex:8];
        dispatch(enc, numTiles);
    }

    // The buffers used outlive their command buffers, so Metal need not retain them.
    id<MTLCommandBuffer> commit(void (^encode)(id<MTLComputeCommandEncoder>), std::function<void()> onDone = nullptr) {
        id<MTLCommandBuffer> cb = [queue commandBufferWithUnretainedReferences];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        encode(enc);
        [enc endEncoding];
        if (onDone) [cb addCompletedHandler:^(id<MTLCommandBuffer>) { onDone(); }];
        [cb commit];
        return cb;
    }

    id<MTLBuffer> atLeast(id<MTLBuffer> b, uint64_t bytes) { return b && b.length >= bytes ? b : buffer(bytes); }

    static double finish(id<MTLCommandBuffer> cb) {
        if (!cb) return 0;
        [cb waitUntilCompleted];
        if (cb.status != MTLCommandBufferStatusCompleted) {
            NSString* msg = cb.error ? cb.error.localizedDescription : @"unknown error";
            throw std::runtime_error(std::string("Metal command buffer failed: ") + msg.UTF8String);
        }
        return cb.GPUEndTime - cb.GPUStartTime;
    }
};

Gpu::Gpu() : p_(new Impl) {}
Gpu::~Gpu() { delete p_; }

bool Gpu::init(std::string& err) {
    @autoreleasepool {
        // Same GPU (Apple Silicon has one), but MTLCreateSystemDefaultDevice also sets up GPU switching
        // for displays, which takes about 40 ms against 8 ms here.
        p_->dev = MTLCopyAllDevices().firstObject;
        if (!p_->dev) p_->dev = MTLCreateSystemDefaultDevice();
        if (!p_->dev) {
            err = "no Metal device";
            return false;
        }
        p_->queue = [p_->dev newCommandQueue];
        MTLCompileOptions* opts = [MTLCompileOptions new];
        opts.languageVersion = MTLLanguageVersion3_0;
        NSError* e = nil;
        id<MTLLibrary> lib = [p_->dev newLibraryWithSource:[NSString stringWithUTF8String:kKernelSource]
                                                   options:opts
                                                     error:&e];
        if (!lib) {
            err = std::string("shader compilation failed: ") + e.localizedDescription.UTF8String;
            return false;
        }
        try {
            p_->census = p_->pipeline(lib, "census");
            p_->extract = p_->pipeline(lib, "extract");
            p_->hist = p_->pipeline(lib, "radix_hist");
            p_->scatter = p_->pipeline(lib, "radix_scatter");
            p_->copyTiles = p_->pipeline(lib, "copy_tiles");
            p_->scanReduce = p_->pipeline(lib, "scan_reduce");
            p_->scanApply = p_->pipeline(lib, "scan_apply");
            p_->scanSingle = p_->pipeline(lib, "scan_single");
            p_->t1Summary = p_->pipeline(lib, "t1_summary");
            p_->t1Carry = p_->pipeline(lib, "t1_carry");
            p_->t1Main = p_->pipeline(lib, "t1_main");
            p_->dummy = p_->buffer(16);
        } catch (const std::exception& ex) {
            err = ex.what();
            return false;
        }
    }
    return true;
}

std::string Gpu::name() const { return p_->dev.name.UTF8String; }
uint64_t Gpu::workingSetBytes() const { return p_->dev.recommendedMaxWorkingSetSize; }
uint64_t Gpu::maxBufferBytes() const { return p_->dev.maxBufferLength; }

void Gpu::setText(const Text& t) {
    p_->n = t.n;
    auto wrap = [&](const PageVec& v) {
        id<MTLBuffer> b = [p_->dev newBufferWithBytesNoCopy:(void*)v.data()
                                                     length:PageAlloc<uint32_t>::bytesFor(v.capacity())
                                                    options:MTLResourceStorageModeShared
                                                deallocator:nil];
        if (!b) throw std::runtime_error("Metal could not wrap the text (" + std::to_string(v.size() * 4) + " bytes)");
        return b;
    };
    p_->T = wrap(t.packed);
    p_->S = wrap(t.segBits);
}

void Gpu::releaseText() {
    p_->T = p_->S = nil;
    p_->n = 0;
}

void Gpu::census(int m, int S, std::vector<uint32_t>& counts, std::vector<uint32_t>& shortCounts,
                 std::vector<uint32_t>& presence, double& gpuSec) {
    Impl* p = p_;
    const uint32_t nbins = (uint32_t)(((1ull << (2 * (m + 2))) - 4) / 3), nreg = 1u << (2 * m);
    const uint64_t tgBytes = ceilDiv((nbins + kCensusPresWords) * 4ull, 16) * 16;
    if (tgBytes > p->dev.maxThreadgroupMemoryLength) throw std::runtime_error("census does not fit threadgroup memory");
    p->censusCounts = p->atLeast(p->censusCounts, nbins * 4ull);
    p->censusShort = p->atLeast(p->censusShort, nreg * 4ull);
    p->censusPres = p->atLeast(p->censusPres, kCensusPresWords * 4ull);
    id<MTLBuffer> cbuf = p->censusCounts, sbuf = p->censusShort, pbuf = p->censusPres;
    memset(cbuf.contents, 0, nbins * 4ull);
    memset(sbuf.contents, 0, nreg * 4ull);
    memset(pbuf.contents, 0, kCensusPresWords * 4ull);
    CensusParams prm{p->n, (uint32_t)m, (uint32_t)S, nbins};
    uint64_t groups = std::min<uint64_t>(ceilDiv(p->n, kTG), 2048);
    gpuSec = Impl::finish(p->commit(^(id<MTLComputeCommandEncoder> enc) {
      [enc setComputePipelineState:p->census];
      [enc setBuffer:p->T offset:0 atIndex:0];
      [enc setBuffer:p->S offset:0 atIndex:1];
      [enc setBuffer:cbuf offset:0 atIndex:2];
      [enc setBuffer:sbuf offset:0 atIndex:3];
      [enc setBytes:&prm length:sizeof(prm) atIndex:4];
      [enc setBuffer:pbuf offset:0 atIndex:5];
      [enc setThreadgroupMemoryLength:tgBytes atIndex:0];
      Impl::dispatch(enc, groups);
    }));
    counts.assign((uint32_t*)cbuf.contents, (uint32_t*)cbuf.contents + nbins);
    shortCounts.assign((uint32_t*)sbuf.contents, (uint32_t*)sbuf.contents + nreg);
    presence.assign((uint32_t*)pbuf.contents, (uint32_t*)pbuf.contents + kCensusPresWords);
}

uint64_t Gpu::tilesFor(const uint64_t* counts, size_t nRegions) {
    uint64_t t = 0;
    for (size_t r = 0; r < nRegions; ++r) t += ceilDiv(counts[r], kSortTile);
    return t;
}

int Gpu::passesFor(int S) { return (int)ceilDiv(2 * S, kRB); }

void Gpu::reserveBatches(uint64_t entries, uint64_t tiles, bool withVals, bool wide, bool tier1) {
    Impl* p = p_;
    if (entries <= p->capEntries && tiles <= p->capTiles && (!withVals || (p->capVals && p->capWide == wide)) &&
        (!tier1 || p->capT1))
        return;
    releaseBatches();
    if (entries * 8 > p->dev.maxBufferLength)
        throw std::runtime_error("batch too large for one Metal buffer; lower --batch");
    const bool hi = withVals && wide;
    for (auto& s : p->slot) {
        s.keys = p->buffer(entries * 8);
        if (withVals) s.vals = p->buffer(entries * 4);
        if (hi) s.hi = p->buffer(entries);
        // Tiles for the main passes plus (at most as many) for the short blocks.
        s.tiles = p->buffer(2 * tiles * sizeof(TileDesc));
        s.regionStart = p->buffer(kMaxRegions * 4);
        s.shortCount = p->buffer(kMaxRegions * 4);
        s.cursor = p->buffer(2 * kMaxRegions * 4);
        if (tier1) {
            const uint64_t t1Tiles = ceilDiv(entries, kT1Tile);
            s.t1Sums = p->buffer(t1Tiles * kT1Stride * 4);
            s.t1Carries = p->buffer(t1Tiles * kT1Stride * 4);
            s.t1Counts = p->buffer(32 * 4);
            s.gbits = p->buffer(t1Tiles * kTG * 4);
        }
    }
    p->keysTmp = p->buffer(entries * 8);
    if (withVals) p->valsTmp = p->buffer(entries * 4);
    if (hi) p->hiTmp = p->buffer(entries);
    p->histBuf = p->buffer(tiles * kRadix * 4);
    uint64_t tmp = 0, c = tiles * kRadix;
    while (c > kScanSingleMax) {
        c = ceilDiv(c, kScanBlock);
        tmp += ceilDiv(c * 4, 256) * 256;
    }
    p->scanTmp = p->buffer(tmp + 256);
    p->capEntries = entries;
    p->capTiles = tiles;
    p->capVals = withVals;
    p->capWide = wide;
    p->capT1 = tier1;
}

void Gpu::releaseBatches() {
    Impl* p = p_;
    for (auto& s : p->slot) s = Impl::Slot{};
    p->keysTmp = p->valsTmp = p->hiTmp = p->histBuf = p->scanTmp = nil;
    p->capEntries = p->capTiles = 0;
    p->capVals = p->capT1 = false;
}

void Gpu::submitBatch(int si, const BatchSpec& b, std::function<void()> onDone) {
    Impl* p = p_;
    Impl::Slot& slot = p->slot[si];
    const uint32_t nr = b.hi - b.lo;
    if (nr > kMaxRegions) throw std::runtime_error("too many regions in one batch");

    std::vector<uint64_t> starts(nr), sizes(nr), shorts(nr);
    uint64_t total = 0, totalShort = 0;
    for (uint32_t r = 0; r < nr; ++r) {
        starts[r] = total;
        sizes[r] = b.counts[r];
        shorts[r] = b.shortCounts[r];
        total += sizes[r];
        totalShort += shorts[r];
    }
    std::vector<TileDesc> td;
    uint32_t histMain = (uint32_t)appendTiles(td, starts, sizes);
    const uint64_t mainTiles = td.size();
    uint32_t histShort = (uint32_t)appendTiles(td, starts, shorts);
    const uint64_t shortTiles = td.size() - mainTiles;
    if (total > p->capEntries || mainTiles > p->capTiles || (b.withVals && (!p->capVals || p->capWide != b.wide)) ||
        (b.tier1 && !p->capT1))
        throw std::runtime_error("batch exceeds reserved capacity");
    slot.cbExtract = slot.cbSort = slot.cbTier1 = nil;
    if (total == 0) {
        if (onDone) onDone();
        return;
    }
    id<MTLBuffer> tiles = slot.tiles, regionStart = slot.regionStart, shortCount = slot.shortCount, cursor = slot.cursor;
    memcpy(tiles.contents, td.data(), td.size() * sizeof(TileDesc));
    uint32_t* rs = (uint32_t*)regionStart.contents;
    uint32_t* sc = (uint32_t*)shortCount.contents;
    for (uint32_t r = 0; r < nr; ++r) rs[r] = (uint32_t)starts[r], sc[r] = (uint32_t)shorts[r];
    memset(cursor.contents, 0, 2 * kMaxRegions * 4);
    id<MTLBuffer> slotKeys = slot.keys, t1Sums = slot.t1Sums, t1Carries = slot.t1Carries, t1Counts = slot.t1Counts,
                  gbits = slot.gbits;
    T1Params tp{(uint32_t)total, (uint32_t)b.m, (uint32_t)b.dLo, (uint32_t)(b.dHi - b.dLo + 1), (uint32_t)b.kRep, nr,
                (uint32_t)ceilDiv(total, kT1Tile), 0};
    if (b.tier1) {
        if (tp.nd < 1 || tp.nd > 26) throw std::runtime_error("tier-1 depth range not supported on the GPU");
        memset(t1Counts.contents, 0, 32 * 4);
    }

    // The passes alternate X -> Y -> X ...; pick X so the last pass writes the slot's buffers.
    const int passes = passesFor(b.S);
    const bool withHi = b.withVals && b.wide;
    id<MTLBuffer> slotVals = b.withVals ? slot.vals : p->dummy, tmpVals = b.withVals ? p->valsTmp : p->dummy;
    id<MTLBuffer> slotHi = withHi ? slot.hi : p->dummy, tmpHi = withHi ? p->hiTmp : p->dummy;
    id<MTLBuffer> kX = passes % 2 ? p->keysTmp : slot.keys, kY = passes % 2 ? slot.keys : p->keysTmp;
    id<MTLBuffer> vX = passes % 2 ? tmpVals : slotVals, vY = passes % 2 ? slotVals : tmpVals;
    id<MTLBuffer> hX = passes % 2 ? tmpHi : slotHi, hY = passes % 2 ? slotHi : tmpHi;

    ExtractParams ep{p->n, (uint32_t)b.m, (uint32_t)b.S, b.lo, b.hi, b.withVals ? 1u : 0u, withHi ? 1u : 0u};
    uint64_t exGroups = ceilDiv(p->n, kExtractTile);
    slot.cbExtract = p->commit(^(id<MTLComputeCommandEncoder> enc) {
      [enc setComputePipelineState:p->extract];
      [enc setBuffer:p->T offset:0 atIndex:0];
      [enc setBuffer:p->S offset:0 atIndex:1];
      [enc setBuffer:regionStart offset:0 atIndex:2];
      [enc setBuffer:shortCount offset:0 atIndex:3];
      [enc setBuffer:cursor offset:0 atIndex:4];
      [enc setBuffer:kX offset:0 atIndex:5];
      [enc setBuffer:vX offset:0 atIndex:6];
      [enc setBytes:&ep length:sizeof(ep) atIndex:7];
      [enc setBuffer:hX offset:0 atIndex:8];
      Impl::dispatch(enc, exGroups);
    });

    const uint32_t withVals = b.withVals ? 1u : 0u, hiFlag = withHi ? 1u : 0u;
    const uint32_t lowBit = 64 - 2 * b.S;
    slot.cbSort = p->commit(^(id<MTLComputeCommandEncoder> enc) {
      if (totalShort) {
          // Order each region's short block by valid length (bits 7..3), in place via Y.
          p->encodePass(enc, tiles, kX, vX, hX, kY, vY, hY, mainTiles, shortTiles, histShort, SortParams{3, 31, withVals, hiFlag});
          SortParams cp{0, 0, withVals, hiFlag};
          [enc setComputePipelineState:p->copyTiles];
          [enc setBuffer:kY offset:0 atIndex:0];
          [enc setBuffer:vY offset:0 atIndex:1];
          [enc setBuffer:kX offset:0 atIndex:2];
          [enc setBuffer:vX offset:0 atIndex:3];
          [enc setBuffer:tiles offset:mainTiles * sizeof(TileDesc) atIndex:4];
          [enc setBytes:&cp length:sizeof(cp) atIndex:5];
          [enc setBuffer:hY offset:0 atIndex:6];
          [enc setBuffer:hX offset:0 atIndex:7];
          Impl::dispatch(enc, shortTiles);
      }
      for (int pass = 0; pass < passes; ++pass) {
          bool even = (pass & 1) == 0;
          p->encodePass(enc, tiles, even ? kX : kY, even ? vX : vY, even ? hX : hY, even ? kY : kX, even ? vY : vX,
                        even ? hY : hX, 0, mainTiles, histMain, SortParams{lowBit + kRB * pass, kRadix - 1, withVals, hiFlag});
      }
    }, b.tier1 ? std::function<void()>() : std::move(onDone));
    if (!b.tier1) return;
    // Tier 1 on the sorted keys (now in the slot's buffer), in its own command buffer so that its
    // GPU time is reported separately.
    slot.cbTier1 = p->commit(^(id<MTLComputeCommandEncoder> enc) {
      [enc setComputePipelineState:p->t1Summary];
      [enc setBuffer:slotKeys offset:0 atIndex:0];
      [enc setBuffer:regionStart offset:0 atIndex:1];
      [enc setBytes:&tp length:sizeof(tp) atIndex:2];
      [enc setBuffer:t1Sums offset:0 atIndex:3];
      Impl::dispatch(enc, tp.nTiles);
      [enc setComputePipelineState:p->t1Carry];
      [enc setBuffer:t1Sums offset:0 atIndex:0];
      [enc setBuffer:t1Carries offset:0 atIndex:1];
      [enc setBytes:&tp length:sizeof(tp) atIndex:2];
      Impl::dispatch(enc, 1);
      [enc setComputePipelineState:p->t1Main];
      [enc setBuffer:slotKeys offset:0 atIndex:0];
      [enc setBuffer:regionStart offset:0 atIndex:1];
      [enc setBytes:&tp length:sizeof(tp) atIndex:2];
      [enc setBuffer:t1Carries offset:0 atIndex:3];
      [enc setBuffer:t1Counts offset:0 atIndex:4];
      [enc setBuffer:gbits offset:0 atIndex:5];
      Impl::dispatch(enc, tp.nTiles);
    }, std::move(onDone));
}

void Gpu::waitBatch(int si, GpuBatchTimes& t) {
    Impl::Slot& slot = p_->slot[si];
    t.extract += Impl::finish(slot.cbExtract);
    t.sort += Impl::finish(slot.cbSort);
    t.tier1 += Impl::finish(slot.cbTier1);
    slot.cbExtract = slot.cbSort = slot.cbTier1 = nil;
}

const uint64_t* Gpu::keys(int si) const { return (const uint64_t*)p_->slot[si].keys.contents; }
const uint32_t* Gpu::vals(int si) const {
    return p_->slot[si].vals ? (const uint32_t*)p_->slot[si].vals.contents : nullptr;
}
const uint32_t* Gpu::tier1Counts(int si) const { return (const uint32_t*)p_->slot[si].t1Counts.contents; }
const uint32_t* Gpu::groupBits(int si) const { return (const uint32_t*)p_->slot[si].gbits.contents; }
const uint8_t* Gpu::hi(int si) const { return p_->slot[si].hi ? (const uint8_t*)p_->slot[si].hi.contents : nullptr; }
