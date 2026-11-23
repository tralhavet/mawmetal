// Copyright (c) 2026 Nicolas Innocenti
// SPDX-License-Identifier: MIT
#pragma once
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "maw.h"

struct GpuBatchTimes {
    double extract = 0, sort = 0, tier1 = 0;  // GPU execution time, seconds
};

// One batch: every position whose m-prefix is in [lo, hi), grouped into regions (one per prefix)
// stored back to back in prefix order, each sorted by the S bases following the prefix.
struct BatchSpec {
    int m = 0, S = 0;
    uint32_t lo = 0, hi = 0;
    const uint64_t* counts = nullptr;       // region sizes, hi - lo of them
    const uint64_t* shortCounts = nullptr;  // positions with fewer than m+S valid bases, per region
    bool withVals = false;                  // also carry positions through the sort
    bool wide = false;                      // also carry position bits 32..39 (texts of 2^32 bases or more)
    // Tier 1 on the GPU: MAW counts for depths [dLo, dHi], and the group bits for depth kRep.
    bool tier1 = false;
    int dLo = 0, dHi = 0, kRep = 0;
};

// Thin wrapper over Metal. All buffers use shared storage, so the CPU reads results in place.
// kSlots batch slots let the GPU sort the next batches while the CPU scans an earlier one.
class Gpu {
public:
    static constexpr int kSlots = 3;
    Gpu();
    ~Gpu();
    bool init(std::string& err);
    std::string name() const;
    uint64_t workingSetBytes() const;
    uint64_t maxBufferBytes() const;

    // Wraps the packed text and segment bitmap (page-aligned, not copied: keep t alive and
    // unchanged until releaseText).
    void setText(const Text& t);
    void releaseText();
    // counts[(4^l - 4)/3 + w] = positions whose word of length min(valid length, m+1) is w;
    // shortCounts[u] = positions with m-prefix u and between m and m+S-1 valid bases; presence has
    // bit (4^l - 4)/3 + w set when w is the word of length min(valid length, kCensusLen) somewhere.
    void census(int m, int S, std::vector<uint32_t>& counts, std::vector<uint32_t>& shortCounts,
                std::vector<uint32_t>& presence, double& gpuSec);

    void reserveBatches(uint64_t entries, uint64_t tiles, bool withVals, bool wide, bool tier1);
    void releaseBatches();
    // Encodes and commits a batch into a slot without waiting; onDone runs (on a Metal thread) once
    // it is sorted. waitBatch blocks until then and collects the GPU times. A slot may be submitted
    // while the other one is still on the GPU.
    void submitBatch(int slot, const BatchSpec& b, std::function<void()> onDone);
    void waitBatch(int slot, GpuBatchTimes& t);
    const uint64_t* keys(int slot) const;
    const uint32_t* vals(int slot) const;  // positions, low 32 bits
    const uint8_t* hi(int slot) const;     // bits 32..39 when BatchSpec::wide, else null
    // With BatchSpec::tier1: MAWs by depth - dLo, and bit i set when sorted suffixes i and i+1
    // (of the same region) share kRep bases.
    const uint32_t* tier1Counts(int slot) const;
    const uint32_t* groupBits(int slot) const;

    // Sort tiles needed for regions of the given sizes (each region is tiled separately).
    static uint64_t tilesFor(const uint64_t* counts, size_t nRegions);
    static int passesFor(int S);

private:
    struct Impl;
    Impl* p_;
};
