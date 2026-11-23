// Copyright (c) 2026 Nicolas Innocenti
// SPDX-License-Identifier: MIT
// Writes a synthetic FASTA genome to stdout without touching the disk: one random sequence of
// LENGTH bases with planted (sometimes mutated) repeats and occasional N runs, emitted COPIES times
// as separate records. MAWs of the copies taken jointly equal those of one copy, which makes
// this a check for texts larger than 2^32 bases.
// usage: gen_genome LENGTH SEED [COPIES]
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

static uint64_t s;
static uint64_t next() {  // splitmix64
    uint64_t z = (s += 0x9e3779b97f4a7c15ull);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: gen_genome LENGTH SEED [COPIES]\n");
        return 1;
    }
    const uint64_t len = strtoull(argv[1], nullptr, 10);
    s = strtoull(argv[2], nullptr, 10);
    const int copies = argc > 3 ? atoi(argv[3]) : 1;
    std::string g(len, 'A');
    const char* acgt = "ACGT";
    for (uint64_t i = 0; i < len; i += 32) {
        uint64_t r = next();
        for (uint64_t j = i; j < i + 32 && j < len; ++j, r >>= 2) g[j] = acgt[r & 3];
    }
    // Repeats of length 32 bases to 64 kb, so every tier (and the deep stage's limit) sees work.
    for (uint64_t k = 0; k < len / 20000 + 4; ++k) {
        uint64_t L = 32ull << (next() % 12);
        if (L * 2 >= len) continue;
        uint64_t a = next() % (len - L), b = next() % (len - L);
        g.replace(b, L, g, a, L);
        if (next() % 2) g[b + next() % L] = acgt[next() & 3];
    }
    for (uint64_t k = 0; k < len / 5000000 + 1; ++k) {
        uint64_t L = 1 + next() % 1000, b = next() % len;
        for (uint64_t j = b; j < b + L && j < len; ++j) g[j] = 'N';
    }
    for (int c = 0; c < copies; ++c) {
        printf(">copy%d\n", c);
        for (uint64_t i = 0; i < len; i += 1 << 20) {
            fwrite(g.data() + i, 1, std::min<uint64_t>(1 << 20, len - i), stdout);
            fputc('\n', stdout);
        }
    }
    return 0;
}
