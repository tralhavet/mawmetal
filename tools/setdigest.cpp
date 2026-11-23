// Copyright (c) 2026 Nicolas Innocenti
// SPDX-License-Identifier: MIT
// Order-independent digest of the words in a MAW-format stream, for comparing outputs too large
// to store or sort. Reads stdin; ignores '>' headers and blank lines; with --acgt-only, ignores
// words containing anything but ACGT (MAW treats N as a letter). Prints the word count and two
// 64-bit sums of independent per-word hashes: equal digests mean equal multisets (w.h.p.).
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static inline uint64_t mix(uint64_t x) {
    x ^= x >> 30, x *= 0xbf58476d1ce4e5b9ull;
    x ^= x >> 27, x *= 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}

int main(int argc, char** argv) {
    bool acgtOnly = argc > 1 && strcmp(argv[1], "--acgt-only") == 0;
    std::vector<char> buf(1 << 24);
    std::string carry;
    uint64_t count = 0, s1 = 0, s2 = 0, skipped = 0;
    auto word = [&](const char* p, size_t n) {
        if (n == 0 || p[0] == '>') return;
        uint64_t h1 = 0x9e3779b97f4a7c15ull ^ n, h2 = 0x632be59bd9b4e019ull + n;
        bool ok = true;
        for (size_t i = 0; i < n; ++i) {
            unsigned char c = (unsigned char)p[i];
            ok &= c == 'A' || c == 'C' || c == 'G' || c == 'T';
        }
        for (size_t i = 0; i < n; i += 8) {
            uint64_t v = 0;
            memcpy(&v, p + i, n - i < 8 ? n - i : 8);
            h1 = mix(h1 ^ v);
            h2 = mix(h2 + v * 0x100000001b3ull);
        }
        if (acgtOnly && !ok) {
            ++skipped;
            return;
        }
        ++count, s1 += h1, s2 += h2;
    };
    size_t r;
    while ((r = fread(buf.data(), 1, buf.size(), stdin)) > 0) {
        const char* p = buf.data();
        const char* end = p + r;
        while (p < end) {
            const char* nl = (const char*)memchr(p, '\n', end - p);
            if (!nl) {
                carry.append(p, end - p);
                break;
            }
            if (!carry.empty()) {
                carry.append(p, nl - p);
                word(carry.data(), carry.size());
                carry.clear();
            } else {
                word(p, nl - p);
            }
            p = nl + 1;
        }
    }
    word(carry.data(), carry.size());
    printf("words %llu digest %016llx%016llx", (unsigned long long)count, (unsigned long long)s1,
           (unsigned long long)s2);
    if (acgtOnly) printf(" (skipped %llu non-ACGT)", (unsigned long long)skipped);
    printf("\n");
    return 0;
}
