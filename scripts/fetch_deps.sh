#!/bin/bash
# Copyright (c) 2026 Nicolas Innocenti
# SPDX-License-Identifier: MIT
# Fetches libsais (needed to build) and, with --reference, the original MAW plus libdivsufsort
# (only needed to validate and benchmark against MAW). Everything lands in third_party/.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p third_party
SDK=$(xcrun --show-sdk-path)

clone() {  # url dir commit
    if [ ! -d "third_party/$2" ]; then
        git clone --quiet https://github.com/$1 "third_party/$2"
        git -C "third_party/$2" checkout --quiet "$3"
    fi
}

clone IlyaGrebnov/libsais libsais c284ba8

# libdeflate: gzip decompression several times faster than zlib, and independent BGZF blocks
# decompressed in parallel. Optional: without it the build falls back to zlib.
clone ebiggers/libdeflate libdeflate v1.22
if [ ! -f third_party/libdeflate/build/libdeflate.a ]; then
    cmake -S third_party/libdeflate -B third_party/libdeflate/build -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_C_COMPILER=clang -DLIBDEFLATE_BUILD_SHARED_LIB=OFF \
        -DLIBDEFLATE_BUILD_GZIP=OFF -DCMAKE_POLICY_VERSION_MINIMUM=3.5 > /dev/null
    cmake --build third_party/libdeflate/build -j 8 > /dev/null
    echo "libdeflate built: third_party/libdeflate/build/libdeflate.a"
fi

LIBCXX=""
echo '#include <vector>' | clang++ -x c++ -fsyntax-only - 2>/dev/null || LIBCXX="-nostdinc++ -isystem $SDK/usr/include/c++/v1"

# An arm64 OpenMP runtime for multithreaded libsais (Homebrew's may be x86_64), linked statically.
if [ ! -f third_party/omp/lib/libomp.a ]; then
    V=19.1.7
    mkdir -p third_party/openmp_src
    (
        cd third_party/openmp_src
        for p in openmp cmake; do
            [ -f $p-$V.src.tar.xz ] || curl -sSfL -o $p-$V.src.tar.xz \
                https://github.com/llvm/llvm-project/releases/download/llvmorg-$V/$p-$V.src.tar.xz
            tar xf $p-$V.src.tar.xz
        done
        rm -rf cmake && mv cmake-$V.src cmake
        cmake -S openmp-$V.src -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_OSX_ARCHITECTURES=arm64 \
            -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_CXX_FLAGS="$LIBCXX" \
            -DLIBOMP_ENABLE_SHARED=OFF -DOPENMP_ENABLE_LIBOMPTARGET=OFF -DLIBOMP_OMPT_SUPPORT=OFF \
            -DLIBOMP_OMPD_SUPPORT=OFF -DOPENMP_ENABLE_OMPT_TOOLS=OFF -DLIBOMP_INSTALL_ALIASES=OFF \
            -DCMAKE_INSTALL_PREFIX="$PWD/../omp" -DCMAKE_POLICY_VERSION_MINIMUM=3.5 > /dev/null
        cmake --build build -j 8 > /dev/null
        cmake --install build > /dev/null
    )
    echo "libomp built: third_party/omp/lib/libomp.a"
fi

if [ "${1:-}" = "--reference" ]; then
    clone solonas13/maw maw d69fa88
    clone y-256/libdivsufsort libdivsufsort 5f60d6f
    # Homebrew under /usr/local may provide an x86_64 cmake, so force the native architecture.
    cmake -S third_party/libdivsufsort -B third_party/libdivsufsort/build -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_OSX_ARCHITECTURES=arm64 -DBUILD_DIVSUFSORT64=ON -DBUILD_SHARED_LIBS=OFF \
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5 > /dev/null
    cmake --build third_party/libdivsufsort/build -j 8 > /dev/null
    # MAW only uses sdsl::bit_vector; a std::vector<bool> stand-in avoids building sdsl-lite.
    mkdir -p third_party/maw_build/sdsl
    cat > third_party/maw_build/sdsl/bit_vectors.hpp <<'EOF'
// Minimal stand-in for sdsl::bit_vector so the reference MAW builds without sdsl-lite.
#pragma once
#include <vector>
#include <cstdint>
namespace sdsl {
class bit_vector : public std::vector<bool> {
public:
    bit_vector() = default;
    bit_vector(uint64_t n, int v) : std::vector<bool>(n, v != 0) {}
};
}
EOF
    D=third_party/libdivsufsort/build
    for bits in 32 64; do
        (cd third_party/maw && clang++ -std=c++11 -O3 -DNDEBUG -D_USE_$bits -w $LIBCXX -I../maw_build \
            -I../libdivsufsort/build/include maw.cc input.cc stack.cc functions.cc \
            ../libdivsufsort/build/lib/libdivsufsort.a ../libdivsufsort/build/lib/libdivsufsort64.a \
            -o ../maw_build/maw$bits)
    done
    echo "reference MAW built: third_party/maw_build/maw32, maw64"
fi
