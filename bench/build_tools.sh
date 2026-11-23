#!/bin/bash
# Copyright (c) 2026 Nicolas Innocenti
# SPDX-License-Identifier: MIT
# Fetches and builds every program of the benchmark into tools/<arch>/ (arm64 or x86_64, so one copy of
# this folder can serve several Macs). Sources and pinned versions are listed in programs.tsv:
#   MAW, pMAW, emMAW   the MAW suite (Barton et al. 2014, 2016; Heliou et al. 2017), with the sdsl-lite
#                      and libdivsufsort 2.0.1 it ships
#   bwtman             Belazzougui and Cunial (2017), Cunial's prototype
#   Pinho et al.       the 2009 release, which cannot be downloaded: built only when the tarball named in
#                      programs.tsv is in vendor/, skipped otherwise
#   mawmetal           this repository, built from a copy of ../src (Apple Silicon only)
#   OpenMP runtime     LLVM 19.1.7, linked statically (Apple clang ships none)
# Porting to current macOS took a few changes, none in the algorithms:
#   * sdsl-lite: louds_tree.hpp refers to tree.m_select1/0, a typo for m_bv_select1/0.
#   * emMAW: int64_t is long long on macOS, so two INT to uint40 conversions are ambiguous.
#   * pSAscan, EM-SparsePhi: GNU parallel-mode sorts (<parallel/algorithm>) map to std::sort;
#     posix_fadvise, absent from macOS, becomes a no-op; integer literals and a getline buffer take
#     the macOS 64-bit types.
#   * Pinho et al.: one declaration, gzFile *fp -> gzFile fp, for current zlib.
#   * The MAW suite's x86-only -msse3 and bwtman's gcc-9 are dropped; everything uses Apple clang.
# Whatever is already built is kept; delete tools/<arch> to rebuild from scratch. mawmetal is rebuilt
# from the current ../src on every call (make only recompiles what changed).
# Requires the Command Line Tools, git, curl and CMake (3.20 or later).
# usage: build_tools.sh
set -euo pipefail
cd "$(dirname "$0")"
HERE=$PWD
ARCH=$(uname -m)
# under Rosetta uname reports x86_64; build for the machine itself
[ "$(sysctl -n sysctl.proc_translated 2>/dev/null)" = 1 ] && ARCH=arm64
T=$HERE/tools/$ARCH
SRC=$T/src
BIN=$T/bin
OMP=$T/omp
NCPU=$(sysctl -n hw.ncpu)
mkdir -p "$SRC" "$BIN"
for c in git curl cmake clang++; do
    command -v $c > /dev/null || { echo "build_tools.sh: $c not found" >&2; exit 1; }
done

SDK=$(xcrun --show-sdk-path)
LIBCXX=""
# Some Command Line Tools installs ship an empty libc++ header directory; fall back to the SDK's copy.
echo '#include <vector>' | clang++ -x c++ -fsyntax-only - 2>/dev/null || LIBCXX="-nostdinc++ -isystem $SDK/usr/include/c++/v1"
CC="clang -arch $ARCH"
CXX="clang++ -arch $ARCH"
CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release -DCMAKE_OSX_ARCHITECTURES=$ARCH -DCMAKE_C_COMPILER=clang
            -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_POLICY_VERSION_MINIMUM=3.5"

prog() {  # name column: a field of programs.tsv (2 source, 3 version)
    awk -F'\t' -v n="$1" -v c="$2" '$1 == n {print $c}' "$HERE/programs.tsv"
}

clone() {  # url dir commit
    if [ ! -d "$SRC/$2" ]; then
        rm -rf "$SRC/$2.tmp"
        git clone --quiet "$1" "$SRC/$2.tmp"
        git -C "$SRC/$2.tmp" checkout --quiet "$3"
        mv "$SRC/$2.tmp" "$SRC/$2"
    fi
}

fetch() {  # url file sha256
    if [ ! -f "$2" ]; then
        curl -sSfL --retry 3 -o "$2.part" "$1"
        mv "$2.part" "$2"
    fi
    echo "$3  $2" | shasum -a 256 -c --status || { echo "build_tools.sh: checksum mismatch for $2" >&2; exit 1; }
}

# OpenMP runtime
if [ ! -f "$OMP/lib/libomp.a" ]; then
    echo "building the OpenMP runtime (LLVM 19.1.7)"
    V=19.1.7
    mkdir -p "$SRC/openmp"
    (
        cd "$SRC/openmp"
        fetch "$(prog openmp 2)" openmp-$V.src.tar.xz "$(prog openmp 3)"
        fetch "$(prog openmp-cmake 2)" cmake-$V.src.tar.xz "$(prog openmp-cmake 3)"
        rm -rf openmp-$V.src cmake-$V.src cmake build
        tar xf openmp-$V.src.tar.xz
        tar xf cmake-$V.src.tar.xz
        mv cmake-$V.src cmake
        cmake -S openmp-$V.src -B build $CMAKE_ARGS -DCMAKE_CXX_FLAGS="$LIBCXX" \
            -DLIBOMP_ENABLE_SHARED=OFF -DOPENMP_ENABLE_LIBOMPTARGET=OFF -DLIBOMP_OMPT_SUPPORT=OFF \
            -DLIBOMP_OMPD_SUPPORT=OFF -DOPENMP_ENABLE_OMPT_TOOLS=OFF -DLIBOMP_INSTALL_ALIASES=OFF \
            -DCMAKE_INSTALL_PREFIX="$OMP" > /dev/null
        cmake --build build -j "$NCPU" > /dev/null
        cmake --install build > /dev/null
    )
fi
OMPFLAGS="-Xclang -fopenmp -I$OMP/include"
OMPLIB=$OMP/lib/libomp.a

clone "$(prog maw 2)" maw "$(prog maw 3)"
clone "$(prog bwtman 2)" portfolio "$(prog bwtman 3)"

# sdsl-lite and its libdivsufsort 2.0.1, from the MAW suite
S=$SRC/sdsl-lite
if [ ! -f "$S/build/lib/libsdsl.a" ]; then
    echo "building sdsl-lite"
    rm -rf "$S"
    (cd "$SRC" && tar -xzf maw/sdsl-lite.tar.gz)
    sed -i '' -e 's/tree\.m_select1,/tree.m_bv_select1,/; s/tree\.m_select0,/tree.m_bv_select0,/' \
        "$S/include/sdsl/louds_tree.hpp"
    cmake -S "$S" -B "$S/build" $CMAKE_ARGS -DCMAKE_CXX_FLAGS="$LIBCXX -w" > /dev/null
    # configuring copies the headers into build/include, typo included
    sed -i '' -e 's/tree\.m_select1,/tree.m_bv_select1,/; s/tree\.m_select0,/tree.m_bv_select0,/' \
        "$S/build/include/sdsl/louds_tree.hpp" 2>/dev/null || true
    cmake --build "$S/build" --target sdsl divsufsort divsufsort64 -j "$NCPU" > /dev/null
fi
D=$S/build/external/libdivsufsort-2.0.1
INC="-I$S/build/include -I$S/include -I$D/include"
LIBS="$S/build/lib/libsdsl.a $D/lib/libdivsufsort.a $D/lib/libdivsufsort64.a"
MAWCXX="$CXX -std=c++11 -O3 -DNDEBUG -fomit-frame-pointer -funroll-loops -w $LIBCXX"

# MAW and pMAW: the suite's flags, minus the x86-only -msse3. pMAW writes one temporary file per
# thread before concatenating them, so it is also built without its output stage (pmaw32-noout).
M=$SRC/maw
for b in 32 64; do
    [ -x "$BIN/maw$b" ] || (cd "$M" && $MAWCXX -D_USE_$b $INC maw.cc input.cc stack.cc functions.cc $LIBS -o "$BIN/maw$b")
done
[ -x "$BIN/pmaw32" ] || (cd "$M/pmaw" && $MAWCXX -D_LOG_WRITE -D_LOG_OUT -D_USE_32 $OMPFLAGS $INC \
    maw.cc input.cc stack.cc functions.cc $LIBS $OMPLIB -o "$BIN/pmaw32")
[ -x "$BIN/pmaw32-noout" ] || (cd "$M/pmaw" && $MAWCXX -D_LOG_WRITE -D_USE_32 $OMPFLAGS $INC \
    maw.cc input.cc stack.cc functions.cc $LIBS $OMPLIB -o "$BIN/pmaw32-noout")

# emMAW and its two helper programs, which it runs from the directory of its own binary
E=$T/emmaw
if [ ! -x "$E/em-maw64" ]; then
    echo "building emMAW"
    rm -rf "$E"
    mkdir -p "$E/shim/parallel" "$E/emmaw-src"
    printf '%s\n' '// libstdc++ parallel mode is not available with libc++: the parallel sorts run sequentially.' \
        '#pragma once' '#include <algorithm>' 'namespace __gnu_parallel { using std::sort; }' > "$E/shim/parallel/algorithm"
    SHIM="-I$E/shim -DPOSIX_FADV_DONTNEED=4 -Dposix_fadvise(fd,off,len,adv)=0"
    (cd "$E" && tar -xzf "$M/em-maw/pSAscan-0.1.0.tar.gz")
    sed -i '' 's/divsufsort64(text, sa, length);/divsufsort64(text, (saidx64_t *)sa, (saidx64_t)length);/' \
        "$E/pSAscan-0.1.0/src/psascan_src/inmem_psascan_src/divsufsort_template.h"
    (cd "$E/pSAscan-0.1.0/src" && $CXX -funroll-loops -pthread -std=c++11 -DNDEBUG -O3 -w $LIBCXX $SHIM $INC \
        $OMPFLAGS -o psascan ./psascan_src/utils.cpp main.cpp $LIBS $OMPLIB)
    (cd "$E" && tar -xzf "$M/em-maw/EM-SparsePhi-0.1.0.tar.gz")
    (cd "$E/EM-SparsePhi-0.1.0/src" &&
        find . \( -name "*.hpp" -o -name "*.cpp" -o -name "*.h" \) -exec perl -pi -e \
            's/\b(\d+)UL\b/$1ULL/g; s/\b(\d+)L\b/$1LL/g; s/(\.\d+)LL\b/$1L/g' {} + &&
        sed -i '' 's/    std::uint64_t buflen = 0;/    size_t buflen = 0;/' main.cpp &&
        $CXX -funroll-loops -std=c++11 -O3 -DNDEBUG -pthread -w $LIBCXX $SHIM $OMPFLAGS \
            -o construct_lcp_parallel em_sparse_phi_src/utils.cpp main.cpp $OMPLIB)
    cp -p "$M/em-maw/"*.cc "$M/em-maw/"*.h "$E/emmaw-src/"
    sed -i '' 's/uint40 start = SAmem;/uint40 start = (long) SAmem;/' "$E/emmaw-src/functions.cc"
    (cd "$E/emmaw-src" && $MAWCXX -D_USE_64 $SHIM $INC $OMPFLAGS maw.cc input.cc functions.cc utils.cc stackinfile.cc \
        $LIBS $OMPLIB -o "$E/em-maw64")
fi

# Pinho et al. (2009), with its own flags (-O2), when its tarball has been put in vendor/
P=$SRC/pinho
PT=$HERE/$(prog pinho 2)
if [ ! -x "$BIN/pinho" ] && [ -f "$PT" ]; then
    echo "$(prog pinho 3)  $PT" | shasum -a 256 -c --status || { echo "build_tools.sh: checksum mismatch for $PT" >&2; exit 1; }
    rm -rf "$P" && mkdir -p "$P"
    tar -xzf "$PT" -C "$P" --strip-components 1
    sed -i '' 's/gzFile \*fp;/gzFile fp;/' "$P/MinimalAbsentWords.c"
    make -C "$P" CC="$CC" > /dev/null 2>&1
    cp -p "$P/MinimalAbsentWords" "$BIN/pinho"
elif [ ! -x "$BIN/pinho" ]; then
    echo "build_tools.sh: $PT not found, so Pinho et al. is skipped (see programs.tsv)" >&2
fi

# bwtman: its bundled libdivsufsort (64-bit), then the index builder and the MAW enumerator
W=$SRC/portfolio/bwtman
if [ ! -x "$BIN/bwtman-index" ] || [ ! -x "$BIN/bwtman-maws" ]; then
    echo "building bwtman"
    cmake -S "$W/libdivsufsort" -B "$W/libdivsufsort/build" $CMAKE_ARGS -DBUILD_DIVSUFSORT64=ON \
        -DBUILD_SHARED_LIBS=OFF > /dev/null
    cmake --build "$W/libdivsufsort/build" -j "$NCPU" > /dev/null
    make -C "$W" program-1 program-3 CC="$CC" CFLAGS="$OMPFLAGS -Wall -O3 -w" LIBS="-lm $OMPLIB" > /dev/null
    cp -p "$W/buildIndex" "$BIN/bwtman-index"
    cp -p "$W/run_MAWs_single" "$BIN/bwtman-maws"
fi

# mawmetal, from a copy of the repository's sources, so that the benchmark never races a build in
# ../build and the result files can record exactly which sources were timed
if [ "$ARCH" = arm64 ] && [ -f "$HERE/../src/main.cpp" ]; then
    clone "$(prog libsais 2)" libsais "$(prog libsais 3)"
    MM=$SRC/mawmetal
    mkdir -p "$MM"
    rsync -a --delete "$HERE/../src" "$HERE/../Makefile" "$MM/"
    make -C "$MM" -j "$NCPU" build/mawmetal CC="$CC" CXX="$CXX" LIBSAIS="$SRC/libsais" OMP="$OMP" > "$MM/make.log" 2>&1 ||
        { cat "$MM/make.log" >&2; exit 1; }
    cp -p "$MM/build/mawmetal" "$BIN/mawmetal"
fi

{
    [ -x "$BIN/pinho" ] && echo "pinho	Pinho et al. 2009 release ($(prog pinho 2), sha256 $(prog pinho 3 | cut -c1-12)), gzFile fix"
    echo "maw	github.com/solonas13/maw $(git -C "$M" rev-parse --short HEAD), maw32 and maw64"
    echo "pmaw	github.com/solonas13/maw $(git -C "$M" rev-parse --short HEAD), pmaw/, with and without output stage"
    echo "emmaw	github.com/solonas13/maw $(git -C "$M" rev-parse --short HEAD), em-maw/ with pSAscan 0.1.0 and EM-SparsePhi 0.1.0"
    echo "bwtman	github.com/fcunial/portfolio $(git -C "$SRC/portfolio" rev-parse --short HEAD), bwtman/"
    [ -x "$BIN/mawmetal" ] && echo "mawmetal	this repository, src/ tree hash $(cd "$SRC/mawmetal" && find src Makefile -type f | LC_ALL=C sort | xargs shasum -a 256 | shasum -a 256 | cut -c1-12), libsais $(git -C "$SRC/libsais" rev-parse --short HEAD)"
    echo "compiler	$(clang --version | head -1)"
    echo "openmp	LLVM 19.1.7 libomp (static)"
} > "$T/versions.tsv"
echo "programs ready in tools/$ARCH/"
