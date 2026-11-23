# Copyright (c) 2026 Nicolas Innocenti
# SPDX-License-Identifier: MIT
CXX      ?= clang++
CC       ?= clang
SDK      := $(shell xcrun --show-sdk-path)
# Some Command Line Tools installs ship an empty libc++ header directory; fall back to the SDK's copy.
LIBCXX   := $(shell echo '\#include <vector>' | $(CXX) -x c++ -fsyntax-only - 2>/dev/null || echo "-nostdinc++ -isystem $(SDK)/usr/include/c++/v1")
LIBSAIS  := third_party/libsais
# A fresh checkout fetches the dependencies before the checks below look for them.
ifeq ($(wildcard $(LIBSAIS)/src/libsais.c),)
$(shell ./scripts/fetch_deps.sh >&2)
endif
# arm64 OpenMP runtime built by scripts/fetch_deps.sh; without it libsais runs single-threaded.
OMP      := third_party/omp
ifneq ($(wildcard $(OMP)/lib/libomp.a),)
OMPFLAGS := -DLIBSAIS_OPENMP -Xclang -fopenmp -I$(OMP)/include
OMPLIBS  := $(OMP)/lib/libomp.a
endif
# libdeflate (scripts/fetch_deps.sh): faster gzip decompression, multithreaded for BGZF files.
DEFLATE  := third_party/libdeflate
ifneq ($(wildcard $(DEFLATE)/build/libdeflate.a),)
DEFLATEFLAGS := -DHAVE_LIBDEFLATE -I$(DEFLATE)
DEFLATELIBS  := $(DEFLATE)/build/libdeflate.a
endif
# CPU=native tunes for the build machine instead of the M1 baseline.
CPU      ?= apple-m1
OPT      := -O3 -mcpu=$(CPU) -flto=thin
CXXFLAGS := -std=c++17 $(OPT) -Wall -Wextra -Wno-unused-parameter $(LIBCXX) -I$(LIBSAIS)/include -Ibuild $(if $(OMPLIBS),-DLIBSAIS_OPENMP -I$(OMP)/include) $(DEFLATEFLAGS)
CFLAGS   := $(OPT) -I$(LIBSAIS)/include $(OMPFLAGS)
LDLIBS   := -framework Metal -framework Foundation -lz $(OMPLIBS) $(DEFLATELIBS)

BIN := build/mawmetal
OBJ := build/main.o build/cpu.o build/input.o build/gpu.o build/libsais.o build/libsais64.o

all: $(BIN) build/setdigest build/gen_genome

$(BIN): $(OBJ)
	$(CXX) $(OPT) -o $@ $^ $(LDLIBS)

build/kernels_src.h: src/kernels.metal
	@mkdir -p build
	{ printf 'static const char* kKernelSource = R"METAL('; cat $<; printf ')METAL";\n'; } > $@

build/gpu.o: src/gpu.mm src/gpu.h src/maw.h build/kernels_src.h
	$(CXX) $(CXXFLAGS) -fobjc-arc -c $< -o $@

build/%.o: src/%.cpp src/maw.h src/gpu.h
	@mkdir -p build
	$(CXX) $(CXXFLAGS) -c $< -o $@

build/libsais.o: $(LIBSAIS)/src/libsais.c $(OMPLIBS)
	@mkdir -p build
	$(CC) $(CFLAGS) -c $< -o $@

build/libsais64.o: $(LIBSAIS)/src/libsais64.c $(OMPLIBS)
	@mkdir -p build
	$(CC) $(CFLAGS) -c $< -o $@

build/setdigest: tools/setdigest.cpp
	@mkdir -p build
	$(CXX) $(CXXFLAGS) $< -o $@

build/gen_genome: tools/gen_genome.cpp
	@mkdir -p build
	$(CXX) $(CXXFLAGS) $< -o $@

clean:
	rm -rf build

.PHONY: all clean
