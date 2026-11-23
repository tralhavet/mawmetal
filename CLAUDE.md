# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

mawmetal computes the minimal absent words (MAWs) of DNA sequences on Apple Silicon, splitting the work
between the GPU (Metal) and the CPU. Its output must match MAW (Barton et al. 2014) exactly, restricted
to ACGT words. README.md holds the full algorithm description, validation record and performance
numbers; read its "How it works" section before changing any stage.

## Build

```sh
scripts/fetch_deps.sh               # once: libsais, libdeflate, a static arm64 OpenMP into third_party/
scripts/fetch_deps.sh --reference   # also the original MAW (third_party/maw_build/maw32), for comparisons
make                                # build/mawmetal, build/setdigest, build/gen_genome
make CPU=native                     # tune for this machine instead of the M1 baseline
```

- macOS on Apple Silicon only, Command Line Tools (no Xcode). libdeflate and OpenMP are optional: the
  Makefile detects them in `third_party/` and falls back to zlib / single-threaded libsais.
- `src/kernels.metal` is not compiled at build time. `make` wraps it in a raw string
  (`build/kernels_src.h`), and `Gpu::init` compiles it at startup on a background thread. **A shader
  error shows up only when you run the binary**, so always run something after editing the kernels.

## Tests

```sh
python3 tools/run_tests.py                    # 200 randomized cases vs the brute-force reference
python3 tools/run_tests.py -n 300 --seed 2    # the README's "full run" is 3 × 300 with different seeds
python3 tools/run_tests.py -n 20 --keep /tmp/fails   # keep the FASTA/outputs of failing cases
```

- Each failing case prints the exact mawmetal command. To reproduce one by hand:
  `python3 tools/brute_maw.py -i case.fa -k K -K K -r 0|1 [--joint] > ref`, then
  `python3 tools/compare_maw.py ours ref [--counts]` (`--counts` when `ours` is the default length
  distribution; `--acgt-only` when comparing against the real MAW, which also reports words with N).
- Large outputs (chr21, chr1) are compared with `build/setdigest`, an order-independent multiset hash,
  since `--list` order differs from MAW's.
- `MAWMETAL_CHECK_SORT=1` verifies every GPU-sorted batch against the text.
- `MAWMETAL_SAIS64=1` forces the 64-bit libsais path.
- `scripts/test_large.sh` exercises positions ≥ 2^32 (the 40-bit high-byte path). It needs ~8 GB of free
  RAM and has never been run; `--wide` only covers the code path on small inputs, where the high byte
  is always 0.

### Test-only CLI knobs

These exist so tiny inputs still hit every boundary. `run_tests.py` randomizes them, and a new tunable
should be added there too:

| flag | controls |
|---|---|
| `--prefix m` | region prefix length, giving 4^m GPU regions: by default 4, or 5 when the largest region exceeds the batch budget (`--batch` counts as one, so tiny tests switch too). `run_tests.py` leaves it out in a quarter of the cases |
| `--split k` | `kRep`, the tier-1 / deep boundary: `min(31, m+28)` by default, must be in `[m+1, m+28]` |
| `--deep D` | deep-stage / libsais tail boundary (1024 by default) |
| `--batch`, `--first` | GPU batch size, and the size of the first (smaller, doubling) batches |
| `--chunk`, `--piece` | CPU scan chunk size, input packing piece size |
| `--pending`, `--spill` | `--list`: output bytes finished chunks may hold while workers run ahead (a batch slot, at least 128 MB), and how often the chunk next in line writes its own output (1 MB) |
| `--wide` | use 40-bit positions even for short texts |
| `--cpu-tier1`, `--gpu-tier1` | force where the tier-1 scan runs (distribution mode only) |

## Architecture

The work is split by `|W|`, the middle of a MAW `aWb`. The stage table and the correctness argument
for the split are in README.md. Where each stage lives:

- `src/main.cpp`: option parsing and the per-text pipeline: census → batch loop (GPU sort, then a
  tier-1 scan on the GPU or CPU, then the deep stage on the CPU) → residue → tier 3 (libsais). It also
  owns the 3-slot batch pipelining and the per-stage timings printed by `-v`.
- `src/gpu.mm` / `src/gpu.h`: thin Objective-C++ wrapper over Metal. All buffers use shared storage
  and wrap host memory without copying, which is why `PageVec` (`src/maw.h`) allocates whole 16 KB
  pages. `BatchSpec` describes one batch (the prefix range `[lo, hi)`, split into regions).
- `src/kernels.metal`: `census`, `extract` (builds the 64-bit suffix keys), the segmented LSD radix
  sort (`radix_hist`, `radix_scatter`, `copy_tiles`), prefix scans, and the GPU tier-1 scan (`t1_*`).
- `src/cpu.cpp`: everything else on the CPU. Sections are marked `// ---- name ----`: the output sink,
  the `WorkPool` thread pool and ordered runner, short MAWs from the census, the tier-1 LCP-interval
  scan, the `DeepScanner` deep stage, the residue, and tier 3.
- `src/input.cpp`: FASTA reading and parallel 2-bit packing (NEON fast path), gzip via libdeflate (BGZF
  in parallel) or a zlib thread, and the reverse complement.
- `src/maw.h`: shared types. `Text` is the packed genome: 16 bases per `uint32`, first base in the high
  bits, A=0 C=1 G=2 T=3. Any non-ACGT character, record boundary or reverse-complement junction only
  starts a new *segment* (`segBits`, `segCoarse`, `segStarts`); no reported word spans one.

The invariants that the stages share and that are easy to break:

- Tier 1 never reports a node shallower than 7 bases (the census already covers those), which lets
  scan chunks be cut wherever adjacent sorted keys share fewer than 7 bases.
- The 64-bit tier-1 key packs the 28 bases after the prefix (zeroed past the segment end), the valid
  length and the preceding base. The CPU scan never reads the genome, only the keys.
- Positions are 32-bit, plus a separate high-byte array (bits 32–39) for texts of 2^32 bases or more.
- `--list` and the default distribution must agree. The tests check the distribution against the
  list's lengths.
- With `--list`, `OutBuf::reserve()` may write the buffer out and restart it (the chunk next in line
  streams its output), so never keep a pointer into an `OutBuf` across `reserve()` calls.

## Benchmarks and paper

- `scripts/bench.sh file.fa ...`: quick wall-clock comparison of MAW and mawmetal.
- `bench/`: the full, reproducible benchmark of every published MAW program (see `bench/README.md`).
  `bench/run.sh --plan` shows what would run; `bench/run.sh --download-only` fetches the pinned
  programs (`programs.tsv`) and genomes (`genomes.tsv`). `bench/tools/` holds a **copy** of `src/`
  (rebuilt by `build_tools.sh` when the sources change), so don't edit mawmetal there.
- `paper/`: the LaTeX paper (pdflatex, PNAS class). The `paper/tools/*.py` scripts write the
  `paper/generated/*.tex` macros, tables and TikZ figures from `bench/results/*.json` and the Claude Code transcripts. Regenerate
  those files rather than editing them.
- `bench/tools/`, `third_party/`, `data/` and `build/` are git-ignored: other people's code, the
  genomes, and build output. So is every other piece of data (`bench/results/`, `paper/2015/`,
  `paper/generated/*.json`) and the PNAS class (`paper/pnastwo.*`, see `paper/README.md`); the
  generated `.tex` files and `paper/main.pdf` are kept. Pinho et al. 2009 (`bench/vendor/` and `reference/`) is part of the project,
  shared with the permission of Armando J. Pinho (October 2026); credit him wherever it is used.
