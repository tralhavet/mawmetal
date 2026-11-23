# MAW benchmark

Times every published program for minimal absent words that we could obtain and run, on genomes from
a 0.24 Mbp virus to the 12 Gbp Norway spruce. Everything needed is in this folder; the genomes and
the other programs' code are fetched from pinned sources the first time.

## Running it

```sh
bench/run.sh --plan                    # what would run on this machine, what would be skipped and why
bench/run.sh                           # profile "full": everything, for the M1 Max (32 GB)
bench/run.sh --profile mbp2012         # for the MacBook Pro Retina 2012 (Intel, 16 GB)
bench/run.sh --check                   # count the MAWs of each length instead of timing (validation)
bench/run.sh --download-only           # only build the programs and download the genomes
bench/run.sh --genomes chr21 --tools maw,mawmetal --reps 1    # any subset
```

`run.sh` builds the programs into `tools/<arch>/` (3 minutes the first time on the M1 Max, longer on an
older Mac; afterwards only mawmetal is recompiled, if its sources changed) and runs `run_bench.py`; `--help` lists every option.

Before a timing run: mains power, other applications closed, and a look at `--plan`. Each run is killed
if swap grows by more than 24 GB, and `--timeout S` caps a single run.

## Results

Each invocation writes one file, `results/<machine>-<YYYYmmdd-HHMMSS>.json`, rewritten after every run
so an interrupted benchmark keeps what it measured. `<machine>` is the processor (`apple-m1-max`,
`intel-core-i7-3615qm`, ...) unless `--machine` names it. The file holds:

* `host`: model, processor, cores, memory, GPU, macOS version, and the version of every program
  (`tools/<arch>/versions.tsv`);
* `genomes`: records, length, ACGT bases and the sha256 of each genome's decompressed text;
* `runs`: one entry per run: program, genome, repetition, exit code, wall-clock, user and system
  seconds, peak resident memory, the MAW total (and with `--check` the count per length), the exact
  command, the input file, and the machine's load, free memory, swap and power source just before;
* `skipped`: each program/genome pair that was not run, with the reason.

## Programs

| name in the runner | program | code |
|---|---|---|
| `pinho`, `pinho-list` | Pinho et al. 2009, the software of our 2015 study | included in `vendor/`, shared with the kind permission of A. J. Pinho |
| `maw` | MAW, Barton et al. 2014 | github.com/solonas13/maw; 32-bit build, 64-bit above 2^31 characters |
| `pmaw` | pMAW, Barton et al. 2016 | same suite; all cores; built without its output stage (`pmaw-out`, used by `--check`, has it) |
| `emmaw` | emMAW, Heliou et al. 2017 | same suite; 8 GiB of RAM, about 20 bytes of temporary disk per character |
| `bwtman` | Belazzougui and Cunial 2017 (bidirectional BWT) | Cunial's prototype, github.com/fcunial/portfolio; builds a BWT index, then enumerates on all cores; both are timed |
| `mawmetal`, `mawmetal-list` | this repository | Apple Silicon only, built from a copy of `../src` |

`programs.tsv` pins the sources (commit or sha256) and gives their licenses; `build_tools.sh` lists the
portability fixes, none of them in the algorithms. The `-list` variants write the words to
`/dev/null`; the others report counts.

On the four smallest genomes (`--check`), all programs report the same number of MAWs of every length,
except pMAW, which on *S. cerevisiae* misses four MAWs and reports three words that occur (see the paper).

**Not included** (searched October 2026):

| work | why |
|---|---|
| Herold et al. 2008 | shortest absent words only |
| Ayad et al. 2021 | implementation "available upon request" |
| r-enum, Nishimoto and Tabei 2021 | the public code enumerates maximal repeats and minimal unique substrings, not MAWs |
| Fujishige et al. 2016/2023, Crochemore et al. 2020, Mieno et al. 2020, Akagi et al. 2022, Okabe et al. 2023, Inenaga et al. 2024, Kimura and I 2025 | no public implementation found |
| scMAW (MAW suite), CD-MAWS 2023 | sequence comparison; CD-MAWS runs MAW |
| cd-maws-sa (ML-MAWS, Saha et al. 2026) | MAWs shorter than 40 only (fixed buffer), no reverse complement, N read as T |

## Profiles

| profile | genomes | programs |
|---|---|---|
| `full` | hhv5, efaecalis, ecoli, yeast, chr21, chr1, grch38p2, spruce | all |
| `mbp2012` | hhv5, efaecalis, ecoli, yeast, chr21, chr1 | all but mawmetal (it needs Apple Silicon) |

Both run the same commands, so the two machines compare directly. Which pairs actually run depends on
the machine: a pair is skipped when its estimated memory or disk exceeds what the machine has
(`--force` runs it anyway). On the M1 Max only mawmetal fits the human genome and the spruce; on a
16 GB Mac, Pinho et al.'s list mode does not fit chr1, and its count mode just does.

## Protocol

All programs compute the MAWs of all lengths (bound 5,000,000) of the genome with its reverse
complement, all records together. Pinho et al., bwtman and mawmetal cut the sequence at every letter
other than ACGT. MAW, pMAW and emMAW take one record and treat N as a fifth letter, so they get the
genome as one record in which every record boundary and every run of other letters is a single N;
their words containing N are not counted. This leaves the MAWs over ACGT unchanged, and it matters: on
12 Mb of human chr21 with its N runs left in, MAW writes 41.7 GB of words containing N and takes 76 s,
against 0.9 GB and 12 s with one N per run. `--keep-n-runs` restores the runs (the protocol of the
first benchmarks).

Programs read uncompressed FASTA, made once from the download when the disk has room; otherwise
mawmetal and Pinho et al. read the gzipped file (the spruce needs about 17 GB free to be
decompressed), and each run records its input. Each program runs `reps` times per genome
(`genomes.tsv`: 3, 2 for chr1, 1 for the human genome and the spruce), the programs taking turns,
under `caffeinate`, with the input in the page cache.

## On the MacBook Pro Retina 2012

* Needs the Command Line Tools (`xcode-select --install`), CMake 3.20 or later (the installer from
  cmake.org runs on macOS 10.13 and later), git, curl, and Python 3.7 or later (the Command Line Tools
  of macOS 10.15 include 3.8).
* Copy this folder or clone the repository; `data/` can be copied along to save the downloads, and
  `tools/` is built per architecture, so both machines can share one copy.
* The Intel build has been built and checked under Rosetta on the M1 Max (same MAW counts), not yet on
  the old machine and its compiler.

## Files

```
run.sh            the one command
build_tools.sh    fetches and builds the programs into tools/<arch>/
run_bench.py      downloads the genomes, plans, runs, writes results/
genomes.tsv       genomes: URL, size, sha256, expected base counts, repetitions
programs.tsv      other programs: source, pinned version, license
vendor/           code that cannot be downloaded: Pinho et al. 2009, shared with permission
tools/<arch>/     built programs (not distributed)
data/             downloaded genomes and derived inputs (not distributed)
results/          one JSON file per run of the benchmark (not distributed)
```
