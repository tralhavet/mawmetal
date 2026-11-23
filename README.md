# mawmetal

> **Abstract.** mawmetal computes the minimal absent words of whole genomes on an Apple Silicon
> laptop, using its unified-memory GPU together with the CPU. Claude, a large language model, drove
> the work under the author's direction. Starting from the published literature, it designed the
> algorithm, implemented it in Metal and C++, optimized and validated it, and benchmarked it. Ten
> years ago, computing the MAW length distribution of the 12-gigabase Norway spruce genome and its
> reverse complement took several days on a supercomputer with 1 TB of memory. mawmetal does it in
> under four minutes on a laptop with 32 GB, and lists every MAW in about seven. It reproduces our
> 2015 distribution for the human genome exactly, all 8,377,095,148 MAWs, in 29 seconds. We tested
> every published MAW program we could obtain and run, and mawmetal is 10 to 100 times faster than
> all of them. The speed-up comes from fitting the algorithm to the hardware, not from newer
> hardware alone. Optimization like this used to take too long to attempt by hand.

Minimal absent words (MAWs) of DNA sequences, using the GPU of an Apple Silicon Mac (Metal) together
with the CPU.

A word is a minimal absent word of a sequence if it does not occur in it but all its proper factors do.
mawmetal computes the same MAWs as [MAW](https://github.com/solonas13/maw) (Barton, Heliou, Mouchard,
Pissis, *BMC Bioinformatics* 2014). By default it reports how many MAWs there are of each length, and
`--list` writes the MAWs themselves in MAW's format.

## Build

Requires macOS on Apple Silicon and the Command Line Tools. Xcode is not needed, because the Metal
shaders are compiled when the program starts.

```sh
scripts/fetch_deps.sh   # clones libsais and libdeflate, builds an arm64 OpenMP runtime (static) into third_party/
make                    # builds build/mawmetal, plus the test tools build/setdigest and build/gen_genome
```

libdeflate is optional: without it, gzip input goes through zlib. `make CPU=native` tunes for the build
machine instead of the M1.

## Usage

```sh
build/mawmetal -i genome.fa[.gz] [-o out.txt] [-k MIN] [-K MAX] [-r 1] [--joint] [--list] [-v]
```

* **Default output** is the length distribution: for each FASTA record, `>id` followed by
  `length<TAB>count` lines and an empty line.
* `--list` writes the MAWs themselves instead, one block per record: `>id`, one MAW per line, then an
  empty line. Within a block the MAWs appear in a different order than MAW's, so compare the two as sets.
* `-k`, `-K` and `-r 1` mean the same as in MAW (`-a DNA` is accepted too). `-r 1` adds the reverse
  complement of each sequence, like the `-rc` switch of Pinho et al.'s program.
* `--joint` computes the MAWs of all records taken together: words absent from every record whose
  proper factors all occur somewhere. Use it for an assembly split into many scaffolds. MAW has no
  equivalent.
* `-v` prints a per-stage timing breakdown on stderr.
* **Size limit:** a run's input can be up to about 68 Gbp, including the reverse complement with
  `-r 1`.

Differences from MAW:

* **N and other IUPAC codes.** MAW treats `N` as a fifth letter and also reports MAWs that contain `N`.
  mawmetal treats every non-ACGT character as a break: no reported word spans one. Restricted to words
  over ACGT, the two outputs are identical (verified on chr21, which has long N runs).

## How it works

For a MAW `aWb` (where `a` and `b` are single bases), `aW` and `Wb` occur but `aWb` does not. The middle
`W` therefore occurs at least twice, and the work splits by the length of `W`:

| stage | handles | where | how |
|---|---|---|---|
| census | `\|W\| ≤ 6` | GPU | one pass records which words of length ≤ 8 occur |
| tier 1 | `7 ≤ \|W\| < 31` | GPU sort; GPU or CPU scan | sorted 64-bit suffix keys: a segmented OR scan (GPU) or a bottom-up LCP-interval scan (CPU) |
| deep | `31 ≤ \|W\| < 1024` | CPU | groups of suffixes sharing 31 bases, sorted further from the packed genome |
| tail | `\|W\| ≥ 1024` | CPU | libsais (multithreaded) on the regions covered by repeated 1024-mers only |

**Input.** A plain FASTA file is mapped and packed in place, 2 bits per base, by all cores. Each core
packs a piece of the file at an offset found by a first counting pass. Blocks of 16 bases that hold no
line break or separator are packed with NEON in one step. With `--joint` the pieces span records, whose
header lines only separate segments, so an assembly of millions of short scaffolds is packed on all
cores too. gzip input is decompressed with libdeflate: BGZF files (bgzip) block by block on all cores,
other files in one call, or through zlib on a separate thread for pipes and multi-member files. The
input's memory is given back as its records are packed. The reverse complement is made from the packed
bases, a word at a time: after each record, or with `--joint` once for the whole text, after all of
them. The Metal device and shaders are set up on another thread while the input is read.

**Census.** One GPU pass over the text counts the positions per 4-base prefix and records which words of
length ≤ 8 occur. That presence is enough for every MAW with `|W| ≤ 6`. Tier 1 then never reports a node
shallower than 7 bases, so its chunks can be cut wherever neighbouring suffixes share fewer than 7 bases,
which happens every few suffixes. That keeps all cores busy.

**Tier 1.** Every position becomes one 64-bit key: the 28 bases after its first four (zeroed past the
segment end), its valid length, and the preceding base. Positions are grouped into 256 regions by their
first four bases, batched, and sorted on the GPU by a segmented LSD radix sort. On texts of a gigabase
or more the regions are made of five bases instead (1024 regions): the sort scatters within a region,
and 4-base regions there outgrow the GPU's caches (the sort of GRCh38 on one strand takes 4.1 s instead
of 7.8 s). A region is never split across batches; when the largest one would not fit a batch, the
regions are also made of five bases, and the census is run again for them.

* **Passes.** The sort covers only the bits the run needs: 6 passes of 9 bits by default, and fewer
  with a small `-K` (3 for `-K 16`).
* **Extract.** Each batch picks its positions out of a pass over the whole text. The pass reads the
  segment bitmap once per 16 positions, and a single test usually settles all their valid lengths.
* **Each pass** ranks a 2048-key tile per simdgroup with `simd_ballot`, reorders the tile in threadgroup
  memory, and then writes each digit's run contiguously.
* **Short suffixes.** The few suffixes cut short by a sequence end are first ordered by length in a
  small pre-pass, so the stable sort places them as if a sentinel ended them.
* **Pipeline.** Three batch slots let the GPU sort the next batches while the CPU threads scan an
  earlier one.
  * The threads are persistent, and move from one batch's chunks to the next without waiting.
  * A slot is refilled as soon as every chunk of its batch is done.
  * About 16 batches; the first ones are a quarter of the size and doubling, so the CPU starts early.
    With `--list`, about 64 smaller ones: its CPU-bound pipeline leaves the GPU time for the extra passes
    over the text, and the batch buffers shrink by as much.
* **The CPU scan** is the bottom-up LCP-interval traversal of Barton et al., run directly over the keys.
  The keys carry the left base, so the genome is never touched. With `--list` the nodes with MAWs are
  recorded during the scan and spelled every 8192 suffixes, which keeps the loop free of calls.
* **List output.** With `--list` the chunks' outputs are written in order.
  * The chunk next in line writes its own output as it goes, so a long chunk is never held whole (a
    frequent 7-mer such as poly-A cannot be cut).
  * Workers run ahead of it only while the finished chunks waiting to be written hold less than one
    batch slot's worth (at least 128 MB). Chunks that take long, large repeat groups, hold back the
    ones after them, the longer on larger genomes: on GRCh38 with both strands this takes the list
    from 60 s (with a fixed 128 MB) to 49 s, for the same peak memory.
    The chunks are smaller than for the distribution (64K suffixes instead of 256K), and a buffer that
    grew large is freed after writing instead of being reused.
  * On chr1 the list mode peaks at about 0.61 GB, below the distribution (0.97 GB, larger batches).
* **The GPU scan** (counts only) uses the fact that, for each depth d from 7 to 30, the suffixes sharing
  their first d bases form a run of the sorted keys.
  * A run's matrix of (preceding base, base at depth d) pairs is the OR of one bit per suffix, and
    its MAWs follow from that matrix.
  * One segmented OR scan over the keys finds the runs of all 24 depths at once. Runs that do not
    branch at depth d have no MAWs, so this gives the CPU traversal's result.
  * For each batch, the tier-1 scan runs on whichever side is less busy at submission: the GPU
    while the CPU has a backlog, the CPU otherwise.

**Deep stage.** Every occurrence of a `W` with 31 or more bases starts with the same 31 bases, so all of
them lie in one group of tier 1's sorted output, inside a single batch.

1. Each group (sorted only to 31 bases) is sorted further, 32 bases per step, reading the packed genome
   directly down to depth 1024. That gives exact LCPs.
   * Those reads are prefetched well ahead, across groups (they average about 5 members): each
     member's text down to depth 1024 (two or three 128-byte lines), and its coarse segment bits.
   * Inside long exact repeats, where a whole group keeps agreeing, one pass compares up to 1024
     further bases of each member at once instead of going 32 bases at a time: four 32-base windows
     per branch, after a single lookup of where the member's segment ends.
   * A coarse bitmap of segment starts (1 bit per 64 bases) answers most "where does this segment
     end" questions without touching the full one. It stays in cache for a human genome (12 MB with
     both strands) but not for the spruce (47 MB), hence its prefetch.
   * Groups are sorted by the next 32 bases: pairs, triples and quadruples by compare-exchanges
     without branches, up to 16 members by insertion, and from 256 by a radix sort that skips the
     bytes all members share.
   * A group of two (60% of the groups on GRCh38) needs none of this: the text gives its LCP
     directly, and with it the group's one node and its marks.
2. The same scan then emits the MAWs with `31 ≤ |W| < 1024`. The bases it needs were saved while
   sorting, so it reads nothing from the genome.
3. Positions still tied at depth 1024 start a repeated 1024-mer. They are marked, and only the regions
   they cover are passed to libsais, which handles the rest.

Refining all the way down would be quadratic inside long exact repeats, where libsais is linear. On
human chr1 the tail's input is 0.5% of the bases, against 21% when every repeated 31-mer went to libsais.
With the reverse complement it is 1.2% for chr1, 7.1% for GRCh38.p2 (whose ALT and patch records nearly
copy parts of the primary assembly) and 5.4% for the spruce.

**Tail.** libsais builds the generalized suffix array of the marked runs and their permuted LCP array
(PLCP). The scan reads each LCP as `PLCP[SA[i]]`, prefetched, which saves the LCP array: 9 bytes per
base in all. When the runs are large, the batch buffers and the packed text are freed first. libsais
runs on about 3/4 of the performance cores (6 on an M1 Max): an efficiency core would hold up each of its
many short parallel steps. Its OpenMP threads spin for 1 ms after each step instead of sleeping at once
(the default of this libomp), since waking them costs more than most steps take. Together this makes
libsais and the tail scan 1.4× faster on GRCh38 (9.3 → 6.6 s) and 1.6× on chr21 with `-r 1`.

**Why the split is exact.** This holds for any repeat length k. If `aWb` is a MAW with `|W| ≥ k`, then
`W` occurs at least twice. Every k-mer of `W` is therefore repeated, and every occurrence of `W` lies
inside a marked run; the one-base extension of the runs keeps `a` and `b`. Conversely, suppose `aW` and
`Wb` occur in the runs but `aWb` does not. Then `aWb` cannot occur in the genome either, because such an
occurrence would contain a repeated `W` and would therefore lie in a run. Runs from different segments
are never merged, so the runs create no new words.

**Positions** are 32-bit. For inputs of 2^32 bases or more, a byte holding bits 32–39 is added in a
separate array, so each position costs 5 bytes instead of 8.

## Validation

* `tools/run_tests.py` runs a randomized differential test against a brute-force reference
  (`tools/brute_maw.py`). The inputs mix planted and mutated repeats, homopolymers, N runs, lowercase,
  and multiple and tiny records. The runs also randomize:
  * the reverse complement, `--joint` and the length bounds;
  * `--prefix` (or its automatic choice), `--split` and `--deep`;
  * the output mode: the list, or the length distribution checked against the list's lengths;
  * where tier 1 runs (`--cpu-tier1`, `--gpu-tier1`, or chosen per batch);
  * tiny `--batch`, `--first`, `--chunk` and `--piece` values, so that small inputs are still cut into
    many GPU batches, scan chunks and packing pieces;
  * tiny `--spill` and `--pending` values, so that small inputs still stream and hold back the list
    output;
  * the 40-bit position path (`--wide`) and the 64-bit libsais path (`MAWMETAL_SAIS64=1`).

  The last full runs passed 3 × 300 cases.
* Against MAW (built by `scripts/fetch_deps.sh --reference`):
  * *E. coli* K-12: identical, 7,973,322 MAWs.
  * Human chr21: identical on ACGT words, 58,101,703 MAWs.
* Self-consistency on chr21 (the 40-bit position path, chunking, batching, `--split`, `-r 1`) and on
  chr1 (341,222,095 MAWs).
  * Large outputs are compared with `build/setdigest`, an order-independent hash of the word multiset.
  * The current version reproduces the previous version's outputs exactly on all of these inputs:
    *E. coli* (also with CRLF line ends and `-r 1`), chr21 (also with `-r 1` and `-k 5 -K 40`), chr1,
    and a multi-record file (per record, and `--joint -r 1`).
  * The same holds for every input path: plain, gzip, BGZF, and a pipe.
* `MAWMETAL_CHECK_SORT=1` verifies every sorted GPU batch against the text.
* **Not yet run:** `scripts/test_large.sh`. It checks inputs above 2^32 bases by streaming a synthetic
  2.2 Gbp genome twice, so the second copy sits above 2^32. Duplicating a record adds no new words, so
  both runs must report the same MAWs. It needs about 8 GB of free memory, which the development machine
  lacked. On small inputs the 40-bit path passes all tests, but its high byte is always 0 there, so only
  this test can catch an error in that byte.

## Reproducing the benchmark

Neither the genomes nor the other programs are distributed with this repository, only where to get them:
`bench/genomes.tsv` lists each genome (URL, size, sha256, expected base counts), and `bench/programs.tsv`
lists the other programs' code (repository and pinned commit, or tarball and sha256, with licenses).

* `bench/run.sh --download-only` rebuilds `bench/tools/` and `bench/data/` from these two files. It
  builds every program, downloads the genomes (resuming broken transfers and checking sha256), and stops.
  `--genomes chr21,chr1` limits it to some genomes.
* `bench/run.sh` then times every program on every genome; `--help` lists the options.
* **Pinho et al. (2009) is the exception.** Its original site is offline, so it cannot be downloaded.
  **Armando J. Pinho has kindly agreed that the original 2009 code be shared as part of this
  project**, so that the comparison with the software of our 2015 study can be reproduced. We are very
  grateful to him. The original archive (`maws-2009-original.tar.gz`, sha256 in `bench/programs.tsv`)
  is in `bench/vendor/`, and its unpacked sources are in `reference/maws_pinho2009/`; the benchmark
  builds it like the others, with a single fix for current zlib
  (`gzFile *fp` to `gzFile fp`).

## Performance

Measured on an M1 Max (10 CPU cores, 32-core GPU, 32 GB), MAWs of all lengths, output discarded. MAW is
the 32-bit build. The machine was under heavy memory pressure and another benchmark kept one core busy,
so the times are approximate (medians of 5 runs).

| input | ACGT bases | MAWs | MAW | mawmetal `--list` | mawmetal (distribution) |
|---|---:|---:|---:|---:|---:|
| *E. coli* K-12 | 4.6 M | 7,973,322 | 2.15 s | 0.12 s | 0.11 s (20×) |
| human chr21 | 40.1 M | 58,101,703 | 25.2 s | 0.37 s | 0.32 s (78×) |
| human chr1 | 230.5 M | 341,222,095 | not measured | 1.19 s | 0.73 s |

MAW on chr1 could not be timed meaningfully under the memory pressure. Against the previous version of
mawmetal, measured the same way:

| chr1 | previous | now |
|---|---:|---:|
| distribution | 2.18 s | 0.73 s |
| `--list` | 3.93 s | 1.19 s |
| `-r 1` (both strands) | 4.58 s | 1.42 s |
| gzip input (one member) | 2.49 s | 0.9–1.1 s |
| BGZF input | 2.7 s | 0.79 s |
| peak memory | 2.9 GB | 1.0 GB |

For chr1 in distribution mode (`-v`), the stages are:

| stage | previous | now |
|---|---:|---:|
| reading and packing | 0.8–1.0 s (one thread) | 0.06 s |
| waiting for the first sorted batch | 0.13–0.47 s | 0.03 s |
| batch pipeline: GPU (extract 0.13 s, sort 0.28 s, part of tier 1 0.09 s) and CPU (tier 1 + deep stage, 3.4 core-seconds) overlapping | 0.96–1.6 s, cores half idle | 0.54–0.62 s, cores 80–85% busy |
| residue, libsais, tail scan | 0.05 s | 0.05 s |
| total | 2.2 s | 0.7 s |

## A 20 Gb conifer

The Norway spruce v1.0 assembly: 12.0 Gbp of ACGT in 10,253,694 scaffolds, so 24.0 Gbp with the reverse
complement, run as `--joint -r 1` on the uncompressed file. On the M1 Max (32 GB, on AC power; runs on
battery took up to 18% longer): 201 s in all, 21,213,697,531 MAWs.

| part | measured |
|---|---|
| reading and packing | 8.4 s for the 12.7 GB file, packed on all cores across the scaffolds (12–18 s when it was packed one scaffold at a time) |
| census | 1.6 s, once: the regions are made of 5 bases from the start, as for any text of a gigabase or more (the largest 4-base regions, AAAA and TTTT, hold 308 M positions each, twice a batch) |
| tier 1 + deep stage | 155 s, 131 batches. The CPU is the limit, the deep stage alone taking 1,358 core-seconds (44% of the positions are in groups of repeated 31-mers, against 14% for chr1); the GPU is busy for about 87 s of it (extract 20 s, sort 55 s, tier 1 12 s) |
| tail | 1.30 G bases (5.4%): residue 3.5 s, libsais 17.4 s, scan 5.6 s |
| memory | peak footprint 26.3 GB (at most 16.4 GB resident; macOS compressed the rest), the same as for a run stopped before the tail. Known parts: text 6.0 GB, segment bitmap 3.0 GB, repeat marks up to 3 GB, three batch slots and the shared temporaries 8.5 GB. The tail frees the batch buffers and the text first, then needs about 12 GB |
| disk | the input only; the distribution output is tiny, while `--list` would write 498 GB (mean MAW length 22.5) |

No other program fits this genome in memory, so the output is checked only for consistency: the counts
for lengths below 1026 equal those of a run stopped before the tail.

## Limitations and next steps

* **GPU and CPU are now roughly balanced** on human chromosomes (tier 1 moves between them). Going
  further needs less work on both sides. On the whole human genome (GRCh38.p2, both strands) the batch
  pipeline now waits on the GPU (23 s, of which about 21 s of GPU work); on the spruce the CPU's deep
  stage is the limit, with the GPU idle nearly half the time.
  * On the GPU: the radix sort is about 19 ms per 16 M keys, the extract about 3 ms, the tier-1 scan
    about 10 ms.
  * On the CPU: the deep stage takes about 2.0 core-seconds on chr1.
* **Each batch rescans the whole text** to pick out its positions. On the spruce (131 batches) the
  rescans take 20 s, against 56 s for the sort.
* **Region prefix length** is 4 bases, or 5 on texts of a gigabase or more and when the largest region
  would not fit a batch.
  The census does not fit in threadgroup memory beyond 5 bases. On a 32 GB Mac, an input about 1.5 times
  the spruce's size would again have a region larger than a batch, and every batch slot would grow with it.
* **Per-record mode over millions of small scaffolds** pays the GPU setup for each record. `--joint`
  avoids that; a CPU-only path for small records would remove it.

## Acknowledgements

We thank Armando J. Pinho for permission to share the 2009 software of Pinho, Ferreira, Garcia and
Rodrigues (*BMC Bioinformatics* 10:137, 2009) with this project. It made our 2015 study possible, and it
is the reference against which ten years of progress are measured here.

## License

mawmetal is released under the MIT license (see `LICENSE`). The code it fetches and builds
(libsais, libdeflate, the OpenMP runtime, MAW and the other benchmarked programs) keeps its own
license; `bench/programs.tsv` lists those of the benchmarked programs.
