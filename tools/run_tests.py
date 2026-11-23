#!/usr/bin/env python3
# Copyright (c) 2026 Nicolas Innocenti
# SPDX-License-Identifier: MIT
"""Randomized differential test: mawmetal vs brute_maw.py on small adversarial inputs.

Inputs mix random DNA with planted (sometimes mutated) repeats, homopolymer runs, N runs,
lowercase, and tiny records. Options are randomized too, notably small --prefix/--split values
so the census, tier-1 and tier-3 code paths and their boundaries all get exercised, tiny
--chunk/--first/--batch/--piece values so the inputs are cut into many chunks, GPU batches and
packing pieces, and tiny --spill/--pending values so the list output is streamed and held back.

usage: run_tests.py [-n CASES] [--seed S] [--bin build/mawmetal] [--keep DIR]
"""
import argparse
import os
import random
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))


def random_seq(rng, n):
    alphabet = rng.choice(["ACGT", "ACGT", "ACGT", "AC", "AT", "A", "ACG"])
    s = [rng.choice(alphabet) for _ in range(n)]
    for _ in range(rng.randint(0, 3)):  # planted repeats
        if len(s) < 4:
            break
        ln = rng.randint(2, max(2, len(s) // 2))
        a = rng.randrange(0, max(1, len(s) - ln))
        piece = s[a:a + ln]
        if rng.random() < 0.4 and piece:
            piece[rng.randrange(len(piece))] = rng.choice("ACGT")
        b = rng.randrange(0, len(s) + 1)
        s[b:b] = piece
    for _ in range(rng.randint(0, 2)):  # homopolymers
        b = rng.randrange(0, len(s) + 1)
        s[b:b] = [rng.choice("ACGT")] * rng.randint(3, 40)
    for _ in range(rng.choice([0, 0, 1, 2, 4])):  # N runs / IUPAC
        b = rng.randrange(0, len(s) + 1)
        s[b:b] = [rng.choice("NNNRY")] * rng.randint(1, 5)
    out = "".join(s)
    if rng.random() < 0.2:
        out = "".join(c.lower() if rng.random() < 0.3 else c for c in out)
    return out[:400]


def write_fasta(path, recs, rng):
    with open(path, "w") as f:
        for rid, seq in recs:
            f.write(f">{rid}\n")
            width = rng.choice([10, 60, 1000])
            for i in range(0, len(seq), width):
                f.write(seq[i:i + width] + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--bin", default=os.path.join(HERE, "..", "build", "mawmetal"))
    ap.add_argument("--keep", help="directory for failing cases")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    tmp = tempfile.mkdtemp(prefix="mawtest")
    fails = 0
    for case in range(a.n):
        nrec = rng.choice([1, 1, 1, 2, 3])
        recs = [(f"r{case}_{i}", random_seq(rng, rng.choice([1, 2, 5, 20, 80, 200, 300])))
                for i in range(nrec)]
        fa = os.path.join(tmp, f"c{case}.fa")
        write_fasta(fa, recs, rng)
        auto = rng.random() < 0.25  # no --prefix: 4, or 5 when the largest region exceeds the batch
        m = 4 if auto else rng.randint(1, 5)
        lo = 6 if auto else m + 1  # --split must then suit both 4 and 5
        split = rng.choice([lo, lo + 1, lo + 2, rng.randint(lo, m + 28), min(31, m + 28)])
        kmin = rng.choice([2, 2, 3, 5, 9])
        kmax = rng.choice([10**6, 10**6, 4, 8, 12, 40])
        rc = rng.random() < 0.3
        joint = rng.random() < 0.3
        deep = rng.choice([split, split + 1, split + 3, split + 33, split + 70, 1024])
        counts = rng.random() < 0.4  # the default output: tier 1 may then run on the GPU
        args = [a.bin] + ([] if counts else ["--list"]) + ["-i", fa, "-k", str(kmin), "-K", str(kmax)] + \
               ([] if auto else ["--prefix", str(m)]) + ["--split", str(split), "--deep", str(deep)]
        if rng.random() < (0.6 if auto else 0.3):
            args += ["--batch", str(rng.choice([1, 16, 100]))]
        if rng.random() < 0.3:
            args += ["--first", str(rng.choice([1, 8, 50]))]
        if rng.random() < 0.6:
            args += ["--chunk", str(rng.choice([1, 2, 5, 30]))]
        if rng.random() < 0.5:
            args += ["--piece", str(rng.choice([1, 2, 7, 33]))]
        if rng.random() < 0.3:
            args += ["--wide"]
        if counts and rng.random() < 0.6:
            args += [rng.choice(["--cpu-tier1", "--gpu-tier1"])]
        if not counts and rng.random() < 0.5:
            args += ["--spill", str(rng.choice([1, 7, 100]))]
        if not counts and rng.random() < 0.3:
            args += ["--pending", str(rng.choice([1, 1000]))]
        if rc:
            args += ["-r", "1"]
        if joint:
            args += ["--joint"]
        ours = os.path.join(tmp, f"c{case}.ours")
        ref = os.path.join(tmp, f"c{case}.ref")
        env = dict(os.environ, MAWMETAL_SAIS64="1") if rng.random() < 0.2 else None
        with open(ours, "w") as f:
            p = subprocess.run(args, stdout=f, stderr=subprocess.PIPE, text=True, env=env)
        if p.returncode != 0:
            print(f"case {case}: mawmetal failed: {p.stderr.strip()}\n  {' '.join(args)}")
            fails += 1
            continue
        bargs = ["-i", fa, "-k", str(kmin), "-K", str(kmax), "-r", str(int(rc))] + (["--joint"] if joint else [])
        with open(ref, "w") as f:
            subprocess.run([sys.executable, os.path.join(HERE, "brute_maw.py")] + bargs, stdout=f, check=True)
        c = subprocess.run([sys.executable, os.path.join(HERE, "compare_maw.py"), "-q", ours, ref] + (["--counts"] if counts else []),
                           capture_output=True, text=True)
        if c.returncode != 0:
            fails += 1
            print(f"case {case}: MISMATCH  {' '.join(args)}\n{c.stdout}")
            if a.keep:
                os.makedirs(a.keep, exist_ok=True)
                for pth in (fa, ours, ref):
                    os.replace(pth, os.path.join(a.keep, os.path.basename(pth)))
    print(f"{a.n - fails}/{a.n} cases passed")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
