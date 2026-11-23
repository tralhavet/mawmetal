#!/usr/bin/env python3
# Copyright (c) 2026 Nicolas Innocenti
# SPDX-License-Identifier: MIT
"""Compare two MAW-format outputs ('>id', one word per line, blank line between records) as sets.

Exits non-zero on any difference or on duplicate words. With --acgt-only, words containing
anything other than ACGT are dropped from both sides first (MAW treats N as a letter). With
--counts, ours is mawmetal's default output ('>id', then 'length<TAB>count' lines) and is checked
against the reference's word lengths.
"""
import argparse
import sys


def load(path, acgt_only):
    recs, cur, dups = [], None, 0
    ok = set("ACGT")
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(">"):
                cur = (line[1:], set())
                recs.append(cur)
            elif line:
                if cur is None:
                    sys.exit(f"{path}: word before the first '>' header")
                if acgt_only and not set(line) <= ok:
                    continue
                if line in cur[1]:
                    dups += 1
                cur[1].add(line)
    return recs, dups


def load_counts(path):
    recs, cur = [], None
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(">"):
                cur = (line[1:], {})
                recs.append(cur)
            elif line:
                length, count = line.split("\t")
                cur[1][int(length)] = int(count)
    return recs


def histogram(words):
    h = {}
    for w in words:
        h[len(w)] = h.get(len(w), 0) + 1
    return h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ours")
    ap.add_argument("reference")
    ap.add_argument("--acgt-only", action="store_true")
    ap.add_argument("-q", action="store_true", help="only print on failure")
    ap.add_argument("--counts", action="store_true", help="ours is a length distribution")
    a = ap.parse_args()
    if a.counts:
        ours = load_counts(a.ours)
        ref, _ = load(a.reference, a.acgt_only)
        bad = [r[0] for r in ours] != [r[0] for r in ref]
        if bad:
            print(f"record headers differ: {[r[0] for r in ours][:5]} vs {[r[0] for r in ref][:5]}")
        for (rid, h1), (_, words) in zip(ours, ref):
            h2 = histogram(words)
            if h1 != h2:
                bad = True
                print(f"{rid}: counts {sorted(h1.items())} vs {sorted(h2.items())}")
        if not bad and not a.q:
            print(f"identical distributions: {len(ours)} record(s)")
        sys.exit(1 if bad else 0)
    ours, d1 = load(a.ours, False)
    ref, _ = load(a.reference, a.acgt_only)
    bad = d1 > 0
    if d1:
        print(f"duplicates in {a.ours}: {d1}")
    if [r[0] for r in ours] != [r[0] for r in ref]:
        print(f"record headers differ: {[r[0] for r in ours][:5]} vs {[r[0] for r in ref][:5]}")
        bad = True
    total = 0
    for (rid, s1), (_, s2) in zip(ours, ref):
        total += len(s2)
        if s1 != s2:
            bad = True
            extra, missing = sorted(s1 - s2, key=len), sorted(s2 - s1, key=len)
            print(f"{rid}: {len(extra)} extra, {len(missing)} missing (of {len(s2)})")
            print("  extra:", extra[:8])
            print("  missing:", missing[:8])
    if not bad and not a.q:
        print(f"identical: {len(ours)} record(s), {total} MAWs")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
