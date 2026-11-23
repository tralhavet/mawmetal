#!/usr/bin/env python3
# Copyright (c) 2026 Nicolas Innocenti
# SPDX-License-Identifier: MIT
"""Brute-force minimal absent words, for validating mawmetal on small inputs.

Semantics match mawmetal: non-ACGT characters split the sequence (no word spanning them is
present), records are processed separately unless --joint, -r 1 adds the reverse complement,
and only MAWs of length >= 2 are reported. Output is MAW-style: '>id', one MAW per line, blank line.
"""
import argparse
import re
import sys

COMP = str.maketrans("ACGT", "TGCA")


def read_fasta(path):
    recs, rid, seq = [], None, []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\r\n")
            if line.startswith(">"):
                if rid is not None:
                    recs.append((rid, "".join(seq)))
                rid, seq = line[1:], []
            elif rid is not None:
                seq.append(line)
    if rid is not None:
        recs.append((rid, "".join(seq)))
    return recs


def segments(seq, revcomp):
    segs = [s for s in re.split(r"[^ACGT]+", seq.upper()) if s]
    if revcomp:
        segs += [s.translate(COMP)[::-1] for s in segs]
    return segs


def maws(segs, kmin, kmax):
    subs = set()
    for s in segs:
        for i in range(len(s)):
            for j in range(i + 1, len(s) + 1):
                subs.add(s[i:j])
    letters = [c for c in "ACGT" if c in subs]
    out = set()
    for w in list(subs) + [""]:
        for a in letters:
            if a + w not in subs:
                continue
            for b in letters:
                x = a + w + b
                if w + b in subs and x not in subs and max(kmin, 2) <= len(x) <= kmax:
                    out.add(x)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", required=True)
    ap.add_argument("-k", type=int, default=2)
    ap.add_argument("-K", type=int, default=10**9)
    ap.add_argument("-r", type=int, default=0)
    ap.add_argument("--joint", action="store_true")
    a = ap.parse_args()
    recs = [(rid, seq) for rid, seq in read_fasta(a.i) if seq]
    if a.joint:
        segs = [s for _, seq in recs for s in segments(seq, a.r)]
        recs_out = [(f"joint ({len(recs)} records)", segs)]
    else:
        recs_out = [(rid, segments(seq, a.r)) for rid, seq in recs]
    w = sys.stdout.write
    for rid, segs in recs_out:
        w(f">{rid}\n")
        for x in sorted(maws(segs, a.k, a.K)):
            w(x + "\n")
        w("\n")


if __name__ == "__main__":
    main()
