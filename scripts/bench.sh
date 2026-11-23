#!/bin/bash
# Copyright (c) 2026 Nicolas Innocenti
# SPDX-License-Identifier: MIT
# Wall-clock comparison of MAW (single-threaded reference) and mawmetal, all MAW lengths, output
# discarded so disk speed does not matter. mawmetal is timed writing the list (--list) and in its
# default mode (length distribution only). Runs under caffeinate so idle sleep cannot stretch it.
# usage: scripts/bench.sh file.fa [file.fa ...]
set -euo pipefail
cd "$(dirname "$0")/.."
MAW=third_party/maw_build/maw32
wall() { /usr/bin/time -p "$@" 2>&1 >/dev/null | awk '/^real/ {print $2}'; }
best() { local b=""; for _ in 1 2 3; do t=$(wall "$@"); b=$(echo "$t $b" | tr ' ' '\n' | grep . | sort -g | head -1); done; echo "$b"; }
printf "%-28s %10s %16s %14s\n" input "MAW (s)" "mawmetal --list" "mawmetal (s)"
for f in "$@"; do
    tm=$(wall caffeinate -i $MAW -a DNA -i "$f" -o /dev/null -k 2 -K 1000000000 || echo "-")
    tl=$(best caffeinate -i ./build/mawmetal -i "$f" --list -o /dev/null)
    tc=$(best caffeinate -i ./build/mawmetal -i "$f" -o /dev/null)
    printf "%-28s %10s %16s %14s\n" "$(basename "$f")" "$tm" "$tl" "$tc"
done
