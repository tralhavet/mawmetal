#!/bin/bash
# Copyright (c) 2026 Nicolas Innocenti
# SPDX-License-Identifier: MIT
# Checks texts above 2^32 bases (positions with a high byte) without using any disk: a synthetic
# genome R of LENGTH bases is streamed twice as two records, so the joint text passes 2^32 while
# its MAWs must equal those of R alone. Needs roughly 8 GB of free memory for the default length.
# usage: scripts/test_large.sh [LENGTH] [SEED]
set -euo pipefail
cd "$(dirname "$0")/.."
L=${1:-2200000000}
SEED=${2:-11}
one=$(./build/gen_genome "$L" "$SEED" 1 | caffeinate -i ./build/mawmetal -i /dev/stdin --joint --batch 30000000 -v | tail -n +2 | md5)
two=$(./build/gen_genome "$L" "$SEED" 2 | caffeinate -i ./build/mawmetal -i /dev/stdin --joint --batch 30000000 -v | tail -n +2 | md5)
echo "one copy: $one"
echo "two copies: $two"
[ "$one" = "$two" ] && echo "OK: identical length distributions" || { echo "MISMATCH"; exit 1; }
