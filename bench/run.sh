#!/bin/bash
# Copyright (c) 2026 Nicolas Innocenti
# SPDX-License-Identifier: MIT
# The MAW benchmark in one command: builds the programs (the first time only, about 5 minutes on the
# M1 Max), downloads the genomes it needs, and times every program on every genome. Each invocation
# writes one file, results/<machine>-<YYYYmmdd-HHMMSS>.json.
#
#   bench/run.sh                     profile "full": everything, for the M1 Max (32 GB)
#   bench/run.sh --profile mbp2012   genomes up to chr1, earlier programs only, for a 16 GB Intel Mac
#                                    (the MacBook Pro Retina 2012)
#   bench/run.sh --plan              print what would run and what would be skipped, run nothing
#   bench/run.sh --download-only     build the programs and download the genomes, run nothing: this
#                                    rebuilds tools/ and data/ from programs.tsv and genomes.tsv
#   bench/run.sh --check             count the MAWs of each length instead of timing (validation)
#   bench/run.sh --help              all options: genomes, programs, repetitions, machine name, ...
set -euo pipefail
cd "$(dirname "$0")"
case " $* " in
    *" --help "* | *" -h "*) ;;
    *) ./build_tools.sh ;;
esac
exec python3 -I run_bench.py "$@"
