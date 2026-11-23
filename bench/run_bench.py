#!/usr/bin/env python3
# Copyright (c) 2026 Nicolas Innocenti
# SPDX-License-Identifier: MIT
"""Times every MAW program on every genome. Each invocation writes one file,
results/<machine>-<YYYYmmdd-HHMMSS>.json, rewritten after every run so an interrupted benchmark keeps
what it measured.

All programs solve the same problem: the minimal absent words of all lengths of a genome together
with its reverse complement, all records taken jointly, as in our 2015 study.
  * Pinho et al. (2009), bwtman and mawmetal do that natively; all three cut the sequence at every
    letter other than ACGT.
  * MAW, pMAW and emMAW compute each FASTA record separately and treat N as a fifth letter, so they
    are given the genome as one record in which every record boundary and every run of other letters
    is a single N (--keep-n-runs: runs kept, as in the first benchmarks). Their words containing N
    are not counted.
Output is discarded in each program's cheapest way: Pinho et al., bwtman and mawmetal report counts
per length (the "-list" variants write the words to /dev/null); MAW writes to /dev/null; emMAW, which
derives temporary file names from its output name, writes into a pipe drained by cat; pMAW, which
writes one temporary file per thread, is built without its output stage ("pmaw"), and with it
("pmaw-out") for --check. bwtman builds its BWT index and then enumerates; both steps are timed.

A program/genome pair is skipped, and recorded as skipped, when the program is not built for this
machine or its estimated memory or disk exceeds what the machine has (--force runs it anyway). A run
is killed if swap grows by more than 24 GB or it exceeds --timeout.

With --check, each run instead records the number of MAWs of each length, which must agree across
programs; timings of check runs are not used.

usage: run_bench.py [--profile full|mbp2012] [--genomes a,b] [--tools a,b] [--reps N] [--check]
                    [--machine NAME] [--tmp DIR] [--timeout S] [--force] [--plan] [--download-only]
"""
import argparse
import datetime
import gzip
import hashlib
import json
import os
import platform
import re
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
# The machine's own architecture, also when this Python is an x86_64 build running under Rosetta.
ARCH = "arm64" if subprocess.run(["sysctl", "-n", "sysctl.proc_translated"], capture_output=True,
                                 text=True).stdout.strip() == "1" else platform.machine()
TOOLS = os.path.join(HERE, "tools", ARCH)
BIN = os.path.join(TOOLS, "bin")
DATA = os.path.join(HERE, "data")
RESULTS = os.path.join(HERE, "results")
GB = 1e9

PINHO = os.path.join(BIN, "pinho")
MAW32, MAW64 = os.path.join(BIN, "maw32"), os.path.join(BIN, "maw64")
PMAW, PMAW_OUT = os.path.join(BIN, "pmaw32-noout"), os.path.join(BIN, "pmaw32")
EMMAW = os.path.join(TOOLS, "emmaw", "em-maw64")
BWT_INDEX, BWT_MAWS = os.path.join(BIN, "bwtman-index"), os.path.join(BIN, "bwtman-maws")
MAWMETAL = os.environ.get("MAWMETAL", os.path.join(BIN, "mawmetal"))

ALL_TOOLS = ["pinho", "pinho-list", "maw", "pmaw", "emmaw", "bwtman", "mawmetal", "mawmetal-list"]
PROFILES = {
    # Everything, for the M1 Max (32 GB). On the human genome and the spruce only the programs that
    # fit run; the rest are recorded as skipped.
    "full": ("hhv5,efaecalis,ecoli,yeast,chr21,chr1,grch38p2,spruce", ALL_TOOLS),
    # A MacBook Pro Retina 2012 (Intel, 16 GB): genomes up to human chromosome 1, the earlier programs
    # only (mawmetal needs Apple Silicon). Same commands as "full", so the two machines compare directly.
    "mbp2012": ("hhv5,efaecalis,ecoli,yeast,chr21,chr1", [t for t in ALL_TOOLS if not t.startswith("mawmetal")]),
}
CHECK_TOOLS = {"pmaw": "pmaw-out", "pinho-list": None}  # --check: pMAW with output; Pinho's list adds nothing

MAXLEN = "5000000"  # the bound used in 2015, far above the longest MAW
EMMAW_RAM_MB = 8192
# Memory estimates, in bytes per character of the indexed text (both strands), from runs on human
# chromosome 21 with a margin, plus a fixed amount in GiB (0.3 unless given). mawmetal's batch buffers
# stop growing with the input, so it gets a fixed part and its own estimate for the spruce (about 21 GB
# for 24.6 G characters, see ../README.md); the largest text each build can index.
MEMORY = {"pinho": 28, "pinho-list": 70, "maw": 18, "maw64": 36, "pmaw": 16, "pmaw-out": 16,
          "bwtman": 11, "mawmetal": 0.9, "mawmetal-list": 0.9}
MEMORY_FIXED = {"mawmetal": 2, "mawmetal-list": 2}
LIMIT = {"pmaw": 2**31 - 1, "pmaw-out": 2**31 - 1, "mawmetal": 68e9, "mawmetal-list": 68e9}
EMMAW_DISK = 20  # bytes of temporary disk per character
SWAP_GROWTH_GB = 24
# Counts MAWs per length from a word list on stdin (FASTA-style headers, blank lines, words with N skipped).
COUNT = "awk '/^>/ || $0 == \"\" || /N/ {next} {c[length($0)]++} END {for (l in c) print l, c[l]}' | sort -n"


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()


def sysctl(name):
    return sh(f"sysctl -n {name} 2>/dev/null")


# ---- genomes ----

def read_manifest():
    g = {}
    for line in open(os.path.join(HERE, "genomes.tsv")):
        if line.startswith("#") or not line.strip():
            continue
        name, file, size, sha, reps, records, length, acgt, url, desc = line.rstrip("\n").split("\t")
        g[name] = dict(name=name, file=file, bytes=int(size), sha256=sha, reps=int(reps), records=int(records),
                       length=int(length), acgt=int(acgt), url=url, description=desc)
    return g


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def free_gb(path):
    return shutil.disk_usage(path).free / GB


def download(g):
    d = os.path.join(DATA, g["name"])
    dst = os.path.join(d, g["file"])
    if os.path.exists(dst):
        return dst
    os.makedirs(d, exist_ok=True)
    part = dst + ".part"
    have = os.path.getsize(part) if os.path.exists(part) else 0
    if free_gb(d) < (g["bytes"] - have) / GB + 2:
        sys.exit(f"not enough disk to download {g['name']} ({g['bytes'] / GB:.1f} GB)")
    print(f"downloading {g['name']} ({g['bytes'] / 1e6:,.0f} MB) from {g['url']}", flush=True)
    stalled = 0
    while stalled < 3:  # resume after dropped connections; stop when attempts make no progress
        before = os.path.getsize(part) if os.path.exists(part) else 0
        if subprocess.run(["curl", "-sSfL", "--retry", "5", "--retry-delay", "5", "-C", "-", "-o", part, g["url"]]).returncode == 0:
            break
        stalled = stalled + 1 if os.path.getsize(part) == before else 0
        time.sleep(5)
    got = sha256_of(part)
    if got != g["sha256"]:
        sys.exit(f"{part}: sha256 {got}, expected {g['sha256']}; delete it to download again")
    os.replace(part, dst)
    return dst


def stats(g, src):
    """Records, sequence length, ACGT bases (checked against genomes.tsv) and sha256 of the decompressed
    text, which does not depend on how a mirror compressed it; cached next to the genome."""
    cache = os.path.join(DATA, g["name"], "stats.json")
    if os.path.exists(cache):
        s = json.load(open(cache))
        if s.get("file_sha256") == g["sha256"]:
            return s
    print(f"counting {g['name']}", flush=True)
    rec = length = acgt = 0
    h = hashlib.sha256()
    tail = b""
    with (gzip.open if src.endswith(".gz") else open)(src, "rb") as f:
        while True:
            buf = f.read(1 << 26)
            h.update(buf)
            if not buf:
                break
            buf = tail + buf
            cut = buf.rfind(b"\n") + 1
            buf, tail = buf[:cut], buf[cut:]
            seq, n = re.subn(rb"(?m)^>.*\n", b"", buf)
            rec += n
            seq = seq.replace(b"\n", b"").replace(b"\r", b"")
            length += len(seq)
            acgt += len(seq) - len(seq.translate(None, b"ACGTacgt"))
    if tail:
        rec += tail.startswith(b">")
        if not tail.startswith(b">"):
            seq = tail.strip()
            length += len(seq)
            acgt += len(seq) - len(seq.translate(None, b"ACGTacgt"))
    s = dict(records=rec, length=length, acgt=acgt, text_sha256=h.hexdigest(), file_sha256=g["sha256"])
    for k in ("records", "length", "acgt"):
        if s[k] != g[k]:
            sys.exit(f"{src}: {k} {s[k]:,}, genomes.tsv says {g[k]:,}")
    json.dump(s, open(cache, "w"))
    return s


def plain_input(g, src, length, required):
    """Uncompressed FASTA, made once from the download if the disk has room. Without room: the download
    itself (gzipped) when the program reads gzip, else None."""
    if not src.endswith(".gz"):
        return src
    dst = os.path.join(DATA, g["name"], g["name"] + ".fa")
    if os.path.exists(dst):
        return dst
    if free_gb(DATA) < length * 1.02 / GB + 5:
        return None if required else src
    print(f"decompressing {g['name']}", flush=True)
    subprocess.run(f"gzip -dc {src} > {dst}.part && mv {dst}.part {dst}", shell=True, check=True)
    return dst


def joined_input(g, src, s, keep_n_runs):
    """MAW, pMAW and emMAW, which take one record and treat N as a fifth letter: the genome as one
    uncompressed record in which every boundary between records, and every run of letters other than
    ACGT, is a single N; made once. The MAWs over ACGT are unchanged. Left in, the long N runs of human
    chromosomes make these programs spend most of their time writing words that contain N (MAW wrote
    41.7 GB of them for 12 Mb of chr21). With keep_n_runs, the runs are left as they are (the protocol
    of the first benchmarks)."""
    dst = os.path.join(DATA, g["name"], g["name"] + (".joined.fa" if keep_n_runs else ".maw.fa"))
    if os.path.exists(dst):
        return dst
    if free_gb(DATA) < s["length"] * 1.02 / GB + 5:
        return None
    print(f"preparing the one-record input of {g['name']}", flush=True)
    other = re.compile(rb"[^ACGTacgt]+")
    with (gzip.open if src.endswith(".gz") else open)(src, "rb") as f, open(dst + ".part", "wb") as out:
        out.write(b">joined\n")
        pending, last_n, first = bytearray(), True, True  # no N before the first base

        def flush(k):  # writes the first k pending characters, 80 per line
            out.write(b"".join(bytes(pending[i:i + 80]) + b"\n" for i in range(0, k, 80)))
            del pending[:k]

        for line in f:
            if line.startswith(b">"):
                seq, first = (b"" if first else b"N"), False
            else:
                seq = line.rstrip(b"\r\n")
            if not keep_n_runs:
                seq = other.sub(b"N", seq)
                if last_n and seq[:1] == b"N":
                    seq = seq[1:]
            if not seq:
                continue
            last_n = seq[-1:] == b"N"
            pending += seq
            if len(pending) >= 1 << 24:
                flush(len(pending) - len(pending) % 80)
        flush(len(pending))
    os.replace(dst + ".part", dst)
    return dst


# ---- programs ----

def plan(tool, s, ram_gb, tmp, force):
    """(binary, input kind, None) if the run fits this machine, else (None, None, reason)."""
    n = 2 * (s["length"] + s["records"])  # characters indexed: both strands
    kind = {"maw": "joined", "pmaw": "joined", "pmaw-out": "joined", "emmaw": "joined", "bwtman": "plain"}.get(tool, "fasta")
    binary = {"pinho": PINHO, "pinho-list": PINHO, "maw": MAW32, "pmaw": PMAW, "pmaw-out": PMAW_OUT, "emmaw": EMMAW,
              "bwtman": BWT_MAWS, "mawmetal": MAWMETAL, "mawmetal-list": MAWMETAL}[tool]
    mem_key = tool
    if tool == "maw" and n >= 2**31:
        binary, mem_key = MAW64, "maw64"
    if not os.path.exists(binary):
        return None, None, f"{os.path.relpath(binary, HERE)} not built for {ARCH}"
    if force:
        return binary, kind, None
    if n > LIMIT.get(tool, float("inf")):
        return None, None, f"{n:,} characters exceed the {LIMIT[tool]:,.0f} this build can index"
    need = (EMMAW_RAM_MB * 2**20 if tool == "emmaw" else MEMORY[mem_key] * n) / 2**30 + MEMORY_FIXED.get(tool, 0.3)
    if need > 0.85 * ram_gb:
        return None, None, f"needs about {need:.0f} GiB of memory, the machine has {ram_gb:.0f} GiB"
    if tool == "emmaw" and free_gb(tmp) < EMMAW_DISK * n / GB + 2:
        return None, None, f"needs about {EMMAW_DISK * n / GB:.0f} GB of temporary disk, {free_gb(tmp):.0f} GB free in {tmp}"
    return binary, kind, None


def command(tool, binary, inp, d, threads, check):
    """Shell command for one run: the program under /usr/bin/time -l, its stderr to d/time.txt."""
    T = "/usr/bin/time -l"
    E = f"2> {d}/time.txt"
    maw_args = f"-a DNA -i {inp} -k 1 -K {MAXLEN} -r 1"
    if tool == "pinho":
        return f"{T} {binary} {MAXLEN} -rc {inp} > /dev/null {E}"
    if tool == "pinho-list":
        return f"{T} {binary} {MAXLEN} -rc -w {inp} > /dev/null {E}"
    if tool == "mawmetal":
        return f"{T} {binary} -i {inp} -r 1 --joint -v -o {d}/out.txt {E}"
    if tool == "mawmetal-list":
        if check:
            return f"{T} {binary} -i {inp} -r 1 --joint --list -o /dev/stdout {E} | {COUNT} > {d}/counts.txt"
        return f"{T} {binary} -i {inp} -r 1 --joint --list -o /dev/null {E}"
    if tool == "maw":
        if check:
            return f"{T} {binary} {maw_args} -o /dev/stdout {E} | {COUNT} > {d}/counts.txt"
        return f"{T} {binary} {maw_args} -o /dev/null {E}"
    if tool == "pmaw":
        return f"{T} {binary} {maw_args} -o {d}/pmaw.out -t {threads} {E}"
    if tool == "pmaw-out":
        return f"{T} {binary} {maw_args} -o {d}/pmaw.out -t {threads} {E} && cat {d}/pmaw.out | {COUNT} > {d}/counts.txt"
    if tool == "emmaw":
        sink = f"cat {d}/out.fifo | {COUNT} > {d}/counts.txt" if check else f"cat {d}/out.fifo > /dev/null"
        return (f"mkfifo {d}/out.fifo && ln -s {inp} {d}/in.fa && ({sink} &) && "
                f"{T} {binary} -a DNA -i {d}/in.fa -o {d}/out.fifo -k 1 -K {MAXLEN} -r 1 -c 1 -m {EMMAW_RAM_MB} -f 1 "
                f"> {d}/stdout.txt {E}; s=$?; wait; exit $s")
    if tool == "bwtman":
        # index: FASTA input, append the reverse complement. MAWs: min length 2 (no letter is absent),
        # histogram only with --check (it prints one line per length up to the maximum).
        hist = f"1 {MAXLEN}" if check else "0 0"
        return (f"{T} sh -c '{BWT_INDEX} {inp} 1 1 {d}/index > {d}/index.txt && "
                f"{binary} {d}/index {threads} 2 {MAXLEN} {hist} 0 0 0 0 > {d}/maws.txt' {E}")
    raise ValueError(tool)


def parse(d, tool):
    r = {}
    txt = open(f"{d}/time.txt", errors="replace").read() if os.path.exists(f"{d}/time.txt") else ""
    m = re.search(r"([\d.]+) real\s+([\d.]+) user\s+([\d.]+) sys", txt)
    if m:
        r["wall"], r["user"], r["sys"] = map(float, m.groups())
    m = re.search(r"(\d+)\s+maximum resident set size", txt)
    if m:
        r["maxrss_gb"] = int(m.group(1)) / GB
    counts, total = None, None
    if tool.startswith("pinho"):
        counts = {int(b): int(a) for a, b in re.findall(r"found (\d+) minimal absent words of size (\d+)", txt)}
    elif tool == "mawmetal" and os.path.exists(f"{d}/out.txt"):
        counts = {}
        for line in open(f"{d}/out.txt"):
            p = line.split()
            if len(p) == 2 and not line.startswith(">"):
                counts[int(p[0])] = counts.get(int(p[0]), 0) + int(p[1])
    elif tool == "bwtman" and os.path.exists(f"{d}/maws.txt"):
        out = open(f"{d}/maws.txt").read()
        m = re.match(r"(\d+),\d+,\d+\|([\d.]+),([\d.]+)\|(\d+)\|(\d+),", out)
        if m:
            total = int(m.group(5))
            r["log"] = [f"text {m.group(1)} chars; load {m.group(2)} s, enumerate {m.group(3)} s, peak malloc {int(m.group(4)) / GB:.2f} GB"]
        idx = open(f"{d}/index.txt").read().strip() if os.path.exists(f"{d}/index.txt") else ""
        m = re.match(r"(\d+),(\d+),\d\|([\d.]+),([\d.]+),([\d.]+)\|(\d+)", idx)
        if m:
            r.setdefault("log", []).insert(0, f"index: input {m.group(1)}, text {m.group(2)} chars; load {m.group(3)} s, "
                                              f"BWT {m.group(4)} s, write {m.group(5)} s, peak malloc {int(m.group(6)) / GB:.2f} GB")
        hist = {int(a): int(b) for a, b in re.findall(r"^(\d+),(\d+) *$", out, re.M) if int(b)}
        if hist:
            counts = hist
    elif os.path.exists(f"{d}/counts.txt"):
        counts = {int(a): int(b) for a, b in (l.split() for l in open(f"{d}/counts.txt") if l.strip())}
    stages = [l for l in txt.splitlines() if l.startswith(("records ", "time: ", "GPU", "residue"))]
    if stages:
        r["log"] = r.get("log", []) + stages
    if counts:
        r["counts"] = {str(k): counts[k] for k in sorted(counts)}
        total = sum(counts.values())
    if total is not None:
        r["total"] = total
    if "wall" not in r:
        r["stderr_tail"] = txt[-1500:]
    return r


# ---- machine ----

def swap_used_gb():
    m = re.search(r"used = ([\d.]+)M", sh("sysctl vm.swapusage"))
    return float(m.group(1)) / 1024 if m else 0.0


def machine_state():
    m = re.search(r"(\d+)%", sh("memory_pressure 2>/dev/null | tail -1"))
    return {"load": os.getloadavg(), "mem_free_pct": int(m.group(1)) if m else None, "swap_used_gb": round(swap_used_gb(), 2),
            "power": sh("pmset -g batt 2>/dev/null | head -1"), "disk_free_gb": round(free_gb(HERE), 1)}


def machine_id():
    """Default machine name: the processor, e.g. apple-m1-max or intel-core-i7-3615qm."""
    s = sysctl("machdep.cpu.brand_string").lower() or platform.node().split(".")[0].lower()
    s = re.sub(r"\(r\)|\(tm\)|\bcpu\b|@.*", " ", s)
    return re.sub(r"[^a-z0-9]+", "-", s).strip("-") or "unknown"


def host_info():
    gpu = sh("system_profiler SPDisplaysDataType 2>/dev/null | grep -E 'Chipset Model|Total Number of Cores'")
    versions = {}
    vf = os.path.join(TOOLS, "versions.tsv")
    if os.path.exists(vf):
        versions = dict(l.rstrip("\n").split("\t", 1) for l in open(vf) if "\t" in l)
    return {
        "model": sysctl("hw.model"), "cpu": sysctl("machdep.cpu.brand_string"), "arch": ARCH,
        "cores": int(sysctl("hw.ncpu") or 0), "physical_cores": int(sysctl("hw.physicalcpu") or 0),
        "performance_cores": int(sysctl("hw.perflevel0.physicalcpu") or 0) or None,
        "efficiency_cores": int(sysctl("hw.perflevel1.physicalcpu") or 0) or None,
        "memory_gb": int(sysctl("hw.memsize") or 0) / 2**30,
        "gpu": " / ".join(l.strip() for l in gpu.splitlines()),
        "os": f"macOS {sh('sw_vers -productVersion')} ({sh('sw_vers -buildVersion')})", "kernel": platform.release(),
        "hostname": platform.node(), "python": platform.python_version(), "programs": versions,
    }


def run_guarded(script, timeout):
    """Runs a script, killing it if the machine starts swapping hard or it runs past the timeout."""
    start, t0 = swap_used_gb(), time.time()
    p = subprocess.Popen(["caffeinate", "-i", "sh", script], start_new_session=True)
    try:
        while p.poll() is None:
            time.sleep(1)
            why = None
            if swap_used_gb() - start > SWAP_GROWTH_GB:
                why = f"killed: swap grew by more than {SWAP_GROWTH_GB:.0f} GB"
            elif timeout and time.time() - t0 > timeout:
                why = f"killed: ran longer than {timeout} s"
            if why:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
                return -9, why
    except KeyboardInterrupt:
        os.killpg(p.pid, signal.SIGKILL)
        p.wait()
        raise
    return p.returncode, None


def write(path, doc):
    with open(path + ".tmp", "w") as f:
        json.dump(doc, f, indent=1)
    os.replace(path + ".tmp", path)


def summary(runs):
    walls = {}
    for r in runs:
        if r.get("exit") == 0 and "wall" in r:
            walls.setdefault((r["genome"], r["tool"]), []).append(r["wall"])
    if walls:
        print("\nmedian wall-clock seconds")
        for (g, t), v in walls.items():
            print(f"  {g:10s} {t:14s} {statistics.median(v):10.2f}  ({len(v)} runs)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--profile", choices=sorted(PROFILES), default="full",
                    help="full: everything, for the M1 Max; mbp2012: up to chr1, for a 16 GB Intel Mac")
    ap.add_argument("--genomes", help="comma-separated, default: the profile's")
    ap.add_argument("--tools", help="comma-separated, default: the profile's (" + ",".join(ALL_TOOLS) + ")")
    ap.add_argument("--reps", type=int, help="runs per program and genome, default: genomes.tsv (1 with --check)")
    ap.add_argument("--check", action="store_true", help="record the number of MAWs of each length instead of timing")
    ap.add_argument("--threads", type=int, default=os.cpu_count(), help="for pMAW and bwtman")
    ap.add_argument("--machine", default=machine_id(), help="name used in the result file name (default: %(default)s)")
    ap.add_argument("--tmp", default=tempfile.gettempdir(), help="directory for temporary files (emMAW needs ~%d bytes/base)" % EMMAW_DISK)
    ap.add_argument("--timeout", type=int, default=0, help="kill a run after this many seconds (default: no limit)")
    ap.add_argument("--force", action="store_true", help="run pairs that look too large for this machine")
    ap.add_argument("--keep-n-runs", action="store_true",
                    help="give MAW, pMAW and emMAW the N runs as they are, instead of one N per run")
    ap.add_argument("--plan", action="store_true", help="print what would run and what would be skipped, then stop")
    ap.add_argument("--download-only", action="store_true", help="download and prepare the genomes, then stop")
    a = ap.parse_args()

    manifest = read_manifest()
    genomes = (a.genomes or PROFILES[a.profile][0]).split(",")
    tools = (a.tools or ",".join(PROFILES[a.profile][1])).split(",")
    for g in genomes:
        if g not in manifest:
            sys.exit(f"unknown genome {g}; known: {','.join(manifest)}")
    for t in tools:
        if t not in ALL_TOOLS + ["pmaw-out"]:
            sys.exit(f"unknown program {t}; known: {','.join(ALL_TOOLS)}")
    if a.check:
        tools = [CHECK_TOOLS.get(t, t) for t in tools if CHECK_TOOLS.get(t, t)]
    ram_gb = int(sysctl("hw.memsize")) / 2**30

    started = datetime.datetime.now()
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, f"{a.machine}-{started:%Y%m%d-%H%M%S}.json")
    doc = dict(format=1, machine=a.machine, started=started.isoformat(timespec="seconds"), finished=None,
               argv=sys.argv[1:], profile=a.profile, check=a.check, threads=a.threads, host=host_info(),
               genomes={}, skipped=[], runs=[])
    if not (a.plan or a.download_only):
        print(f"results: {os.path.relpath(path)}", flush=True)
    try:
        for g in genomes:
            gm = manifest[g]
            planned = [(t,) + plan(t, gm, ram_gb, a.tmp, a.force) for t in tools]
            if a.plan:
                for t, binary, kind, why in planned:
                    if why:
                        print(f"skip {t:14s} {g:10s} {why}")
                print(f"run  {g:10s} {gm['acgt'] / 1e6:10,.2f} Mbp ACGT, "
                      f"{'downloaded' if os.path.exists(os.path.join(DATA, g, gm['file'])) else 'to download'}: "
                      + (" ".join(t for t, binary, kind, why in planned if binary) or "-"))
                continue
            if not a.download_only and not any(p[1] for p in planned):
                doc["skipped"] += [dict(genome=g, tool=t, reason=why) for t, binary, kind, why in planned]
                print(f"skip {g}: no program fits this machine", flush=True)
                continue
            src = download(gm)
            s = stats(gm, src)
            doc["genomes"][g] = dict(description=gm["description"], file=gm["file"], url=gm["url"], **s)
            jobs = []
            for t, binary, kind, why in planned:
                inp = None
                if binary:
                    if kind == "joined":
                        inp = joined_input(gm, src, s, a.keep_n_runs)
                    else:
                        inp = plain_input(gm, src, s["length"], kind == "plain")
                    if inp is None:
                        binary, why = None, f"not enough disk for the uncompressed {'joined ' if kind == 'joined' else ''}input"
                if binary:
                    jobs.append((t, binary, inp))
                else:
                    doc["skipped"].append(dict(genome=g, tool=t, reason=why))
                    print(f"skip {t:14s} {g:10s} {why}", flush=True)
            if a.download_only or not jobs:
                continue
            sh("cat " + " ".join(sorted({j[2] for j in jobs})) + " > /dev/null")  # warm the page cache
            reps = a.reps or (1 if a.check else gm["reps"])
            for rep in range(reps):  # round robin, so slow drifts in machine state hit every program
                for t, binary, inp in jobs:
                    d = tempfile.mkdtemp(prefix=f"{t}_{g}_", dir=a.tmp)
                    cmd = command(t, binary, inp, d, a.threads, a.check)
                    open(f"{d}/run.sh", "w").write(cmd + "\n")
                    state = machine_state()
                    t0 = time.time()
                    code, killed = run_guarded(f"{d}/run.sh", a.timeout)
                    rec = dict(time=datetime.datetime.now().isoformat(timespec="seconds"), tool=t, genome=g, rep=rep,
                               exit=code, elapsed=round(time.time() - t0, 2), input=os.path.relpath(inp, HERE),
                               binary=os.path.basename(binary), cmd=cmd, state=state, **parse(d, t))
                    if killed:
                        rec["killed"] = killed
                    doc["runs"].append(rec)
                    write(path, doc)
                    print(f"{t:14s} {g:10s} rep {rep} exit {code:3d} wall {rec.get('wall', -1):9.2f} s "
                          f"user {rec.get('user', -1):9.2f} s rss {rec.get('maxrss_gb', -1):6.2f} GB "
                          f"MAWs {rec.get('total', '-')}" + (f"  {killed}" if killed else ""), flush=True)
                    shutil.rmtree(d, ignore_errors=True)
    except KeyboardInterrupt:
        doc["interrupted"] = True
        print("\ninterrupted", flush=True)
    if a.plan or a.download_only:
        return
    doc["finished"] = datetime.datetime.now().isoformat(timespec="seconds")
    write(path, doc)
    summary(doc["runs"])
    print(f"\nresults: {os.path.relpath(path)}")


if __name__ == "__main__":
    main()
