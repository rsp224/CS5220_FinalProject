#!/usr/bin/env python3
"""
Parse nsys stats for each rank's .nsys-rep file, aggregate per GPU-count,
and write prof/overlap_timing.csv.

Iteration and exposed-wait timings come from CPU-side NVTX wall time
(nvtx_sum). Compute and communication timings come from GPU-projected NVTX
ranges (nvtx_gpu_proj_sum), so async kernel launches are counted by GPU work
instead of CPU enqueue time.
"""

import csv
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

SCRATCH = os.environ.get("SCRATCH", "")
if not SCRATCH:
    sys.exit("ERROR: $SCRATCH is not set")

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUT_CSV = PROJECT_DIR / "prof" / "overlap_timing.csv"
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

COMPUTE_RANGES = {
    "h2d",
    "overlap/fwd",
    "overlap/bwd_loss",
    "overlap/bwd_layer3_grads",
    "overlap/bwd_layer3_input",
    "overlap/bwd_layer2_grads",
    "overlap/bwd_layer2_input",
    "overlap/bwd_layer1_grads",
    "overlap/bwd_layer1_input",
    "overlap/opt",
}


def run_nsys_stats(nsys_rep: Path) -> Dict[str, Dict[str, Dict[str, int]]]:
    """Run nsys stats and return parsed CPU/GPU NVTX summary tables."""
    result = subprocess.run(
        [
            "nsys", "stats",
            "--report", "nvtx_sum",
            "--report", "nvtx_gpu_proj_sum",
            "--format", "csv",
            "--force-export=true",
            str(nsys_rep),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    if result.returncode != 0:
        print(f"  WARN: nsys stats failed for {nsys_rep.name}: {result.stderr.strip()}")
        return {"cpu": {}, "gpu": {}}
    return parse_nvtx_csv(result.stdout)


def normalize_range_name(name: str) -> str:
    # nsys prefixes default-domain NVTX names with ":" (for example,
    # ":overlap/iter"). Strip only that marker; named domains keep their prefix.
    return name.strip().lstrip(":")


def parse_int(value: str) -> int:
    return int(value.replace(",", ""))


def parse_nvtx_csv(text: str) -> Dict[str, Dict[str, Dict[str, int]]]:
    """Parse combined nvtx_sum and nvtx_gpu_proj_sum CSV output."""
    tables: Dict[str, Dict[str, Dict[str, int]]] = {"cpu": {}, "gpu": {}}
    current: Optional[str] = None
    header: Optional[List[str]] = None

    for line in text.splitlines():
        line = line.strip()
        if not line:
            current = None
            header = None
            continue

        if line.startswith("Time (%),Total Time (ns),Instances,"):
            current = "cpu"
            header = next(csv.reader([line]))
            continue

        if line.startswith("Range,Style,Total Proj Time (ns),"):
            current = "gpu"
            header = next(csv.reader([line]))
            continue

        if current is None or header is None:
            continue

        row = next(csv.reader([line]))
        if len(row) != len(header):
            continue

        values = dict(zip(header, row))
        try:
            if current == "cpu":
                range_name = normalize_range_name(values["Range"])
                total_ns = parse_int(values["Total Time (ns)"])
                instances = parse_int(values["Instances"])
            else:
                range_name = normalize_range_name(values["Range"])
                total_ns = parse_int(values["Total Proj Time (ns)"])
                instances = parse_int(values["Range Instances"])
        except (KeyError, ValueError):
            continue

        tables[current][range_name] = {
            "total_ns": total_ns,
            "instances": instances,
        }

    return tables


def compute_metrics(tables: Dict[str, Dict[str, Dict[str, int]]]) -> Optional[Dict[str, float]]:
    """Compute per-iteration timing metrics from parsed NVTX tables."""
    cpu_ranges = tables["cpu"]
    gpu_ranges = tables["gpu"]

    if "overlap/iter" not in cpu_ranges:
        return None

    num_iters = cpu_ranges["overlap/iter"]["instances"]
    if num_iters == 0:
        return None

    def ms_per_iter(ranges: Dict[str, Dict[str, int]], name: str) -> float:
        if name not in ranges:
            return 0.0
        return ranges[name]["total_ns"] / num_iters / 1e6

    iter_ms = ms_per_iter(cpu_ranges, "overlap/iter")
    compute_ms = sum(ms_per_iter(gpu_ranges, r) for r in COMPUTE_RANGES)
    comm_total_ms = ms_per_iter(gpu_ranges, "overlap/comm_layer")
    exposed_ms = ms_per_iter(cpu_ranges, "overlap/wait_comm")
    hidden_ms = comm_total_ms - exposed_ms
    overlap_pct = (hidden_ms / comm_total_ms * 100) if comm_total_ms > 0 else 0.0

    return {
        "iter_ms": iter_ms,
        "compute_ms": compute_ms,
        "comm_total_ms": comm_total_ms,
        "exposed_comm_ms": exposed_ms,
        "overlap_pct": overlap_pct,
    }


def mean_metrics(metrics_list: List[Dict]) -> Dict:
    # Only average numeric metric fields — skip "rank" (rank index, not a metric).
    keys = [k for k, v in metrics_list[0].items()
            if k != "rank" and isinstance(v, (int, float))]
    return {k: sum(m[k] for m in metrics_list) / len(metrics_list) for k in keys}


GPU_COUNTS = [2, 4, 8, 16]

rows: List[Dict] = []

for n in GPU_COUNTS:
    rep_dir = Path(SCRATCH) / "prof" / f"overlap_{n}gpu"
    rep_files = sorted(rep_dir.glob("nsys_report_*.nsys-rep")) if rep_dir.exists() else []

    if not rep_files:
        print(f"[{n} GPUs] No .nsys-rep files found in {rep_dir}, skipping.")
        continue

    print(f"\n[{n} GPUs] Found {len(rep_files)} rank file(s)")
    rank_metrics: List[Dict] = []

    for rep in rep_files:
        rank = int(re.search(r"_(\d+)\.nsys-rep$", rep.name).group(1))
        print(f"  rank {rank}: {rep.name} ... ", end="", flush=True)
        nvtx = run_nsys_stats(rep)
        m = compute_metrics(nvtx)
        if m is None:
            print("no overlap/iter range found, skipping rank")
            continue
        m["rank"] = rank
        print(
            f"iter={m['iter_ms']:.1f}ms  compute={m['compute_ms']:.1f}ms  "
            f"comm={m['comm_total_ms']:.1f}ms  exposed={m['exposed_comm_ms']:.1f}ms  "
            f"overlap={m['overlap_pct']:.1f}%"
        )
        rows.append({"num_gpus": n, **m})
        rank_metrics.append(m)

    if rank_metrics:
        agg = mean_metrics(rank_metrics)
        rows.append({"num_gpus": n, "rank": "agg", **agg})
        print(
            f"  → AGG  iter={agg['iter_ms']:.1f}ms  compute={agg['compute_ms']:.1f}ms  "
            f"comm={agg['comm_total_ms']:.1f}ms  exposed={agg['exposed_comm_ms']:.1f}ms  "
            f"overlap={agg['overlap_pct']:.1f}%"
        )

# ── Write CSV ──────────────────────────────────────────────────────────────
fieldnames = ["num_gpus", "rank", "iter_ms", "compute_ms",
              "comm_total_ms", "exposed_comm_ms", "overlap_pct"]

with open(OUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow({
            "num_gpus": row["num_gpus"],
            "rank": row["rank"],
            "iter_ms": f"{row['iter_ms']:.3f}",
            "compute_ms": f"{row['compute_ms']:.3f}",
            "comm_total_ms": f"{row['comm_total_ms']:.3f}",
            "exposed_comm_ms": f"{row['exposed_comm_ms']:.3f}",
            "overlap_pct": f"{row['overlap_pct']:.2f}",
        })

print(f"\nCSV written → {OUT_CSV}")

# ── Summary table ──────────────────────────────────────────────────────────
agg_rows = [r for r in rows if r["rank"] == "agg"]
if agg_rows:
    print("\n── Aggregate summary across ranks ──────────────────────────────────")
    print(f"{'GPUs':>4}  {'iter_ms':>9}  {'compute_ms':>10}  {'comm_ms':>8}  {'exposed_ms':>10}  {'overlap%':>8}")
    print(f"{'----':>4}  {'---------':>9}  {'----------':>10}  {'--------':>8}  {'----------':>10}  {'--------':>8}")
    for r in agg_rows:
        print(
            f"{r['num_gpus']:>4}  {r['iter_ms']:>9.1f}  {r['compute_ms']:>10.1f}  "
            f"{r['comm_total_ms']:>8.1f}  {r['exposed_comm_ms']:>10.1f}  {r['overlap_pct']:>8.1f}%"
        )
