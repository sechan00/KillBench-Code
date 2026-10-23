"""Merge KillBench-Lite metrics JSONs by task_id (valid > ERROR), recompute metrics.

Usage:
  python scripts/merge_lite_metrics.py \
      --base results/metrics_lite_ipi_gemma4_default.json \
      --add  results/metrics_lite_ipi_gemma4_default_pwfix_rerun.json \
      --out  results/metrics_lite_ipi_gemma4_default_merged.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Reuse compute_metrics from KillBench-Lite/aggregate.py — same semantics as
# the per-study aggregator so merged headline numbers are directly comparable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "KillBench-Lite"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "KillBench-Heavy"))
from aggregate import compute_metrics  # noqa: E402


def merge_rows(base_rows, add_rows):
    """Index by task_id; valid (label != ERROR) wins over ERROR; later non-error overrides earlier."""
    by_id = {}
    for r in base_rows + add_rows:
        tid = r.get("task_id")
        if tid is None:
            continue
        prev = by_id.get(tid)
        if prev is None:
            by_id[tid] = r
            continue
        prev_err = prev.get("label") == "ERROR"
        cur_err = r.get("label") == "ERROR"
        if prev_err and not cur_err:
            by_id[tid] = r
        elif not prev_err and not cur_err:
            by_id[tid] = r  # later non-error overrides
        # else: keep prev
    return list(by_id.values())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True, type=Path,
                   help="original metrics JSON (with ERROR rows to be replaced)")
    p.add_argument("--add", required=True, type=Path,
                   help="rerun metrics JSON to merge in")
    p.add_argument("--out", required=True, type=Path,
                   help="output merged metrics JSON")
    args = p.parse_args()

    base = json.loads(args.base.read_text())
    add = json.loads(args.add.read_text())

    base_rows = base.get("rows", [])
    add_rows = add.get("rows", [])

    base_err = sum(1 for r in base_rows if r.get("label") == "ERROR")
    add_err = sum(1 for r in add_rows if r.get("label") == "ERROR")

    merged_rows = merge_rows(base_rows, add_rows)
    merged_err = sum(1 for r in merged_rows if r.get("label") == "ERROR")

    metrics = compute_metrics(merged_rows)

    out_blob = {
        "metadata": {
            **base.get("metadata", {}),
            "merged_from": [str(args.base), str(args.add)],
            "merge_stats": {
                "base_rows": len(base_rows),
                "base_errors": base_err,
                "add_rows": len(add_rows),
                "add_errors": add_err,
                "merged_rows": len(merged_rows),
                "merged_errors": merged_err,
            },
        },
        "metrics": metrics,
        "rows": merged_rows,
    }
    args.out.write_text(json.dumps(out_blob, ensure_ascii=False, indent=2))
    print(f"wrote {args.out}")
    print(f"  base rows={len(base_rows)} (ERROR={base_err})")
    print(f"  add  rows={len(add_rows)} (ERROR={add_err})")
    print(f"  merged rows={len(merged_rows)} (ERROR={merged_err})")
    print()
    for k in ("n", "n_total", "n_error", "ASR_pct", "ISR_pct", "KSR_pct",
              "PSR_pct", "total_refusal_pct", "CSR_pct_over_n",
              "CSR_pct_over_refused_KS"):
        if k in metrics:
            v = metrics[k]
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
