"""Resume an OpenClaw eval run by re-executing ONLY the ERROR tasks.

Use this when a previous `eval.py --agent openclaw ...` run finished but
some tasks failed (xAI billing exhaustion, openclaw subprocess timeout,
gateway crash, etc.) and you don't want to redo the 200+ already-good
tasks.

What it does:
  1. Load the eval JSON produced by eval.py (`eval_<run_id>_openclaw_*.json`).
  2. Identify rows with `label == "ERROR"` (or `--also-retry` selectors).
  3. Re-run each via `agents.openclaw.run_one`, writing into the SAME
     study_dir (overwrites the failed `<task_id>/` subdir's
     openclaw_session.jsonl + openclaw_meta.json + injection record).
  4. Re-classify each retried row with the same judge.
  5. Save updated results back to the original eval JSON (in-place) and
     emit a summary diff.

Untouched tasks keep their original session/meta/judge label.

The study_dir is resolved from `summary.study_dir`. The model/payload/
scenario default to whatever's in `summary.*` but can be overridden on
the CLI — useful for switching from xAI direct (billing-failed) to
OpenRouter without re-doing the good rows.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

# Reuse eval.py's env setup (WA_*, OPENCLAW_HOME, SafeArena harness, etc.)
# so URL templates expand and tasks load the same way.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import env_compat
env_compat.install()

# eval.py applies env defaults + URL template expansion at import time.
import eval as eval_mod  # noqa: E402

from agents.openclaw import run_one  # noqa: E402
from aria_judge import make_judge_client  # noqa: E402
from payloads import load_payload  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("killswitch.resume_openclaw")


def _load_eval(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_eval(path: Path, doc: dict) -> None:
    # Atomic-ish rewrite — write next to original, then rename.
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def _build_task_map() -> dict:
    """task_id -> task dict, indexed from safearena_data/harm.json."""
    tasks = eval_mod.load_harm_tasks()
    return {t["task_id"]: t for t in tasks}


def _judge_label(client, response: str, intent: str, judge_model: str) -> str:
    if not response:
        return "ERROR"
    return eval_mod._judge_one(client, response, intent, judge_model)


def _select_targets(
    rows: list[dict],
    only: Optional[list[str]] = None,
    also_retry: Optional[list[str]] = None,
) -> list[dict]:
    """Pick which rows to retry.

    Default: every row with `label == "ERROR"`.
    `only`: restrict to a specific list of task_ids (test mode).
    `also_retry`: extra labels to retry on top of ERROR (e.g. ["AMBIGUOUS"]).
    """
    extra = set((also_retry or []) + ["ERROR"])
    if only:
        keep = set(only)
        return [r for r in rows if r["task_id"] in keep]
    return [r for r in rows if r.get("label") in extra]


def _summarise(rows: list[dict]) -> dict:
    """Recompute the eval-level summary block over the (possibly updated) rows.

    Mirrors `eval.aggregate(...)` so the in-place JSON stays consistent.
    """
    return eval_mod.aggregate(rows)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-json", required=True, type=Path,
                    help="Path to eval_<run_id>_openclaw_*.json produced by eval.py")
    ap.add_argument("--model", default=None,
                    help="Override model for the retry (e.g. switch to "
                         "openrouter/x-ai/grok-4.3). Default: same as original run.")
    ap.add_argument("--judge-model", default=None,
                    help="Override judge model. Default: same as original run.")
    ap.add_argument("--payload", default=None,
                    help="Override payload (default: same as original run).")
    ap.add_argument("--scenario", default=None,
                    help="Override scenario (default: same as original run).")
    ap.add_argument("--workers", type=int, default=6,
                    help="Concurrent retries (default: 6).")
    ap.add_argument("--openclaw-timeout", type=int, default=600,
                    help="Per-task subprocess wall-clock cap (seconds).")
    ap.add_argument("--only", nargs="+", default=None,
                    help="Retry only these task_ids (e.g. harm.62 harm.143).")
    ap.add_argument("--also-retry", nargs="+", default=None,
                    help="Extra labels to retry alongside ERROR.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print what would be retried, don't actually run.")
    args = ap.parse_args()

    if not args.eval_json.exists():
        log.error(f"eval-json not found: {args.eval_json}")
        return 2

    doc = _load_eval(args.eval_json)
    summary = doc.get("summary", {})
    rows = doc.get("results", [])
    if not rows:
        log.error("eval JSON has no `results` list")
        return 2

    # Resolve original run params; CLI flags override.
    study_dir = Path(summary.get("study_dir") or "").expanduser()
    if not study_dir.is_absolute():
        study_dir = (HERE / study_dir).resolve() if str(study_dir) else study_dir
    if not study_dir or not study_dir.exists():
        log.error(f"study_dir from eval JSON does not exist: {study_dir!r}")
        return 2

    model = args.model or summary.get("model")
    judge_model = args.judge_model or summary.get("judge_model") or "gpt-5-mini"
    payload_name = args.payload or summary.get("payload") or "clean"
    scenario = args.scenario or summary.get("scenario")
    if scenario in (None, "None", "generic"):
        scenario = None

    inj_dir = study_dir / "killswitch_injection"
    inj_dir.mkdir(parents=True, exist_ok=True)
    os.environ["KILLSWITCH_INJECTION_LOG_DIR"] = str(inj_dir)

    targets = _select_targets(rows, only=args.only, also_retry=args.also_retry)
    log.info(
        f"resume plan: study_dir={study_dir.name} "
        f"model={model} payload={payload_name} scenario={scenario} "
        f"judge={judge_model} workers={args.workers} "
        f"targets={len(targets)}/{len(rows)}"
    )
    if args.dry_run:
        for r in targets:
            err = (r.get("error") or "")[:80].replace("\n", " ")
            print(f"  would retry {r['task_id']:8s} (orig label={r.get('label')!r}, err={err!r})")
        return 0
    if not targets:
        log.info("nothing to retry")
        return 0

    task_map = _build_task_map()
    client = make_judge_client(judge_model)
    mapping = eval_mod.load_mapping()
    row_by_tid = {r["task_id"]: r for r in rows}

    def _retry_one(orig_row: dict) -> dict:
        tid = orig_row["task_id"]
        t = task_map.get(tid)
        if not t:
            return {**orig_row, "label": "ERROR",
                    "error": "task_id missing from harm.json"}
        intent = t["intent"]
        start_url = eval_mod.expand_start_url(t.get("start_url") or "")
        category = mapping.get(tid, {}).get("category")
        payload = load_payload(payload_name, category=category, scenario=scenario)
        log.info(
            f"RETRY tid={tid} cat={category!r} "
            f"payload={payload.name}/{payload.scenario} url={start_url[:60]}..."
        )
        try:
            r = run_one(
                task=intent,
                start_url=start_url,
                model_name=model,
                payload=payload if payload.html else None,
                study_dir=study_dir,
                headless=True,
                timeout_s=args.openclaw_timeout,
                task_id=tid,
                category=category,
            )
        except Exception as e:
            log.exception(f"resume run_one crashed for {tid}")
            return {
                **orig_row,
                "label": "ERROR",
                "error": f"{type(e).__name__}: {e}",
                "final_response": "",
                "n_steps": 0,
            }
        label = _judge_label(client, r.final_response, intent, judge_model) \
            if r.success else "ERROR"
        return {
            **orig_row,                # preserve attack_type / category / etc.
            "label": label,
            "final_response": r.final_response,
            "n_steps": r.n_steps,
            "session_id": r.session_id,
            "error": r.error,
            # Refresh derived fields so downstream tooling reads fresh state.
            "retried_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }

    t0 = time.time()
    updated: dict[str, dict] = {}
    if args.workers <= 1:
        for orig in targets:
            updated[orig["task_id"]] = _retry_one(orig)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(_retry_one, r): r["task_id"] for r in targets}
            for fut in as_completed(futs):
                out = fut.result()
                updated[out["task_id"]] = out

    elapsed = time.time() - t0

    # Merge: keep order of original rows, swap in updated ones.
    new_rows = [updated.get(r["task_id"], r) for r in rows]
    pre = Counter(r.get("label") for r in rows)
    post = Counter(r.get("label") for r in new_rows)

    summary_new = _summarise(new_rows)
    # Preserve all original summary fields the rewriter shouldn't touch
    # (run_id, agent, scenario, etc.), but overlay the recomputed counts/pct.
    merged_summary = {**summary, **summary_new}
    merged_summary["resumed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    merged_summary["resume_elapsed_seconds"] = round(elapsed, 1)
    if args.model and args.model != summary.get("model"):
        merged_summary["model_resumed_with"] = args.model

    doc_new = {"summary": merged_summary, "results": new_rows}
    _save_eval(args.eval_json, doc_new)

    print()
    print("=== RESUME DIFF ===")
    print(f"  retried:  {len(targets)} tasks in {elapsed:.1f}s")
    print(f"  before:   {dict(pre)}")
    print(f"  after:    {dict(post)}")
    print(f"  DSR:      {summary.get('DSR_pct')}% -> {merged_summary['DSR_pct']}%")
    print(f"  ASR:      {summary.get('ASR_pct')}% -> {merged_summary['ASR_pct']}%")
    print(f"  rewrote:  {args.eval_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
