"""Run the SafeArena harness on our 250 jailbreak-wrapped prompts using
Claude 4.5 Sonnet (via Anthropic API directly, no OpenRouter).

This drives a real browser via Playwright against the locally-hosted SafeArena
Docker containers and uses SafeArena's own evaluation router to decide
completion. Output: AgentLab study folder + a small summary file we write to
`safearena_eval_summary.json`.

Costs: a multi-turn agent run can use ~5-20k tokens per task across screenshots
and reasoning. 250 tasks × ~$0.05–0.20 each => roughly $15-50 total.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from dotenv import load_dotenv
load_dotenv(HERE.parent / ".env")

# Set SafeArena-specific env BEFORE importing safearena.*
os.environ.setdefault("SAFEARENA_DATA_DIR", str(HERE / "safearena_data"))
os.environ.setdefault("SAFEARENA_TASK", "harm")

# WebArena URLs (localhost docker containers, ports from env-setup/00_vars.sh
# with USER_NUM=21 INSTANCE_NUM=0).
_WA = {
    "WA_HOMEPAGE":       "http://127.0.0.1:21000",
    "WA_SHOPPING":       "http://127.0.0.1:21001",
    "WA_SHOPPING_ADMIN": "http://127.0.0.1:21002/admin",
    "WA_REDDIT":         "http://127.0.0.1:21003",
    "WA_GITLAB":         "http://127.0.0.1:21004",
    "WA_WIKIPEDIA":      "http://127.0.0.1:21005/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing",
    "WA_MAP":            "http://127.0.0.1:21006",
    "WA_FULL_RESET":     "http://127.0.0.1:21009",
}
for k, v in _WA.items():
    os.environ.setdefault(k, v)

# Now import the SafeArena harness pieces
from agentlab.experiments.study import Study  # noqa: E402

from safearena import create_default_benchmark  # noqa: E402
from safearena.config import HARM_TASK_IDS  # noqa: E402

from anthropic_chat_adapter import prepare_claude_direct  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("killswitch.eval")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-sonnet-4-5-20250929")
    ap.add_argument("--n-jobs", type=int, default=2,
                    help="Concurrent browser sessions (keep low — each is heavy)")
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap on number of tasks for a quick test")
    ap.add_argument("--suffix", default="killbench")
    ap.add_argument("--parallel", default="sequential",
                    choices=["sequential", "ray", "joblib"])
    args = ap.parse_args()

    task_ids = list(HARM_TASK_IDS)
    if args.limit is not None:
        task_ids = task_ids[:args.limit]
        log.info(f"Limiting to first {len(task_ids)} task_ids")
    log.info(f"Total task_ids: {len(task_ids)}")

    benchmark = create_default_benchmark(task_ids=task_ids, name=f"killbench-harm")

    agent = prepare_claude_direct(model_name=args.model)
    log.info(f"Agent: {type(agent).__name__}  model={args.model}")

    study = Study([agent], benchmark, logging_level_stdout=logging.INFO,
                  suffix=args.suffix)
    log.info(f"Study dir: {study.dir}")

    study.run(
        n_jobs=args.n_jobs,
        parallel_backend=args.parallel,
        strict_reproducibility=False,
        n_relaunch=2,
    )
    log.info("Study finished")


if __name__ == "__main__":
    main()
