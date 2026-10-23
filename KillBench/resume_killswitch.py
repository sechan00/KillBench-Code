"""Resume a Kill Switch payload run that crashed mid-execution.

Same as resume_baseline.py but ALSO applies task_patch.install() to inject the
payload, and propagates the worker env vars (KILLSWITCH_PAYLOAD_NAME etc.) so
joblib worker subprocesses re-apply the patch via bootstrap/sitecustomize.py.

Usage:
    python resume_killswitch.py --study-dir DIR --payload autoguard \
        --inject-delay-steps 1 --n-jobs 8 --parallel joblib
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
_SAFEARENA_REPO = HERE / "safearena_repo"
if _SAFEARENA_REPO.exists() and str(_SAFEARENA_REPO) not in sys.path:
    sys.path.insert(0, str(_SAFEARENA_REPO))

import env_compat  # noqa: E402
env_compat.install()

from dotenv import load_dotenv  # noqa: E402
load_dotenv(HERE.parent / ".env", override=True)

os.environ.setdefault("SAFEARENA_DATA_DIR", str(HERE / "safearena_data"))
os.environ.setdefault("SAFEARENA_TASK", "harm")
os.environ.setdefault("KILLBENCH_OBS_TIMEOUT_S", "45")

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

import timeout_patch  # noqa: E402
timeout_patch.install()

# Apply parent-process task_patch BEFORE Study.load() materialises agents.
# Worker subprocesses re-apply via sitecustomize.py reading our env vars.
import task_patch  # noqa: E402
import openai_reasoning_patch  # noqa: E402
openai_reasoning_patch.install()

try:
    import tokenizer_patch  # noqa: E402
    tokenizer_patch.install()
except Exception as e:
    logging.warning(f"tokenizer_patch.install() failed: {e}")

from agentlab.experiments.study import Study, get_most_recent_study  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("killswitch.resume_ks")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study-dir", required=True,
                    help="Study directory to resume")
    ap.add_argument("--payload", required=True,
                    help="Payload name (autoguard / gcg / ipi / ...)")
    ap.add_argument("--scenario", default=None,
                    help="Scenario override for scenario-based payloads")
    ap.add_argument("--inject-delay-steps", type=int, default=0,
                    help="Skip injection for first N _get_obs calls per task")
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--parallel", default="joblib",
                    choices=["sequential", "ray", "joblib"])
    ap.add_argument("--n-relaunch", type=int, default=2)
    args = ap.parse_args()

    study_dir = Path(args.study_dir).expanduser().resolve()
    if not study_dir.exists():
        raise SystemExit(f"study dir does not exist: {study_dir}")
    log.info(f"resuming kill-switch study at: {study_dir}")

    # 1) Install task_patch in parent (workers re-install via sitecustomize)
    task_patch.install(
        payload_name=args.payload,
        scenario=args.scenario,
        delay_steps=args.inject_delay_steps,
    )

    # 2) Propagate to workers via env vars (bootstrap/sitecustomize.py reads these)
    os.environ["KILLSWITCH_PAYLOAD_NAME"] = args.payload
    if args.scenario:
        os.environ["KILLSWITCH_SCENARIO_OVERRIDE"] = args.scenario
    else:
        os.environ.pop("KILLSWITCH_SCENARIO_OVERRIDE", None)
    if args.inject_delay_steps > 0:
        os.environ["KILLSWITCH_INJECTION_DELAY_STEPS"] = str(args.inject_delay_steps)
    else:
        os.environ.pop("KILLSWITCH_INJECTION_DELAY_STEPS", None)
    os.environ["KILLSWITCH_BENCH_DIR"] = str(HERE)

    # 3) Re-use existing injection log dir (created by the original eval.py run)
    inj_dir = study_dir / "killswitch_injection"
    inj_dir.mkdir(parents=True, exist_ok=True)
    os.environ["KILLSWITCH_INJECTION_LOG_DIR"] = str(inj_dir)
    log.info(f"injection log dir: {inj_dir}")

    # 4) Load study and resume
    study = Study.load(study_dir)
    log.info(f"study.name = {study.name}")
    n_incomplete, n_error = study.find_incomplete(include_errors=True)
    log.info(f"incomplete: {n_incomplete}  errors: {n_error}")
    if n_incomplete == 0:
        log.info("study already complete — nothing to do")
        return

    study.run(
        n_jobs=args.n_jobs,
        parallel_backend=args.parallel,
        strict_reproducibility=False,
        n_relaunch=args.n_relaunch,
    )
    log.info("resume finished")


if __name__ == "__main__":
    main()
