"""Resume the SafeArena baseline run, picking up only the tasks that did not
finish (no `summary_info.json`). Applies the BrowserGym timeout patch so hangs
like the harm.56 frame-detach trap (1.5 h block in `_pre_extract`) become
recoverable TimeoutErrors → task is marked as errored → harness moves on.

Usage:
    python3 resume_baseline.py [--study-dir DIR] [--n-jobs 1] [--parallel sequential]
If --study-dir is omitted, the most recent killbench study under
~/agentlab_results is used.
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

# Apply env compatibility shims (langchain.schema alias +
# reproducibility_util._get_repo stub). See env_compat.py.
import env_compat  # noqa: E402
env_compat.install()

from dotenv import load_dotenv
load_dotenv(HERE.parent / ".env")

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

# Patch BEFORE any browsergym import path matters.
import timeout_patch  # noqa: E402
timeout_patch.install()

from agentlab.experiments.study import Study, get_most_recent_study  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("killswitch.resume")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study-dir", default=None,
                    help="Study directory to resume (default: most recent killbench)")
    ap.add_argument("--n-jobs", type=int, default=1,
                    help="Concurrent workers (default: 1 to keep docker happy)")
    ap.add_argument("--parallel", default="sequential",
                    choices=["sequential", "ray", "joblib"])
    ap.add_argument("--n-relaunch", type=int, default=2,
                    help="Max relaunches per failing experiment")
    args = ap.parse_args()

    if args.study_dir:
        study_dir = Path(args.study_dir).expanduser().resolve()
    else:
        # New runs land in `<KillBench>/results/` (see eval.py).
        # Fall back to the legacy `~/agentlab_results/` if no recent run is
        # found in the new location.
        here = Path(__file__).resolve().parent
        new_root = here / "results"
        study_dir = None
        if new_root.exists():
            study_dir = get_most_recent_study(new_root, contains="killbench")
        if study_dir is None:
            study_dir = get_most_recent_study(
                Path.home() / "agentlab_results", contains="killbench"
            )
    log.info(f"resuming study at: {study_dir}")

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
