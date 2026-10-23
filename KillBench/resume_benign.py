"""Resume an interrupted benign warning_universal study IN PLACE.

Loads the existing agentlab Study dir, re-applies the exact payload-injection
patches + KILLSWITCH_* env (mirrors eval.run_study), then relaunches only the
incomplete/errored experiments via agentlab's find_incomplete. Completed tasks
(those with summary_info.json) are skipped, so results accumulate in the SAME
study dir under results/benign_tasks — nothing is re-run or moved.

Params below match the original launch:
    --model gpt-5.2 --payload warning_universal --inject-delay-steps 1
    --parallel sequential --suffix warning-benign-delay1-gpt52-250
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.chdir(HERE)
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# eval.py's top-level wires up sys.path (safearena repo), .env, AGENTLAB_EXP_ROOT.
import eval as _eval  # noqa: F401,E402

PAYLOAD_NAME = "warning_universal"
SCENARIO = None
DELAY_STEPS = 1
INJECT_AFTER_SITE = 0

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
log = logging.getLogger("resume_benign")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", required=True, type=Path,
                        help="Existing benign study directory to resume")
    args = parser.parse_args()
    run_dir = args.study_dir.expanduser().resolve()
    if not run_dir.is_dir():
        parser.error(f"Study directory does not exist: {run_dir}")
    # --- replicate eval.run_study() patch + env setup (parent process) ---
    import openai_reasoning_patch
    import tokenizer_patch
    import login_patch
    import task_patch

    openai_reasoning_patch.install()
    tokenizer_patch.install()
    login_patch.install()
    task_patch.install(payload_name=PAYLOAD_NAME, scenario=SCENARIO,
                       delay_steps=DELAY_STEPS, inject_after_site=INJECT_AFTER_SITE)

    # gpt-* must hit api.openai.com, not a shell-exported vLLM base url.
    for _ev in ("OPENAI_BASE_URL", "OPENAI_API_BASE"):
        if _ev in os.environ:
            log.info(f"clearing {_ev}={os.environ[_ev]!r}")
            os.environ.pop(_ev, None)

    inj_dir = run_dir / "killswitch_injection"
    inj_dir.mkdir(parents=True, exist_ok=True)
    os.environ["KILLSWITCH_INJECTION_LOG_DIR"] = str(inj_dir)
    os.environ["KILLSWITCH_PAYLOAD_NAME"] = PAYLOAD_NAME
    os.environ.pop("KILLSWITCH_SCENARIO_OVERRIDE", None)
    os.environ["KILLSWITCH_INJECTION_DELAY_STEPS"] = str(DELAY_STEPS)
    os.environ.pop("KILLSWITCH_INJECT_AFTER_SITE", None)
    os.environ["KILLSWITCH_BENCH_DIR"] = str(HERE)
    os.environ.pop("KILLSWITCH_SYSTEM_PROMPT_MODE", None)

    # --- load existing study and relaunch incomplete experiments in place ---
    from agentlab.experiments.study import Study

    study = Study.load(run_dir)
    study.dir = run_dir
    n_incomplete, n_error = study.find_incomplete(include_errors=True)
    log.info(f"resume target: {n_incomplete} incomplete ({n_error} errored) in {run_dir}")

    study.run(
        n_jobs=1,
        parallel_backend="sequential",
        strict_reproducibility=False,
        n_relaunch=2,
    )
    log.info("resume run() returned")


if __name__ == "__main__":
    main()
