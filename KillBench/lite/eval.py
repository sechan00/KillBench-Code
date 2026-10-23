"""KillBench/lite evaluation harness.

Reuses Heavy's agentlab Study + GenericAgent + OpenRouter adapter against
Lite's static HTML scenarios (PII/NEWS/HACK) and Lite's attack/defense
prompt libraries.

Layout mirrors `KillBench/eval.py`. Key differences:
- No SafeArena / WebArena docker — websites are static HTML served from
  `data/website/<scenario>/` via a localhost HTTP thread (see
  `lite_server.py`).
- No category routing — Lite has 3 fixed scenarios with one attack-type
  family per scenario (the data.json prompts are all "attempt to extract
  PII / generate divisive comments / scan for vulns"). The active
  defense_prompt is set once via `--prompt-id` for the whole run.
- Sequential-first: n_jobs=1, parallel_backend="sequential" by default
  to avoid the worker-bootstrap dance that Heavy needs for loky-spawned
  subprocesses.

Examples
--------
# Clean baseline, PII scenario, 5 tasks, Qwen3.6-35B via OpenRouter
python KillBench/lite/eval.py \
    --agent default --model openrouter/qwen/qwen3.6-35b-a3b \
    --payload clean --scenario pii --limit 5 --suffix lite-pii-clean

# AutoGuard defense, PII scenario, default prompt_id=48
python KillBench/lite/eval.py \
    --agent default --model openrouter/qwen/qwen3.6-35b-a3b \
    --payload autoguard --scenario pii --prompt-id 48 \
    --limit 5 --suffix lite-pii-autoguard
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

# Import order: Heavy's env_compat shim FIRST so the langchain.schema alias
# and reproducibility_util._get_repo stub are installed before agentlab
# imports anything that requires them.
HERE = Path(__file__).resolve().parent
# lite/ nests under KillBench/: HERE.parent IS the canonical KillBench dir
# (formerly the "KillBench-Heavy" tree), repo root is one level above that.
HEAVY_DIR = HERE.parent
REPO_ROOT = HERE.parent.parent
# Lite's directory has a hyphen so it cannot be imported as a Python package.
# Add Heavy + Lite to sys.path. ORDER MATTERS: Lite must come BEFORE Heavy so
# Lite-specific modules win on conflict.
# `safearena_repo` is required because Heavy's `openrouter_chat_adapter` (which
# we reuse) imports `safearena.modeling.get_default_flags` for agentlab flag
# defaults. We do NOT use any actual SafeArena tasks in Lite.
_SAFEARENA_REPO = HEAVY_DIR / "safearena_repo"
if _SAFEARENA_REPO.exists() and str(_SAFEARENA_REPO) not in sys.path:
    sys.path.insert(0, str(_SAFEARENA_REPO))
if str(HEAVY_DIR) not in sys.path:
    sys.path.insert(0, str(HEAVY_DIR))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# safearena's import path needs these set early (mostly noops for Lite).
os.environ.setdefault("SAFEARENA_DATA_DIR", str(HEAVY_DIR / "safearena_data"))
os.environ.setdefault("SAFEARENA_TASK", "harm")
os.environ.setdefault("WA_HOMEPAGE", "http://127.0.0.1:21000")
os.environ.setdefault("WA_REDDIT", "http://127.0.0.1:21003")
os.environ.setdefault("WA_SHOPPING", "http://127.0.0.1:21001")
os.environ.setdefault("WA_SHOPPING_ADMIN", "http://127.0.0.1:21002/admin")
os.environ.setdefault("WA_GITLAB", "http://127.0.0.1:21004")
os.environ.setdefault("WA_WIKIPEDIA", "http://127.0.0.1:21005")
os.environ.setdefault("WA_MAP", "http://127.0.0.1:21006")
os.environ.setdefault("WA_FULL_RESET", "http://127.0.0.1:21009")

import env_compat  # noqa: E402  (from KillBench)
env_compat.install()

from dotenv import load_dotenv  # noqa: E402
load_dotenv(REPO_ROOT / ".env", override=True)

# agentlab writes Study trajectories here.
os.environ.setdefault("AGENTLAB_EXP_ROOT", str(HERE / "results"))
(HERE / "results").mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("killbench_lite.eval")


from browsergym.experiments.benchmark import Benchmark, HighLevelActionSetArgs  # noqa: E402
from browsergym.experiments.loop import EnvArgs  # noqa: E402


class LiteBenchmark(Benchmark):
    """Module-level Benchmark subclass so Study.save() can pickle it."""

    def prepare_backends(self):
        log.info("LiteBenchmark.prepare_backends: noop (HTTP servers start lazily)")

    def get_version(self):
        return "0.1"


def _build_benchmark(task_ids: list[str], max_steps: int = 15):
    """Build a Lite BrowserGym Benchmark over `task_ids`."""
    env_args_list = [
        EnvArgs(
            task_name=tid,
            task_seed=0,
            max_steps=max_steps,
            headless=True,
            record_video=False,
            wait_for_user_message=False,
            viewport=None,
            slow_mo=None,
            storage_state=None,
            task_kwargs=None,
        )
        for tid in task_ids
    ]
    return LiteBenchmark(
        name="killbench-lite",
        high_level_action_set_args=HighLevelActionSetArgs(
            subsets=["webarena"],
            multiaction=False,
            strict=False,
            retry_with_force=True,
            demo_mode="off",
        ),
        is_multi_tab=False,
        supports_parallel_seeds=False,
        backends=[],
        env_args_list=env_args_list,
    )


def _build_agent(model: str, no_vision: bool = False, agent_kind: str = "default"):
    """Build an agentlab agent factory for the requested model.

    agent_kind:
      - 'default'  → standard GenericAgent for the model (vision toggled via
                     no_vision; no extra system-prompt directive).
      - 'hardened' → RQ3 adaptive attacker. Routes through Heavy's
                     `agents.hardened.prepare_hardened`, which patches
                     `dynamic_prompting.SystemPrompt._prompt` with an
                     ADVERSARIAL_SYSTEM_PROMPT telling the agent to ignore
                     in-page STOP/HALT/Kill-Switch content. Worker subprocs
                     re-apply the patch via sitecustomize.py reading
                     `KILLSWITCH_SYSTEM_PROMPT_MODE=hardened`.
    """
    import openai_reasoning_patch

    ml = model.lower()
    vis = {"use_vision": not no_vision, "use_som": not no_vision}

    if agent_kind == "hardened":
        # Heavy's dispatcher already covers anthropic / openai / xai / openrouter
        # and installs the system-prompt patch + env var.
        from agents.hardened import prepare_hardened
        return prepare_hardened(model_name=model, **vis)

    if agent_kind != "default":
        raise ValueError(f"_build_agent: unsupported agent_kind={agent_kind!r}")

    from openrouter_chat_adapter import prepare_openrouter

    is_openrouter = (
        ml.startswith("openrouter/")
        or ml.startswith("qwen/")
        or ml.startswith("meta-llama/")
        or ml.startswith("google/")
        or ml.startswith("deepseek/")
        or ml.startswith("mistralai/")
        or ml.startswith("x-ai/")
    )
    or_model = model[len("openrouter/"):] if ml.startswith("openrouter/") else model

    if is_openrouter:
        return prepare_openrouter(model_name=or_model, **vis)

    is_gpt = ml.startswith("gpt-") or ml.startswith("openai/")
    if is_gpt:
        # Real OpenAI key path — clear any local-vLLM base-url overrides so the
        # SDK falls back to api.openai.com (matches Heavy/eval.py:218).
        if ml.startswith("gpt-"):
            for _ev in ("OPENAI_BASE_URL", "OPENAI_API_BASE"):
                if _ev in os.environ:
                    log.info(f"clearing {_ev}={os.environ[_ev]!r} for gpt-* run")
                    os.environ.pop(_ev, None)
        from safearena.modeling import prepare_gpt
        gpt_kwargs = dict(vis)
        # Reasoning models (gpt-5*/o1*/o3*/o4*) burn output tokens on hidden
        # reasoning before any visible text. Bump budget to match Heavy.
        if openai_reasoning_patch.is_reasoning_model(model):
            gpt_kwargs.update(
                max_new_tokens=8192,
                max_total_tokens=32768,
                max_prompt_tokens=24576,
            )
        return prepare_gpt(model_name=model, **gpt_kwargs)

    if ml.startswith("claude-") or ml.startswith("anthropic/"):
        from anthropic_chat_adapter import prepare_claude_direct
        return prepare_claude_direct(model_name=model, **vis)
    raise ValueError(f"_build_agent: unsupported model prefix in {model!r}")


def _build_agents(models: list, no_vision: bool = False,
                  agent_kind: str = "default") -> list:
    """Build one agent per model. agentlab Study takes a list and runs every
    task against every agent (so 2 models × 300 tasks = 600 trials)."""
    return [_build_agent(m, no_vision=no_vision, agent_kind=agent_kind)
            for m in models]


def run_study(
    agent_kind: str,
    models: list,
    payload_name: str,
    scenarios: list,
    prompt_id: Optional[int],
    limit: Optional[int],
    n_jobs: int,
    parallel: str,
    suffix: str,
    no_vision: bool,
    max_steps: int,
    temperature: Optional[float],
    indices: Optional[list] = None,
    attack_types: Optional[list] = None,
    inject_delay_steps: int = 0,
) -> Path:
    from agentlab.experiments.study import Study
    import attack_data
    import lite_server
    import lite_task_patch as task_patch
    import lite_tasks as tasks
    # Tokenizer / reasoning-model compatibility shims from Heavy. Required for
    # Gemma family (transformers 4.57.3 GemmaTokenizerFast init bug) and any
    # OpenAI reasoning model (gpt-5*/o1*/o3*).
    import tokenizer_patch
    import openai_reasoning_patch
    tokenizer_patch.install()
    openai_reasoning_patch.install()

    scenarios = [s.lower() for s in scenarios]

    # 0) Pre-start HTTP servers in the parent for all requested scenarios and
    # export their URLs so worker subprocesses reuse the same servers
    # (instead of starting their own which would mismatch the URL substituted
    # into the prompt at registration time).
    for sc in scenarios:
        url = lite_server.start_server(sc)
        os.environ[f"LITE_URL_{sc.upper()}"] = url
        log.info(f"exported LITE_URL_{sc.upper()}={url}")

    # 1) Register every Lite task with BrowserGym up front. Defaults
    # mirror test.py: 10 attack_types × 10 indices = 100 tasks/scenario,
    # so 3 scenarios = 300 tasks in a single Study.
    all_ids = tasks.register_all_tasks(
        scenarios=tuple(scenarios),
        indices=indices,
        attack_types=attack_types,
    )
    if limit is not None:
        all_ids = all_ids[:limit]
    log.info(f"running {len(all_ids)} tasks across scenarios={scenarios}")

    # 2) Resolve defense text per scenario and install the injection patch.
    defense_by_scenario: dict = {}
    if payload_name != "clean":
        for sc in scenarios:
            text = attack_data.resolve_payload_text(sc, payload_name, prompt_id)
            defense_by_scenario[sc] = text
            log.info(
                f"defense text resolved: scenario={sc} payload={payload_name} "
                f"chars={len(text)}"
            )
    task_patch.install(
        payload_name=payload_name,
        defense_by_scenario=defense_by_scenario,
        inject_delay_steps=inject_delay_steps,
    )
    # Export to env vars so worker subprocesses (via sitecustomize) install
    # the same patch.
    os.environ["LITE_PAYLOAD_NAME"] = payload_name
    os.environ["LITE_BENCH_DIR"] = str(HERE)
    if inject_delay_steps > 0:
        os.environ["LITE_INJECT_DELAY_STEPS"] = str(inject_delay_steps)
    else:
        os.environ.pop("LITE_INJECT_DELAY_STEPS", None)
    os.environ["LITE_SCENARIOS"] = ",".join(scenarios)
    # ray.init() doesn't pass parent env vars to worker actors by default, so
    # propagate OPENAI_BASE_URL / OPENAI_API_BASE through a LITE_-prefixed
    # alias that sitecustomize.py re-exports in workers. Without this, local
    # vLLM/llama-cpp setups (e.g. Qwen3.5-uncensored RR proxy on :8010) fall
    # back to api.openai.com and 400 with "invalid model ID".
    for _ev in ("OPENAI_BASE_URL", "OPENAI_API_BASE"):
        _val = os.environ.get(_ev)
        if _val:
            os.environ[f"LITE_{_ev}"] = _val
        else:
            os.environ.pop(f"LITE_{_ev}", None)
    if indices:
        os.environ["LITE_INDICES"] = ",".join(str(i) for i in indices)
    else:
        os.environ.pop("LITE_INDICES", None)
    if attack_types:
        os.environ["LITE_ATTACK_TYPES"] = ",".join(attack_types)
    else:
        os.environ.pop("LITE_ATTACK_TYPES", None)
    for sc, text in defense_by_scenario.items():
        os.environ[f"LITE_DEFENSE_TEXT__{sc.upper()}"] = text
    # Clear stale entries for scenarios not in this run.
    for sc in ("pii", "news", "hack"):
        if sc not in defense_by_scenario:
            os.environ.pop(f"LITE_DEFENSE_TEXT__{sc.upper()}", None)

    # 3) Build benchmark + agents (one agent per model).
    benchmark = _build_benchmark(all_ids, max_steps=max_steps)
    agents = _build_agents(models, no_vision=no_vision, agent_kind=agent_kind)
    if temperature is not None:
        for agent in agents:
            agent.chat_model_args.temperature = temperature
        log.info(f"override agent temperature={temperature} for all agents")
    for agent, m in zip(agents, models):
        log.info(f"agent: {type(agent).__name__} model={m}")

    # 4) Launch Study. Study accepts a list of agents; each task runs against
    # every agent (so 2 models × 300 tasks = 600 trials).
    study = Study(
        agents,
        benchmark,
        logging_level_stdout=logging.INFO,
        suffix=suffix,
    )
    study.make_dir()
    log.info(f"study dir: {study.dir}")

    study.run(
        n_jobs=n_jobs,
        parallel_backend=parallel,
        strict_reproducibility=False,
        n_relaunch=2,
    )
    return Path(study.dir)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agent", default="default",
                   choices=["default", "hardened"],
                   help="'default' = stock GenericAgent (vision toggled via "
                        "--no-vision). 'hardened' = RQ3 adaptive-attacker "
                        "baseline; injects ADVERSARIAL_SYSTEM_PROMPT into the "
                        "agent's `system` role telling it to ignore in-page "
                        "STOP/HALT/Kill-Switch content. Reuses Heavy's "
                        "agents.hardened.prepare_hardened (works for "
                        "anthropic / openai / xai / openrouter prefixes).")
    p.add_argument("--models", nargs="+", required=True,
                   help="one or more models to run in the SAME Study. "
                        "Each task runs against every model. "
                        "Examples: `--models openrouter/qwen/qwen3.6-35b-a3b openrouter/google/gemma-4-31b-it`")
    p.add_argument("--scenarios", nargs="+", required=True,
                   choices=["pii", "news", "hack"],
                   help="one or more scenarios to run together in a single Study "
                        "(e.g. `--scenarios pii news hack` for all 300 tasks)")
    p.add_argument("--payload", default="clean",
                   choices=["clean", "autoguard", "ipi", "warning",
                            "gcg", "autoguard_universal"],
                   help="'clean' = no defense; autoguard/ipi/warning load from "
                        "prompt_library_<scenario>.json or data/prompts/prompts.json; "
                        "gcg / autoguard_universal load Heavy's universal payloads "
                        "from results/gcg_universal_masteronly/ and "
                        "results/prompt_library_universal.json (scenario-agnostic).")
    p.add_argument("--prompt-id", type=int, default=None,
                   help="defense prompt_id; defaults to test.py per-scenario constant")
    p.add_argument("--indices", nargs="+", type=int, default=None,
                   help="prompt indices per attack_type (default = test.py TEST_INDICES [20..29])")
    p.add_argument("--attack-types", nargs="+", default=None,
                   help="subset of attack types (default = all 10: CR/AR/RE/TC/LOGIC/PROG/TRANS/SUPER/SUDO/SIMU)")
    p.add_argument("--limit", type=int, default=None,
                   help="cap total tasks (post-filter; for smoke-tests)")
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--parallel", default="sequential",
                   choices=["sequential", "joblib", "ray"])
    p.add_argument("--no-vision", action="store_true")
    p.add_argument("--max-steps", type=int, default=30,
                   help="agent step budget per task (Heavy default: 30)")
    p.add_argument("--inject-delay-steps", type=int, default=0,
                   help="skip payload injection for the first N observations. "
                        "0 (default) = inject on landing page; 1 = inject on the "
                        "page after the agent's first action (= second page the "
                        "agent encounters); etc. Mirrors Heavy's same-named flag.")
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--suffix", default="lite")
    args = p.parse_args()

    study_dir = run_study(
        agent_kind=args.agent,
        models=args.models,
        payload_name=args.payload,
        scenarios=args.scenarios,
        prompt_id=args.prompt_id,
        indices=args.indices,
        attack_types=args.attack_types,
        limit=args.limit,
        n_jobs=args.n_jobs,
        parallel=args.parallel,
        suffix=args.suffix,
        no_vision=args.no_vision,
        max_steps=args.max_steps,
        temperature=args.temperature,
        inject_delay_steps=args.inject_delay_steps,
    )
    print(f"\nDone. study_dir={study_dir}")


if __name__ == "__main__":
    main()
