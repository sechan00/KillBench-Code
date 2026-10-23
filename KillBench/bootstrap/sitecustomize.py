"""Auto-apply KillBench shims for joblib/loky worker subprocesses.

KillBench's eval.py applies env_compat + task_patch in the parent, but
loky spawns fresh worker processes that re-import everything from scratch and
never hit those calls. This module is auto-imported by Python at startup for
every process in this conda env, so it ensures every worker has the shims
applied before joblib unpickles the agent and before SafeArena's task runs.

This file lives in the repo at KillBench/bootstrap/sitecustomize.py.
Use bootstrap/install.sh to symlink it into the active conda env's
site-packages — that way edits here flow automatically to every Python
process in the env without re-running the installer.

Shims applied here:
  1. langchain.schema -> langchain_classic.schema  (agentlab import path)
  2. task_patch.install(payload_name=$KILLSWITCH_PAYLOAD_NAME,
                        scenario=$KILLSWITCH_SCENARIO_OVERRIDE)
     — only when those env vars are set (i.e. the parent eval.py exported
     them right before study.run()). This patches
     GenericSafeArenaTask.setup + BrowserEnv._get_obs in the worker process
     so the Kill Switch payload is injected and re-injected at obs time.

All shims are best-effort: ImportErrors are swallowed so this file is safe to
ship in a conda env that doesn't have KillBench checked out.
"""
import os
import sys

# ---- (1) langchain.schema alias --------------------------------------------
try:
    import langchain.schema  # noqa: F401
except ImportError:
    try:
        import langchain_classic.schema as _lc_schema
        sys.modules["langchain.schema"] = _lc_schema
    except ImportError:
        pass

# ---- (2) task_patch.install() in worker subprocesses -----------------------
_payload = os.environ.get("KILLSWITCH_PAYLOAD_NAME")
_scenario = os.environ.get("KILLSWITCH_SCENARIO_OVERRIDE") or None
if _payload:
    _killbench_dir = os.environ.get("KILLSWITCH_BENCH_DIR")
    if _killbench_dir:
        if _killbench_dir not in sys.path:
            sys.path.insert(0, _killbench_dir)
        # task_patch imports `from safearena.task import GenericSafeArenaTask`
        # at module load time, and the `safearena` package is shipped in-tree
        # at `<KillBench>/safearena_repo/`. Mirror the parent eval.py
        # sys.path setup so the import resolves in worker processes.
        _safearena_repo = os.path.join(_killbench_dir, "safearena_repo")
        if os.path.isdir(_safearena_repo) and _safearena_repo not in sys.path:
            sys.path.insert(0, _safearena_repo)
        # safearena.config reads SAFEARENA_DATA_DIR / SAFEARENA_TASK at import
        # time. Workers don't always inherit these from the parent, so default
        # them to the in-tree task data directory.
        _safearena_data = os.path.join(_killbench_dir, "safearena_data")
        if os.path.isdir(_safearena_data):
            os.environ.setdefault("SAFEARENA_DATA_DIR", _safearena_data)
        os.environ.setdefault("SAFEARENA_TASK", "harm")
    try:
        import task_patch  # type: ignore
        _delay_raw = os.environ.get("KILLSWITCH_INJECTION_DELAY_STEPS", "0")
        try:
            _delay = max(0, int(_delay_raw))
        except (TypeError, ValueError):
            _delay = 0
        _after_site_raw = os.environ.get("KILLSWITCH_INJECT_AFTER_SITE", "0")
        try:
            _after_site = max(0, int(_after_site_raw))
        except (TypeError, ValueError):
            _after_site = 0
        task_patch.install(payload_name=_payload, scenario=_scenario,
                           delay_steps=_delay, inject_after_site=_after_site)
    except Exception:
        # Don't crash the worker if KillBench isn't on sys.path — the parent
        # process is the only place this *must* work; the worker will simply
        # run without injection (which we'll catch via missing records).
        pass
    # ---- (3) reasoning-model compatibility for OpenAI gpt-5*/o1*/o3* -----
    # agentlab's ChatModel.__call__ always passes temperature + max_tokens,
    # which 400s on reasoning models. This patch routes those models through
    # max_completion_tokens (no temperature) so workers running gpt-5.5 etc.
    # don't fall into an indefinite "Connection error" retry loop.
    try:
        import openai_reasoning_patch  # type: ignore
        openai_reasoning_patch.install()
    except Exception:
        pass
    # ---- (3a) broaden agentlab.get_tokenizer fallback ----------------------
    # transformers 4.57.3 + Gemma fast tokenizer raises AttributeError during
    # __init__; agentlab only catches OSError, so without this patch every
    # step crashes for OpenRouter-routed `google/gemma-*` models.
    try:
        import tokenizer_patch  # type: ignore
        tokenizer_patch.install()
    except Exception:
        pass
    # ---- (3b) robust per-site UI login under concurrency -------------------
    # gitlab sign-in page can exceed the 10s default timeout when several
    # workers reset/login at once → setup fails (n_steps=0 ERROR). Wrap
    # ui_login with a higher timeout + retry/backoff so concurrent logins
    # don't turn into hard task failures.
    try:
        import login_patch  # type: ignore
        login_patch.install()
    except Exception:
        pass

# ---- (4) hardened-agent system prompt in worker subprocesses ---------------
# When the parent eval.py builds a hardened agent, it sets
# KILLSWITCH_SYSTEM_PROMPT_MODE=hardened. Workers re-import dynamic_prompting
# fresh, so the parent's patch doesn't survive — re-apply it here so the
# worker's GenericAgent.get_action() reads the bypass directive at step 0.
if os.environ.get("KILLSWITCH_SYSTEM_PROMPT_MODE") == "hardened":
    _killbench_dir_sp = os.environ.get("KILLSWITCH_BENCH_DIR")
    if _killbench_dir_sp and _killbench_dir_sp not in sys.path:
        sys.path.insert(0, _killbench_dir_sp)
    try:
        import system_patch  # type: ignore
        system_patch.install_hardened()
        with open("/tmp/sitecustomize_hardened_debug.log", "a") as _dbg:
            _dbg.write(f"pid={os.getpid()} ppid={os.getppid()} OK system_patch.install_hardened()\n")
    except Exception as _e:
        with open("/tmp/sitecustomize_hardened_debug.log", "a") as _dbg:
            import traceback
            _dbg.write(f"pid={os.getpid()} ppid={os.getppid()} FAIL {type(_e).__name__}: {_e}\n")
            _dbg.write(traceback.format_exc() + "\n")


# ---- (5) KillBench/lite worker bootstrap -----------------------------------
# Parent's KillBench/lite/eval.py exports LITE_PAYLOAD_NAME +
# LITE_DEFENSE_TEXT__<SCENARIO> + LITE_URL_<SCENARIO> + LITE_BENCH_DIR before
# study.run(). Workers spawn fresh and don't inherit the monkey-patches; this
# block re-applies them so defense_prompt injection actually fires in worker
# tasks (without it, parallel runs would silently bypass the defense and
# report artificially low DSR).
_lite_payload = os.environ.get("LITE_PAYLOAD_NAME")
_lite_dir = os.environ.get("LITE_BENCH_DIR")
if _lite_payload and _lite_dir:
    # Restore OpenAI base URL in ray worker actors (ray.init() doesn't pass
    # parent env vars). Parent eval.py exports LITE_OPENAI_BASE_URL /
    # LITE_OPENAI_API_BASE; mirror them back into the canonical env vars
    # before the OpenAI SDK is constructed.
    for _ev in ("OPENAI_BASE_URL", "OPENAI_API_BASE"):
        _alias = os.environ.get(f"LITE_{_ev}")
        if _alias and not os.environ.get(_ev):
            os.environ[_ev] = _alias
    if _lite_dir not in sys.path:
        sys.path.insert(0, _lite_dir)
    # lite_task_patch imports `from injection import inject_payload` and
    # `from payloads.base import ...` — both live in KillBench. Mirror
    # the parent's sys.path so workers resolve them.
    # lite/ nests under KillBench/, so the core dir IS the parent of _lite_dir.
    _heavy_dir_lite = os.path.dirname(_lite_dir)
    if os.path.isdir(_heavy_dir_lite) and _heavy_dir_lite not in sys.path:
        sys.path.insert(0, _heavy_dir_lite)
    _safearena_repo_lite = os.path.join(_heavy_dir_lite, "safearena_repo")
    if os.path.isdir(_safearena_repo_lite) and _safearena_repo_lite not in sys.path:
        sys.path.insert(0, _safearena_repo_lite)
    # Tokenizer / OpenAI-reasoning compatibility for Lite workers — same as
    # Heavy's path but here we must apply unconditionally (Lite supports
    # Gemma & GPT-5.x reasoning models out of the box).
    try:
        import tokenizer_patch  # type: ignore
        tokenizer_patch.install()
    except Exception:
        pass
    try:
        import openai_reasoning_patch  # type: ignore
        openai_reasoning_patch.install()
    except Exception:
        pass
    # ---- (5a) playwright singleton self-heal -------------------------------
    # browsergym/core/__init__.py caches a module-global sync_playwright
    # instance and reuses it across env.reset() calls. When a prior task
    # crashes the underlying greenlet / asyncio loop (frame detach, page
    # crash, oversized DOM evaluate timeout), the singleton's cached
    # `_connection._loop` becomes closed but `_PLAYWRIGHT` is never reset,
    # so every subsequent env.reset() dies on the first sync API call with
    # `RuntimeError: Selectors.set_test_id_attribute: no running event loop`.
    # Observed in KillBench/lite IPI runs on `hack` scenarios (large DOM +
    # heavy payload combo). Patch wraps the getter to detect a dead loop
    # and recycle the singleton transparently.
    try:
        import browsergym.core as _bg_core  # type: ignore

        _orig_get_pw = _bg_core._get_global_playwright

        def _safe_get_global_playwright():
            pw = getattr(_bg_core, "_PLAYWRIGHT", None)
            if pw is not None:
                loop_dead = False
                try:
                    conn = pw._impl_obj._connection
                    loop = getattr(conn, "_loop", None)
                    if loop is None or loop.is_closed():
                        loop_dead = True
                except Exception:
                    loop_dead = True
                if loop_dead:
                    try:
                        pw.stop()
                    except Exception:
                        pass
                    _bg_core._set_global_playwright(None)
            return _orig_get_pw()

        _bg_core._get_global_playwright = _safe_get_global_playwright
    except Exception:
        pass

    try:
        import lite_tasks  # type: ignore   # noqa: F401  (registers tasks on import)
        # Re-register tasks in the worker so gym.make("browsergym/killbench_lite.…")
        # resolves. Indices/attack_types pulled from env vars when set; otherwise
        # the defaults match the parent's register call.
        _indices_env = os.environ.get("LITE_INDICES")
        _attack_env = os.environ.get("LITE_ATTACK_TYPES")
        _indices = (
            [int(x) for x in _indices_env.split(",") if x.strip()]
            if _indices_env else None
        )
        _attack_types = (
            [x for x in _attack_env.split(",") if x.strip()]
            if _attack_env else None
        )
        _scenarios_env = os.environ.get("LITE_SCENARIOS", "pii,news,hack")
        _scenarios = tuple(x for x in _scenarios_env.split(",") if x.strip())
        lite_tasks.register_all_tasks(
            scenarios=_scenarios,
            indices=_indices,
            attack_types=_attack_types,
        )
        import lite_task_patch  # type: ignore
        lite_task_patch.install_from_env()
    except Exception:
        # Don't crash worker — parent will detect missing injection records
        # (or empty trajectories) and we can debug from there.
        pass
