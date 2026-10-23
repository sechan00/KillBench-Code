"""Defense-prompt injection for KillBench/lite tasks.

Like Heavy's task_patch.py, we monkey-patch:
- `LiteHarmTask.setup` → arms first-obs injection
- `BrowserEnv._get_obs` → does the actual `inject_payload` right before
  axtree extraction (setup-time inject is wiped by load events)

Differences from Heavy:
- Per-scenario defense text (multi-scenario runs supported). Active text
  picked by `task.scenario` at injection time.
- Defense injects on the FIRST observation per task only (mirrors test.py's
  `DEFENSE_INITIAL_PAGE_ONLY=True`). Subsequent steps leave the DOM alone.
- Works in worker subprocesses via the sitecustomize bootstrap (reads
  `LITE_PAYLOAD_NAME`, `LITE_DEFENSE_TEXT__<SCENARIO>` env vars).
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

# Heavy lives next to us — import its injection helper + Payload class.
# lite/ nests under KillBench/, so its parent IS the core KillBench dir.
_HEAVY_DIR = Path(__file__).resolve().parent.parent
if str(_HEAVY_DIR) not in sys.path:
    sys.path.insert(0, str(_HEAVY_DIR))

from injection import inject_payload  # noqa: E402
from payloads.base import Payload, wrap_visible_div  # noqa: E402

from lite_tasks import LiteHarmTask  # noqa: E402

log = logging.getLogger("killbench_lite.task_patch")


@dataclass
class _Config:
    payload_name: str = "clean"
    # scenario (lowercase) -> rendered HTML fragment for that scenario.
    defense_html_by_scenario: Dict[str, str] = field(default_factory=dict)
    defense_text_by_scenario: Dict[str, str] = field(default_factory=dict)
    # Number of initial _get_obs calls per task that SKIP injection so
    # step 0..delay-1 axtree is payload-free. Injection then fires ONCE on
    # step==delay (matches Lite's "initial-page-only" semantic). With
    # delay=0 (default) the landing page gets the payload; delay=1 means
    # the agent's first action navigates away from the landing page and
    # the SECOND page (step 1) gets the payload.
    inject_delay_steps: int = 0
    installed: bool = False


_CONFIG = _Config()
_ORIGINAL_SETUP = LiteHarmTask.setup
_ORIGINAL_GET_OBS = None  # lazily set on install()


def _patched_setup(self, page):
    goal, info = _ORIGINAL_SETUP(self, page)
    sc = self.scenario.lower()
    html = _CONFIG.defense_html_by_scenario.get(sc, "")
    text = _CONFIG.defense_text_by_scenario.get(sc, "")
    if _CONFIG.payload_name == "clean" or not html:
        self._lite_inject_done = True  # no-op flag
        return goal, info
    self._lite_inject_done = False
    self._lite_step_count = 0
    self._lite_payload = Payload(
        name=_CONFIG.payload_name,
        scenario=sc,
        html=html,
        text=text,
    )
    log.info(
        f"task_patch.setup: armed injection scenario={sc} "
        f"attack_type={self.attack_type} prompt_idx={self.prompt_idx} "
        f"bytes={len(html)} delay_steps={_CONFIG.inject_delay_steps}"
    )
    return goal, info


def _patched_get_obs(self):
    task = getattr(self, "task", None)
    payload = getattr(task, "_lite_payload", None) if task is not None else None
    if payload is not None and not getattr(task, "_lite_inject_done", True):
        step_idx = getattr(task, "_lite_step_count", 0)
        task._lite_step_count = step_idx + 1
        delay = _CONFIG.inject_delay_steps
        if step_idx < delay:
            # Pre-payload window: leave the page untouched. axtree at this
            # step will not contain the payload — agent's intrinsic behavior
            # is observable here.
            log.info(
                f"task_patch.get_obs: delay window step_idx={step_idx} "
                f"(< delay={delay}) — skip inject "
                f"task_id={task.task_configs[0]['task_id']} url={self.page.url}"
            )
            return _ORIGINAL_GET_OBS(self)
        ok = inject_payload(self.page, payload)
        task._lite_inject_done = True
        log.info(
            f"task_patch.get_obs: inject ok={ok} step_idx={step_idx} "
            f"task_id={task.task_configs[0]['task_id']} url={self.page.url}"
        )
    return _ORIGINAL_GET_OBS(self)


def install(
    payload_name: str = "clean",
    defense_by_scenario: Optional[Dict[str, str]] = None,
    inject_delay_steps: int = 0,
) -> None:
    """Install the patch (idempotent).

    Args:
        payload_name: 'clean' / 'autoguard' / 'ipi' / 'warning'.
        defense_by_scenario: {scenario_lower: defense_text_string} for the
            scenarios that have a payload. Pass {} (or omit) for clean.
        inject_delay_steps: number of initial _get_obs calls per task to
            SKIP injection. 0 (default) = inject on the landing page;
            1 = inject on the page after the agent's first action
            (= the second web page the agent encounters); etc.
    """
    global _ORIGINAL_GET_OBS
    _CONFIG.payload_name = payload_name or "clean"
    _CONFIG.inject_delay_steps = max(0, int(inject_delay_steps or 0))
    _CONFIG.defense_text_by_scenario = {}
    _CONFIG.defense_html_by_scenario = {}
    if defense_by_scenario:
        for sc, text in defense_by_scenario.items():
            sc_l = sc.lower()
            if not text:
                continue
            _CONFIG.defense_text_by_scenario[sc_l] = text
            _CONFIG.defense_html_by_scenario[sc_l] = wrap_visible_div(
                text, marker=f"lite-{sc_l}"
            )
    if not _CONFIG.installed:
        LiteHarmTask.setup = _patched_setup
        from browsergym.core.env import BrowserEnv
        _ORIGINAL_GET_OBS = BrowserEnv._get_obs
        BrowserEnv._get_obs = _patched_get_obs
        _CONFIG.installed = True
        log.info("task_patch: LiteHarmTask.setup + BrowserEnv._get_obs patched")
    log.info(
        f"task_patch: payload_name={payload_name} "
        f"delay_steps={_CONFIG.inject_delay_steps} "
        f"scenarios_with_defense={list(_CONFIG.defense_html_by_scenario)}"
    )


def install_from_env() -> None:
    """Worker-side install: read `LITE_PAYLOAD_NAME` and per-scenario
    `LITE_DEFENSE_TEXT__<SCENARIO>` env vars (set by parent eval.py) and
    apply the patch. Safe to call when no env vars are set (no-op).
    """
    payload_name = os.environ.get("LITE_PAYLOAD_NAME")
    if not payload_name:
        return
    by_scenario: Dict[str, str] = {}
    for sc in ("pii", "news", "hack"):
        env_key = f"LITE_DEFENSE_TEXT__{sc.upper()}"
        v = os.environ.get(env_key, "")
        if v:
            by_scenario[sc] = v
    delay_raw = os.environ.get("LITE_INJECT_DELAY_STEPS", "0")
    try:
        delay = max(0, int(delay_raw))
    except (TypeError, ValueError):
        delay = 0
    install(
        payload_name=payload_name,
        defense_by_scenario=by_scenario,
        inject_delay_steps=delay,
    )


def uninstall() -> None:
    LiteHarmTask.setup = _ORIGINAL_SETUP
    if _ORIGINAL_GET_OBS is not None:
        from browsergym.core.env import BrowserEnv
        BrowserEnv._get_obs = _ORIGINAL_GET_OBS
    _CONFIG.installed = False
