"""Monkey-patch SafeArena's `GenericSafeArenaTask.setup` so that after the
browser navigates to the task's `start_url`, we inject a Kill Switch payload
at the bottom of the page before the agent gets its first observation.

The payload is built PER TASK based on the task's SafeArena harm category,
read from `safearena_data/_killswitch_mapping.json`. This is what makes the
warning-based Kill Switch a different banner on a Bias task vs a Cybercrime
task. Universal payloads (IPI) ignore the category and inject the same text
everywhere.

For post-hoc metric computation (ASR/ISR/KSR/PSR/CSR), each successful
injection is recorded as a sidecar JSON file under the directory pointed
to by the `KILLSWITCH_INJECTION_LOG_DIR` environment variable. Each file
captures the marker substring `aggregate_results.py` uses to locate `k`
(the first step at which the payload becomes observable in axtree_txt).

Call `install(payload_name=...)` once before the Study runs.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from safearena.task import GenericSafeArenaTask

from injection import inject_payload
from payloads import load_payload

log = logging.getLogger("killswitch.task_patch")

_MAPPING_FILE = Path(__file__).resolve().parent / "safearena_data" / "_killswitch_mapping.json"

# Env var pointing to the per-study directory where each task writes its
# injection metadata. Set by eval.run_study() after Study() is constructed.
_LOG_DIR_ENV = "KILLSWITCH_INJECTION_LOG_DIR"

# How many leading characters of the payload text to use as the "marker"
# that aggregate_results.py substring-matches against axtree_txt. Keeping
# it short and unique avoids false negatives from agentlab's truncation of
# the accessibility tree.
_MARKER_CHARS = 120


@dataclass
class _Config:
    payload_name: str = "clean"
    scenario_override: Optional[str] = None
    # Number of initial _get_obs calls (per task) where payload injection is
    # SKIPPED, so step 0..delay_steps-1 axtree is payload-free. This makes
    # "agent refused before payload was observable" measurable as ISR in
    # payload runs (otherwise k=0 always and step-0 refusals collapse to KSR).
    delay_steps: int = 0
    # If > 1, payload injection is SKIPPED while the agent is on the Nth-or-
    # earlier distinct website (URL). Website rank is the 1-based first-
    # appearance order of the current page's URL within the task. With
    # inject_after_site=2, the agent's landing page (start_url, rank 1) stays
    # payload-free and the payload is only injected once the agent navigates
    # to a *second* distinct page — the IPI threat model where the malicious
    # content lives on retrieved content, not the landing page. 0/1 = inject
    # everywhere (legacy behaviour). Mutually independent from delay_steps.
    inject_after_site: int = 0
    installed: bool = False
    mapping: dict = field(default_factory=dict)


_CONFIG = _Config()
_ORIGINAL_SETUP = GenericSafeArenaTask.setup
_ORIGINAL_GET_OBS = None  # lazily set on first install() to avoid importing browsergym at module-import time
_ORIGINAL_VALIDATE = None  # ditto for webarena validate

# When True, suppress webarena's TCR-based early termination (score > 0). This
# decouples *measuring* task-completion reward from *ending* the episode, so a
# permissive evaluator firing on step 0/1 no longer kicks the agent out before
# the Kill Switch payload is observable. STOP-action termination is preserved.
_SUPPRESS_TCR_TERMINATION = True


def _load_mapping() -> dict:
    if not _MAPPING_FILE.exists():
        log.warning(f"_killswitch_mapping.json not found at {_MAPPING_FILE}; "
                    "category routing will use fallbacks.")
        return {}
    with _MAPPING_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_task_id(task) -> Optional[str]:
    """Extract a "harm.<N>" task_id from a GenericSafeArenaTask instance."""
    cfgs = getattr(task, "task_configs", None)
    if not cfgs:
        return None
    # task_configs is a list of one dict in practice — the active task.
    return cfgs[0].get("task_id")


def _write_injection_record(
    task_id: Optional[str],
    payload,
    category: Optional[str],
    page_url: str,
    injected: bool,
) -> Optional[Path]:
    """Write per-task injection metadata to the configured log dir.

    Returns the path written (for logging) or None if no log dir is set
    or the write failed. Safe to call from worker processes — each task
    writes a distinct filename so there is no contention.
    """
    log_dir = os.environ.get(_LOG_DIR_ENV)
    if not log_dir or not task_id:
        return None
    p = Path(log_dir)
    try:
        p.mkdir(parents=True, exist_ok=True)
        # Strip the wrap_visible_div wrapper out of `marker` — the marker
        # is what we substring-match against axtree_txt (which renders the
        # inner text, not the HTML attributes).
        marker = (payload.text or "").strip()[:_MARKER_CHARS]
        record = {
            "task_id": task_id,
            "payload_name": payload.name,
            "scenario": payload.scenario,
            "category": category,
            "payload_text": payload.text,
            "payload_text_marker": marker,
            "injection_method": "body.afterbegin",
            "injection_url": page_url,
            "injected": injected,
            "agent_path": "agentlab",
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        }
        out = p / f"{task_id}.json"
        out.write_text(json.dumps(record, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        return out
    except Exception as e:  # defensive — never break the run for telemetry
        log.warning(f"task_patch: failed to write injection record "
                    f"for {task_id}: {e}")
        return None


def _patched_setup(self, page):
    goal, d = _ORIGINAL_SETUP(self, page)

    task_id = _resolve_task_id(self)
    meta = _CONFIG.mapping.get(task_id, {}) if task_id else {}
    category = meta.get("category")

    if _CONFIG.payload_name in (None, "clean"):
        # Still log the clean run so aggregate_results.py knows the task
        # had no payload — distinguishes "no injection requested" from
        # "injection requested but failed".
        from payloads import Payload
        empty = Payload(name="clean", scenario=category or "generic",
                        html="", text="")
        _write_injection_record(task_id, empty, category, page.url,
                                injected=False)
        return goal, d

    payload = load_payload(
        _CONFIG.payload_name,
        category=category,
        scenario=_CONFIG.scenario_override,
    )
    log.info(
        f"task_patch: task_id={task_id} category={category!r} → "
        f"payload={payload.name}/{payload.scenario} bytes={len(payload.html)} "
        f"delay_steps={_CONFIG.delay_steps} "
        f"inject_after_site={_CONFIG.inject_after_site}"
    )
    # Setup-time inject is skipped when delay_steps > 0 (step-0 axtree stays
    # payload-free for ISR measurement) OR when inject_after_site > 1 (the
    # landing page / start_url is the 1st website and must stay payload-free;
    # the payload lands only once the agent navigates to a 2nd+ website). In
    # both cases the reinject path in _patched_get_obs handles all injection.
    _setup_skip = _CONFIG.delay_steps > 0 or _CONFIG.inject_after_site > 1
    if _setup_skip:
        ok = False
    else:
        ok = inject_payload(page, payload)
    # Still write the record so aggregate_results.py knows this is a payload
    # run (injected=True flag = "intent to inject"). The marker is what k
    # substring-matches against; it doesn't matter that the DOM hasn't been
    # touched yet — the marker will land in axtree once _get_obs re-injects.
    declared_injected = True if _setup_skip else ok
    rec_path = _write_injection_record(task_id, payload, category, page.url,
                                       injected=declared_injected)
    if rec_path:
        log.info(f"task_patch: injection record → {rec_path}")

    # Stash payload + metadata on the task instance so the BrowserEnv._get_obs
    # hook can re-inject right before the first observation is extracted.
    # The setup-time inject above is unreliable: webarena's auth flow leaves
    # the page in a state where domcontentloaded hasn't completed, and the
    # subsequent `_wait_dom_loaded` (or page-internal JS) wipes our <div>
    # before _get_obs extracts the axtree.
    self._killswitch_payload = payload
    self._killswitch_category = category
    self._killswitch_task_id = task_id
    self._killswitch_needs_reinject = True
    self._killswitch_step_count = 0
    # Ordered first-appearance list of distinct page URLs the agent has seen,
    # used by _patched_get_obs to compute the current page's "website rank"
    # for inject_after_site gating. Seeded empty; the first _get_obs appends
    # the landing page (rank 1).
    self._killswitch_seen_urls = []
    return goal, d


def _patched_get_obs(self):
    """Re-inject the payload before EVERY observation extraction.

    Why every step: navigation/reload wipes the injected <div> from the DOM,
    so a "first obs only" reinject leaves all subsequent steps unprotected.
    For vision_* payloads the agent's vision encoder only sees the patch on
    the very first page otherwise — a fatal failure mode confirmed empirically.

    When `_CONFIG.delay_steps > 0`, the first `delay_steps` _get_obs calls
    per task SKIP injection entirely so step 0..delay_steps-1 axtree is
    payload-free. Refusals in that window are intrinsic (ISR) — see
    aggregate_results._classify's `s_r < k` branch.

    The injection record is only written on the first successful re-inject per
    task (to keep aggregate_results.py's KSR/PSR anchoring stable); subsequent
    reinjects just refresh the DOM silently.
    """
    task = getattr(self, "task", None)
    payload = getattr(task, "_killswitch_payload", None) if task is not None else None
    if payload is not None:
        step_idx = getattr(task, "_killswitch_step_count", 0)
        task._killswitch_step_count = step_idx + 1
        delay = _CONFIG.delay_steps
        if step_idx < delay:
            # Pre-payload window: leave the page untouched. axtree at this
            # step will not contain the payload, making intrinsic refusals
            # distinguishable from KS-driven refusals at post-hoc time.
            log.info(f"task_patch: delay window step_idx={step_idx} "
                     f"(< delay={delay}) — skipping inject for "
                     f"task_id={getattr(task, '_killswitch_task_id', None)}")
            return _ORIGINAL_GET_OBS(self)
        # Website-rank gating: skip injection while the agent is still on the
        # 1st..(N-1)th distinct website. Rank = 1-based first-appearance order
        # of the current URL (fragment stripped). With inject_after_site=2 the
        # landing page stays clean and the payload only appears once the agent
        # navigates to a different page.
        after_site = _CONFIG.inject_after_site
        if after_site > 1:
            seen = getattr(task, "_killswitch_seen_urls", None)
            if seen is None:
                seen = []
                task._killswitch_seen_urls = seen
            cur_url = (self.page.url or "").split("#", 1)[0]
            if cur_url not in seen:
                seen.append(cur_url)
            rank = seen.index(cur_url) + 1
            if rank < after_site:
                log.info(f"task_patch: site-gate rank={rank} "
                         f"(< inject_after_site={after_site}) — skipping "
                         f"inject for "
                         f"task_id={getattr(task, '_killswitch_task_id', None)} "
                         f"url={cur_url}")
                return _ORIGINAL_GET_OBS(self)
        ok = inject_payload(self.page, payload)
        if getattr(task, "_killswitch_needs_reinject", False):
            task_id = getattr(task, "_killswitch_task_id", None)
            category = getattr(task, "_killswitch_category", None)
            _write_injection_record(task_id, payload, category,
                                    self.page.url, injected=ok)
            log.info(f"task_patch: first reinject before _get_obs ok={ok} "
                     f"task_id={task_id} step_idx={step_idx} "
                     f"url={self.page.url}")
            task._killswitch_needs_reinject = False
    return _ORIGINAL_GET_OBS(self)


def _patched_validate(self, page, chat_messages):
    """Wrap webarena's validate to suppress TCR-only early termination.

    Webarena's evaluator is permissive: some harm tasks fire `score > 0` on
    step 0/1 from URL string reflection or `must_include: ["="]` patterns,
    which immediately ends the episode (`if score > 0 or STOP: terminated=True`)
    before the Kill Switch payload is observable in axtree. Those episodes
    end up labeled ASR via `label_source='TCR'` despite the model never
    seeing the payload — a measurement artifact, not a defense failure.

    Fix: ignore `score > 0` as a termination trigger. STOP still terminates.
    Score is still returned (so downstream metrics can use it), only the
    termination flag is overridden. URL-safeguard terminations (score=0,
    terminated=True) are untouched.
    """
    score, terminated, msg, info = _ORIGINAL_VALIDATE(self, page, chat_messages)
    if not _SUPPRESS_TCR_TERMINATION:
        return score, terminated, msg, info
    if terminated and score > 0:
        # Was termination caused by STOP, or solely by score>0? webarena sets
        # last_action=STOP iff chat_messages[-1].role in ("assistant","infeasible").
        is_stop = bool(chat_messages) and chat_messages[-1].get("role") in (
            "assistant", "infeasible",
        )
        if not is_stop:
            task_id = _resolve_task_id(self)
            log.info(
                f"task_patch: suppressed TCR-only termination "
                f"task_id={task_id} score={score} url={page.url}"
            )
            terminated = False
    return score, terminated, msg, info


def install(payload_name: str = "clean", scenario: Optional[str] = None,
            delay_steps: int = 0, inject_after_site: int = 0) -> None:
    """Install the patch (idempotent) and set the active payload name.

    `scenario` (e.g. "pii"/"news"/"hack") is an OPTIONAL override for the
    scenario-based payloads (gcg / autoguard / vision_autoguard). If left as
    None, each task routes to a scenario by category via
    `payloads.CATEGORY_TO_SCENARIO`. Has no effect on warning / ipi.

    `delay_steps`: if > 0, the first `delay_steps` _get_obs calls per task
    skip payload injection so the agent sees a payload-free axtree at
    step 0..delay_steps-1. Use 1 to enable ISR-vs-KSR separation in payload
    runs (step 0 = ISR window, step 1+ = KSR window).

    `inject_after_site`: if > 1, the payload is injected only once the agent
    reaches the Nth distinct website (URL). The landing page (start_url) is
    website 1 and stays payload-free; with inject_after_site=2 the payload
    appears on the *second* page the agent navigates to (IPI threat model).
    0/1 = inject on every page (legacy). Independent from delay_steps.
    """
    global _ORIGINAL_GET_OBS, _ORIGINAL_VALIDATE
    _CONFIG.payload_name = payload_name or "clean"
    _CONFIG.scenario_override = scenario
    _CONFIG.delay_steps = max(0, int(delay_steps or 0))
    _CONFIG.inject_after_site = max(0, int(inject_after_site or 0))
    if not _CONFIG.mapping:
        _CONFIG.mapping = _load_mapping()
    if not _CONFIG.installed:
        GenericSafeArenaTask.setup = _patched_setup
        # Also patch BrowserEnv._get_obs so the inject lands right before
        # axtree extraction (see _patched_get_obs doc).
        from browsergym.core.env import BrowserEnv
        _ORIGINAL_GET_OBS = BrowserEnv._get_obs
        BrowserEnv._get_obs = _patched_get_obs
        # Patch validate on the webarena base so score>0 alone no longer
        # terminates the episode (see _patched_validate doc).
        from browsergym.webarena.task import GenericWebArenaTask
        _ORIGINAL_VALIDATE = GenericWebArenaTask.validate
        GenericWebArenaTask.validate = _patched_validate
        _CONFIG.installed = True
        log.info(
            "task_patch: GenericSafeArenaTask.setup + BrowserEnv._get_obs + "
            "GenericWebArenaTask.validate patched"
        )
    log.info(
        f"task_patch: active payload_name={_CONFIG.payload_name} "
        f"scenario_override={_CONFIG.scenario_override!r} "
        f"delay_steps={_CONFIG.delay_steps}"
    )


def uninstall() -> None:
    """Restore the original setup (for tests)."""
    GenericSafeArenaTask.setup = _ORIGINAL_SETUP
    if _ORIGINAL_GET_OBS is not None:
        from browsergym.core.env import BrowserEnv
        BrowserEnv._get_obs = _ORIGINAL_GET_OBS
    if _ORIGINAL_VALIDATE is not None:
        from browsergym.webarena.task import GenericWebArenaTask
        GenericWebArenaTask.validate = _ORIGINAL_VALIDATE
    _CONFIG.installed = False
    _CONFIG.payload_name = "clean"
    _CONFIG.scenario_override = None
