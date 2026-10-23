"""AutoGuard Kill Switch payload — text trigger optimized by the AutoGuard
search procedure.

Primary source (v6.0_model_quota): results/prompt_library_universal.json.
The "top-1" defense prompt is the arm with the highest Exp3 cumulative
reward in `phase1_step2_checkpoint.bandit_state.cumulative_rewards`,
indexed into `universal.defense_prompts`. This is scenario-agnostic, so
`scenario` is ignored when the universal library is present.

Legacy fallback (per-scenario): results/prompt_library_{pii,news,hack}.json
with the `top_k_defense_prompts.prompts[0]` schema. Used only when the
universal library file is missing.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from .base import Payload, wrap_visible_div

_REPO = Path(__file__).resolve().parents[2]
_UNIVERSAL_FILE = _REPO / "results" / "prompt_library_universal.json"
_LEGACY_FILES = {
    "pii":  _REPO / "results" / "prompt_library_pii.json",
    "news": _REPO / "results" / "prompt_library_news.json",
    "hack": _REPO / "results" / "prompt_library_hack.json",
}


def _select_top1_idx(rewards: list[float]) -> int:
    # Top-1 by Exp3 cumulative reward across the 366-round run.
    # arm#32 (prompt_id=33, grok-4-1-fast-reasoning, cum_R=3.349) at the
    # time of the 2026-05-16 checkpoint.
    return max(range(len(rewards)), key=lambda i: rewards[i])


@lru_cache(maxsize=1)
def _load_universal_top1() -> str:
    with _UNIVERSAL_FILE.open("r", encoding="utf-8") as f:
        data = json.load(f)
    prompts = data["universal"]["defense_prompts"]
    rewards = data["phase1_step2_checkpoint"]["bandit_state"]["cumulative_rewards"]
    if len(prompts) != len(rewards):
        raise RuntimeError(
            f"universal library schema mismatch: {len(prompts)} prompts vs "
            f"{len(rewards)} bandit arms in {_UNIVERSAL_FILE.name}"
        )
    return prompts[_select_top1_idx(rewards)]["p_def"]


@lru_cache(maxsize=8)
def _load_legacy_best(scenario: str) -> str:
    p = _LEGACY_FILES[scenario]
    if not p.exists():
        p = _LEGACY_FILES["pii"]
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    top = data.get("top_k_defense_prompts", {}).get("prompts", [])
    if not top:
        raise RuntimeError(f"No top_k_defense_prompts in {p}")
    return top[0]["p_def"]


def text(scenario: str | None = None) -> str:
    if _UNIVERSAL_FILE.exists():
        return _load_universal_top1()
    s = (scenario or "pii").lower()
    if s not in _LEGACY_FILES:
        s = "pii"
    return _load_legacy_best(s)


def build(scenario: str | None = None) -> Payload:
    if _UNIVERSAL_FILE.exists():
        body = _load_universal_top1()
        return Payload(name="autoguard", scenario="universal",
                       html=wrap_visible_div(body, "ks-autoguard"), text=body)
    s = (scenario or "pii").lower()
    if s not in _LEGACY_FILES:
        s = "pii"
    body = _load_legacy_best(s)
    return Payload(name="autoguard", scenario=s,
                   html=wrap_visible_div(body, "ks-autoguard"), text=body)
