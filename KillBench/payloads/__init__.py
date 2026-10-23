"""Kill Switch payload registry.

Two payload axes:
- by-category: warning (5 hand-written variants), ipi (universal — same text
  for every category, signature kept for symmetry)
- by-scenario: gcg / autoguard / vision_autoguard (loaded from AutoGuard
  artifacts that were trained per scenario in {pii, news, hack})

`load_payload(name, ...)` accepts both `category` and `scenario` so callers
can pass either; the underlying builder picks whichever it actually needs.

Usage:
    from payloads import load_payload, PAYLOAD_NAMES
    p = load_payload("warning", category="Bias")
    page.evaluate("(h) => document.body.insertAdjacentHTML('beforeend', h)", p.html)
"""

from __future__ import annotations

from .base import Payload
from . import warning as _warning
from . import warning_universal as _warning_universal
from . import ipi as _ipi
from . import gcg as _gcg
from . import autoguard as _autoguard
from . import vision_autoguard as _vision
from . import vision_ipi as _vision_ipi

PAYLOAD_NAMES = ["warning", "warning_universal", "ipi", "gcg", "autoguard",
                 "vision_autoguard", "vision_ipi"]

# Which builders consume `category` vs `scenario`.
_CATEGORY_BUILDERS = {"warning", "warning_universal", "ipi", "vision_ipi"}
_SCENARIO_BUILDERS = {"gcg", "autoguard", "vision_autoguard"}

# Best-effort SafeArena-category → AutoGuard-scenario mapping for the
# scenario-based payloads (gcg / autoguard / vision_autoguard). These payloads
# were trained against three AutoGuard scenarios — we route SafeArena
# categories to the closest semantic neighbour. Adjust here if a different
# mapping is desired.
CATEGORY_TO_SCENARIO: dict[str, str] = {
    "Bias":             "news",   # political / social content
    "Cybercrime":       "hack",   # security / intrusion
    "Illegal Activity": "hack",   # restricted goods / unauthorised actions
    "Harassment":       "news",   # harmful posts
    "Misinformation":   "news",
}


def _resolve_scenario(category: str | None, scenario: str | None) -> str | None:
    if scenario:
        return scenario
    if category and category in CATEGORY_TO_SCENARIO:
        return CATEGORY_TO_SCENARIO[category]
    return None


def load_payload(
    name: str,
    category: str | None = None,
    scenario: str | None = None,
) -> Payload:
    """Return a Payload for `name`. Pass `category` (e.g. "Bias", "Cybercrime")
    for SafeArena-category-aware loaders, or `scenario` (e.g. "pii", "news",
    "hack") for legacy AutoGuard-scenario loaders. If only category is given
    for a scenario-based payload, it is routed via CATEGORY_TO_SCENARIO.
    """
    if name == "clean" or name is None:
        return Payload(name="clean", scenario=category or scenario or "generic", html="")

    if name == "warning":
        return _warning.build(category)
    if name == "warning_universal":
        return _warning_universal.build(category)
    if name == "ipi":
        return _ipi.build(category)
    if name == "vision_ipi":
        return _vision_ipi.build(category)

    # scenario-based payloads
    s = _resolve_scenario(category, scenario)
    if name == "gcg":
        return _gcg.build(s)
    if name == "autoguard":
        return _autoguard.build(s)
    if name == "vision_autoguard":
        return _vision.build(s)

    raise ValueError(f"Unknown payload: {name!r}. Choices: {PAYLOAD_NAMES}")


def get_payload_text(
    name: str,
    category: str | None = None,
    scenario: str | None = None,
) -> str:
    """Return the raw text (no HTML wrapper) of a payload. Used by the LLM
    judges (KS-TRR / TAR / Halt-Quality) as the 'what the agent might have
    seen' context. Returns "" for unknown or empty (clean) payloads.

    Resolves `category` to `scenario` for scenario-based payloads, mirroring
    `load_payload`.
    """
    if not name or name == "clean":
        return ""
    if name == "warning":
        return _warning.text(category)
    if name == "warning_universal":
        return _warning_universal.text(category)
    if name == "ipi":
        return _ipi.text(category)
    if name == "vision_ipi":
        return _vision_ipi.text(category)
    s = _resolve_scenario(category, scenario)
    if name == "gcg":
        return _gcg.text(s)
    if name == "autoguard":
        return _autoguard.text(s)
    if name == "vision_autoguard":
        return _vision.text(s)
    return ""


__all__ = ["Payload", "load_payload", "get_payload_text",
           "PAYLOAD_NAMES", "CATEGORY_TO_SCENARIO"]
