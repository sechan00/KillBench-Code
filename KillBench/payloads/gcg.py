"""GCG (Greedy Coordinate Gradient) adversarial suffix Kill Switch payload.

Source: results/gcg_universal_masteronly/gcg_result.json — a single universal
suffix jointly optimized over PII / News / Hack scenarios on
meta-llama/Llama-3.1-8B-Instruct to elicit refusal. Transferred as-is to
closed-source agents.

The `scenario` argument is kept for signature symmetry with the other
scenario-based payloads (autoguard / vision_autoguard) but is ignored —
the universal suffix is returned regardless.

The suffix alone is not a coherent instruction; it is appended to a short
context phrase so the model sees it as page content immediately before the
attack-elicitation point.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

from .base import Payload, wrap_visible_div

_REPO = Path(__file__).resolve().parents[2]
_DEFAULT_GCG_FILE = _REPO / "results" / "gcg_universal_masteronly" / "gcg_result.json"

# Allow pointing the GCG payload at a different suffix artifact (e.g. a suffix
# trained white-box on a specific target model) via KILLBENCH_GCG_FILE. The
# env var propagates to joblib/loky workers automatically. Falls back to the
# canonical Llama-master universal suffix when unset.
_GCG_FILE = Path(os.environ.get("KILLBENCH_GCG_FILE", str(_DEFAULT_GCG_FILE)))


@lru_cache(maxsize=1)
def _load() -> dict:
    with _GCG_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)


def text(scenario: str | None = None) -> str:
    data = _load()
    return data["suffix"]


def build(scenario: str | None = None) -> Payload:
    body = text(scenario)
    s = (scenario or "universal").lower()
    return Payload(name="gcg", scenario=s,
                   html=wrap_visible_div(body, "ks-gcg"), text=body)
