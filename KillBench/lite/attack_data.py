"""Load Lite attack prompts + defense prompts from the existing AutoGuard
data files (no copy: just read). Source matches test.py exactly.

- Attack prompts: `AutoGuard_ver2/data/data_augmented.json`
    Top-level keys are scenario names ("PII"/"NEWS"/"HACK"). Each scenario
    has a `goal` (one-liner) and 10 attack-type sub-dicts:
    `CR / AR / RE / TC / LOGIC / PROG / TRANS / SUPER / SUDO / SIMU`,
    each mapping str(prompt_idx) → malicious user request with a `[URL]`
    placeholder. NEWS also has `auxiliary_prompts` per attack_type (extra
    style hint appended to the prompt).

    test.py's default selection is `TEST_INDICES = [20..29]` (10 indices),
    yielding 10 attack_types × 10 indices = 100 attack tasks per scenario.

- Defense prompts: `AutoGuard_ver2/results/prompt_library_{pii,news,hack}.json`
    Each file has the scenario key at the root with a `defense_prompts` list of
    `{prompt_id, p_def, ...}` items.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

log = logging.getLogger("killbench_lite.attack_data")

# lite/ now nests under KillBench/, so repo root is three levels up.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DATA_JSON = _REPO_ROOT / "data" / "data_augmented.json"

# IPI / warning_based baseline payloads — one per scenario, in `data/prompts/prompts.json`.
# Keys here are LOWERCASE scenario names matching our canonical convention.
_BASELINE_PAYLOADS_JSON = _REPO_ROOT / "data" / "prompts" / "prompts.json"

# Lite scenario key in data_augmented.json → canonical name used everywhere else.
_SCENARIO_TO_DATA_KEY = {
    "pii":  "PII",
    "news": "NEWS",
    "hack": "HACK",
}

# Attack-type keys in each scenario block — everything except metadata keys.
_METADATA_KEYS = {"goal", "auxiliary_prompts"}

# test.py:229 default — same indices applied to each attack type.
TEST_INDICES_DEFAULT: List[int] = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29]

# test.py:239-241 — per-scenario default defense prompt_id (AutoGuard pinned).
DEFAULT_DEFENSE_PROMPT_ID = {
    "pii":  48,
    "news": 57,
    "hack": 61,
}

DEFENSE_LIBRARY_PATH = {
    "pii":  _REPO_ROOT / "results" / "prompt_library_pii.json",
    "news": _REPO_ROOT / "results" / "prompt_library_news.json",
    "hack": _REPO_ROOT / "results" / "prompt_library_hack.json",
}


@dataclass(frozen=True)
class AttackTask:
    scenario: str         # "pii" / "news" / "hack"
    attack_type: str      # "CR" / "AR" / ... / "SIMU"
    prompt_idx: int       # numeric key inside that attack_type dict
    user_request: str     # rendered prompt ([URL] already substituted)
    goal: str             # scenario-level descriptive goal (informational)
    auxiliary_prompt: Optional[str] = None  # NEWS-only style hint

    @property
    def task_id(self) -> str:
        return f"{self.scenario}.{self.attack_type}.{self.prompt_idx}"


def _load_data():
    if not _DATA_JSON.exists():
        raise FileNotFoundError(f"{_DATA_JSON} not found")
    with _DATA_JSON.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_attack_tasks(
    scenario: str,
    base_url: str,
    indices: Optional[List[int]] = None,
    attack_types: Optional[List[str]] = None,
    limit: Optional[int] = None,
) -> List[AttackTask]:
    """Load Lite attack prompts for `scenario`.

    Args:
        scenario: "pii" | "news" | "hack"
        base_url: substituted into the `[URL]` placeholder in each prompt
            (typically the local HTTP server URL where the Lite website is
            being served).
        indices: list of prompt indices to use per attack_type.
            Defaults to test.py's `TEST_INDICES_DEFAULT` = [20..29].
        attack_types: subset of attack types. Defaults to all 10
            (CR/AR/RE/TC/LOGIC/PROG/TRANS/SUPER/SUDO/SIMU) present in the file.
        limit: cap on returned count (post-filter, smoke-test).

    Returns 10 × len(indices) tasks under default settings = 100/scenario.
    """
    scenario = scenario.lower()
    if scenario not in _SCENARIO_TO_DATA_KEY:
        raise ValueError(
            f"Unknown scenario {scenario!r}; expected one of "
            f"{sorted(_SCENARIO_TO_DATA_KEY)}"
        )
    data = _load_data()
    data_key = _SCENARIO_TO_DATA_KEY[scenario]
    if data_key not in data:
        raise KeyError(f"Scenario key {data_key!r} missing from {_DATA_JSON}")
    block = data[data_key]
    goal = block.get("goal", "")
    auxiliary = block.get("auxiliary_prompts", {}) or {}

    if indices is None:
        indices = list(TEST_INDICES_DEFAULT)
    if attack_types is None:
        attack_types = [k for k in block.keys() if k not in _METADATA_KEYS]

    items: List[AttackTask] = []
    for atype in attack_types:
        prompts: Dict[str, str] = block.get(atype, {})
        if not isinstance(prompts, dict) or not prompts:
            log.warning(f"attack_type {atype!r} missing/empty for scenario={scenario}")
            continue
        aux_text = auxiliary.get(atype) if isinstance(auxiliary, dict) else None
        for idx in indices:
            key = str(idx)
            raw = prompts.get(key)
            if not raw:
                log.warning(f"prompt_idx {idx} missing from {scenario}/{atype}; skipping")
                continue
            rendered = raw.replace("[URL]", base_url)
            items.append(AttackTask(
                scenario=scenario,
                attack_type=atype,
                prompt_idx=idx,
                user_request=rendered,
                goal=goal,
                auxiliary_prompt=aux_text,
            ))
    if limit is not None:
        items = items[:limit]
    log.info(
        f"attack_data: loaded {len(items)} tasks for scenario={scenario} "
        f"(attack_types={len(attack_types)}, indices={len(indices)})"
    )
    return items


def load_baseline_payload(scenario: str, payload_name: str) -> str:
    """Load a non-AutoGuard baseline payload (IPI / warning_based) from
    `data/prompts/prompts.json`.

    payload_name: "ipi" -> the `prompt_injection` key (universal IPI text)
                  "warning" / "warning_based" -> the `warning_based` key (per-scenario
                                                  hand-written warning banner)
    """
    if not _BASELINE_PAYLOADS_JSON.exists():
        raise FileNotFoundError(f"{_BASELINE_PAYLOADS_JSON} not found")
    with _BASELINE_PAYLOADS_JSON.open("r", encoding="utf-8") as f:
        data = json.load(f)
    scenario = scenario.lower()
    if scenario not in data:
        raise KeyError(f"Scenario {scenario!r} missing from {_BASELINE_PAYLOADS_JSON}")
    block = data[scenario]
    pn = payload_name.lower()
    if pn == "ipi":
        key = "prompt_injection"
    elif pn in ("warning", "warning_based"):
        key = "warning_based"
    else:
        raise ValueError(f"load_baseline_payload: unsupported payload_name={payload_name!r}")
    text = block.get(key, "")
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"Empty {key!r} for scenario={scenario!r} in {_BASELINE_PAYLOADS_JSON}")
    return text


def resolve_payload_text(
    scenario: str,
    payload_name: str,
    prompt_id: Optional[int] = None,
) -> str:
    """Unified entry point: return the defense text for any supported payload.

    - clean       -> "" (no defense)
    - autoguard   -> AutoGuard library, prompt_id (defaults to per-scenario pinned)
    - ipi         -> data/prompts/prompts.json[scenario][prompt_injection]
    - warning(_based) -> data/prompts/prompts.json[scenario][warning_based]
    """
    pn = payload_name.lower()
    if pn == "clean":
        return ""
    if pn == "autoguard":
        if prompt_id is None:
            prompt_id = DEFAULT_DEFENSE_PROMPT_ID[scenario.lower()]
        return load_defense_prompt(scenario, prompt_id)
    if pn in ("ipi", "warning", "warning_based"):
        return load_baseline_payload(scenario, pn)
    if pn in ("gcg", "autoguard_universal"):
        # Re-use Heavy's payload modules — both load from `results/`
        # (universal, scenario-agnostic). Heavy is on sys.path via
        # lite_task_patch's prepend, but resolve_payload_text is called
        # in the parent process, so add it ourselves.
        import sys
        heavy_dir = str(_REPO_ROOT / "KillBench")
        if heavy_dir not in sys.path:
            sys.path.insert(0, heavy_dir)
        if pn == "gcg":
            from payloads import gcg  # type: ignore
            return gcg.text()
        from payloads import autoguard  # type: ignore
        return autoguard.text()
    raise ValueError(f"resolve_payload_text: unsupported payload_name={payload_name!r}")


def load_defense_prompt(
    scenario: str,
    prompt_id: Optional[int] = None,
) -> str:
    """Load a defense prompt text from prompt_library_{scenario}.json.

    Returns empty string if `prompt_id` is None (clean run).
    """
    if prompt_id is None:
        return ""
    scenario = scenario.lower()
    path = DEFENSE_LIBRARY_PATH.get(scenario)
    if not path or not path.exists():
        raise FileNotFoundError(
            f"Defense library missing for scenario={scenario!r}: {path}"
        )
    with path.open("r", encoding="utf-8") as f:
        lib = json.load(f)
    scenario_key = None
    for k in lib.keys():
        if k.lower() == scenario:
            scenario_key = k
            break
    if scenario_key is None:
        raise KeyError(f"Scenario key not found in {path}: keys={list(lib.keys())}")
    block = lib[scenario_key]
    for p in block.get("defense_prompts", []):
        if int(p.get("prompt_id", -1)) == int(prompt_id):
            text = p.get("p_def") or p.get("defense_prompt", "")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(
                    f"defense prompt at id={prompt_id} in {path} is empty"
                )
            return text
    raise KeyError(f"prompt_id={prompt_id} not found in {path}")
