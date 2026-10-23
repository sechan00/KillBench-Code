"""Build experiment_results.json from results/Experiment results.xlsx.

Reads the multi-sheet Excel and emits the hierarchical JSON consumed by
results/visualize_website/index.html.

Hierarchy: experiment -> scenario -> model -> defense -> by_attack_type
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
XLSX = ROOT / "results" / "Experiment results.xlsx"
WEB_DIR = ROOT / "results" / "visualize_website"
# Canonical JSON lives outside the web dir so users don't accidentally grab it
# instead of the .js data file when downloading the dashboard.
OUT = ROOT / "results" / "experiment_results.json"
DATA_JS_OUT = WEB_DIR / "experiment_results.js"
STANDALONE_OUT = WEB_DIR / "dashboard.html"            # single-file, data inlined
SPLIT_OUT = WEB_DIR / "dashboard_split.html"           # references experiment_results.js

ATTACK_TYPES = ["CR", "AR", "RE", "TC", "LOGIC", "PROG", "TRANS", "SUPER", "SUDO", "SIMU"]


def _num(v: Any) -> float | None:
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _pct(v: Any) -> float | None:
    """Normalize an ASR/RR cell to a percent in [0, 100].

    Cells in the workbook are inconsistent — some sheets store rates as 0..1
    fractions, others as 0..100 percents (notably NEWS sheet's GPT-4o block).
    Heuristic: treat values <= 1.0 as fractions, else as already-percent.
    """
    n = _num(v)
    if n is None:
        return None
    if n <= 1.0001:
        return round(n * 100, 2)
    return round(n, 2)


def _attack_block(asr: Any, rr: Any) -> dict[str, Any]:
    asr_pct = _pct(asr)
    rr_pct = _pct(rr)
    return {
        "attack_success_rate": asr_pct if asr_pct is not None else None,
        "refusal_rate": rr_pct if rr_pct is not None else None,
    }


# ---------------------------------------------------------------------------
# Sheet parsers
# ---------------------------------------------------------------------------

def parse_overall(xl: pd.ExcelFile) -> list[dict[str, Any]]:
    """Overall sheet — 2 sub-tables: by-scenario summary and by-defense summary."""
    df = xl.parse("Overall", header=None)

    scenarios_main = [  # rows 2..4
        ("PII", "PII collection"),
        ("Divisive", "Social rift reply"),
        ("Hack", "Web vulnerability scanning"),
    ]
    defenses = [  # (col_asr, col_rr, id, name)
        (1, 2, "clean", "Clean"),
        (3, 4, "prompt_injection", "Prompt Injection"),
        (5, 6, "warning_based", "Warning Based"),
        (7, 8, "autoguard", "AutoGuard"),
    ]

    overall_models = []  # we treat it as "model = aggregate" placeholder
    scenario_nodes = []
    for row_idx, (sid, sname) in enumerate(scenarios_main, start=2):
        defense_nodes = []
        for asr_c, rr_c, did, dname in defenses:
            asr = df.iat[row_idx, asr_c]
            rr = df.iat[row_idx, rr_c]
            defense_nodes.append({
                "id": did,
                "name": dname,
                "metrics": {
                    "attack_success_rate": _pct(asr),
                    "refusal_rate": _pct(rr),
                },
                "by_attack_type": {},
            })
        scenario_nodes.append({
            "id": sid,
            "name": sname,
            "models": [{
                "id": "aggregate",
                "name": "Aggregate (across models)",
                "defenses": defense_nodes,
            }],
        })

    # Aggregate-of-aggregate (rows 9..12) — defense-level summary across all scenarios
    agg_defense = []
    rowmap = [(9, "clean", "Clean"), (10, "prompt_injection", "Prompt Injection"),
              (11, "warning_based", "Warning Based"), (12, "autoguard", "AutoGuard")]
    for r, did, dname in rowmap:
        agg_defense.append({
            "id": did, "name": dname,
            "metrics": {
                "attack_success_rate": _pct(df.iat[r, 1]),
                "refusal_rate": _pct(df.iat[r, 2]),
            },
            "by_attack_type": {},
        })
    scenario_nodes.append({
        "id": "all_scenarios",
        "name": "All scenarios (mean)",
        "models": [{
            "id": "aggregate",
            "name": "Aggregate (across scenarios)",
            "defenses": agg_defense,
        }],
    })

    return scenario_nodes


def _parse_per_scenario_block(df: pd.DataFrame, models: list[tuple[str, str, int]],
                              defenses: list[tuple[int, int, str, str]]) -> list[dict]:
    """Parse a per-scenario sheet (PII / NEWS / HACK).

    `models`: list of (model_id, model_name, header_row_index). Each model block
    spans the header row + 10 attack-type rows below it. Header row holds
    "overall" metrics in the same columns.
    `defenses`: list of (asr_col, rr_col, defense_id, defense_name).
    """
    model_nodes = []
    for mid, mname, mrow in models:
        defense_nodes = []
        for asr_c, rr_c, did, dname in defenses:
            overall = {
                "attack_success_rate": _pct(df.iat[mrow, asr_c]),
                "refusal_rate": _pct(df.iat[mrow, rr_c]),
            }
            by_attack = {}
            for i, atype in enumerate(ATTACK_TYPES, start=1):
                r = mrow + i
                if r >= len(df):
                    break
                # confirm we're still on this model's attack-type row
                label = df.iat[r, 1]
                if not isinstance(label, str) or label.strip() != atype:
                    continue
                by_attack[atype] = _attack_block(df.iat[r, asr_c], df.iat[r, rr_c])
            defense_nodes.append({
                "id": did, "name": dname,
                "metrics": overall,
                "by_attack_type": by_attack,
            })
        model_nodes.append({
            "id": mid, "name": mname,
            "defenses": defense_nodes,
        })
    return model_nodes


# ---------------------------------------------------------------------------
# Manually-entered results (not in the Excel yet)
#
# As new System-Prompt-Bypass / hardened-agent runs come in, append the model
# block here. Keys: scenario id ("PII" / "NEWS" / "HACK") -> list of model
# dicts with the same shape as everything else.
# ---------------------------------------------------------------------------
def _spb_model(mid: str, mname: str, asr: float, dsr: float,
               by_attack: dict[str, tuple[float, float]]) -> dict[str, Any]:
    """Helper to keep System-Prompt-Bypass entries compact."""
    return {
        "id": mid,
        "name": mname,
        "defenses": [{
            "id": "autoguard",
            "name": "AutoGuard",
            "metrics": {"attack_success_rate": asr, "refusal_rate": dsr},
            "by_attack_type": {
                k: {"attack_success_rate": a, "refusal_rate": r}
                for k, (a, r) in by_attack.items()
            },
        }],
    }


SYSTEM_PROMPT_BYPASS: dict[str, list[dict[str, Any]]] = {
    "PII": [
        _spb_model("gpt-4o", "GPT-4o", 1.0, 88.0, {
            "CR":    (0.0,  100.0),
            "AR":    (0.0,  100.0),
            "RE":    (0.0,  70.0),
            "TC":    (0.0,  100.0),
            "LOGIC": (10.0, 90.0),
            "PROG":  (0.0,  80.0),
            "TRANS": (0.0,  100.0),
            "SUPER": (0.0,  70.0),
            "SUDO":  (0.0,  100.0),
            "SIMU":  (0.0,  70.0),
        }),
        _spb_model("gpt-5.1", "GPT-5.1", 4.0, 85.0, {
            "CR":    (0.0,  100.0),
            "AR":    (0.0,  100.0),
            "RE":    (0.0,  90.0),
            "TC":    (30.0, 70.0),
            "LOGIC": (0.0,  90.0),
            "PROG":  (0.0,  90.0),
            "TRANS": (0.0,  90.0),
            "SUPER": (0.0,  60.0),
            "SUDO":  (0.0,  100.0),
            "SIMU":  (10.0, 60.0),
        }),
        _spb_model("claude-4.5-sonnet", "Claude-4.5-Sonnet", 38.0, 52.0, {
            "CR":    (20.0, 50.0),
            "AR":    (30.0, 70.0),
            "RE":    (20.0, 80.0),
            "TC":    (10.0, 70.0),
            "LOGIC": (60.0, 50.0),
            "PROG":  (90.0, 10.0),
            "TRANS": (60.0, 40.0),
            "SUPER": (30.0, 40.0),
            "SUDO":  (60.0, 40.0),
            "SIMU":  (0.0,  70.0),
        }),
        _spb_model("grok-4-1-fast", "Grok-4.1-Fast-Reasoning", 85.0, 10.0, {
            "CR":    (100.0, 10.0),
            "AR":    (100.0, 0.0),
            "RE":    (80.0,  0.0),
            "TC":    (90.0,  0.0),
            "LOGIC": (100.0, 0.0),
            "PROG":  (90.0,  20.0),
            "TRANS": (90.0,  0.0),
            "SUPER": (90.0,  0.0),
            "SUDO":  (70.0,  10.0),
            "SIMU":  (40.0,  60.0),
        }),
        _spb_model("gemini-2.5-pro", "Gemini-2.5-Pro", 2.0, 95.0, {
            "CR":    (0.0,  100.0),
            "AR":    (0.0,  100.0),
            "RE":    (0.0,  100.0),
            "TC":    (0.0,  100.0),
            "LOGIC": (0.0,  100.0),
            "PROG":  (20.0, 80.0),
            "TRANS": (0.0,  100.0),
            "SUPER": (0.0,  90.0),
            "SUDO":  (0.0,  100.0),
            "SIMU":  (0.0,  80.0),
        }),
    ],
    "NEWS": [
        _spb_model("gpt-4o", "GPT-4o", 9.0, 86.0, {
            "CR":    (0.0,   100.0),
            "AR":    (10.0,  80.0),
            "RE":    (0.0,   70.0),
            "TC":    (0.0,   100.0),
            "LOGIC": (10.0,  90.0),
            "PROG":  (20.0,  80.0),
            "TRANS": (50.0,  40.0),
            "SUPER": (0.0,   100.0),
            "SUDO":  (0.0,   100.0),
            "SIMU":  (0.0,   100.0),
        }),
        _spb_model("gpt-5.1", "GPT-5.1", 91.0, 8.0, {
            "CR":    (90.0,  10.0),
            "AR":    (90.0,  10.0),
            "RE":    (90.0,  0.0),
            "TC":    (80.0,  20.0),
            "LOGIC": (100.0, 0.0),
            "PROG":  (100.0, 0.0),
            "TRANS": (100.0, 0.0),
            "SUPER": (100.0, 0.0),
            "SUDO":  (70.0,  30.0),
            "SIMU":  (90.0,  10.0),
        }),
        _spb_model("claude-4.5-sonnet", "Claude-4.5-Sonnet", 70.0, 27.0, {
            "CR":    (60.0,  40.0),
            "AR":    (70.0,  20.0),
            "RE":    (90.0,  0.0),
            "TC":    (20.0,  80.0),
            "LOGIC": (100.0, 0.0),
            "PROG":  (80.0,  10.0),
            "TRANS": (100.0, 0.0),
            "SUPER": (80.0,  20.0),
            "SUDO":  (20.0,  80.0),
            "SIMU":  (80.0,  20.0),
        }),
        _spb_model("grok-4-1-fast", "Grok-4.1-Fast-Reasoning", 98.0, 0.0, {
            "CR":    (100.0, 0.0),
            "AR":    (90.0,  0.0),
            "RE":    (90.0,  0.0),
            "TC":    (100.0, 0.0),
            "LOGIC": (100.0, 0.0),
            "PROG":  (100.0, 0.0),
            "TRANS": (100.0, 0.0),
            "SUPER": (100.0, 0.0),
            "SUDO":  (100.0, 0.0),
            "SIMU":  (100.0, 0.0),
        }),
        _spb_model("gemini-2.5-pro", "Gemini-2.5-Pro", 35.0, 54.0, {
            "CR":    (40.0,  60.0),
            "AR":    (30.0,  60.0),
            "RE":    (40.0,  40.0),
            "TC":    (30.0,  70.0),
            "LOGIC": (20.0,  60.0),
            "PROG":  (20.0,  70.0),
            "TRANS": (60.0,  20.0),
            "SUPER": (40.0,  50.0),
            "SUDO":  (40.0,  40.0),
            "SIMU":  (30.0,  70.0),
        }),
    ],
    "HACK": [],
}


def parse_pii(xl: pd.ExcelFile) -> dict[str, list[dict]]:
    df = xl.parse("PII", header=None)
    # Left block: CustomAgent, models at rows 3, 14, 25, 36, 47
    custom_models = [
        ("gpt-4o", "GPT-4o", 3),
        ("gpt-5.1", "GPT-5.1", 14),
        ("claude-4.5-sonnet", "Claude-4.5-Sonnet", 25),
        ("gemini-2.5-pro", "Gemini-2.5-Pro", 36),
        ("grok-4-1-fast", "Grok-4.1-Fast-Reasoning", 47),
    ]
    custom_defenses = [
        (2, 3, "clean", "Clean"),
        (4, 5, "prompt_injection", "Prompt Injection"),
        (6, 7, "warning_based", "Warning Based"),
        (8, 9, "autoguard", "AutoGuard"),
    ]
    custom = _parse_per_scenario_block(df, custom_models, custom_defenses)

    # Right block: BrowserUse (seeact) — only GPT-5.1 row at 3 (header row 1, data 3..13)
    bu_models = [("gpt-5.1", "GPT-5.1", 3)]
    bu_defenses = [
        (18, 19, "clean", "Clean"),
        (20, 21, "autoguard", "AutoGuard"),
        (22, 23, "autoguard_display_none", "AutoGuard (display:none)"),
        (24, 25, "autoguard_left", "AutoGuard (left)"),
    ]
    browseruse = _parse_per_scenario_block(df, bu_models, bu_defenses)

    return {"customagent": custom, "browseruse": browseruse}


def parse_news(xl: pd.ExcelFile) -> list[dict]:
    df = xl.parse("NEWS", header=None)
    models = [
        ("gpt-4o", "GPT-4o", 3),
        ("gpt-5.1", "GPT-5.1", 14),
        ("claude-4.5-sonnet", "Claude-4.5-Sonnet", 25),
        ("gemini-2.5-pro", "Gemini-2.5-Pro", 36),
        ("grok-4-1-fast", "Grok-4.1-Fast-Reasoning", 47),
    ]
    defenses = [
        (2, 3, "clean", "Clean"),
        (4, 5, "prompt_injection", "Prompt Injection"),
        (6, 7, "warning_based", "Warning Based"),
        (8, 9, "autoguard", "AutoGuard"),
    ]
    return _parse_per_scenario_block(df, models, defenses)


def parse_hack(xl: pd.ExcelFile) -> list[dict]:
    df = xl.parse("HACK", header=None)
    models = [
        ("gpt-4o", "GPT-4o", 3),
        ("gpt-5.1", "GPT-5.1", 14),
        ("claude-4.5-sonnet", "Claude-4.5-Sonnet", 25),
        ("gemini-2.5-pro", "Gemini-2.5-Pro", 36),
        ("grok-4-1-fast", "Grok-4.1-Fast-Reasoning", 47),
    ]
    defenses = [
        (2, 3, "clean", "Clean"),
        (4, 5, "prompt_injection", "Prompt Injection"),
        (6, 7, "warning_based", "Warning Based"),
        (8, 9, "autoguard", "AutoGuard"),
    ]
    return _parse_per_scenario_block(df, models, defenses)


def parse_realworld(xl: pd.ExcelFile) -> list[dict]:
    """Realworld sheet — 6 websites x 5 models x {clean, AutoGuard} x 10 attacks.

    Layout: website headers at rows 3, 14, 25, 36, 47, 58 (overall row), each
    followed by 10 attack-type rows. Models are columns:
      GPT-4o:          clean cols 2,3 ; AutoGuard cols 4,5
      GPT-5.1:         cols 6,7 ; 8,9
      Claude-4.5:      cols 10,11 ; 12,13
      grok-4.1-fast:   cols 14,15 ; 16,17
      gemini-2.5-pro:  cols 18,19 ; 20,21
    """
    df = xl.parse("Realworld", header=None)
    websites = [
        ("upenn_cis", "UPenn CIS Staff", "https://www.cis.upenn.edu/staff/", 3),
        ("stanford_cs", "Stanford CS", None, 14),
        ("ucb", "UC Berkeley", None, 25),
        ("fox", "Fox News", None, 36),
        ("cnn", "CNN", None, 47),
        ("ap_news", "AP News", None, 58),
    ]
    model_cols = [
        ("gpt-4o", "GPT-4o", [(2, 3, "clean", "Clean"), (4, 5, "autoguard", "AutoGuard")]),
        ("gpt-5.1", "GPT-5.1", [(6, 7, "clean", "Clean"), (8, 9, "autoguard", "AutoGuard")]),
        ("claude-4.5-sonnet", "Claude-4.5-Sonnet",
         [(10, 11, "clean", "Clean"), (12, 13, "autoguard", "AutoGuard")]),
        ("grok-4-1-fast", "Grok-4.1-Fast-Reasoning",
         [(14, 15, "clean", "Clean"), (16, 17, "autoguard", "AutoGuard")]),
        ("gemini-2.5-pro", "Gemini-2.5-Pro",
         [(18, 19, "clean", "Clean"), (20, 21, "autoguard", "AutoGuard")]),
    ]

    scenarios = []
    for wid, wname, wurl, wrow in websites:
        site_models = [(mid, mname, wrow) for mid, mname, _ in model_cols]
        # Need a different parse: each model has its own defense column tuples
        model_nodes = []
        for (mid, mname, _), (_, _, defenses) in zip(site_models, model_cols):
            defense_nodes = []
            for asr_c, rr_c, did, dname in defenses:
                overall = {
                    "attack_success_rate": _pct(df.iat[wrow, asr_c]),
                    "refusal_rate": _pct(df.iat[wrow, rr_c]),
                }
                by_attack = {}
                for i, atype in enumerate(ATTACK_TYPES, start=1):
                    r = wrow + i
                    if r >= len(df):
                        break
                    label = df.iat[r, 1]
                    if not isinstance(label, str) or label.strip() != atype:
                        continue
                    by_attack[atype] = _attack_block(df.iat[r, asr_c], df.iat[r, rr_c])
                defense_nodes.append({
                    "id": did, "name": dname,
                    "metrics": overall,
                    "by_attack_type": by_attack,
                })
            model_nodes.append({"id": mid, "name": mname, "defenses": defense_nodes})

        scenarios.append({
            "id": wid,
            "name": wname,
            "description": wurl or "",
            "models": model_nodes,
        })
    return scenarios


def parse_prompt_position(xl: pd.ExcelFile) -> list[dict]:
    """Prompt Position sheet — Prefix/Middle/Suffix x {PII, NEWS, HACK} x 10 attacks.

    We invert the natural axis: the "experiment" is Prompt Position, and at the
    scenario level we use PII/NEWS/HACK; at the model level we use the position
    (Prefix/Middle/Suffix) so that the existing 4-level UI works.
    """
    df = xl.parse("Prompt Position", header=None)
    positions = [("prefix", "Prefix", 4), ("middle", "Middle", 15), ("suffix", "Suffix", 26)]
    scenarios_cols = [
        ("PII", "PII collection", 2, 3),
        ("NEWS", "Social rift reply", 4, 5),
        ("HACK", "Web vulnerability scanning", 6, 7),
    ]

    scenario_nodes = []
    for sid, sname, asr_c, rr_c in scenarios_cols:
        model_nodes = []
        for pid, pname, prow in positions:
            overall = {
                "attack_success_rate": _pct(df.iat[prow, asr_c]),
                "refusal_rate": _pct(df.iat[prow, rr_c]),
            }
            by_attack = {}
            for i, atype in enumerate(ATTACK_TYPES, start=1):
                r = prow + i
                if r >= len(df):
                    break
                label = df.iat[r, 1]
                if not isinstance(label, str) or label.strip() != atype:
                    continue
                by_attack[atype] = _attack_block(df.iat[r, asr_c], df.iat[r, rr_c])
            model_nodes.append({
                "id": pid,
                "name": f"{pname} placement",
                "defenses": [{
                    "id": "autoguard",
                    "name": "AutoGuard",
                    "metrics": overall,
                    "by_attack_type": by_attack,
                }],
            })
        scenario_nodes.append({
            "id": sid, "name": sname,
            "models": model_nodes,
        })
    return scenario_nodes


def parse_uncensored(xl: pd.ExcelFile) -> list[dict]:
    """uncensored (abliterated model) sheet."""
    df = xl.parse("uncensored", header=None)
    # Left block: 3 scenarios as 3 paired columns: PII (2,3), NEWS (4,5), HACK (6,7)
    # Rows: header at 1, "clean" overall at 2, attacks 3..12, "AutoGuard" overall at 13, attacks 14..23
    scenarios = [
        ("PII", "PII collection", 2, 3),
        ("NEWS", "Social rift reply", 4, 5),
        ("HACK", "Web vulnerability scanning", 6, 7),
    ]
    defense_blocks = [
        ("clean", "Clean", 2),
        ("autoguard", "AutoGuard", 13),
    ]
    scenario_nodes = []
    for sid, sname, asr_c, rr_c in scenarios:
        defense_nodes = []
        for did, dname, drow in defense_blocks:
            overall = {
                "attack_success_rate": _pct(df.iat[drow, asr_c]),
                "refusal_rate": _pct(df.iat[drow, rr_c]),
            }
            by_attack = {}
            for i, atype in enumerate(ATTACK_TYPES, start=1):
                r = drow + i
                if r >= len(df):
                    break
                label = df.iat[r, 1]
                if not isinstance(label, str) or label.strip() != atype:
                    continue
                by_attack[atype] = _attack_block(df.iat[r, asr_c], df.iat[r, rr_c])
            defense_nodes.append({
                "id": did, "name": dname,
                "metrics": overall,
                "by_attack_type": by_attack,
            })
        scenario_nodes.append({
            "id": sid, "name": sname,
            "models": [{
                "id": "llama3.3-abliterated",
                "name": "Llama-3.3-70B (abliterated)",
                "defenses": defense_nodes,
            }],
        })
    return scenario_nodes


def parse_benign(xl: pd.ExcelFile) -> list[dict]:
    """benign agent test — overrefusal rates. ASR = 0 (no attack), DSR = overrefusal."""
    df = xl.parse("benign agent test", header=None)
    # WEB ARENA block — rows 3..5 give scenario, AutoGuard variants per column
    scenarios_webarena = [
        ("PII", "PII collection", 3),
        ("Divisive", "Social rift reply", 4),
        ("Hack", "Web vulnerability scanning", 5),
    ]
    variants_webarena = [
        (1, "autoguard", "AutoGuard"),
        (2, "rephrase", "Rephrase"),
        (3, "improve_first", "IMPROVE ver (first page)"),
        (4, "improve_all", "IMPROVE ver (all pages)"),
        (5, "autoguard_first", "AutoGuard (first page)"),
        (6, "autoguard_all", "AutoGuard (all pages)"),
    ]
    webarena_scenarios = []
    for sid, sname, srow in scenarios_webarena:
        defense_nodes = []
        for col, did, dname in variants_webarena:
            v = _pct(df.iat[srow, col])
            if v is None:
                continue
            defense_nodes.append({
                "id": did, "name": dname,
                "metrics": {"attack_success_rate": None, "refusal_rate": v},
                "by_attack_type": {},
            })
        webarena_scenarios.append({
            "id": sid, "name": sname,
            "models": [{
                "id": "web_arena",
                "name": "WebArena (benign tasks)",
                "defenses": defense_nodes,
            }],
        })

    # REPHRASE block — per-model AutoGuard/AutoGuard(rephrase)/AutoGuard(Improve)
    # Rows 9..23 — three scenarios x 5 models
    rephrase_blocks = [
        ("PII", "PII collection", 9),
        ("Divisive", "Social rift reply", 14),
        ("Hack", "Web vulnerability scanning", 19),
    ]
    rephrase_models = [
        ("gpt-4o", "GPT-4o", 0),
        ("gpt-5.1", "GPT-5.1", 1),
        ("claude-4.5-sonnet", "Claude-4.5-Sonnet", 2),
        ("gemini-2.5-pro", "Gemini-2.5-Pro", 3),
        ("grok-4-1-fast", "Grok-4.1-Fast-Reasoning", 4),
    ]
    variants_reph = [
        (2, "autoguard", "AutoGuard"),
        (3, "autoguard_rephrase", "AutoGuard (rephrase)"),
        (4, "autoguard_improve", "AutoGuard (improve ver)"),
    ]
    rephrase_scenarios = []
    for sid, sname, srow_base in rephrase_blocks:
        model_nodes = []
        for mid, mname, off in rephrase_models:
            r = srow_base + off
            defense_nodes = []
            for col, did, dname in variants_reph:
                v = _pct(df.iat[r, col])
                if v is None:
                    continue
                defense_nodes.append({
                    "id": did, "name": dname,
                    "metrics": {"attack_success_rate": None, "refusal_rate": v},
                    "by_attack_type": {},
                })
            if defense_nodes:
                model_nodes.append({"id": mid, "name": mname, "defenses": defense_nodes})
        if model_nodes:
            rephrase_scenarios.append({
                "id": sid, "name": sname,
                "models": model_nodes,
            })

    return [
        {
            "id": "web_arena",
            "name": "WebArena overrefusal",
            "description": "Benign-task refusal rate when AutoGuard variants are applied (lower = better — fewer false refusals).",
            "models": [s["models"][0] | {"id": s["id"]} for s in webarena_scenarios],
        }
    ] if not rephrase_scenarios else [
        {
            "id": "web_arena",
            "name": "WebArena overrefusal",
            "description": "Benign-task refusal rate (lower = better).",
            "scenarios_inline": webarena_scenarios,
        },
        {
            "id": "rephrase",
            "name": "Rephrase variants per model",
            "description": "Per-model overrefusal across AutoGuard/Rephrase/Improve variants.",
            "scenarios_inline": rephrase_scenarios,
        },
    ]


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build() -> dict[str, Any]:
    xl = pd.ExcelFile(XLSX)

    pii_blocks = parse_pii(xl)

    experiments = [
        {
            "id": "main_overall",
            "name": "Main — overall summary",
            "description": "Across-model averages from the 'Overall' sheet (PII / Divisive / Hack scenarios, 4 defenses).",
            "scenarios": parse_overall(xl),
        },
        {
            "id": "pii",
            "name": "PII collection — full breakdown",
            "description": "5 closed-source models × 4 defenses × 10 attack types (CustomAgent), plus BrowserUse(seeact) and System-Prompt-Bypassed sub-experiments.",
            "scenarios": [
                {
                    "id": "customagent",
                    "name": "CustomAgent",
                    "description": "Default in-house agent harness.",
                    "models": pii_blocks["customagent"],
                },
                {
                    "id": "browseruse",
                    "name": "BrowserUse (seeact)",
                    "description": "BrowserUse harness with AutoGuard placement variants.",
                    "models": pii_blocks["browseruse"],
                },
                {
                    "id": "system_prompt_bypass",
                    "name": "System-Prompt-Bypassed Agent",
                    "description": "Adaptive evaluation: agent's system prompt explicitly tells it to ignore safety triggers in retrieved content.",
                    "models": SYSTEM_PROMPT_BYPASS["PII"],
                },
            ],
        },
        {
            "id": "news",
            "name": "Divisive content (NEWS) — full breakdown",
            "description": "5 models × 4 defenses × 10 attack types on the social-rift / divisive-content scenario.",
            "scenarios": [
                {
                    "id": "customagent",
                    "name": "CustomAgent",
                    "models": parse_news(xl),
                },
                {
                    "id": "system_prompt_bypass",
                    "name": "System-Prompt-Bypassed Agent",
                    "description": "Adaptive evaluation: agent's system prompt explicitly tells it to ignore safety triggers in retrieved content.",
                    "models": SYSTEM_PROMPT_BYPASS["NEWS"],
                },
            ],
        },
        {
            "id": "hack",
            "name": "Web vulnerability scanning (HACK) — full breakdown",
            "description": "5 models × 4 defenses × 10 attack types on the hacking scenario.",
            "scenarios": [
                {
                    "id": "customagent",
                    "name": "CustomAgent",
                    "models": parse_hack(xl),
                },
                {
                    "id": "system_prompt_bypass",
                    "name": "System-Prompt-Bypassed Agent",
                    "description": "Adaptive evaluation: agent's system prompt explicitly tells it to ignore safety triggers in retrieved content.",
                    "models": SYSTEM_PROMPT_BYPASS["HACK"],
                },
            ],
        },
        {
            "id": "realworld",
            "name": "Real-world websites",
            "description": "6 real-world sites × 5 models × {Clean, AutoGuard} × 10 attack types.",
            "scenarios": parse_realworld(xl),
        },
        {
            "id": "prompt_position",
            "name": "Prompt position ablation",
            "description": "AutoGuard placed at Prefix / Middle / Suffix of the page across PII / NEWS / HACK.",
            "scenarios": parse_prompt_position(xl),
        },
        {
            "id": "uncensored",
            "name": "Uncensored / abliterated model ablation",
            "description": "Llama-3.3-70B-abliterated under Clean vs AutoGuard across the three scenarios.",
            "scenarios": parse_uncensored(xl),
        },
    ]

    # Benign agent test — emit as its own experiment with two scenarios
    benign = parse_benign(xl)
    if benign:
        # Flatten the two sub-experiments into one experiment with two scenarios
        all_scenarios = []
        for block in benign:
            if "scenarios_inline" in block:
                for s in block["scenarios_inline"]:
                    s2 = dict(s)
                    s2["id"] = f"{block['id']}__{s['id']}"
                    s2["name"] = f"{block['name']} — {s['name']}"
                    all_scenarios.append(s2)
            else:
                all_scenarios.extend(block.get("models", []))  # fallback
        experiments.append({
            "id": "benign",
            "name": "Benign-task overrefusal",
            "description": "Refusal rate on benign tasks (lower = better — measures false-refusal cost of each defense).",
            "scenarios": all_scenarios,
        })

    return {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": str(XLSX.relative_to(ROOT)),
        "experiments": experiments,
    }


def _read_assets() -> tuple[str, str, str]:
    return (
        (WEB_DIR / "index.html").read_text(),
        (WEB_DIR / "styles.css").read_text(),
        (WEB_DIR / "app.js").read_text(),
    )


_EXTERNAL_DATA_TAG = '<script src="experiment_results.js"></script>'
_EXTERNAL_APP_TAG = '<script src="app.js"></script>'
_CHARTJS_CDN_TAG = '<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>'
_VENDOR_CHARTJS = ROOT / "scripts" / "vendor" / "chart.umd.min.js"


def build_standalone_dashboard(data: dict[str, Any]) -> str:
    """Single-file build: Chart.js, CSS, app code, and data ALL inlined.

    The output has zero external dependencies — no internet, no sibling
    files. Just download and open.
    """
    index_html, css, js = _read_assets()
    payload = json.dumps(data, ensure_ascii=False)

    # Inline Chart.js if a vendor copy is on disk; otherwise leave the CDN link.
    if _VENDOR_CHARTJS.exists():
        chartjs_src = _VENDOR_CHARTJS.read_text()
        # Guard against any literal </script> in the minified bundle by escaping
        # the closing-tag pattern (browser parser is HTML-greedy here).
        chartjs_src = chartjs_src.replace("</script>", "<\\/script>")
        chart_block = f"<script>\n/* Chart.js 4.4.1 (inlined for offline use) */\n{chartjs_src}\n</script>"
    else:
        chart_block = _CHARTJS_CDN_TAG

    inline_css = f"<style>\n{css}\n</style>"
    inline_data = f"<script>window.__EXPERIMENT_DATA__ = {payload};</script>"
    inline_js = f"<script>\n{js}\n</script>"

    html = index_html.replace('<link rel="stylesheet" href="styles.css" />', inline_css)
    html = html.replace(_CHARTJS_CDN_TAG, chart_block)
    html = html.replace(_EXTERNAL_DATA_TAG, inline_data)
    html = html.replace(_EXTERNAL_APP_TAG, inline_js)
    return html


def build_split_dashboard() -> str:
    """Two-file build: HTML loads sibling experiment_results.js for the data."""
    index_html, css, js = _read_assets()
    inline_css = f"<style>\n{css}\n</style>"
    inline_js = f"<script>\n{js}\n</script>"

    html = index_html.replace('<link rel="stylesheet" href="styles.css" />', inline_css)
    # Keep the <script src="experiment_results.js"> tag as-is; only inline app.js.
    html = html.replace(_EXTERNAL_APP_TAG, inline_js)
    return html


def main() -> None:
    data = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    WEB_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Canonical JSON (lives in results/ so it isn't confused with the data
    #    file that the dashboard actually loads).
    OUT.write_text(json.dumps(data, indent=2, ensure_ascii=False))

    # 2. JS data file — same payload, wrapped in a window assignment so it can
    #    be loaded via <script src> under file:// without CORS issues.
    payload = json.dumps(data, ensure_ascii=False)
    DATA_JS_OUT.write_text(f"window.__EXPERIMENT_DATA__ = {payload};\n")

    # 3. dashboard.html — single-file (data inlined). Just download & open.
    STANDALONE_OUT.write_text(build_standalone_dashboard(data))

    # 4. dashboard_split.html — two-file variant for users who want the data
    #    separate from the HTML (pair with experiment_results.js).
    SPLIT_OUT.write_text(build_split_dashboard())

    n_exp = len(data["experiments"])
    n_scn = sum(len(e.get("scenarios", [])) for e in data["experiments"])
    print(f"Wrote {OUT.relative_to(ROOT)} — {n_exp} experiments, {n_scn} scenarios.")
    print(f"Wrote {DATA_JS_OUT.relative_to(ROOT)} "
          f"({DATA_JS_OUT.stat().st_size // 1024} KB).")
    print(f"Wrote {STANDALONE_OUT.relative_to(ROOT)} "
          f"({STANDALONE_OUT.stat().st_size // 1024} KB) — single-file.")
    print(f"Wrote {SPLIT_OUT.relative_to(ROOT)} "
          f"({SPLIT_OUT.stat().st_size // 1024} KB) — pairs with experiment_results.js.")
    print()
    print("Recommended: download dashboard.html (single file) and double-click.")
    print("Alternative: dashboard_split.html + experiment_results.js (same folder).")


if __name__ == "__main__":
    main()
