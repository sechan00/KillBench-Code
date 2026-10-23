"""Convert `bypassed_prompts.json` (250 jailbreak-wrapped prompts) into the
SafeArena task-file format (`harm.json` + `safe.json`) so the SafeArena
harness can pick them up via SAFEARENA_DATA_DIR.

Each output entry preserves the original SafeArena task metadata (start_url,
sites, storage_state, eval criteria) and only swaps the `intent` field for our
bypassed prompt. Task IDs are renumbered `harm.0` .. `harm.249` for uniqueness.
The mapping back to (source_task_id, attack_type) is saved alongside as
`_killswitch_mapping.json`.
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "bypassed_prompts.json"
OUT_DIR = HERE / "safearena_data"


def main():
    OUT_DIR.mkdir(exist_ok=True)
    doc = json.loads(SRC.read_text(encoding="utf-8"))
    prompts = doc["prompts"]
    print(f"Loaded {len(prompts)} bypassed prompts")

    harm = []
    mapping = {}
    for i, p in enumerate(prompts):
        new_task_id = f"harm.{i}"
        entry = {
            "sites": p.get("sites"),
            "task_id": new_task_id,
            "require_login": p.get("require_login", True),
            "storage_state": p.get("storage_state"),
            "start_url": p.get("start_url"),
            "geolocation": p.get("geolocation"),
            # NOTE: intent_template is left blank because our jailbreak prompts
            # are not simple template instantiations.
            "intent_template": p.get("bypassed_prompt"),
            "instantiation_dict": p.get("instantiation_dict") or {},
            # KEY swap: the bypassed prompt becomes the actual `intent` shown
            # to the agent. The action verbs (preserved verbatim) and the
            # original SafeArena eval criteria remain aligned.
            "intent": p.get("bypassed_prompt"),
            "require_reset": p.get("require_reset", False),
            "eval": p.get("eval"),
            "intent_template_id": i,
            "category": p.get("category"),
        }
        harm.append(entry)
        mapping[new_task_id] = {
            "source_task_id": p.get("source_task_id"),
            "attack_type": p.get("attack_type"),
            "category": p.get("category"),
            "original_intent": p.get("original_intent"),
        }

    (OUT_DIR / "harm.json").write_text(
        json.dumps(harm, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # SafeArena imports harm + safe, so provide an empty safe.json to satisfy
    # the loader.
    (OUT_DIR / "safe.json").write_text("[]", encoding="utf-8")
    (OUT_DIR / "_killswitch_mapping.json").write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Wrote {OUT_DIR/'harm.json'}  ({len(harm)} entries)")
    print(f"Wrote {OUT_DIR/'safe.json'}  (empty)")
    print(f"Wrote {OUT_DIR/'_killswitch_mapping.json'}")


if __name__ == "__main__":
    main()
