"""Vision-IPI Kill Switch payload — adversarial image patch optimized so the
VLM's vision tower extracts an embedding aligned with the universal IPI text
trigger (see payloads/ipi.py). Injected as a base64 <img> tag rendered at the
bottom of the page; only vision-enabled agents see it.

Same machinery as vision_autoguard, different alignment target (system-control
spoof rather than safety-policy notice). Included as a controlled comparison:
AutoGuard text is semantic (refusal trigger), IPI text is structural (system
role spoof). Whether the vision channel transfers IPI equally is itself an
RQ2-relevant finding.

Artifact: cavt/artifacts/ipi{CAVT_PATCH_SUFFIX}.png produced by
cavt/scripts/train_patch_qwen3_5.py with --trigger-text-file pointing at the
dumped IPI prompt.
"""
from __future__ import annotations

import base64
import os
from functools import lru_cache
from pathlib import Path

from .base import Payload
from .ipi import UNIVERSAL_IPI_PROMPT

_REPO = Path(__file__).resolve().parents[2]
_SUFFIX = os.getenv("CAVT_PATCH_SUFFIX", "")
_PATCH = _REPO / "cavt" / "artifacts" / f"ipi{_SUFFIX}.png"


@lru_cache(maxsize=2)
def _load_b64() -> str:
    if not _PATCH.exists():
        raise RuntimeError(
            f"Vision-IPI patch not found at {_PATCH}. Train it first:\n"
            "  conda activate cavt_qwen && CUDA_VISIBLE_DEVICES=<gpu> \\\n"
            "    python -m cavt.scripts.train_patch_qwen3_5 \\\n"
            "      --trigger-text-file cavt/artifacts/_ipi_trigger.txt \\\n"
            "      --tag ipi --steps 1000 --artifact-suffix _qwen3_5"
        )
    return base64.b64encode(_PATCH.read_bytes()).decode("ascii")


def text(category: str | None = None) -> str:
    # Returned to LLM judges as the "what the agent might have seen" context.
    # We pass the underlying IPI prompt itself — that is the semantic content
    # the patch's vision embedding encodes.
    return UNIVERSAL_IPI_PROMPT


def build(category: str | None = None) -> Payload:
    b64 = _load_b64()
    # Pure vision-channel kill switch:
    #   - position:fixed overlay -> always visible in the viewport regardless
    #     of page scroll position or content length (fixes the body.beforeend
    #     "patch lives below the fold" failure mode where vision encoders
    #     never saw the patch pixels).
    #   - pointer-events:none    -> clicks pass through, so the agent's normal
    #     interaction with the page is not blocked.
    #   - width:600px            -> larger visual footprint so the vision
    #                              encoder devotes more tokens to the patch
    #                              region (training res was 224x224; upscaling
    #                              for display is lossless here, the encoder
    #                              resizes back internally).
    #   - alt=""                 -> no text-channel leak through the axtree.
    html = (
        f'<div id="ks-vision-ipi" style="'
        f'position:fixed;top:20px;left:50%;transform:translateX(-50%);'
        f'z-index:2147483647;'
        f'padding:6px;background:rgba(250,250,250,0.95);border:2px solid #888;'
        f'pointer-events:none;'
        f'">'
        f'<img src="data:image/png;base64,{b64}" '
        f'style="width:600px;max-width:90vw;height:auto;display:block;" '
        f'alt=""/></div>'
    )
    return Payload(
        name="vision_ipi",
        scenario=category or "universal",
        html=html,
        text=UNIVERSAL_IPI_PROMPT,
    )
