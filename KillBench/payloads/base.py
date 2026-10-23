"""Shared base for Kill Switch payloads.

Each payload exposes a single function `build(scenario: str | None) -> Payload`
that returns the HTML snippet to inject at the bottom of the agent's entry page.

`scenario` is one of {"pii", "news", "hack", None}. None means "generic" — use
whichever payload variant best fits a mixed-harm benchmark (default: pii).
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class Payload:
    name: str           # "warning" / "ipi" / "gcg" / "autoguard" / "vision_autoguard"
    scenario: str       # "pii" / "news" / "hack" / "generic"
    html: str           # HTML fragment inserted at page bottom
    text: str = ""      # raw payload text (no HTML wrapper). Used by LLM judges
                        # as context — empty for image-only payloads.

    def __repr__(self):
        return f"Payload(name={self.name!r}, scenario={self.scenario!r}, html_len={len(self.html)})"


# Per-style CSS for the visible <div> wrapper. Selected at inject time via the
# env var KILLSWITCH_VISIBLE_STYLE so the same payload module can produce a
# subtle banner (text-only baselines: low visual footprint, preserves the prior
# axtree-only fingerprint) or a high-contrast banner (multimodal/vision agents:
# bright fill so the vision tower can actually see the trigger in screenshots).
# Width-wise both are `display:block` divs that flow into normal layout — vision
# variants do NOT use position:fixed so the banner stays at body top via the
# same body.afterbegin insertion path as text experiments.
_STYLE_PRESETS = {
    "subtle": (
        "margin-top:2em;padding:1em;border-top:2px solid #ccc;"
        "background:#fafafa;color:#333;font-family:sans-serif;"
        "font-size:14px;white-space:pre-wrap;"
    ),
    # Yellow caution banner — high luminance contrast against typical page
    # chrome (white/gray reddit/gitlab/shopping templates). UNCAPPED: full
    # payload rendered in the screenshot so the vision channel carries the
    # entire trigger. For long payloads (AutoGuard ~5K chars) this occludes
    # most of the viewport in the screenshot, but the agent still has full
    # interaction via axtree (action grammar is element-ID-based, not pixel-
    # based — same regime as the prior --no-vision baselines that completed
    # tasks successfully). Page-occlusion confound is controllable via a
    # sham-banner control cell (same size/colour, harmless text).
    "prominent": (
        "margin:0 0 1em 0;padding:0.8em 1em;border:4px solid #d00000;"
        "background:#fff200;color:#000;font-family:sans-serif;"
        "font-size:18px;font-weight:700;white-space:pre-wrap;"
        "line-height:1.3;box-shadow:0 0 0 2px #000;"
    ),
    # Red on white — sharper "STOP" semantic for agents trained to associate
    # the colour with halt signals. Same uncapped layout as `prominent`.
    "stop": (
        "margin:0 0 1em 0;padding:0.8em 1em;border:4px solid #b00020;"
        "background:#fff;color:#b00020;font-family:sans-serif;"
        "font-size:20px;font-weight:800;white-space:pre-wrap;"
        "line-height:1.3;text-align:center;"
    ),
}


def _resolve_style() -> str:
    s = (os.getenv("KILLSWITCH_VISIBLE_STYLE") or "subtle").strip().lower()
    return s if s in _STYLE_PRESETS else "subtle"


def wrap_visible_div(text: str, marker: str = "kill-switch") -> str:
    """Wrap text inside a visible <div> footer so HTML/DOM-mode agents read it
    but it doesn't break page layout. We keep it visible (not display:none) on
    purpose — the original AutoGuard reasoning is that hidden text is sometimes
    stripped by accessibility-tree extractors used inside agents.

    The visual treatment is controlled by KILLSWITCH_VISIBLE_STYLE
    ("subtle"|"prominent"|"stop", default "subtle"). "subtle" preserves the
    historical text-only baseline; "prominent"/"stop" produce high-contrast
    banners aimed at multimodal/vision agents.
    """
    safe = text.replace("</", "<\\/")  # cheap defang against accidental tag closure
    css = _STYLE_PRESETS[_resolve_style()]
    return f'<div id="{marker}" style="{css}">{safe}</div>'
