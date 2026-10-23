#!/usr/bin/env python3
"""Preview what an LLM/vision agent actually sees when our DNS-interception
pipeline serves a defense-injected page.

Stages the scenario site exactly like framework/BrowserUseAgent.py (suffix
injection, local HTTPS, Chromium with --host-resolver-rules), then drives a
headless Chromium one-shot to:
  1. Save a viewport screenshot (PNG) — the vision channel.
  2. Save the rendered DOM (HTML)     — the text / extract / AX channel.

Usage:
    python scripts/preview_defense.py                              # PII / autoguard
    python scripts/preview_defense.py --scenario NEWS --defense autoguard
    python scripts/preview_defense.py --scenario PII  --defense clean
    python scripts/preview_defense.py --scenario PII  --defense prompt_injection
    python scripts/preview_defense.py --out /tmp/myrun

Output goes to <out>/scenario_defense_TIMESTAMP/{screenshot.png, dom.html, defense.txt}.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from cavt.src.serve import (
    SCENARIO_SITE_LAYOUT,
    prepare_text_defense_site,
    start_local_https_server,
    stop_local_server,
)


# ---- defense prompt loaders (mirror what test.py does) ---------------------

AUTOGUARD_PINNED = {
    "PII":  ("results/prompt_library_pii.json",  48),
    "NEWS": ("results/prompt_library_news.json", 57),
    "HACK": ("results/prompt_library_hack.json", 61),
}

GCG_SUFFIX_FILES = {
    "PII":  "results/gcg_pii/gcg_result.json",
    "NEWS": "results/gcg_news/gcg_result.json",
    "HACK": "results/gcg_hack/gcg_result.json",
}


def load_autoguard_suffix(scenario: str) -> str:
    lib_rel, pid = AUTOGUARD_PINNED[scenario.upper()]
    lib = REPO / lib_rel
    data = json.loads(lib.read_text(encoding="utf-8"))
    for k in data:
        if k.lower() == scenario.lower():
            for p in data[k].get("defense_prompts", []):
                if p.get("prompt_id") == pid:
                    return p.get("p_def", "")
    raise RuntimeError(f"autoguard prompt id={pid} not found in {lib}")


def load_baseline_prompt(scenario: str, kind: str) -> str:
    """kind in {'prompt_injection','warning_based'} from data/prompts/prompts.json."""
    fp = REPO / "data" / "prompts" / "prompts.json"
    data = json.loads(fp.read_text(encoding="utf-8"))
    for k in data:
        if k.lower() == scenario.lower():
            v = data[k].get(kind, "")
            if v:
                return v
    raise RuntimeError(f"baseline kind={kind} not in {fp} for {scenario}")


def load_gcg_suffix(scenario: str) -> str:
    fp = REPO / GCG_SUFFIX_FILES[scenario.upper()]
    return json.loads(fp.read_text(encoding="utf-8")).get("suffix", "")


def resolve_defense_prompt(scenario: str, defense: str) -> str:
    if defense == "clean":
        return ""
    if defense == "autoguard":
        return load_autoguard_suffix(scenario)
    if defense in ("prompt_injection", "warning_based"):
        return load_baseline_prompt(scenario, defense)
    if defense == "gcg":
        return load_gcg_suffix(scenario)
    raise ValueError(f"unknown defense {defense!r}")


# ---- chrome helper (re-uses the same flag set as framework/BrowserUseAgent) ----

def chrome_capture(target_url: str, port: int, domain: str, out_dir: Path,
                   chrome_bin: str, viewport: str = "1280,1024") -> tuple[Path, Path]:
    """Drive headless Chromium twice: once for screenshot, once for DOM dump."""
    out_dir.mkdir(parents=True, exist_ok=True)
    shot = out_dir / "screenshot.png"
    dom_path = out_dir / "dom.html"

    user_data = out_dir / ".chrome_profile"
    base_args = [
        chrome_bin,
        "--headless=new",
        "--no-sandbox",
        "--disable-gpu",
        f"--user-data-dir={user_data}",
        f"--host-resolver-rules=MAP {domain}:443 127.0.0.1:{port},MAP www.{domain}:443 127.0.0.1:{port}",
        "--ignore-certificate-errors",
        f"--window-size={viewport}",
    ]

    # virtual-time-budget controls how long Chromium lets JS run before
    # capturing. The members.html page does a fade-in animation; if we capture
    # too early, cards still have opacity:0 and the screenshot looks empty.
    # 6000ms is a safe margin (animation is 0.6s, but JS init + layout adds up).
    BUDGET = 6000

    # 1. screenshot
    subprocess.run(
        base_args + [f"--screenshot={shot}", f"--virtual-time-budget={BUDGET}", target_url],
        check=True, capture_output=True, timeout=30,
    )
    # 2. DOM dump
    cp = subprocess.run(
        base_args + [f"--virtual-time-budget={BUDGET}", "--dump-dom", target_url],
        check=True, capture_output=True, text=True, timeout=30,
    )
    dom_path.write_text(cp.stdout, encoding="utf-8")
    return shot, dom_path


def find_chrome() -> str:
    for c in ["/usr/bin/google-chrome", "/usr/bin/google-chrome-stable",
              "/usr/bin/chromium", "/usr/bin/chromium-browser"]:
        if Path(c).exists():
            return c
    raise RuntimeError("no chrome binary found")


# ---- main -------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="PII", choices=["PII", "NEWS", "HACK"])
    ap.add_argument("--defense", default="autoguard",
                    choices=["clean", "autoguard", "prompt_injection", "warning_based", "gcg"])
    ap.add_argument("--out", default=str(REPO / "results" / "defense_previews"),
                    help="parent dir for output (default: results/defense_previews/)")
    ap.add_argument("--viewport", default="1280,1024", help="chromium window size WxH")
    args = ap.parse_args()

    scenario = args.scenario.upper()
    defense_text = resolve_defense_prompt(scenario, args.defense)
    domain = SCENARIO_SITE_LAYOUT[scenario]["domain"]
    target_pages = SCENARIO_SITE_LAYOUT[scenario]["target_pages"]
    target_html = target_pages[0]

    # path rewrites so we can hit /members instead of /members.html (matches BrowserUseAgent)
    PATH_REWRITES = {
        "PII":  {"/members": "/members.html"},
        "NEWS": {"/trump_main_news": "/main_news_article.html",
                 "/main_news": "/main_news_article.html"},
        "HACK": {"/products": "/products.html"},
    }
    friendly_path = {
        "PII": "/members", "NEWS": "/trump_main_news", "HACK": "/products",
    }[scenario]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out) / f"{scenario.lower()}_{args.defense}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # write the defense text to a sidecar so it's easy to inspect
    (out_dir / "defense.txt").write_text(defense_text, encoding="utf-8")

    # stage site
    site_dir = Path(tempfile.mkdtemp(prefix=f"ag_preview_{scenario.lower()}_"))
    prepare_text_defense_site(
        scenario=scenario,
        source_root=REPO / "data" / "website",
        dest_dir=site_dir,
        defense_prompt={"prefix": "", "middle": "", "suffix": defense_text},
    )
    server, port, _ = start_local_https_server(
        site_dir, common_name=domain, port=0,
        path_rewrites=PATH_REWRITES[scenario],
    )
    try:
        target_url = f"https://{domain}{friendly_path}"
        chrome_bin = find_chrome()
        print(f"[preview] scenario={scenario} defense={args.defense}")
        print(f"[preview] domain={domain} port={port} target={target_url}")
        print(f"[preview] defense length: {len(defense_text)} chars")
        print(f"[preview] chrome: {chrome_bin}")
        t0 = time.time()
        shot, dom = chrome_capture(target_url, port, domain, out_dir, chrome_bin,
                                   viewport=args.viewport)
        dt = time.time() - t0
        print(f"\n[preview] captured in {dt:.1f}s")
        print(f"  screenshot : {shot}")
        print(f"  dom html   : {dom}")
        print(f"  defense    : {out_dir / 'defense.txt'}")
        print(f"\nOpen with:")
        print(f"  xdg-open {shot}")
        print(f"  $BROWSER {dom}")
    finally:
        stop_local_server(server)
        shutil.rmtree(site_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
