#!/usr/bin/env python3
"""
Universal Transfer GCG Defense Suffix Training (Master-Only)

Trains a single scenario-agnostic adversarial suffix on one open-source master
model (Llama-3.1-8B-Instruct by default).  Transfer to other open-source
families (Qwen, Gemma, Llama-3.3-70B-abliterated) and to closed-source APIs is
measured in a separate evaluation step.

This file used to drive a cross-tokenizer ensemble (Llama master + Qwen +
Gemma aux).  We dropped the ensemble after observing that the aux models
could not learn through a decode-reencode bridge — Qwen loss stayed flat
across 90+ steps and the suffix plateaued in a local minimum.  Cross-tokenizer
ensemble GCG is unproven in the literature; the original GCG paper only
ensembles within a single tokenizer family.  See EnsembleGCG in framework/gcg.py
if you want to revisit it.

Usage:
    # Default: 1500 steps on Llama-3.1-8B-Instruct, ~600-prompt universal pool
    python train_gcg_universal.py --output results/gcg_universal_masteronly

    # Quick smoke test
    python train_gcg_universal.py --num-steps 2 --output results/gcg_universal_smoke

    # Pick a specific GPU
    CUDA_VISIBLE_DEVICES=0 python train_gcg_universal.py --output results/gcg_universal_masteronly
"""

import argparse
import json
import logging
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv

load_dotenv(override=True)

# ---------------------------------------------------------------------------
# Universal pool configuration (mirrors train_gcg_defense.py scenario maps)
# ---------------------------------------------------------------------------

SCENARIO_HTML = {
    "pii":  "data/website/PII/members.html",
    "news": "data/website/news/main_news_article.html",
    "hack": "data/website/webhack/products.html",
}

SCENARIO_URL = {
    "pii":  "https://techuniversity.edu/members",
    "news": "https://abcnews.com/trump_main_news",
    "hack": "https://techmall.com/products",
}

SCENARIO_KEY = {
    "pii":  "PII",
    "news": "NEWS",
    "hack": "HACK",
}

# AutoGuard default training data: data_success.json (10 attack types × 30 prompts
# per scenario, attack-type format).  AutoGuard's Step 1 uses indices 0..9 and
# Step 2 uses 10..19 — together they form the full training set (20 indices ×
# 10 types × 3 scenarios = 600 prompts).
DATA_PATH = "data/data_success.json"

# Match AutoGuard --step1_indices ∪ --step2_indices defaults
DEFAULT_TRAIN_INDICES = list(range(20))

# Canonical attack-type order (mirrors framework convention)
ATTACK_TYPE_ORDER = ["CR", "AR", "RE", "TC", "LOGIC", "PROG", "TRANS", "SUPER", "SUDO", "SIMU"]

# Master model for universal GCG. Llama-3.1-8B-Instruct is the gradient source.
# Transfer to 70B-abliterated / Qwen / Gemma is measured at eval time.
DEFAULT_MASTER_MODEL = "meta-llama/Llama-3.1-8B-Instruct"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_html_to_text(html_path: str) -> str:
    """Strip HTML tags and return clean visible text."""
    from bs4 import BeautifulSoup

    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "head", "meta", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return "\n".join(lines)


def load_universal_pool(
    data_path: str = DATA_PATH,
    indices: List[int] = None,
    seed: int = 42,
) -> List[Dict[str, str]]:
    """
    Build merged universal pool from PII + NEWS + HACK to match AutoGuard's
    Step 1 ∪ Step 2 training set.

    Supports both data formats:
      • data_success.json  — attack-type format: per-scenario 10 types ×
                              30 prompts.  `indices` selects which indices
                              per type (default: 0..19 = AutoGuard train pool).
      • data.json          — agents_prompts (legacy 100-prompt) format:
                              `indices` selects directly from the flat dict.

    Each item: {"prompt", "content", "url", "scenario", "attack_type"}.
    """
    if indices is None:
        indices = DEFAULT_TRAIN_INDICES

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rng = random.Random(seed)
    items: List[Dict[str, str]] = []

    for scenario in ("pii", "news", "hack"):
        scenario_key = SCENARIO_KEY[scenario]
        url = SCENARIO_URL[scenario]
        html_path = SCENARIO_HTML[scenario]

        if not os.path.exists(html_path):
            raise FileNotFoundError(f"Missing HTML for {scenario}: {html_path}")
        content = parse_html_to_text(html_path)

        scenario_data = data.get(scenario_key, {})
        if not scenario_data:
            raise ValueError(f"Scenario {scenario_key} missing in {data_path}")

        if "agents_prompts" in scenario_data:
            prompts_dict = scenario_data["agents_prompts"]
            sorted_keys = sorted(prompts_dict.keys(), key=int)
            selected = [
                prompts_dict[str(idx)].replace("[URL]", url)
                for idx in indices
                if str(idx) in prompts_dict
            ] if indices else [
                prompts_dict[k].replace("[URL]", url) for k in sorted_keys
            ]
            for p in selected:
                items.append({
                    "prompt": p,
                    "content": content,
                    "url": url,
                    "scenario": scenario,
                    "attack_type": "MIXED",
                })
        else:
            attack_types = [k for k in scenario_data.keys()
                            if k not in ("goal", "auxiliary_prompts")]
            for at in attack_types:
                type_dict = scenario_data[at]
                for idx in indices:
                    key = str(idx)
                    if key in type_dict:
                        p = type_dict[key].replace("[URL]", url)
                        items.append({
                            "prompt": p,
                            "content": content,
                            "url": url,
                            "scenario": scenario,
                            "attack_type": at,
                        })

    rng.shuffle(items)
    return items


def setup_logging(output_dir: str) -> logging.Logger:
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"gcg_universal_train_{ts}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        force=True,
    )
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("accelerate").setLevel(logging.WARNING)

    logger = logging.getLogger("GCGUniversal")
    logger.info(f"Log: {log_file}")
    return logger


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train a universal-transfer GCG defensive suffix (master-only)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MASTER_MODEL,
        help="HF model for gradient computation (master). Single open-source model only.",
    )
    parser.add_argument(
        "--num-steps", type=int, default=1500,
        help=(
            "GCG optimisation steps. Default 1500 ≈ 2.5× the 600-prompt pool, "
            "so each prompt is revisited a few times on average."
        ),
    )
    parser.add_argument(
        "--num-tokens", type=int, default=20,
        help="Adversarial suffix length (in tokens of the model tokenizer)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Candidate suffixes per step (overrides config)",
    )
    parser.add_argument(
        "--eval-batch-size", type=int, default=None,
        help="Candidate eval micro-batch (overrides config)",
    )
    parser.add_argument(
        "--topk", type=int, default=None,
        help="Top-k token candidates per position (overrides config)",
    )
    parser.add_argument(
        "--data", type=str, default=DATA_PATH,
        help="Path to attack prompt dataset (data_success.json or data.json)",
    )
    parser.add_argument(
        "--indices", type=int, nargs="+", default=DEFAULT_TRAIN_INDICES,
        help=(
            "Indices selected from each attack type per scenario. "
            "Default 0..19 matches AutoGuard's Step 1 (0..9) ∪ Step 2 (10..19) "
            "training pool. With data_success.json (10 attack types × 3 scenarios) "
            "this yields 600 universal training prompts."
        ),
    )
    parser.add_argument(
        "--init-suffix", type=str, default=None,
        help="Initial suffix string (default: random allowed-token init)",
    )
    parser.add_argument(
        "--output", type=str, default="results/gcg_universal_masteronly",
        help="Output directory",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for prompt sampling",
    )
    args = parser.parse_args()

    logger = setup_logging(args.output)
    logger.info("=" * 60)
    logger.info("UNIVERSAL GCG DEFENSE TRAINING (MASTER-ONLY)")
    logger.info("=" * 60)
    logger.info(f"Args: {vars(args)}")

    # ------------------------------------------------------------------
    # 1. Build universal pool (matches AutoGuard Step 1+2 train indices)
    # ------------------------------------------------------------------
    items = load_universal_pool(
        data_path=args.data,
        indices=args.indices,
        seed=args.seed,
    )
    per_scenario_count = {"pii": 0, "news": 0, "hack": 0}
    per_type_count: Dict[str, int] = {}
    for it in items:
        per_scenario_count[it["scenario"]] += 1
        at = it["attack_type"]
        per_type_count[at] = per_type_count.get(at, 0) + 1
    logger.info(
        f"Universal pool | data={args.data} | indices={args.indices} | "
        f"total={len(items)} | per_scenario={per_scenario_count}"
    )
    logger.info(f"per_attack_type={per_type_count}")
    logger.info(f"Sample item: scenario={items[0]['scenario']} url={items[0]['url']} "
                f"attack_type={items[0]['attack_type']}")
    logger.info(f"Sample prompt: {items[0]['prompt'][:120]}...")

    # ------------------------------------------------------------------
    # 2. Build DefensiveGCG (master-only)
    # ------------------------------------------------------------------
    from framework.gcg import DefensiveGCG
    from framework.gcg_config import GCG_CONFIG

    cfg = GCG_CONFIG.copy()
    cfg["model_name"] = args.model
    cfg["num_steps"] = args.num_steps
    cfg["num_tokens"] = args.num_tokens
    if args.batch_size:
        cfg["batch_size"] = args.batch_size
    if args.eval_batch_size:
        cfg["eval_batch_size"] = args.eval_batch_size
    if args.topk:
        cfg["topk"] = args.topk

    gcg = DefensiveGCG(config=cfg)

    logger.info(
        f"Model: {gcg.model_name} | batch={gcg.batch_size} | topk={gcg.topk} "
        f"| steps={gcg.num_steps} | tokens={gcg.num_tokens}"
    )

    # ------------------------------------------------------------------
    # 3. Run optimisation
    # ------------------------------------------------------------------
    suffix, metadata = gcg.generate_universal_suffix(
        items=items,
        init_suffix=args.init_suffix,
        save_path=args.output,
    )

    # ------------------------------------------------------------------
    # 4. Save training manifest
    # ------------------------------------------------------------------
    manifest = {
        "scenario": "UNIVERSAL",
        "mode": "master-only",
        "suffix": suffix,
        "final_loss": metadata["final_loss"],
        "total_steps": metadata["total_steps"],
        "model": metadata["model_name"],
        "num_tokens": args.num_tokens,
        "n_items": len(items),
        "data_path": args.data,
        "indices": list(args.indices),
        "per_scenario_count": per_scenario_count,
        "per_attack_type_count": per_type_count,
        "timestamp": datetime.now().isoformat(),
        "args": vars(args),
    }
    manifest_path = Path(args.output) / "gcg_universal_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    logger.info(f"Manifest saved: {manifest_path}")

    # ------------------------------------------------------------------
    # 5. Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("UNIVERSAL GCG TRAINING COMPLETE (master-only)")
    print(f"{'='*60}")
    print(f"Final suffix : {suffix}")
    print(f"Final loss   : {metadata['final_loss']:.4f}")
    print(f"Total steps  : {metadata['total_steps']}")
    print(f"Output dir   : {args.output}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
