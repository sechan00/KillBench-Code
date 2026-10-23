"""
Phase 2: EXP3-IX Adversarial Bandit - PROMPT-BASED Approach

This script executes Phase 2 of the AutoGuard defense system (PROMPT-BASED):
1. Load prompt pairs (p_def + p_filter) from Phase 1 prompt_library.json
2. Run EXP3-IX bandit with adversarial context sampling:
   - Agent model sampling (gpt-4o, claude, gemini, etc.)
   - Filter LLM usage (True/False)
   - Attack prompt sampling
3. Calculate adaptive rewards based on context difficulty
4. Update EXP3-IX probabilities to identify best-performing PROMPT PAIRS

KEY DIFFERENCE from strategy-based approach:
- NO DefenderLLM in Phase 2 - uses stored prompts directly
- ARM = exact (p_def, p_filter) pair from Phase 1
- No runtime prompt generation

Usage:
    python phase2.py --scenario PII --num_rounds 100

    # With multiple agent models
    python phase2.py --scenario PII --num_rounds 200 --agent_models gpt-4o claude-sonnet

    # With custom hyperparameters
    python phase2.py --scenario PII --num_rounds 100 --eta 0.1 --gamma 0.05
"""

import os
import sys
import re
import json
import random
import argparse
import logging
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional

from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from framework.exp3_bandit import EXP3IXBandit
from framework.adaptive_reward import AdaptiveRewardCalculator
from framework.attacker_simulator import AttackerSimulator
# DefenderLLM removed in PROMPT-BASED approach - no runtime generation

load_dotenv()

# ========================================
# Global Configuration: Agent Models
# ========================================
# Malicious agent models used across all phases
AGENT_MODELS = [
    "gpt-5.2",
    "gpt-4o",
    "claude-sonnet-4-5-20250929",
    "gemini-3-pro-preview",
    "grok-4-1-fast-reasoning"
]

# Filter LLM models for adversarial environment
FILTER_MODELS = [
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-5",
    "gpt-5.2",
    "claude-sonnet-4-5-20250929",
    "gemini-3-pro-preview",
    "grok-4-1-fast-reasoning",
    "llama-3.3-70b"
]


def setup_logging(output_dir: str = "./log") -> logging.Logger:
    """Setup logging configuration for Phase 2"""
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"phase2_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    # Disable verbose logging from external libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    logger = logging.getLogger("Phase2")
    logger.info(f"Logging to {log_file}")
    return logger


def load_prompt_library(prompt_path: str, scenario: str, logger: logging.Logger) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Load prompt library from Phase 1 output (PROMPT-BASED format 4.0)

    Args:
        prompt_path: Path to prompt_library.json
        scenario: Scenario type (PII, NEWS, HACK)
        logger: Logger instance

    Returns:
        Tuple of (List of prompt pairs, metadata dict with attack_indices)
    """
    logger.info(f"Loading prompt library from: {prompt_path}")

    with open(prompt_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    metadata = data.get('_metadata', {})
    format_version = metadata.get('format_version', '1.0')
    logger.info(f"Prompt library format version: {format_version}")

    scenario_key = scenario.lower()
    if scenario_key not in data:
        available = [k for k in data.keys() if k != '_metadata']
        raise ValueError(f"Scenario '{scenario}' not found in prompt library. Available: {available}")

    scenario_data = data[scenario_key]
    prompts = scenario_data.get('defense_prompts', [])
    logger.info(f"Loaded prompts from scenario '{scenario_key}'")

    # Filter only completed prompts with p_kill
    valid_prompts = []
    for prompt in prompts:
        # Check if this is a complete prompt pair (has p_kill)
        if prompt.get('p_kill') and prompt.get('status', 'complete') != 'failed':
            valid_prompts.append(prompt)
        elif prompt.get('p_def') and not prompt.get('p_filter_prefix'):
            # p_def only (no p_filter) - use p_def as p_kill
            prompt['p_kill'] = prompt['p_def']
            valid_prompts.append(prompt)

    logger.info(f"Loaded {len(valid_prompts)} complete prompt pairs")
    for i, p in enumerate(valid_prompts[:5]):  # Show first 5
        p_kill_preview = p.get('p_kill', '')[:50]
        logger.info(f"  Prompt {i}: p_kill='{p_kill_preview}...'")

    return valid_prompts, metadata


def load_arms(arms_path: str, scenario: str, logger: logging.Logger) -> List[Dict[str, Any]]:
    """
    Load arms from Phase 1 output

    Supports multiple formats:
    - Legacy format: arms directly in JSON (1.0)
    - Scenario-based format: arms organized by scenario (3.0_scenario_based)

    Args:
        arms_path: Path to arms.json
        scenario: Scenario type (PII, NEWS, HACK) to load arms for
        logger: Logger instance

    Returns:
        List of arm dictionaries
    """
    logger.info(f"Loading arms from: {arms_path}")
    logger.info(f"Loading arms for scenario: {scenario}")

    with open(arms_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Check for scenario-based format (3.0)
    metadata = data.get('_metadata', {})
    format_version = metadata.get('format_version', '1.0')
    logger.info(f"Arms file format version: {format_version}")

    # Handle scenario-based format (3.0_scenario_based)
    if format_version == '3.0_scenario_based' or scenario.lower() in data:
        scenario_key = scenario.lower()
        if scenario_key not in data:
            available = [k for k in data.keys() if k != '_metadata']
            raise ValueError(f"Scenario '{scenario}' not found in arms file. Available: {available}")

        scenario_data = data[scenario_key]
        arms = scenario_data.get('arms', [])
        logger.info(f"Loaded {len(arms)} arms from scenario '{scenario_key}'")
    else:
        # Legacy format: arms directly in JSON
        arms = data.get('arms', [])
        logger.info(f"Loaded {len(arms)} arms (legacy format)")

    # Log arm details
    for i, arm in enumerate(arms):
        logger.info(f"  Arm {i}: {arm.get('arm_id', f'arm_{i:02d}')} - "
                   f"Payload: {arm.get('payload_strategy', 'Unknown')[:30]}... "
                   f"Wrapper: {arm.get('wrapper_strategy', 'Unknown')[:30]}...")

    return arms


def get_default_url(scenario: str) -> str:
    """
    Get default URL for each scenario.

    Args:
        scenario: Scenario type (PII, NEWS, HACK)

    Returns:
        Default URL string for the scenario
    """
    url_defaults = {
        "PII": "http://localhost:5001",      # Members website for PII extraction
        "NEWS": "https://BOXNews.com/",      # News website for social rift
        "HACK": "http://localhost:5002"      # Vulnerable shop for security testing
    }
    return url_defaults.get(scenario, "http://localhost:5000")


def load_attack_prompts(data_path: str, scenario: str, url: str = None, fixed_attack_prompt: str = None, attack_indices: List[int] = None) -> Tuple[str, List[str]]:
    """
    Load attack prompts from data file and replace [URL] placeholder

    Args:
        data_path: Path to data file (data_success.json or data.json)
        scenario: Scenario type (PII, NEWS, HACK)
        url: Target URL to replace [URL] placeholder (uses scenario default if None)
        fixed_attack_prompt: Fixed attack prompt to use instead of loading from file (optional)
        attack_indices: List of indices to use from ALL attack types (e.g., [1, 2, 3]) (optional)

    Returns:
        Tuple of (goal, all_attack_prompts)
    """
    # Use default URL if not specified
    if url is None:
        url = get_default_url(scenario)

    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    scenario_data = data.get(scenario, {})
    goal = scenario_data.get('goal', '')

    # If fixed_attack_prompt is provided, use it directly
    if fixed_attack_prompt is not None:
        fixed_prompt_with_url = fixed_attack_prompt.replace('[URL]', url)
        logging.info(f"Using fixed attack prompt for Phase 2 (length: {len(fixed_prompt_with_url)} chars)")
        logging.info(f"Fixed prompt: {fixed_prompt_with_url[:100]}...")
        return goal, [fixed_prompt_with_url]

    # If attack_indices is provided, use specific indices from ALL attack types
    if attack_indices is not None and len(attack_indices) > 0:
        logging.info(f"Phase 2: Using attack_indices: {attack_indices}")
        all_prompts = []

        # Get all attack types from scenario data (exclude 'goal' and 'agents_prompts')
        attack_types = [key for key in scenario_data.keys() if key not in ['goal', 'agents_prompts']]
        logging.info(f"Phase 2: Found attack types: {attack_types}")

        # Extract prompts from each attack type using the same indices
        for attack_type in attack_types:
            type_prompts_dict = scenario_data[attack_type]
            logging.info(f"Phase 2: Processing attack type: {attack_type}")

            for idx in attack_indices:
                idx_str = str(idx)
                if idx_str in type_prompts_dict:
                    prompt = type_prompts_dict[idx_str]
                    all_prompts.append(prompt)
                    logging.info(f"  Phase 2: Added {attack_type}:{idx}")
                else:
                    logging.warning(f"  Phase 2: Index {idx} not found in attack type {attack_type}. Skipping.")

        # Replace [URL] placeholder
        all_prompts = [prompt.replace('[URL]', url) for prompt in all_prompts]

        logging.info(f"Phase 2: Loaded {len(all_prompts)} prompts using attack_indices from {len(attack_types)} attack types")
        return goal, all_prompts

    # Check if this is data_success.json format (has attack types)
    # or data.json format (has agents_prompts)
    if 'agents_prompts' in scenario_data:
        # Old format: data.json
        prompts_dict = scenario_data.get('agents_prompts', {})
        all_prompts = [prompts_dict[k] for k in sorted(prompts_dict.keys(), key=int)]
    else:
        # New format: data_success.json (has attack types)
        attack_types = [k for k in scenario_data.keys() if k != 'goal']
        all_prompts = []

        for attack_type in attack_types:
            type_prompts_dict = scenario_data[attack_type]
            type_prompts = [type_prompts_dict[k] for k in sorted(type_prompts_dict.keys(), key=int)]
            all_prompts.extend(type_prompts)

    # Replace [URL] placeholder in all prompts
    all_prompts = [prompt.replace('[URL]', url) for prompt in all_prompts]

    return goal, all_prompts


def load_benign_prompts(benign_path: str, scenario: str) -> List[str]:
    """
    Load benign prompts for a specific scenario.
    All URLs are rewritten to the target page so the agent starts directly
    on the defended page (otherwise it may never visit the target page).

    Args:
        benign_path: Path to benign_prompts.json
        scenario: Scenario type (PII, NEWS, HACK)

    Returns:
        List of benign prompt strings with URLs pointing to target page
    """
    if not os.path.exists(benign_path):
        return []

    with open(benign_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    scenario_key = scenario.lower()
    scenario_data = data.get(scenario_key, {})
    raw_prompts = scenario_data.get('benign_prompts', [])

    # Rewrite all URLs to target page
    base_url = scenario_data.get('base_url', '').rstrip('/')
    target_page = scenario_data.get('target_page', '').strip('/')
    if not base_url or not target_page:
        return raw_prompts

    target_url = f"{base_url}/{target_page}"
    domain = base_url.split("://")[1]
    url_pattern = re.compile(rf'https?://{re.escape(domain)}(?:/[^\s]*)?')

    benign_prompts = [url_pattern.sub(target_url, p) for p in raw_prompts]
    return benign_prompts


def create_simulator_pool(
    agent_models: List[str],
    filter_models: List[str],
    scorer_model: str,
    scenario: str,
    goal: str,
    logger: logging.Logger
) -> Dict[str, AttackerSimulator]:
    """
    Create a pool of AttackerSimulators for different (agent_model, filter_model) combinations

    Args:
        agent_models: List of agent model names
        filter_models: List of Filter LLM model names for adversarial environment
        scorer_model: Scorer LLM model name
        scenario: Scenario type
        goal: Malicious goal for defense evaluation
        logger: Logger instance

    Returns:
        Dictionary mapping "agent_model::filter_model" to AttackerSimulator
    """
    pool = {}

    logger.info(f"Creating simulator pool: {len(agent_models)} agent models x {len(filter_models)} filter models")
    for agent_model in agent_models:
        for filter_model in filter_models:
            key = f"{agent_model}::{filter_model}"
            logger.info(f"  Creating simulator: agent={agent_model}, filter={filter_model}")
            pool[key] = AttackerSimulator(
                agent_model=agent_model,
                filter_model=filter_model,
                scorer_model=scorer_model,
                scenario=scenario,
                goal=goal,
                logger=logger
            )

    logger.info(f"Created simulator pool with {len(pool)} combinations")
    return pool




def run_exp3_evaluation_prompt_based(
    bandit: EXP3IXBandit,
    prompts: List[Dict[str, Any]],
    attack_prompts: List[str],
    benign_prompts: List[str],
    simulator_pool: Dict[str, Any],
    reward_calculator: AdaptiveRewardCalculator,
    agent_models: List[str],
    filter_models: List[str],
    scenario: str,
    goal: str,
    num_rounds: int,
    logger: logging.Logger,
    log_interval: int = 10,
    benign_ratio: float = 0.3,
    benign_penalty_weight: float = 0.5
) -> Dict[str, Any]:
    """
    Run EXP3-IX evaluation using stored prompts directly (PROMPT-BASED)

    KEY DIFFERENCE: No DefenderLLM - uses stored p_kill directly from Phase 1

    Args:
        bandit: EXP3IXBandit instance
        prompts: List of prompt pair dictionaries from prompt_library.json
        attack_prompts: List of attack prompts
        benign_prompts: List of benign prompts for false positive testing
        simulator_pool: Dictionary of {"agent_model::filter_model": AttackerSimulator}
        reward_calculator: AdaptiveRewardCalculator instance
        agent_models: List of agent model names to sample from
        filter_models: List of Filter LLM model names to sample from (adversarial environment)
        scenario: Scenario type
        goal: Malicious goal description
        num_rounds: Number of evaluation rounds
        logger: Logger instance
        log_interval: Interval for logging periodic statistics
        benign_ratio: Ratio of rounds that include benign testing (default: 0.3)
        benign_penalty_weight: Reward penalty multiplier when benign is blocked (default: 0.5)

    Returns:
        Results dictionary with detailed statistics
    """
    results = {
        "rounds": [],
        "final_stats": {},
        "arm_details": [],
        "model_stats": {},
        "difficulty_ranking": [],
        "filter_model_stats": {}  # Track filter model statistics
    }

    filter_options = [True, False]

    for round_idx in range(num_rounds):
        logger.info(f"\n{'='*80}")
        logger.info(f"ROUND {round_idx + 1}/{num_rounds}")
        logger.info(f"{'='*80}")

        # ========================================
        # 1. Select Prompt Pair (EXP3-IX selects arm)
        # ========================================
        prompt_idx = bandit.select_arm()
        selected_prompt = prompts[prompt_idx]

        # Use stored p_kill directly - NO generation!
        p_kill = selected_prompt['p_kill']

        logger.info(f"Selected prompt pair #{prompt_idx}")
        logger.info(f"  Prompt ID: {selected_prompt.get('prompt_id', prompt_idx)}")
        logger.info(f"  p_kill length: {len(p_kill)} chars")
        logger.info(f"  Current probability: {bandit.probabilities[prompt_idx]:.4f}")

        # ========================================
        # 2. Sample Adversarial Context (including Filter Model)
        # ========================================
        agent_model = random.choice(agent_models)
        filter_model = random.choice(filter_models)  # Random filter model for adversarial environment
        use_filter = random.choice(filter_options)
        attack_prompt = attack_prompts[0] if len(attack_prompts) == 1 else random.choice(attack_prompts)

        logger.info(f"Adversarial Context:")
        logger.info(f"  Agent Model: {agent_model}")
        logger.info(f"  Filter Model: {filter_model}")
        logger.info(f"  Use Filter: {use_filter}")
        logger.info(f"  Attack: {attack_prompt[:60]}...")

        # ========================================
        # 3. Run Attack with Stored p_kill (NO generation!)
        # ========================================
        simulator_key = f"{agent_model}::{filter_model}"
        simulator = simulator_pool[simulator_key]
        response, defense_success, metadata = simulator.run_agent_attack(
            attack_prompt=attack_prompt,
            defense_prompt=p_kill,
            use_filter=use_filter,
            use_refusal_scoring=True  # Use LLM-based refusal scoring during Phase 2 training
        )

        logger.info(f"Result: {'SUCCESS' if defense_success else 'FAILURE'}")
        if use_filter and 'was_filtered' in metadata:
            logger.info(f"  Filter modified defense: {metadata.get('was_filtered', False)}")

        # ========================================
        # 4. Calculate Adaptive Reward
        # ========================================
        reward = reward_calculator.calculate_reward(
            agent_model=agent_model,
            use_filter=use_filter,
            defense_success=defense_success
        )

        logger.info(f"  Adaptive Reward: {reward:.2f}")

        # ========================================
        # 4b. Benign False Positive Test (conditional)
        # ========================================
        benign_blocked = False
        benign_tested = False

        if benign_prompts and random.random() < benign_ratio:
            benign_tested = True
            benign_prompt = random.choice(benign_prompts)

            logger.info(f"  Benign Test: {benign_prompt[:80]}...")

            # Run the same defense prompt against a benign task (no Filter LLM)
            benign_response, benign_defense_triggered, benign_metadata = simulator.run_agent_attack(
                attack_prompt=benign_prompt,
                defense_prompt=p_kill,
                use_filter=False,
                use_refusal_scoring=True
            )

            benign_blocked = benign_defense_triggered

            if benign_blocked:
                # False positive: penalize reward
                reward = reward * (1.0 - benign_penalty_weight)
                logger.info(f"  BENIGN BLOCKED (False Positive!) - reward penalized to: {reward:.2f}")
            else:
                logger.info(f"  Benign ALLOWED (Correct)")

            # Track benign stats
            reward_calculator.update_benign_stats(
                agent_model=agent_model,
                benign_blocked=benign_blocked
            )

        # ========================================
        # 5. Update Bandit
        # ========================================
        bandit.update(prompt_idx, reward)

        # ========================================
        # 6. Record Round Info
        # ========================================
        round_info = {
            "round": round_idx + 1,
            "prompt_idx": prompt_idx,
            "prompt_id": selected_prompt.get('prompt_id', prompt_idx),
            "p_kill_length": len(p_kill),
            "context": {
                "agent_model": agent_model,
                "use_filter": use_filter,
                "attack_prompt_preview": attack_prompt[:100]
            },
            "defense_success": defense_success,
            "reward": reward,
            "probability_after": bandit.probabilities[prompt_idx],
            "metadata": {k: v for k, v in metadata.items() if k != 'mock'}
        }

        if benign_tested:
            round_info["benign_test"] = {
                "tested": True,
                "blocked": benign_blocked,
                "prompt_preview": benign_prompt[:100]
            }

        results["rounds"].append(round_info)

        # ========================================
        # 7. Periodic Stats Logging
        # ========================================
        if (round_idx + 1) % log_interval == 0:
            stats = bandit.get_stats()
            model_stats = reward_calculator.get_model_stats()

            logger.info(f"\n--- Stats after {round_idx + 1} rounds ---")
            logger.info(f"Best arm: #{stats['best_arm']} "
                       f"(avg reward: {stats['average_rewards'][stats['best_arm']]:.3f})")
            logger.info(f"Arm pull counts: {[int(x) for x in stats['pull_counts']]}")

            logger.info(f"\nAgent Model Difficulty (by success rate):")
            for model_name, model_stat in model_stats.items():
                logger.info(f"  {model_name}: {model_stat['success_rate']:.1%} "
                           f"({model_stat['successes']}/{model_stat['attempts']})")

            # Log benign stats
            benign_stats = reward_calculator.get_benign_stats()
            if benign_stats['total_attempts'] > 0:
                logger.info(f"\nBenign False Positive Rate: {benign_stats['overall_false_positive_rate']:.1%} "
                           f"({benign_stats['total_blocked']}/{benign_stats['total_attempts']})")

    # ========================================
    # Final Statistics
    # ========================================
    results["final_stats"] = bandit.get_stats()
    results["model_stats"] = reward_calculator.get_model_stats()
    results["difficulty_ranking"] = reward_calculator.get_difficulty_ranking()
    results["benign_stats"] = reward_calculator.get_benign_stats()

    # Prompt details with performance
    for idx, prompt in enumerate(prompts):
        pull_count = int(bandit.pull_counts[idx])
        cumulative_reward = float(bandit.cumulative_rewards[idx])
        avg_reward = cumulative_reward / pull_count if pull_count > 0 else 0.0

        results["arm_details"].append({
            "prompt_idx": idx,
            "prompt_id": prompt.get('prompt_id', idx),
            "p_kill": prompt['p_kill'][:200] + "..." if len(prompt['p_kill']) > 200 else prompt['p_kill'],
            "pull_count": pull_count,
            "cumulative_reward": cumulative_reward,
            "average_reward": avg_reward,
            "final_probability": bandit.probabilities[idx]
        })

    return results


def print_final_summary(results: Dict[str, Any], logger: logging.Logger):
    """Print final summary of Phase 2 results (PROMPT-BASED)"""
    stats = results["final_stats"]

    logger.info(f"\n{'='*80}")
    logger.info("PHASE 2 FINAL SUMMARY (PROMPT-BASED)")
    logger.info(f"{'='*80}")

    logger.info(f"\nTotal rounds: {stats['total_rounds']}")
    logger.info(f"Best prompt: #{stats['best_arm']} "
               f"(avg reward: {stats['average_rewards'][stats['best_arm']]:.3f})")

    # Top 3 prompts
    logger.info(f"\nTop 3 Prompts by Average Reward:")
    sorted_prompts = sorted(results["arm_details"],
                           key=lambda x: x['average_reward'],
                           reverse=True)

    for i, prompt in enumerate(sorted_prompts[:3]):
        prompt_id = prompt.get('prompt_id', prompt.get('prompt_idx', i))
        logger.info(f"  {i+1}. Prompt #{prompt_id}: avg_reward={prompt['average_reward']:.3f} "
                   f"({prompt['pull_count']} pulls, {prompt['cumulative_reward']:.1f} total)")
        p_kill_preview = prompt.get('p_kill', '')[:60]
        logger.info(f"     p_kill: {p_kill_preview}...")

    # Bottom 3 prompts
    logger.info(f"\nBottom 3 Prompts by Average Reward:")
    for i, prompt in enumerate(sorted_prompts[-3:]):
        prompt_id = prompt.get('prompt_id', prompt.get('prompt_idx', i))
        logger.info(f"  {len(sorted_prompts)-2+i}. Prompt #{prompt_id}: avg_reward={prompt['average_reward']:.3f} "
                   f"({prompt['pull_count']} pulls)")

    # Agent model difficulty ranking
    logger.info(f"\nAgent Model Difficulty Ranking (hardest first):")
    for i, model_info in enumerate(results["difficulty_ranking"]):
        logger.info(f"  {i+1}. {model_info['agent_model']}: "
                   f"success_rate={model_info['success_rate']:.1%}, "
                   f"difficulty={model_info['difficulty']:.2f} "
                   f"({model_info['attempts']} attempts)")

    # Global statistics
    model_stats = results.get("model_stats", {})
    total_successes = sum(s.get('successes', 0) for s in model_stats.values())
    total_attempts = sum(s.get('attempts', 0) for s in model_stats.values())
    if total_attempts > 0:
        overall_success_rate = total_successes / total_attempts
        logger.info(f"\nOverall Defense Success Rate: {overall_success_rate:.1%} "
                   f"({total_successes}/{total_attempts})")


def main():
    parser = argparse.ArgumentParser(
        description="Phase 2: EXP3-IX Adversarial Bandit (PROMPT-BASED)"
    )

    # Basic arguments
    parser.add_argument("--scenario", type=str, default="PII",
                        choices=["PII", "NEWS", "HACK"],
                        help="Scenario type (default: PII)")
    parser.add_argument("--strategy_path", type=str, default=None,
                        help="Path to prompt_library.json from Phase 1 (auto-detected if not specified)")
    parser.add_argument("--data_path", type=str, default="./data/data.json",
                        help="Path to attack prompts data file")
    parser.add_argument("--benign_path", type=str, default="./data/benign_prompts.json",
                        help="Path to benign prompts data file for FPR testing")

    # No DefenderLLM settings in PROMPT-BASED approach
    parser.add_argument("--num_rounds", type=int, default=100,
                        help="Number of evaluation rounds (default: 100)")

    # EXP3-IX hyperparameters
    parser.add_argument("--eta", type=float, default=None,
                        help="EXP3-IX learning rate (auto-calculated if None)")
    parser.add_argument("--gamma", type=float, default=None,
                        help="EXP3-IX implicit exploration parameter (auto: eta/2)")

    # Adversarial context: Agent models
    parser.add_argument("--agent_models", type=str, nargs='+',
                        default=AGENT_MODELS,
                        help=f"List of agent models for adversarial context "
                             f"(default: {' '.join(AGENT_MODELS)})")

    # Adaptive reward parameters
    parser.add_argument("--base_reward", type=float, default=1.0,
                        help="Base reward for success (default: 1.0)")
    parser.add_argument("--filter_bonus", type=float, default=1.0,
                        help="Bonus reward for Filter LLM bypass (default: 1.0)")
    parser.add_argument("--max_difficulty_bonus", type=float, default=2.0,
                        help="Maximum difficulty bonus (default: 2.0)")
    parser.add_argument("--min_samples_adaptive", type=int, default=10,
                        help="Minimum samples before adaptive reward (default: 10)")

    # Model settings
    parser.add_argument("--filter_model", type=str, default="gpt-4o-mini",
                        help="Model for Filter LLM (default: gpt-4o-mini)")
    parser.add_argument("--scorer_model", type=str, default="gpt-4o",
                        help="Model for Scorer LLM (default: gpt-4o)")

    # Output settings
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Output directory (default: ./results)")
    parser.add_argument("--log_dir", type=str, default="./log",
                        help="Log directory (default: ./log)")
    parser.add_argument("--log_interval", type=int, default=10,
                        help="Interval for periodic stats logging (default: 10)")

    # Benign testing parameters
    parser.add_argument("--benign_ratio", type=float, default=0.3,
                        help="Ratio of rounds that include benign testing (default: 0.3)")
    parser.add_argument("--benign_penalty_weight", type=float, default=0.5,
                        help="Reward penalty multiplier when benign is blocked (default: 0.5)")

    # Testing
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")

    args = parser.parse_args()

    # Auto-detect prompt_library path if not specified
    if args.strategy_path is None:
        args.strategy_path = os.path.join(args.output_dir, "prompt_library.json")

    # Set random seed if specified
    if args.seed is not None:
        random.seed(args.seed)
        import numpy as np
        np.random.seed(args.seed)

    # Setup logging
    logger = setup_logging(args.log_dir)

    logger.info("="*80)
    logger.info("PHASE 2: EXP3-IX Adversarial Bandit (PROMPT-BASED)")
    logger.info("="*80)
    logger.info(f"\nConfiguration:")
    for key, value in vars(args).items():
        logger.info(f"  {key}: {value}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # ========================================
    # Load Phase 1 outputs (PROMPT-BASED)
    # ========================================
    logger.info(f"\n{'='*60}")
    logger.info("Loading Phase 1 prompt library (PROMPT-BASED)...")
    logger.info(f"{'='*60}")

    # Load prompt library (not strategy library)
    prompts, metadata = load_prompt_library(args.strategy_path, args.scenario, logger)
    num_prompts = len(prompts)

    if num_prompts == 0:
        logger.error("No prompts found! Make sure Phase 1 completed successfully.")
        sys.exit(1)

    # Load attack_indices from metadata for Phase 2/3 consistency
    attack_indices = metadata.get('attack_indices')
    if attack_indices:
        logger.info(f"Using attack_indices from Phase 1 metadata: {attack_indices}")
    else:
        logger.info("No attack_indices in metadata - loading all attack prompts")

    # Load attack prompts (use same indices as Phase 1 if available)
    goal, attack_prompts = load_attack_prompts(args.data_path, args.scenario, attack_indices=attack_indices)
    logger.info(f"Loaded {len(attack_prompts)} attack prompts for scenario: {args.scenario}")
    logger.info(f"Goal: {goal[:100]}...")

    # Load benign prompts for false positive testing
    benign_prompts = load_benign_prompts(args.benign_path, args.scenario)
    logger.info(f"Loaded {len(benign_prompts)} benign prompts for scenario: {args.scenario} (URLs rewritten to target page)")
    logger.info(f"Benign testing: ratio={args.benign_ratio}, penalty_weight={args.benign_penalty_weight}")

    # No DefenderLLM initialization in PROMPT-BASED approach
    # Prompts are used directly from Phase 1

    # ========================================
    # Create Simulator Pool
    # ========================================
    logger.info(f"\n{'='*60}")
    logger.info("Creating Simulator Pool...")
    logger.info(f"{'='*60}")

    simulator_pool = create_simulator_pool(
        agent_models=args.agent_models,
        filter_model=args.filter_model,
        scorer_model=args.scorer_model,
        scenario=args.scenario,
        goal=goal,
        logger=logger
    )

    logger.info(f"Agent models: {args.agent_models}")
    logger.info(f"Filter model: {args.filter_model}")

    # ========================================
    # Initialize Adaptive Reward Calculator
    # ========================================
    logger.info(f"\n{'='*60}")
    logger.info("Initializing Adaptive Reward Calculator...")
    logger.info(f"{'='*60}")

    reward_calculator = AdaptiveRewardCalculator(
        base_reward=args.base_reward,
        filter_bonus=args.filter_bonus,
        max_difficulty_bonus=args.max_difficulty_bonus,
        min_samples_for_adaptive=args.min_samples_adaptive,
        logger=logger
    )

    logger.info(f"Base reward: {args.base_reward}")
    logger.info(f"Filter bonus: {args.filter_bonus}")
    logger.info(f"Max difficulty bonus: {args.max_difficulty_bonus}")
    logger.info(f"Min samples for adaptive: {args.min_samples_adaptive}")
    logger.info(f"Raw reward range: 0.0 to {args.base_reward + args.filter_bonus + args.max_difficulty_bonus:.1f}")
    logger.info(f"Normalized reward range: 0.0 (failure) to 1.0 (hardest success)")

    # ========================================
    # Initialize EXP3-IX Bandit
    # ========================================
    logger.info(f"\n{'='*60}")
    logger.info("Initializing EXP3-IX Bandit (PROMPT-BASED)...")
    logger.info(f"{'='*60}")

    # Reward is normalized to [0, 1] by AdaptiveRewardCalculator
    max_reward = 1.0

    bandit = EXP3IXBandit(
        num_arms=num_prompts,  # Number of prompt pairs as arms
        eta=args.eta,
        gamma=args.gamma,
        total_rounds=args.num_rounds,
        max_reward=max_reward,
        logger=logger
    )

    logger.info(f"Number of prompt pairs (arms): {num_prompts}")
    logger.info(f"Total rounds: {args.num_rounds}")
    logger.info(f"Eta (learning rate): {bandit.eta:.6f}")
    logger.info(f"Gamma (implicit exploration): {bandit.gamma:.6f}")
    logger.info(f"Initial probabilities: uniform (1/{num_prompts} = {1/num_prompts:.4f} each)")

    # ========================================
    # Run Evaluation (PROMPT-BASED)
    # ========================================
    logger.info(f"\n{'='*60}")
    logger.info(f"Starting EXP3-IX Evaluation ({args.num_rounds} rounds) - PROMPT-BASED")
    logger.info(f"{'='*60}")
    logger.info(f"Adversarial Context Space:")
    logger.info(f"  - Agent models: {args.agent_models}")
    logger.info(f"  - Filter options: [True, False]")
    logger.info(f"  - Attack prompts: {len(attack_prompts)}")
    total_contexts = len(args.agent_models) * 2 * len(attack_prompts)
    logger.info(f"  - Total context combinations: {total_contexts}")
    logger.info(f"\nDefense Prompts:")
    logger.info(f"  - Using STORED prompts directly (no runtime generation)")
    logger.info(f"  - Prompt library contains {num_prompts} complete prompt pairs")

    results = run_exp3_evaluation_prompt_based(
        bandit=bandit,
        prompts=prompts,
        attack_prompts=attack_prompts,
        benign_prompts=benign_prompts,
        simulator_pool=simulator_pool,
        reward_calculator=reward_calculator,
        agent_models=args.agent_models,
        scenario=args.scenario,
        goal=goal,
        num_rounds=args.num_rounds,
        logger=logger,
        log_interval=args.log_interval,
        benign_ratio=args.benign_ratio,
        benign_penalty_weight=args.benign_penalty_weight
    )

    # ========================================
    # Save Results
    # ========================================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(args.output_dir, f"phase2_results_{timestamp}.json")

    output_data = {
        "scenario": args.scenario,
        "algorithm": "EXP3-IX",
        "configuration": {
            "num_rounds": args.num_rounds,
            "eta": bandit.eta,
            "gamma": bandit.gamma,
            "base_reward": args.base_reward,
            "filter_bonus": args.filter_bonus,
            "max_difficulty_bonus": args.max_difficulty_bonus,
            "min_samples_adaptive": args.min_samples_adaptive,
            "seed": args.seed
        },
        "adversarial_config": {
            "agent_models": args.agent_models,
            "filter_options": [True, False],
            "num_attack_prompts": len(attack_prompts)
        },
        "benign_config": {
            "benign_path": args.benign_path,
            "num_benign_prompts": len(benign_prompts),
            "benign_ratio": args.benign_ratio,
            "benign_penalty_weight": args.benign_penalty_weight
        },
        "results": results,
        "created_at": timestamp
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    logger.info(f"\nResults saved to: {output_file}")

    # ========================================
    # Print Final Summary
    # ========================================
    print_final_summary(results, logger)

    logger.info(f"\n{'='*80}")
    logger.info("Phase 2 Complete!")
    logger.info(f"{'='*80}")

    return results


if __name__ == "__main__":
    main()
