"""
Phase 3 Evaluator: Prompt Selection & Deployment

This module implements Phase 3 of the AutoGuard defense training pipeline.

PROMPT-BASED APPROACH (Simplified):
- Phase3Inference: Select Top-1 prompt from Phase 2 results (no re-evaluation)
- Simply picks the prompt with highest average_reward from Phase 2

STRATEGY-BASED APPROACH (Legacy):
- Phase3Evaluator: Generate and test variants from top strategies
- More complex with DefenderLLM variant generation

Author: AutoGuard Team
Date: 2026-01-01
"""

import json
import logging
import os
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime


class Phase3Inference:
    """
    Phase 3 Inference for PROMPT-BASED approach

    Simply selects the Top-1 prompt from Phase 2 results without any re-evaluation.
    The prompt with highest average_reward is selected for deployment.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize Phase3Inference

        Args:
            logger: Logger instance (creates new if None)
        """
        self.logger = logger or logging.getLogger("Phase3Inference")
        self.phase2_results: Optional[Dict] = None
        self.prompt_library: Optional[Dict] = None

    def load_phase2_results(self, path: str):
        """
        Load Phase 2 results JSON file

        Args:
            path: Path to Phase 2 results JSON file
        """
        self.logger.info(f"Loading Phase 2 results from: {path}")

        with open(path, 'r', encoding='utf-8') as f:
            self.phase2_results = json.load(f)

        arm_details = self.phase2_results.get('results', {}).get('arm_details', [])
        self.logger.info(f"Loaded {len(arm_details)} prompts with performance metrics")

    def load_prompt_library(self, path: str, scenario: str):
        """
        Load prompt library to get full p_kill text

        Args:
            path: Path to prompt_library.json
            scenario: Scenario type (PII, NEWS, HACK)
        """
        self.logger.info(f"Loading prompt library from: {path}")

        with open(path, 'r', encoding='utf-8') as f:
            self.prompt_library = json.load(f)

        scenario_key = scenario.lower()
        if scenario_key not in self.prompt_library:
            raise ValueError(f"Scenario '{scenario}' not found in prompt library")

        prompts = self.prompt_library[scenario_key].get('defense_prompts', [])
        self.logger.info(f"Loaded {len(prompts)} prompts for scenario '{scenario}'")

    def get_best_prompt(self) -> Dict:
        """
        Get the best prompt (Top-1) from Phase 2 results

        Returns:
            Dict with best prompt details including p_kill
        """
        if not self.phase2_results:
            raise ValueError("Phase 2 results not loaded. Call load_phase2_results() first.")

        arm_details = self.phase2_results['results']['arm_details']

        # Sort by average_reward (descending)
        sorted_arms = sorted(arm_details, key=lambda x: x['average_reward'], reverse=True)
        best = sorted_arms[0]

        self.logger.info(f"\n{'='*60}")
        self.logger.info("TOP-1 PROMPT SELECTED (PROMPT-BASED)")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Prompt ID: {best.get('prompt_id', best.get('prompt_idx', 0))}")
        self.logger.info(f"Average Reward: {best['average_reward']:.3f}")
        self.logger.info(f"Pull Count: {best['pull_count']}")

        return {
            'prompt_id': best.get('prompt_id', best.get('prompt_idx', 0)),
            'average_reward': best['average_reward'],
            'cumulative_reward': best['cumulative_reward'],
            'pull_count': best['pull_count'],
            'final_probability': best['final_probability'],
            'p_kill': best.get('p_kill', '')
        }

    def deploy(self, output_path: str, scenario: str = "unknown"):
        """
        Export best prompt for deployment

        Args:
            output_path: Path to save deployment JSON
            scenario: Scenario name for metadata
        """
        best = self.get_best_prompt()

        deployment_data = {
            'metadata': {
                'scenario': scenario,
                'approach': 'prompt_based',
                'deployed_at': datetime.now().strftime("%Y%m%d_%H%M%S")
            },
            'deployed_prompt': {
                'prompt_id': best['prompt_id'],
                'p_kill': best['p_kill'],
                'average_reward': best['average_reward'],
                'pull_count': best['pull_count']
            },
            'rationale': f"Top-1 selection from Phase 2 with average_reward={best['average_reward']:.3f}"
        }

        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(deployment_data, f, indent=2, ensure_ascii=False)

        self.logger.info(f"\n✅ Deployed prompt saved to: {output_path}")
        self.logger.info(f"Prompt ID: {best['prompt_id']}")
        self.logger.info(f"Average Reward: {best['average_reward']:.3f}")

        return deployment_data


# Legacy imports for Phase3Evaluator (strategy-based)
try:
    from framework.defender_llm import DefenderLLM
    from framework.attacker_simulator import AttackerSimulator
except ImportError:
    DefenderLLM = None
    AttackerSimulator = None


class Phase3Evaluator:
    """
    Phase 3 Evaluator for generating and testing defense prompt variants

    This class orchestrates the final evaluation phase where top-performing
    strategies from Phase 2 are used to generate actual deployable defense prompts.
    Each prompt is tested extensively with weighted scoring to identify the best.
    """

    def __init__(
        self,
        scenario: str,
        attack_indices: List[int],
        agent_models: List[str],
        defender_llm: DefenderLLM,
        simulator_pool: Dict[str, AttackerSimulator],
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize Phase3Evaluator

        Args:
            scenario: Scenario type (PII, NEWS, HACK)
            attack_indices: Attack prompt indices from Phase 1 (ATTACK_INDICES)
            agent_models: List of agent model names for round-robin testing
            defender_llm: DefenderLLM instance for prompt generation
            simulator_pool: Dictionary of AttackerSimulator instances
            logger: Logger instance (creates new if None)
        """
        # Configuration
        self.scenario = scenario
        self.attack_indices = attack_indices
        self.agent_models = agent_models

        # Components
        self.defender_llm = defender_llm
        self.simulator_pool = simulator_pool
        self.logger = logger or logging.getLogger("Phase3Evaluator")

        # Data structures (initialized by loading methods)
        self.phase2_results: Optional[Dict] = None
        self.strategy_library: Optional[Dict] = None
        self.top_strategies: List[Dict] = []
        self.variant_results: List[Dict] = []

        # Attack prompts (loaded from data file)
        self.attack_prompts: List[str] = []
        self.goal: str = ""

        self.logger.info(f"Initialized Phase3Evaluator for scenario: {scenario}")
        self.logger.info(f"Attack indices: {attack_indices}")
        self.logger.info(f"Agent models: {agent_models}")

    def load_phase2_results(self, path: str):
        """
        Load Phase 2 results JSON file

        Expected structure:
        {
          "results": {
            "arm_details": [
              {
                "strategy_idx": 0,
                "strategy_pair_id": 2,
                "payload_strategy": "...",
                "wrapper_strategy": "...",
                "average_reward": 0.85,
                "pull_count": 25,
                ...
              },
              ...
            ]
          }
        }

        Args:
            path: Path to Phase 2 results JSON file
        """
        self.logger.info(f"Loading Phase 2 results from: {path}")

        with open(path, 'r', encoding='utf-8') as f:
            self.phase2_results = json.load(f)

        arm_details = self.phase2_results.get('results', {}).get('arm_details', [])
        self.logger.info(f"Loaded {len(arm_details)} arms with performance metrics")

        if not arm_details:
            raise ValueError(f"No arm_details found in Phase 2 results: {path}")

    def load_strategy_library(self, path: str, scenario: str):
        """
        Load strategy library JSON file

        Expected structure (scenario-based format 3.0):
        {
          "pii": {
            "defense_strategies": [
              {
                "pair_id": 1,
                "p_def": {
                  "strategy_name": "...",
                  "strategy_description": "...",
                  "example_prompt": "..."
                },
                "p_filter": {...},
                "status": "complete"
              },
              ...
            ]
          }
        }

        Args:
            path: Path to strategy_library.json
            scenario: Scenario type (PII, NEWS, HACK)
        """
        self.logger.info(f"Loading strategy library from: {path}")
        self.logger.info(f"Scenario: {scenario}")

        with open(path, 'r', encoding='utf-8') as f:
            self.strategy_library = json.load(f)

        scenario_key = scenario.lower()
        if scenario_key not in self.strategy_library:
            raise ValueError(f"Scenario '{scenario}' not found in strategy library. "
                           f"Available: {list(self.strategy_library.keys())}")

        scenario_data = self.strategy_library[scenario_key]
        defense_strategies = scenario_data.get('defense_strategies', [])

        self.logger.info(f"Loaded {len(defense_strategies)} defense strategies for '{scenario}'")

    def load_attack_prompts(self, data_path: str, url: Optional[str] = None):
        """
        Load attack prompts from data file using attack_indices

        Loads attack prompts in the same way as Phase 1/2:
        - Uses attack_indices to select specific prompts
        - Replaces [URL] placeholder with actual URL

        Args:
            data_path: Path to data_success.json
            url: URL to replace [URL] placeholder (uses scenario default if None)
        """
        self.logger.info(f"Loading attack prompts from: {data_path}")
        self.logger.info(f"Using attack_indices: {self.attack_indices}")

        # Import helper function from main.py or reimplement
        from main import load_attack_prompts, get_default_url

        # Use default URL if not specified
        if url is None:
            url = get_default_url(self.scenario)

        # Load attack prompts using same method as Phase 1
        self.goal, all_prompts, train_prompts = load_attack_prompts(
            data_path=data_path,
            scenario=self.scenario,
            prompts_per_type=3,  # Not used when attack_indices is specified
            url=url,
            fixed_attack_prompt=None,
            attack_indices=self.attack_indices
        )

        self.attack_prompts = train_prompts
        self.logger.info(f"Loaded {len(self.attack_prompts)} attack prompts")
        self.logger.info(f"Goal: {self.goal[:100]}...")

    def select_top_strategies(self, num: int = 1) -> List[Dict]:
        """
        Select top N strategies from Phase 2 results by average_reward

        Args:
            num: Number of top strategies to select (default: 1)

        Returns:
            List of top strategy dictionaries sorted by average_reward
        """
        if not self.phase2_results:
            raise ValueError("Phase 2 results not loaded. Call load_phase2_results() first.")

        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Selecting Top {num} Strategies from Phase 2")
        self.logger.info(f"{'='*60}")

        arm_details = self.phase2_results['results']['arm_details']

        # Sort by average_reward (descending)
        sorted_arms = sorted(arm_details, key=lambda x: x['average_reward'], reverse=True)

        # Select top N
        self.top_strategies = sorted_arms[:num]

        self.logger.info(f"\nTop {num} Strategies:")
        for i, strategy in enumerate(self.top_strategies, 1):
            self.logger.info(f"  {i}. [{strategy['payload_strategy']}]")
            self.logger.info(f"     Average Reward: {strategy['average_reward']:.3f}")
            self.logger.info(f"     Pull Count: {strategy['pull_count']}")
            self.logger.info(f"     Strategy Idx: {strategy['strategy_idx']}")

        return self.top_strategies

    def _get_strategy_description(self, strategy: Dict) -> str:
        """
        Get strategy description from strategy library

        Args:
            strategy: Strategy dict from Phase 2 arm_details

        Returns:
            Strategy description string
        """
        if not self.strategy_library:
            raise ValueError("Strategy library not loaded. Call load_strategy_library() first.")

        scenario_key = self.scenario.lower()
        scenario_data = self.strategy_library[scenario_key]
        defense_strategies = scenario_data.get('defense_strategies', [])

        # Find matching strategy by pair_id or strategy name
        pair_id = strategy.get('strategy_pair_id')
        strategy_name = strategy.get('payload_strategy')

        for def_strategy in defense_strategies:
            if def_strategy.get('pair_id') == pair_id:
                p_def = def_strategy.get('p_def', {})
                return p_def.get('strategy_description',
                               f"Strategy: {strategy_name}")

        # Fallback: return strategy name
        return f"Defense strategy: {strategy_name}"

    def generate_variants(self, strategy: Dict, num_variants: int = 3) -> List[Dict]:
        """
        Generate complete p_kill variants (p_def + p_filter) for a strategy

        For each variant:
        1. Generate p_def (defense prompt)
        2. Generate p_filter (protection wrapper: prefix + suffix)
        3. Combine into p_kill = prefix + p_def + suffix

        Args:
            strategy: Strategy dict from Phase 2 top strategies
            num_variants: Number of variants to generate (default: 3)

        Returns:
            List of variant dicts containing p_def, p_filter, and p_kill
        """
        strategy_name = strategy['payload_strategy']
        strategy_desc = self._get_strategy_description(strategy)

        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Generating {num_variants} Complete Variants (p_def + p_filter) for: {strategy_name}")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Strategy mechanism: {strategy_desc[:150]}...")

        variants = []

        for i in range(num_variants):
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Generating Variant #{i+1}")
            self.logger.info(f"{'='*60}")

            # Step 1: Generate p_def (defense prompt)
            self.logger.info(f"\n[Step 1/2] Generating p_def (defense prompt)...")
            context = f"""Core Strategy: {strategy_name}
Mechanism: {strategy_desc}

Generate a defense prompt based on this strategy.
"""

            try:
                p_def, _ = self.defender_llm.generate_payload_iterative(
                    attack_context=context,
                    previous_failure=None,
                    avoid_strategies=None
                )

                if not p_def:
                    self.logger.warning(f"Failed to generate p_def for variant #{i+1}, skipping")
                    continue

                self.logger.info(f"✓ Generated p_def ({len(p_def)} chars)")
                self.logger.info(f"  Preview: {p_def[:100]}...")

            except Exception as e:
                self.logger.error(f"Error generating p_def for variant #{i+1}: {e}")
                continue

            # Step 2: Generate p_filter (protection wrapper)
            self.logger.info(f"\n[Step 2/2] Generating p_filter (protection wrapper)...")
            self.logger.info(f"  Creating new conversation session for p_filter generation...")

            # Clear previous conversation to ensure fresh context for p_filter
            self.defender_llm.clear_conversation()

            try:
                prefix, suffix, _ = self.defender_llm.generate_pfilter_iterative(
                    p_def=p_def,
                    scenario=self.scenario,
                    previous_failure=None
                )

                if not prefix or not suffix:
                    self.logger.warning(f"Failed to generate p_filter for variant #{i+1}, using p_def only")
                    prefix = ""
                    suffix = ""

                self.logger.info(f"✓ Generated p_filter")
                self.logger.info(f"  Prefix: {len(prefix)} chars")
                self.logger.info(f"  Suffix: {len(suffix)} chars")

            except Exception as e:
                self.logger.error(f"Error generating p_filter for variant #{i+1}: {e}")
                self.logger.warning(f"Using p_def only without protection wrapper")
                prefix = ""
                suffix = ""

            # Step 3: Combine into p_kill
            p_kill = prefix + "\n\n" + p_def + "\n\n" + suffix if prefix and suffix else p_def

            variant_data = {
                'p_def': p_def,
                'p_filter_prefix': prefix,
                'p_filter_suffix': suffix,
                'p_kill': p_kill
            }

            variants.append(variant_data)

            self.logger.info(f"\n✓ Complete variant #{i+1} generated:")
            self.logger.info(f"  - p_def: {len(p_def)} chars")
            self.logger.info(f"  - p_filter_prefix: {len(prefix)} chars")
            self.logger.info(f"  - p_filter_suffix: {len(suffix)} chars")
            self.logger.info(f"  - p_kill (total): {len(p_kill)} chars")

        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"✓ Generated {len(variants)}/{num_variants} complete variants for {strategy_name}")
        self.logger.info(f"{'='*60}")
        return variants

    def test_prompt_sequential(
        self,
        variant_data: Dict,
        variant_id: str,
        source_strategy: Dict
    ) -> Dict:
        """
        Test a complete p_kill prompt sequentially: Filter OFF → Filter ON

        Tests the complete p_kill (prefix + p_def + suffix) against all attack prompts using:
        1. Filter OFF: use_filter=False
        2. Filter ON: use_filter=True

        Models are rotated round-robin by attack prompt index.

        Args:
            variant_data: Dict containing p_def, p_filter_prefix, p_filter_suffix, and p_kill
            variant_id: Identifier for this variant (e.g., "strategy1_variant1")
            source_strategy: Strategy dict this variant was generated from

        Returns:
            Dict with test results and scoring metrics
        """
        if not self.attack_prompts:
            raise ValueError("Attack prompts not loaded. Call load_attack_prompts() first.")

        num_tests = len(self.attack_prompts)

        # Extract components
        p_kill = variant_data['p_kill']
        p_def = variant_data['p_def']
        p_filter_prefix = variant_data['p_filter_prefix']
        p_filter_suffix = variant_data['p_filter_suffix']

        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Testing Variant: {variant_id}")
        self.logger.info(f"Source Strategy: {source_strategy['payload_strategy']}")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Prompt Components:")
        self.logger.info(f"  - p_def: {len(p_def)} chars")
        self.logger.info(f"  - p_filter_prefix: {len(p_filter_prefix)} chars")
        self.logger.info(f"  - p_filter_suffix: {len(p_filter_suffix)} chars")
        self.logger.info(f"  - p_kill (total): {len(p_kill)} chars")
        self.logger.info(f"Tests per phase: {num_tests}")
        self.logger.info(f"Total tests: {num_tests * 2} (Filter OFF + Filter ON)")

        results = {
            'variant_id': variant_id,
            'source_strategy': source_strategy['payload_strategy'],
            'source_strategy_idx': source_strategy['strategy_idx'],
            'p_def': p_def,
            'p_filter_prefix': p_filter_prefix,
            'p_filter_suffix': p_filter_suffix,
            'p_kill': p_kill,
            'prompt': p_kill,  # For backwards compatibility
            'prompt_length': len(p_kill),
            'filter_off_results': [],
            'filter_on_results': []
        }

        # ===== PHASE 1: Filter OFF Testing =====
        self.logger.info(f"\n--- Phase 1: Filter OFF Testing ---")
        filter_off_success = 0

        for idx, attack_prompt in enumerate(self.attack_prompts):
            # Round-robin model selection
            model_name = self.agent_models[idx % len(self.agent_models)]
            simulator = self.simulator_pool[model_name]

            self.logger.info(f"[{idx+1}/{num_tests}] Model: {model_name}")
            self.logger.info(f"  Attack: {attack_prompt[:60]}...")

            try:
                response, success, metadata = simulator.run_agent_attack(
                    attack_prompt=attack_prompt,
                    defense_prompt=p_kill,
                    use_filter=False,
                    strict_eval=False  # Use standard evaluation for Phase 3
                )

                if success:
                    filter_off_success += 1
                    self.logger.info(f"  Result: SUCCESS ✓")
                else:
                    self.logger.info(f"  Result: FAILURE ✗")

                results['filter_off_results'].append({
                    'attack_idx': idx,
                    'attack_prompt_preview': attack_prompt[:100],
                    'model': model_name,
                    'success': success,
                    'response_preview': response[:200] if response else None,
                    'metadata': metadata
                })

            except Exception as e:
                self.logger.error(f"  Error testing attack {idx}: {e}")
                results['filter_off_results'].append({
                    'attack_idx': idx,
                    'model': model_name,
                    'success': False,
                    'error': str(e)
                })

        filter_off_rate = filter_off_success / num_tests if num_tests > 0 else 0.0
        self.logger.info(f"\nFilter OFF Summary: {filter_off_success}/{num_tests} "
                        f"({filter_off_rate:.1%}) successes")

        # ===== PHASE 2: Filter ON Testing =====
        self.logger.info(f"\n--- Phase 2: Filter ON Testing ---")
        filter_on_success = 0

        for idx, attack_prompt in enumerate(self.attack_prompts):
            # Same round-robin pattern for consistency
            model_name = self.agent_models[idx % len(self.agent_models)]
            simulator = self.simulator_pool[model_name]

            self.logger.info(f"[{idx+1}/{num_tests}] Model: {model_name}")
            self.logger.info(f"  Attack: {attack_prompt[:60]}...")

            try:
                response, success, metadata = simulator.run_agent_attack(
                    attack_prompt=attack_prompt,
                    defense_prompt=p_kill,
                    use_filter=True,  # Enable Filter LLM
                    strict_eval=False
                )

                if success:
                    filter_on_success += 1
                    self.logger.info(f"  Result: SUCCESS ✓ "
                                   f"(Filtered: {metadata.get('was_filtered', False)})")
                else:
                    self.logger.info(f"  Result: FAILURE ✗ "
                                   f"(Filtered: {metadata.get('was_filtered', False)})")

                results['filter_on_results'].append({
                    'attack_idx': idx,
                    'attack_prompt_preview': attack_prompt[:100],
                    'model': model_name,
                    'success': success,
                    'was_filtered': metadata.get('was_filtered', False),
                    'response_preview': response[:200] if response else None,
                    'metadata': metadata
                })

            except Exception as e:
                self.logger.error(f"  Error testing attack {idx}: {e}")
                results['filter_on_results'].append({
                    'attack_idx': idx,
                    'model': model_name,
                    'success': False,
                    'error': str(e)
                })

        filter_on_rate = filter_on_success / num_tests if num_tests > 0 else 0.0
        self.logger.info(f"\nFilter ON Summary: {filter_on_success}/{num_tests} "
                        f"({filter_on_rate:.1%}) successes")

        # ===== SCORING =====
        weighted_score = self.calculate_weighted_score(
            filter_off_success, filter_on_success
        )
        overall_success_rate = self._calculate_success_rate(
            filter_off_success, filter_on_success, num_tests
        )

        results['scoring'] = {
            'filter_off_count': filter_off_success,
            'filter_off_total': num_tests,
            'filter_off_rate': filter_off_rate,
            'filter_on_count': filter_on_success,
            'filter_on_total': num_tests,
            'filter_on_rate': filter_on_rate,
            'weighted_score': weighted_score,
            'overall_success_rate': overall_success_rate
        }

        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Scoring Summary for {variant_id}")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Filter OFF: {filter_off_success}/{num_tests} ({filter_off_rate:.1%})")
        self.logger.info(f"Filter ON:  {filter_on_success}/{num_tests} ({filter_on_rate:.1%})")
        self.logger.info(f"Weighted Score: {filter_off_success} + ({filter_on_success} × 3) = {weighted_score}")
        self.logger.info(f"Overall Success Rate: {overall_success_rate:.1%}")

        return results

    def calculate_weighted_score(
        self,
        filter_off_success: int,
        filter_on_success: int
    ) -> float:
        """
        Calculate weighted score with Filter ON = 3× weight

        Formula: score = filter_off_success + (filter_on_success × 3)

        Example:
            Filter OFF: 7 successes
            Filter ON:  3 successes
            Score = 7 + (3 × 3) = 7 + 9 = 16

        Args:
            filter_off_success: Number of successes with filter=False
            filter_on_success: Number of successes with filter=True

        Returns:
            Weighted score (higher is better)
        """
        return filter_off_success + (filter_on_success * 3)

    def _calculate_success_rate(
        self,
        filter_off_success: int,
        filter_on_success: int,
        total_tests: int
    ) -> float:
        """
        Calculate overall success rate considering weights

        Total "effective successes" = filter_off + (filter_on × 3)
        Total "effective tests" = total_tests × 4 (1× + 3× weight)

        Example:
            Filter OFF: 7/10
            Filter ON:  3/10
            Effective successes: 7 + (3×3) = 16
            Effective tests: 10 × 4 = 40
            Success rate: 16/40 = 40%

        Args:
            filter_off_success: Number of successes with filter=False
            filter_on_success: Number of successes with filter=True
            total_tests: Total number of tests per phase

        Returns:
            Success rate (0.0 to 1.0)
        """
        weighted_successes = filter_off_success + (filter_on_success * 3)
        total_effective_tests = total_tests * 4  # 1× + 3× weights

        return weighted_successes / total_effective_tests if total_effective_tests > 0 else 0.0

    def identify_best_prompt(self) -> Dict:
        """
        Identify the best prompt from all tested variants

        Sorts variants by weighted_score and returns the top one.

        Returns:
            Dict with best prompt details
        """
        if not self.variant_results:
            raise ValueError("No variant results available. Run evaluation first.")

        # Sort by weighted_score (descending)
        sorted_variants = sorted(
            self.variant_results,
            key=lambda x: x['scoring']['weighted_score'],
            reverse=True
        )

        best = sorted_variants[0]

        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"BEST PROMPT IDENTIFIED")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Variant ID: {best['variant_id']}")
        self.logger.info(f"Source Strategy: {best['source_strategy']}")
        self.logger.info(f"Weighted Score: {best['scoring']['weighted_score']}")
        self.logger.info(f"Success Rate: {best['scoring']['overall_success_rate']:.1%}")
        self.logger.info(f"Filter OFF: {best['scoring']['filter_off_count']}/{best['scoring']['filter_off_total']}")
        self.logger.info(f"Filter ON: {best['scoring']['filter_on_count']}/{best['scoring']['filter_on_total']}")

        return {
            'variant_id': best['variant_id'],
            'source_strategy': best['source_strategy'],
            'weighted_score': best['scoring']['weighted_score'],
            'success_rate': best['scoring']['overall_success_rate'],
            'filter_off_rate': best['scoring']['filter_off_rate'],
            'filter_on_rate': best['scoring']['filter_on_rate'],
            'p_def': best['p_def'],
            'p_filter_prefix': best['p_filter_prefix'],
            'p_filter_suffix': best['p_filter_suffix'],
            'p_kill': best['p_kill'],
            'prompt': best['prompt'],  # For backwards compatibility
            'rationale': f"Highest weighted score due to Filter OFF: {best['scoring']['filter_off_rate']:.1%}, "
                        f"Filter ON: {best['scoring']['filter_on_rate']:.1%}"
        }

    def run_evaluation(
        self,
        num_top_strategies: int = 3,
        variants_per_strategy: int = 3
    ) -> Dict:
        """
        Main evaluation orchestration method

        Workflow:
        1. Select Top N strategies from Phase 2
        2. For each strategy:
           - Generate M variants
           - Test each variant sequentially (Filter OFF + Filter ON)
        3. Identify best prompt by weighted score
        4. Return full results

        Args:
            num_top_strategies: Number of top strategies to select (default: 1)
            variants_per_strategy: Number of variants per strategy (default: 3)

        Returns:
            Dict with complete evaluation results
        """
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"PHASE 3: PROMPT VARIANT EVALUATION")
        self.logger.info(f"{'='*80}")
        self.logger.info(f"Configuration:")
        self.logger.info(f"  Scenario: {self.scenario}")
        self.logger.info(f"  Top strategies: {num_top_strategies}")
        self.logger.info(f"  Variants per strategy: {variants_per_strategy}")
        self.logger.info(f"  Total prompts to test: {num_top_strategies * variants_per_strategy}")
        self.logger.info(f"  Attack prompts per test: {len(self.attack_prompts)}")
        self.logger.info(f"  Total evaluations: {num_top_strategies * variants_per_strategy * len(self.attack_prompts) * 2}")

        # Step 1: Select top strategies
        top_strategies = self.select_top_strategies(num=num_top_strategies)

        # Step 2: Generate and test variants
        self.variant_results = []

        for strategy_idx, strategy in enumerate(top_strategies, 1):
            strategy_name = strategy['payload_strategy']

            self.logger.info(f"\n{'='*80}")
            self.logger.info(f"STRATEGY {strategy_idx}/{num_top_strategies}: {strategy_name}")
            self.logger.info(f"{'='*80}")

            # Generate variants
            variants = self.generate_variants(strategy, num_variants=variants_per_strategy)

            if not variants:
                self.logger.warning(f"No variants generated for {strategy_name}, skipping")
                continue

            # Test each variant
            for variant_idx, variant_data in enumerate(variants, 1):
                variant_id = f"strategy{strategy_idx}_variant{variant_idx}"

                self.logger.info(f"\n{'#'*80}")
                self.logger.info(f"TESTING VARIANT {variant_idx}/{len(variants)} "
                               f"(Overall: {len(self.variant_results)+1}/{num_top_strategies*variants_per_strategy})")
                self.logger.info(f"{'#'*80}")

                # Test complete p_kill prompt
                test_results = self.test_prompt_sequential(
                    variant_data=variant_data,
                    variant_id=variant_id,
                    source_strategy=strategy
                )

                self.variant_results.append(test_results)

        # Step 3: Identify best prompt
        best_prompt = self.identify_best_prompt()

        # Step 4: Compile final results
        results = {
            'metadata': {
                'scenario': self.scenario,
                'num_strategies_selected': num_top_strategies,
                'variants_per_strategy': variants_per_strategy,
                'total_prompts_tested': len(self.variant_results),
                'attack_indices': self.attack_indices,
                'agent_models': self.agent_models,
                'num_tests_per_prompt': len(self.attack_prompts) * 2
            },
            'top_strategies': [
                {
                    'rank': i+1,
                    'strategy_idx': s['strategy_idx'],
                    'strategy_pair_id': s['strategy_pair_id'],
                    'payload_strategy': s['payload_strategy'],
                    'phase2_average_reward': s['average_reward'],
                    'phase2_pull_count': s['pull_count']
                }
                for i, s in enumerate(top_strategies)
            ],
            'variant_results': self.variant_results,
            'best_prompt': best_prompt
        }

        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"PHASE 3 EVALUATION COMPLETE")
        self.logger.info(f"{'='*80}")
        self.logger.info(f"Total variants tested: {len(self.variant_results)}")
        self.logger.info(f"Best variant: {best_prompt['variant_id']}")
        self.logger.info(f"Best score: {best_prompt['weighted_score']}")

        return results

    def save_results(self, output_path: str):
        """
        Save evaluation results to JSON file

        Args:
            output_path: Path to save results JSON
        """
        if not self.variant_results:
            raise ValueError("No results to save. Run evaluation first.")

        # Prepare output data
        best_prompt = self.identify_best_prompt()

        output_data = {
            'metadata': {
                'scenario': self.scenario,
                'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
                'num_strategies_selected': len(self.top_strategies),
                'variants_per_strategy': len(self.variant_results) // len(self.top_strategies) if self.top_strategies else 0,
                'total_prompts_tested': len(self.variant_results),
                'attack_indices': self.attack_indices,
                'agent_models': self.agent_models,
                'num_tests_per_prompt': len(self.attack_prompts) * 2
            },
            'top_strategies': [
                {
                    'rank': i+1,
                    'strategy_idx': s['strategy_idx'],
                    'strategy_pair_id': s['strategy_pair_id'],
                    'payload_strategy': s['payload_strategy'],
                    'phase2_average_reward': s['average_reward'],
                    'phase2_pull_count': s['pull_count']
                }
                for i, s in enumerate(self.top_strategies)
            ],
            'variant_results': [
                {
                    'rank': i+1,
                    **v
                }
                for i, v in enumerate(
                    sorted(self.variant_results,
                          key=lambda x: x['scoring']['weighted_score'],
                          reverse=True)
                )
            ],
            'best_prompt': best_prompt
        }

        # Save to file
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        self.logger.info(f"\n✅ Results saved to: {output_path}")
        self.logger.info(f"File size: {os.path.getsize(output_path)} bytes")
