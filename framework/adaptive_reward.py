"""
Adaptive Reward Calculator for Phase 1 Step 2 EXP3 Bandit

Calculates rewards based on multiple difficulty factors:
- Agent model type (harder models = higher difficulty bonus)
- Attack type difficulty (harder attack types = higher bonus)
- Tracks per-model and per-attack-type success rates to learn difficulty
- All difficulty bonuses are ADAPTIVE (based on observed success rates)
"""

from typing import Dict, Tuple, Any, List, Optional
from collections import defaultdict
import logging


class AdaptiveRewardCalculator:
    """
    Context difficulty-based adaptive reward calculator.

    Features:
    - Tracks success rate per agent model
    - Lower success rate = harder agent model = higher reward
    - Handles cold start with global success rate

    Reward Formula:
        reward = 0.0 (if failure)
        reward = base_reward + model_difficulty_bonus + attack_type_bonus (if success)

        model_difficulty_bonus = (1 - agent_model_success_rate) * max_difficulty_bonus
        attack_type_bonus = (1 - attack_type_success_rate) * max_attack_type_bonus
    """

    def __init__(self,
                 base_reward: float = 1.0,
                 max_difficulty_bonus: float = 2.0,
                 max_attack_type_bonus: float = 1.0,
                 min_samples_for_adaptive: int = 10,
                 logger: logging.Logger = None,
                 **kwargs):  # Accept but ignore legacy filter_bonus parameter
        """
        Initialize adaptive reward calculator.

        Args:
            base_reward: Base reward for any success (default: 1.0)
            max_difficulty_bonus: Maximum agent model difficulty bonus (default: 2.0)
            max_attack_type_bonus: Maximum attack type difficulty bonus (default: 1.0)
            min_samples_for_adaptive: Minimum samples before using context-specific
                                      success rate (default: 10)
            logger: Logger instance
        """
        self.base_reward = base_reward
        self.max_difficulty_bonus = max_difficulty_bonus
        self.max_attack_type_bonus = max_attack_type_bonus
        self.min_samples = min_samples_for_adaptive
        self.logger = logger or logging.getLogger(__name__)

        # Agent model-specific statistics
        # Key: agent_model (string)
        # Value: {'attempts': N, 'successes': M}
        self.model_stats: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {'attempts': 0, 'successes': 0}
        )

        # Attack type-specific statistics
        # Key: attack_type (string, e.g., "CR", "TRANS")
        # Value: {'attempts': N, 'successes': M}
        self.attack_type_stats: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {'attempts': 0, 'successes': 0}
        )

        # Global statistics (for cold start)
        self.total_attempts = 0
        self.total_successes = 0

        # Benign task statistics (for false positive tracking)
        self.benign_stats: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {'attempts': 0, 'blocked': 0}
        )
        self.total_benign_attempts = 0
        self.total_benign_blocked = 0

        self.logger.debug(f"Initialized AdaptiveRewardCalculator: "
                         f"base={base_reward}, "
                         f"max_model_difficulty={max_difficulty_bonus}, "
                         f"max_attack_type_bonus={max_attack_type_bonus}")

    def get_model_key(self, agent_model: str) -> str:
        """Generate model key from agent model name."""
        return agent_model

    def calculate_reward(self,
                         agent_model: str,
                         defense_success: bool,
                         attack_type: Optional[str] = None,
                         **kwargs) -> float:  # Accept but ignore legacy use_filter parameter
        """
        Calculate context difficulty-based reward.

        Reward Components:
        1. base_reward: Always given on success (default: 1.0)
        2. model_difficulty_bonus: Based on agent model success rate (0.0 ~ 2.0)
           - 0% success rate model: +2.0 bonus
           - 50% success rate model: +1.0 bonus
           - 100% success rate model: +0.0 bonus
        3. attack_type_bonus: Based on attack type success rate (0.0 ~ 1.0)
           - 0% success rate attack type: +1.0 bonus
           - 50% success rate attack type: +0.5 bonus
           - 100% success rate attack type: +0.0 bonus

        Total reward range: 0.0 (failure) to 1.0 (hardest context success)
        - Rewards are normalized to [0, 1] range by dividing by max possible reward

        Args:
            agent_model: Name of the agent model (e.g., "gpt-4o", "grok-4")
            defense_success: Whether defense succeeded
            attack_type: Attack type (e.g., "CR", "TRANS"). Optional for backward compatibility.

        Returns:
            Reward value (0.0 ~ 1.0)
        """
        model_key = self.get_model_key(agent_model)
        model_stats = self.model_stats[model_key]

        # Update model statistics FIRST (before calculating reward)
        model_stats['attempts'] += 1
        self.total_attempts += 1

        if defense_success:
            model_stats['successes'] += 1
            self.total_successes += 1

        # Update attack type statistics if provided
        if attack_type:
            attack_stats = self.attack_type_stats[attack_type]
            attack_stats['attempts'] += 1
            if defense_success:
                attack_stats['successes'] += 1

        # Failure = 0 reward
        if not defense_success:
            return 0.0

        # Start with base reward
        reward = self.base_reward

        # Add adaptive difficulty bonus based on agent model
        model_difficulty_bonus = self._calculate_difficulty_bonus(model_key)
        reward += model_difficulty_bonus

        # Add adaptive difficulty bonus based on attack type
        attack_type_bonus = 0.0
        if attack_type:
            attack_type_bonus = self._calculate_attack_type_bonus(attack_type)
            reward += attack_type_bonus

        # Calculate max possible reward for normalization
        max_possible_reward = (self.base_reward +
                               self.max_difficulty_bonus + self.max_attack_type_bonus)

        # Normalize to [0, 1] range
        normalized_reward = reward / max_possible_reward

        self.logger.debug(f"Reward for {agent_model} (type={attack_type}): "
                         f"base={self.base_reward}, "
                         f"model_difficulty={model_difficulty_bonus:.2f}, "
                         f"attack_type={attack_type_bonus:.2f}, "
                         f"raw={reward:.2f}, normalized={normalized_reward:.4f}")

        return normalized_reward

    def update_benign_stats(self, agent_model: str, benign_blocked: bool):
        """
        Update benign task statistics for false positive tracking.

        Args:
            agent_model: Agent model used for the benign test
            benign_blocked: True if the benign task was incorrectly blocked
        """
        model_key = self.get_model_key(agent_model)
        self.benign_stats[model_key]['attempts'] += 1
        self.total_benign_attempts += 1

        if benign_blocked:
            self.benign_stats[model_key]['blocked'] += 1
            self.total_benign_blocked += 1

        self.logger.debug(f"Benign stats update: model={agent_model}, blocked={benign_blocked}, "
                         f"total={self.total_benign_blocked}/{self.total_benign_attempts}")

    def get_benign_stats(self) -> Dict[str, Any]:
        """
        Get benign task false positive statistics.

        Returns:
            Dictionary with benign blocking stats per model and overall
        """
        per_model = {}
        for model_key, stats in self.benign_stats.items():
            attempts = stats['attempts']
            blocked = stats['blocked']
            per_model[model_key] = {
                'attempts': attempts,
                'blocked': blocked,
                'false_positive_rate': blocked / attempts if attempts > 0 else 0.0
            }

        overall_fpr = (self.total_benign_blocked / self.total_benign_attempts
                      if self.total_benign_attempts > 0 else 0.0)

        return {
            'per_model': per_model,
            'total_attempts': self.total_benign_attempts,
            'total_blocked': self.total_benign_blocked,
            'overall_false_positive_rate': overall_fpr
        }

    def _calculate_difficulty_bonus(self, model_key: str) -> float:
        """
        Calculate difficulty bonus based on agent model success rate.

        Uses model-specific success rate if enough samples,
        otherwise falls back to global success rate.

        Args:
            model_key: agent model name (string)

        Returns:
            Difficulty bonus (0.0 ~ max_difficulty_bonus)
        """
        stats = self.model_stats[model_key]

        if stats['attempts'] >= self.min_samples:
            # Use model-specific success rate
            success_rate = stats['successes'] / stats['attempts']
            difficulty_bonus = (1.0 - success_rate) * self.max_difficulty_bonus
        else:
            # Cold start: use global success rate with reduced weight
            if self.total_attempts > 0:
                global_success_rate = self.total_successes / self.total_attempts
                # Apply half weight during cold start
                difficulty_bonus = (1.0 - global_success_rate) * (self.max_difficulty_bonus / 2)
            else:
                # No data yet: assume 50% success rate
                difficulty_bonus = self.max_difficulty_bonus / 2

        return difficulty_bonus

    def _calculate_attack_type_bonus(self, attack_type: str) -> float:
        """
        Calculate difficulty bonus based on attack type success rate.

        Uses attack type-specific success rate if enough samples,
        otherwise falls back to global success rate.

        Args:
            attack_type: Attack type string (e.g., "CR", "TRANS")

        Returns:
            Difficulty bonus (0.0 ~ max_attack_type_bonus)
        """
        stats = self.attack_type_stats[attack_type]

        if stats['attempts'] >= self.min_samples:
            # Use attack type-specific success rate
            success_rate = stats['successes'] / stats['attempts']
            attack_type_bonus = (1.0 - success_rate) * self.max_attack_type_bonus
        else:
            # Cold start: use global success rate with reduced weight
            if self.total_attempts > 0:
                global_success_rate = self.total_successes / self.total_attempts
                # Apply half weight during cold start
                attack_type_bonus = (1.0 - global_success_rate) * (self.max_attack_type_bonus / 2)
            else:
                # No data yet: assume 50% success rate
                attack_type_bonus = self.max_attack_type_bonus / 2

        return attack_type_bonus

    def get_model_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for all observed agent models.

        Returns:
            Dictionary mapping model name to stats:
            {
                "gpt-4o": {
                    "attempts": 50,
                    "successes": 10,
                    "success_rate": 0.20,
                    "estimated_difficulty": 0.80
                },
                "grok-4": {
                    "attempts": 50,
                    "successes": 5,
                    "success_rate": 0.10,
                    "estimated_difficulty": 0.90
                },
                ...
            }
        """
        result = {}

        for model_key, stats in self.model_stats.items():
            attempts = stats['attempts']
            successes = stats['successes']
            success_rate = successes / attempts if attempts > 0 else 0.0

            result[model_key] = {
                'agent_model': model_key,
                'attempts': attempts,
                'successes': successes,
                'success_rate': success_rate,
                'estimated_difficulty': 1.0 - success_rate
            }

        return result

    def get_difficulty_ranking(self) -> List[Dict[str, Any]]:
        """
        Get agent models ranked by difficulty (hardest first).

        Lower success rate = harder agent model.

        Returns:
            List of model info dictionaries sorted by difficulty (descending)
        """
        models = []

        for model_key, stats in self.model_stats.items():
            if stats['attempts'] > 0:
                success_rate = stats['successes'] / stats['attempts']

                models.append({
                    'agent_model': model_key,
                    'success_rate': success_rate,
                    'difficulty': 1.0 - success_rate,
                    'attempts': stats['attempts'],
                    'successes': stats['successes']
                })

        # Sort by success_rate ascending (lowest success = hardest = first)
        return sorted(models, key=lambda x: x['success_rate'])

    def get_attack_type_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for all observed attack types.

        Returns:
            Dictionary mapping attack type to stats:
            {
                "CR": {
                    "attempts": 50,
                    "successes": 25,
                    "success_rate": 0.50,
                    "estimated_difficulty": 0.50
                },
                "TRANS": {
                    "attempts": 50,
                    "successes": 5,
                    "success_rate": 0.10,
                    "estimated_difficulty": 0.90
                },
                ...
            }
        """
        result = {}

        for attack_type, stats in self.attack_type_stats.items():
            attempts = stats['attempts']
            successes = stats['successes']
            success_rate = successes / attempts if attempts > 0 else 0.0

            result[attack_type] = {
                'attack_type': attack_type,
                'attempts': attempts,
                'successes': successes,
                'success_rate': success_rate,
                'estimated_difficulty': 1.0 - success_rate
            }

        return result

    def get_attack_type_ranking(self) -> List[Dict[str, Any]]:
        """
        Get attack types ranked by difficulty (hardest first).

        Lower success rate = harder attack type.

        Returns:
            List of attack type info dictionaries sorted by difficulty (descending)
        """
        attack_types = []

        for attack_type, stats in self.attack_type_stats.items():
            if stats['attempts'] > 0:
                success_rate = stats['successes'] / stats['attempts']

                attack_types.append({
                    'attack_type': attack_type,
                    'success_rate': success_rate,
                    'difficulty': 1.0 - success_rate,
                    'attempts': stats['attempts'],
                    'successes': stats['successes']
                })

        # Sort by success_rate ascending (lowest success = hardest = first)
        return sorted(attack_types, key=lambda x: x['success_rate'])

    def get_global_stats(self) -> Dict[str, Any]:
        """
        Get global statistics across all contexts.

        Returns:
            Dictionary with global attempt/success counts and rate
        """
        global_success_rate = (
            self.total_successes / self.total_attempts
            if self.total_attempts > 0 else 0.0
        )

        return {
            'total_attempts': self.total_attempts,
            'total_successes': self.total_successes,
            'global_success_rate': global_success_rate,
            'num_models': len(self.model_stats),
            'num_attack_types': len(self.attack_type_stats)
        }

    def reset(self):
        """Reset all statistics."""
        self.model_stats = defaultdict(
            lambda: {'attempts': 0, 'successes': 0}
        )
        self.attack_type_stats = defaultdict(
            lambda: {'attempts': 0, 'successes': 0}
        )
        self.total_attempts = 0
        self.total_successes = 0
        self.logger.info("AdaptiveRewardCalculator reset")

    def save_state(self) -> Dict[str, Any]:
        """
        Save current state for checkpoint persistence.

        Returns:
            Dictionary containing all state needed to restore the calculator
        """
        return {
            'model_stats': dict(self.model_stats),  # Convert defaultdict to regular dict
            'attack_type_stats': dict(self.attack_type_stats),  # Convert defaultdict to regular dict
            'total_attempts': self.total_attempts,
            'total_successes': self.total_successes,
            'config': {
                'base_reward': self.base_reward,
                'max_difficulty_bonus': self.max_difficulty_bonus,
                'max_attack_type_bonus': self.max_attack_type_bonus,
                'min_samples': self.min_samples
            }
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """
        Restore state from checkpoint.

        Args:
            state: Dictionary from save_state()
        """
        if not state:
            self.logger.warning("Empty state provided to load_state, skipping")
            return

        # Restore model stats
        saved_model_stats = state.get('model_stats', {})
        self.model_stats = defaultdict(
            lambda: {'attempts': 0, 'successes': 0}
        )
        for model_key, stats in saved_model_stats.items():
            self.model_stats[model_key] = stats.copy()

        # Restore attack type stats
        saved_attack_type_stats = state.get('attack_type_stats', {})
        self.attack_type_stats = defaultdict(
            lambda: {'attempts': 0, 'successes': 0}
        )
        for attack_type, stats in saved_attack_type_stats.items():
            self.attack_type_stats[attack_type] = stats.copy()

        # Restore global counters
        self.total_attempts = state.get('total_attempts', 0)
        self.total_successes = state.get('total_successes', 0)

        self.logger.info(f"Loaded AdaptiveRewardCalculator state: "
                        f"{len(self.model_stats)} models, "
                        f"{len(self.attack_type_stats)} attack types, "
                        f"{self.total_attempts} attempts, "
                        f"{self.total_successes} successes")


if __name__ == "__main__":
    # Test the adaptive reward calculator
    logging.basicConfig(level=logging.DEBUG)

    print("Testing AdaptiveRewardCalculator...")
    calc = AdaptiveRewardCalculator(
        base_reward=1.0,
        max_difficulty_bonus=2.0,
        max_attack_type_bonus=1.0,
        min_samples_for_adaptive=5
    )

    # Simulate different agent models with varying success rates
    models = ["gpt-4o", "gpt-4o-mini", "claude-sonnet-4", "grok-4"]

    # Model difficulty (lower success rate = harder model)
    model_success_probs = {
        "grok-4": 0.1,          # 10% success (hardest)
        "gpt-4o": 0.3,          # 30% success (hard)
        "claude-sonnet-4": 0.6, # 60% success (medium)
        "gpt-4o-mini": 0.9      # 90% success (easiest)
    }

    # Attack type difficulty (lower success rate = harder type)
    attack_types = ["CR", "AR", "RE", "TC", "LOGIC", "PROG", "TRANS", "SUPER", "SUDO", "SIMU"]
    attack_type_success_probs = {
        "TRANS": 0.1,   # 10% success (hardest - multilingual)
        "PROG": 0.2,    # 20% success (hard - programming)
        "LOGIC": 0.3,   # 30% success (hard - logical)
        "SIMU": 0.4,    # 40% success (medium)
        "SUDO": 0.5,    # 50% success (medium)
        "SUPER": 0.5,   # 50% success (medium)
        "TC": 0.6,      # 60% success (medium)
        "RE": 0.7,      # 70% success (easy)
        "AR": 0.8,      # 80% success (easy)
        "CR": 0.9       # 90% success (easiest)
    }

    import random
    # Test each model and attack type combination
    for _ in range(20):
        for model in models:
            attack_type = random.choice(attack_types)
            # Combined success probability (product of model and attack type probs)
            combined_prob = model_success_probs[model] * attack_type_success_probs[attack_type]
            success = random.random() < combined_prob
            reward = calc.calculate_reward(model, success, attack_type)
            if success:
                print(f"Model {model}, Type {attack_type}: SUCCESS, reward={reward:.2f}")

    print("\n=== Final Statistics ===")
    print("\nAgent Model Statistics:")
    for model_name, stats in calc.get_model_stats().items():
        print(f"  {model_name}: success_rate={stats['success_rate']:.1%}, "
              f"difficulty={stats['estimated_difficulty']:.2f}, "
              f"attempts={stats['attempts']}")

    print("\nModel Difficulty Ranking (hardest first):")
    for i, model_info in enumerate(calc.get_difficulty_ranking()):
        print(f"  {i+1}. {model_info['agent_model']}: "
              f"success_rate={model_info['success_rate']:.1%}, "
              f"difficulty={model_info['difficulty']:.2f}")

    print("\nAttack Type Statistics:")
    for attack_type, stats in calc.get_attack_type_stats().items():
        print(f"  {attack_type}: success_rate={stats['success_rate']:.1%}, "
              f"difficulty={stats['estimated_difficulty']:.2f}, "
              f"attempts={stats['attempts']}")

    print("\nAttack Type Difficulty Ranking (hardest first):")
    for i, type_info in enumerate(calc.get_attack_type_ranking()):
        print(f"  {i+1}. {type_info['attack_type']}: "
              f"success_rate={type_info['success_rate']:.1%}, "
              f"difficulty={type_info['difficulty']:.2f}")

    print("\nGlobal Stats:")
    global_stats = calc.get_global_stats()
    print(f"  Total: {global_stats['total_successes']}/{global_stats['total_attempts']} "
          f"({global_stats['global_success_rate']:.1%})")
    print(f"  Models: {global_stats['num_models']}, Attack Types: {global_stats['num_attack_types']}")
