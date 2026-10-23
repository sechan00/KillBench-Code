"""
EXP3-IX (Implicit eXploration) Multi-Armed Bandit

Implementation based on Neu & Bartók (2013) for adversarial settings.
EXP3-IX provides implicit exploration through modified importance weighting,
offering better numerical stability and tighter regret bounds than standard EXP3.
"""

import numpy as np
from typing import Dict, List, Any, Optional
import logging


class EXP3IXBandit:
    """
    EXP3-IX (Implicit eXploration) Multi-Armed Bandit for adversarial settings.

    Features:
    - Implicit exploration via modified importance weighting
    - Better numerical stability than standard EXP3
    - Automatic hyperparameter calculation based on T and K
    - Handles normalized rewards (0.0 ~ 1.0 range)

    Key difference from EXP3:
    - EXP3: ℓ̂ = ℓ / P(i), explicit mixing with γ/K
    - EXP3-IX: ℓ̂ = ℓ / (P(i) + γ), implicit exploration through optimistic loss estimate

    Loss conversion: ℓ = max_reward - reward (higher reward = lower loss)
    """

    def __init__(self,
                 num_arms: int,
                 eta: Optional[float] = None,
                 gamma: Optional[float] = None,
                 total_rounds: int = 100,
                 max_reward: float = 1.0,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize EXP3-IX bandit.

        Args:
            num_arms: Number of arms (K)
            eta: Learning rate. If None, auto-calculated as √(2 log(K) / (T × K))
            gamma: Implicit exploration parameter. If None, auto-calculated as η/2
            total_rounds: Expected total rounds (T), used for auto hyperparameter calculation
            max_reward: Maximum possible reward for loss conversion (default: 1.0)
            logger: Logger instance for debugging
        """
        if num_arms <= 0:
            raise ValueError("num_arms must be positive")
        if total_rounds <= 0:
            raise ValueError("total_rounds must be positive")

        self.num_arms = num_arms
        self.total_rounds = total_rounds
        self.max_reward = max_reward
        self.logger = logger or logging.getLogger(__name__)

        # Auto-calculate hyperparameters if not provided (Neu & Bartók, 2013)
        if eta is None:
            eta = np.sqrt(2 * np.log(num_arms) / (total_rounds * num_arms))
        if gamma is None:
            gamma = eta / 2

        self.eta = eta      # Learning rate
        self.gamma = gamma  # Implicit exploration parameter

        # Initialize weights to 1 for all arms
        self.weights = np.ones(num_arms, dtype=np.float64)

        # Calculate initial probabilities
        self.probabilities = self._calculate_probabilities()

        # Statistics tracking
        self.cumulative_rewards = np.zeros(num_arms, dtype=np.float64)
        self.pull_counts = np.zeros(num_arms, dtype=np.int32)

        # History for analysis
        self.history: List[Dict[str, Any]] = []

        self.logger.debug(f"Initialized EXP3-IX with {num_arms} arms, "
                         f"eta={eta:.6f}, gamma={gamma:.6f}")

    def _calculate_probabilities(self) -> np.ndarray:
        """
        Calculate probability distribution from weights.

        EXP3-IX Formula: P_t(i) = w_i / Σw_j

        Unlike standard EXP3, there is no explicit exploration mixing.
        Exploration is handled implicitly through the modified update rule.

        Returns:
            Probability distribution over arms
        """
        weight_sum = np.sum(self.weights)
        return self.weights / weight_sum

    def select_arm(self) -> int:
        """
        Sample arm according to probability distribution.

        Returns:
            Index of selected arm (0 to num_arms-1)
        """
        selected = np.random.choice(self.num_arms, p=self.probabilities)
        return int(selected)

    def update(self, arm_idx: int, reward: float):
        """
        Update arm weight based on received reward.

        EXP3-IX Update Rule (Loss-based, following Neu & Bartók 2013):
        1. Convert reward to loss: ℓ = max_reward - reward
        2. Estimated loss: ℓ̂ = ℓ / (P_t(arm_idx) + γ)
           - γ added to denominator provides optimistic bias (lower loss estimate)
           - This encourages exploration of low-probability arms
        3. Weight update: w_i ← w_i * exp(-η * ℓ̂)
           - Negative sign: lower loss increases weight
           - η is the learning rate

        Args:
            arm_idx: Index of pulled arm
            reward: Reward received (normalized to [0.0, 1.0] range)
        """
        if not (0 <= arm_idx < self.num_arms):
            raise ValueError(f"arm_idx must be in [0, {self.num_arms-1}]")

        # Convert reward to loss (higher reward = lower loss)
        loss = self.max_reward - reward

        # EXP3-IX: Implicit exploration via γ in denominator
        # Adding γ creates optimistic bias (underestimates loss)
        # This is crucial for exploration: low-probability arms get optimistic estimates
        estimated_loss = loss / (self.probabilities[arm_idx] + self.gamma)

        # Exponential weight update with NEGATIVE sign (loss minimization)
        # Lower loss → higher exp(-η * ℓ̂) → weight increases
        self.weights[arm_idx] *= np.exp(-self.eta * estimated_loss)

        # Prevent numerical overflow by normalizing if weights get too large
        max_weight = np.max(self.weights)
        if max_weight > 1e10:
            self.weights /= max_weight
            self.logger.debug("Normalized weights to prevent overflow")

        # Recalculate probabilities with new weights
        self.probabilities = self._calculate_probabilities()

        # Track statistics
        self.cumulative_rewards[arm_idx] += reward
        self.pull_counts[arm_idx] += 1

        # Record history
        self.history.append({
            "round": len(self.history) + 1,
            "arm_idx": arm_idx,
            "reward": reward,
            "loss": loss,
            "estimated_loss": estimated_loss,
            "probability": self.probabilities[arm_idx],
            "cumulative_reward": float(self.cumulative_rewards[arm_idx]),
            "pull_count": int(self.pull_counts[arm_idx])
        })

    def get_stats(self) -> Dict[str, Any]:
        """
        Return current bandit statistics.

        Returns:
            Dictionary with:
            - weights: Current arm weights
            - probabilities: Current selection probabilities
            - cumulative_rewards: Total rewards per arm
            - pull_counts: Number of times each arm was pulled
            - average_rewards: Mean reward per arm
            - best_arm: Arm with highest average reward
            - total_rounds: Number of completed rounds
        """
        # Calculate average rewards, handling division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            avg_rewards = np.where(
                self.pull_counts > 0,
                self.cumulative_rewards / self.pull_counts,
                0.0
            )

        # Find best arm by average reward
        best_arm = int(np.argmax(avg_rewards)) if np.any(self.pull_counts > 0) else 0

        return {
            "eta": self.eta,
            "gamma": self.gamma,
            "weights": self.weights.tolist(),
            "probabilities": self.probabilities.tolist(),
            "cumulative_rewards": self.cumulative_rewards.tolist(),
            "pull_counts": self.pull_counts.tolist(),
            "average_rewards": avg_rewards.tolist(),
            "best_arm": best_arm,
            "total_rounds": len(self.history)
        }

    def get_history(self) -> List[Dict[str, Any]]:
        """
        Return full history of arm pulls and rewards.

        Returns:
            List of dictionaries, one per round
        """
        return self.history

    def get_arm_ranking(self) -> List[Dict[str, Any]]:
        """
        Get arms ranked by average reward.

        Returns:
            List of arm info dictionaries, sorted by avg reward (highest first)
        """
        stats = self.get_stats()

        arms = []
        for idx in range(self.num_arms):
            arms.append({
                "arm_idx": idx,
                "pull_count": stats["pull_counts"][idx],
                "cumulative_reward": stats["cumulative_rewards"][idx],
                "average_reward": stats["average_rewards"][idx],
                "probability": stats["probabilities"][idx],
                "weight": stats["weights"][idx]
            })

        # Sort by average reward (descending)
        return sorted(arms, key=lambda x: x["average_reward"], reverse=True)

    def reset(self):
        """
        Reset bandit to initial state.

        Clears all learned weights and history.
        """
        self.weights = np.ones(self.num_arms, dtype=np.float64)
        self.probabilities = self._calculate_probabilities()
        self.cumulative_rewards = np.zeros(self.num_arms, dtype=np.float64)
        self.pull_counts = np.zeros(self.num_arms, dtype=np.int32)
        self.history = []

        self.logger.info("EXP3-IX bandit reset to initial state")

    def save_state(self) -> Dict[str, Any]:
        """
        Save complete bandit state for checkpoint/resume.

        Returns:
            Dictionary containing all state needed to restore the bandit
        """
        return {
            "num_arms": self.num_arms,
            "eta": self.eta,
            "gamma": self.gamma,
            "total_rounds": self.total_rounds,
            "max_reward": self.max_reward,
            "weights": self.weights.tolist(),
            "probabilities": self.probabilities.tolist(),
            "cumulative_rewards": self.cumulative_rewards.tolist(),
            "pull_counts": self.pull_counts.tolist(),
            "completed_rounds": len(self.history)
        }

    def load_state(self, state: Dict[str, Any]):
        """
        Restore bandit state from checkpoint.

        Args:
            state: Dictionary from save_state()
        """
        # Validate state
        if state["num_arms"] != self.num_arms:
            raise ValueError(f"State num_arms ({state['num_arms']}) != bandit num_arms ({self.num_arms})")

        # Restore hyperparameters
        self.eta = state["eta"]
        self.gamma = state["gamma"]

        # Restore weights and statistics
        self.weights = np.array(state["weights"], dtype=np.float64)
        self.probabilities = np.array(state["probabilities"], dtype=np.float64)
        self.cumulative_rewards = np.array(state["cumulative_rewards"], dtype=np.float64)
        self.pull_counts = np.array(state["pull_counts"], dtype=np.int32)

        # History is not restored (can be rebuilt from prompts if needed)
        self.history = []

        completed = state.get("completed_rounds", 0)
        self.logger.info(f"Restored EXP3-IX state: {completed} rounds completed, "
                        f"eta={self.eta:.6f}, gamma={self.gamma:.6f}")


if __name__ == "__main__":
    # Simple test
    logging.basicConfig(level=logging.DEBUG)

    print("Testing EXP3IXBandit...")
    bandit = EXP3IXBandit(num_arms=5, total_rounds=50, max_reward=1.0)

    print(f"Auto-calculated hyperparameters:")
    print(f"  eta (learning rate): {bandit.eta:.6f}")
    print(f"  gamma (implicit exploration): {bandit.gamma:.6f}")
    print(f"  max_reward: {bandit.max_reward}")

    # Simulate 50 rounds where arm 0 is best
    for _ in range(50):
        arm = bandit.select_arm()
        if arm == 0:
            reward = 0.75  # High reward (low loss = 0.25)
        else:
            reward = 0.25  # Low reward (high loss = 0.75)
        bandit.update(arm, reward)

    stats = bandit.get_stats()
    print(f"\nAfter 50 rounds:")
    print(f"Best arm: {stats['best_arm']}")
    print(f"Probabilities: {[f'{p:.3f}' for p in stats['probabilities']]}")
    print(f"Pull counts: {stats['pull_counts']}")
    print(f"Average rewards: {[f'{r:.2f}' for r in stats['average_rewards']]}")

    ranking = bandit.get_arm_ranking()
    print(f"\nArm ranking by avg reward:")
    for i, arm in enumerate(ranking[:3]):
        print(f"  {i+1}. Arm {arm['arm_idx']}: avg={arm['average_reward']:.2f}, pulls={arm['pull_count']}")
