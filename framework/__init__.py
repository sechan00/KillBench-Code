"""
Framework module for AutoGuard EXP3 Defense System

Contains:
- Agent: Malicious agent simulation
- Scorer: Defense effectiveness evaluation (LLM-based)
- defender_llm: Defense prompt generator with iterative improvement (Phase 1)
- attacker_simulator: Attack simulation and testing (Phase 1)
- phase3_evaluator: Top-1 inference and deployment (Phase 3)
- adaptive_reward: Adaptive reward calculation (Phase 2)
- exp3_bandit: EXP3-IX bandit algorithm (Phase 2)
"""

# Phase 1 components
from .defender_llm import DefenderLLM
from .attacker_simulator import AttackerSimulator, MockAttackerSimulator

# Evaluation components
from .Scorer import Scorer

# Phase 2 components
from .adaptive_reward import AdaptiveRewardCalculator
from .exp3_bandit import EXP3IXBandit

# Phase 3 components
from .phase3_evaluator import Phase3Inference

# Agent utilities
from .Agent import run_gpt, run_llama

__all__ = [
    # Phase 1
    'DefenderLLM',
    'AttackerSimulator',
    'MockAttackerSimulator',
    # Evaluation
    'Scorer',
    # Phase 2
    'AdaptiveRewardCalculator',
    'EXP3IXBandit',
    # Phase 3
    'Phase3Inference',
    # Agent utilities
    'run_gpt',
    'run_llama',
]
