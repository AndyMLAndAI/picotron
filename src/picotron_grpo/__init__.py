"""Scriptable Group Relative Policy Optimization built on Picotron."""

from picotron_grpo.grpo_trainer import GRPOTrainer, group_relative_advantages, run_grpo
from picotron_grpo.reward_functions import completion_length_reward, keyword_match_reward

__all__ = [
    "GRPOTrainer",
    "completion_length_reward",
    "group_relative_advantages",
    "keyword_match_reward",
    "run_grpo",
]
