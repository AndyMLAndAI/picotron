"""Small illustrative reward functions for the GRPO callable interface."""

from __future__ import annotations


def completion_length_reward(_: str, completion: str) -> float:
    """Return the completion's character count as a deliberately simple reward."""

    return float(len(completion))


def keyword_match_reward(keyword: str):
    """Build a case-insensitive binary reward function for one keyword."""

    if not keyword:
        raise ValueError("keyword must be non-empty.")
    normalized_keyword = keyword.casefold()

    def reward(_: str, completion: str) -> float:
        return float(normalized_keyword in completion.casefold())

    return reward
