"""
Random agent — selects uniformly at random from legal actions.

This is the sanity-check baseline; every other agent should beat it
(eventually).
"""

from __future__ import annotations

import random
from typing import Any

from agents.base import BaseAgent


class RandomAgent(BaseAgent):
    """Picks a legal action uniformly at random.

    Parameters
    ----------
    name:
        Human-readable identifier (default: ``"random"``).
    seed:
        Optional RNG seed for reproducibility.
    """

    def __init__(self, name: str = "random", seed: int | None = None) -> None:
        super().__init__(name)
        self._rng = random.Random(seed)

    def choose_action(self, state: Any) -> int:
        """Return a random legal action index."""
        actions = state.legal_actions()
        if not actions:
            raise ValueError("choose_action called on a state with no legal actions")
        return self._rng.choice(actions)
