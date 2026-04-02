"""
Random agent — selects uniformly at random from legal actions.

This is the sanity-check baseline; every other agent should beat it
(eventually).
"""

from __future__ import annotations

import random
from typing import Any

from agents.base import ActionContext, BaseAgent


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
        self._seed = seed

    @property
    def seed(self) -> int | None:
        """Return the RNG seed (for serialization)."""
        return self._seed

    def select_action(
        self,
        state_view: Any,
        legal_actions: list[int],
        context: ActionContext | None = None,
    ) -> int:
        """Return a random legal action index."""
        if not legal_actions:
            raise ValueError("select_action called with no legal actions")
        return self._rng.choice(legal_actions)
