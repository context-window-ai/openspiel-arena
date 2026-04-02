"""
MCTS agent — wraps OpenSpiel's built-in Monte Carlo Tree Search evaluator.

See: open_spiel.python.algorithms.mcts
"""

from __future__ import annotations

from typing import Any

import numpy as np
from open_spiel.python.algorithms import mcts

from agents.base import ActionContext, BaseAgent


class MCTSAgent(BaseAgent):
    """Wraps OpenSpiel's MCTS bot.

    Parameters
    ----------
    game:
        A ``pyspiel.Game`` object (or compatible wrapper exposing the raw game
        via a ``game`` attribute).  The bot is re-created per game type.
    num_simulations:
        Number of MCTS rollouts per move (higher → stronger, slower).
    player_id:
        Which player index this agent controls (0 or 1).
    name:
        Human-readable identifier.
    seed:
        RNG seed for the MCTS bot.
    """

    def __init__(
        self,
        game: Any,
        num_simulations: int = 100,
        player_id: int = 0,
        name: str | None = None,
        seed: int = 42,
    ) -> None:
        label = name or f"mcts-{num_simulations}"
        super().__init__(label)
        self._player_id = player_id
        self._num_simulations = num_simulations
        self._seed = seed

        # Resolve the raw pyspiel.Game object if a wrapper was passed.
        raw_game = getattr(game, "_game", game)

        self._bot = mcts.MCTSBot(
            game=raw_game,
            uct_c=2,
            max_simulations=num_simulations,
            evaluator=mcts.RandomRolloutEvaluator(
                n_rollouts=1,
                random_state=np.random.RandomState(seed),
            ),
            random_state=np.random.RandomState(seed + 1),
        )

    @property
    def num_simulations(self) -> int:
        """Return the number of MCTS simulations per move."""
        return self._num_simulations

    @property
    def seed(self) -> int:
        """Return the RNG seed."""
        return self._seed

    def select_action(
        self,
        state_view: Any,
        legal_actions: list[int],
        context: ActionContext | None = None,
    ) -> int:
        """Return the MCTS-selected action."""
        # Resolve the raw pyspiel.State if a wrapper was passed.
        raw_state = getattr(state_view, "_state", state_view)
        return self._bot.step(raw_state)
