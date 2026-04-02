"""
games.tic_tac_toe — OpenSpiel tic-tac-toe adapter
===================================================
Wraps ``pyspiel.load_game("tic_tac_toe")`` with the ``GameWrapper`` protocol
so the arena and agents never need to touch the raw pyspiel API.

The class is intentionally thin: all game logic stays inside OpenSpiel.
"""

from __future__ import annotations

from typing import Any


class TicTacToeGame:
    """
    Adapter around the OpenSpiel ``tic_tac_toe`` game.

    Attributes
    ----------
    name : str
        Always ``"tic_tac_toe"``.
    num_players : int
        Always ``2``.
    """

    name: str = "tic_tac_toe"
    num_players: int = 2

    def __init__(self) -> None:
        try:
            import pyspiel  # type: ignore[import]
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "pyspiel is required to use TicTacToeGame. "
                "Install it with: pip install open_spiel"
            ) from exc

        self._game = pyspiel.load_game("tic_tac_toe")

    # ------------------------------------------------------------------
    # GameWrapper protocol
    # ------------------------------------------------------------------

    def new_state(self) -> Any:
        """Return a fresh initial state."""
        return self._game.new_initial_state()

    def legal_actions(self, state: Any) -> list[int]:
        """Return a list of legal action indices for the current player."""
        return list(state.legal_actions())

    def apply_action(self, state: Any, action: int) -> Any:
        """Apply *action* in-place and return the state (OpenSpiel mutates)."""
        state.apply_action(action)
        return state

    def is_terminal(self, state: Any) -> bool:
        """Return ``True`` when the game has ended."""
        return state.is_terminal()

    def returns(self, state: Any) -> list[float]:
        """
        Return per-player payoffs.

        Only valid after ``is_terminal()`` is ``True``.
        Values are in {-1.0, 0.0, 1.0} for loss / draw / win.
        """
        return list(state.returns())

    def state_string(self, state: Any) -> str:
        """Return a human-readable board representation."""
        return str(state)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def current_player(self, state: Any) -> int:
        """Return the index (0 or 1) of the player to move."""
        return state.current_player()

    def observation_string(self, state: Any, player: int) -> str:
        """Return the observation string for *player* (for LLM prompts)."""
        return state.observation_string(player)
