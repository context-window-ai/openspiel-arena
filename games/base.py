"""
games.base — shared types and the GameWrapper protocol
=======================================================
Every concrete game adapter (e.g. ``games.tic_tac_toe``) must implement the
``GameWrapper`` protocol so the arena can call it in a uniform way.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class GameWrapper(Protocol):
    """Minimal interface every game adapter must expose."""

    name: str  # human-readable game name, e.g. "tic_tac_toe"
    num_players: int

    def new_state(self) -> Any:
        """Return a fresh OpenSpiel ``State`` object."""
        ...

    def legal_actions(self, state: Any) -> list[int]:
        """Return a list of legal action indices for the current player."""
        ...

    def apply_action(self, state: Any, action: int) -> Any:
        """Apply *action* to *state* and return the (possibly mutated) state."""
        ...

    def is_terminal(self, state: Any) -> bool:
        """Return ``True`` when the game has ended."""
        ...

    def returns(self, state: Any) -> list[float]:
        """Return per-player payoffs; only valid on terminal states."""
        ...

    def state_string(self, state: Any) -> str:
        """Return a human-readable string representation of *state*."""
        ...
