"""
games.breakthrough — OpenSpiel breakthrough adapter
=====================================================
Wraps ``pyspiel.load_game("breakthrough")`` with the ``GameWrapper`` protocol
so the arena and agents never need to touch the raw pyspiel API.

Breakthrough is a two-player strategy game where each player tries to get
a pawn to the opponent's back row or capture all opponent pieces.

Board representation
--------------------
OpenSpiel renders the board with rows numbered 1-N from bottom to top
and columns labeled a-h (or more for larger boards). Player 0 (black)
starts at the top and moves down; player 1 (white) starts at the bottom
and moves up.

Default board (8x8)::

    8bbbbbbbb
    7bbbbbbbb
    6........
    5........
    4........
    3........
    2wwwwwwww
    1wwwwwwww
     abcdefgh

Action format
-------------
Actions are encoded as integers. The string representation is like "a7a6"
meaning "move from a7 to a6". Capture moves are indicated with an asterisk,
e.g., "a7b6*".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BreakthroughAction:
    """Human-readable action metadata for Breakthrough.

    Attributes
    ----------
    action_id : int
        The integer action ID used by OpenSpiel.
    from_square : str
        The source square in algebraic notation (e.g., "a7").
    to_square : str
        The destination square in algebraic notation (e.g., "a6").
    is_capture : bool
        True if this move captures an opponent piece.
    notation : str
        Full move notation (e.g., "a7a6" or "a7b6*").
    """

    action_id: int
    from_square: str
    to_square: str
    is_capture: bool
    notation: str


class BreakthroughGame:
    """
    Adapter around the OpenSpiel ``breakthrough`` game.

    Parameters
    ----------
    columns : int, optional
        Number of columns on the board (default 8).
    rows : int, optional
        Number of rows on the board (default 8).

    Attributes
    ----------
    name : str
        Always ``"breakthrough"``.
    num_players : int
        Always ``2``.
    columns : int
        Number of columns on the board.
    rows : int
        Number of rows on the board.
    """

    name: str = "breakthrough"
    num_players: int = 2

    def __init__(self, columns: int = 8, rows: int = 8) -> None:
        try:
            import pyspiel  # type: ignore[import]
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "pyspiel is required to use BreakthroughGame. "
                "Install it with: pip install open_spiel"
            ) from exc

        self.columns = columns
        self.rows = rows
        self._game = pyspiel.load_game(
            "breakthrough", {"columns": columns, "rows": rows}
        )

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
        In Breakthrough, draws are not possible.
        """
        return list(state.returns())

    def state_string(self, state: Any) -> str:
        """Return a human-readable board representation."""
        return str(state)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def current_player(self, state: Any) -> int:
        """Return the index (0 or 1) of the player to move.

        Player 0 is black (starts at top, moves down).
        Player 1 is white (starts at bottom, moves up).
        """
        return state.current_player()

    def observation_string(self, state: Any, player: int) -> str:
        """Return the observation string for *player* (for LLM prompts)."""
        return state.observation_string(player)

    def action_to_string(self, state: Any, action: int) -> str:
        """Convert an action ID to its string representation.

        Example: 98 -> "a7a6" or "a7b6*" for captures.
        """
        return state.action_to_string(state.current_player(), action)

    def string_to_action(self, state: Any, notation: str) -> int:
        """Convert a move notation string to an action ID.

        Parameters
        ----------
        state : Any
            The current game state.
        notation : str
            Move notation like "a7a6" or "a7b6*".

        Returns
        -------
        int
            The action ID.
        """
        return state.string_to_action(notation)

    def get_action_metadata(self, state: Any, action: int) -> BreakthroughAction:
        """Get detailed metadata for an action.

        Parameters
        ----------
        state : Any
            The current game state.
        action : int
            The action ID.

        Returns
        -------
        BreakthroughAction
            Detailed action information for human display or LLM prompts.
        """
        notation = self.action_to_string(state, action)
        is_capture = notation.endswith("*")
        clean_notation = notation.rstrip("*")

        # Parse the move: e.g., "a7a6" -> from="a7", to="a6"
        # Format is always <col><row><col><row>
        from_square = clean_notation[:2]
        to_square = clean_notation[2:4]

        return BreakthroughAction(
            action_id=action,
            from_square=from_square,
            to_square=to_square,
            is_capture=is_capture,
            notation=notation,
        )

    def get_all_legal_actions_metadata(
        self, state: Any
    ) -> list[BreakthroughAction]:
        """Get metadata for all legal actions in the current state.

        Useful for generating action lists for LLM agents.

        Parameters
        ----------
        state : Any
            The current game state.

        Returns
        -------
        list[BreakthroughAction]
            Metadata for each legal action.
        """
        return [
            self.get_action_metadata(state, action)
            for action in self.legal_actions(state)
        ]

    def winner(self, state: Any) -> int | None:
        """Return the winner's player index, or None if game is not terminal.

        Parameters
        ----------
        state : Any
            The game state to check.

        Returns
        -------
        int | None
            Player index (0 or 1) of the winner, or None if not terminal.
        """
        if not self.is_terminal(state):
            return None
        returns = self.returns(state)
        if returns[0] > 0:
            return 0
        elif returns[1] > 0:
            return 1
        return None  # Draw (not possible in Breakthrough but handled)

    def player_name(self, player_id: int) -> str:
        """Return a human-readable player name.

        Parameters
        ----------
        player_id : int
            Player index (0 or 1).

        Returns
        -------
        str
            "Black" for player 0, "White" for player 1.
        """
        return "Black" if player_id == 0 else "White"

    def render_compact(self, state: Any) -> str:
        """Render a compact board suitable for prompts and debugging.

        This is the default state string from OpenSpiel, which shows
        the board with row numbers and column letters.

        Parameters
        ----------
        state : Any
            The game state to render.

        Returns
        -------
        str
            Compact board representation.
        """
        return self.state_string(state)

    def render_with_context(self, state: Any) -> str:
        """Render the board with additional context for LLM prompts.

        Includes the board, current player, and legal moves.

        Parameters
        ----------
        state : Any
            The game state to render.

        Returns
        -------
        str
            Board with context information.
        """
        lines = []
        lines.append(f"Current player: {self.player_name(self.current_player(state))}")
        lines.append("")
        lines.append(self.render_compact(state))

        if not self.is_terminal(state):
            lines.append("")
            lines.append("Legal moves:")
            actions = self.get_all_legal_actions_metadata(state)
            # Group by from_square for readability
            by_from: dict[str, list[BreakthroughAction]] = {}
            for action in actions:
                if action.from_square not in by_from:
                    by_from[action.from_square] = []
                by_from[action.from_square].append(action)

            for from_sq in sorted(by_from.keys()):
                moves = by_from[from_sq]
                move_strs = []
                for m in moves:
                    if m.is_capture:
                        move_strs.append(f"{m.to_square}*")
                    else:
                        move_strs.append(m.to_square)
                lines.append(f"  {from_sq} -> {', '.join(move_strs)}")
        else:
            winner = self.winner(state)
            if winner is not None:
                lines.append("")
                lines.append(f"Game over. Winner: {self.player_name(winner)}")

        return "\n".join(lines)
