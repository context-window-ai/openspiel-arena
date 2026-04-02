"""
arena.result — MatchResult and TournamentResult data models
=============================================================
These are the core data types that flow from the match runner to the rating
engine and analysis layer.  Both are serialisable to / from JSON.

Construction styles accepted by ``MatchResult``:

    # Canonical: supply the winner's name (or None for a draw)
    MatchResult(agent_a="alice", agent_b="bob", winner="alice")

    # Alternate: supply outcome from agent_a's perspective
    MatchResult(agent_a="alice", agent_b="bob", outcome="win")
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Literal

# Sentinel object that lets us distinguish "winner not supplied" from
# the intentional value "winner=None" (which means a draw).
_MISSING = object()


class MatchResult:
    """Record of a single completed match between two agents.

    Parameters
    ----------
    game_name:
        Name of the OpenSpiel game (e.g. ``"tic_tac_toe"``).
    agent_a:
        Name of the agent playing as player 0.
    agent_b:
        Name of the agent playing as player 1.
    winner:
        Name of the winning agent, or ``None`` for a draw.  Mutually
        exclusive with *outcome*; if both are given *winner* takes
        precedence.
    outcome:
        ``"win"``, ``"loss"``, or ``"draw"`` from *agent_a*'s perspective.
        Resolved to a *winner* value automatically.
    match_id:
        Unique identifier (auto-generated UUID if omitted).
    num_moves:
        Number of half-moves played.  If omitted, inferred from ``len(moves)``.
    returns:
        Raw per-player payoffs from OpenSpiel.
    timestamp:
        UTC ISO-8601 string (auto-generated if omitted).
    moves:
        Full move history as a list of action indices.
    """

    __slots__ = (
        "game_name",
        "agent_a",
        "agent_b",
        "winner",
        "match_id",
        "num_moves",
        "returns",
        "timestamp",
        "moves",
        # New fields for canonical results logging
        "seed",
        "agent_a_side",
        "agent_b_side",
        "invalid_move_retries",
        "agent_a_latency_ms",
        "agent_b_latency_ms",
        "termination_reason",
    )

    def __init__(
        self,
        game_name: str = "",
        agent_a: str = "",
        agent_b: str = "",
        winner: str | None | object = _MISSING,
        outcome: Literal["win", "loss", "draw"] | None = None,
        match_id: str | None = None,
        num_moves: int | None = None,
        returns: list[float] | None = None,
        timestamp: str | None = None,
        moves: list[int] | None = None,
        # New fields for canonical results logging
        seed: int | None = None,
        agent_a_side: int | None = None,
        agent_b_side: int | None = None,
        invalid_move_retries: int = 0,
        agent_a_latency_ms: float | None = None,
        agent_b_latency_ms: float | None = None,
        termination_reason: str = "normal",
    ) -> None:
        self.game_name = game_name
        self.agent_a = agent_a
        self.agent_b = agent_b
        self.match_id = match_id or str(uuid.uuid4())
        self.returns = list(returns) if returns is not None else []
        self.timestamp = timestamp or datetime.now(timezone.utc).isoformat()
        self.moves = list(moves) if moves is not None else []

        # num_moves: explicit value wins; otherwise infer from move list
        self.num_moves = num_moves if num_moves is not None else len(self.moves)

        # New fields
        self.seed = seed
        self.agent_a_side = agent_a_side
        self.agent_b_side = agent_b_side
        self.invalid_move_retries = invalid_move_retries
        self.agent_a_latency_ms = agent_a_latency_ms
        self.agent_b_latency_ms = agent_b_latency_ms
        self.termination_reason = termination_reason

        # Resolve winner from whichever argument was provided
        if winner is not _MISSING:
            # Explicit winner= (including winner=None for draws)
            self.winner: str | None = winner  # type: ignore[assignment]
        elif outcome is not None:
            if outcome == "win":
                self.winner = agent_a
            elif outcome == "loss":
                self.winner = agent_b
            else:  # "draw"
                self.winner = None
        else:
            # Neither supplied → default to draw
            self.winner = None

    # ------------------------------------------------------------------
    # Derived helpers
    # ------------------------------------------------------------------

    @property
    def is_draw(self) -> bool:
        """Return ``True`` when the match ended in a draw."""
        return self.winner is None

    @property
    def outcome(self) -> Literal["win", "loss", "draw"]:
        """Return ``"win"``, ``"loss"``, or ``"draw"`` from *agent_a*'s perspective."""
        if self.winner is None:
            return "draw"
        return "win" if self.winner == self.agent_a else "loss"

    def resolved_outcome(self) -> Literal["win", "loss", "draw"]:
        """Alias for :attr:`outcome`; used by the rating engines."""
        return self.outcome

    @property
    def played_at(self) -> str:
        """Alias for :attr:`timestamp` (ISO-8601 UTC string)."""
        return self.timestamp

    @property
    def run_id(self) -> str:
        """Alias for :attr:`match_id` for canonical schema compatibility."""
        return self.match_id

    @property
    def game(self) -> str:
        """Alias for :attr:`game_name` for canonical schema compatibility."""
        return self.game_name

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialise to a plain dict suitable for ``json.dump``."""
        return {
            "game_name": self.game_name,
            "agent_a": self.agent_a,
            "agent_b": self.agent_b,
            "winner": self.winner,
            "match_id": self.match_id,
            "num_moves": self.num_moves,
            "returns": self.returns,
            "timestamp": self.timestamp,
            "moves": self.moves,
            # New fields
            "seed": self.seed,
            "agent_a_side": self.agent_a_side,
            "agent_b_side": self.agent_b_side,
            "invalid_move_retries": self.invalid_move_retries,
            "agent_a_latency_ms": self.agent_a_latency_ms,
            "agent_b_latency_ms": self.agent_b_latency_ms,
            "termination_reason": self.termination_reason,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MatchResult":
        """Deserialise from a plain dict produced by ``to_dict``.

        Any ``outcome`` convenience key (if present) is ignored because
        ``winner`` is the canonical stored field.
        """
        cleaned = {k: v for k, v in data.items() if k != "outcome"}
        # Handle missing new fields with defaults for backward compatibility
        defaults = {
            "seed": None,
            "agent_a_side": None,
            "agent_b_side": None,
            "invalid_move_retries": 0,
            "agent_a_latency_ms": None,
            "agent_b_latency_ms": None,
            "termination_reason": "normal",
        }
        for key, default_value in defaults.items():
            if key not in cleaned:
                cleaned[key] = default_value
        return cls(**cleaned)

    # ------------------------------------------------------------------
    # Magic methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"MatchResult(game={self.game_name!r}, "
            f"a={self.agent_a!r}, b={self.agent_b!r}, "
            f"outcome={self.outcome!r})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MatchResult):
            return NotImplemented
        return self.to_dict() == other.to_dict()


# ---------------------------------------------------------------------------
# TournamentResult
# ---------------------------------------------------------------------------


class TournamentResult:
    """Aggregated results from a complete tournament.

    Attributes
    ----------
    matches:
        All ``MatchResult`` objects produced during the tournament.
    game_name:
        Name of the game played.
    """

    def __init__(
        self,
        matches: list[MatchResult] | None = None,
        game_name: str = "",
    ) -> None:
        self.matches: list[MatchResult] = list(matches or [])
        self.game_name = game_name

    def __len__(self) -> int:
        return len(self.matches)

    def __iter__(self):  # type: ignore[override]
        return iter(self.matches)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"TournamentResult(game={self.game_name!r}, "
            f"matches={len(self.matches)})"
        )

    def to_dict(self) -> dict:
        """Serialise to a plain dict."""
        return {
            "game_name": self.game_name,
            "matches": [m.to_dict() for m in self.matches],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TournamentResult":
        """Deserialise from a plain dict produced by ``to_dict``."""
        return cls(
            game_name=data.get("game_name", ""),
            matches=[MatchResult.from_dict(m) for m in data.get("matches", [])],
        )
