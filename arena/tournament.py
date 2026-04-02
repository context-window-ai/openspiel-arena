"""
arena.tournament — full tournament runner
==========================================
Schedules and executes a complete tournament over a set of agents, then
returns a ``TournamentResult`` containing every ``MatchResult``.

Delegates individual game execution to ``arena.match.run_match`` and
match-up generation to ``arena.scheduler.round_robin``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from arena.result import TournamentResult

if TYPE_CHECKING:
    from agents.base import BaseAgent
    from games.base import GameWrapper


def run_tournament(
    game: "GameWrapper",
    agents: "list[BaseAgent]",
    rounds: int = 1,
) -> TournamentResult:
    """Run a full round-robin tournament.

    Parameters
    ----------
    game:
        A ``GameWrapper``-compatible game object.
    agents:
        List of participating agents (must contain at least 2).
    rounds:
        Number of times to repeat the full round-robin schedule.

    Returns
    -------
    TournamentResult
        Aggregated results from every match.
    """
    from arena.match import run_match
    from arena.scheduler import round_robin

    pairs = round_robin(agents, repeat=rounds)
    result = TournamentResult()
    for a, b in pairs:
        result.matches.append(run_match(a, b, game))
    return result
