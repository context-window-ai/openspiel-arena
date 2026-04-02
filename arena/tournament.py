"""
arena.tournament — full tournament runner
==========================================
Schedules and executes a complete tournament over a set of agents, then
returns a ``TournamentResult`` containing every ``MatchResult``.

Delegates individual game execution to ``arena.match.run_match`` and
match-up generation to ``arena.scheduler.round_robin``.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

from arena.match import run_match
from arena.result import MatchResult, TournamentResult
from arena.scheduler import round_robin

if TYPE_CHECKING:
    from agents.base import BaseAgent
    from games.base import GameWrapper

log = logging.getLogger(__name__)


def run_tournament(
    agents: "list[BaseAgent]",
    game: "GameWrapper",
    rounds: int = 1,
    results_dir: str | Path | None = None,
) -> list[MatchResult]:
    """Run a full round-robin tournament and persist results to disk.

    Parameters
    ----------
    agents:
        Participating agents (must contain at least 2).
    game:
        A ``GameWrapper``-compatible game object.
    rounds:
        Number of times to repeat the full round-robin schedule.
    results_dir:
        Directory where per-match JSON files are written.  Defaults to the
        ``ARENA_RESULTS_DIR`` environment variable or ``"results/"``.

    Returns
    -------
    list[MatchResult]
        All match results in the order they were played.
    """
    out_dir = Path(results_dir or os.getenv("ARENA_RESULTS_DIR", "results/"))
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = round_robin(agents, repeat=rounds)
    log.info(
        "Tournament: %d agents, %d matches on %s → %s",
        len(agents),
        len(pairs),
        game.name,
        out_dir,
    )

    results: list[MatchResult] = []
    for i, (a, b) in enumerate(pairs, start=1):
        log.info("[%d/%d] %s vs %s", i, len(pairs), a.name, b.name)
        result = run_match(a, b, game)
        results.append(result)

        # Persist to disk immediately so partial runs aren't lost.
        out_path = out_dir / f"{result.match_id}.json"
        out_path.write_text(json.dumps(result.to_dict(), indent=2))
        log.debug("Saved result → %s", out_path)

    log.info("Tournament complete. %d matches played.", len(results))
    return results
