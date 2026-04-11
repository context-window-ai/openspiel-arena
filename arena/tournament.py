"""
arena.tournament — full tournament runner
==========================================
Schedules and executes a complete tournament over a set of agents, then
returns a ``TournamentResult`` containing every ``MatchResult``.

Delegates individual game execution to ``arena.match.run_match`` and
match-up generation to ``arena.scheduler.round_robin_with_side_swap``.

Features
--------
- Round-robin scheduling with balanced side swaps
- Persistent results via CSV logging
- Manifest JSON with tournament metadata (run_id, timing, agent list, etc.)
- Graceful handling of failed matches without crashing the tournament
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from arena.logging import ResultsWriter
from arena.result import MatchResult, TournamentResult
from arena.scheduler import count_pairings, round_robin_with_side_swap

if TYPE_CHECKING:
    from agents.base import BaseAgent
    from games.base import GameWrapper

logger = logging.getLogger(__name__)


@dataclass
class TournamentManifest:
    """Metadata record for a completed tournament run.

    Attributes
    ----------
    run_id:
        Unique identifier for this tournament run.
    game_name:
        Name of the game played.
    agents:
        List of agent names that participated.
    rounds_per_pairing:
        Number of matches scheduled per agent pair.
    total_matches:
        Total number of matches scheduled.
    completed_matches:
        Number of matches that completed successfully.
    failed_matches:
        Number of matches that failed.
    start_time:
        ISO-8601 UTC timestamp when tournament started.
    end_time:
        ISO-8601 UTC timestamp when tournament ended.
    duration_seconds:
        Total duration in seconds.
    results_path:
        Path to the results CSV file.
    failed_match_ids:
        List of match IDs that failed (if any).
    """

    run_id: str
    game_name: str
    agents: list[str]
    rounds_per_pairing: int
    total_matches: int
    completed_matches: int = 0
    failed_matches: int = 0
    start_time: str = ""
    end_time: str = ""
    duration_seconds: float = 0.0
    results_path: str = ""
    failed_match_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize to a plain dict for JSON output."""
        return {
            "run_id": self.run_id,
            "game_name": self.game_name,
            "agents": self.agents,
            "rounds_per_pairing": self.rounds_per_pairing,
            "total_matches": self.total_matches,
            "completed_matches": self.completed_matches,
            "failed_matches": self.failed_matches,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_seconds": self.duration_seconds,
            "results_path": self.results_path,
            "failed_match_ids": self.failed_match_ids,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TournamentManifest":
        """Deserialize from a plain dict."""
        return cls(
            run_id=data["run_id"],
            game_name=data["game_name"],
            agents=data["agents"],
            rounds_per_pairing=data["rounds_per_pairing"],
            total_matches=data["total_matches"],
            completed_matches=data.get("completed_matches", 0),
            failed_matches=data.get("failed_matches", 0),
            start_time=data.get("start_time", ""),
            end_time=data.get("end_time", ""),
            duration_seconds=data.get("duration_seconds", 0.0),
            results_path=data.get("results_path", ""),
            failed_match_ids=data.get("failed_match_ids", []),
        )

    def save(self, path: Path) -> None:
        """Save manifest to a JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "TournamentManifest":
        """Load manifest from a JSON file."""
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


def run_tournament(
    game: "GameWrapper",
    agents: "list[BaseAgent]",
    rounds_per_pairing: int = 2,
    *,
    results_dir: str | Path | None = None,
    run_id: str | None = None,
    transcripts_dir: str | Path | None = None,
) -> tuple[TournamentResult, TournamentManifest]:
    """Run a full round-robin tournament with side swaps.

    Schedules all-vs-all match-ups where each unordered agent pair plays
    ``rounds_per_pairing`` matches with balanced sides. Results are persisted
    to disk via CSV, and a manifest JSON is written with tournament metadata.

    Parameters
    ----------
    game:
        A ``GameWrapper``-compatible game object.
    agents:
        List of participating agents (must contain at least 2).
    rounds_per_pairing:
        Number of matches per unordered agent pair (default: 2 for side swap).
        E.g., 2 = once per side, 4 = twice per side.
    results_dir:
        Directory for output files (CSV results and manifest JSON).
        If None, results are not persisted to disk (manifest still tracks them).
    run_id:
        Unique identifier for this run. Auto-generated if omitted.
    transcripts_dir:
        If provided, per-move transcript JSONs are written to this directory.

    Returns
    -------
    tuple[TournamentResult, TournamentManifest]
        The aggregated match results and tournament metadata.

    Raises
    ------
    ValueError
        If fewer than 2 agents are provided.
    """
    from arena.match import run_match

    if len(agents) < 2:
        raise ValueError("Tournament requires at least 2 agents")

    # Initialize tournament metadata
    run_id = run_id or str(uuid.uuid4())
    start_time = datetime.now(timezone.utc)
    agent_names = [a.name for a in agents]

    # Generate schedule
    pairs = round_robin_with_side_swap(agents, rounds_per_pairing)
    total_matches = len(pairs)

    logger.info(
        f"Starting tournament {run_id}: "
        f"{len(agents)} agents, {total_matches} matches, "
        f"{rounds_per_pairing} rounds per pairing"
    )

    # Initialize results
    tournament_result = TournamentResult(game_name=game.name)
    failed_match_ids: list[str] = []

    # Set up CSV writer if we have an output directory
    writer = None
    results_path = None
    if results_dir is not None:
        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        results_path = results_dir / f"results_{run_id}.csv"
        writer = ResultsWriter(results_path)

    try:
        # Run all matches
        for idx, (agent_a, agent_b) in enumerate(pairs):
            match_num = idx + 1
            logger.debug(f"Match {match_num}/{total_matches}: {agent_a.name} vs {agent_b.name}")

            try:
                save_td = str(transcripts_dir) if transcripts_dir else None
                result = run_match(agent_a, agent_b, game, save_transcript_dir=save_td)
                tournament_result.matches.append(result)

                # Write to CSV immediately so data isn't lost on crash
                if writer is not None:
                    writer.write(result)

            except Exception as e:
                # Create a placeholder failed match result
                failed_id = str(uuid.uuid4())
                failed_match_ids.append(failed_id)
                logger.error(
                    f"Match {match_num} failed ({agent_a.name} vs {agent_b.name}): {e}"
                )

                # Record a failed match with error info
                failed_result = MatchResult(
                    game_name=game.name,
                    agent_a=agent_a.name,
                    agent_b=agent_b.name,
                    winner=None,
                    match_id=failed_id,
                    num_moves=0,
                    termination_reason=f"error: {type(e).__name__}",
                )
                tournament_result.matches.append(failed_result)

                if writer is not None:
                    writer.write(failed_result)

    finally:
        if writer is not None:
            writer.close()

    # Finalize manifest
    end_time = datetime.now(timezone.utc)
    duration = (end_time - start_time).total_seconds()
    completed = total_matches - len(failed_match_ids)

    manifest = TournamentManifest(
        run_id=run_id,
        game_name=game.name,
        agents=agent_names,
        rounds_per_pairing=rounds_per_pairing,
        total_matches=total_matches,
        completed_matches=completed,
        failed_matches=len(failed_match_ids),
        start_time=start_time.isoformat(),
        end_time=end_time.isoformat(),
        duration_seconds=duration,
        results_path=str(results_path) if results_path else "",
        failed_match_ids=failed_match_ids,
    )

    # Write manifest JSON
    if results_dir is not None:
        manifest_path = Path(results_dir) / f"manifest_{run_id}.json"
        manifest.save(manifest_path)
        logger.info(f"Tournament complete. Manifest saved to: {manifest_path}")

    logger.info(
        f"Tournament {run_id} finished: "
        f"{completed}/{total_matches} matches completed, "
        f"{len(failed_match_ids)} failed, "
        f"{duration:.1f}s"
    )

    return tournament_result, manifest


def load_tournament_results(results_dir: str | Path) -> TournamentResult:
    """Load tournament results from a CSV file.

    Scans the directory for results_*.csv files and loads all matches.

    Parameters
    ----------
    results_dir:
        Directory containing results CSV files.

    Returns
    -------
    TournamentResult
        Aggregated results from all found CSV files.
    """
    from arena.logging import load_results_csv

    results_dir = Path(results_dir)
    tournament_result = TournamentResult()

    for csv_file in results_dir.glob("results_*.csv"):
        rows = load_results_csv(csv_file)
        for row in rows:
            # Convert row back to MatchResult
            match = MatchResult(
                game_name=row.get("game", ""),
                agent_a=row.get("agent_a", ""),
                agent_b=row.get("agent_b", ""),
                winner=row.get("winner") or None,
                match_id=row.get("run_id", ""),
                num_moves=int(row.get("num_moves", 0)),
                seed=int(row["seed"]) if row.get("seed") else None,
                agent_a_side=int(row.get("agent_a_side", 0)),
                agent_b_side=int(row.get("agent_b_side", 1)),
                invalid_move_retries=int(row.get("invalid_move_retries", 0)),
                agent_a_latency_ms=float(row.get("agent_a_latency_ms", 0.0)) or None,
                agent_b_latency_ms=float(row.get("agent_b_latency_ms", 0.0)) or None,
                termination_reason=row.get("termination_reason", "normal"),
            )
            tournament_result.matches.append(match)

    return tournament_result
