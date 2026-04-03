"""
ratings.elo — Elo rating computation with CSV support and leaderboards
======================================================================
Processes match results from CSV files and computes Elo ratings with
comprehensive statistics.

Features:
- Reads results from CSV files (canonical arena format)
- Configurable K-factor and starting rating
- Handles wins, losses, and draws
- Produces leaderboards with full statistics

CLI Usage:
    python3 -m ratings.elo results/matches.csv

References
----------
- Elo, A. E. (1978). *The Rating of Chessplayers, Past and Present*.
- https://en.wikipedia.org/wiki/Elo_rating_system
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from arena.result import MatchResult

DEFAULT_RATING: float = 1500.0
K_FACTOR: float = 32.0


# ---------------------------------------------------------------------------
# Core Elo computation functions
# ---------------------------------------------------------------------------


def expected_score(rating_a: float, rating_b: float) -> float:
    """Return the expected score for player A against player B.

    The score is a probability in [0, 1] where 1 means certain win.

    Parameters
    ----------
    rating_a:
        Current Elo rating of player A.
    rating_b:
        Current Elo rating of player B.
    """
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def update_elo(
    results: list[MatchResult],
    initial_ratings: dict[str, float] | None = None,
    k: float = K_FACTOR,
    default: float = DEFAULT_RATING,
) -> dict[str, float]:
    """Compute Elo ratings after processing every result in order.

    Parameters
    ----------
    results:
        Match results in chronological order.
    initial_ratings:
        Seed ratings (agent name → float).  Agents not listed start at
        *default*.
    k:
        K-factor controlling how much each result shifts ratings.
    default:
        Starting rating for agents not present in *initial_ratings*.

    Returns
    -------
    dict[str, float]
        Current rating for every agent that appeared in *results*.
    """
    if not results:
        return {}

    ratings: dict[str, float] = dict(initial_ratings or {})

    def _get(name: str) -> float:
        return ratings.setdefault(name, default)

    for r in results:
        ra = _get(r.agent_a)
        rb = _get(r.agent_b)

        ea = expected_score(ra, rb)
        eb = expected_score(rb, ra)

        # Score from agent_a's perspective
        outcome = r.outcome
        if outcome == "win":
            sa, sb = 1.0, 0.0
        elif outcome == "loss":
            sa, sb = 0.0, 1.0
        else:  # draw
            sa, sb = 0.5, 0.5

        ratings[r.agent_a] = ra + k * (sa - ea)
        ratings[r.agent_b] = rb + k * (sb - eb)

    return ratings


# ---------------------------------------------------------------------------
# Leaderboard data structures
# ---------------------------------------------------------------------------


@dataclass
class MatchupStats:
    """Head-to-head record against a specific opponent."""

    opponent: str
    wins: int = 0
    losses: int = 0
    draws: int = 0

    @property
    def games(self) -> int:
        return self.wins + self.losses + self.draws

    @property
    def win_rate(self) -> float:
        if self.games == 0:
            return 0.0
        return (self.wins + 0.5 * self.draws) / self.games

    def to_dict(self) -> dict[str, Any]:
        return {
            "opponent": self.opponent,
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "games": self.games,
            "win_rate": self.win_rate,
        }


@dataclass
class AgentStats:
    """Full statistics for an agent including rating and matchup history."""

    name: str
    rating: float
    wins: int = 0
    losses: int = 0
    draws: int = 0
    matchups: dict[str, MatchupStats] = field(default_factory=dict)

    @property
    def games(self) -> int:
        return self.wins + self.losses + self.draws

    @property
    def win_rate(self) -> float:
        if self.games == 0:
            return 0.0
        return (self.wins + 0.5 * self.draws) / self.games

    def get_matchup(self, opponent: str) -> MatchupStats:
        """Get or create matchup stats for an opponent."""
        if opponent not in self.matchups:
            self.matchups[opponent] = MatchupStats(opponent=opponent)
        return self.matchups[opponent]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "rating": round(self.rating, 2),
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "games": self.games,
            "win_rate": round(self.win_rate, 4),
            "matchups": {
                opp: m.to_dict() for opp, m in sorted(self.matchups.items())
            },
        }


@dataclass
class Leaderboard:
    """Complete leaderboard with all agent statistics."""

    game: str = ""
    agents: dict[str, AgentStats] = field(default_factory=dict)
    total_matches: int = 0

    def get_agent(self, name: str, default_rating: float = DEFAULT_RATING) -> AgentStats:
        """Get or create agent stats."""
        if name not in self.agents:
            self.agents[name] = AgentStats(name=name, rating=default_rating)
        return self.agents[name]

    def sorted_by_rating(self) -> list[AgentStats]:
        """Return agents sorted by rating (descending)."""
        return sorted(self.agents.values(), key=lambda a: a.rating, reverse=True)

    def to_dict(self) -> dict[str, Any]:
        return {
            "game": self.game,
            "total_matches": self.total_matches,
            "agents": [a.to_dict() for a in self.sorted_by_rating()],
        }

    def __str__(self) -> str:
        """Pretty-print the leaderboard as a table."""
        lines = []
        lines.append("=" * 80)
        if self.game:
            lines.append(f"Leaderboard: {self.game}")
        else:
            lines.append("Leaderboard")
        lines.append("=" * 80)
        lines.append(
            f"{'Rank':<6}{'Agent':<20}{'Rating':<10}{'W':<6}{'L':<6}{'D':<6}"
            f"{'Games':<8}{'WinRate':<10}"
        )
        lines.append("-" * 80)

        for rank, agent in enumerate(self.sorted_by_rating(), 1):
            lines.append(
                f"{rank:<6}{agent.name:<20}{agent.rating:<10.1f}"
                f"{agent.wins:<6}{agent.losses:<6}{agent.draws:<6}"
                f"{agent.games:<8}{agent.win_rate:<10.2%}"
            )

        lines.append("-" * 80)
        lines.append(f"Total matches: {self.total_matches}")
        lines.append("=" * 80)

        # Add matchup summaries for each agent
        lines.append("\nMatchup Details:")
        lines.append("-" * 80)
        for agent in self.sorted_by_rating():
            if agent.matchups:
                lines.append(f"\n{agent.name}:")
                for opp, m in sorted(agent.matchups.items()):
                    lines.append(
                        f"  vs {opp}: {m.wins}W-{m.losses}L-{m.draws}D "
                        f"({m.win_rate:.1%})"
                    )

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CSV Reading and Leaderboard Computation
# ---------------------------------------------------------------------------


def load_results_from_csv(path: str | Path) -> list[dict[str, Any]]:
    """Load results from a CSV file in canonical arena format.

    Parameters
    ----------
    path:
        Path to the CSV file.

    Returns
    -------
    list[dict]
        List of row dictionaries with canonical column names.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def row_to_match_result(row: dict[str, Any]) -> MatchResult:
    """Convert a CSV row dict to a MatchResult object.

    Parameters
    ----------
    row:
        Dictionary with canonical column names from the CSV.

    Returns
    -------
    MatchResult
        MatchResult object suitable for Elo computation.
    """
    # Handle winner field: empty string means draw
    winner_raw = row.get("winner", "")
    winner = None if winner_raw == "" or row.get("is_draw", "").lower() == "true" else winner_raw

    return MatchResult(
        game_name=row.get("game", ""),
        agent_a=row.get("agent_a", ""),
        agent_b=row.get("agent_b", ""),
        winner=winner,
        match_id=row.get("run_id", ""),
        num_moves=int(row.get("num_moves", 0) or 0),
    )


def compute_leaderboard(
    results: list[MatchResult],
    k: float = K_FACTOR,
    default_rating: float = DEFAULT_RATING,
) -> Leaderboard:
    """Compute a full leaderboard with Elo ratings and statistics.

    Parameters
    ----------
    results:
        List of MatchResult objects in chronological order.
    k:
        K-factor for Elo updates.
    default_rating:
        Starting rating for new agents.

    Returns
    -------
    Leaderboard
        Complete leaderboard with ratings and matchup statistics.
    """
    if not results:
        return Leaderboard()

    # Determine game name from first result
    game_name = results[0].game_name if results else ""

    # Track ratings incrementally while also tracking stats
    ratings: dict[str, float] = {}
    leaderboard = Leaderboard(game=game_name, total_matches=len(results))

    for r in results:
        # Ensure both agents exist in leaderboard
        agent_a = leaderboard.get_agent(r.agent_a, default_rating)
        agent_b = leaderboard.get_agent(r.agent_b, default_rating)

        # Get current ratings (default if not yet set)
        ra = ratings.setdefault(r.agent_a, default_rating)
        rb = ratings.setdefault(r.agent_b, default_rating)

        # Compute expected scores
        ea = expected_score(ra, rb)
        eb = expected_score(rb, ra)

        # Determine actual scores
        outcome = r.outcome
        if outcome == "win":
            sa, sb = 1.0, 0.0
            agent_a.wins += 1
            agent_b.losses += 1
            # Update matchups
            agent_a.get_matchup(r.agent_b).wins += 1
            agent_b.get_matchup(r.agent_a).losses += 1
        elif outcome == "loss":
            sa, sb = 0.0, 1.0
            agent_a.losses += 1
            agent_b.wins += 1
            # Update matchups
            agent_a.get_matchup(r.agent_b).losses += 1
            agent_b.get_matchup(r.agent_a).wins += 1
        else:  # draw
            sa, sb = 0.5, 0.5
            agent_a.draws += 1
            agent_b.draws += 1
            # Update matchups
            agent_a.get_matchup(r.agent_b).draws += 1
            agent_b.get_matchup(r.agent_a).draws += 1

        # Update ratings
        ratings[r.agent_a] = ra + k * (sa - ea)
        ratings[r.agent_b] = rb + k * (sb - eb)

    # Set final ratings
    for name, rating in ratings.items():
        leaderboard.get_agent(name, default_rating).rating = rating

    return leaderboard


def compute_leaderboard_from_csv(
    csv_path: str | Path,
    k: float = K_FACTOR,
    default_rating: float = DEFAULT_RATING,
) -> Leaderboard:
    """Load results from CSV and compute the leaderboard.

    Parameters
    ----------
    csv_path:
        Path to the CSV file with match results.
    k:
        K-factor for Elo updates.
    default_rating:
        Starting rating for new agents.

    Returns
    -------
    Leaderboard
        Complete leaderboard with ratings and matchup statistics.
    """
    rows = load_results_from_csv(csv_path)
    results = [row_to_match_result(row) for row in rows]
    return compute_leaderboard(results, k=k, default_rating=default_rating)


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------


def main(args: list[str] | None = None) -> int:
    """CLI entry point for computing Elo ratings from a CSV file.

    Usage:
        python3 -m ratings.elo results/some_file.csv [--k 32] [--default 1500]

    Parameters
    ----------
    args:
        Command line arguments (defaults to sys.argv[1:]).

    Returns
    -------
    int
        Exit code (0 for success, 1 for error).
    """
    if args is None:
        args = sys.argv[1:]

    if not args or args[0] in ("-h", "--help"):
        print("Usage: python3 -m ratings.elo <results.csv> [--k K_FACTOR] [--default RATING]")
        print("\nComputes Elo ratings and prints a leaderboard from match results.")
        print("\nOptions:")
        print("  --k K_FACTOR     K-factor for Elo updates (default: 32)")
        print("  --default RATING Starting rating (default: 1500)")
        print("\nExample:")
        print("  python3 -m ratings.elo results/matches.csv --k 20 --default 1000")
        return 0 if not args else 0

    # Parse arguments
    csv_path = args[0]
    k = K_FACTOR
    default_rating = DEFAULT_RATING

    i = 1
    while i < len(args):
        if args[i] == "--k" and i + 1 < len(args):
            k = float(args[i + 1])
            i += 2
        elif args[i] == "--default" and i + 1 < len(args):
            default_rating = float(args[i + 1])
            i += 2
        else:
            print(f"Unknown argument: {args[i]}", file=sys.stderr)
            return 1

    try:
        leaderboard = compute_leaderboard_from_csv(
            csv_path, k=k, default_rating=default_rating
        )
        print(leaderboard)
        return 0
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error processing file: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
