"""
ratings.payoff_matrix — Pairwise payoff matrix from match results
==================================================================
Aggregates head-to-head CSV results into a pairwise payoff matrix showing
win rates between agents.

Features:
- Reads results from CSV files (canonical arena format)
- Computes win rate of row agent vs column agent
- Supports multiple games and agents
- JSON and table output formats

CLI Usage:
    python3 -m ratings.payoff_matrix results/some_file.csv
    python3 -m ratings.payoff_matrix results/*.csv --format table
    python3 -m ratings.payoff_matrix results/file.csv --output matrix.json

References
----------
- Omidshafiei, S., et al. (2019). "AlphaRank: Multi-Agent Evaluation by Evolution"
- https://arxiv.org/abs/1903.01373
"""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Data structures for payoff matrix
# ---------------------------------------------------------------------------


@dataclass
class HeadToHeadRecord:
    """Win/loss/draw record for one agent against another."""

    wins: int = 0
    losses: int = 0
    draws: int = 0

    @property
    def games(self) -> int:
        return self.wins + self.losses + self.draws

    @property
    def win_rate(self) -> float:
        """Win rate from the perspective of the row agent (0.0 to 1.0)."""
        if self.games == 0:
            return 0.0
        # Wins count 1, draws count 0.5
        return (self.wins + 0.5 * self.draws) / self.games

    def to_dict(self) -> dict[str, Any]:
        return {
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "games": self.games,
            "win_rate": round(self.win_rate, 4),
        }


@dataclass
class PayoffMatrix:
    """Pairwise payoff matrix for a set of agents.

    The matrix M[i][j] gives the win rate of agent i against agent j.
    Diagonal entries are 0.5 (self-play).
    """

    agents: list[str] = field(default_factory=list)
    game: str = ""
    _records: dict[tuple[str, str], HeadToHeadRecord] = field(default_factory=dict)

    def get_record(self, agent_a: str, agent_b: str) -> HeadToHeadRecord:
        """Get or create the head-to-head record for agent_a vs agent_b."""
        key = (agent_a, agent_b)
        if key not in self._records:
            self._records[key] = HeadToHeadRecord()
        return self._records[key]

    def add_result(
        self, agent_a: str, agent_b: str, outcome: str
    ) -> None:
        """Add a match result.

        Parameters
        ----------
        agent_a:
            Name of the first agent (row agent).
        agent_b:
            Name of the second agent (column agent).
        outcome:
            "win", "loss", or "draw" from agent_a's perspective.
        """
        record = self.get_record(agent_a, agent_b)
        if outcome == "win":
            record.wins += 1
        elif outcome == "loss":
            record.losses += 1
        else:  # draw
            record.draws += 1

        # Track agents
        if agent_a not in self.agents:
            self.agents.append(agent_a)
        if agent_b not in self.agents:
            self.agents.append(agent_b)

    def get_win_rate(self, agent_a: str, agent_b: str) -> float:
        """Get win rate of agent_a against agent_b.

        For self-play (agent_a == agent_b), returns 0.5.
        """
        if agent_a == agent_b:
            return 0.5
        record = self.get_record(agent_a, agent_b)
        if record.games == 0:
            # Check reverse direction
            reverse = self.get_record(agent_b, agent_a)
            if reverse.games == 0:
                return 0.5  # No data, assume even
            return 1.0 - reverse.win_rate
        return record.win_rate

    def to_matrix(self) -> list[list[float]]:
        """Return the payoff matrix as a 2D list of win rates.

        Row i, column j gives win rate of agents[i] vs agents[j].
        """
        # Sort agents for consistent ordering
        sorted_agents = sorted(self.agents)
        matrix = []
        for agent_a in sorted_agents:
            row = []
            for agent_b in sorted_agents:
                row.append(self.get_win_rate(agent_a, agent_b))
            matrix.append(row)
        return matrix

    def to_numpy(self) -> "numpy.ndarray":
        """Return the payoff matrix as a numpy array."""
        import numpy as np
        return np.array(self.to_matrix())

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary suitable for JSON export."""
        sorted_agents = sorted(self.agents)
        matrix = self.to_matrix()

        # Build detailed records
        records_dict = {}
        for (a, b), record in self._records.items():
            records_dict[f"{a}_vs_{b}"] = record.to_dict()

        return {
            "game": self.game,
            "agents": sorted_agents,
            "matrix": matrix,
            "records": records_dict,
        }

    def to_json(self, indent: int = 2) -> str:
        """Return JSON representation of the payoff matrix."""
        return json.dumps(self.to_dict(), indent=indent)

    def __str__(self) -> str:
        """Pretty-print the payoff matrix as a table."""
        sorted_agents = sorted(self.agents)
        if not sorted_agents:
            return "Empty PayoffMatrix"

        # Calculate column widths
        max_name = max(len(a) for a in sorted_agents)
        col_width = max(max_name, 8)  # At least 8 chars for numbers

        lines = []
        lines.append("=" * (col_width * (len(sorted_agents) + 1) + len(sorted_agents)))
        if self.game:
            lines.append(f"Payoff Matrix: {self.game}")
        else:
            lines.append("Payoff Matrix")
        lines.append("=" * (col_width * (len(sorted_agents) + 1) + len(sorted_agents)))

        # Header row
        header = " " * col_width
        for agent in sorted_agents:
            header += f" {agent[:col_width]:>{col_width}}"
        lines.append(header)
        lines.append("-" * len(header))

        # Data rows
        matrix = self.to_matrix()
        for i, agent_a in enumerate(sorted_agents):
            row = f"{agent_a[:col_width]:<{col_width}}"
            for j, win_rate in enumerate(matrix[i]):
                row += f" {win_rate:>{col_width}.3f}"
            lines.append(row)

        lines.append("-" * len(header))
        lines.append("Note: Values are win rate of row agent vs column agent")
        lines.append("=" * (col_width * (len(sorted_agents) + 1) + len(sorted_agents)))

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CSV Loading and Matrix Computation
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


def row_to_outcome(row: dict[str, Any]) -> tuple[str, str, str]:
    """Extract agent names and outcome from a CSV row.

    Parameters
    ----------
    row:
        Dictionary with canonical column names from the CSV.

    Returns
    -------
    tuple[str, str, str]
        (agent_a, agent_b, outcome) where outcome is "win", "loss", or "draw".
    """
    agent_a = row.get("agent_a", "")
    agent_b = row.get("agent_b", "")

    # Determine outcome from winner field
    winner_raw = row.get("winner", "")
    is_draw = row.get("is_draw", "").lower() == "true" or winner_raw == ""

    if is_draw:
        outcome = "draw"
    elif winner_raw == agent_a:
        outcome = "win"
    else:
        outcome = "loss"

    return agent_a, agent_b, outcome


def compute_payoff_matrix(
    results: list[dict[str, Any]],
    game: str = "",
) -> PayoffMatrix:
    """Compute a payoff matrix from loaded CSV results.

    Parameters
    ----------
    results:
        List of row dictionaries from CSV files.
    game:
        Optional game name to include in the matrix metadata.

    Returns
    -------
    PayoffMatrix
        Pairwise payoff matrix with win rates.
    """
    matrix = PayoffMatrix(game=game)

    for row in results:
        agent_a, agent_b, outcome = row_to_outcome(row)
        if agent_a and agent_b:
            matrix.add_result(agent_a, agent_b, outcome)

    return matrix


def compute_payoff_matrix_from_csv(
    *csv_paths: str | Path,
) -> PayoffMatrix:
    """Load results from CSV files and compute the payoff matrix.

    Parameters
    ----------
    *csv_paths:
        One or more paths to CSV files with match results.

    Returns
    -------
    PayoffMatrix
        Pairwise payoff matrix with win rates.
    """
    all_results: list[dict[str, Any]] = []
    game = ""

    for path in csv_paths:
        path = Path(path)
        results = load_results_from_csv(path)
        all_results.extend(results)

        # Extract game name from first result if not set
        if not game and results:
            game = results[0].get("game", "")

    return compute_payoff_matrix(all_results, game=game)


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------


def main(args: list[str] | None = None) -> int:
    """CLI entry point for computing payoff matrix from CSV files.

    Usage:
        python3 -m ratings.payoff_matrix results/some_file.csv [--format json|table]
        python3 -m ratings.payoff_matrix results/*.csv --output matrix.json

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
        print("Usage: python3 -m ratings.payoff_matrix <results.csv>... [--format json|table] [--output FILE]")
        print("\nComputes pairwise payoff matrix from match results.")
        print("\nOptions:")
        print("  --format FORMAT   Output format: 'json' or 'table' (default: table)")
        print("  --output FILE     Write output to file instead of stdout")
        print("\nExamples:")
        print("  python3 -m ratings.payoff_matrix results/matches.csv")
        print("  python3 -m ratings.payoff_matrix results/*.csv --format json")
        print("  python3 -m ratings.payoff_matrix results/file.csv --output matrix.json")
        return 0 if not args else 0

    # Parse arguments
    csv_paths: list[str] = []
    output_format = "table"
    output_file = None

    i = 0
    while i < len(args):
        if args[i] == "--format" and i + 1 < len(args):
            output_format = args[i + 1]
            if output_format not in ("json", "table"):
                print(f"Error: Unknown format '{output_format}'. Use 'json' or 'table'.", file=sys.stderr)
                return 1
            i += 2
        elif args[i] == "--output" and i + 1 < len(args):
            output_file = args[i + 1]
            i += 2
        elif args[i].startswith("--"):
            print(f"Unknown argument: {args[i]}", file=sys.stderr)
            return 1
        else:
            csv_paths.append(args[i])
            i += 1

    if not csv_paths:
        print("Error: No CSV files specified.", file=sys.stderr)
        return 1

    try:
        # Verify files exist
        for path in csv_paths:
            if not Path(path).exists():
                print(f"Error: File not found: {path}", file=sys.stderr)
                return 1

        matrix = compute_payoff_matrix_from_csv(*csv_paths)

        if output_format == "json":
            output = matrix.to_json()
        else:
            output = str(matrix)

        if output_file:
            Path(output_file).write_text(output, encoding="utf-8")
            print(f"Wrote payoff matrix to {output_file}")
        else:
            print(output)

        return 0
    except Exception as e:
        print(f"Error processing files: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
