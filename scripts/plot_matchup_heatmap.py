#!/usr/bin/env python3
"""
scripts.plot_matchup_heatmap — Agent vs Agent Win Rate Heatmap
===============================================================
Generates a heatmap visualization of pairwise win rates between agents.

Features:
- Reads results from CSV files (canonical arena format)
- Computes win rate of row agent vs column agent
- Supports game filtering and side-balanced aggregation
- Outputs PNG heatmap and CSV matrix
- Annotates cells with win rate and optional sample size

CLI Usage:
    python3 scripts/plot_matchup_heatmap.py results/*.csv
    python3 scripts/plot_matchup_heatmap.py results/*.csv --output-dir output/
    python3 scripts/plot_matchup_heatmap.py results/*.csv --game breakthrough
    python3 scripts/plot_matchup_heatmap.py results/*.csv --no-annotate
    python3 scripts/plot_matchup_heatmap.py results/*.csv --show-samples
    python3 scripts/plot_matchup_heatmap.py results/*.csv --side-balanced

References
----------
- Omidshafiei, S., et al. (2019). "AlphaRank: Multi-Agent Evaluation by Evolution"
- https://arxiv.org/abs/1903.01373
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Data structures for matchup matrix
# ---------------------------------------------------------------------------


@dataclass
class MatchupRecord:
    """Win/loss/draw record for one agent against another, with side tracking."""

    wins: int = 0
    losses: int = 0
    draws: int = 0
    # Track by which side the agent played (for side-balanced aggregation)
    wins_as_side_0: int = 0
    wins_as_side_1: int = 0
    losses_as_side_0: int = 0
    losses_as_side_1: int = 0
    draws_as_side_0: int = 0
    draws_as_side_1: int = 0

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
class MatchupMatrix:
    """Pairwise matchup matrix for a set of agents.

    The matrix M[i][j] gives the win rate of agent i against agent j.
    Diagonal entries are NaN (no self-play).
    """

    agents: list[str] = field(default_factory=list)
    game: str = ""
    _records: dict[tuple[str, str], MatchupRecord] = field(default_factory=dict)

    def get_record(self, agent_a: str, agent_b: str) -> MatchupRecord:
        """Get or create the head-to-head record for agent_a vs agent_b."""
        key = (agent_a, agent_b)
        if key not in self._records:
            self._records[key] = MatchupRecord()
        return self._records[key]

    def add_result(
        self,
        agent_a: str,
        agent_b: str,
        outcome: str,
        agent_a_side: int = 0,
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
        agent_a_side:
            Which side agent_a played (0 or 1).
        """
        record = self.get_record(agent_a, agent_b)

        if outcome == "win":
            record.wins += 1
            if agent_a_side == 0:
                record.wins_as_side_0 += 1
            else:
                record.wins_as_side_1 += 1
        elif outcome == "loss":
            record.losses += 1
            if agent_a_side == 0:
                record.losses_as_side_0 += 1
            else:
                record.losses_as_side_1 += 1
        else:  # draw
            record.draws += 1
            if agent_a_side == 0:
                record.draws_as_side_0 += 1
            else:
                record.draws_as_side_1 += 1

        # Track agents
        if agent_a not in self.agents:
            self.agents.append(agent_a)
        if agent_b not in self.agents:
            self.agents.append(agent_b)

    def get_win_rate(self, agent_a: str, agent_b: str, side_balanced: bool = False) -> float:
        """Get win rate of agent_a against agent_b.

        Parameters
        ----------
        agent_a:
            Row agent name.
        agent_b:
            Column agent name.
        side_balanced:
            If True, average win rate across both starting positions.

        Returns
        -------
        float
            Win rate (0.0 to 1.0), or NaN for self-play.
        """
        if agent_a == agent_b:
            return float("nan")

        record = self.get_record(agent_a, agent_b)
        reverse = self.get_record(agent_b, agent_a)

        if side_balanced:
            # Average win rate across both positions
            # Position 1: agent_a vs agent_b (agent_a as side 0/1)
            # Position 2: agent_b vs agent_a (agent_b as side 0/1)
            # We need to compute the average from both perspectives

            # Games where agent_a was the row agent
            games_a = record.games
            wins_a = record.wins + 0.5 * record.draws

            # Games where agent_b was the row agent (agent_a was column)
            games_b = reverse.games
            wins_b = reverse.wins + 0.5 * reverse.draws  # wins for agent_b

            total_games = games_a + games_b
            if total_games == 0:
                return float("nan")

            # agent_a's wins from games_a, agent_b's losses from games_b
            agent_a_wins = wins_a + (games_b - wins_b)
            return agent_a_wins / total_games
        else:
            if record.games == 0:
                # Check reverse direction
                if reverse.games == 0:
                    return float("nan")
                return 1.0 - reverse.win_rate
            return record.win_rate

    def get_sample_count(self, agent_a: str, agent_b: str) -> int:
        """Get total number of games between agent_a and agent_b."""
        if agent_a == agent_b:
            return 0
        record = self.get_record(agent_a, agent_b)
        reverse = self.get_record(agent_b, agent_a)
        return record.games + reverse.games

    def to_matrix(self, side_balanced: bool = False) -> list[list[float]]:
        """Return the matchup matrix as a 2D list of win rates.

        Row i, column j gives win rate of agents[i] vs agents[j].
        Diagonal is NaN.
        """
        sorted_agents = sorted(self.agents)
        matrix = []
        for agent_a in sorted_agents:
            row = []
            for agent_b in sorted_agents:
                row.append(self.get_win_rate(agent_a, agent_b, side_balanced))
            matrix.append(row)
        return matrix

    def to_numpy(self, side_balanced: bool = False) -> np.ndarray:
        """Return the matchup matrix as a numpy array."""
        return np.array(self.to_matrix(side_balanced))

    def to_sample_matrix(self) -> list[list[int]]:
        """Return sample count matrix as a 2D list."""
        sorted_agents = sorted(self.agents)
        matrix = []
        for agent_a in sorted_agents:
            row = []
            for agent_b in sorted_agents:
                row.append(self.get_sample_count(agent_a, agent_b))
            matrix.append(row)
        return matrix

    def to_dict(self, side_balanced: bool = False) -> dict[str, Any]:
        """Serialize to a dictionary suitable for CSV export."""
        sorted_agents = sorted(self.agents)
        matrix = self.to_matrix(side_balanced)

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


def row_to_outcome(row: dict[str, Any]) -> tuple[str, str, str, int]:
    """Extract agent names, outcome, and side from a CSV row.

    Parameters
    ----------
    row:
        Dictionary with canonical column names from the CSV.

    Returns
    -------
    tuple[str, str, str, int]
        (agent_a, agent_b, outcome, agent_a_side) where outcome is "win", "loss", or "draw".
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

    # Get side information
    try:
        agent_a_side = int(row.get("agent_a_side", 0))
    except (ValueError, TypeError):
        agent_a_side = 0

    return agent_a, agent_b, outcome, agent_a_side


def compute_matchup_matrix(
    results: list[dict[str, Any]],
    game: str = "",
    game_filter: str | None = None,
) -> MatchupMatrix:
    """Compute a matchup matrix from loaded CSV results.

    Parameters
    ----------
    results:
        List of row dictionaries from CSV files.
    game:
        Optional game name to include in the matrix metadata.
    game_filter:
        If provided, only include results for this game.

    Returns
    -------
    MatchupMatrix
        Pairwise matchup matrix with win rates.
    """
    matrix = MatchupMatrix(game=game)

    for row in results:
        # Apply game filter
        if game_filter:
            row_game = row.get("game", "")
            if row_game != game_filter:
                continue

        agent_a, agent_b, outcome, agent_a_side = row_to_outcome(row)
        if agent_a and agent_b:
            matrix.add_result(agent_a, agent_b, outcome, agent_a_side)

    return matrix


def compute_matchup_matrix_from_csv(
    *csv_paths: str | Path,
    game_filter: str | None = None,
) -> MatchupMatrix:
    """Load results from CSV files and compute the matchup matrix.

    Parameters
    ----------
    *csv_paths:
        One or more paths to CSV files with match results.
    game_filter:
        If provided, only include results for this game.

    Returns
    -------
    MatchupMatrix
        Pairwise matchup matrix with win rates.
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

    return compute_matchup_matrix(all_results, game=game, game_filter=game_filter)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_heatmap(
    matrix: MatchupMatrix,
    output_path: str | Path,
    side_balanced: bool = False,
    annotate: bool = True,
    show_samples: bool = False,
    title: str | None = None,
    figsize: tuple[int, int] | None = None,
) -> None:
    """Plot the matchup heatmap and save to file.

    Parameters
    ----------
    matrix:
        MatchupMatrix with win rate data.
    output_path:
        Path to save the PNG file.
    side_balanced:
        If True, use side-balanced win rates.
    annotate:
        If True, annotate cells with win rate values.
    show_samples:
        If True, show sample size (n=X) below win rate.
    title:
        Custom title for the plot.
    figsize:
        Figure size as (width, height).
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError as exc:
        raise ImportError(
            "matplotlib and seaborn are required for plotting. "
            "Install them with: pip install -e '.[analysis]'"
        ) from exc

    sorted_agents = sorted(matrix.agents)
    n = len(sorted_agents)

    if n == 0:
        print("No agents found in the data.")
        return

    # Get win rate matrix
    win_rates = np.array(matrix.to_matrix(side_balanced))

    # Get sample count matrix if needed
    sample_counts = None
    if show_samples:
        sample_counts = np.array(matrix.to_sample_matrix())

    # Determine figure size
    if figsize is None:
        size = max(8, n + 2)
        figsize = (size, size - 1)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Use RdYlGn colormap centered at 0.5 (diverging)
    cmap = sns.diverging_palette(10, 130, s=90, l=50, as_cmap=True)

    # Create heatmap mask for diagonal (NaN values)
    mask = np.isnan(win_rates)

    # Plot heatmap
    sns.heatmap(
        win_rates,
        annot=False,  # We'll add custom annotations
        cmap=cmap,
        center=0.5,
        vmin=0,
        vmax=1,
        square=True,
        mask=mask,
        cbar_kws={"label": "Win Rate (row vs column)"},
        ax=ax,
        linewidths=0.5,
        linecolor="gray",
    )

    # Set axis labels
    ax.set_xticks(np.arange(n) + 0.5)
    ax.set_yticks(np.arange(n) + 0.5)
    ax.set_xticklabels(sorted_agents, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(sorted_agents, rotation=0, fontsize=10)

    # Set title
    if title is None:
        game_name = matrix.game or "All Games"
        balance_str = " (side-balanced)" if side_balanced else ""
        title = f"Agent vs Agent Win Rates{balance_str}\n{game_name}"
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

    # Axis labels
    ax.set_xlabel("Opponent (column)", fontsize=12)
    ax.set_ylabel("Agent (row)", fontsize=12)

    # Add custom annotations
    if annotate:
        for i in range(n):
            for j in range(n):
                if mask[i, j]:
                    # Gray out diagonal
                    ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=True, color="#e0e0e0", lw=0))
                    ax.text(
                        j + 0.5,
                        i + 0.5,
                        "—",
                        ha="center",
                        va="center",
                        fontsize=10,
                        color="gray",
                    )
                else:
                    win_rate = win_rates[i, j]
                    if not np.isnan(win_rate):
                        # Main annotation: win rate percentage
                        text = f"{win_rate:.0%}"

                        # Add sample count if requested
                        if show_samples and sample_counts is not None:
                            n_samples = sample_counts[i, j]
                            if n_samples > 0:
                                text += f"\nn={n_samples}"

                        # Choose text color based on background
                        text_color = "white" if 0.3 < win_rate < 0.7 else "black"

                        ax.text(
                            j + 0.5,
                            i + 0.5,
                            text,
                            ha="center",
                            va="center",
                            fontsize=9 if show_samples else 10,
                            color=text_color,
                            fontweight="bold" if not show_samples else "normal",
                        )

    plt.tight_layout()

    # Save figure
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Saved heatmap to {output_path}")


def save_matrix_csv(
    matrix: MatchupMatrix,
    output_path: str | Path,
    side_balanced: bool = False,
) -> None:
    """Save the matchup matrix to a CSV file.

    Parameters
    ----------
    matrix:
        MatchupMatrix with win rate data.
    output_path:
        Path to save the CSV file.
    side_balanced:
        If True, use side-balanced win rates.
    """
    sorted_agents = sorted(matrix.agents)
    win_rates = matrix.to_matrix(side_balanced)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Header row
        writer.writerow(["Agent"] + sorted_agents)

        # Data rows
        for i, agent in enumerate(sorted_agents):
            row = [agent]
            for j, _ in enumerate(sorted_agents):
                val = win_rates[i][j]
                if isinstance(val, float) and (val != val):  # NaN check
                    row.append("")
                else:
                    row.append(f"{val:.4f}")
            writer.writerow(row)

    print(f"Saved matrix CSV to {output_path}")


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------


def expand_glob_paths(patterns: list[str]) -> list[str]:
    """Expand glob patterns to actual file paths.

    Parameters
    ----------
    patterns:
        List of file paths or glob patterns.

    Returns
    -------
    list[str]
        Expanded list of file paths.
    """
    from glob import glob

    expanded: list[str] = []
    for pattern in patterns:
        # Check if it's a glob pattern
        if "*" in pattern or "?" in pattern:
            matches = sorted(glob(pattern))
            expanded.extend(matches)
        else:
            expanded.append(pattern)

    return expanded


def main(args: list[str] | None = None) -> int:
    """CLI entry point for plotting matchup heatmap.

    Usage:
        python3 scripts/plot_matchup_heatmap.py results/*.csv
        python3 scripts/plot_matchup_heatmap.py results/*.csv --output-dir output/
        python3 scripts/plot_matchup_heatmap.py results/*.csv --game breakthrough
        python3 scripts/plot_matchup_heatmap.py results/*.csv --no-annotate
        python3 scripts/plot_matchup_heatmap.py results/*.csv --show-samples
        python3 scripts/plot_matchup_heatmap.py results/*.csv --side-balanced

    Parameters
    ----------
    args:
        Command line arguments (defaults to sys.argv[1:]).

    Returns
    -------
    int
        Exit code (0 for success, 1 for error).
    """
    parser = argparse.ArgumentParser(
        description="Generate agent vs agent win rate heatmap",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 scripts/plot_matchup_heatmap.py results/*.csv
  python3 scripts/plot_matchup_heatmap.py results/*.csv --game breakthrough
  python3 scripts/plot_matchup_heatmap.py results/*.csv --output-dir output/
  python3 scripts/plot_matchup_heatmap.py results/*.csv --side-balanced --show-samples
""",
    )

    parser.add_argument(
        "csv_paths",
        nargs="+",
        help="One or more CSV file paths (supports glob patterns)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/",
        help="Output directory for PNG and CSV files (default: output/)",
    )
    parser.add_argument(
        "--game",
        type=str,
        default=None,
        help="Filter results to a specific game",
    )
    parser.add_argument(
        "--no-annotate",
        action="store_true",
        default=False,
        help="Disable cell annotations (default: annotations on)",
    )
    parser.add_argument(
        "--show-samples",
        action="store_true",
        help="Show sample size (n=X) in cells",
    )
    parser.add_argument(
        "--side-balanced",
        action="store_true",
        help="Use side-balanced win rate aggregation",
    )

    parsed = parser.parse_args(args)

    # Expand glob patterns
    csv_paths = expand_glob_paths(parsed.csv_paths)

    if not csv_paths:
        print("Error: No CSV files found.", file=sys.stderr)
        return 1

    # Verify files exist
    for path in csv_paths:
        if not Path(path).exists():
            print(f"Error: File not found: {path}", file=sys.stderr)
            return 1

    print(f"Processing {len(csv_paths)} CSV file(s)...")

    try:
        # Compute matchup matrix
        matrix = compute_matchup_matrix_from_csv(*csv_paths, game_filter=parsed.game)

        if not matrix.agents:
            print("Error: No agents found in the data.", file=sys.stderr)
            return 1

        print(f"Found {len(matrix.agents)} agents: {', '.join(sorted(matrix.agents))}")

        # Create output directory
        output_dir = Path(parsed.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate output filenames
        game_suffix = f"_{parsed.game}" if parsed.game else ""
        balance_suffix = "_side_balanced" if parsed.side_balanced else ""
        png_path = output_dir / f"matchup_heatmap{game_suffix}{balance_suffix}.png"
        csv_path = output_dir / f"matchup_matrix{game_suffix}{balance_suffix}.csv"

        # Plot heatmap
        plot_heatmap(
            matrix,
            output_path=png_path,
            side_balanced=parsed.side_balanced,
            annotate=not parsed.no_annotate,
            show_samples=parsed.show_samples,
        )

        # Save matrix CSV
        save_matrix_csv(
            matrix,
            output_path=csv_path,
            side_balanced=parsed.side_balanced,
        )

        print(f"\nDone! Generated:")
        print(f"  - {png_path}")
        print(f"  - {csv_path}")

        return 0

    except Exception as e:
        print(f"Error processing files: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
