#!/usr/bin/env python3
"""
scripts.plot_elo_over_time — Elo Rating Evolution Visualization
================================================================
Generates a line chart showing how agent Elo ratings evolve over the course
of a tournament.

Features:
- Reads results from CSV files (canonical arena format)
- Computes Elo trajectories incrementally after each match
- Supports game and agent filtering
- Color scheme: search-based agents (random, mcts-*) in blue/grey tones;
  LLM-based agents (llm-*) in warm tones (orange/red/yellow)
- Outputs PNG chart and CSV of trajectories

CLI Usage:
    python3 scripts/plot_elo_over_time.py results/*.csv
    python3 scripts/plot_elo_over_time.py results/*.csv --output-dir output/
    python3 scripts/plot_elo_over_time.py results/*.csv --game breakthrough
    python3 scripts/plot_elo_over_time.py results/*.csv --agents mcts-500,random
    python3 scripts/plot_elo_over_time.py results/*.csv --k 20 --start 1000

References
----------
- Elo, A. E. (1978). *The Rating of Chessplayers, Past and Present*.
- https://en.wikipedia.org/wiki/Elo_rating_system
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Default constants
# ---------------------------------------------------------------------------

DEFAULT_RATING: float = 1500.0
K_FACTOR: float = 32.0


# ---------------------------------------------------------------------------
# Data structures for Elo trajectories
# ---------------------------------------------------------------------------


@dataclass
class EloTrajectory:
    """Elo rating history for a single agent over the course of a tournament.

    Attributes
    ----------
    agent:
        Agent name.
    ratings:
        List of Elo ratings, one per match index (0 = initial rating).
    """

    agent: str
    ratings: list[float] = field(default_factory=list)
    start_rating: float = DEFAULT_RATING

    def record(self, rating: float) -> None:
        """Append a rating snapshot."""
        self.ratings.append(rating)

    @property
    def final_rating(self) -> float:
        """Return the final (most recent) rating."""
        return self.ratings[-1] if self.ratings else self.start_rating


@dataclass
class EloHistory:
    """Complete Elo history for all agents across a tournament.

    Attributes
    ----------
    trajectories:
        Dict mapping agent name to its EloTrajectory.
    game:
        Name of the game (if filtered).
    total_matches:
        Total number of matches processed.
    """

    trajectories: dict[str, EloTrajectory] = field(default_factory=dict)
    game: str = ""
    total_matches: int = 0
    start_rating: float = DEFAULT_RATING

    def get_trajectory(self, agent: str) -> EloTrajectory:
        """Get or create a trajectory for an agent."""
        if agent not in self.trajectories:
            self.trajectories[agent] = EloTrajectory(
                agent=agent, start_rating=self.start_rating
            )
        return self.trajectories[agent]

    def to_csv_rows(self) -> list[dict[str, Any]]:
        """Convert to flat list of rows for CSV export.

        Returns
        -------
        list[dict]
            Each row has: match_index, agent, elo_rating
        """
        rows = []
        for agent, traj in sorted(self.trajectories.items()):
            for match_idx, rating in enumerate(traj.ratings):
                rows.append(
                    {
                        "match_index": match_idx,
                        "agent": agent,
                        "elo_rating": round(rating, 2),
                    }
                )
        return rows


# ---------------------------------------------------------------------------
# Elo computation
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


def compute_elo_trajectories(
    results: list[dict[str, Any]],
    k: float = K_FACTOR,
    start_rating: float = DEFAULT_RATING,
    game_filter: str | None = None,
    agent_filter: set[str] | None = None,
) -> EloHistory:
    """Compute Elo trajectories from match results.

    Processes matches in order, updating ratings after each game and
    recording a snapshot for all known agents.

    Parameters
    ----------
    results:
        List of row dictionaries from CSV files.
    k:
        K-factor controlling how much each result shifts ratings.
    start_rating:
        Starting rating for new agents.
    game_filter:
        If provided, only include results for this game.
    agent_filter:
        If provided, only track these agents. Matches involving other
        agents are still processed for Elo updates, but those agents
        are not included in the output trajectories.

    Returns
    -------
    EloHistory
        Complete history with trajectories for each agent.
    """
    history = EloHistory(start_rating=start_rating)
    current_ratings: dict[str, float] = {}

    def _get_rating(name: str) -> float:
        """Get current rating, initializing new agents at start_rating."""
        return current_ratings.setdefault(name, start_rating)

    # Track which agents we've seen (for initial snapshot)
    agents_seen: set[str] = set()

    match_idx = 0
    for row in results:
        # Apply game filter
        if game_filter:
            row_game = row.get("game", "")
            if row_game != game_filter:
                continue

        agent_a = row.get("agent_a", "")
        agent_b = row.get("agent_b", "")

        if not agent_a or not agent_b:
            continue

        # Determine outcome from winner field
        winner_raw = row.get("winner", "")
        is_draw = row.get("is_draw", "").lower() == "true" or winner_raw == ""

        if is_draw:
            outcome = "draw"
        elif winner_raw == agent_a:
            outcome = "win"
        else:
            outcome = "loss"

        # Get current ratings (initialize if first time seen)
        ra = _get_rating(agent_a)
        rb = _get_rating(agent_b)

        # Record initial ratings for new agents before the match
        for agent in [agent_a, agent_b]:
            if agent not in agents_seen:
                agents_seen.add(agent)
                traj = history.get_trajectory(agent)
                traj.record(current_ratings[agent])

        # Compute expected scores
        ea = expected_score(ra, rb)
        eb = expected_score(rb, ra)

        # Determine actual scores
        if outcome == "win":
            sa, sb = 1.0, 0.0
        elif outcome == "loss":
            sa, sb = 0.0, 1.0
        else:  # draw
            sa, sb = 0.5, 0.5

        # Update ratings
        current_ratings[agent_a] = ra + k * (sa - ea)
        current_ratings[agent_b] = rb + k * (sb - eb)

        match_idx += 1

        # Record post-match ratings for all seen agents
        for agent in agents_seen:
            traj = history.get_trajectory(agent)
            traj.record(current_ratings.get(agent, start_rating))

    # Apply agent filter if provided
    if agent_filter:
        filtered_trajectories = {
            agent: traj
            for agent, traj in history.trajectories.items()
            if agent in agent_filter
        }
        history.trajectories = filtered_trajectories

    # Set metadata
    history.total_matches = match_idx
    if results and not game_filter:
        history.game = results[0].get("game", "")

    return history


# ---------------------------------------------------------------------------
# CSV Loading
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


def load_all_results(*csv_paths: str | Path) -> list[dict[str, Any]]:
    """Load and concatenate results from multiple CSV files.

    Parameters
    ----------
    *csv_paths:
        One or more paths to CSV files.

    Returns
    -------
    list[dict]
        Combined list of all rows.
    """
    all_results: list[dict[str, Any]] = []
    for path in csv_paths:
        all_results.extend(load_results_from_csv(path))
    return all_results


# ---------------------------------------------------------------------------
# Color assignment
# ---------------------------------------------------------------------------


def get_agent_color(agent: str, agents: list[str]) -> tuple[float, float, float, float]:
    """Assign a color to an agent based on its type.

    Color scheme:
    - Search-based agents (random, mcts-*): blue/grey tones
    - LLM-based agents (llm-*): warm tones (orange/red/yellow)

    Parameters
    ----------
    agent:
        Agent name.
    agents:
        Sorted list of all agents (for consistent color assignment).

    Returns
    -------
    tuple
        RGBA color tuple.
    """
    # Determine agent type
    is_llm = agent.startswith("llm-") or "gpt" in agent.lower() or "claude" in agent.lower()
    is_search = agent.startswith("mcts") or agent == "random" or agent.startswith("mcts-")

    # Sort agents within each category for consistent colors
    llm_agents = sorted([a for a in agents if a.startswith("llm-") or "gpt" in a.lower() or "claude" in a.lower()])
    search_agents = sorted([a for a in agents if a.startswith("mcts") or a == "random" or a.startswith("mcts-")])
    other_agents = sorted([a for a in agents if a not in llm_agents and a not in search_agents])

    if is_llm:
        # Warm tones: orange, red, yellow, coral, gold
        warm_colors = [
            (1.0, 0.6, 0.2, 1.0),    # Orange
            (0.9, 0.3, 0.24, 1.0),   # Red
            (0.96, 0.87, 0.3, 1.0),  # Yellow/Gold
            (1.0, 0.5, 0.31, 1.0),   # Coral
            (0.85, 0.45, 0.2, 1.0),  # Burnt orange
            (0.93, 0.4, 0.36, 1.0),  # Tomato
            (1.0, 0.75, 0.0, 1.0),   # Amber
            (0.8, 0.3, 0.3, 1.0),    # Indian red
        ]
        idx = llm_agents.index(agent) % len(warm_colors)
        return warm_colors[idx]

    elif is_search:
        # Cool tones: blue, grey, teal
        cool_colors = [
            (0.2, 0.4, 0.8, 1.0),    # Blue
            (0.5, 0.5, 0.55, 1.0),   # Grey
            (0.25, 0.55, 0.65, 1.0), # Teal
            (0.3, 0.45, 0.7, 1.0),   # Steel blue
            (0.15, 0.35, 0.6, 1.0),  # Navy
            (0.4, 0.6, 0.8, 1.0),    # Sky blue
            (0.45, 0.45, 0.5, 1.0),  # Dim grey
            (0.2, 0.5, 0.6, 1.0),    # Dark cyan
        ]
        idx = search_agents.index(agent) % len(cool_colors)
        return cool_colors[idx]

    else:
        # Other agents: green/purple tones
        other_colors = [
            (0.3, 0.7, 0.4, 1.0),    # Green
            (0.6, 0.4, 0.7, 1.0),    # Purple
            (0.4, 0.75, 0.55, 1.0),  # Sea green
            (0.7, 0.5, 0.8, 1.0),    # Lavender
            (0.2, 0.6, 0.35, 1.0),   # Forest green
        ]
        idx = other_agents.index(agent) % len(other_colors)
        return other_colors[idx]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_elo_over_time(
    history: EloHistory,
    output_path: str | Path,
    title: str | None = None,
    figsize: tuple[int, int] = (12, 7),
    dpi: int = 150,
) -> None:
    """Plot Elo trajectories and save to file.

    Parameters
    ----------
    history:
        EloHistory with trajectories for each agent.
    output_path:
        Path to save the PNG file.
    title:
        Custom title for the plot.
    figsize:
        Figure size as (width, height).
    dpi:
        Resolution for the output image.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install it with: pip install -e '.[analysis]'"
        ) from exc

    if not history.trajectories:
        print("No agent trajectories to plot.")
        return

    # Get sorted list of agents
    sorted_agents = sorted(history.trajectories.keys())
    n_agents = len(sorted_agents)

    # Create figure with appropriate size for slide resolution
    fig, ax = plt.subplots(figsize=figsize)

    # Plot each agent's trajectory
    for agent in sorted_agents:
        traj = history.trajectories[agent]
        if not traj.ratings:
            continue

        color = get_agent_color(agent, sorted_agents)
        ax.plot(
            range(len(traj.ratings)),
            traj.ratings,
            label=agent,
            color=color,
            linewidth=2,
            alpha=0.85,
        )

    # Set labels and title
    ax.set_xlabel("Match Number", fontsize=14, fontweight="bold")
    ax.set_ylabel("Elo Rating", fontsize=14, fontweight="bold")

    if title is None:
        game_name = history.game or "Tournament"
        title = f"Elo Rating Evolution — {game_name}"
    ax.set_title(title, fontsize=16, fontweight="bold", pad=15)

    # Configure legend
    # Place legend outside plot if many agents
    if n_agents > 6:
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1),
            fontsize=11,
            framealpha=0.95,
        )
        plt.tight_layout(rect=[0, 0, 0.85, 1])
    else:
        ax.legend(loc="best", fontsize=11, framealpha=0.95)
        plt.tight_layout()

    # Grid
    ax.grid(True, alpha=0.3, linestyle="--")

    # Ensure x-axis starts at 0
    ax.set_xlim(left=0)

    # Make tick labels readable
    ax.tick_params(axis="both", labelsize=11)

    # Save figure
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Saved Elo chart to {output_path}")


def save_trajectory_csv(
    history: EloHistory,
    output_path: str | Path,
) -> None:
    """Save Elo trajectories to a CSV file.

    Parameters
    ----------
    history:
        EloHistory with trajectories for each agent.
    output_path:
        Path to save the CSV file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = history.to_csv_rows()

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["match_index", "agent", "elo_rating"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved Elo trajectory CSV to {output_path}")


# ---------------------------------------------------------------------------
# CLI utilities
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


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------


def main(args: list[str] | None = None) -> int:
    """CLI entry point for plotting Elo over time.

    Usage:
        python3 scripts/plot_elo_over_time.py results/*.csv
        python3 scripts/plot_elo_over_time.py results/*.csv --output-dir output/
        python3 scripts/plot_elo_over_time.py results/*.csv --game breakthrough
        python3 scripts/plot_elo_over_time.py results/*.csv --agents mcts-500,random
        python3 scripts/plot_elo_over_time.py results/*.csv --k 20 --start 1000

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
        description="Generate Elo rating evolution chart from match results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 scripts/plot_elo_over_time.py results/*.csv
  python3 scripts/plot_elo_over_time.py results/*.csv --game breakthrough
  python3 scripts/plot_elo_over_time.py results/*.csv --agents mcts-500,llm-gpt-4
  python3 scripts/plot_elo_over_time.py results/*.csv --k 20 --start 1000
  python3 scripts/plot_elo_over_time.py results/*.csv --output-dir output/
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
        "--agents",
        type=str,
        default=None,
        help="Comma-separated list of agents to include (others filtered out)",
    )
    parser.add_argument(
        "--k",
        type=float,
        default=K_FACTOR,
        help=f"Elo K-factor (default: {K_FACTOR})",
    )
    parser.add_argument(
        "--start",
        type=float,
        default=DEFAULT_RATING,
        help=f"Starting Elo rating (default: {DEFAULT_RATING})",
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
        # Load all results
        all_results = load_all_results(*csv_paths)
        print(f"Loaded {len(all_results)} match records")

        # Parse agent filter
        agent_filter = None
        if parsed.agents:
            agent_filter = set(a.strip() for a in parsed.agents.split(","))

        # Compute trajectories
        history = compute_elo_trajectories(
            all_results,
            k=parsed.k,
            start_rating=parsed.start,
            game_filter=parsed.game,
            agent_filter=agent_filter,
        )

        if not history.trajectories:
            print("Error: No agent trajectories computed.", file=sys.stderr)
            return 1

        print(f"Computed trajectories for {len(history.trajectories)} agents")
        print(f"Total matches processed: {history.total_matches}")

        # Show final ratings
        sorted_agents = sorted(
            history.trajectories.keys(),
            key=lambda a: history.trajectories[a].final_rating,
            reverse=True,
        )
        print("\nFinal Elo Ratings:")
        for agent in sorted_agents:
            traj = history.trajectories[agent]
            print(f"  {agent}: {traj.final_rating:.1f}")

        # Create output directory
        output_dir = Path(parsed.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate output filenames
        game_suffix = f"_{parsed.game}" if parsed.game else ""
        png_path = output_dir / f"elo_over_time{game_suffix}.png"
        csv_path = output_dir / f"elo_over_time{game_suffix}.csv"

        # Plot chart
        plot_elo_over_time(history, output_path=png_path)

        # Save trajectory CSV
        save_trajectory_csv(history, output_path=csv_path)

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
