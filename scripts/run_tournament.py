"""
scripts/run_tournament.py — CLI entry-point for openspiel-arena
================================================================
Launch a round-robin tournament between a configurable set of agents on a
chosen OpenSpiel game, persist results to disk, and print an Elo rating table.

Usage
-----
Direct::

    python scripts/run_tournament.py --game tic_tac_toe --rounds 5

Via the installed ``arena`` console script (after ``pip install -e .``)::

    arena --game tic_tac_toe --rounds 5 --results-dir my_results/
"""

from __future__ import annotations

import logging
import os
import sys

import click
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

# Bootstrap: load .env before touching any project imports.
load_dotenv()

console = Console()


def _configure_logging(level: str) -> None:
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )


@click.command()
@click.option(
    "--game",
    default="tic_tac_toe",
    show_default=True,
    help="OpenSpiel game name to use for all matches.",
)
@click.option(
    "--rounds",
    default=1,
    show_default=True,
    type=int,
    help="How many times each ordered agent pair plays.",
)
@click.option(
    "--results-dir",
    default=None,
    help="Directory for match result JSON files (default: $ARENA_RESULTS_DIR or results/).",
)
@click.option(
    "--mcts-sims",
    default=100,
    show_default=True,
    type=int,
    help="MCTS rollouts per move for the MCTS agent.",
)
@click.option(
    "--log-level",
    default=None,
    help="Logging verbosity (default: $ARENA_LOG_LEVEL or INFO).",
)
def main(
    game: str,
    rounds: int,
    results_dir: str | None,
    mcts_sims: int,
    log_level: str | None,
) -> None:
    """Run an openspiel-arena tournament and print Elo ratings."""

    level = log_level or os.getenv("ARENA_LOG_LEVEL", "INFO")
    _configure_logging(level)

    # ------------------------------------------------------------------
    # Build game wrapper
    # ------------------------------------------------------------------
    if game == "tic_tac_toe":
        from games.tic_tac_toe import TicTacToeGame
        game_obj = TicTacToeGame()
    else:
        console.print(f"[red]Unsupported game: {game!r}[/red]")
        console.print("Currently supported games: tic_tac_toe")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Build agent roster
    # ------------------------------------------------------------------
    from agents.random_agent import RandomAgent
    from agents.mcts_agent import MCTSAgent

    agents = [
        RandomAgent(name="random", seed=0),
        MCTSAgent(
            game=game_obj,
            num_simulations=mcts_sims,
            player_id=0,
            name=f"mcts-{mcts_sims}",
            seed=42,
        ),
        MCTSAgent(
            game=game_obj,
            num_simulations=mcts_sims,
            player_id=1,
            name=f"mcts-{mcts_sims}-p1",
            seed=7,
        ),
    ]

    console.rule("[bold blue]openspiel-arena[/bold blue]")
    console.print(f"Game      : [cyan]{game_obj.name}[/cyan]")
    console.print(f"Agents    : {', '.join(a.name for a in agents)}")
    console.print(f"Rounds    : {rounds}")
    console.print()

    # ------------------------------------------------------------------
    # Run tournament
    # ------------------------------------------------------------------
    from arena.tournament import run_tournament

    out = results_dir or os.getenv("ARENA_RESULTS_DIR", "results/")
    results = run_tournament(agents, game_obj, rounds=rounds, results_dir=out)

    # ------------------------------------------------------------------
    # Compute and display Elo ratings
    # ------------------------------------------------------------------
    from ratings.elo import update_elo

    ratings = update_elo(results)

    table = Table(title="Final Elo Ratings", show_header=True)
    table.add_column("Agent", style="cyan")
    table.add_column("Elo", justify="right", style="green")
    table.add_column("Matches", justify="right")

    match_counts: dict[str, int] = {}
    for r in results:
        match_counts[r.agent_a] = match_counts.get(r.agent_a, 0) + 1
        match_counts[r.agent_b] = match_counts.get(r.agent_b, 0) + 1

    for agent_name, elo in sorted(ratings.items(), key=lambda x: -x[1]):
        table.add_row(agent_name, f"{elo:.1f}", str(match_counts.get(agent_name, 0)))

    console.print()
    console.print(table)
    console.print(f"\n[dim]Results written to: {out}[/dim]")


if __name__ == "__main__":
    main()
