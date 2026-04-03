"""
scripts/run_tournament.py — CLI entry-point for openspiel-arena
================================================================
Launch a round-robin tournament between a configurable set of agents on a
chosen OpenSpiel game, persist results to disk, and print an Elo rating table.

Usage
-----
Direct::

    python scripts/run_tournament.py --game tic_tac_toe --rounds-per-pairing 2

Via the installed ``arena`` console script (after ``pip install -e .``)::

    arena --game tic_tac_toe --rounds-per-pairing 2 --results-dir my_results/
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

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


def _get_game(game_name: str):
    """Factory function to create a game wrapper by name."""
    if game_name == "tic_tac_toe":
        from games.tic_tac_toe import TicTacToeGame

        return TicTacToeGame()
    elif game_name == "breakthrough":
        from games.breakthrough import BreakthroughGame

        return BreakthroughGame()
    else:
        console.print(f"[red]Unsupported game: {game_name!r}[/red]")
        console.print("Currently supported games: tic_tac_toe, breakthrough")
        sys.exit(1)


def _build_agents(game, mcts_sims: int, llm_models: tuple) -> list:
    """Build the roster of agents for the tournament."""
    from agents.random_agent import RandomAgent

    agents = [
        RandomAgent(name="random", seed=0),
    ]

    # Add MCTS agents
    try:
        from agents.mcts_agent import MCTSAgent

        agents.append(
            MCTSAgent(
                game=game,
                num_simulations=mcts_sims,
                player_id=0,
                name=f"mcts-{mcts_sims}",
                seed=42,
            )
        )
        agents.append(
            MCTSAgent(
                game=game,
                num_simulations=mcts_sims,
                player_id=1,
                name=f"mcts-{mcts_sims}-p1",
                seed=7,
            )
        )
    except ImportError:
        console.print("[yellow]MCTSAgent not available, skipping MCTS agents[/yellow]")

    for llm_model in llm_models:
        try:
            from agents.llm_agent import LLMAgent, LLMAgentConfig
            from agents.prompts import PromptStyle
            config = LLMAgentConfig(model=llm_model, prompt_style=PromptStyle.BOARD_SUMMARY_THEN_CHOICE, memory_turns=1)
            slug = llm_model.split("/")[-1]
            agents.append(LLMAgent(config=config, name=f"llm-{slug}"))
        except Exception as exc:
            console.print(f"[yellow]LLM agent unavailable: {exc}[/yellow]")

    return agents


@click.command()
@click.option(
    "--game",
    default="tic_tac_toe",
    show_default=True,
    help="OpenSpiel game name to use for all matches.",
)
@click.option(
    "--rounds-per-pairing",
    default=2,
    show_default=True,
    type=int,
    help="How many matches per unordered agent pair (2 = once per side).",
)
@click.option(
    "--results-dir",
    default=None,
    help="Directory for match result files (default: $ARENA_RESULTS_DIR or results/).",
)
@click.option(
    "--mcts-sims",
    default=100,
    show_default=True,
    type=int,
    help="MCTS rollouts per move for the MCTS agent.",
)
@click.option(
    "--llm-model",
    multiple=True,
    help="LLM model identifier (repeatable). e.g. --llm-model openai/gpt-4o-mini --llm-model openai/gpt-5-mini",
)
@click.option(
    "--log-level",
    default=None,
    help="Logging verbosity (default: $ARENA_LOG_LEVEL or INFO).",
)
@click.option(
    "--run-id",
    default=None,
    help="Unique identifier for this tournament run (auto-generated if omitted).",
)
def main(
    game: str,
    rounds_per_pairing: int,
    results_dir: str | None,
    mcts_sims: int,
    llm_model: tuple,
    log_level: str | None,
    run_id: str | None,
) -> None:
    """Run an openspiel-arena tournament and print Elo ratings."""

    level = log_level or os.getenv("ARENA_LOG_LEVEL", "INFO")
    _configure_logging(level)

    # ------------------------------------------------------------------
    # Build game wrapper
    # ------------------------------------------------------------------
    game_obj = _get_game(game)

    # ------------------------------------------------------------------
    # Build agent roster
    # ------------------------------------------------------------------
    agents = _build_agents(game_obj, mcts_sims, llm_model)

    console.rule("[bold blue]openspiel-arena[/bold blue]")
    console.print(f"Game              : [cyan]{game_obj.name}[/cyan]")
    console.print(f"Agents            : {', '.join(a.name for a in agents)}")
    console.print(f"Rounds per pairing: {rounds_per_pairing}")
    console.print()

    # ------------------------------------------------------------------
    # Determine output directory
    # ------------------------------------------------------------------
    out_dir = Path(results_dir or os.getenv("ARENA_RESULTS_DIR", "results/"))

    # ------------------------------------------------------------------
    # Run tournament
    # ------------------------------------------------------------------
    from arena.tournament import run_tournament

    tournament_result, manifest = run_tournament(
        game=game_obj,
        agents=agents,
        rounds_per_pairing=rounds_per_pairing,
        results_dir=out_dir,
        run_id=run_id,
    )

    # ------------------------------------------------------------------
    # Display manifest summary
    # ------------------------------------------------------------------
    console.print()
    console.print("[bold]Tournament Summary[/bold]")
    console.print(f"  Run ID           : {manifest.run_id}")
    console.print(f"  Total matches    : {manifest.total_matches}")
    console.print(f"  Completed        : {manifest.completed_matches}")
    console.print(f"  Failed           : {manifest.failed_matches}")
    console.print(f"  Duration         : {manifest.duration_seconds:.1f}s")
    if manifest.results_path:
        console.print(f"  Results CSV      : {manifest.results_path}")
    console.print()

    # ------------------------------------------------------------------
    # Compute and display Elo ratings
    # ------------------------------------------------------------------
    from ratings.elo import update_elo

    # Only use completed matches for rating
    completed_matches = [
        m for m in tournament_result.matches
        if not m.termination_reason.startswith("error")
    ]

    if completed_matches:
        ratings = update_elo(completed_matches)

        match_counts: dict[str, int] = {}
        for r in completed_matches:
            match_counts[r.agent_a] = match_counts.get(r.agent_a, 0) + 1
            match_counts[r.agent_b] = match_counts.get(r.agent_b, 0) + 1

        table = Table(title="Final Elo Ratings", show_header=True)
        table.add_column("Agent", style="cyan")
        table.add_column("Elo", justify="right", style="green")
        table.add_column("Matches", justify="right")

        for agent_name, elo in sorted(ratings.items(), key=lambda x: -x[1]):
            table.add_row(
                agent_name, f"{elo:.1f}", str(match_counts.get(agent_name, 0))
            )

        console.print(table)
    else:
        console.print("[yellow]No completed matches to rate.[/yellow]")

    n = len(tournament_result.matches)
    console.print(f"\n[dim]{n} match(es) written to: {out_dir}[/dim]")

    # Print manifest path
    if manifest.results_path:
        manifest_path = Path(out_dir) / f"manifest_{manifest.run_id}.json"
        console.print(f"[dim]Manifest saved to: {manifest_path}[/dim]")


if __name__ == "__main__":
    main()
