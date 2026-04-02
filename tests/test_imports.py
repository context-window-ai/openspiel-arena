"""
Smoke tests — verify that every top-level package is importable and that
key classes/functions are present at their expected import paths.

These tests do NOT require OpenSpiel or API keys.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Package-level imports
# ---------------------------------------------------------------------------

def test_import_games() -> None:
    import games
    from games.base import GameWrapper
    from games.tic_tac_toe import TicTacToeGame

    assert GameWrapper is not None
    assert TicTacToeGame is not None


def test_import_agents() -> None:
    import agents
    from agents.base import BaseAgent, Agent
    from agents.random_agent import RandomAgent

    assert BaseAgent is not None
    assert Agent is not None
    assert RandomAgent is not None


def test_import_arena() -> None:
    import arena
    from arena.result import MatchResult, TournamentResult
    from arena.match import run_match
    from arena.scheduler import round_robin
    from arena.tournament import run_tournament

    assert MatchResult is not None
    assert TournamentResult is not None
    assert run_match is not None
    assert round_robin is not None
    assert run_tournament is not None


def test_import_ratings() -> None:
    import ratings
    from ratings.elo import update_elo, expected_score
    from ratings.glicko2 import Glicko2Rating, update_glicko2

    assert update_elo is not None
    assert expected_score is not None
    assert Glicko2Rating is not None
    assert update_glicko2 is not None


def test_import_analysis() -> None:
    import analysis
    from analysis.loader import load_results

    assert load_results is not None


def test_import_scripts() -> None:
    import scripts
    from scripts.run_tournament import main

    assert main is not None


# ---------------------------------------------------------------------------
# Protocol / shape checks
# ---------------------------------------------------------------------------

def test_game_wrapper_protocol_shape() -> None:
    from games.base import GameWrapper
    assert callable(GameWrapper)


def test_agent_protocol_shape() -> None:
    from agents.base import Agent
    assert callable(Agent)


def test_match_result_roundtrip() -> None:
    """MatchResult serialises to dict and back without data loss."""
    from arena.result import MatchResult

    r = MatchResult(
        agent_a="mcts",
        agent_b="random",
        game_name="tic_tac_toe",
        outcome="win",
        returns=[1.0, -1.0],
        moves=[4, 0, 2, 6, 8],
    )
    d = r.to_dict()
    r2 = MatchResult.from_dict(d)

    assert r2.agent_a == "mcts"
    assert r2.agent_b == "random"
    assert r2.outcome == "win"
    assert r2.returns == [1.0, -1.0]
    assert r2.moves == [4, 0, 2, 6, 8]
    assert r2.match_id == r.match_id
