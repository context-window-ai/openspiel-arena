"""
Smoke tests — verify that every top-level package is importable and that
key classes/functions are present at their expected import paths.
"""

from __future__ import annotations


def test_import_games():
    import games
    from games.base import GameWrapper
    from games.tic_tac_toe import TicTacToeGame

    assert GameWrapper is not None
    assert TicTacToeGame is not None


def test_import_agents():
    import agents
    from agents.base import BaseAgent
    from agents.random_agent import RandomAgent

    assert BaseAgent is not None
    assert RandomAgent is not None


def test_import_arena():
    import arena
    from arena.result import MatchResult
    from arena.match import run_match
    from arena.scheduler import round_robin
    from arena.tournament import run_tournament

    assert MatchResult is not None
    assert run_match is not None
    assert round_robin is not None
    assert run_tournament is not None


def test_import_ratings():
    import ratings
    from ratings.elo import update_elo, expected_score
    from ratings.glicko2 import Glicko2Rating, update_glicko2

    assert update_elo is not None
    assert expected_score is not None
    assert Glicko2Rating is not None
    assert update_glicko2 is not None


def test_import_analysis():
    import analysis
    from analysis.loader import load_results

    assert load_results is not None


def test_import_scripts():
    import scripts
    from scripts.run_tournament import main

    assert main is not None
