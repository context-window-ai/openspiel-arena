"""
Integration tests for arena.match — run_match.

Requires OpenSpiel.
"""

from __future__ import annotations

import pytest

pytest.importorskip("pyspiel", reason="open_spiel not installed")

from agents.random_agent import RandomAgent  # noqa: E402
from arena.match import run_match  # noqa: E402
from arena.result import MatchResult  # noqa: E402
from games.tic_tac_toe import TicTacToeGame  # noqa: E402


@pytest.fixture()
def game():
    return TicTacToeGame()


def test_run_match_returns_match_result(game):
    a = RandomAgent(seed=0)
    b = RandomAgent(seed=1)
    result = run_match(a, b, game)
    assert isinstance(result, MatchResult)


def test_run_match_agents_named_correctly(game):
    a = RandomAgent(name="player-0", seed=0)
    b = RandomAgent(name="player-1", seed=1)
    result = run_match(a, b, game)
    assert result.agent_a == "player-0"
    assert result.agent_b == "player-1"


def test_run_match_game_name(game):
    result = run_match(RandomAgent(seed=0), RandomAgent(seed=1), game)
    assert result.game_name == "tic_tac_toe"


def test_run_match_outcome_valid(game):
    result = run_match(RandomAgent(seed=0), RandomAgent(seed=1), game)
    assert result.outcome in ("win", "loss", "draw")


def test_run_match_winner_valid(game):
    """winner is agent_a name, agent_b name, or None for a draw."""
    a = RandomAgent(name="alice", seed=0)
    b = RandomAgent(name="bob", seed=1)
    result = run_match(a, b, game)
    assert result.winner in ("alice", "bob", None)


def test_run_match_moves_nonempty(game):
    result = run_match(RandomAgent(seed=0), RandomAgent(seed=1), game)
    assert len(result.moves) > 0


def test_run_match_is_draw_property(game):
    """is_draw should be True when winner is None."""
    result = run_match(RandomAgent(seed=0), RandomAgent(seed=1), game)
    assert result.is_draw == (result.winner is None)


def test_run_match_records_seed(game):
    """Test that seed is recorded in result."""
    result = run_match(RandomAgent(seed=0), RandomAgent(seed=1), game, seed=12345)
    assert result.seed == 12345


def test_run_match_records_agent_sides(game):
    """Test that agent sides are recorded in result."""
    result = run_match(RandomAgent(seed=0), RandomAgent(seed=1), game)
    assert result.agent_a_side == 0
    assert result.agent_b_side == 1


def test_run_match_records_latency(game):
    """Test that latency is recorded for both agents."""
    result = run_match(RandomAgent(seed=0), RandomAgent(seed=1), game)
    assert result.agent_a_latency_ms is not None
    assert result.agent_b_latency_ms is not None
    # RandomAgent should be very fast
    assert result.agent_a_latency_ms >= 0
    assert result.agent_b_latency_ms >= 0


def test_run_match_records_termination_reason(game):
    """Test that termination reason is recorded."""
    result = run_match(RandomAgent(seed=0), RandomAgent(seed=1), game)
    assert result.termination_reason == "normal"


def test_run_match_records_invalid_move_retries(game):
    """Test that invalid move retries are recorded."""
    result = run_match(RandomAgent(seed=0), RandomAgent(seed=1), game)
    # RandomAgent always returns valid moves
    assert result.invalid_move_retries == 0
