"""
Tests for arena.result — MatchResult dataclass.
"""

from __future__ import annotations

from arena.result import MatchResult


def test_match_result_required_fields():
    r = MatchResult(
        agent_a="random",
        agent_b="mcts",
        game_name="tic_tac_toe",
        outcome="loss",          # agent_a lost → agent_b won
    )
    assert r.game_name == "tic_tac_toe"
    assert r.agent_a == "random"
    assert r.agent_b == "mcts"
    assert r.winner == "mcts"   # derived property
    assert r.is_draw is False


def test_match_result_draw():
    r = MatchResult(
        agent_a="random",
        agent_b="random",
        game_name="tic_tac_toe",
        outcome="draw",
    )
    assert r.is_draw is True
    assert r.winner is None


def test_match_result_default_moves():
    r = MatchResult(agent_a="a", agent_b="b", game_name="g", outcome="draw")
    assert r.moves == []


def test_match_result_with_moves(sample_match_result):
    assert sample_match_result.moves == [0, 4, 1, 5, 2]
    assert len(sample_match_result.moves) == 5
