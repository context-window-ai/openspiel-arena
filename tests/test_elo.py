"""
Tests for ratings.elo — expected_score and update_elo.
"""

from __future__ import annotations

import pytest

from ratings.elo import DEFAULT_RATING, expected_score, update_elo
from arena.result import MatchResult


def test_expected_score_equal_ratings():
    assert expected_score(1500, 1500) == pytest.approx(0.5)


def test_expected_score_higher_beats_lower():
    assert expected_score(1600, 1400) > 0.5


def test_expected_score_lower_vs_higher():
    assert expected_score(1400, 1600) < 0.5


def test_update_elo_winner_gains(sample_results):
    ratings = update_elo(sample_results)
    assert "alpha" in ratings
    assert "beta" in ratings


def test_update_elo_empty_results():
    ratings = update_elo([])
    assert ratings == {}


def test_update_elo_default_rating():
    r = MatchResult(agent_a="x", agent_b="y", game_name="g", outcome="win", returns=[1.0, -1.0])
    ratings = update_elo([r])
    assert ratings["x"] > DEFAULT_RATING
    assert ratings["y"] < DEFAULT_RATING


def test_update_elo_draw_minimal_change():
    r = MatchResult(agent_a="x", agent_b="y", game_name="g", outcome="draw", returns=[0.0, 0.0])
    ratings = update_elo([r])
    # Equal starting ratings + draw → no change
    assert ratings["x"] == pytest.approx(DEFAULT_RATING, abs=1e-6)
    assert ratings["y"] == pytest.approx(DEFAULT_RATING, abs=1e-6)
