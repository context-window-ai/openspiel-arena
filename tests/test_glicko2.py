"""
Tests for ratings.glicko2 — Glicko2Rating and update_glicko2.
"""

from __future__ import annotations

import pytest

from ratings.glicko2 import (
    DEFAULT_DISPLAY_RATING,
    DEFAULT_DISPLAY_RD,
    Glicko2Rating,
    update_glicko2,
)
from arena.result import MatchResult


def test_default_rating_display_values():
    r = Glicko2Rating.default()
    assert r.display_rating == pytest.approx(DEFAULT_DISPLAY_RATING, abs=1.0)
    assert r.display_rd == pytest.approx(DEFAULT_DISPLAY_RD, abs=1.0)


def test_update_glicko2_winner_gains():
    results = [
        MatchResult(agent_a="strong", agent_b="weak", game_name="g", outcome="win", returns=[1.0, -1.0]),
    ]
    ratings = update_glicko2(results)
    assert ratings["strong"].display_rating > DEFAULT_DISPLAY_RATING
    assert ratings["weak"].display_rating < DEFAULT_DISPLAY_RATING


def test_update_glicko2_draw_near_equal():
    results = [
        MatchResult(agent_a="x", agent_b="y", game_name="g", outcome="draw", returns=[0.0, 0.0]),
    ]
    ratings = update_glicko2(results)
    assert abs(ratings["x"].display_rating - DEFAULT_DISPLAY_RATING) < 50
    assert abs(ratings["y"].display_rating - DEFAULT_DISPLAY_RATING) < 50


def test_update_glicko2_rd_decreases_after_games():
    results = [
        MatchResult(agent_a="a", agent_b="b", game_name="g", outcome="win", returns=[1.0, -1.0]),
    ]
    ratings = update_glicko2(results)
    assert ratings["a"].display_rd < DEFAULT_DISPLAY_RD
    assert ratings["b"].display_rd < DEFAULT_DISPLAY_RD


def test_update_glicko2_empty():
    ratings = update_glicko2([])
    assert ratings == {}
