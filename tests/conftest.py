"""
tests/conftest.py — shared pytest fixtures
"""

from __future__ import annotations

import pytest

from arena.result import MatchResult


@pytest.fixture()
def sample_match_result() -> MatchResult:
    """Return a minimal MatchResult suitable for unit tests."""
    return MatchResult(
        game_name="tic_tac_toe",
        agent_a="agent-a",
        agent_b="agent-b",
        winner="agent-a",
        moves=[0, 4, 1, 5, 2],
    )


@pytest.fixture()
def sample_results() -> list[MatchResult]:
    """Return a small list of MatchResults covering win / loss / draw."""
    return [
        MatchResult(
            game_name="tic_tac_toe",
            agent_a="alpha",
            agent_b="beta",
            winner="alpha",
            moves=[0, 4, 1, 5, 2],
        ),
        MatchResult(
            game_name="tic_tac_toe",
            agent_a="beta",
            agent_b="alpha",
            winner="alpha",   # alpha wins again (as player 1 this time → outcome="loss" for beta)
            moves=[0, 4, 1, 5, 2],
        ),
        MatchResult(
            game_name="tic_tac_toe",
            agent_a="alpha",
            agent_b="beta",
            winner=None,  # draw
            moves=[0, 4, 1, 5, 2, 3, 6, 7, 8],
        ),
    ]
