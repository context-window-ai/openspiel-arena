"""
Tests for arena.scheduler — round_robin.
"""

from __future__ import annotations

import pytest

from agents.random_agent import RandomAgent
from arena.scheduler import round_robin


def _agents(n: int) -> list[RandomAgent]:
    return [RandomAgent(name=f"agent-{i}") for i in range(n)]


def test_round_robin_match_count():
    agents = _agents(3)
    pairs = round_robin(agents)
    # n*(n-1) with repeat=1
    assert len(pairs) == 3 * 2


def test_round_robin_repeat():
    agents = _agents(2)
    pairs = round_robin(agents, repeat=3)
    # 2*(2-1)*3 = 6
    assert len(pairs) == 6


def test_round_robin_no_self_play():
    agents = _agents(4)
    for a, b in round_robin(agents):
        assert a is not b


def test_round_robin_requires_two_agents():
    with pytest.raises(ValueError):
        round_robin(_agents(1))


def test_round_robin_pair_types():
    agents = _agents(2)
    for a, b in round_robin(agents):
        assert isinstance(a, RandomAgent)
        assert isinstance(b, RandomAgent)
