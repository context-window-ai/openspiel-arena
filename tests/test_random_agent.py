"""
Tests for agents.random_agent — RandomAgent.
"""

from __future__ import annotations

import pytest

from agents.random_agent import RandomAgent


class _FakeState:
    """Minimal OpenSpiel-like State stub."""

    def __init__(self, actions: list[int]) -> None:
        self._actions = actions

    def legal_actions(self) -> list[int]:
        return list(self._actions)


def test_random_agent_returns_legal_action():
    agent = RandomAgent(seed=0)
    state = _FakeState([0, 1, 2, 3, 4])
    action = agent.choose_action(state)
    assert action in state.legal_actions()


def test_random_agent_raises_on_empty():
    agent = RandomAgent()
    with pytest.raises(ValueError, match="no legal actions"):
        agent.choose_action(_FakeState([]))


def test_random_agent_reproducible():
    a = RandomAgent(seed=42)
    b = RandomAgent(seed=42)
    state = _FakeState(list(range(9)))
    assert a.choose_action(state) == b.choose_action(state)


def test_random_agent_name():
    agent = RandomAgent(name="my-random")
    assert agent.name == "my-random"
