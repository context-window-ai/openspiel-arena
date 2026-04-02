"""
Tests for agents.registry and agents.base — agent interface and registry.
"""

from __future__ import annotations

import pytest

from agents.base import ActionContext, Agent, BaseAgent
from agents.random_agent import RandomAgent
from agents.registry import AgentConfig, AgentRegistry, get_default_registry


class _FakeState:
    """Minimal OpenSpiel-like State stub."""

    def __init__(self, actions: list[int]) -> None:
        self._actions = actions

    def legal_actions(self) -> list[int]:
        return list(self._actions)


# ---------------------------------------------------------------------------
# ActionContext tests
# ---------------------------------------------------------------------------


def test_action_context_defaults():
    ctx = ActionContext()
    assert ctx.game_name == ""
    assert ctx.player_id == 0
    assert ctx.turn_number == 0
    assert ctx.history == []
    assert ctx.extra == {}


def test_action_context_frozen():
    ctx = ActionContext(game_name="tic_tac_toe", player_id=1)
    assert ctx.game_name == "tic_tac_toe"
    assert ctx.player_id == 1
    with pytest.raises(AttributeError):
        ctx.player_id = 0  # type: ignore[misc]


def test_action_context_with_history():
    ctx = ActionContext(history=[0, 4, 1], turn_number=3)
    assert ctx.history == [0, 4, 1]
    assert ctx.turn_number == 3


# ---------------------------------------------------------------------------
# AgentConfig tests
# ---------------------------------------------------------------------------


def test_agent_config_basic():
    config = AgentConfig("random")
    assert config.agent_type == "random"
    assert config.name is None
    assert config.params == {}


def test_agent_config_with_params():
    config = AgentConfig("mcts", params={"num_simulations": 100, "seed": 42})
    assert config.agent_type == "mcts"
    assert config.params == {"num_simulations": 100, "seed": 42}


def test_agent_config_frozen():
    """AgentConfig is frozen - cannot reassign attributes."""
    config = AgentConfig("random", params={"seed": 42})
    with pytest.raises(AttributeError):
        config.name = "new_name"  # type: ignore[misc]
    with pytest.raises(AttributeError):
        config.params = {}  # type: ignore[misc]


def test_agent_config_registry_id_with_name():
    config = AgentConfig("mcts", name="fast", params={"num_simulations": 50})
    assert config.registry_id() == "mcts:fast"


def test_agent_config_registry_id_without_name():
    config1 = AgentConfig("random", params={"seed": 42})
    config2 = AgentConfig("random", params={"seed": 42})
    config3 = AgentConfig("random", params={"seed": 43})

    # Same params → same ID
    assert config1.registry_id() == config2.registry_id()
    # Different params → different ID
    assert config1.registry_id() != config3.registry_id()


def test_agent_config_registry_id_no_params():
    config = AgentConfig("random")
    assert config.registry_id() == "random"


def test_agent_config_display_name_with_name():
    config = AgentConfig("mcts", name="deep_blue")
    assert config.display_name() == "deep_blue"


def test_agent_config_display_name_without_params():
    config = AgentConfig("random")
    assert config.display_name() == "random"


def test_agent_config_display_name_with_params():
    config = AgentConfig("mcts", params={"num_simulations": 100, "seed": 42})
    # Params are sorted alphabetically
    assert "mcts" in config.display_name()
    assert "num_simulations=100" in config.display_name()
    assert "seed=42" in config.display_name()


def test_agent_config_serialization():
    config = AgentConfig("mcts", name="fast", params={"num_simulations": 100})
    data = config.to_dict()

    assert data["agent_type"] == "mcts"
    assert data["name"] == "fast"
    assert data["params"] == {"num_simulations": 100}

    restored = AgentConfig.from_dict(data)
    assert restored.agent_type == config.agent_type
    assert restored.name == config.name
    assert restored.params == config.params


def test_agent_config_different_configs_different_ids():
    """Two agent instances with different configs produce different IDs."""
    configs = [
        AgentConfig("mcts", name="mcts_fast", params={"num_simulations": 50}),
        AgentConfig("mcts", name="mcts_deep", params={"num_simulations": 500}),
        AgentConfig("llm", name="llm_zero_shot", params={"model": "gpt-4"}),
        AgentConfig("llm", name="llm_structured_choice_retry", params={"model": "gpt-4", "retry": True}),
    ]

    ids = [c.registry_id() for c in configs]
    # All IDs should be unique
    assert len(ids) == len(set(ids))

    # Verify expected IDs
    assert "mcts:mcts_fast" in ids
    assert "mcts:mcts_deep" in ids
    assert "llm:llm_zero_shot" in ids
    assert "llm:llm_structured_choice_retry" in ids


# ---------------------------------------------------------------------------
# AgentRegistry tests
# ---------------------------------------------------------------------------


def test_registry_register_and_create():
    registry = AgentRegistry()
    registry.register(
        "random",
        lambda cfg: RandomAgent(name=cfg.name or "random", seed=cfg.params.get("seed")),
    )

    config = AgentConfig("random", name="testy", params={"seed": 123})
    agent = registry.create(config)

    assert isinstance(agent, RandomAgent)
    assert agent.name == "testy"
    assert agent.seed == 123


def test_registry_unknown_type():
    registry = AgentRegistry()
    with pytest.raises(KeyError, match="unknown_type"):
        registry.create(AgentConfig("unknown_type"))


def test_registry_is_registered():
    registry = AgentRegistry()
    registry.register("random", lambda cfg: RandomAgent())

    assert registry.is_registered("random")
    assert not registry.is_registered("mcts")


def test_registry_registered_types():
    registry = AgentRegistry()
    registry.register("random", lambda cfg: RandomAgent())
    registry.register("mcts", lambda cfg: RandomAgent())  # placeholder

    types = registry.registered_types()
    assert "random" in types
    assert "mcts" in types


def test_default_registry_has_random():
    registry = get_default_registry()
    assert registry.is_registered("random")

    config = AgentConfig("random", name="lucky", params={"seed": 42})
    agent = registry.create(config)

    assert isinstance(agent, RandomAgent)
    assert agent.name == "lucky"


# ---------------------------------------------------------------------------
# Agent interface conformance tests
# ---------------------------------------------------------------------------


def test_random_agent_implements_protocol():
    agent = RandomAgent(seed=42)
    assert isinstance(agent, Agent)  # Protocol check


def test_random_agent_select_action():
    agent = RandomAgent(seed=42)
    legal_actions = [0, 1, 2, 3, 4]
    action = agent.select_action(None, legal_actions)
    assert action in legal_actions


def test_random_agent_select_action_with_context():
    agent = RandomAgent(seed=42)
    legal_actions = [0, 1, 2]
    ctx = ActionContext(game_name="tic_tac_toe", player_id=0, turn_number=1)
    action = agent.select_action(None, legal_actions, context=ctx)
    assert action in legal_actions


def test_random_agent_select_action_empty_raises():
    agent = RandomAgent()
    with pytest.raises(ValueError, match="no legal actions"):
        agent.select_action(None, [])


def test_random_agent_choose_action_backward_compat():
    """Test that choose_action still works via default implementation."""
    agent = RandomAgent(seed=42)
    state = _FakeState([0, 1, 2, 3, 4])
    action = agent.choose_action(state)
    assert action in state.legal_actions()


def test_random_agent_reproducible():
    a = RandomAgent(seed=42)
    b = RandomAgent(seed=42)
    legal_actions = list(range(9))
    assert a.select_action(None, legal_actions) == b.select_action(None, legal_actions)


class _MinimalAgent(BaseAgent):
    """Minimal agent implementation for testing."""

    def select_action(self, state_view, legal_actions, context=None):
        return legal_actions[0]


def test_base_agent_subclass():
    agent = _MinimalAgent(name="first_pick")
    assert agent.name == "first_pick"

    legal_actions = [5, 3, 1]
    assert agent.select_action(None, legal_actions) == 5


def test_base_agent_choose_action_delegates():
    agent = _MinimalAgent(name="first_pick")
    state = _FakeState([10, 20, 30])
    # choose_action should delegate to select_action which returns first legal
    assert agent.choose_action(state) == 10


# ---------------------------------------------------------------------------
# Variant naming tests (acceptance criteria)
# ---------------------------------------------------------------------------


def test_variant_registry_ids():
    """Test that the registry can represent common variants."""
    variants = [
        ("mcts", "mcts_fast", {"num_simulations": 50}),
        ("mcts", "mcts_deep", {"num_simulations": 500}),
        ("llm", "llm_zero_shot", {"model": "gpt-4", "prompt_style": "zero_shot"}),
        ("llm", "llm_structured_choice_retry", {"model": "gpt-4", "retry": True}),
    ]

    configs = [AgentConfig(t, name=n, params=p) for t, n, p in variants]
    ids = [c.registry_id() for c in configs]

    # All unique
    assert len(ids) == len(set(ids))

    # Check expected format
    assert ids[0] == "mcts:mcts_fast"
    assert ids[1] == "mcts:mcts_deep"
    assert ids[2] == "llm:llm_zero_shot"
    assert ids[3] == "llm:llm_structured_choice_retry"


def test_stable_ids_for_same_config():
    """Downstream code can refer to competitors by stable IDs."""
    config1 = AgentConfig("mcts", name="standard", params={"num_simulations": 100})
    config2 = AgentConfig("mcts", name="standard", params={"num_simulations": 100})

    # Same config → same ID, stable over time
    assert config1.registry_id() == config2.registry_id()
    assert config1.registry_id() == "mcts:standard"


def test_config_hash_stability():
    """Param order should not affect the registry ID."""
    config1 = AgentConfig("mcts", params={"seed": 42, "num_simulations": 100})
    config2 = AgentConfig("mcts", params={"num_simulations": 100, "seed": 42})

    # Different param order → same ID (sorted keys in hash)
    assert config1.registry_id() == config2.registry_id()
