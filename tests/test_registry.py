"""
Tests for agents.registry — AgentConfig, AgentRegistry, and serialization.
"""

from __future__ import annotations

from typing import Any

import pytest

from agents.base import ActionContext, Agent, BaseAgent
from agents.random_agent import RandomAgent
from agents.registry import (
    AgentConfig,
    AgentRegistry,
    _serialize_value,
    _stable_json,
    get_default_registry,
)


class TestSerializeValue:
    """Tests for the _serialize_value helper."""

    def test_primitives(self) -> None:
        assert _serialize_value(None) is None
        assert _serialize_value(True) is True
        assert _serialize_value(False) is False
        assert _serialize_value(42) == 42
        assert _serialize_value(3.14) == 3.14
        assert _serialize_value("hello") == "hello"

    def test_list(self) -> None:
        assert _serialize_value([1, 2, 3]) == [1, 2, 3]
        assert _serialize_value((1, 2, 3)) == [1, 2, 3]  # tuple becomes list

    def test_nested(self) -> None:
        data = {"a": [1, 2], "b": {"c": 3}}
        result = _serialize_value(data)
        assert result == {"a": [1, 2], "b": {"c": 3}}

    def test_dict_sorted_keys(self) -> None:
        # Keys should be sorted
        data = {"z": 1, "a": 2, "m": 3}
        result = _serialize_value(data)
        assert list(result.keys()) == ["a", "m", "z"]

    def test_unknown_type_converts_to_string(self) -> None:
        class CustomType:
            def __str__(self) -> str:
                return "custom"

        assert _serialize_value(CustomType()) == "custom"


class TestStableJson:
    """Tests for the _stable_json helper."""

    def test_sorted_keys(self) -> None:
        result = _stable_json({"z": 1, "a": 2})
        assert result == '{"a":2,"z":1}'

    def test_no_spaces(self) -> None:
        result = _stable_json({"key": "value"})
        assert result == '{"key":"value"}'
        assert " " not in result

    def test_deterministic(self) -> None:
        data = {"b": 2, "a": 1, "c": [3, 2, 1]}
        assert _stable_json(data) == _stable_json(data)


class TestAgentConfig:
    """Tests for AgentConfig dataclass."""

    def test_basic_config(self) -> None:
        config = AgentConfig("random")
        assert config.agent_type == "random"
        assert config.name is None
        assert config.params == {}

    def test_config_with_params(self) -> None:
        config = AgentConfig("mcts", params={"num_simulations": 100, "seed": 42})
        assert config.agent_type == "mcts"
        assert config.params["num_simulations"] == 100
        assert config.params["seed"] == 42

    def test_config_frozen(self) -> None:
        config = AgentConfig("random", params={"seed": 42})
        with pytest.raises(AttributeError):
            config.agent_type = "mcts"  # type: ignore[misc]

    def test_registry_id_no_params(self) -> None:
        config = AgentConfig("random")
        assert config.registry_id() == "random"

    def test_registry_id_with_name(self) -> None:
        config = AgentConfig("mcts", name="fast", params={"num_simulations": 50})
        assert config.registry_id() == "mcts:fast"

    def test_registry_id_with_params_hash(self) -> None:
        config1 = AgentConfig("mcts", params={"num_simulations": 100})
        config2 = AgentConfig("mcts", params={"num_simulations": 100})
        config3 = AgentConfig("mcts", params={"num_simulations": 200})

        # Same params should produce same ID
        assert config1.registry_id() == config2.registry_id()
        # Different params should produce different IDs
        assert config1.registry_id() != config3.registry_id()
        # IDs should start with agent_type
        assert config1.registry_id().startswith("mcts:")

    def test_different_configs_different_ids(self) -> None:
        """Acceptance criterion: two agent instances with different configs
        produce different registry identifiers."""
        configs = [
            AgentConfig("mcts", name="fast", params={"num_simulations": 50}),
            AgentConfig("mcts", name="deep", params={"num_simulations": 1000}),
            AgentConfig("llm", name="zero_shot", params={"model": "gpt-4"}),
            AgentConfig(
                "llm",
                name="structured_choice_retry",
                params={"model": "gpt-4", "retry_on_invalid": True},
            ),
        ]
        ids = [c.registry_id() for c in configs]
        # All IDs should be unique
        assert len(ids) == len(set(ids))
        # Expected IDs
        assert "mcts:fast" in ids
        assert "mcts:deep" in ids
        assert "llm:zero_shot" in ids
        assert "llm:structured_choice_retry" in ids

    def test_display_name_with_name(self) -> None:
        config = AgentConfig("mcts", name="my_agent")
        assert config.display_name() == "my_agent"

    def test_display_name_no_params(self) -> None:
        config = AgentConfig("random")
        assert config.display_name() == "random"

    def test_display_name_with_params(self) -> None:
        config = AgentConfig("mcts", params={"num_simulations": 100, "seed": 42})
        # Should include params in sorted order
        display = config.display_name()
        assert "mcts" in display
        assert "num_simulations=100" in display
        assert "seed=42" in display

    def test_to_dict(self) -> None:
        config = AgentConfig("mcts", name="fast", params={"num_simulations": 100})
        d = config.to_dict()
        assert d == {
            "agent_type": "mcts",
            "name": "fast",
            "params": {"num_simulations": 100},
        }

    def test_from_dict(self) -> None:
        d = {"agent_type": "random", "name": "lucky", "params": {"seed": 42}}
        config = AgentConfig.from_dict(d)
        assert config.agent_type == "random"
        assert config.name == "lucky"
        assert config.params == {"seed": 42}

    def test_roundtrip(self) -> None:
        original = AgentConfig("llm", name="gpt4", params={"temperature": 0.7})
        restored = AgentConfig.from_dict(original.to_dict())
        assert restored.agent_type == original.agent_type
        assert restored.name == original.name
        assert restored.params == original.params


class TestAgentRegistry:
    """Tests for AgentRegistry."""

    def test_register_and_create(self) -> None:
        registry = AgentRegistry()
        registry.register(
            "random",
            lambda cfg: RandomAgent(
                name=cfg.name or "random", seed=cfg.params.get("seed")
            ),
        )
        config = AgentConfig("random", name="lucky", params={"seed": 42})
        agent = registry.create(config)
        assert isinstance(agent, RandomAgent)
        assert agent.name == "lucky"
        assert agent.seed == 42

    def test_create_unknown_type(self) -> None:
        registry = AgentRegistry()
        with pytest.raises(KeyError, match="unknown_agent"):
            registry.create(AgentConfig("unknown_agent"))

    def test_is_registered(self) -> None:
        registry = AgentRegistry()
        assert not registry.is_registered("random")
        registry.register("random", lambda cfg: RandomAgent())
        assert registry.is_registered("random")

    def test_registered_types(self) -> None:
        registry = AgentRegistry()
        registry.register("random", lambda cfg: RandomAgent())
        registry.register("mcts", lambda cfg: RandomAgent())  # placeholder
        types = registry.registered_types()
        assert set(types) == {"random", "mcts"}


class TestDefaultRegistry:
    """Tests for the global default registry."""

    def test_has_random_agent(self) -> None:
        registry = get_default_registry()
        assert registry.is_registered("random")

    def test_create_random_agent(self) -> None:
        registry = get_default_registry()
        config = AgentConfig("random", params={"seed": 123})
        agent = registry.create(config)
        assert isinstance(agent, RandomAgent)

    def test_singleton(self) -> None:
        registry1 = get_default_registry()
        registry2 = get_default_registry()
        assert registry1 is registry2


class TestAgentProtocol:
    """Tests for Agent protocol conformance."""

    def test_random_agent_satisfies_protocol(self) -> None:
        agent: Agent = RandomAgent(seed=42)
        assert hasattr(agent, "name")
        assert hasattr(agent, "select_action")
        assert callable(agent.select_action)

    def test_protocol_runtime_checkable(self) -> None:
        agent = RandomAgent(seed=42)
        assert isinstance(agent, Agent)

    def test_select_action_signature(self) -> None:
        agent = RandomAgent(seed=42)
        legal_actions = [0, 1, 2]
        context = ActionContext(
            game_name="test_game",
            player_id=0,
            turn_number=1,
            history=[],
        )
        action = agent.select_action(None, legal_actions, context)
        assert action in legal_actions


class TestActionContext:
    """Tests for ActionContext dataclass."""

    def test_defaults(self) -> None:
        ctx = ActionContext()
        assert ctx.game_name == ""
        assert ctx.player_id == 0
        assert ctx.turn_number == 0
        assert ctx.history == []
        assert ctx.extra == {}

    def test_all_fields(self) -> None:
        ctx = ActionContext(
            game_name="tic_tac_toe",
            player_id=1,
            turn_number=5,
            history=[0, 4, 1],
            extra={"difficulty": "hard"},
        )
        assert ctx.game_name == "tic_tac_toe"
        assert ctx.player_id == 1
        assert ctx.turn_number == 5
        assert ctx.history == [0, 4, 1]
        assert ctx.extra == {"difficulty": "hard"}

    def test_frozen(self) -> None:
        ctx = ActionContext()
        with pytest.raises(AttributeError):
            ctx.game_name = "new_game"  # type: ignore[misc]


class MockAgent(BaseAgent):
    """Minimal agent implementation for testing."""

    def __init__(self, name: str = "mock", action: int = 0) -> None:
        super().__init__(name)
        self._action = action

    def select_action(
        self,
        state_view: Any,
        legal_actions: list[int],
        context: ActionContext | None = None,
    ) -> int:
        # Return the first legal action that matches our preferred action,
        # or the first legal action if not found
        if self._action in legal_actions:
            return self._action
        return legal_actions[0]


class TestBaseAgent:
    """Tests for BaseAgent abstract class."""

    def test_name_attribute(self) -> None:
        agent = MockAgent(name="test_agent")
        assert agent.name == "test_agent"

    def test_repr(self) -> None:
        agent = MockAgent(name="test_agent")
        assert "MockAgent" in repr(agent)
        assert "test_agent" in repr(agent)

    def test_choose_action_legacy(self) -> None:
        """Test the legacy choose_action interface."""

        class FakeState:
            def legal_actions(self) -> list[int]:
                return [1, 2, 3]

        agent = MockAgent(action=2)
        action = agent.choose_action(FakeState())
        assert action == 2


class TestRegistryVariants:
    """Acceptance tests for registry representing agent variants.

    This tests that the registry can represent variants like:
    - mcts_fast
    - mcts_deep
    - llm_zero_shot
    - llm_structured_choice_retry
    """

    def test_mcts_variants(self) -> None:
        """Test MCTS variants with different simulation counts."""
        registry = AgentRegistry()
        registry.register("mcts", lambda cfg: MockAgent(name=cfg.display_name()))

        fast_config = AgentConfig("mcts", name="fast", params={"num_simulations": 50})
        deep_config = AgentConfig(
            "mcts", name="deep", params={"num_simulations": 1000}
        )

        registry.create(fast_config)
        registry.create(deep_config)

        assert fast_config.registry_id() == "mcts:fast"
        assert deep_config.registry_id() == "mcts:deep"
        assert fast_config.registry_id() != deep_config.registry_id()

    def test_llm_variants(self) -> None:
        """Test LLM variants with different prompting strategies."""
        registry = AgentRegistry()
        registry.register("llm", lambda cfg: MockAgent(name=cfg.display_name()))

        zero_shot = AgentConfig("llm", name="zero_shot", params={"model": "gpt-4"})
        structured_retry = AgentConfig(
            "llm",
            name="structured_choice_retry",
            params={"model": "gpt-4", "retry_on_invalid": True},
        )

        assert zero_shot.registry_id() == "llm:zero_shot"
        assert structured_retry.registry_id() == "llm:structured_choice_retry"
        assert zero_shot.registry_id() != structured_retry.registry_id()

    def test_unnamed_variants_get_unique_hashes(self) -> None:
        """Test that unnamed configs with different params get unique IDs."""
        config1 = AgentConfig("mcts", params={"num_simulations": 100})
        config2 = AgentConfig("mcts", params={"num_simulations": 200})
        config3 = AgentConfig("mcts", params={"num_simulations": 100, "seed": 42})

        ids = [config1.registry_id(), config2.registry_id(), config3.registry_id()]
        # All should be unique
        assert len(ids) == len(set(ids))
        # All should start with mcts:
        for id_ in ids:
            assert id_.startswith("mcts:")

    def test_stable_ids_across_sessions(self) -> None:
        """Test that IDs are stable (deterministic hash)."""
        config = AgentConfig("mcts", params={"num_simulations": 100, "seed": 42})
        id1 = config.registry_id()
        id2 = config.registry_id()
        assert id1 == id2
