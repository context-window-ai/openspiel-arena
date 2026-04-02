"""
Agent registry — manages agent configurations and creates stable identifiers.

The registry provides:
- Stable identifiers for agents based on their type and configuration
- Config serialization for reproducibility
- Factory pattern for creating agents from configurations

This enables downstream rating systems to treat different agent variants
(e.g., ``mcts_fast`` vs ``mcts_deep``) as distinct competitors.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from agents.base import Agent


def _serialize_value(value: Any) -> Any:
    """Serialize a value to a JSON-compatible representation."""
    if value is None or isinstance(value, (bool, int, str, float)):
        return value
    if isinstance(value, (list, tuple)):
        return [_serialize_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _serialize_value(v) for k, v in sorted(value.items())}
    # For other types, use string representation
    return str(value)


def _stable_json(obj: dict[str, Any]) -> str:
    """Create a stable JSON string from a dict (sorted keys, no spaces)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class AgentConfig:
    """Immutable configuration for an agent.

    This is the primary way to specify an agent for the registry.
    Configs are hashable and comparable, making them suitable as dict keys.

    Attributes
    ----------
    agent_type:
        The type identifier (e.g., "random", "mcts", "llm_openai", "llm_anthropic").
    name:
        Optional human-readable name. If not provided, one will be generated
        from the type and params.
    params:
        Agent-specific parameters (e.g., ``{"seed": 42, "num_simulations": 100}``).
    """

    agent_type: str
    name: str | None = None
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Ensure params is frozen by creating a frozen copy
        object.__setattr__(self, "params", dict(self.params))

    def _params_digest(self) -> str:
        """Return a short hash of the params for unique identification."""
        if not self.params:
            return ""
        serialized = _stable_json(_serialize_value(self.params))
        return hashlib.sha256(serialized.encode()).hexdigest()[:8]

    def registry_id(self) -> str:
        """Return the stable registry identifier for this config.

        The ID format is:
        - If name is provided: ``{agent_type}:{name}``
        - Otherwise: ``{agent_type}:{params_hash}``

        Examples
        --------
        >>> config = AgentConfig("random", params={"seed": 42})
        >>> config.registry_id()
        'random:a1b2c3d4'  # hash varies

        >>> config = AgentConfig("mcts", name="fast", params={"num_simulations": 50})
        >>> config.registry_id()
        'mcts:fast'
        """
        if self.name:
            return f"{self.agent_type}:{self.name}"
        digest = self._params_digest()
        if digest:
            return f"{self.agent_type}:{digest}"
        return self.agent_type

    def display_name(self) -> str:
        """Return a human-readable name for this config.

        Falls back to the registry ID if no name is set.
        """
        if self.name:
            return self.name
        # Generate a readable name from params
        if not self.params:
            return self.agent_type
        parts = [self.agent_type]
        for key, value in sorted(self.params.items()):
            parts.append(f"{key}={value}")
        return "_".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict."""
        return {
            "agent_type": self.agent_type,
            "name": self.name,
            "params": dict(self.params),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentConfig:
        """Deserialize from a plain dict."""
        return cls(
            agent_type=data["agent_type"],
            name=data.get("name"),
            params=data.get("params", {}),
        )


# Type alias for agent factory functions
AgentFactory = Callable[[AgentConfig], Agent]


class AgentRegistry:
    """Registry for agent types and their factories.

    The registry maps agent type strings to factory functions that can
    create agent instances from configurations. It also provides utilities
    for creating agents and managing their stable identifiers.

    Example
    -------
    >>> registry = AgentRegistry()
    >>> registry.register("random", lambda cfg: RandomAgent(
    ...     name=cfg.display_name(), **cfg.params
    ... ))
    >>> config = AgentConfig("random", params={"seed": 42})
    >>> agent = registry.create(config)
    """

    def __init__(self) -> None:
        self._factories: dict[str, AgentFactory] = {}

    def register(self, agent_type: str, factory: AgentFactory) -> None:
        """Register a factory function for an agent type.

        Parameters
        ----------
        agent_type:
            The type identifier (e.g., "random", "mcts").
        factory:
            A callable that takes an :class:`AgentConfig` and returns
            an :class:`Agent` instance.
        """
        self._factories[agent_type] = factory

    def create(self, config: AgentConfig) -> Agent:
        """Create an agent instance from a configuration.

        Parameters
        ----------
        config:
            The agent configuration.

        Returns
        -------
        Agent
            A new agent instance.

        Raises
        ------
        KeyError
            If the agent type is not registered.
        """
        if config.agent_type not in self._factories:
            raise KeyError(f"Unknown agent type: {config.agent_type!r}")
        factory = self._factories[config.agent_type]
        return factory(config)

    def is_registered(self, agent_type: str) -> bool:
        """Check if an agent type is registered."""
        return agent_type in self._factories

    def registered_types(self) -> list[str]:
        """Return a list of all registered agent types."""
        return list(self._factories.keys())


# Global default registry
_default_registry: AgentRegistry | None = None


def get_default_registry() -> AgentRegistry:
    """Get the global default agent registry.

    The default registry is lazily initialized with built-in agent types.
    """
    global _default_registry
    if _default_registry is None:
        _default_registry = AgentRegistry()
        _register_builtin_agents(_default_registry)
    return _default_registry


def _register_builtin_agents(registry: AgentRegistry) -> None:
    """Register built-in agent types with the registry."""
    # Import here to avoid circular imports
    from agents.random_agent import RandomAgent
    from agents.llm_agent import LLMAgent, LLMAgentConfig, FallbackStrategy
    from agents.prompts import PromptStyle

    registry.register(
        "random",
        lambda cfg: RandomAgent(
            name=cfg.name or cfg.display_name(),
            seed=cfg.params.get("seed"),
        ),
    )

    def _create_llm_agent(cfg: AgentConfig) -> "LLMAgent":
        """Factory for LLM agents."""
        params = cfg.params
        prompt_style = PromptStyle(params.get("prompt_style", "legal_moves_only"))
        fallback = FallbackStrategy(params.get("fallback_strategy", "random"))

        llm_config = LLMAgentConfig(
            model=params.get("model", "minimax/minimax-m2.5:free"),
            prompt_style=prompt_style,
            memory_turns=params.get("memory_turns", 0),
            temperature=params.get("temperature", 0.7),
            max_tokens=params.get("max_tokens", 1024),
            fallback_strategy=fallback,
            log_responses=params.get("log_responses", False),
            response_log_dir=params.get("response_log_dir", "logs/llm_responses"),
        )

        return LLMAgent(
            name=cfg.name or cfg.display_name(),
            config=llm_config,
        )

    registry.register("llm", _create_llm_agent)
