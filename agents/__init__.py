"""
agents — Agent implementations that share a common interface.

Every agent must implement the ``Agent`` protocol (defined in
``agents.base``)::

    class Agent(Protocol):
        name: str
        def select_action(self, state_view, legal_actions, context) -> int: ...

Modules
-------
- base.py          : Agent protocol, BaseAgent abstract class, ActionContext
- registry.py      : AgentConfig, AgentRegistry for stable identifiers
- random_agent.py  : Uniformly random legal-move agent (sanity baseline)
- mcts_agent.py    : Wraps OpenSpiel's built-in MCTS solver
- openai_agent.py  : LLM agent backed by the OpenAI API
- anthropic_agent.py : LLM agent backed by the Anthropic API

Usage example::

    from agents import RandomAgent, AgentConfig, AgentRegistry

    # Direct instantiation
    agent = RandomAgent(seed=42)

    # Via registry (recommended for tournaments)
    registry = AgentRegistry()
    registry.register("random", lambda cfg: RandomAgent(
        name=cfg.display_name(), **cfg.params
    ))
    config = AgentConfig("random", name="lucky", params={"seed": 42})
    agent = registry.create(config)
    print(config.registry_id())  # "random:lucky"
"""

from agents.base import ActionContext, Agent, BaseAgent
from agents.registry import AgentConfig, AgentRegistry, get_default_registry

__all__ = [
    "ActionContext",
    "Agent",
    "BaseAgent",
    "AgentConfig",
    "AgentRegistry",
    "get_default_registry",
]
