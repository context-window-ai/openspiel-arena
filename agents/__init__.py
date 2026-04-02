"""
agents — Agent implementations that share a common interface.

Every agent must implement the ``BaseAgent`` protocol (defined in
``agents.base``):

    class BaseAgent(Protocol):
        name: str
        def choose_action(self, state) -> int: ...

Planned modules
---------------
- base.py          : BaseAgent protocol / abstract class
- random_agent.py  : Uniformly random legal-move agent (sanity baseline)
- mcts_agent.py    : Wraps OpenSpiel's built-in MCTS solver
- openai_agent.py  : LLM agent backed by the OpenAI API
- anthropic_agent.py : LLM agent backed by the Anthropic API

Usage example (once implemented)::

    from agents.random_agent import RandomAgent
    from agents.mcts_agent import MCTSAgent

    agents = [RandomAgent(seed=42), MCTSAgent(num_simulations=100)]
"""
