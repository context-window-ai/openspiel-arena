"""
Base agent interface.

All concrete agents must satisfy this interface so the arena can treat
every agent uniformly, regardless of whether it is a classic search algorithm
or an LLM.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Protocol, runtime_checkable


class BaseAgent(ABC):
    """Abstract base class for all arena agents.

    Parameters
    ----------
    name:
        Human-readable identifier used in results and rating tables.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def choose_action(self, state: Any) -> int:
        """Return the index of the chosen legal action for *state*.

        Parameters
        ----------
        state:
            An OpenSpiel ``State`` object (or compatible wrapper) at a
            decision node for this agent.

        Returns
        -------
        int
            Index into ``state.legal_actions()``.
        """

    def __repr__(self) -> str:  # pragma: no cover
        return f"{self.__class__.__name__}(name={self.name!r})"


@runtime_checkable
class Agent(Protocol):
    """Structural (duck-typing) protocol for all arena agents.

    Use this when you want to accept *any* agent without requiring
    inheritance from ``BaseAgent``.
    """

    name: str

    def choose_action(self, state: Any) -> int:  # pragma: no cover
        """Return the index of the chosen legal action for *state*."""
        ...
