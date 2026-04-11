"""
Base agent interface.

All concrete agents must satisfy this interface so the arena can treat
every agent uniformly, regardless of whether it is a classic search algorithm
or an LLM.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class ActionContext:
    """Additional context passed to agents when selecting an action.

    This allows agents to make decisions based on game-level information
    without depending on the full OpenSpiel state object.

    Attributes
    ----------
    game_name:
        Name of the game being played (e.g., "tic_tac_toe").
    player_id:
        The player index (0 or 1 in two-player games).
    turn_number:
        Current turn/move number in the game.
    history:
        List of actions taken so far in the game.
    extra:
        Optional game-specific metadata.
    """

    game_name: str = ""
    player_id: int = 0
    turn_number: int = 0
    history: list[int] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Agent(Protocol):
    """Structural (duck-typing) protocol for all arena agents.

    Use this when you want to accept *any* agent without requiring
    inheritance from ``BaseAgent``.
    """

    name: str

    def select_action(
        self,
        state_view: Any,
        legal_actions: list[int],
        context: ActionContext | None = None,
    ) -> int:
        """Return the chosen action from ``legal_actions``.

        Parameters
        ----------
        state_view:
            A game-specific view of the current state. This may be the raw
            OpenSpiel state or a simplified representation (e.g., a string
            or dict) suitable for LLM consumption.
        legal_actions:
            List of legal action indices the agent may choose from.
        context:
            Optional additional context about the game state.

        Returns
        -------
        int
            The chosen action index (must be in ``legal_actions``).
        """
        ...  # pragma: no cover


class BaseAgent(ABC):
    """Abstract base class for all arena agents.

    Subclasses must implement :meth:`select_action`. A default
    :meth:`choose_action` implementation is provided for backward
    compatibility with existing code.

    Parameters
    ----------
    name:
        Human-readable identifier used in results and rating tables.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        # Optional: most recent LLM prompt/response (set after each select_action call)
        self.last_prompt: str | None = None
        self.last_response: str | None = None

    @abstractmethod
    def select_action(
        self,
        state_view: Any,
        legal_actions: list[int],
        context: ActionContext | None = None,
    ) -> int:
        """Return the chosen action from ``legal_actions``.

        Parameters
        ----------
        state_view:
            A game-specific view of the current state.
        legal_actions:
            List of legal action indices the agent may choose from.
        context:
            Optional additional context about the game state.

        Returns
        -------
        int
            The chosen action index (must be in ``legal_actions``).
        """

    def choose_action(self, state: Any) -> int:
        """Legacy interface for backward compatibility.

        Extracts legal actions from the state and delegates to
        :meth:`select_action`. Subclasses generally should not override
        this method.

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
        legal_actions = list(state.legal_actions())
        return self.select_action(state, legal_actions, context=None)

    def reset(self) -> None:
        """Reset agent state for a new game.

        Subclasses may override to clear per-game caches (e.g. turn history).
        """
        self.last_prompt = None
        self.last_response = None

    def __repr__(self) -> str:  # pragma: no cover
        return f"{self.__class__.__name__}(name={self.name!r})"
