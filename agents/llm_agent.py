"""
agents.llm_agent — LLM-backed agent using OpenRouter API
=========================================================
Provides an LLM agent that chooses from legal moves only, supports multiple
prompt styles, and can include the last n turns of game history.

Configuration
-------------
- model: The model to use (default: "minimax/minimax-m2.5:free")
- prompt_style: One of the PromptStyle enum values
- memory_turns: Number of recent turns to include in context (0 = stateless)
- fallback_mode: What to do after retry failure ("random" or "first")
- temperature: Sampling temperature for the LLM
- max_tokens: Maximum tokens in response

The agent uses OpenRouter API (OpenAI-compatible):
- Base URL: https://openrouter.ai/api/v1
- API key from OPENROUTER_API_KEY environment variable
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from openai import OpenAI

from agents.base import ActionContext, BaseAgent
from agents.prompts import (
    PromptContext,
    PromptStyle,
    get_template,
    parse_action_from_response,
)

logger = logging.getLogger(__name__)


class FallbackMode(str, Enum):
    """What to do when the LLM fails to return a valid action."""

    RANDOM = "random"  # Choose randomly from legal actions
    FIRST = "first"  # Choose the first legal action


@dataclass
class LLMAgentConfig:
    """Configuration for the LLM agent.

    Attributes
    ----------
    model:
        The model identifier for OpenRouter (e.g., "minimax/minimax-m2.5:free").
    prompt_style:
        The prompt template style to use.
    memory_turns:
        Number of recent turns to include in context. 0 = stateless.
    fallback_mode:
        What to do after retry failure.
    temperature:
        Sampling temperature (0.0 to 2.0).
    max_tokens:
        Maximum tokens in the response.
    debug_dir:
        Optional directory to save debug logs (prompts and responses).
    """

    model: str = "minimax/minimax-m2.5:free"
    prompt_style: PromptStyle | str = PromptStyle.LEGAL_MOVES_ONLY
    memory_turns: int = 0
    fallback_mode: FallbackMode | str = FallbackMode.RANDOM
    temperature: float = 0.7
    max_tokens: int = 500
    reasoning_effort: str | None = None  # 'low', 'medium', 'high' (o-series / gpt-5.x)
    debug_dir: str | Path | None = None

    def __post_init__(self) -> None:
        # Convert string to enum if needed
        if isinstance(self.prompt_style, str):
            self.prompt_style = PromptStyle(self.prompt_style)
        if isinstance(self.fallback_mode, str):
            self.fallback_mode = FallbackMode(self.fallback_mode)
        if isinstance(self.debug_dir, str):
            self.debug_dir = Path(self.debug_dir)

        # Validate
        if self.memory_turns < 0:
            raise ValueError(f"memory_turns must be >= 0, got {self.memory_turns}")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(
                f"temperature must be in [0.0, 2.0], got {self.temperature}"
            )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LLMAgentConfig:
        """Create config from a dictionary."""
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        """Convert config to a dictionary."""
        return {
            "model": self.model,
            "prompt_style": self.prompt_style.value
            if isinstance(self.prompt_style, PromptStyle)
            else self.prompt_style,
            "memory_turns": self.memory_turns,
            "fallback_mode": self.fallback_mode.value
            if isinstance(self.fallback_mode, FallbackMode)
            else self.fallback_mode,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "reasoning_effort": self.reasoning_effort,
            "debug_dir": str(self.debug_dir) if self.debug_dir else None,
        }


@dataclass
class TurnRecord:
    """Record of a single turn for memory and debugging.

    Attributes
    ----------
    turn_number:
        The turn number in the game.
    player_id:
        The player who moved.
    state_description:
        Description of the state before the move.
    legal_actions:
        List of legal action descriptions.
    legal_action_ids:
        List of legal action IDs.
    prompt:
        The full prompt sent to the LLM.
    raw_response:
        The raw response from the LLM.
    parsed_action:
        The action that was parsed from the response.
    chosen_action:
        The final action taken (may differ if fallback used).
    was_retry:
        Whether a retry was needed.
    was_fallback:
        Whether fallback was used.
    """

    turn_number: int
    player_id: int
    state_description: str
    legal_actions: list[str]
    legal_action_ids: list[int]
    prompt: str
    raw_response: str
    parsed_action: int | None
    chosen_action: int
    was_retry: bool = False
    was_fallback: bool = False


class LLMAgent(BaseAgent):
    """LLM-backed agent that uses OpenRouter API.

    This agent prompts an LLM to choose from legal moves, with configurable
    prompt styles and optional turn memory.

    Parameters
    ----------
    name:
        Human-readable identifier for the agent.
    config:
        Configuration for the agent. If not provided, uses defaults.

    Example
    -------
    >>> config = LLMAgentConfig(
    ...     model="minimax/minimax-m2.5:free",
    ...     prompt_style="legal_moves_only",
    ...     memory_turns=3,
    ... )
    >>> agent = LLMAgent("llm-agent", config)
    >>> action = agent.select_action(state, legal_actions, context)
    """

    def __init__(
        self,
        name: str = "llm",
        config: LLMAgentConfig | None = None,
    ) -> None:
        super().__init__(name)
        self._config = config or LLMAgentConfig()
        self._template = get_template(self._config.prompt_style)
        self._turn_history: list[TurnRecord] = []
        self._game_history: list[str] = []  # For memory_turns feature

        # Initialize OpenAI client for OpenRouter
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY environment variable is required. "
                "Set it in your .env file or export it."
            )

        self._client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            timeout=60.0,
        )

        # Create debug directory if specified
        if self._config.debug_dir:
            Path(self._config.debug_dir).mkdir(parents=True, exist_ok=True)

    @property
    def config(self) -> LLMAgentConfig:
        """Return the agent configuration."""
        return self._config

    @property
    def turn_history(self) -> list[TurnRecord]:
        """Return the history of turns for this game."""
        return list(self._turn_history)

    def reset(self) -> None:
        """Reset the agent state for a new game."""
        super().reset()
        self._turn_history = []
        self._game_history = []

    def select_action(
        self,
        state_view: Any,
        legal_actions: list[int],
        context: ActionContext | None = None,
    ) -> int:
        """Select an action using the LLM.

        Parameters
        ----------
        state_view:
            A game-specific view of the current state.
        legal_actions:
            List of legal action indices.
        context:
            Optional context about the game state.

        Returns
        -------
        int
            The chosen action index from legal_actions.
        """
        if not legal_actions:
            raise ValueError("select_action called with no legal actions")

        # Build context
        context = context or ActionContext()
        state_description = self._get_state_description(state_view)
        legal_action_descriptions = self._get_action_descriptions(
            state_view, legal_actions
        )

        prompt_context = PromptContext(
            game_name=context.game_name or "unknown_game",
            state_description=state_description,
            legal_actions=legal_action_descriptions,
            legal_action_ids=legal_actions,
            player_id=context.player_id,
            turn_number=context.turn_number,
            history=self._get_memory_history(),
            extra=context.extra,
        )

        # Build prompt
        prompt = self._template.build_prompt(prompt_context)

        # Call LLM
        raw_response, was_retry = self._call_llm_with_retry(
            prompt, prompt_context, legal_actions
        )

        # Parse response
        parsed_action = parse_action_from_response(raw_response, legal_actions)

        # Determine final action
        if parsed_action is not None:
            chosen_action = parsed_action
            was_fallback = False
        else:
            # Use fallback
            chosen_action = self._fallback_action(legal_actions)
            was_fallback = True
            logger.warning(
                f"LLM returned invalid action. Raw response: {raw_response[:200]!r}. "
                f"Using fallback: {chosen_action}"
            )

        # Expose prompt/response for transcript capture
        self.last_prompt = prompt
        self.last_response = raw_response

        # Record turn
        record = TurnRecord(
            turn_number=context.turn_number,
            player_id=context.player_id,
            state_description=state_description,
            legal_actions=legal_action_descriptions,
            legal_action_ids=legal_actions,
            prompt=prompt,
            raw_response=raw_response,
            parsed_action=parsed_action,
            chosen_action=chosen_action,
            was_retry=was_retry,
            was_fallback=was_fallback,
        )
        self._turn_history.append(record)

        # Update game history for memory
        turn_summary = self._make_turn_summary(record)
        self._game_history.append(turn_summary)

        # Debug logging
        self._log_debug_info(record)

        return chosen_action

    def _call_llm_with_retry(
        self,
        prompt: str,
        context: PromptContext,
        legal_actions: list[int],
    ) -> tuple[str, bool]:
        """Call the LLM with one retry on invalid response.

        Returns
        -------
        tuple[str, bool]
            (raw_response, was_retry)
        """
        # First attempt
        raw_response = self._call_llm(prompt)
        parsed = parse_action_from_response(raw_response, legal_actions)

        if parsed is not None:
            return raw_response, False

        # Log the invalid response
        logger.warning(
            f"Invalid action parsed from LLM response. "
            f"Response: {raw_response[:200]!r}"
        )

        # Retry with correction prompt
        correction_prompt = self._template.build_correction_prompt(
            context, raw_response, "Could not parse a valid action ID"
        )
        raw_response = self._call_llm(correction_prompt)

        return raw_response, True

    def _call_llm(self, prompt: str) -> str:
        """Make a single call to the LLM.

        Parameters
        ----------
        prompt:
            The prompt to send.

        Returns
        -------
        str
            The raw response text.
        """
        try:
            call_kwargs: dict = dict(
                model=self._config.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a game-playing AI. Follow instructions precisely.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=self._config.max_tokens,
            )
            if self._config.reasoning_effort:
                call_kwargs["reasoning_effort"] = self._config.reasoning_effort
                call_kwargs.pop("max_tokens", None)  # no cap: let reasoning finish
            else:
                call_kwargs["temperature"] = self._config.temperature
            response = self._client.chat.completions.create(**call_kwargs)

            content = response.choices[0].message.content
            if content is None:
                return ""
            return content.strip()

        except Exception as e:
            logger.error(f"LLM API call failed: {e}")
            raise

    def _fallback_action(self, legal_actions: list[int]) -> int:
        """Choose a fallback action when LLM fails.

        Parameters
        ----------
        legal_actions:
            List of legal actions.

        Returns
        -------
        int
            A legal action.
        """
        if self._config.fallback_mode == FallbackMode.FIRST:
            return legal_actions[0]
        else:  # RANDOM
            import random

            return random.choice(legal_actions)

    def _get_state_description(self, state_view: Any) -> str:
        """Get a human-readable description of the state.

        Parameters
        ----------
        state_view:
            The state view (may be an OpenSpiel state or string).

        Returns
        -------
        str
            Human-readable state description.
        """
        # If it has a state_string method, use it
        if hasattr(state_view, "state_string"):
            return str(state_view.state_string())
        # If it's already a string, use it
        if isinstance(state_view, str):
            return state_view
        # Otherwise, use string representation
        return str(state_view)

    def _get_action_descriptions(
        self, state_view: Any, legal_actions: list[int]
    ) -> list[str]:
        """Get human-readable descriptions for actions.

        Parameters
        ----------
        state_view:
            The state view (may have action_to_string method).
        legal_actions:
            List of legal action IDs.

        Returns
        -------
        list[str]
            Human-readable action descriptions.
        """
        descriptions = []
        for action in legal_actions:
            # Try to get action string from state
            if hasattr(state_view, "action_to_string"):
                try:
                    player = (
                        state_view.current_player()
                        if hasattr(state_view, "current_player")
                        else 0
                    )
                    desc = state_view.action_to_string(player, action)
                    descriptions.append(desc)
                    continue
                except Exception:
                    pass

            # Fallback to just the ID
            descriptions.append(f"Action {action}")

        return descriptions

    def _get_memory_history(self) -> list[str]:
        """Get the history to include in the prompt based on memory_turns.

        Returns
        -------
        list[str]
            List of turn descriptions for context.
        """
        if self._config.memory_turns == 0:
            return []

        # Get last n turns
        n = self._config.memory_turns
        return self._game_history[-n:] if n > 0 else []

    def _make_turn_summary(self, record: TurnRecord) -> str:
        """Create a summary string for a turn to include in history.

        Parameters
        ----------
        record:
            The turn record.

        Returns
        -------
        str
            A brief summary of the turn.
        """
        # Find the action description
        action_desc = "unknown"
        if record.chosen_action in record.legal_action_ids:
            idx = record.legal_action_ids.index(record.chosen_action)
            action_desc = record.legal_actions[idx]
        else:
            action_desc = f"Action {record.chosen_action}"

        return f"Player {record.player_id} played {action_desc}"

    def _log_debug_info(self, record: TurnRecord) -> None:
        """Log debug information to file if debug_dir is set."""
        if not self._config.debug_dir:
            return

        debug_dir = Path(self._config.debug_dir)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"turn_{record.turn_number:03d}_{timestamp}.json"
        filepath = debug_dir / filename

        debug_data = {
            "turn_number": record.turn_number,
            "player_id": record.player_id,
            "state_description": record.state_description,
            "legal_actions": record.legal_actions,
            "legal_action_ids": record.legal_action_ids,
            "prompt": record.prompt,
            "raw_response": record.raw_response,
            "parsed_action": record.parsed_action,
            "chosen_action": record.chosen_action,
            "was_retry": record.was_retry,
            "was_fallback": record.was_fallback,
            "config": self._config.to_dict(),
        }

        try:
            with open(filepath, "w") as f:
                json.dump(debug_data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to write debug file: {e}")


def create_llm_agent(
    name: str = "llm",
    model: str = "minimax/minimax-m2.5:free",
    prompt_style: PromptStyle | str = "legal_moves_only",
    memory_turns: int = 0,
    fallback_mode: FallbackMode | str = "random",
    temperature: float = 0.7,
    **kwargs: Any,
) -> LLMAgent:
    """Factory function to create an LLM agent with common options.

    Parameters
    ----------
    name:
        Agent name.
    model:
        Model identifier for OpenRouter.
    prompt_style:
        Prompt template style.
    memory_turns:
        Number of turns to remember (0 = stateless).
    fallback_mode:
        Fallback behavior on invalid response.
    temperature:
        Sampling temperature.
    **kwargs:
        Additional config options.

    Returns
    -------
    LLMAgent
        Configured LLM agent instance.
    """
    config = LLMAgentConfig(
        model=model,
        prompt_style=prompt_style,
        memory_turns=memory_turns,
        fallback_mode=fallback_mode,
        temperature=temperature,
        **kwargs,
    )
    return LLMAgent(name, config)
