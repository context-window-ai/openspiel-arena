"""
agents.prompts — Prompt templates for LLM agents
=================================================
Provides configurable prompt templates for game-playing LLM agents.

Supported prompt styles:
- zero_shot: Basic prompt with no guidance
- legal_moves_only: Lists legal moves and asks agent to pick one
- board_summary_then_choice: Provides board analysis then asks for choice
- reason_then_choice: Asks agent to reason before selecting
- critic_then_choice: Agent proposes, critiques, then finalizes choice

Each template receives:
- game_name: Name of the game being played
- state_description: Human-readable board/state representation
- legal_actions: List of legal action descriptions
- legal_action_ids: List of legal action IDs (parallel to legal_actions)
- player_id: Current player's ID
- turn_number: Current turn number
- history: Optional list of previous turn descriptions (for memory)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class PromptStyle(str, Enum):
    """Available prompt styles for LLM agents."""

    ZERO_SHOT = "zero_shot"
    LEGAL_MOVES_ONLY = "legal_moves_only"
    BOARD_SUMMARY_THEN_CHOICE = "board_summary_then_choice"
    REASON_THEN_CHOICE = "reason_then_choice"
    CRITIC_THEN_CHOICE = "critic_then_choice"


@dataclass(frozen=True)
class PromptContext:
    """Context data for constructing prompts.

    Attributes
    ----------
    game_name:
        Name of the game (e.g., "tic_tac_toe", "breakthrough").
    state_description:
        Human-readable representation of the current game state.
    legal_actions:
        Human-readable descriptions of legal actions.
    legal_action_ids:
        Corresponding integer action IDs.
    player_id:
        Current player's index (0 or 1).
    turn_number:
        Current turn/move number.
    history:
        Optional list of previous turn descriptions for context.
    extra:
        Additional game-specific metadata.
    """

    game_name: str
    state_description: str
    legal_actions: list[str]
    legal_action_ids: list[int]
    player_id: int
    turn_number: int
    history: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def format_legal_actions(self) -> str:
        """Format legal actions as a numbered list."""
        lines = []
        for i, (action_desc, action_id) in enumerate(
            zip(self.legal_actions, self.legal_action_ids)
        ):
            lines.append(f"  {i + 1}. {action_desc} (ID: {action_id})")
        return "\n".join(lines)

    def format_history(self, max_turns: int | None = None) -> str:
        """Format history for inclusion in prompt.

        Parameters
        ----------
        max_turns:
            Maximum number of recent turns to include. None means all.

        Returns
        -------
        str
            Formatted history string.
        """
        if not self.history:
            return "No previous moves."

        turns = self.history
        if max_turns is not None and max_turns > 0:
            turns = turns[-max_turns:]

        lines = [f"Last {len(turns)} turn(s):"]
        for i, turn_desc in enumerate(turns):
            lines.append(f"  Turn {i + 1}: {turn_desc}")
        return "\n".join(lines)


class PromptTemplate(ABC):
    """Abstract base class for prompt templates."""

    @property
    @abstractmethod
    def style(self) -> PromptStyle:
        """Return the prompt style identifier."""
        ...

    @abstractmethod
    def build_prompt(self, context: PromptContext) -> str:
        """Build the full prompt from context.

        Parameters
        ----------
        context:
            The prompt context containing game state and history.

        Returns
        -------
        str
            The complete prompt string.
        """
        ...

    @abstractmethod
    def build_correction_prompt(
        self,
        context: PromptContext,
        invalid_response: str,
        error_message: str,
    ) -> str:
        """Build a correction prompt after an invalid response.

        Parameters
        ----------
        context:
            The original prompt context.
        invalid_response:
            The model's invalid response.
        error_message:
            Description of why the response was invalid.

        Returns
        -------
        str
            A stricter prompt asking for correction.
        """
        ...


class ZeroShotPrompt(PromptTemplate):
    """Minimal prompt with no guidance - just asks for a move."""

    @property
    def style(self) -> PromptStyle:
        return PromptStyle.ZERO_SHOT

    def build_prompt(self, context: PromptContext) -> str:
        lines = [
            f"You are playing {context.game_name}.",
            f"You are Player {context.player_id}.",
            f"Turn: {context.turn_number}",
            "",
            "Current board state:",
            context.state_description,
            "",
            "Choose your next move.",
        ]

        if context.history:
            lines.append("")
            lines.append(context.format_history())

        lines.append("")
        lines.append("Respond with only the action ID number.")

        return "\n".join(lines)

    def build_correction_prompt(
        self,
        context: PromptContext,
        invalid_response: str,
        error_message: str,
    ) -> str:
        return "\n".join([
            "Your previous response was invalid.",
            f"Response: {invalid_response}",
            f"Error: {error_message}",
            "",
            "You MUST respond with ONLY a single number from this list:",
            ", ".join(str(a) for a in context.legal_action_ids),
        ])


class LegalMovesOnlyPrompt(PromptTemplate):
    """Prompt that lists legal moves and asks agent to pick one."""

    @property
    def style(self) -> PromptStyle:
        return PromptStyle.LEGAL_MOVES_ONLY

    def build_prompt(self, context: PromptContext) -> str:
        lines = [
            f"You are playing {context.game_name}.",
            f"You are Player {context.player_id}.",
            f"Turn: {context.turn_number}",
            "",
            "Current board state:",
            context.state_description,
            "",
            "Legal moves:",
            context.format_legal_actions(),
        ]

        if context.history:
            lines.append("")
            lines.append(context.format_history())

        lines.extend([
            "",
            "Choose one move from the legal moves above.",
            "Respond with ONLY the action ID number (nothing else).",
        ])

        return "\n".join(lines)

    def build_correction_prompt(
        self,
        context: PromptContext,
        invalid_response: str,
        error_message: str,
    ) -> str:
        return "\n".join([
            "Your previous response was INVALID.",
            f"Response: {invalid_response}",
            f"Error: {error_message}",
            "",
            "Choose a move from this EXACT list:",
            context.format_legal_actions(),
            "",
            "IMPORTANT: Respond with ONLY the action ID number.",
            "Valid IDs: " + ", ".join(str(a) for a in context.legal_action_ids),
        ])


class BoardSummaryThenChoicePrompt(PromptTemplate):
    """Prompt that asks for board analysis before move selection."""

    @property
    def style(self) -> PromptStyle:
        return PromptStyle.BOARD_SUMMARY_THEN_CHOICE

    def build_prompt(self, context: PromptContext) -> str:
        lines = [
            f"You are playing {context.game_name}.",
            f"You are Player {context.player_id}.",
            f"Turn: {context.turn_number}",
            "",
            "Current board state:",
            context.state_description,
            "",
            "Legal moves:",
            context.format_legal_actions(),
        ]

        if context.history:
            lines.append("")
            lines.append(context.format_history())

        lines.extend([
            "",
            "First, briefly analyze the board position.",
            "Then, select your move.",
            "",
            "Format your response as:",
            "ANALYSIS: <your brief analysis>",
            "MOVE: <action ID>",
        ])

        return "\n".join(lines)

    def build_correction_prompt(
        self,
        context: PromptContext,
        invalid_response: str,
        error_message: str,
    ) -> str:
        return "\n".join([
            "Your previous response was INVALID.",
            f"Response: {invalid_response}",
            f"Error: {error_message}",
            "",
            "You must include a valid MOVE line.",
            "Choose from these legal moves:",
            context.format_legal_actions(),
            "",
            "Format:",
            "ANALYSIS: <brief analysis>",
            "MOVE: <action ID from: " + ", ".join(str(a) for a in context.legal_action_ids) + ">",
        ])


class ReasonThenChoicePrompt(PromptTemplate):
    """Prompt that asks agent to reason through the decision."""

    @property
    def style(self) -> PromptStyle:
        return PromptStyle.REASON_THEN_CHOICE

    def build_prompt(self, context: PromptContext) -> str:
        lines = [
            f"You are playing {context.game_name}.",
            f"You are Player {context.player_id}.",
            f"Turn: {context.turn_number}",
            "",
            "Current board state:",
            context.state_description,
            "",
            "Legal moves:",
            context.format_legal_actions(),
        ]

        if context.history:
            lines.append("")
            lines.append(context.format_history())

        lines.extend([
            "",
            "Think through your decision carefully:",
            "1. What is the current game state?",
            "2. What are your strategic options?",
            "3. Which move gives you the best position?",
            "",
            "Format your response as:",
            "REASONING: <your step-by-step reasoning>",
            "CHOICE: <action ID>",
        ])

        return "\n".join(lines)

    def build_correction_prompt(
        self,
        context: PromptContext,
        invalid_response: str,
        error_message: str,
    ) -> str:
        return "\n".join([
            "Your previous response was INVALID.",
            f"Response: {invalid_response}",
            f"Error: {error_message}",
            "",
            "You must include a valid CHOICE line.",
            "Choose from these legal moves:",
            context.format_legal_actions(),
            "",
            "Format:",
            "REASONING: <your reasoning>",
            "CHOICE: <action ID from: " + ", ".join(str(a) for a in context.legal_action_ids) + ">",
        ])


class CriticThenChoicePrompt(PromptTemplate):
    """Prompt that asks agent to propose, critique, then finalize."""

    @property
    def style(self) -> PromptStyle:
        return PromptStyle.CRITIC_THEN_CHOICE

    def build_prompt(self, context: PromptContext) -> str:
        lines = [
            f"You are playing {context.game_name}.",
            f"You are Player {context.player_id}.",
            f"Turn: {context.turn_number}",
            "",
            "Current board state:",
            context.state_description,
            "",
            "Legal moves:",
            context.format_legal_actions(),
        ]

        if context.history:
            lines.append("")
            lines.append(context.format_history())

        lines.extend([
            "",
            "Use a two-step decision process:",
            "1. PROPOSE: Identify a candidate move",
            "2. CRITIQUE: Evaluate potential issues with your proposal",
            "3. FINAL: Make your final decision (can be same or different)",
            "",
            "Format your response as:",
            "PROPOSE: <action ID>",
            "CRITIQUE: <evaluation of the proposed move>",
            "FINAL: <action ID>",
        ])

        return "\n".join(lines)

    def build_correction_prompt(
        self,
        context: PromptContext,
        invalid_response: str,
        error_message: str,
    ) -> str:
        return "\n".join([
            "Your previous response was INVALID.",
            f"Response: {invalid_response}",
            f"Error: {error_message}",
            "",
            "You must include a valid FINAL line.",
            "Choose from these legal moves:",
            context.format_legal_actions(),
            "",
            "Format:",
            "PROPOSE: <action ID>",
            "CRITIQUE: <your evaluation>",
            "FINAL: <action ID from: " + ", ".join(str(a) for a in context.legal_action_ids) + ">",
        ])


# Registry of prompt templates
_PROMPT_TEMPLATES: dict[PromptStyle, type[PromptTemplate]] = {
    PromptStyle.ZERO_SHOT: ZeroShotPrompt,
    PromptStyle.LEGAL_MOVES_ONLY: LegalMovesOnlyPrompt,
    PromptStyle.BOARD_SUMMARY_THEN_CHOICE: BoardSummaryThenChoicePrompt,
    PromptStyle.REASON_THEN_CHOICE: ReasonThenChoicePrompt,
    PromptStyle.CRITIC_THEN_CHOICE: CriticThenChoicePrompt,
}


def get_template(style: PromptStyle | str) -> PromptTemplate:
    """Get a prompt template instance by style.

    Parameters
    ----------
    style:
        The prompt style (enum value or string name).

    Returns
    -------
    PromptTemplate
        A new instance of the requested template.

    Raises
    ------
    ValueError
        If the style is not recognized.
    """
    if isinstance(style, str):
        try:
            style = PromptStyle(style)
        except ValueError:
            valid = [s.value for s in PromptStyle]
            raise ValueError(
                f"Unknown prompt style: {style!r}. Valid styles: {valid}"
            ) from None

    template_class = _PROMPT_TEMPLATES.get(style)
    if template_class is None:
        raise ValueError(f"No template registered for style: {style}")

    return template_class()


def parse_action_from_response(response: str, legal_action_ids: list[int]) -> int | None:
    """Parse an action ID from an LLM response.

    Tries multiple parsing strategies:
    1. Look for explicit action markers (MOVE:, CHOICE:, FINAL:)
    2. Find standalone numbers that match legal actions
    3. Find action ID in parentheses format "(ID: N)"

    Parameters
    ----------
    response:
        The raw LLM response text.
    legal_action_ids:
        List of valid action IDs.

    Returns
    -------
    int | None
        The parsed action ID if found and valid, None otherwise.
    """
    response = response.strip()
    legal_set = set(legal_action_ids)

    # Strategy 1: Look for explicit action markers
    markers = ["MOVE:", "CHOICE:", "FINAL:", "ACTION:", "PLAY:"]
    for marker in markers:
        if marker in response.upper():
            # Find the marker (case-insensitive)
            idx = response.upper().find(marker)
            after_marker = response[idx + len(marker) :].strip()
            # Extract first number from the text after marker
            num = _extract_first_number(after_marker)
            if num is not None and num in legal_set:
                return num

    # Strategy 2: Look for "(ID: N)" format
    import re

    id_pattern = r"\(ID:\s*(\d+)\)"
    matches = re.findall(id_pattern, response)
    for match in matches:
        num = int(match)
        if num in legal_set:
            return num

    # Strategy 3: Find standalone numbers that match legal actions
    # Prefer numbers that appear alone on a line
    lines = response.strip().split("\n")
    for line in reversed(lines):  # Check from end (usually where answer is)
        line = line.strip()
        # Check if line is just a number
        if line.isdigit():
            num = int(line)
            if num in legal_set:
                return num
        # Check for "N." or "N)" patterns at start of line
        match = re.match(r"^(\d+)[.\)]\s*$", line)
        if match:
            num = int(match.group(1))
            if num in legal_set:
                return num

    # Strategy 4: Extract all numbers and find first legal one
    num = _extract_first_number(response)
    if num is not None and num in legal_set:
        return num

    return None


def _extract_first_number(text: str) -> int | None:
    """Extract the first integer from text."""
    import re

    match = re.search(r"\b(\d+)\b", text)
    if match:
        return int(match.group(1))
    return None
