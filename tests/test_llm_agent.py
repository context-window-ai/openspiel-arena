"""
Tests for agents.llm_agent and agents.prompts — LLM agent with prompts and memory.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from agents.llm_agent import (
    LLMAgent,
    LLMAgentConfig,
    TurnRecord,
    FallbackMode,
    create_llm_agent,
)
from agents.prompts import (
    PromptStyle,
    PromptContext,
    PromptTemplate,
    get_template,
    parse_action_from_response,
)
from agents.base import ActionContext


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def prompt_context() -> PromptContext:
    """Create a sample prompt context for testing."""
    return PromptContext(
        game_name="tic_tac_toe",
        state_description="...\n...\n...",
        legal_actions=["0", "1", "2", "3", "4", "5", "6", "7", "8"],
        legal_action_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8],
        player_id=0,
        turn_number=1,
    )


@pytest.fixture
def breakthrough_context() -> PromptContext:
    """Create a breakthrough game context for testing."""
    return PromptContext(
        game_name="breakthrough",
        state_description="8bbbbbbbb\n7bbbbbbbb\n6........\n5........\n4........\n3........\n2wwwwwwww\n1wwwwwwww\n abcdefgh",
        legal_actions=["a7a6", "a7b6*", "b7b6"],
        legal_action_ids=[98, 99, 100],
        player_id=0,
        turn_number=1,
    )


@pytest.fixture
def prompt_context_with_history() -> PromptContext:
    """Create a prompt context with history for memory testing."""
    return PromptContext(
        game_name="tic_tac_toe",
        state_description="O..\n.X.\n...",
        legal_actions=["6", "7", "8"],
        legal_action_ids=[6, 7, 8],
        player_id=0,
        turn_number=3,
        history=[
            "Player 0 played 4",
            "Player 1 played 0",
        ],
    )


# ---------------------------------------------------------------------------
# Tests for PromptStyle enum
# ---------------------------------------------------------------------------


def test_prompt_style_values():
    """Test that all required prompt styles are defined."""
    assert PromptStyle.ZERO_SHOT.value == "zero_shot"
    assert PromptStyle.LEGAL_MOVES_ONLY.value == "legal_moves_only"
    assert PromptStyle.BOARD_SUMMARY_THEN_CHOICE.value == "board_summary_then_choice"
    assert PromptStyle.REASON_THEN_CHOICE.value == "reason_then_choice"
    assert PromptStyle.CRITIC_THEN_CHOICE.value == "critic_then_choice"


# ---------------------------------------------------------------------------
# Tests for PromptContext
# ---------------------------------------------------------------------------


def test_prompt_context_creation(prompt_context):
    """Test PromptContext dataclass creation."""
    assert prompt_context.game_name == "tic_tac_toe"
    assert prompt_context.player_id == 0
    assert len(prompt_context.legal_actions) == 9
    assert prompt_context.turn_number == 1


def test_prompt_context_format_legal_actions(prompt_context):
    """Test format_legal_actions method."""
    formatted = prompt_context.format_legal_actions()
    assert "0 (ID: 0)" in formatted
    assert "4 (ID: 4)" in formatted


def test_prompt_context_format_history(prompt_context_with_history):
    """Test format_history method."""
    formatted = prompt_context_with_history.format_history()
    assert "Last 2 turn(s):" in formatted
    assert "Player 0 played 4" in formatted


def test_prompt_context_format_history_limited(prompt_context_with_history):
    """Test format_history with max_turns limit."""
    formatted = prompt_context_with_history.format_history(max_turns=1)
    assert "Last 1 turn(s):" in formatted
    # Should only include the last turn
    assert "Player 1 played 0" in formatted


def test_prompt_context_format_history_empty(prompt_context):
    """Test format_history with no history."""
    formatted = prompt_context.format_history()
    assert "No previous moves" in formatted


# ---------------------------------------------------------------------------
# Tests for get_template
# ---------------------------------------------------------------------------


def test_get_template_by_enum():
    """Test getting template by enum value."""
    template = get_template(PromptStyle.LEGAL_MOVES_ONLY)
    assert template.style == PromptStyle.LEGAL_MOVES_ONLY


def test_get_template_by_string():
    """Test getting template by string name."""
    template = get_template("legal_moves_only")
    assert template.style == PromptStyle.LEGAL_MOVES_ONLY


def test_get_template_invalid():
    """Test getting template with invalid style."""
    with pytest.raises(ValueError, match="Unknown prompt style"):
        get_template("invalid_style")


# ---------------------------------------------------------------------------
# Tests for prompt templates
# ---------------------------------------------------------------------------


class TestZeroShotPrompt:
    """Tests for ZeroShotPrompt template."""

    def test_style(self):
        template = get_template(PromptStyle.ZERO_SHOT)
        assert template.style == PromptStyle.ZERO_SHOT

    def test_build_prompt(self, prompt_context):
        template = get_template(PromptStyle.ZERO_SHOT)
        prompt = template.build_prompt(prompt_context)
        assert "tic_tac_toe" in prompt.lower()
        assert "Player 0" in prompt
        assert "action ID" in prompt.lower()

    def test_build_correction_prompt(self, prompt_context):
        template = get_template(PromptStyle.ZERO_SHOT)
        correction = template.build_correction_prompt(
            prompt_context, "invalid response", "Could not parse"
        )
        assert "invalid" in correction.lower()
        assert "invalid response" in correction


class TestLegalMovesOnlyPrompt:
    """Tests for LegalMovesOnlyPrompt template."""

    def test_style(self):
        template = get_template(PromptStyle.LEGAL_MOVES_ONLY)
        assert template.style == PromptStyle.LEGAL_MOVES_ONLY

    def test_build_prompt(self, prompt_context):
        template = get_template(PromptStyle.LEGAL_MOVES_ONLY)
        prompt = template.build_prompt(prompt_context)
        assert "Legal moves:" in prompt
        assert "0 (ID: 0)" in prompt
        assert "ONLY the action ID" in prompt

    def test_build_prompt_with_history(self, prompt_context_with_history):
        template = get_template(PromptStyle.LEGAL_MOVES_ONLY)
        prompt = template.build_prompt(prompt_context_with_history)
        assert "Last 2 turn(s):" in prompt

    def test_build_correction_prompt(self, prompt_context):
        template = get_template(PromptStyle.LEGAL_MOVES_ONLY)
        correction = template.build_correction_prompt(
            prompt_context, "99", "Invalid action ID"
        )
        assert "INVALID" in correction
        assert "99" in correction
        assert "Valid IDs:" in correction


class TestBoardSummaryThenChoicePrompt:
    """Tests for BoardSummaryThenChoicePrompt template."""

    def test_style(self):
        template = get_template(PromptStyle.BOARD_SUMMARY_THEN_CHOICE)
        assert template.style == PromptStyle.BOARD_SUMMARY_THEN_CHOICE

    def test_build_prompt(self, prompt_context):
        template = get_template(PromptStyle.BOARD_SUMMARY_THEN_CHOICE)
        prompt = template.build_prompt(prompt_context)
        assert "ANALYSIS:" in prompt
        assert "MOVE:" in prompt

    def test_build_correction_prompt(self, prompt_context):
        template = get_template(PromptStyle.BOARD_SUMMARY_THEN_CHOICE)
        correction = template.build_correction_prompt(
            prompt_context, "no move line", "Missing MOVE line"
        )
        assert "MOVE line" in correction


class TestReasonThenChoicePrompt:
    """Tests for ReasonThenChoicePrompt template."""

    def test_style(self):
        template = get_template(PromptStyle.REASON_THEN_CHOICE)
        assert template.style == PromptStyle.REASON_THEN_CHOICE

    def test_build_prompt(self, prompt_context):
        template = get_template(PromptStyle.REASON_THEN_CHOICE)
        prompt = template.build_prompt(prompt_context)
        assert "REASONING:" in prompt
        assert "CHOICE:" in prompt
        assert "Think through" in prompt

    def test_build_correction_prompt(self, prompt_context):
        template = get_template(PromptStyle.REASON_THEN_CHOICE)
        correction = template.build_correction_prompt(
            prompt_context, "no choice", "Missing CHOICE line"
        )
        assert "CHOICE line" in correction


class TestCriticThenChoicePrompt:
    """Tests for CriticThenChoicePrompt template."""

    def test_style(self):
        template = get_template(PromptStyle.CRITIC_THEN_CHOICE)
        assert template.style == PromptStyle.CRITIC_THEN_CHOICE

    def test_build_prompt(self, prompt_context):
        template = get_template(PromptStyle.CRITIC_THEN_CHOICE)
        prompt = template.build_prompt(prompt_context)
        assert "PROPOSE:" in prompt
        assert "CRITIQUE:" in prompt
        assert "FINAL:" in prompt

    def test_build_correction_prompt(self, prompt_context):
        template = get_template(PromptStyle.CRITIC_THEN_CHOICE)
        correction = template.build_correction_prompt(
            prompt_context, "no final", "Missing FINAL line"
        )
        assert "FINAL line" in correction


# ---------------------------------------------------------------------------
# Tests for action parsing
# ---------------------------------------------------------------------------


class TestParseActionFromResponse:
    """Tests for parse_action_from_response function."""

    def test_parse_direct_integer(self):
        """Test parsing direct integer action."""
        result = parse_action_from_response("4", [0, 1, 2, 3, 4, 5])
        assert result == 4

    def test_parse_with_move_marker(self):
        """Test parsing with MOVE: marker."""
        result = parse_action_from_response(
            "After analysis...\nMOVE: 4",
            [0, 1, 2, 3, 4, 5],
        )
        assert result == 4

    def test_parse_with_choice_marker(self):
        """Test parsing with CHOICE: marker."""
        result = parse_action_from_response(
            "REASONING: Center is good\nCHOICE: 4",
            [0, 1, 2, 3, 4, 5],
        )
        assert result == 4

    def test_parse_with_final_marker(self):
        """Test parsing with FINAL: marker."""
        result = parse_action_from_response(
            "PROPOSE: 3\nCRITIQUE: Not great\nFINAL: 5",
            [0, 1, 2, 3, 4, 5],
        )
        assert result == 5

    def test_parse_with_id_format(self):
        """Test parsing with (ID: N) format."""
        result = parse_action_from_response(
            "I choose 4 (ID: 4)",
            [0, 1, 2, 3, 4, 5],
        )
        assert result == 4

    def test_parse_standalone_number_on_line(self):
        """Test parsing standalone number on a line."""
        result = parse_action_from_response(
            "The best move is:\n4",
            [0, 1, 2, 3, 4, 5],
        )
        assert result == 4

    def test_parse_invalid_response(self):
        """Test parsing invalid response returns None."""
        result = parse_action_from_response(
            "I have no idea",
            [0, 1, 2],
        )
        assert result is None

    def test_parse_out_of_range(self):
        """Test parsing action not in legal moves."""
        result = parse_action_from_response("99", [0, 1, 2])
        assert result is None

    def test_parse_case_insensitive_marker(self):
        """Test case-insensitive marker parsing."""
        result = parse_action_from_response(
            "move: 3",
            [0, 1, 2, 3, 4],
        )
        assert result == 3

    def test_parse_chooses_first_legal_from_markers(self):
        """Test that marker-based parsing gets the first number after marker."""
        result = parse_action_from_response(
            "MOVE: 2 is good but I want 3",
            [0, 1, 2, 3, 4],
        )
        assert result == 2


# ---------------------------------------------------------------------------
# Tests for LLMAgentConfig
# ---------------------------------------------------------------------------


class TestLLMAgentConfig:
    """Tests for LLMAgentConfig dataclass."""

    def test_defaults(self):
        """Test default config values."""
        config = LLMAgentConfig()
        assert config.model == "minimax/minimax-m2.5:free"
        assert config.prompt_style == PromptStyle.LEGAL_MOVES_ONLY
        assert config.memory_turns == 0
        assert config.fallback_mode == FallbackMode.RANDOM
        assert config.temperature == 0.7
        assert config.max_tokens == 500

    def test_custom_values(self):
        """Test custom config values."""
        config = LLMAgentConfig(
            model="gpt-4",
            prompt_style=PromptStyle.REASON_THEN_CHOICE,
            memory_turns=5,
            temperature=0.5,
            fallback_mode=FallbackMode.FIRST,
        )
        assert config.model == "gpt-4"
        assert config.prompt_style == PromptStyle.REASON_THEN_CHOICE
        assert config.memory_turns == 5
        assert config.fallback_mode == FallbackMode.FIRST

    def test_string_to_enum_conversion(self):
        """Test automatic string to enum conversion."""
        config = LLMAgentConfig(
            prompt_style="reason_then_choice",
            fallback_mode="first",
        )
        assert config.prompt_style == PromptStyle.REASON_THEN_CHOICE
        assert config.fallback_mode == FallbackMode.FIRST

    def test_invalid_memory_turns(self):
        """Test validation of memory_turns."""
        with pytest.raises(ValueError, match="memory_turns"):
            LLMAgentConfig(memory_turns=-1)

    def test_invalid_temperature(self):
        """Test validation of temperature."""
        with pytest.raises(ValueError, match="temperature"):
            LLMAgentConfig(temperature=3.0)

    def test_serialization(self):
        """Test config serialization."""
        config = LLMAgentConfig(
            model="gpt-4",
            prompt_style=PromptStyle.CRITIC_THEN_CHOICE,
            memory_turns=3,
        )
        data = config.to_dict()
        restored = LLMAgentConfig.from_dict(data)
        assert restored.model == config.model
        assert restored.prompt_style == config.prompt_style
        assert restored.memory_turns == config.memory_turns


# ---------------------------------------------------------------------------
# Tests for LLMAgent
# ---------------------------------------------------------------------------


class MockState:
    """Mock OpenSpiel state for testing."""

    def __init__(self, legal_actions: list[int]):
        self._legal_actions = legal_actions

    def legal_actions(self):
        return list(self._legal_actions)

    def current_player(self):
        return 0

    def __str__(self):
        return "Mock board state"


class TestLLMAgent:
    """Tests for LLMAgent class."""

    def test_agent_creation_without_api_key(self):
        """Test that agent raises error without API key."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
                LLMAgent("test")

    def test_agent_creation_with_config(self):
        """Test agent creation with config."""
        config = LLMAgentConfig(
            model="test-model",
            prompt_style=PromptStyle.LEGAL_MOVES_ONLY,
        )
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
            agent = LLMAgent("test", config=config)
            assert agent.name == "test"
            assert agent.config.model == "test-model"

    def test_agent_reset(self):
        """Test resetting agent state."""
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
            agent = LLMAgent("test")
            agent._game_history.append("turn 1")
            agent.reset()
            assert len(agent._game_history) == 0
            assert len(agent.turn_history) == 0

    @patch("agents.llm_agent.LLMAgent._call_llm")
    def test_agent_select_action_valid(self, mock_call):
        """Test select_action with valid LLM response."""
        mock_call.return_value = "4"

        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
            agent = LLMAgent("test")
            state = MockState([0, 1, 2, 3, 4, 5])
            context = ActionContext(game_name="test_game", player_id=0, turn_number=1)
            action = agent.select_action(state, [0, 1, 2, 3, 4, 5], context)
            assert action == 4

    @patch("agents.llm_agent.LLMAgent._call_llm")
    def test_agent_select_action_fallback_random(self, mock_call):
        """Test select_action with fallback to random."""
        mock_call.return_value = "invalid response"

        config = LLMAgentConfig(fallback_mode=FallbackMode.RANDOM)
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
            agent = LLMAgent("test", config=config)
            state = MockState([0, 1, 2])
            context = ActionContext(game_name="test_game", player_id=0, turn_number=1)
            action = agent.select_action(state, [0, 1, 2], context)
            assert action in [0, 1, 2]

    @patch("agents.llm_agent.LLMAgent._call_llm")
    def test_agent_select_action_fallback_first(self, mock_call):
        """Test select_action with fallback to first legal."""
        mock_call.return_value = "invalid response"

        config = LLMAgentConfig(fallback_mode=FallbackMode.FIRST)
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
            agent = LLMAgent("test", config=config)
            state = MockState([0, 1, 2])
            context = ActionContext(game_name="test_game", player_id=0, turn_number=1)
            action = agent.select_action(state, [0, 1, 2], context)
            assert action == 0

    def test_agent_select_action_empty_legal(self):
        """Test select_action raises with empty legal actions."""
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
            agent = LLMAgent("test")
            state = MockState([])
            with pytest.raises(ValueError, match="no legal actions"):
                agent.select_action(state, [])

    @patch("agents.llm_agent.LLMAgent._call_llm")
    def test_agent_records_turn(self, mock_call):
        """Test that agent records turn history."""
        mock_call.return_value = "4"

        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
            agent = LLMAgent("test")
            state = MockState([0, 1, 2, 3, 4, 5])
            context = ActionContext(game_name="test_game", player_id=0, turn_number=1)
            agent.select_action(state, [0, 1, 2, 3, 4, 5], context)

            assert len(agent.turn_history) == 1
            record = agent.turn_history[0]
            assert record.turn_number == 1
            assert record.chosen_action == 4
            assert not record.was_fallback


# ---------------------------------------------------------------------------
# Tests for create_llm_agent factory
# ---------------------------------------------------------------------------


class TestCreateLLMAgent:
    """Tests for create_llm_agent factory function."""

    def test_defaults(self):
        """Test factory with defaults."""
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
            agent = create_llm_agent("test")
            assert agent.name == "test"
            assert agent.config.prompt_style == PromptStyle.LEGAL_MOVES_ONLY
            assert agent.config.memory_turns == 0

    def test_custom_options(self):
        """Test factory with custom options."""
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
            agent = create_llm_agent(
                "test",
                model="gpt-4",
                prompt_style="reason_then_choice",
                memory_turns=5,
                temperature=0.3,
                fallback_mode="first",
            )
            assert agent.config.model == "gpt-4"
            assert agent.config.prompt_style == PromptStyle.REASON_THEN_CHOICE
            assert agent.config.memory_turns == 5
            assert agent.config.temperature == 0.3
            assert agent.config.fallback_mode == FallbackMode.FIRST


# ---------------------------------------------------------------------------
# Tests for memory_turns configuration
# ---------------------------------------------------------------------------


class TestMemoryTurns:
    """Tests for different memory_turns settings."""

    @patch("agents.llm_agent.LLMAgent._call_llm")
    def test_memory_turns_zero(self, mock_call):
        """Test that memory_turns=0 results in stateless prompts."""
        mock_call.return_value = "0"

        config = LLMAgentConfig(memory_turns=0)
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
            agent = LLMAgent("test", config=config)
            state = MockState([0, 1, 2])

            # Make a move
            agent.select_action(state, [0, 1, 2], ActionContext(turn_number=1))

            # Check that game history is not used in prompts
            history = agent._get_memory_history()
            assert history == []

    @patch("agents.llm_agent.LLMAgent._call_llm")
    def test_memory_turns_positive(self, mock_call):
        """Test that memory_turns > 0 includes history in prompts."""
        mock_call.return_value = "0"

        config = LLMAgentConfig(memory_turns=3)
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
            agent = LLMAgent("test", config=config)
            state = MockState([0, 1, 2])

            # Make several moves
            for i in range(5):
                agent.select_action(
                    state, [0, 1, 2], ActionContext(turn_number=i + 1)
                )

            # Check that history is limited to 3 turns
            history = agent._get_memory_history()
            assert len(history) == 3


# ---------------------------------------------------------------------------
# Tests for retry behavior
# ---------------------------------------------------------------------------


class TestRetryBehavior:
    """Tests for retry and correction behavior."""

    @patch("agents.llm_agent.LLMAgent._call_llm")
    def test_retry_on_invalid_response(self, mock_call):
        """Test that agent retries with correction prompt on invalid response."""
        # First call returns invalid, second call returns valid
        mock_call.side_effect = ["invalid response", "4"]

        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
            agent = LLMAgent("test")
            state = MockState([0, 1, 2, 3, 4, 5])
            context = ActionContext(game_name="test_game", player_id=0, turn_number=1)
            action = agent.select_action(state, [0, 1, 2, 3, 4, 5], context)

            assert action == 4
            assert mock_call.call_count == 2

            # Check that retry was recorded
            record = agent.turn_history[0]
            assert record.was_retry

    @patch("agents.llm_agent.LLMAgent._call_llm")
    def test_fallback_after_retry_failure(self, mock_call):
        """Test fallback after both attempts fail."""
        mock_call.return_value = "always invalid"

        config = LLMAgentConfig(fallback_mode=FallbackMode.FIRST)
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
            agent = LLMAgent("test", config=config)
            state = MockState([0, 1, 2])
            context = ActionContext(game_name="test_game", player_id=0, turn_number=1)
            action = agent.select_action(state, [0, 1, 2], context)

            assert action == 0  # First legal fallback

            # Check that fallback was recorded
            record = agent.turn_history[0]
            assert record.was_fallback


# ---------------------------------------------------------------------------
# Tests for all prompt styles
# ---------------------------------------------------------------------------


class TestAllPromptStyles:
    """Test that all prompt styles work."""

    @pytest.mark.parametrize("style", list(PromptStyle))
    def test_get_template_all_styles(self, style):
        """Test that all styles can be retrieved."""
        template = get_template(style)
        assert template.style == style

    @pytest.mark.parametrize("style", list(PromptStyle))
    def test_build_prompt_all_styles(self, style, prompt_context):
        """Test that all styles produce valid prompts."""
        template = get_template(style)
        prompt = template.build_prompt(prompt_context)
        assert len(prompt) > 0
        assert "tic_tac_toe" in prompt.lower()

    @pytest.mark.parametrize("style", list(PromptStyle))
    def test_build_correction_prompt_all_styles(self, style, prompt_context):
        """Test that all styles produce valid correction prompts."""
        template = get_template(style)
        correction = template.build_correction_prompt(
            prompt_context, "invalid", "test error"
        )
        assert len(correction) > 0
        assert "invalid" in correction.lower() or "error" in correction.lower()


# ---------------------------------------------------------------------------
# Tests for FallbackMode enum
# ---------------------------------------------------------------------------


class TestFallbackMode:
    """Tests for FallbackMode enum."""

    def test_values(self):
        """Test enum values."""
        assert FallbackMode.RANDOM.value == "random"
        assert FallbackMode.FIRST.value == "first"

    def test_string_conversion(self):
        """Test string to enum conversion."""
        assert FallbackMode("random") == FallbackMode.RANDOM
        assert FallbackMode("first") == FallbackMode.FIRST
