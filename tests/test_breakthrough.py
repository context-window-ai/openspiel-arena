"""
tests.test_breakthrough — Unit tests for the Breakthrough game wrapper.
"""

import pytest

from games.breakthrough import BreakthroughAction, BreakthroughGame


class TestBreakthroughGame:
    """Tests for BreakthroughGame class."""

    def test_default_initialization(self) -> None:
        """Test default 8x8 game initialization."""
        game = BreakthroughGame()
        assert game.name == "breakthrough"
        assert game.num_players == 2
        assert game.columns == 8
        assert game.rows == 8

    def test_custom_board_size(self) -> None:
        """Test custom board size initialization."""
        game = BreakthroughGame(columns=6, rows=6)
        assert game.columns == 6
        assert game.rows == 6

        state = game.new_state()
        board_str = str(state)
        # 6x6 board should have rows 1-6 and columns a-f
        assert "6" in board_str
        assert "abcdef" in board_str

    def test_new_state(self) -> None:
        """Test creating a new initial state."""
        game = BreakthroughGame()
        state = game.new_state()

        assert not game.is_terminal(state)
        assert game.current_player(state) == 0  # Black moves first

    def test_initial_board_setup(self) -> None:
        """Test that initial board has correct piece placement."""
        game = BreakthroughGame()
        state = game.new_state()
        board = game.state_string(state)

        # Black pieces on rows 7 and 8
        lines = board.strip().split("\n")
        assert "bbbbbbbb" in lines[0]  # Row 8
        assert "bbbbbbbb" in lines[1]  # Row 7

        # White pieces on rows 1 and 2
        assert "wwwwwwww" in lines[-3]  # Row 2
        assert "wwwwwwww" in lines[-2]  # Row 1

    def test_legal_actions_initial_state(self) -> None:
        """Test legal actions in initial state."""
        game = BreakthroughGame()
        state = game.new_state()
        actions = game.legal_actions(state)

        # Should have legal moves for black's front row
        assert len(actions) > 0
        # In 8x8, black has 22 initial moves (front row can move forward or diagonal)
        assert len(actions) == 22

    def test_legal_actions_returns_list(self) -> None:
        """Test that legal_actions returns a list of integers."""
        game = BreakthroughGame()
        state = game.new_state()
        actions = game.legal_actions(state)

        assert isinstance(actions, list)
        assert all(isinstance(a, int) for a in actions)

    def test_apply_action(self) -> None:
        """Test applying an action changes the state."""
        game = BreakthroughGame()
        state = game.new_state()

        initial_actions = game.legal_actions(state)
        action = initial_actions[0]

        game.apply_action(state, action)

        # State should have changed
        assert game.current_player(state) == 1  # Now white's turn

    def test_apply_action_returns_state(self) -> None:
        """Test that apply_action returns the state object."""
        game = BreakthroughGame()
        state = game.new_state()
        action = game.legal_actions(state)[0]

        returned_state = game.apply_action(state, action)
        assert returned_state is state

    def test_action_to_string(self) -> None:
        """Test converting action ID to string notation."""
        game = BreakthroughGame()
        state = game.new_state()
        action = game.legal_actions(state)[0]

        notation = game.action_to_string(state, action)
        # Should be something like "a7a6"
        assert len(notation) >= 4
        assert notation[0] in "abcdefgh"
        assert notation[2] in "abcdefgh"

    def test_string_to_action(self) -> None:
        """Test converting string notation to action ID."""
        game = BreakthroughGame()
        state = game.new_state()

        # Get a known action
        actions = game.legal_actions(state)
        original_action = actions[0]
        notation = game.action_to_string(state, original_action)

        # Convert back
        converted_action = game.string_to_action(state, notation)
        assert converted_action == original_action

    def test_current_player_alternates(self) -> None:
        """Test that current player alternates after each move."""
        game = BreakthroughGame()
        state = game.new_state()

        assert game.current_player(state) == 0

        action = game.legal_actions(state)[0]
        game.apply_action(state, action)
        assert game.current_player(state) == 1

        action = game.legal_actions(state)[0]
        game.apply_action(state, action)
        assert game.current_player(state) == 0

    def test_is_terminal_initial_state(self) -> None:
        """Test that initial state is not terminal."""
        game = BreakthroughGame()
        state = game.new_state()
        assert not game.is_terminal(state)

    def test_returns_initial_state(self) -> None:
        """Test returns in initial state are zeros."""
        game = BreakthroughGame()
        state = game.new_state()
        returns = game.returns(state)
        assert returns == [0.0, 0.0]

    def test_winner_none_when_not_terminal(self) -> None:
        """Test winner returns None for non-terminal state."""
        game = BreakthroughGame()
        state = game.new_state()
        assert game.winner(state) is None


class TestBreakthroughActionMetadata:
    """Tests for action metadata extraction."""

    def test_get_action_metadata(self) -> None:
        """Test getting metadata for a single action."""
        game = BreakthroughGame()
        state = game.new_state()
        action = game.legal_actions(state)[0]

        metadata = game.get_action_metadata(state, action)

        assert isinstance(metadata, BreakthroughAction)
        assert metadata.action_id == action
        assert len(metadata.from_square) == 2
        assert len(metadata.to_square) == 2
        assert metadata.notation is not None

    def test_get_all_legal_actions_metadata(self) -> None:
        """Test getting metadata for all legal actions."""
        game = BreakthroughGame()
        state = game.new_state()

        all_metadata = game.get_all_legal_actions_metadata(state)

        assert len(all_metadata) == len(game.legal_actions(state))
        assert all(isinstance(m, BreakthroughAction) for m in all_metadata)

    def test_action_metadata_from_square_format(self) -> None:
        """Test that from_square has correct format."""
        game = BreakthroughGame()
        state = game.new_state()

        for metadata in game.get_all_legal_actions_metadata(state):
            # Square format: <col_letter><row_number>
            assert metadata.from_square[0] in "abcdefgh"
            assert metadata.from_square[1].isdigit()

    def test_action_metadata_to_square_format(self) -> None:
        """Test that to_square has correct format."""
        game = BreakthroughGame()
        state = game.new_state()

        for metadata in game.get_all_legal_actions_metadata(state):
            assert metadata.to_square[0] in "abcdefgh"
            assert metadata.to_square[1].isdigit()


class TestBreakthroughCaptureMoves:
    """Tests for capture move detection."""

    def test_capture_move_detection(self) -> None:
        """Test that capture moves are detected."""
        game = BreakthroughGame(columns=4, rows=4)
        state = game.new_state()

        # Play moves to set up a capture situation
        # 1. Black a4b3
        state.apply_action(game.string_to_action(state, "a4b3"))
        # 2. White a1a2
        state.apply_action(game.string_to_action(state, "a1a2"))
        # 3. Black b4a3
        state.apply_action(game.string_to_action(state, "b4a3"))

        # Now white should be able to capture at b3
        metadata_list = game.get_all_legal_actions_metadata(state)
        capture_moves = [m for m in metadata_list if m.is_capture]

        assert len(capture_moves) > 0
        for move in capture_moves:
            assert move.notation.endswith("*")


class TestBreakthroughWinCondition:
    """Tests for win condition detection."""

    def test_white_wins_by_reaching_back_row(self) -> None:
        """Test that white wins by reaching black's back row."""
        game = BreakthroughGame(columns=4, rows=4)
        state = game.new_state()

        # Play a game where white reaches row 4
        moves = [
            "a4b3",   # Black
            "a1a2",   # White
            "d4c3",   # Black
            "b1c2",   # White
            "c4d3",   # Black
            "a2b3*",  # White captures
            "b4a3",   # Black
            "c1d2",   # White
            "d3c2*",  # Black captures
            "b3a4",   # White reaches row 4 - wins!
        ]

        for move in moves:
            if game.is_terminal(state):
                break
            action = game.string_to_action(state, move)
            game.apply_action(state, action)

        assert game.is_terminal(state)
        assert game.winner(state) == 1  # White wins
        assert game.returns(state) == [-1.0, 1.0]

    def test_black_wins_by_reaching_back_row(self) -> None:
        """Test that black wins by reaching white's back row."""
        game = BreakthroughGame(columns=4, rows=4)
        state = game.new_state()

        # Play a game where black reaches row 1
        moves = [
            "a4a3",  # Black
            "a1b2",  # White
            "a3a2",  # Black
            "b1c2",  # White
            "a2a1",  # Black reaches row 1 - wins!
        ]

        for move in moves:
            if game.is_terminal(state):
                break
            action = game.string_to_action(state, move)
            game.apply_action(state, action)

        assert game.is_terminal(state)
        assert game.winner(state) == 0  # Black wins
        assert game.returns(state) == [1.0, -1.0]


class TestBreakthroughRendering:
    """Tests for board rendering methods."""

    def test_state_string(self) -> None:
        """Test basic state string rendering."""
        game = BreakthroughGame()
        state = game.new_state()
        board = game.state_string(state)

        assert "8" in board  # Row numbers
        assert "1" in board
        assert "abcdefgh" in board  # Column labels

    def test_render_compact(self) -> None:
        """Test compact rendering."""
        game = BreakthroughGame()
        state = game.new_state()
        compact = game.render_compact(state)

        assert compact == game.state_string(state)

    def test_render_with_context(self) -> None:
        """Test contextual rendering for LLM prompts."""
        game = BreakthroughGame()
        state = game.new_state()
        context = game.render_with_context(state)

        assert "Current player:" in context
        assert "Black" in context
        assert "Legal moves:" in context

    def test_render_with_context_terminal(self) -> None:
        """Test contextual rendering for terminal state."""
        game = BreakthroughGame(columns=4, rows=4)
        state = game.new_state()

        # Play to a win
        moves = ["a4b3", "a1a2", "d4c3", "b1c2", "c4d3", "a2b3*",
                 "b4a3", "c1d2", "d3c2*", "b3a4"]
        for move in moves:
            if game.is_terminal(state):
                break
            action = game.string_to_action(state, move)
            game.apply_action(state, action)

        context = game.render_with_context(state)
        assert "Game over" in context
        assert "Winner:" in context


class TestBreakthroughPlayerNames:
    """Tests for player name helpers."""

    def test_player_name_black(self) -> None:
        """Test player 0 is Black."""
        game = BreakthroughGame()
        assert game.player_name(0) == "Black"

    def test_player_name_white(self) -> None:
        """Test player 1 is White."""
        game = BreakthroughGame()
        assert game.player_name(1) == "White"


class TestBreakthroughObservation:
    """Tests for observation methods."""

    def test_observation_string(self) -> None:
        """Test observation string generation."""
        game = BreakthroughGame()
        state = game.new_state()

        obs = game.observation_string(state, 0)
        assert obs is not None
        assert len(obs) > 0

    def test_observation_same_for_both_players(self) -> None:
        """Test that both players see the same board (perfect information)."""
        game = BreakthroughGame()
        state = game.new_state()

        obs0 = game.observation_string(state, 0)
        obs1 = game.observation_string(state, 1)
        assert obs0 == obs1


class TestBreakthroughGameWrapperProtocol:
    """Tests to verify BreakthroughGame implements GameWrapper protocol."""

    def test_has_name_attribute(self) -> None:
        """Test game has name attribute."""
        game = BreakthroughGame()
        assert hasattr(game, "name")
        assert game.name == "breakthrough"

    def test_has_num_players_attribute(self) -> None:
        """Test game has num_players attribute."""
        game = BreakthroughGame()
        assert hasattr(game, "num_players")
        assert game.num_players == 2

    def test_has_new_state_method(self) -> None:
        """Test game has new_state method."""
        game = BreakthroughGame()
        assert callable(getattr(game, "new_state", None))

    def test_has_legal_actions_method(self) -> None:
        """Test game has legal_actions method."""
        game = BreakthroughGame()
        assert callable(getattr(game, "legal_actions", None))

    def test_has_apply_action_method(self) -> None:
        """Test game has apply_action method."""
        game = BreakthroughGame()
        assert callable(getattr(game, "apply_action", None))

    def test_has_is_terminal_method(self) -> None:
        """Test game has is_terminal method."""
        game = BreakthroughGame()
        assert callable(getattr(game, "is_terminal", None))

    def test_has_returns_method(self) -> None:
        """Test game has returns method."""
        game = BreakthroughGame()
        assert callable(getattr(game, "returns", None))

    def test_has_state_string_method(self) -> None:
        """Test game has state_string method."""
        game = BreakthroughGame()
        assert callable(getattr(game, "state_string", None))
