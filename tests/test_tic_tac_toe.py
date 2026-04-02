"""
Integration tests for games.tic_tac_toe — TicTacToeGame.

These tests require OpenSpiel to be installed (``open_spiel``).
"""

from __future__ import annotations

import pytest

pytest.importorskip("pyspiel", reason="open_spiel not installed")

from games.tic_tac_toe import TicTacToeGame  # noqa: E402


@pytest.fixture()
def game() -> TicTacToeGame:
    return TicTacToeGame()


def test_game_name(game):
    assert game.name == "tic_tac_toe"


def test_num_players(game):
    assert game.num_players == 2


def test_new_state_not_terminal(game):
    state = game.new_state()
    assert not game.is_terminal(state)


def test_legal_actions_nonempty_at_start(game):
    state = game.new_state()
    actions = game.legal_actions(state)
    assert len(actions) == 9  # all cells free


def test_apply_action_reduces_legal_actions(game):
    state = game.new_state()
    action = game.legal_actions(state)[0]
    game.apply_action(state, action)
    assert len(game.legal_actions(state)) == 8


def test_full_game_terminates(game):
    """Play a complete game; verify it reaches a terminal state."""
    from agents.random_agent import RandomAgent

    agents = [RandomAgent(seed=0), RandomAgent(seed=1)]
    state = game.new_state()
    while not game.is_terminal(state):
        cp = state.current_player()
        action = agents[cp].choose_action(state)
        game.apply_action(state, action)

    returns = game.returns(state)
    assert len(returns) == 2
    # Tic-tac-toe returns are in {-1, 0, 1}
    assert all(r in (-1.0, 0.0, 1.0) for r in returns)


def test_state_string_nonempty(game):
    state = game.new_state()
    s = game.state_string(state)
    assert isinstance(s, str) and len(s) > 0
