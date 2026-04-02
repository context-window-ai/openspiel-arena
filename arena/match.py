"""
arena.match — single-game executor
=====================================
Runs one game between two agents on an OpenSpiel game and returns a
``MatchResult``.

The function is intentionally side-effect-free (no file I/O); callers are
responsible for persisting the returned result.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from arena.result import MatchResult

if TYPE_CHECKING:
    from agents.base import BaseAgent
    from games.base import GameWrapper


def run_match(
    agent_a: "BaseAgent",
    agent_b: "BaseAgent",
    game: "GameWrapper",
) -> MatchResult:
    """Run a single game and return the outcome.

    Parameters
    ----------
    agent_a:
        The agent playing as player 0.
    agent_b:
        The agent playing as player 1.
    game:
        A ``GameWrapper``-compatible game object.

    Returns
    -------
    MatchResult
        Full record of the completed match.
    """
    state = game.new_state()
    moves: list[int] = []

    while not game.is_terminal(state):
        current_player = game.current_player(state)
        agent = agent_a if current_player == 0 else agent_b
        action = agent.choose_action(state)
        game.apply_action(state, action)
        moves.append(action)

    returns = game.returns(state)
    if returns[0] > 0:
        outcome = "win"
    elif returns[0] < 0:
        outcome = "loss"
    else:
        outcome = "draw"

    return MatchResult(
        agent_a=agent_a.name,
        agent_b=agent_b.name,
        game_name=game.name,
        outcome=outcome,
        returns=list(returns),
        moves=moves,
    )
