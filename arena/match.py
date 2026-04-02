"""
arena.match — single-game executor
=====================================
Runs one game between two agents on an OpenSpiel game and returns a
``MatchResult``.

The function is intentionally side-effect-free (no file I/O); callers are
responsible for persisting the returned result.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from arena.result import MatchResult

if TYPE_CHECKING:
    from agents.base import BaseAgent
    from games.base import GameWrapper

log = logging.getLogger(__name__)


def run_match(
    agent_a: "BaseAgent",
    agent_b: "BaseAgent",
    game: "GameWrapper",
) -> MatchResult:
    """Run a single game between *agent_a* (player 0) and *agent_b* (player 1).

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
    agents = [agent_a, agent_b]
    state = game.new_state()
    moves: list[int] = []

    log.debug("Match start: %s vs %s on %s", agent_a.name, agent_b.name, game.name)

    while not game.is_terminal(state):
        current_player: int = state.current_player()
        action: int = agents[current_player].choose_action(state)
        game.apply_action(state, action)
        moves.append(action)

    rets = game.returns(state)

    if rets[0] > rets[1]:
        winner: str | None = agent_a.name
    elif rets[0] < rets[1]:
        winner = agent_b.name
    else:
        winner = None  # draw

    log.debug(
        "Match end: winner=%s, returns=%s, moves=%d",
        winner,
        rets,
        len(moves),
    )

    return MatchResult(
        game_name=game.name,
        agent_a=agent_a.name,
        agent_b=agent_b.name,
        winner=winner,
        returns=rets,
        moves=moves,
    )
