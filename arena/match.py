"""
arena.match — single-game executor
=====================================
Runs one game between two agents on an OpenSpiel game and returns a
``MatchResult``.

The function is intentionally side-effect-free (no file I/O); callers are
responsible for persisting the returned result.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Literal

from arena.result import MatchResult

if TYPE_CHECKING:
    from agents.base import BaseAgent
    from games.base import GameWrapper


# Default maximum retries for invalid moves before terminating
DEFAULT_MAX_INVALID_RETRIES = 3


def run_match(
    agent_a: "BaseAgent",
    agent_b: "BaseAgent",
    game: "GameWrapper",
    *,
    seed: int | None = None,
    max_invalid_retries: int = DEFAULT_MAX_INVALID_RETRIES,
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
    seed:
        Optional seed for reproducibility (recorded in result).
    max_invalid_retries:
        Maximum number of retries per turn when an agent returns an
        invalid action (default: 3).

    Returns
    -------
    MatchResult
        Full record of the completed match, including latency metrics,
        invalid move retries, and termination reason.
    """
    state = game.new_state()
    moves: list[int] = []

    # Track metrics
    total_invalid_retries = 0
    agent_a_total_latency_ms = 0.0
    agent_b_total_latency_ms = 0.0
    termination_reason: Literal["normal", "invalid_move_limit"] = "normal"

    while not game.is_terminal(state):
        current_player = game.current_player(state)
        agent = agent_a if current_player == 0 else agent_b
        legal_actions = game.legal_actions(state)

        # Track per-move latency
        retries_this_turn = 0
        action: int | None = None

        while retries_this_turn <= max_invalid_retries:
            start_time = time.perf_counter()
            try:
                # Use the agent's select_action method
                action = agent.select_action(state, legal_actions)
            except Exception:
                # If agent throws an exception, treat as invalid move
                action = None

            elapsed_ms = (time.perf_counter() - start_time) * 1000

            # Update latency for the appropriate agent
            if current_player == 0:
                agent_a_total_latency_ms += elapsed_ms
            else:
                agent_b_total_latency_ms += elapsed_ms

            # Check if action is valid
            if action is not None and action in legal_actions:
                break

            # Invalid move - retry
            retries_this_turn += 1
            total_invalid_retries += 1

            # If we've exceeded retries, terminate the game
            if retries_this_turn > max_invalid_retries:
                termination_reason = "invalid_move_limit"
                # Forfeit: the current player loses
                # Create a result with the opponent as winner
                winner = agent_b.name if current_player == 0 else agent_a.name
                return MatchResult(
                    agent_a=agent_a.name,
                    agent_b=agent_b.name,
                    game_name=game.name,
                    winner=winner,
                    num_moves=len(moves),
                    returns=[-1.0 if current_player == 0 else 1.0, 1.0 if current_player == 0 else -1.0],
                    moves=moves,
                    seed=seed,
                    agent_a_side=0,
                    agent_b_side=1,
                    invalid_move_retries=total_invalid_retries,
                    agent_a_latency_ms=agent_a_total_latency_ms,
                    agent_b_latency_ms=agent_b_total_latency_ms,
                    termination_reason=termination_reason,
                )

        # Apply the valid action
        if action is not None:
            game.apply_action(state, action)
            moves.append(action)

    # Game completed normally
    returns = game.returns(state)
    if returns[0] > 0:
        winner = agent_a.name
    elif returns[0] < 0:
        winner = agent_b.name
    else:
        winner = None

    return MatchResult(
        agent_a=agent_a.name,
        agent_b=agent_b.name,
        game_name=game.name,
        winner=winner,
        num_moves=len(moves),
        returns=list(returns),
        moves=moves,
        seed=seed,
        agent_a_side=0,
        agent_b_side=1,
        invalid_move_retries=total_invalid_retries,
        agent_a_latency_ms=agent_a_total_latency_ms,
        agent_b_latency_ms=agent_b_total_latency_ms,
        termination_reason=termination_reason,
    )
