"""
arena.match — single-game executor
=====================================
Runs one game between two agents on an OpenSpiel game and returns a
``MatchResult``.

The function is intentionally side-effect-free (no file I/O); callers are
responsible for persisting the returned result.

Transcript saving
-----------------
When *save_transcript_dir* is provided, a per-move transcript JSON is written
to ``{save_transcript_dir}/{match_id}.json`` after the game completes.  Each
entry captures the board state, legal actions, agent name, chosen action,
and (for LLM agents) the prompt and response text.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
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
    save_transcript_dir: str | None = None,
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
    save_transcript_dir:
        If provided, save a per-move transcript JSON to this directory.

    Returns
    -------
    MatchResult
        Full record of the completed match, including latency metrics,
        invalid move retries, and termination reason.
    """
    # Reset agent state for a new game (clears last_prompt/last_response etc.)
    if hasattr(agent_a, "reset"):
        agent_a.reset()
    if hasattr(agent_b, "reset"):
        agent_b.reset()

    state = game.new_state()
    moves: list[int] = []

    # Track metrics
    total_invalid_retries = 0
    agent_a_total_latency_ms = 0.0
    agent_b_total_latency_ms = 0.0
    termination_reason: Literal["normal", "invalid_move_limit"] = "normal"

    # Transcript capture
    transcript_entries: list[dict] = []
    move_num = 0

    while not game.is_terminal(state):
        current_player = game.current_player(state)
        agent = agent_a if current_player == 0 else agent_b
        legal_actions = game.legal_actions(state)

        # Clear per-move prompt/response so we can read them after select_action
        if hasattr(agent, "last_prompt"):
            agent.last_prompt = None
        if hasattr(agent, "last_response"):
            agent.last_response = None

        # Track per-move latency
        retries_this_turn = 0
        action: int | None = None
        was_invalid_retry = False

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
            was_invalid_retry = True
            retries_this_turn += 1
            total_invalid_retries += 1

            # If we've exceeded retries, terminate the game
            if retries_this_turn > max_invalid_retries:
                termination_reason = "invalid_move_limit"

                # Record transcript entry for the failed move
                if save_transcript_dir is not None:
                    board_str = _safe_observation_string(game, state)
                    transcript_entries.append({
                        "move_num": move_num,
                        "player": current_player,
                        "agent_name": agent.name,
                        "action": None,
                        "board_str": board_str,
                        "legal_actions": legal_actions,
                        "llm_prompt": getattr(agent, "last_prompt", None),
                        "llm_response": getattr(agent, "last_response", None),
                        "was_invalid_retry": True,
                    })

                # Forfeit: the current player loses
                winner = agent_b.name if current_player == 0 else agent_a.name
                result = MatchResult(
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

                # Save transcript
                if save_transcript_dir is not None:
                    _save_transcript(
                        save_transcript_dir, result.match_id,
                        transcript_entries, result,
                    )

                return result

        # Apply the valid action
        if action is not None:
            # Record transcript entry before applying
            if save_transcript_dir is not None:
                board_str = _safe_observation_string(game, state)
                transcript_entries.append({
                    "move_num": move_num,
                    "player": current_player,
                    "agent_name": agent.name,
                    "action": action,
                    "board_str": board_str,
                    "legal_actions": legal_actions,
                    "llm_prompt": getattr(agent, "last_prompt", None),
                    "llm_response": getattr(agent, "last_response", None),
                    "was_invalid_retry": was_invalid_retry,
                })

            game.apply_action(state, action)
            moves.append(action)
            move_num += 1

    # Game completed normally
    returns = game.returns(state)
    if returns[0] > 0:
        winner = agent_a.name
    elif returns[0] < 0:
        winner = agent_b.name
    else:
        winner = None

    result = MatchResult(
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

    # Save transcript
    if save_transcript_dir is not None:
        _save_transcript(
            save_transcript_dir, result.match_id,
            transcript_entries, result,
        )

    return result


def _safe_observation_string(game: "GameWrapper", state: object) -> str:
    """Try to get observation_string; fall back to str(state)."""
    if hasattr(state, "observation_string"):
        try:
            return state.observation_string(player_id=0)  # type: ignore[call-arg]
        except Exception:
            pass
    if hasattr(game, "state_string"):
        return game.state_string(state)
    return str(state)


def _save_transcript(
    transcript_dir: str,
    match_id: str,
    entries: list[dict],
    result: MatchResult,
) -> None:
    """Persist a transcript JSON to disk.

    Parameters
    ----------
    transcript_dir:
        Target directory (created if it does not exist).
    match_id:
        Unique match identifier used as the filename stem.
    entries:
        Per-move transcript dictionaries.
    result:
        The final ``MatchResult`` for this game.
    """
    out_path = Path(transcript_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    payload = {
        "match_id": match_id,
        "game_name": result.game_name,
        "agent_a": result.agent_a,
        "agent_b": result.agent_b,
        "winner": result.winner,
        "num_moves": result.num_moves,
        "returns": result.returns,
        "termination_reason": result.termination_reason,
        "invalid_move_retries": result.invalid_move_retries,
        "moves": result.moves,
        "entries": entries,
    }

    with open(out_path / f"{match_id}.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
