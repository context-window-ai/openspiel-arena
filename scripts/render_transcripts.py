#!/usr/bin/env python3
"""
scripts.render_transcripts — turn saved transcript JSONs into HTML + PNG artefacts
====================================================================================
Selects a "good" and a "bad" game from a transcripts directory, then produces
slide-ready HTML move-by-move transcripts and matplotlib board snapshots.

Usage::

    python scripts/render_transcripts.py --transcripts-dir transcripts/ --output-dir output/
    python scripts/render_transcripts.py --good-game <MATCH_ID> --bad-game <MATCH_ID>
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import click
import matplotlib

matplotlib.use("Agg")  # non-interactive backend for headless rendering
import matplotlib.pyplot as plt  # noqa: E402


# ---------------------------------------------------------------------------
# Transcript loading
# ---------------------------------------------------------------------------


def load_transcripts(transcripts_dir: str | Path) -> list[dict]:
    """Load all transcript JSON files from *transcripts_dir*.

    Parameters
    ----------
    transcripts_dir:
        Directory containing ``{match_id}.json`` files.

    Returns
    -------
    list[dict]
        Parsed transcript payloads sorted by filename for determinism.
    """
    tdir = Path(transcripts_dir)
    if not tdir.is_dir():
        raise FileNotFoundError(f"Transcripts directory not found: {tdir}")

    transcripts: list[dict] = []
    for fp in sorted(tdir.glob("*.json")):
        with open(fp, encoding="utf-8") as f:
            transcripts.append(json.load(f))
    return transcripts


def load_transcript(path: str | Path) -> dict:
    """Load a single transcript JSON file."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Game selection logic
# ---------------------------------------------------------------------------


def _is_llm_match(t: dict) -> bool:
    """Return True if either agent is an LLM agent (name contains 'llm')."""
    return "llm" in t.get("agent_a", "").lower() or "llm" in t.get("agent_b", "").lower()


def _llm_won(t: dict) -> bool:
    """Return True if the LLM agent won this match."""
    if not _is_llm_match(t):
        return False
    winner = t.get("winner")
    if not winner:
        return False
    return "llm" in winner.lower()


def select_good_game(transcripts: list[dict]) -> dict | None:
    """Select the best game where the LLM performed well.

    Criteria (in priority order):
    1. LLM agent won
    2. Low ``invalid_move_retries``
    3. Reasonable number of moves (5–15)

    Falls back to non-LLM games with clean play if no LLM win is found.
    """
    # Prefer LLM wins
    llm_wins = [t for t in transcripts if _llm_won(t)]
    if llm_wins:
        return sorted(llm_wins, key=lambda t: (t["invalid_move_retries"], t["num_moves"]))[0]

    # Fallback: any game with 0 retries and reasonable length
    candidates = [
        t for t in transcripts
        if t.get("invalid_move_retries", 0) == 0
        and 5 <= t.get("num_moves", 0) <= 15
    ]
    if candidates:
        return candidates[len(candidates) // 2]  # pick a middle one

    # Last resort: just pick the first transcript
    return transcripts[0] if transcripts else None


def select_bad_game(transcripts: list[dict]) -> dict | None:
    """Select a game where the LLM performed poorly.

    Criteria:
    1. LLM agent lost OR high ``invalid_move_retries``
    2. Highest retries first
    """
    # Prefer LLM losses with retries
    llm_losses = [t for t in transcripts if _is_llm_match(t) and not _llm_won(t)]
    if llm_losses:
        return sorted(llm_losses, key=lambda t: -t.get("invalid_move_retries", 0))[0]

    # Fallback: highest retries
    if transcripts:
        return sorted(transcripts, key=lambda t: -t.get("invalid_move_retries", 0))[0]

    return None


# ---------------------------------------------------------------------------
# Board rendering (matplotlib)
# ---------------------------------------------------------------------------


def render_tic_tac_toe_board(
    transcript: dict,
    output_path: str | Path,
    move_index: int | None = None,
    figsize: tuple[float, float] = (19.2, 10.8),
    dpi: int = 100,
) -> Path:
    """Render a tic-tac-toe board snapshot as a PNG.

    Parameters
    ----------
    transcript:
        Loaded transcript dict with ``entries`` and ``moves``.
    output_path:
        Destination file path for the PNG.
    move_index:
        Which move to render (None = final state).
    figsize:
        Figure size in inches (default 1920×1080 @ 100 DPI).
    dpi:
        Resolution.

    Returns
    -------
    Path
        The path the image was written to.
    """
    entries = transcript.get("entries", [])
    if not entries:
        # Render an empty board
        board = [""] * 9
    else:
        # Replay up to move_index (or all moves)
        idx = move_index if move_index is not None else len(entries) - 1
        idx = min(idx, len(entries) - 1)

        # Parse the board from the last entry's board_str
        # We replay the moves to construct the board state
        board = _replay_tic_tac_toe(entries, idx)

    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-0.5, 2.5)
    ax.set_aspect("equal")
    ax.axis("off")

    # Title
    agent_a = transcript.get("agent_a", "Player 0")
    agent_b = transcript.get("agent_b", "Player 1")
    winner = transcript.get("winner")
    title = f"{agent_a} (X) vs {agent_b} (O)"
    if winner:
        title += f"  —  Winner: {winner}"
    elif transcript.get("num_moves", 0) >= 9:
        title += "  —  Draw"
    ax.set_title(title, fontsize=28, fontweight="bold", pad=20)

    # Grid lines
    for i in range(1, 3):
        ax.axhline(y=i - 0.5, color="black", linewidth=3)
        ax.axvline(x=i - 0.5, color="black", linewidth=3)

    # Pieces
    symbols = {0: "X", 1: "O"}
    colors = {0: "#2196F3", 1: "#F44336"}  # blue / red

    for pos, piece in enumerate(board):
        if piece == "":
            continue
        row, col = divmod(pos, 3)
        player_id = int(piece)
        ax.text(
            col, 2 - row,
            symbols.get(player_id, "?"),
            ha="center", va="center",
            fontsize=80, fontweight="bold",
            color=colors.get(player_id, "black"),
        )

    # Position labels
    for pos in range(9):
        row, col = divmod(pos, 3)
        if board[pos] == "":
            ax.text(
                col, 2 - row,
                str(pos),
                ha="center", va="center",
                fontsize=24, color="#CCCCCC",
            )

    plt.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    return out


def _replay_tic_tac_toe(entries: list[dict], up_to: int) -> list[str]:
    """Replay tic-tac-toe moves to reconstruct the board at index *up_to*.

    Returns a list of 9 strings: "" for empty, "0" for X (player 0), "1" for O (player 1).
    """
    board = [""] * 9
    for i in range(up_to + 1):
        entry = entries[i]
        action = entry.get("action")
        if action is None:
            continue
        player = entry.get("player", i % 2)
        if 0 <= action < 9:
            board[action] = str(player)
    return board


def render_board_png(
    transcript: dict,
    output_path: str | Path,
    move_index: int | None = None,
) -> Path:
    """Auto-detect game type and render the board PNG.

    Currently supports ``tic_tac_toe``.
    """
    game_name = transcript.get("game_name", "")
    if game_name == "tic_tac_toe":
        return render_tic_tac_toe_board(transcript, output_path, move_index)
    else:
        # Generic fallback: render a text-based info card
        return _render_generic_card(transcript, output_path)


def _render_generic_card(transcript: dict, output_path: str | Path) -> Path:
    """Render a generic info card for unsupported games."""
    fig, ax = plt.subplots(figsize=(19.2, 10.8), dpi=100)
    ax.axis("off")
    lines = [
        f"Game: {transcript.get('game_name', '?')}",
        f"{transcript.get('agent_a', '?')} vs {transcript.get('agent_b', '?')}",
        f"Winner: {transcript.get('winner', 'Draw')}",
        f"Moves: {transcript.get('num_moves', 0)}",
    ]
    ax.text(0.5, 0.5, "\n".join(lines), ha="center", va="center",
            fontsize=36, fontfamily="monospace", transform=ax.transAxes)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=100)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------


def render_transcript_html(transcript: dict, output_path: str | Path, label: str = "") -> Path:
    """Render a move-by-move transcript as a standalone HTML file.

    The HTML uses inline styles so it looks good without an external stylesheet.

    Parameters
    ----------
    transcript:
        Loaded transcript dict.
    output_path:
        Destination HTML file path.
    label:
        Optional label like "Good Game" or "Bad Game" for the heading.

    Returns
    -------
    Path
        Path the HTML was written to.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    entries = transcript.get("entries", [])
    winner = transcript.get("winner", "Draw")
    game_name = transcript.get("game_name", "unknown")
    agent_a = transcript.get("agent_a", "?")
    agent_b = transcript.get("agent_b", "?")
    num_moves = transcript.get("num_moves", 0)
    retries = transcript.get("invalid_move_retries", 0)
    termination = transcript.get("termination_reason", "normal")

    # Build HTML
    heading = f" ({label})" if label else ""
    rows_html = ""
    for entry in entries:
        move_num = entry.get("move_num", "?")
        player = entry.get("player", "?")
        agent_name = entry.get("agent_name", "?")
        action = entry.get("action", "N/A")
        board_str = entry.get("board_str", "")
        legal_actions = entry.get("legal_actions", [])
        llm_prompt = entry.get("llm_prompt")
        llm_response = entry.get("llm_response")
        was_invalid = entry.get("was_invalid_retry", False)

        invalid_badge = ""
        if was_invalid:
            invalid_badge = (
                '<span style="background:#F44336;color:white;padding:2px 8px;'
                'border-radius:4px;font-size:12px;margin-left:8px;">'
                "INVALID RETRY</span>"
            )

        prompt_section = ""
        if llm_prompt:
            prompt_section = (
                '<div style="margin-top:8px;">'
                '<span style="font-weight:bold;color:#1565C0;">LLM Prompt:</span>'
                f'<pre style="background:#F5F5F5;padding:8px;border-radius:4px;'
                f'font-size:13px;overflow-x:auto;white-space:pre-wrap;">'
                f"{escape_html(llm_prompt)}</pre></div>"
            )

        response_section = ""
        if llm_response:
            response_section = (
                '<div style="margin-top:4px;">'
                '<span style="font-weight:bold;color:#C62828;">LLM Response:</span>'
                f'<pre style="background:#FFF3E0;padding:8px;border-radius:4px;'
                f'font-size:13px;overflow-x:auto;white-space:pre-wrap;">'
                f"{escape_html(llm_response)}</pre></div>"
            )

        board_html = ""
        if board_str:
            board_html = (
                '<pre style="font-size:16px;letter-spacing:4px;'
                'background:#E8F5E9;padding:8px;border-radius:4px;'
                'display:inline-block;">'
                f"{escape_html(board_str)}</pre>"
            )

        rows_html += f"""
        <tr style="border-bottom:1px solid #E0E0E0;">
            <td style="padding:8px;text-align:center;font-weight:bold;">{move_num}</td>
            <td style="padding:8px;text-align:center;">Player {player}</td>
            <td style="padding:8px;">{escape_html(agent_name)}</td>
            <td style="padding:8px;text-align:center;font-weight:bold;">{action}{invalid_badge}</td>
            <td style="padding:8px;">{board_html}</td>
            <td style="padding:8px;font-size:12px;">{legal_actions}</td>
            <td style="padding:8px;">
                {prompt_section}
                {response_section}
            </td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Game Transcript: {game_name} — {agent_a} vs {agent_b}{heading}</title>
</head>
<body style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
             max-width:1400px;margin:0 auto;padding:24px;background:#FAFAFA;color:#212121;">

<h1 style="text-align:center;color:#1565C0;">
    Game Transcript{heading}: {escape_html(game_name)}
</h1>
<p style="text-align:center;font-size:18px;">
    <strong>{escape_html(agent_a)}</strong> (Player 0)
    vs
    <strong>{escape_html(agent_b)}</strong> (Player 1)
</p>

<table style="width:100%;margin:16px auto;border-collapse:collapse;background:white;
              border-radius:8px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,0.1);">
<tr style="background:#1565C0;color:white;">
    <th style="padding:10px;">Move</th>
    <th style="padding:10px;">Player</th>
    <th style="padding:10px;">Agent</th>
    <th style="padding:10px;">Action</th>
    <th style="padding:10px;">Board</th>
    <th style="padding:10px;">Legal Actions</th>
    <th style="padding:10px;">LLM Details</th>
</tr>
{rows_html}
</table>

<div style="margin-top:24px;padding:16px;background:white;border-radius:8px;
            box-shadow:0 2px 8px rgba(0,0,0,0.1);">
<h2 style="color:#1565C0;">Game Summary</h2>
<table style="width:100%;font-size:16px;">
<tr><td style="padding:4px 8px;font-weight:bold;">Winner:</td>
    <td style="padding:4px 8px;">{escape_html(str(winner))}</td></tr>
<tr><td style="padding:4px 8px;font-weight:bold;">Total Moves:</td>
    <td style="padding:4px 8px;">{num_moves}</td></tr>
<tr><td style="padding:4px 8px;font-weight:bold;">Invalid Move Retries:</td>
    <td style="padding:4px 8px;">{retries}</td></tr>
<tr><td style="padding:4px 8px;font-weight:bold;">Termination:</td>
    <td style="padding:4px 8px;">{escape_html(termination)}</td></tr>
<tr><td style="padding:4px 8px;font-weight:bold;">Returns:</td>
    <td style="padding:4px 8px;">{transcript.get('returns', [])}</td></tr>
</table>
</div>

</body>
</html>"""

    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    return out


def escape_html(text: str) -> str:
    """Escape HTML special characters."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command()
@click.option(
    "--transcripts-dir",
    default="transcripts/",
    show_default=True,
    help="Directory containing transcript JSON files.",
)
@click.option(
    "--output-dir",
    default="output/",
    show_default=True,
    help="Directory for output HTML and PNG files.",
)
@click.option(
    "--good-game",
    default=None,
    help="Specific match_id for the good game (auto-selected if omitted).",
)
@click.option(
    "--bad-game",
    default=None,
    help="Specific match_id for the bad game (auto-selected if omitted).",
)
def main(
    transcripts_dir: str,
    output_dir: str,
    good_game: str | None,
    bad_game: str | None,
) -> None:
    """Render HTML + PNG transcripts from saved game transcripts."""
    transcripts = load_transcripts(transcripts_dir)
    if not transcripts:
        click.echo("No transcripts found.", err=True)
        sys.exit(1)

    click.echo(f"Loaded {len(transcripts)} transcripts from {transcripts_dir}")

    # Resolve good game
    if good_game:
        good_t = _find_by_match_id(transcripts, good_game)
        if not good_t:
            click.echo(f"Good game match_id not found: {good_game}", err=True)
            sys.exit(1)
    else:
        good_t = select_good_game(transcripts)

    # Resolve bad game
    if bad_game:
        bad_t = _find_by_match_id(transcripts, bad_game)
        if not bad_t:
            click.echo(f"Bad game match_id not found: {bad_game}", err=True)
            sys.exit(1)
    else:
        bad_t = select_bad_game(transcripts)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if good_t:
        mid = good_t["match_id"]
        click.echo(f"Rendering good game: {mid}")
        render_board_png(good_t, out / "example_good_llm_game.png")
        render_transcript_html(good_t, out / "example_good_llm_game.html", label="Good Game")
        click.echo(f"  → output/example_good_llm_game.png")
        click.echo(f"  → output/example_good_llm_game.html")

    if bad_t:
        mid = bad_t["match_id"]
        click.echo(f"Rendering bad game: {mid}")
        render_board_png(bad_t, out / "example_bad_llm_game.png")
        render_transcript_html(bad_t, out / "example_bad_llm_game.html", label="Bad Game")
        click.echo(f"  → output/example_bad_llm_game.png")
        click.echo(f"  → output/example_bad_llm_game.html")

    click.echo("Done.")


def _find_by_match_id(transcripts: list[dict], match_id: str) -> dict | None:
    """Find a transcript by its match_id."""
    for t in transcripts:
        if t.get("match_id") == match_id:
            return t
    return None


if __name__ == "__main__":
    main()
