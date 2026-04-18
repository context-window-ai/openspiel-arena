"""
tests.test_transcript_renderer — tests for the transcript rendering pipeline
=============================================================================
Covers transcript JSON format validation, game selection logic, board rendering,
and HTML output.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# We need matplotlib for board rendering tests
# ---------------------------------------------------------------------------
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from scripts.render_transcripts import (  # noqa: E402
    _is_llm_match,
    _llm_won,
    escape_html,
    load_transcripts,
    render_board_png,
    render_tic_tac_toe_board,
    render_transcript_html,
    select_bad_game,
    select_good_game,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_transcript_good() -> dict:
    """A 'good' transcript where an LLM agent wins cleanly."""
    return {
        "match_id": "good-001",
        "game_name": "tic_tac_toe",
        "agent_a": "llm-gpt-mini",
        "agent_b": "random",
        "winner": "llm-gpt-mini",
        "num_moves": 7,
        "returns": [1.0, -1.0],
        "termination_reason": "normal",
        "invalid_move_retries": 0,
        "moves": [4, 0, 1, 5, 2, 3, 7],
        "entries": [
            {"move_num": 0, "player": 0, "agent_name": "llm-gpt-mini", "action": 4,
             "board_str": "...\n...\n...", "legal_actions": [0,1,2,3,4,5,6,7,8],
             "llm_prompt": "You are playing tic-tac-toe...", "llm_response": "I choose action 4 (center)",
             "was_invalid_retry": False},
            {"move_num": 1, "player": 1, "agent_name": "random", "action": 0,
             "board_str": "O..\n...\n.X.", "legal_actions": [0,1,2,3,5,6,7,8],
             "llm_prompt": None, "llm_response": None, "was_invalid_retry": False},
            {"move_num": 2, "player": 0, "agent_name": "llm-gpt-mini", "action": 1,
             "board_str": "O..\n...\n.X.", "legal_actions": [1,2,3,5,6,7,8],
             "llm_prompt": "Board state:\nO..\n...\n.X.\nChoose action.", "llm_response": "Action 1",
             "was_invalid_retry": False},
        ],
    }


@pytest.fixture()
def sample_transcript_bad() -> dict:
    """A 'bad' transcript where the LLM agent lost with many retries."""
    return {
        "match_id": "bad-001",
        "game_name": "tic_tac_toe",
        "agent_a": "llm-gpt-mini",
        "agent_b": "mcts-50",
        "winner": "mcts-50",
        "num_moves": 1,
        "returns": [-1.0, 1.0],
        "termination_reason": "invalid_move_limit",
        "invalid_move_retries": 4,
        "moves": [8],
        "entries": [
            {"move_num": 0, "player": 0, "agent_name": "llm-gpt-mini", "action": None,
             "board_str": "...\n...\n...", "legal_actions": [0,1,2,3,4,5,6,7,8],
             "llm_prompt": "Choose an action...", "llm_response": "Let me place at position 99",
             "was_invalid_retry": True},
        ],
    }


@pytest.fixture()
def transcripts_dir(tmp_path: Path, sample_transcript_good: dict, sample_transcript_bad: dict) -> Path:
    """Write sample transcripts to a temp dir."""
    for t in [sample_transcript_good, sample_transcript_bad]:
        (tmp_path / f"{t['match_id']}.json").write_text(json.dumps(t), encoding="utf-8")
    return tmp_path


# ---------------------------------------------------------------------------
# Test 1: Transcript JSON format validation
# ---------------------------------------------------------------------------

class TestTranscriptFormat:
    """Validate the structure of transcript JSON payloads."""

    REQUIRED_KEYS = [
        "match_id", "game_name", "agent_a", "agent_b",
        "winner", "num_moves", "entries",
    ]

    ENTRY_KEYS = [
        "move_num", "player", "agent_name", "action",
        "board_str", "legal_actions", "was_invalid_retry",
    ]

    def test_transcript_has_required_keys(self, sample_transcript_good: dict):
        for key in self.REQUIRED_KEYS:
            assert key in sample_transcript_good, f"Missing key: {key}"

    def test_entries_are_list_of_dicts(self, sample_transcript_good: dict):
        entries = sample_transcript_good["entries"]
        assert isinstance(entries, list)
        assert len(entries) > 0
        for entry in entries:
            assert isinstance(entry, dict)

    def test_each_entry_has_required_keys(self, sample_transcript_good: dict):
        for entry in sample_transcript_good["entries"]:
            for key in self.ENTRY_KEYS:
                assert key in entry, f"Entry missing key: {key}"

    def test_moves_list_matches_num_moves(self, sample_transcript_good: dict):
        assert len(sample_transcript_good["moves"]) == sample_transcript_good["num_moves"]

    def test_llm_fields_present(self, sample_transcript_good: dict):
        """LLM entries should have llm_prompt and llm_response keys (can be None)."""
        for entry in sample_transcript_good["entries"]:
            assert "llm_prompt" in entry
            assert "llm_response" in entry


# ---------------------------------------------------------------------------
# Test 2: Game selection logic
# ---------------------------------------------------------------------------

class TestGameSelection:
    """Test good/bad game auto-selection criteria."""

    def test_select_good_game_prefers_llm_win(self, sample_transcript_good: dict, sample_transcript_bad: dict):
        transcripts = [sample_transcript_bad, sample_transcript_good]
        result = select_good_game(transcripts)
        assert result is not None
        assert result["match_id"] == "good-001"

    def test_select_bad_game_prefers_llm_loss(self, sample_transcript_good: dict, sample_transcript_bad: dict):
        transcripts = [sample_transcript_good, sample_transcript_bad]
        result = select_bad_game(transcripts)
        assert result is not None
        assert result["match_id"] == "bad-001"

    def test_select_bad_game_sorts_by_retries(self):
        t1 = {"match_id": "a", "agent_a": "llm", "agent_b": "x", "winner": "x",
               "invalid_move_retries": 2, "num_moves": 5, "entries": []}
        t2 = {"match_id": "b", "agent_a": "llm", "agent_b": "x", "winner": "x",
               "invalid_move_retries": 10, "num_moves": 0, "entries": []}
        result = select_bad_game([t1, t2])
        assert result["match_id"] == "b"

    def test_select_good_game_fallback_no_llm(self):
        t = {
            "match_id": "fallback-1", "game_name": "tic_tac_toe",
            "agent_a": "random", "agent_b": "mcts",
            "winner": "mcts", "num_moves": 7,
            "invalid_move_retries": 0, "entries": [],
        }
        result = select_good_game([t])
        assert result["match_id"] == "fallback-1"

    def test_is_llm_match(self, sample_transcript_good: dict, sample_transcript_bad: dict):
        assert _is_llm_match(sample_transcript_good) is True
        non_llm = {"agent_a": "random", "agent_b": "mcts"}
        assert _is_llm_match(non_llm) is False

    def test_llm_won(self, sample_transcript_good: dict, sample_transcript_bad: dict):
        assert _llm_won(sample_transcript_good) is True
        assert _llm_won(sample_transcript_bad) is False


# ---------------------------------------------------------------------------
# Test 3: Board rendering produces valid matplotlib figure
# ---------------------------------------------------------------------------

class TestBoardRendering:
    """Test that board PNGs are produced correctly."""

    def test_render_tic_tac_toe_produces_png(self, sample_transcript_good: dict, tmp_path: Path):
        out = tmp_path / "board.png"
        result = render_tic_tac_toe_board(sample_transcript_good, out)
        assert result.exists()
        assert result.stat().st_size > 0
        # PNG files start with the signature bytes 0x89504e47
        data = result.read_bytes()
        assert data[:4] == b"\x89PNG"

    def test_render_board_png_auto_detects_ttt(self, sample_transcript_good: dict, tmp_path: Path):
        out = tmp_path / "auto_board.png"
        result = render_board_png(sample_transcript_good, out)
        assert result.exists()
        assert result.stat().st_size > 0

    def test_render_at_specific_move_index(self, sample_transcript_good: dict, tmp_path: Path):
        out = tmp_path / "move0.png"
        render_tic_tac_toe_board(sample_transcript_good, out, move_index=0)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_render_empty_board(self, tmp_path: Path):
        """A transcript with no entries should still produce a valid PNG."""
        t = {
            "match_id": "empty", "game_name": "tic_tac_toe",
            "agent_a": "a", "agent_b": "b", "winner": None,
            "num_moves": 0, "entries": [],
        }
        out = tmp_path / "empty.png"
        render_tic_tac_toe_board(t, out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_render_generic_game(self, tmp_path: Path):
        """Non-tic-tac-toe games should produce a generic card PNG."""
        t = {
            "match_id": "gen", "game_name": "breakthrough",
            "agent_a": "a", "agent_b": "b", "winner": "a",
            "num_moves": 10, "entries": [],
        }
        out = tmp_path / "generic.png"
        result = render_board_png(t, out)
        assert result.exists()


# ---------------------------------------------------------------------------
# Test 4: HTML output contains expected sections
# ---------------------------------------------------------------------------

class TestHTMLOutput:
    """Validate the structure and content of rendered HTML."""

    def test_html_contains_game_title(self, sample_transcript_good: dict, tmp_path: Path):
        out = tmp_path / "transcript.html"
        render_transcript_html(sample_transcript_good, out, label="Good Game")
        html = out.read_text(encoding="utf-8")
        assert "tic_tac_toe" in html
        assert "Good Game" in html

    def test_html_contains_agent_names(self, sample_transcript_good: dict, tmp_path: Path):
        out = tmp_path / "transcript.html"
        render_transcript_html(sample_transcript_good, out)
        html = out.read_text(encoding="utf-8")
        assert "llm-gpt-mini" in html
        assert "random" in html

    def test_html_contains_move_table(self, sample_transcript_good: dict, tmp_path: Path):
        out = tmp_path / "transcript.html"
        render_transcript_html(sample_transcript_good, out)
        html = out.read_text(encoding="utf-8")
        assert "<table" in html
        assert "Move" in html
        assert "Action" in html
        assert "Agent" in html

    def test_html_contains_llm_prompt_and_response(self, sample_transcript_good: dict, tmp_path: Path):
        out = tmp_path / "transcript.html"
        render_transcript_html(sample_transcript_good, out)
        html = out.read_text(encoding="utf-8")
        assert "LLM Prompt" in html
        assert "LLM Response" in html
        assert "tic-tac-toe" in html

    def test_html_contains_summary_section(self, sample_transcript_good: dict, tmp_path: Path):
        out = tmp_path / "transcript.html"
        render_transcript_html(sample_transcript_good, out)
        html = out.read_text(encoding="utf-8")
        assert "Game Summary" in html
        assert "Winner" in html

    def test_html_uses_inline_styles(self, sample_transcript_good: dict, tmp_path: Path):
        out = tmp_path / "transcript.html"
        render_transcript_html(sample_transcript_good, out)
        html = out.read_text(encoding="utf-8")
        assert 'style="' in html  # inline styles present
        assert '<link rel="stylesheet"' not in html  # no external stylesheets

    def test_html_escapes_special_chars(self):
        result = escape_html('<script>alert("xss")</script>')
        assert "<script>" not in result
        assert "&lt;script&gt;" in result


# ---------------------------------------------------------------------------
# Test 5: Loading transcripts from disk
# ---------------------------------------------------------------------------

class TestLoadTranscripts:
    """Test transcript loading from a directory."""

    def test_load_from_dir(self, transcripts_dir: Path):
        transcripts = load_transcripts(transcripts_dir)
        assert len(transcripts) == 2

    def test_load_missing_dir_raises(self):
        with pytest.raises(FileNotFoundError):
            load_transcripts("/nonexistent/path")

    def test_load_single_transcript(self, transcripts_dir: Path):
        from scripts.render_transcripts import load_transcript
        t = load_transcript(transcripts_dir / "good-001.json")
        assert t["match_id"] == "good-001"
