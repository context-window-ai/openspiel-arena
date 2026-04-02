"""
Tests for arena.logging — CSV results logging.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from arena.logging import (
    CANONICAL_COLUMNS,
    ResultsWriter,
    match_result_to_row,
    load_results_csv,
)
from arena.result import MatchResult


CSV_COLUMNS = CANONICAL_COLUMNS  # Alias for backward compatibility in tests


class TestMatchResultToRow:
    """Tests for match_result_to_row function."""

    def test_basic_conversion(self):
        """Test basic conversion of MatchResult to row dict."""
        result = MatchResult(
            game_name="tic_tac_toe",
            agent_a="alice",
            agent_b="bob",
            winner="alice",
            num_moves=7,
            seed=42,
            agent_a_side=0,
            agent_b_side=1,
            invalid_move_retries=0,
            agent_a_latency_ms=123.45,
            agent_b_latency_ms=67.89,
            termination_reason="normal",
        )
        row = match_result_to_row(result)

        assert row["game"] == "tic_tac_toe"
        assert row["agent_a"] == "alice"
        assert row["agent_b"] == "bob"
        assert row["winner"] == "alice"
        assert row["is_draw"] is False
        assert row["num_moves"] == 7
        assert row["seed"] == 42
        assert row["agent_a_side"] == 0
        assert row["agent_b_side"] == 1
        assert row["invalid_move_retries"] == 0
        assert row["agent_a_latency_ms"] == 123.45
        assert row["agent_b_latency_ms"] == 67.89
        assert row["termination_reason"] == "normal"

    def test_draw_result(self):
        """Test conversion of a draw result."""
        result = MatchResult(
            game_name="tic_tac_toe",
            agent_a="alice",
            agent_b="bob",
            winner=None,
            num_moves=9,
        )
        row = match_result_to_row(result)

        assert row["winner"] == ""
        assert row["is_draw"] is True

    def test_null_seed(self):
        """Test handling of null seed."""
        result = MatchResult(
            game_name="tic_tac_toe",
            agent_a="alice",
            agent_b="bob",
            winner="alice",
            seed=None,
        )
        row = match_result_to_row(result)

        assert row["seed"] == ""

    def test_null_latency(self):
        """Test handling of null latency values."""
        result = MatchResult(
            game_name="tic_tac_toe",
            agent_a="alice",
            agent_b="bob",
            winner="alice",
            agent_a_latency_ms=None,
            agent_b_latency_ms=None,
        )
        row = match_result_to_row(result)

        assert row["agent_a_latency_ms"] == 0.0
        assert row["agent_b_latency_ms"] == 0.0

    def test_all_columns_present(self):
        """Test that all expected columns are present."""
        result = MatchResult(
            game_name="tic_tac_toe",
            agent_a="alice",
            agent_b="bob",
            winner="alice",
        )
        row = match_result_to_row(result)

        for col in CSV_COLUMNS:
            assert col in row, f"Missing column: {col}"


class TestResultsWriter:
    """Tests for ResultsWriter class."""

    def test_creates_file_with_headers(self, tmp_path: Path):
        """Test that writer creates a file with correct headers."""
        filepath = tmp_path / "results.csv"
        writer = ResultsWriter(filepath)

        result = MatchResult(
            game_name="tic_tac_toe",
            agent_a="alice",
            agent_b="bob",
            winner="alice",
        )
        writer.write(result)

        writer.close()

        assert filepath.exists()

        with open(filepath, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            assert headers is not None
            assert set(headers) == set(CSV_COLUMNS)

    def test_logs_single_result(self, tmp_path: Path):
        """Test logging a single result."""
        filepath = tmp_path / "results.csv"
        writer = ResultsWriter(filepath)

        result = MatchResult(
            game_name="tic_tac_toe",
            agent_a="alice",
            agent_b="bob",
            winner="alice",
            num_moves=5,
            seed=42,
        )
        writer.write(result)
        writer.close()

        rows = load_results_csv(filepath)
        assert len(rows) == 1
        assert rows[0]["game"] == "tic_tac_toe"
        assert rows[0]["agent_a"] == "alice"
        assert rows[0]["agent_b"] == "bob"
        assert rows[0]["winner"] == "alice"
        assert rows[0]["num_moves"] == "5"
        assert rows[0]["seed"] == "42"

    def test_logs_multiple_results(self, tmp_path: Path):
        """Test logging multiple results."""
        filepath = tmp_path / "results.csv"
        writer = ResultsWriter(filepath)

        results = [
            MatchResult(game_name="tic_tac_toe", agent_a="a", agent_b="b", winner="a"),
            MatchResult(game_name="tic_tac_toe", agent_a="a", agent_b="b", winner="b"),
            MatchResult(game_name="tic_tac_toe", agent_a="a", agent_b="b", winner=None),
        ]
        writer.write_many(results)
        writer.close()

        rows = load_results_csv(filepath)
        assert len(rows) == 3
        assert rows[0]["winner"] == "a"
        assert rows[1]["winner"] == "b"
        assert rows[2]["winner"] == ""

    def test_appends_to_existing_file(self, tmp_path: Path):
        """Test that writer appends to existing file."""
        filepath = tmp_path / "results.csv"

        # First writer writes one result
        writer1 = ResultsWriter(filepath)
        result1 = MatchResult(game_name="tic_tac_toe", agent_a="a", agent_b="b", winner="a")
        writer1.write(result1)
        writer1.close()

        # Second writer appends another result
        writer2 = ResultsWriter(filepath, append=True)
        result2 = MatchResult(game_name="tic_tac_toe", agent_a="c", agent_b="d", winner="c")
        writer2.write(result2)
        writer2.close()

        rows = load_results_csv(filepath)
        assert len(rows) == 2
        assert rows[0]["agent_a"] == "a"
        assert rows[1]["agent_a"] == "c"

    def test_overwrite_mode(self, tmp_path: Path):
        """Test that writer can overwrite existing file."""
        filepath = tmp_path / "results.csv"

        # First writer writes one result
        writer1 = ResultsWriter(filepath)
        result1 = MatchResult(game_name="tic_tac_toe", agent_a="a", agent_b="b", winner="a")
        writer1.write(result1)
        writer1.close()

        # Second writer overwrites with new result
        writer2 = ResultsWriter(filepath, append=False)
        result2 = MatchResult(game_name="tic_tac_toe", agent_a="c", agent_b="d", winner="c")
        writer2.write(result2)
        writer2.close()

        rows = load_results_csv(filepath)
        assert len(rows) == 1
        assert rows[0]["agent_a"] == "c"

    def test_creates_parent_directories(self, tmp_path: Path):
        """Test that writer creates parent directories if needed."""
        filepath = tmp_path / "subdir" / "nested" / "results.csv"
        writer = ResultsWriter(filepath)

        result = MatchResult(game_name="tic_tac_toe", agent_a="a", agent_b="b", winner="a")
        writer.write(result)
        writer.close()

        assert filepath.exists()
        assert filepath.parent.exists()


class TestReadResultsCSV:
    """Tests for load_results_csv function."""

    def test_reads_empty_file(self, tmp_path: Path):
        """Test reading an empty/missing file."""
        filepath = tmp_path / "nonexistent.csv"
        rows = load_results_csv(filepath)
        assert rows == []

    def test_reads_existing_file(self, tmp_path: Path):
        """Test reading an existing CSV file."""
        filepath = tmp_path / "results.csv"

        # Write some data
        writer = ResultsWriter(filepath)
        results = [
            MatchResult(game_name="tic_tac_toe", agent_a="a", agent_b="b", winner="a"),
            MatchResult(game_name="tic_tac_toe", agent_a="c", agent_b="d", winner="d"),
        ]
        writer.write_many(results)
        writer.close()

        # Read it back
        rows = load_results_csv(filepath)
        assert len(rows) == 2
        assert rows[0]["agent_a"] == "a"
        assert rows[1]["agent_a"] == "c"


class TestSchemaCompleteness:
    """Tests to ensure schema includes all required fields."""

    def test_schema_has_required_fields(self):
        """Test that the schema includes all required fields from spec."""
        required_fields = {
            "run_id",
            "game",
            "agent_a",
            "agent_b",
            "agent_a_side",
            "agent_b_side",
            "winner",
            "is_draw",
            "num_moves",
            "seed",
            "invalid_move_retries",
            "agent_a_latency_ms",
            "agent_b_latency_ms",
            "termination_reason",
        }

        schema_fields = set(CSV_COLUMNS)

        for field in required_fields:
            assert field in schema_fields, f"Missing required field: {field}"
