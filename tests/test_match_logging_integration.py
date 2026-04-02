"""
Integration tests for arena.match + arena.logging.

Tests that a match between two stub agents completes and writes one row
to a results file.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("pyspiel", reason="open_spiel not installed")

from agents.random_agent import RandomAgent  # noqa: E402
from arena.logging import (  # noqa: E402
    CANONICAL_COLUMNS,
    ResultsWriter,
    load_results_csv,
    write_results_csv,
)
from arena.match import run_match  # noqa: E402
from games.tic_tac_toe import TicTacToeGame  # noqa: E402


@pytest.fixture()
def game():
    return TicTacToeGame()


def test_match_writes_one_row_to_csv(game, tmp_path: Path):
    """Test that a match between two agents writes one row to results file."""
    csv_path = tmp_path / "results.csv"

    # Run a match
    agent_a = RandomAgent(name="agent_a", seed=42)
    agent_b = RandomAgent(name="agent_b", seed=43)
    result = run_match(agent_a, agent_b, game, seed=12345)

    # Log the result
    with ResultsWriter(csv_path) as writer:
        writer.write(result)

    # Verify one row was written
    rows = load_results_csv(csv_path)
    assert len(rows) == 1

    # Verify the row contains expected data
    row = rows[0]
    assert row["run_id"] == result.match_id
    assert row["game"] == "tic_tac_toe"
    assert row["agent_a"] == "agent_a"
    assert row["agent_b"] == "agent_b"
    assert row["seed"] == "12345"


def test_csv_schema_includes_all_required_fields(game, tmp_path: Path):
    """Test that CSV schema includes all fields from spec."""
    csv_path = tmp_path / "results.csv"

    agent_a = RandomAgent(name="agent_a", seed=42)
    agent_b = RandomAgent(name="agent_b", seed=43)
    result = run_match(agent_a, agent_b, game, seed=12345)

    write_results_csv([result], csv_path)

    rows = load_results_csv(csv_path)
    assert len(rows) == 1

    # Check all required fields are present
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

    for field in required_fields:
        assert field in rows[0], f"Missing required field: {field}"


def test_latency_recorded_for_each_match(game, tmp_path: Path):
    """Test that per-agent latency is recorded for each match."""
    csv_path = tmp_path / "results.csv"

    agent_a = RandomAgent(name="agent_a", seed=42)
    agent_b = RandomAgent(name="agent_b", seed=43)
    result = run_match(agent_a, agent_b, game)

    write_results_csv([result], csv_path)

    rows = load_results_csv(csv_path)
    assert len(rows) == 1

    # Latency should be non-negative floats
    latency_a = float(rows[0]["agent_a_latency_ms"])
    latency_b = float(rows[0]["agent_b_latency_ms"])
    assert latency_a >= 0
    assert latency_b >= 0


def test_multiple_matches_logged_correctly(game, tmp_path: Path):
    """Test that multiple matches are logged correctly."""
    csv_path = tmp_path / "results.csv"

    # Run 3 matches
    results = []
    for i in range(3):
        agent_a = RandomAgent(name="agent_a", seed=42 + i)
        agent_b = RandomAgent(name="agent_b", seed=43 + i)
        result = run_match(agent_a, agent_b, game, seed=100 + i)
        results.append(result)

    write_results_csv(results, csv_path)

    rows = load_results_csv(csv_path)
    assert len(rows) == 3

    # Verify each row has unique run_id and correct seed
    run_ids = set()
    for i, row in enumerate(rows):
        assert row["run_id"] not in run_ids, "Duplicate run_id"
        run_ids.add(row["run_id"])
        assert row["seed"] == str(100 + i)


def test_canonical_columns_constant():
    """Test that CANONICAL_COLUMNS contains all required fields."""
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

    for field in required_fields:
        assert field in CANONICAL_COLUMNS, f"Missing field in CANONICAL_COLUMNS: {field}"
