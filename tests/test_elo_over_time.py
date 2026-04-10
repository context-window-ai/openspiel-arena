"""
Tests for scripts.plot_elo_over_time — Elo trajectory computation and visualization.
"""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import pytest

from scripts.plot_elo_over_time import (
    DEFAULT_RATING,
    K_FACTOR,
    EloHistory,
    EloTrajectory,
    compute_elo_trajectories,
    expected_score,
    get_agent_color,
    load_all_results,
    load_results_from_csv,
    main,
    save_trajectory_csv,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_results() -> list[dict[str, str]]:
    """Return a small list of match result dictionaries."""
    return [
        {
            "run_id": "match1",
            "game": "tic_tac_toe",
            "agent_a": "alpha",
            "agent_b": "beta",
            "winner": "alpha",
            "is_draw": "False",
        },
        {
            "run_id": "match2",
            "game": "tic_tac_toe",
            "agent_a": "beta",
            "agent_b": "alpha",
            "winner": "alpha",
            "is_draw": "False",
        },
        {
            "run_id": "match3",
            "game": "tic_tac_toe",
            "agent_a": "alpha",
            "agent_b": "beta",
            "winner": "",
            "is_draw": "True",
        },
    ]


@pytest.fixture()
def sample_csv_file(tmp_path: Path) -> Path:
    """Create a sample CSV file with match results."""
    csv_path = tmp_path / "test_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "run_id",
                "game",
                "agent_a",
                "agent_b",
                "winner",
                "is_draw",
                "num_moves",
            ]
        )
        # Alpha beats beta
        writer.writerow(["id1", "tic_tac_toe", "alpha", "beta", "alpha", "False", "5"])
        # Beta beats gamma
        writer.writerow(["id2", "tic_tac_toe", "beta", "gamma", "beta", "False", "7"])
        # Alpha vs gamma draw
        writer.writerow(["id3", "tic_tac_toe", "alpha", "gamma", "", "True", "9"])
    return csv_path


@pytest.fixture()
def late_entry_csv_file(tmp_path: Path) -> Path:
    """Create a CSV file where an agent enters mid-tournament."""
    csv_path = tmp_path / "late_entry.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["run_id", "game", "agent_a", "agent_b", "winner", "is_draw", "num_moves"]
        )
        # First few matches: only alpha and beta
        writer.writerow(["id1", "test", "alpha", "beta", "alpha", "False", "5"])
        writer.writerow(["id2", "test", "alpha", "beta", "alpha", "False", "5"])
        writer.writerow(["id3", "test", "alpha", "beta", "alpha", "False", "5"])
        # Now gamma enters
        writer.writerow(["id4", "test", "gamma", "alpha", "alpha", "False", "5"])
        writer.writerow(["id5", "test", "gamma", "beta", "beta", "False", "5"])
    return csv_path


@pytest.fixture()
def multi_game_csv_file(tmp_path: Path) -> Path:
    """Create a CSV file with results from multiple games."""
    csv_path = tmp_path / "multi_game.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["run_id", "game", "agent_a", "agent_b", "winner", "is_draw", "num_moves"]
        )
        writer.writerow(["id1", "game_a", "agent1", "agent2", "agent1", "False", "5"])
        writer.writerow(["id2", "game_b", "agent1", "agent2", "agent2", "False", "5"])
        writer.writerow(["id3", "game_a", "agent1", "agent2", "agent1", "False", "5"])
    return csv_path


# ---------------------------------------------------------------------------
# Basic Elo computation tests
# ---------------------------------------------------------------------------


def test_expected_score_equal_ratings():
    """Equal ratings should give 50% expected score."""
    assert expected_score(1500, 1500) == pytest.approx(0.5)


def test_expected_score_higher_vs_lower():
    """Higher rated player should have >50% expected score."""
    assert expected_score(1600, 1400) > 0.5


def test_expected_score_symmetry():
    """Expected scores should sum to 1."""
    assert expected_score(1500, 1700) + expected_score(1700, 1500) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# EloTrajectory tests
# ---------------------------------------------------------------------------


def test_elo_trajectory_record():
    """Should record rating snapshots correctly."""
    traj = EloTrajectory(agent="test_agent", start_rating=1500.0)
    traj.record(1510.0)
    traj.record(1520.0)
    traj.record(1505.0)

    assert len(traj.ratings) == 3
    assert traj.ratings == [1510.0, 1520.0, 1505.0]


def test_elo_trajectory_final_rating():
    """Final rating should return the last recorded value."""
    traj = EloTrajectory(agent="test_agent", start_rating=1500.0)
    assert traj.final_rating == 1500.0  # No ratings recorded

    traj.record(1510.0)
    traj.record(1520.0)
    assert traj.final_rating == 1520.0


# ---------------------------------------------------------------------------
# EloHistory tests
# ---------------------------------------------------------------------------


def test_elo_history_get_trajectory():
    """Should create and retrieve trajectories."""
    history = EloHistory()
    traj = history.get_trajectory("agent1")
    assert isinstance(traj, EloTrajectory)
    assert traj.agent == "agent1"

    # Same agent should return same trajectory
    traj2 = history.get_trajectory("agent1")
    assert traj is traj2


def test_elo_history_to_csv_rows():
    """Should convert to flat CSV rows correctly."""
    history = EloHistory()
    traj1 = history.get_trajectory("alpha")
    traj1.record(1500.0)
    traj1.record(1510.0)
    traj1.record(1505.0)

    traj2 = history.get_trajectory("beta")
    traj2.record(1500.0)
    traj2.record(1490.0)
    traj2.record(1495.0)

    rows = history.to_csv_rows()

    # Should have 3 rows per agent
    assert len(rows) == 6

    # Check structure
    assert all("match_index" in row for row in rows)
    assert all("agent" in row for row in rows)
    assert all("elo_rating" in row for row in rows)


# ---------------------------------------------------------------------------
# Trajectory computation tests
# ---------------------------------------------------------------------------


def test_compute_trajectories_basic(sample_results: list[dict[str, str]]):
    """Should compute trajectories with correct number of snapshots."""
    history = compute_elo_trajectories(sample_results)

    assert len(history.trajectories) == 2
    assert "alpha" in history.trajectories
    assert "beta" in history.trajectories
    assert history.total_matches == 3

    # Each trajectory should have 4 points: initial + after each of 3 matches
    alpha_traj = history.trajectories["alpha"]
    assert len(alpha_traj.ratings) == 4  # 1 initial + 3 updates


def test_compute_trajectories_alpha_wins(sample_results: list[dict[str, str]]):
    """Alpha wins twice and draws once, should have highest final rating."""
    history = compute_elo_trajectories(sample_results)

    alpha_final = history.trajectories["alpha"].final_rating
    beta_final = history.trajectories["beta"].final_rating

    assert alpha_final > beta_final


def test_compute_trajectories_custom_params(sample_results: list[dict[str, str]]):
    """Should accept custom K-factor and start rating."""
    history = compute_elo_trajectories(sample_results, k=50, start_rating=1000)

    # Alpha should still be higher rated
    alpha_final = history.trajectories["alpha"].final_rating
    beta_final = history.trajectories["beta"].final_rating

    assert alpha_final > 1000
    assert beta_final < 1000


def test_compute_trajectories_empty_results():
    """Empty results should give empty history."""
    history = compute_elo_trajectories([])
    assert len(history.trajectories) == 0
    assert history.total_matches == 0


def test_compute_trajectories_late_entry_agent(late_entry_csv_file: Path):
    """Agents entering mid-tournament should initialize at start rating."""
    results = load_results_from_csv(late_entry_csv_file)
    history = compute_elo_trajectories(results)

    # All three agents should be present
    assert "alpha" in history.trajectories
    assert "beta" in history.trajectories
    assert "gamma" in history.trajectories

    gamma_traj = history.trajectories["gamma"]

    # Gamma's first rating should be the default start rating
    assert gamma_traj.ratings[0] == DEFAULT_RATING

    # Gamma's trajectory should be shorter than alpha's (entered later)
    alpha_traj = history.trajectories["alpha"]
    assert len(gamma_traj.ratings) < len(alpha_traj.ratings)


def test_compute_trajectories_game_filter(multi_game_csv_file: Path):
    """Game filter should exclude results from other games."""
    results = load_results_from_csv(multi_game_csv_file)

    # Filter to game_a
    history_a = compute_elo_trajectories(results, game_filter="game_a")
    assert history_a.total_matches == 2  # Only 2 game_a matches

    # Filter to game_b
    history_b = compute_elo_trajectories(results, game_filter="game_b")
    assert history_b.total_matches == 1  # Only 1 game_b match


def test_compute_trajectories_agent_filter(sample_results: list[dict[str, str]]):
    """Agent filter should only include specified agents."""
    # Add another agent to the results
    extended_results = list(sample_results) + [
        {
            "run_id": "match4",
            "game": "tic_tac_toe",
            "agent_a": "gamma",
            "agent_b": "delta",
            "winner": "gamma",
            "is_draw": "False",
        }
    ]

    # Filter to only alpha and beta
    history = compute_elo_trajectories(
        extended_results, agent_filter={"alpha", "beta"}
    )

    assert len(history.trajectories) == 2
    assert "alpha" in history.trajectories
    assert "beta" in history.trajectories
    assert "gamma" not in history.trajectories
    assert "delta" not in history.trajectories


# ---------------------------------------------------------------------------
# CSV loading tests
# ---------------------------------------------------------------------------


def test_load_results_from_csv(sample_csv_file: Path):
    """Should load CSV rows correctly."""
    rows = load_results_from_csv(sample_csv_file)
    assert len(rows) == 3
    assert rows[0]["agent_a"] == "alpha"
    assert rows[0]["winner"] == "alpha"


def test_load_results_from_csv_not_found():
    """Should raise FileNotFoundError for missing file."""
    with pytest.raises(FileNotFoundError):
        load_results_from_csv("/nonexistent/path/file.csv")


def test_load_all_results(sample_csv_file: Path, tmp_path: Path):
    """Should concatenate results from multiple files."""
    # Create second CSV file
    csv2 = tmp_path / "results2.csv"
    with open(csv2, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["run_id", "game", "agent_a", "agent_b", "winner", "is_draw"])
        writer.writerow(["id4", "test", "x", "y", "x", "False"])

    all_results = load_all_results(sample_csv_file, csv2)
    assert len(all_results) == 4  # 3 from first file + 1 from second


# ---------------------------------------------------------------------------
# Color assignment tests
# ---------------------------------------------------------------------------


def test_get_agent_color_llm():
    """LLM agents should get warm colors."""
    agents = ["llm-gpt-4", "llm-claude", "random"]
    color = get_agent_color("llm-gpt-4", agents)
    # Warm colors have higher red component
    assert color[0] > 0.5  # Red channel


def test_get_agent_color_search():
    """Search-based agents should get cool colors."""
    agents = ["random", "mcts-500", "llm-gpt-4"]
    color = get_agent_color("mcts-500", agents)
    # Cool colors have higher blue component relative to red
    assert color[2] >= color[0]  # Blue >= Red


def test_get_agent_color_consistency():
    """Same agent should always get same color given same agent list."""
    agents = ["alpha", "beta", "gamma"]
    color1 = get_agent_color("alpha", agents)
    color2 = get_agent_color("alpha", agents)
    assert color1 == color2


# ---------------------------------------------------------------------------
# CSV output tests
# ---------------------------------------------------------------------------


def test_save_trajectory_csv(tmp_path: Path):
    """Should save trajectory data to CSV correctly."""
    history = EloHistory()
    traj = history.get_trajectory("test_agent")
    traj.record(1500.0)
    traj.record(1510.0)
    traj.record(1505.0)

    output_path = tmp_path / "test_output.csv"
    save_trajectory_csv(history, output_path)

    assert output_path.exists()

    # Read back and verify
    with open(output_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 3
    assert rows[0]["agent"] == "test_agent"
    assert rows[0]["match_index"] == "0"
    assert float(rows[0]["elo_rating"]) == 1500.0


def test_save_trajectory_csv_format(tmp_path: Path):
    """CSV should have correct column headers."""
    history = EloHistory()
    traj = history.get_trajectory("agent")
    traj.record(1500.0)

    output_path = tmp_path / "test.csv"
    save_trajectory_csv(history, output_path)

    with open(output_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert set(reader.fieldnames) == {"match_index", "agent", "elo_rating"}


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------


def test_main_help():
    """Help should print usage and exit with 0."""
    # argparse calls sys.exit() when --help is passed
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    assert exc_info.value.code == 0


def test_main_file_not_found():
    """Missing file should return 1."""
    assert main(["/nonexistent/file.csv"]) == 1


def test_main_success(sample_csv_file: Path, tmp_path: Path):
    """Valid file should generate outputs and return 0."""
    output_dir = tmp_path / "output"
    result = main([str(sample_csv_file), "--output-dir", str(output_dir)])
    assert result == 0

    # Check outputs exist
    assert (output_dir / "elo_over_time.png").exists()
    assert (output_dir / "elo_over_time.csv").exists()


def test_main_game_filter(multi_game_csv_file: Path, tmp_path: Path):
    """Game filter should work via CLI."""
    output_dir = tmp_path / "output"
    result = main(
        [str(multi_game_csv_file), "--output-dir", str(output_dir), "--game", "game_a"]
    )
    assert result == 0

    # Check output has game suffix
    assert (output_dir / "elo_over_time_game_a.png").exists()


def test_main_agent_filter(sample_csv_file: Path, tmp_path: Path):
    """Agent filter should work via CLI."""
    output_dir = tmp_path / "output"
    result = main(
        [
            str(sample_csv_file),
            "--output-dir",
            str(output_dir),
            "--agents",
            "alpha",
        ]
    )
    assert result == 0

    # Check CSV only has alpha
    csv_path = output_dir / "elo_over_time.csv"
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        agents = {row["agent"] for row in reader}
    assert agents == {"alpha"}


def test_main_custom_k_and_start(sample_csv_file: Path, tmp_path: Path):
    """Custom K and start rating should affect results."""
    output_dir = tmp_path / "output"
    result = main(
        [
            str(sample_csv_file),
            "--output-dir",
            str(output_dir),
            "--k",
            "50",
            "--start",
            "1000",
        ]
    )
    assert result == 0

    # Read CSV and verify ratings are around 1000
    csv_path = output_dir / "elo_over_time.csv"
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        ratings = [float(row["elo_rating"]) for row in reader]

    # All ratings should be closer to 1000 than 1500
    assert all(abs(r - 1000) < 200 for r in ratings)


def test_main_no_csv_files():
    """No CSV files should return 1."""
    result = main(["/nonexistent/*.csv"])
    assert result == 1
