"""
Tests for ratings.elo — Elo computation, CSV loading, and leaderboards.
"""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import pytest

from arena.result import MatchResult
from ratings.elo import (
    DEFAULT_RATING,
    K_FACTOR,
    AgentStats,
    Leaderboard,
    MatchupStats,
    compute_leaderboard,
    compute_leaderboard_from_csv,
    expected_score,
    load_results_from_csv,
    main,
    row_to_match_result,
    update_elo,
)


# ---------------------------------------------------------------------------
# Basic Elo computation tests
# ---------------------------------------------------------------------------


def test_expected_score_equal_ratings():
    """Equal ratings should give 50% expected score."""
    assert expected_score(1500, 1500) == pytest.approx(0.5)


def test_expected_score_higher_beats_lower():
    """Higher rated player should have >50% expected score."""
    assert expected_score(1600, 1400) > 0.5


def test_expected_score_lower_vs_higher():
    """Lower rated player should have <50% expected score."""
    assert expected_score(1400, 1600) < 0.5


def test_expected_score_symmetry():
    """Expected scores should sum to 1."""
    assert expected_score(1500, 1700) + expected_score(1700, 1500) == pytest.approx(1.0)


def test_update_elo_winner_gains(sample_results):
    """Winner should gain rating, loser should lose."""
    ratings = update_elo(sample_results)
    assert "alpha" in ratings
    assert "beta" in ratings
    # Alpha won twice and drew once, should be higher rated
    assert ratings["alpha"] > ratings["beta"]


def test_update_elo_empty_results():
    """Empty results should return empty dict."""
    ratings = update_elo([])
    assert ratings == {}


def test_update_elo_default_rating():
    """New agents should start at default rating."""
    r = MatchResult(
        agent_a="x", agent_b="y", game_name="g", outcome="win", returns=[1.0, -1.0]
    )
    ratings = update_elo([r])
    assert ratings["x"] > DEFAULT_RATING
    assert ratings["y"] < DEFAULT_RATING


def test_update_elo_draw_minimal_change():
    """Draw between equal players should keep ratings close to default."""
    r = MatchResult(
        agent_a="x", agent_b="y", game_name="g", outcome="draw", returns=[0.0, 0.0]
    )
    ratings = update_elo([r])
    assert ratings["x"] == pytest.approx(DEFAULT_RATING, abs=1e-6)
    assert ratings["y"] == pytest.approx(DEFAULT_RATING, abs=1e-6)


def test_update_elo_custom_k_factor():
    """Higher K-factor should produce larger rating changes."""
    r = MatchResult(agent_a="x", agent_b="y", game_name="g", outcome="win")

    ratings_low_k = update_elo([r], k=10)
    ratings_high_k = update_elo([r], k=50)

    # Higher K should produce bigger change from default
    assert abs(ratings_high_k["x"] - DEFAULT_RATING) > abs(
        ratings_low_k["x"] - DEFAULT_RATING
    )


def test_update_elo_custom_default_rating():
    """Should be able to set custom default rating."""
    r = MatchResult(agent_a="x", agent_b="y", game_name="g", outcome="draw")
    ratings = update_elo([r], default=1000)
    assert ratings["x"] == pytest.approx(1000, abs=1e-6)
    assert ratings["y"] == pytest.approx(1000, abs=1e-6)


def test_update_elo_initial_ratings():
    """Should be able to provide initial ratings."""
    r = MatchResult(agent_a="x", agent_b="y", game_name="g", outcome="draw")
    initial = {"x": 2000, "y": 1000}
    ratings = update_elo([r], initial_ratings=initial)
    # X is higher rated, expected to win, draw is worse for x
    assert ratings["x"] < 2000
    assert ratings["y"] > 1000


# ---------------------------------------------------------------------------
# CSV Loading tests
# ---------------------------------------------------------------------------


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
            ]
        )
        # Alpha beats beta (alpha wins)
        writer.writerow(
            [
                "id1",
                "tic_tac_toe",
                "alpha",
                "beta",
                0,
                1,
                "alpha",
                "False",
                5,
                "",
                0,
                0.0,
                0.0,
                "normal",
            ]
        )
        # Beta beats gamma (beta wins)
        writer.writerow(
            [
                "id2",
                "tic_tac_toe",
                "beta",
                "gamma",
                0,
                1,
                "beta",
                "False",
                7,
                "",
                0,
                0.0,
                0.0,
                "normal",
            ]
        )
        # Alpha vs gamma draw
        writer.writerow(
            [
                "id3",
                "tic_tac_toe",
                "alpha",
                "gamma",
                0,
                1,
                "",
                "True",
                9,
                "",
                0,
                0.0,
                0.0,
                "normal",
            ]
        )
    return csv_path


def test_load_results_from_csv(sample_csv_file: Path):
    """Should load CSV rows correctly."""
    rows = load_results_from_csv(sample_csv_file)
    assert len(rows) == 3
    assert rows[0]["agent_a"] == "alpha"
    assert rows[0]["winner"] == "alpha"
    assert rows[2]["is_draw"] == "True"


def test_load_results_from_csv_not_found():
    """Should raise FileNotFoundError for missing file."""
    with pytest.raises(FileNotFoundError):
        load_results_from_csv("/nonexistent/path/file.csv")


def test_row_to_match_result_win():
    """Should convert win row correctly."""
    row = {
        "run_id": "test-id",
        "game": "tic_tac_toe",
        "agent_a": "alice",
        "agent_b": "bob",
        "winner": "alice",
        "is_draw": "False",
        "num_moves": "5",
    }
    result = row_to_match_result(row)
    assert result.agent_a == "alice"
    assert result.agent_b == "bob"
    assert result.winner == "alice"
    assert result.outcome == "win"
    assert result.game_name == "tic_tac_toe"


def test_row_to_match_result_loss():
    """Should convert loss row correctly."""
    row = {
        "run_id": "test-id",
        "game": "tic_tac_toe",
        "agent_a": "alice",
        "agent_b": "bob",
        "winner": "bob",
        "is_draw": "False",
        "num_moves": "5",
    }
    result = row_to_match_result(row)
    assert result.outcome == "loss"


def test_row_to_match_result_draw():
    """Should convert draw row correctly."""
    row = {
        "run_id": "test-id",
        "game": "tic_tac_toe",
        "agent_a": "alice",
        "agent_b": "bob",
        "winner": "",
        "is_draw": "True",
        "num_moves": "9",
    }
    result = row_to_match_result(row)
    assert result.winner is None
    assert result.outcome == "draw"


# ---------------------------------------------------------------------------
# Leaderboard tests
# ---------------------------------------------------------------------------


def test_matchup_stats():
    """MatchupStats should compute stats correctly."""
    m = MatchupStats(opponent="bob", wins=3, losses=1, draws=2)
    assert m.games == 6
    assert m.win_rate == pytest.approx((3 + 0.5 * 2) / 6)


def test_matchup_stats_zero_games():
    """Win rate should be 0 for zero games."""
    m = MatchupStats(opponent="bob")
    assert m.games == 0
    assert m.win_rate == 0.0


def test_agent_stats():
    """AgentStats should track matchups correctly."""
    agent = AgentStats(name="alice", rating=1600, wins=5, losses=2, draws=1)
    assert agent.games == 8
    assert agent.win_rate == pytest.approx((5 + 0.5 * 1) / 8)


def test_leaderboard_sorted_by_rating():
    """Leaderboard should sort agents by rating descending."""
    lb = Leaderboard()
    lb.agents["low"] = AgentStats(name="low", rating=1000)
    lb.agents["high"] = AgentStats(name="high", rating=2000)
    lb.agents["mid"] = AgentStats(name="mid", rating=1500)

    sorted_agents = lb.sorted_by_rating()
    assert sorted_agents[0].name == "high"
    assert sorted_agents[1].name == "mid"
    assert sorted_agents[2].name == "low"


def test_compute_leaderboard_empty():
    """Empty results should give empty leaderboard."""
    lb = compute_leaderboard([])
    assert len(lb.agents) == 0
    assert lb.total_matches == 0


def test_compute_leaderboard_single_match():
    """Single match should produce correct stats."""
    results = [
        MatchResult(
            game_name="test_game",
            agent_a="alice",
            agent_b="bob",
            winner="alice",
        )
    ]
    lb = compute_leaderboard(results)

    assert lb.game == "test_game"
    assert lb.total_matches == 1
    assert len(lb.agents) == 2

    alice = lb.agents["alice"]
    bob = lb.agents["bob"]

    assert alice.wins == 1
    assert alice.losses == 0
    assert alice.rating > bob.rating

    assert bob.wins == 0
    assert bob.losses == 1


def test_compute_leaderboard_with_draws():
    """Draws should be tracked correctly."""
    results = [
        MatchResult(game_name="test", agent_a="a", agent_b="b", outcome="draw"),
        MatchResult(game_name="test", agent_a="a", agent_b="b", outcome="draw"),
    ]
    lb = compute_leaderboard(results)

    assert lb.agents["a"].draws == 2
    assert lb.agents["b"].draws == 2


def test_compute_leaderboard_matchups():
    """Matchup stats should be tracked correctly."""
    results = [
        MatchResult(game_name="test", agent_a="alice", agent_b="bob", outcome="win"),
        MatchResult(game_name="test", agent_a="alice", agent_b="bob", outcome="win"),
        MatchResult(game_name="test", agent_a="alice", agent_b="bob", outcome="loss"),
        MatchResult(game_name="test", agent_a="alice", agent_b="carol", outcome="draw"),
    ]
    lb = compute_leaderboard(results)

    alice = lb.agents["alice"]

    # Alice vs Bob: 2-1-0
    assert alice.matchups["bob"].wins == 2
    assert alice.matchups["bob"].losses == 1
    assert alice.matchups["bob"].draws == 0

    # Alice vs Carol: 0-0-1
    assert alice.matchups["carol"].wins == 0
    assert alice.matchups["carol"].losses == 0
    assert alice.matchups["carol"].draws == 1


def test_compute_leaderboard_rating_ordering():
    """Stronger agents should have higher ratings."""
    # Create a scenario where alpha clearly dominates
    results = []
    for _ in range(10):
        results.append(
            MatchResult(game_name="test", agent_a="alpha", agent_b="beta", outcome="win")
        )
        results.append(
            MatchResult(game_name="test", agent_a="beta", agent_b="gamma", outcome="win")
        )
        results.append(
            MatchResult(game_name="test", agent_a="alpha", agent_b="gamma", outcome="win")
        )

    lb = compute_leaderboard(results)
    sorted_agents = lb.sorted_by_rating()

    # Alpha should be highest, gamma lowest
    assert sorted_agents[0].name == "alpha"
    assert sorted_agents[1].name == "beta"
    assert sorted_agents[2].name == "gamma"

    # Verify rating ordering
    assert sorted_agents[0].rating > sorted_agents[1].rating > sorted_agents[2].rating


def test_compute_leaderboard_custom_params():
    """Should accept custom K-factor and default rating."""
    results = [
        MatchResult(game_name="test", agent_a="a", agent_b="b", outcome="win"),
    ]
    lb = compute_leaderboard(results, k=50, default_rating=1000)

    assert lb.agents["a"].rating > 1000
    assert lb.agents["b"].rating < 1000


# ---------------------------------------------------------------------------
# CSV Leaderboard Integration tests
# ---------------------------------------------------------------------------


def test_compute_leaderboard_from_csv(sample_csv_file: Path):
    """Should compute leaderboard from CSV file."""
    lb = compute_leaderboard_from_csv(sample_csv_file)

    assert lb.game == "tic_tac_toe"
    assert lb.total_matches == 3
    assert len(lb.agents) == 3

    # Alpha: 1 win, 0 losses, 1 draw
    assert lb.agents["alpha"].wins == 1
    assert lb.agents["alpha"].losses == 0
    assert lb.agents["alpha"].draws == 1

    # Beta: 1 win, 1 loss, 0 draws
    assert lb.agents["beta"].wins == 1
    assert lb.agents["beta"].losses == 1
    assert lb.agents["beta"].draws == 0

    # Gamma: 0 wins, 1 loss, 1 draw
    assert lb.agents["gamma"].wins == 0
    assert lb.agents["gamma"].losses == 1
    assert lb.agents["gamma"].draws == 1


def test_compute_leaderboard_from_csv_rating_order(sample_csv_file: Path):
    """Alpha should have highest rating (1 win, 1 draw, no losses)."""
    lb = compute_leaderboard_from_csv(sample_csv_file)
    sorted_agents = lb.sorted_by_rating()

    # Alpha is undefeated with a win and a draw, should be top
    assert sorted_agents[0].name == "alpha"


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------


def test_main_help():
    """Help should print usage and return 0."""
    assert main(["--help"]) == 0


def test_main_no_args():
    """No args should print usage and return 0."""
    assert main([]) == 0


def test_main_file_not_found():
    """Missing file should return 1."""
    assert main(["/nonexistent/file.csv"]) == 1


def test_main_success(sample_csv_file: Path, capsys):
    """Valid file should print leaderboard and return 0."""
    result = main([str(sample_csv_file)])
    assert result == 0

    captured = capsys.readouterr()
    assert "Leaderboard" in captured.out
    assert "alpha" in captured.out
    assert "beta" in captured.out
    assert "gamma" in captured.out


def test_main_custom_params(sample_csv_file: Path, capsys):
    """Custom K and default should be accepted."""
    result = main([str(sample_csv_file), "--k", "50", "--default", "1000"])
    assert result == 0

    captured = capsys.readouterr()
    assert "Leaderboard" in captured.out


def test_main_unknown_arg(sample_csv_file: Path):
    """Unknown argument should return 1."""
    result = main([str(sample_csv_file), "--unknown"])
    assert result == 1


# ---------------------------------------------------------------------------
# Leaderboard serialization tests
# ---------------------------------------------------------------------------


def test_leaderboard_to_dict():
    """Leaderboard should serialize to dict correctly."""
    lb = Leaderboard(game="test_game", total_matches=5)
    lb.agents["alice"] = AgentStats(name="alice", rating=1600, wins=3, losses=1, draws=1)

    d = lb.to_dict()
    assert d["game"] == "test_game"
    assert d["total_matches"] == 5
    assert len(d["agents"]) == 1
    assert d["agents"][0]["name"] == "alice"
    assert d["agents"][0]["rating"] == 1600.0


def test_agent_stats_to_dict():
    """AgentStats should serialize correctly."""
    agent = AgentStats(name="alice", rating=1600, wins=3, losses=1, draws=1)
    agent.get_matchup("bob").wins = 2

    d = agent.to_dict()
    assert d["name"] == "alice"
    assert d["wins"] == 3
    assert "bob" in d["matchups"]


def test_leaderboard_str():
    """Leaderboard should have pretty string representation."""
    lb = Leaderboard(game="test")
    lb.agents["alice"] = AgentStats(name="alice", rating=1600, wins=3, losses=1)
    lb.agents["bob"] = AgentStats(name="bob", rating=1400, wins=1, losses=3)
    lb.total_matches = 4

    s = str(lb)
    assert "Leaderboard: test" in s
    assert "alice" in s
    assert "bob" in s
    assert "Total matches: 4" in s
