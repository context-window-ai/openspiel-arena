"""
Tests for arena.tournament — tournament runner with round-robin scheduling.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from arena.result import MatchResult, TournamentResult
from arena.scheduler import count_pairings, round_robin_with_side_swap
from arena.tournament import (
    TournamentManifest,
    load_tournament_results,
    run_tournament,
)


# ---------------------------------------------------------------------------
# Mock Agent for Testing (avoids dependency on agents/__init__.py)
# ---------------------------------------------------------------------------


class MockAgent:
    """Simple mock agent for testing without external dependencies.
    
    This class is structurally compatible with the Agent protocol but
    doesn't require importing from agents.base (which triggers the
    agents/__init__.py that has optional dependencies).
    """

    def __init__(self, name: str, seed: int | None = None) -> None:
        self.name = name
        import random
        self._rng = random.Random(seed)

    def select_action(
        self,
        state_view: Any,
        legal_actions: list[int],
        context: Any = None,
    ) -> int:
        if not legal_actions:
            raise ValueError("No legal actions")
        return self._rng.choice(legal_actions)

    def __repr__(self) -> str:
        return f"MockAgent(name={self.name!r})"


def _make_agents(n: int) -> list[MockAgent]:
    """Create n mock agents for testing."""
    return [MockAgent(name=f"agent-{i}", seed=i) for i in range(n)]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def game():
    """Tic-tac-toe game fixture."""
    pytest.importorskip("pyspiel", reason="open_spiel not installed")
    from games.tic_tac_toe import TicTacToeGame

    return TicTacToeGame()


@pytest.fixture()
def agents():
    """Three mock agents for testing."""
    return [
        MockAgent(name="alice", seed=0),
        MockAgent(name="bob", seed=1),
        MockAgent(name="carol", seed=2),
    ]


# ---------------------------------------------------------------------------
# Scheduler Tests
# ---------------------------------------------------------------------------


class TestRoundRobinWithSideSwap:
    """Tests for round_robin_with_side_swap scheduler."""

    def test_requires_two_agents(self):
        """Scheduler should require at least 2 agents."""
        with pytest.raises(ValueError, match="at least 2"):
            round_robin_with_side_swap([MockAgent(name="solo")])

    def test_requires_positive_rounds(self):
        """Scheduler should require rounds_per_pairing >= 1."""
        agents = [MockAgent(name="a"), MockAgent(name="b")]
        with pytest.raises(ValueError, match="at least 1"):
            round_robin_with_side_swap(agents, rounds_per_pairing=0)

    def test_two_agents_two_rounds(self):
        """With 2 agents and 2 rounds per pairing, should get 2 matches (one per side)."""
        agents = [MockAgent(name="alice"), MockAgent(name="bob")]
        pairs = round_robin_with_side_swap(agents, rounds_per_pairing=2)

        assert len(pairs) == 2
        # Should have one (alice, bob) and one (bob, alice)
        names = [(a.name, b.name) for a, b in pairs]
        assert ("alice", "bob") in names
        assert ("bob", "alice") in names

    def test_three_agents_two_rounds(self):
        """With 3 agents and 2 rounds per pairing, should get 6 matches."""
        agents = [
            MockAgent(name="alice"),
            MockAgent(name="bob"),
            MockAgent(name="carol"),
        ]
        pairs = round_robin_with_side_swap(agents, rounds_per_pairing=2)

        # 3 pairs * 2 rounds = 6 matches
        assert len(pairs) == 6

    def test_four_agents_one_round(self):
        """With 4 agents and 1 round per pairing, should get 6 matches."""
        agents = _make_agents(4)
        pairs = round_robin_with_side_swap(agents, rounds_per_pairing=1)

        # 4*3/2 = 6 unordered pairs * 1 round = 6 matches
        assert len(pairs) == 6

    def test_four_agents_four_rounds(self):
        """With 4 agents and 4 rounds per pairing, should get 24 matches."""
        agents = _make_agents(4)
        pairs = round_robin_with_side_swap(agents, rounds_per_pairing=4)

        # 4*3/2 = 6 unordered pairs * 4 rounds = 24 matches
        assert len(pairs) == 24

    def test_side_balance_even_rounds(self):
        """With even rounds_per_pairing, sides should be perfectly balanced."""
        agents = [MockAgent(name="alice"), MockAgent(name="bob")]
        pairs = round_robin_with_side_swap(agents, rounds_per_pairing=4)

        # Count directions
        alice_first = sum(1 for a, b in pairs if a.name == "alice")
        bob_first = sum(1 for a, b in pairs if a.name == "bob")

        assert alice_first == 2
        assert bob_first == 2

    def test_side_balance_odd_rounds(self):
        """With odd rounds_per_pairing, sides should be as balanced as possible."""
        agents = [MockAgent(name="alice"), MockAgent(name="bob")]
        pairs = round_robin_with_side_swap(agents, rounds_per_pairing=3)

        # Count directions
        alice_first = sum(1 for a, b in pairs if a.name == "alice")
        bob_first = sum(1 for a, b in pairs if a.name == "bob")

        # One direction gets 2, the other gets 1
        assert abs(alice_first - bob_first) == 1
        assert alice_first + bob_first == 3

    def test_no_self_play(self):
        """Agents should never play against themselves."""
        agents = _make_agents(5)
        pairs = round_robin_with_side_swap(agents, rounds_per_pairing=2)

        for a, b in pairs:
            assert a is not b
            assert a.name != b.name


class TestCountPairings:
    """Tests for count_pairings helper."""

    def test_two_agents(self):
        """2 agents = 1 pairing."""
        agents = [MockAgent(name="a"), MockAgent(name="b")]
        assert count_pairings(agents) == 1

    def test_three_agents(self):
        """3 agents = 3 pairings."""
        agents = [MockAgent(name="a"), MockAgent(name="b"), MockAgent(name="c")]
        assert count_pairings(agents) == 3

    def test_four_agents(self):
        """4 agents = 6 pairings."""
        agents = [MockAgent(name=f"a{i}") for i in range(4)]
        assert count_pairings(agents) == 6


# ---------------------------------------------------------------------------
# TournamentManifest Tests
# ---------------------------------------------------------------------------


class TestTournamentManifest:
    """Tests for TournamentManifest dataclass."""

    def test_to_dict(self):
        """Manifest should serialize to dict correctly."""
        manifest = TournamentManifest(
            run_id="test-123",
            game_name="tic_tac_toe",
            agents=["alice", "bob"],
            rounds_per_pairing=2,
            total_matches=2,
            completed_matches=2,
            failed_matches=0,
            start_time="2024-01-01T00:00:00+00:00",
            end_time="2024-01-01T00:01:00+00:00",
            duration_seconds=60.0,
            results_path="/tmp/results.csv",
            failed_match_ids=[],
        )

        d = manifest.to_dict()
        assert d["run_id"] == "test-123"
        assert d["game_name"] == "tic_tac_toe"
        assert d["agents"] == ["alice", "bob"]
        assert d["total_matches"] == 2
        assert d["completed_matches"] == 2
        assert d["failed_matches"] == 0

    def test_from_dict(self):
        """Manifest should deserialize from dict correctly."""
        data = {
            "run_id": "test-456",
            "game_name": "breakthrough",
            "agents": ["x", "y", "z"],
            "rounds_per_pairing": 4,
            "total_matches": 12,
            "completed_matches": 10,
            "failed_matches": 2,
            "start_time": "2024-01-01T00:00:00Z",
            "end_time": "2024-01-01T00:05:00Z",
            "duration_seconds": 300.0,
            "results_path": "/data/results.csv",
            "failed_match_ids": ["match-1", "match-2"],
        }

        manifest = TournamentManifest.from_dict(data)
        assert manifest.run_id == "test-456"
        assert manifest.game_name == "breakthrough"
        assert manifest.agents == ["x", "y", "z"]
        assert manifest.total_matches == 12
        assert manifest.failed_matches == 2
        assert manifest.failed_match_ids == ["match-1", "match-2"]

    def test_save_and_load(self, tmp_path: Path):
        """Manifest should save to and load from JSON file."""
        manifest = TournamentManifest(
            run_id="test-789",
            game_name="tic_tac_toe",
            agents=["alice", "bob"],
            rounds_per_pairing=2,
            total_matches=2,
        )

        json_path = tmp_path / "manifest.json"
        manifest.save(json_path)

        assert json_path.exists()

        loaded = TournamentManifest.load(json_path)
        assert loaded.run_id == "test-789"
        assert loaded.game_name == "tic_tac_toe"
        assert loaded.agents == ["alice", "bob"]


# ---------------------------------------------------------------------------
# Tournament Runner Tests
# ---------------------------------------------------------------------------


class TestRunTournament:
    """Tests for run_tournament function."""

    def test_basic_tournament(self, game, agents):
        """Run a basic tournament and check results."""
        result, manifest = run_tournament(
            game=game,
            agents=agents,
            rounds_per_pairing=2,
        )

        assert isinstance(result, TournamentResult)
        assert isinstance(manifest, TournamentManifest)

        # 3 agents -> 3 pairings * 2 rounds = 6 matches
        assert len(result.matches) == 6
        assert manifest.total_matches == 6
        assert manifest.completed_matches == 6
        assert manifest.failed_matches == 0

    def test_tournament_with_output_dir(self, game, agents, tmp_path: Path):
        """Tournament should write CSV and manifest to output directory."""
        results_dir = tmp_path / "results"
        result, manifest = run_tournament(
            game=game,
            agents=agents,
            rounds_per_pairing=2,
            results_dir=results_dir,
            run_id="test-run",
        )

        # Check manifest
        assert manifest.run_id == "test-run"
        assert manifest.results_path != ""

        # Check files exist
        results_csv = results_dir / "results_test-run.csv"
        manifest_json = results_dir / "manifest_test-run.json"
        assert results_csv.exists()
        assert manifest_json.exists()

        # Verify manifest content
        with open(manifest_json) as f:
            manifest_data = json.load(f)
        assert manifest_data["run_id"] == "test-run"
        assert manifest_data["game_name"] == "tic_tac_toe"
        assert set(manifest_data["agents"]) == {"alice", "bob", "carol"}

    def test_tournament_requires_two_agents(self, game):
        """Tournament should require at least 2 agents."""
        single_agent = [MockAgent(name="solo")]
        with pytest.raises(ValueError, match="at least 2"):
            run_tournament(game=game, agents=single_agent)

    def test_tournament_game_name(self, game, agents):
        """Tournament result should have correct game name."""
        result, _ = run_tournament(game=game, agents=agents, rounds_per_pairing=1)
        assert result.game_name == "tic_tac_toe"

    def test_tournament_agent_names_in_manifest(self, game, agents):
        """Manifest should contain all agent names."""
        _, manifest = run_tournament(game=game, agents=agents, rounds_per_pairing=1)
        assert "alice" in manifest.agents
        assert "bob" in manifest.agents
        assert "carol" in manifest.agents

    def test_tournament_timing_info(self, game, agents):
        """Manifest should contain timing information."""
        _, manifest = run_tournament(game=game, agents=agents, rounds_per_pairing=1)

        assert manifest.start_time != ""
        assert manifest.end_time != ""
        assert manifest.duration_seconds >= 0

    def test_tournament_all_matches_have_valid_winner(self, game, agents):
        """All completed matches should have a valid winner or be a draw."""
        result, _ = run_tournament(game=game, agents=agents, rounds_per_pairing=1)

        agent_names = {"alice", "bob", "carol"}
        for match in result.matches:
            assert match.winner is None or match.winner in agent_names

    def test_tournament_rounds_per_pairing_affects_count(self, game, agents):
        """More rounds per pairing should produce more matches."""
        result1, _ = run_tournament(game=game, agents=agents, rounds_per_pairing=1)
        result2, _ = run_tournament(game=game, agents=agents, rounds_per_pairing=2)
        result4, _ = run_tournament(game=game, agents=agents, rounds_per_pairing=4)

        # 3 agents -> 3 unordered pairs
        # 1 round: 3 matches
        # 2 rounds: 6 matches
        # 4 rounds: 12 matches
        assert len(result1.matches) == 3
        assert len(result2.matches) == 6
        assert len(result4.matches) == 12

    def test_tournament_with_custom_run_id(self, game, agents):
        """Tournament should use provided run_id."""
        _, manifest = run_tournament(
            game=game,
            agents=agents,
            rounds_per_pairing=1,
            run_id="custom-id-12345",
        )
        assert manifest.run_id == "custom-id-12345"


class TestLoadTournamentResults:
    """Tests for load_tournament_results function."""

    def test_load_empty_directory(self, tmp_path: Path):
        """Loading from empty directory should return empty result."""
        result = load_tournament_results(tmp_path)
        assert len(result.matches) == 0

    def test_load_results(self, game, agents, tmp_path: Path):
        """Should be able to load results saved by tournament."""
        results_dir = tmp_path / "results"

        # Run tournament and save
        result1, manifest1 = run_tournament(
            game=game,
            agents=agents,
            rounds_per_pairing=2,
            results_dir=results_dir,
            run_id="load-test",
        )

        # Load results back
        result2 = load_tournament_results(results_dir)

        assert len(result2.matches) == len(result1.matches)

        # Check match content matches
        for m1, m2 in zip(result1.matches, result2.matches):
            assert m1.agent_a == m2.agent_a
            assert m1.agent_b == m2.agent_b
            assert m1.winner == m2.winner


# ---------------------------------------------------------------------------
# Side Balance Integration Tests
# ---------------------------------------------------------------------------


class TestSideBalance:
    """Integration tests for side balancing in tournaments."""

    def test_two_agents_side_swap(self, game):
        """With 2 agents and 2 rounds, each should play both sides."""
        agents = [MockAgent(name="alice", seed=0), MockAgent(name="bob", seed=1)]
        result, manifest = run_tournament(
            game=game,
            agents=agents,
            rounds_per_pairing=2,
        )

        # Count how many times each agent was player 0 (agent_a)
        alice_as_p0 = sum(1 for m in result.matches if m.agent_a == "alice")
        bob_as_p0 = sum(1 for m in result.matches if m.agent_a == "bob")

        assert alice_as_p0 == 1
        assert bob_as_p0 == 1

    def test_three_agents_side_balance(self, game, agents):
        """With 3 agents and 2 rounds, sides should be balanced."""
        result, _ = run_tournament(game=game, agents=agents, rounds_per_pairing=2)

        # Each agent plays 2 opponents * 1 time per side = 4 matches
        # As player 0: 2 matches, As player 1: 2 matches
        for agent_name in ["alice", "bob", "carol"]:
            as_p0 = sum(1 for m in result.matches if m.agent_a == agent_name)
            as_p1 = sum(1 for m in result.matches if m.agent_b == agent_name)
            assert as_p0 == 2, f"{agent_name} played as p0 {as_p0} times, expected 2"
            assert as_p1 == 2, f"{agent_name} played as p1 {as_p1} times, expected 2"


# ---------------------------------------------------------------------------
# Error Handling Tests
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Tests for error handling in tournaments."""

    def test_failed_match_recorded(self, tmp_path: Path):
        """Failed matches should be recorded in manifest."""
        pytest.importorskip("pyspiel", reason="open_spiel not installed")

        from games.tic_tac_toe import TicTacToeGame

        # Create a game and agents
        game = TicTacToeGame()
        agents = [MockAgent(name="alice"), MockAgent(name="bob")]

        # Run tournament (all should succeed since MockAgent is well-behaved)
        result, manifest = run_tournament(
            game=game,
            agents=agents,
            rounds_per_pairing=2,
            results_dir=tmp_path,
        )

        # For this test, just verify the error tracking fields exist
        assert hasattr(manifest, "failed_matches")
        assert hasattr(manifest, "failed_match_ids")
        assert manifest.failed_matches == 0
        assert manifest.failed_match_ids == []
