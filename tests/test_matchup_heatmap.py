"""
Tests for scripts.plot_matchup_heatmap module.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from scripts.plot_matchup_heatmap import (
    MatchupMatrix,
    MatchupRecord,
    compute_matchup_matrix,
    compute_matchup_matrix_from_csv,
    expand_glob_paths,
    load_results_from_csv,
    main,
    row_to_outcome,
    save_matrix_csv,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_csv(tmp_path: Path) -> Path:
    """Create a sample CSV file for testing."""
    csv_content = """run_id,game,agent_a,agent_b,agent_a_side,agent_b_side,winner,is_draw,num_moves,seed,invalid_move_retries,agent_a_latency_ms,agent_b_latency_ms,termination_reason
id1,breakthrough,alice,bob,0,1,alice,False,10,,0,0,0,normal
id2,breakthrough,alice,bob,0,1,bob,False,12,,0,0,0,normal
id3,breakthrough,alice,charlie,0,1,alice,False,8,,0,0,0,normal
id4,breakthrough,bob,charlie,0,1,bob,False,15,,0,0,0,normal
id5,breakthrough,charlie,alice,1,0,alice,False,20,,0,0,0,normal
id6,tic_tac_toe,alice,bob,0,1,,True,9,,0,0,0,normal
"""
    csv_file = tmp_path / "test_results.csv"
    csv_file.write_text(csv_content)
    return csv_file


@pytest.fixture
def multi_game_csv(tmp_path: Path) -> Path:
    """Create a CSV with multiple games."""
    csv_content = """run_id,game,agent_a,agent_b,agent_a_side,agent_b_side,winner,is_draw,num_moves,seed,invalid_move_retries,agent_a_latency_ms,agent_b_latency_ms,termination_reason
id1,breakthrough,alice,bob,0,1,alice,False,10,,0,0,0,normal
id2,breakthrough,bob,alice,0,1,bob,False,12,,0,0,0,normal
id3,tic_tac_toe,alice,bob,0,1,alice,False,8,,0,0,0,normal
id4,tic_tac_toe,bob,alice,0,1,alice,False,15,,0,0,0,normal
"""
    csv_file = tmp_path / "multi_game.csv"
    csv_file.write_text(csv_content)
    return csv_file


# ---------------------------------------------------------------------------
# Test MatchupRecord
# ---------------------------------------------------------------------------


class TestMatchupRecord:
    """Tests for MatchupRecord dataclass."""

    def test_empty_record(self) -> None:
        """Empty record should have zero games and win rate."""
        record = MatchupRecord()
        assert record.games == 0
        assert record.win_rate == 0.0

    def test_wins_only(self) -> None:
        """Record with only wins."""
        record = MatchupRecord(wins=10)
        assert record.games == 10
        assert record.win_rate == 1.0

    def test_losses_only(self) -> None:
        """Record with only losses."""
        record = MatchupRecord(losses=10)
        assert record.games == 10
        assert record.win_rate == 0.0

    def test_draws_only(self) -> None:
        """Record with only draws should have 0.5 win rate."""
        record = MatchupRecord(draws=10)
        assert record.games == 10
        assert record.win_rate == 0.5

    def test_mixed_record(self) -> None:
        """Mixed win/loss/draw record."""
        record = MatchupRecord(wins=6, losses=2, draws=2)
        assert record.games == 10
        # Win rate: (6 + 0.5 * 2) / 10 = 0.7
        assert record.win_rate == 0.7

    def test_side_tracking(self) -> None:
        """Side tracking attributes exist."""
        record = MatchupRecord(wins=5, wins_as_side_0=3, wins_as_side_1=2)
        assert record.wins_as_side_0 == 3
        assert record.wins_as_side_1 == 2


# ---------------------------------------------------------------------------
# Test MatchupMatrix
# ---------------------------------------------------------------------------


class TestMatchupMatrix:
    """Tests for MatchupMatrix class."""

    def test_empty_matrix(self) -> None:
        """Empty matrix should have no agents."""
        matrix = MatchupMatrix()
        assert matrix.agents == []

    def test_add_win_result(self) -> None:
        """Adding a win result."""
        matrix = MatchupMatrix()
        matrix.add_result("alice", "bob", "win")
        assert "alice" in matrix.agents
        assert "bob" in matrix.agents
        assert matrix.get_win_rate("alice", "bob") == 1.0
        assert matrix.get_win_rate("bob", "alice") == 0.0

    def test_diagonal_is_nan(self) -> None:
        """Diagonal entries should be NaN (no self-play)."""
        matrix = MatchupMatrix()
        matrix.add_result("alice", "bob", "win")

        # Self-play should return NaN
        win_rate = matrix.get_win_rate("alice", "alice")
        assert isinstance(win_rate, float)
        assert win_rate != win_rate  # NaN check

    def test_matrix_diagonal_nan(self) -> None:
        """Matrix diagonal should contain NaN values."""
        matrix = MatchupMatrix()
        matrix.add_result("alice", "bob", "win")
        matrix.add_result("bob", "charlie", "loss")

        mat = matrix.to_matrix()
        agents = sorted(matrix.agents)

        # All diagonal elements should be NaN
        for i, agent in enumerate(agents):
            assert np.isnan(mat[i][i])

    def test_side_balanced_aggregation(self) -> None:
        """Side-balanced aggregation averages across positions."""
        matrix = MatchupMatrix()

        # alice beats bob when alice is row agent (1 game)
        matrix.add_result("alice", "bob", "win", agent_a_side=0)

        # bob beats alice when bob is row agent (1 game)
        # This means alice lost when she was column agent
        matrix.add_result("bob", "alice", "win", agent_a_side=0)

        # Without side-balancing:
        # alice vs bob: 1 win in 1 game = 1.0
        assert matrix.get_win_rate("alice", "bob", side_balanced=False) == 1.0

        # With side-balancing:
        # alice has 1 win as row, 0 wins as column (1 loss)
        # Total: 1 win out of 2 games = 0.5
        assert matrix.get_win_rate("alice", "bob", side_balanced=True) == 0.5

    def test_side_balanced_with_multiple_games(self) -> None:
        """Side-balanced with multiple games per position."""
        matrix = MatchupMatrix()

        # alice vs bob: 3 wins, 1 loss as row agent
        matrix.add_result("alice", "bob", "win", agent_a_side=0)
        matrix.add_result("alice", "bob", "win", agent_a_side=0)
        matrix.add_result("alice", "bob", "win", agent_a_side=0)
        matrix.add_result("alice", "bob", "loss", agent_a_side=0)

        # bob vs alice: 2 wins, 2 losses (alice wins 2 as column)
        matrix.add_result("bob", "alice", "win", agent_a_side=0)
        matrix.add_result("bob", "alice", "win", agent_a_side=0)
        matrix.add_result("bob", "alice", "loss", agent_a_side=0)
        matrix.add_result("bob", "alice", "loss", agent_a_side=0)

        # Side-balanced: alice has 3+2=5 wins out of 8 games = 0.625
        win_rate = matrix.get_win_rate("alice", "bob", side_balanced=True)
        assert win_rate == pytest.approx(5 / 8)

    def test_get_sample_count(self) -> None:
        """Sample count sums games from both directions."""
        matrix = MatchupMatrix()

        # 3 games alice vs bob
        matrix.add_result("alice", "bob", "win")
        matrix.add_result("alice", "bob", "win")
        matrix.add_result("alice", "bob", "loss")

        # 2 games bob vs alice
        matrix.add_result("bob", "alice", "win")
        matrix.add_result("bob", "alice", "loss")

        # Total: 5 games
        assert matrix.get_sample_count("alice", "bob") == 5
        assert matrix.get_sample_count("bob", "alice") == 5

    def test_sample_count_self_play_zero(self) -> None:
        """Sample count for self-play should be 0."""
        matrix = MatchupMatrix()
        matrix.add_result("alice", "bob", "win")

        assert matrix.get_sample_count("alice", "alice") == 0

    def test_to_numpy(self) -> None:
        """Conversion to numpy array."""
        matrix = MatchupMatrix()
        matrix.add_result("alice", "bob", "win")
        matrix.add_result("bob", "charlie", "win")

        arr = matrix.to_numpy()
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (3, 3)

    def test_to_sample_matrix(self) -> None:
        """Sample count matrix generation."""
        matrix = MatchupMatrix()
        matrix.add_result("alice", "bob", "win")
        matrix.add_result("alice", "bob", "win")
        matrix.add_result("bob", "alice", "win")

        sample_matrix = matrix.to_sample_matrix()
        # 3 total games
        assert sample_matrix[0][1] == 3  # alice vs bob
        assert sample_matrix[1][0] == 3  # bob vs alice


# ---------------------------------------------------------------------------
# Test row_to_outcome
# ---------------------------------------------------------------------------


class TestRowToOutcome:
    """Tests for row_to_outcome function."""

    def test_win_outcome(self) -> None:
        """Win when agent_a is the winner."""
        row = {"agent_a": "alice", "agent_b": "bob", "winner": "alice", "is_draw": "False", "agent_a_side": "0"}
        agent_a, agent_b, outcome, side = row_to_outcome(row)
        assert agent_a == "alice"
        assert agent_b == "bob"
        assert outcome == "win"
        assert side == 0

    def test_loss_outcome(self) -> None:
        """Loss when agent_b is the winner."""
        row = {"agent_a": "alice", "agent_b": "bob", "winner": "bob", "is_draw": "False", "agent_a_side": "1"}
        _, _, outcome, side = row_to_outcome(row)
        assert outcome == "loss"
        assert side == 1

    def test_draw_outcome(self) -> None:
        """Draw when is_draw is True."""
        row = {"agent_a": "alice", "agent_b": "bob", "winner": "", "is_draw": "True", "agent_a_side": "0"}
        _, _, outcome, _ = row_to_outcome(row)
        assert outcome == "draw"


# ---------------------------------------------------------------------------
# Test CSV Loading
# ---------------------------------------------------------------------------


class TestCSVLoading:
    """Tests for CSV loading functions."""

    def test_load_results_from_csv(self, sample_csv: Path) -> None:
        """Load results from a CSV file."""
        results = load_results_from_csv(sample_csv)
        assert len(results) == 6
        assert results[0]["agent_a"] == "alice"
        assert results[0]["game"] == "breakthrough"

    def test_load_nonexistent_file(self) -> None:
        """Loading a nonexistent file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_results_from_csv("/nonexistent/file.csv")

    def test_compute_from_csv(self, sample_csv: Path) -> None:
        """Compute matchup matrix directly from CSV file."""
        matrix = compute_matchup_matrix_from_csv(str(sample_csv))

        assert len(matrix.agents) == 3
        assert "alice" in matrix.agents
        assert "bob" in matrix.agents
        assert "charlie" in matrix.agents

    def test_game_filter(self, sample_csv: Path) -> None:
        """Game filter should exclude non-matching games."""
        matrix = compute_matchup_matrix_from_csv(str(sample_csv), game_filter="breakthrough")

        # The draw (id6) is from tic_tac_toe, should be excluded
        # Check that we only have breakthrough games
        assert matrix.game == "breakthrough"


# ---------------------------------------------------------------------------
# Test CLI
# ---------------------------------------------------------------------------


class TestCLI:
    """Tests for CLI functionality."""

    def test_main_basic(self, sample_csv: Path, tmp_path: Path) -> None:
        """Basic CLI run should create output files."""
        output_dir = tmp_path / "output"
        result = main([str(sample_csv), "--output-dir", str(output_dir)])

        assert result == 0
        assert (output_dir / "matchup_heatmap.png").exists()
        assert (output_dir / "matchup_matrix.csv").exists()

    def test_main_with_game_filter(self, multi_game_csv: Path, tmp_path: Path) -> None:
        """Game filter should create appropriately named output."""
        output_dir = tmp_path / "output"
        result = main([str(multi_game_csv), "--output-dir", str(output_dir), "--game", "breakthrough"])

        assert result == 0
        assert (output_dir / "matchup_heatmap_breakthrough.png").exists()
        assert (output_dir / "matchup_matrix_breakthrough.csv").exists()

    def test_main_side_balanced(self, sample_csv: Path, tmp_path: Path) -> None:
        """Side-balanced flag should create appropriately named output."""
        output_dir = tmp_path / "output"
        result = main([str(sample_csv), "--output-dir", str(output_dir), "--side-balanced"])

        assert result == 0
        assert (output_dir / "matchup_heatmap_side_balanced.png").exists()
        assert (output_dir / "matchup_matrix_side_balanced.csv").exists()

    def test_main_no_annotate(self, sample_csv: Path, tmp_path: Path) -> None:
        """No-annotate flag should run without error."""
        output_dir = tmp_path / "output"
        result = main([str(sample_csv), "--output-dir", str(output_dir), "--no-annotate"])

        assert result == 0
        assert (output_dir / "matchup_heatmap.png").exists()

    def test_main_show_samples(self, sample_csv: Path, tmp_path: Path) -> None:
        """Show-samples flag should run without error."""
        output_dir = tmp_path / "output"
        result = main([str(sample_csv), "--output-dir", str(output_dir), "--show-samples"])

        assert result == 0
        assert (output_dir / "matchup_heatmap.png").exists()

    def test_main_nonexistent_file(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        """Nonexistent file should return error."""
        output_dir = tmp_path / "output"
        result = main(["/nonexistent/file.csv", "--output-dir", str(output_dir)])

        assert result == 1
        captured = capsys.readouterr()
        assert "Error" in captured.err

    def test_main_multiple_files(self, tmp_path: Path) -> None:
        """Multiple CSV files should be combined."""
        # Create two CSV files
        csv1 = tmp_path / "test1.csv"
        csv1.write_text("""run_id,game,agent_a,agent_b,agent_a_side,agent_b_side,winner,is_draw,num_moves,seed,invalid_move_retries,agent_a_latency_ms,agent_b_latency_ms,termination_reason
id1,test,alice,bob,0,1,alice,False,10,,0,0,0,normal
""")
        csv2 = tmp_path / "test2.csv"
        csv2.write_text("""run_id,game,agent_a,agent_b,agent_a_side,agent_b_side,winner,is_draw,num_moves,seed,invalid_move_retries,agent_a_latency_ms,agent_b_latency_ms,termination_reason
id2,test,bob,charlie,0,1,bob,False,10,,0,0,0,normal
""")

        output_dir = tmp_path / "output"
        result = main([str(csv1), str(csv2), "--output-dir", str(output_dir)])

        assert result == 0


# ---------------------------------------------------------------------------
# Test save_matrix_csv
# ---------------------------------------------------------------------------


class TestSaveMatrixCSV:
    """Tests for save_matrix_csv function."""

    def test_save_csv_output(self, tmp_path: Path) -> None:
        """CSV output should have correct format."""
        matrix = MatchupMatrix(game="test_game")
        matrix.add_result("alice", "bob", "win")
        matrix.add_result("bob", "charlie", "loss")

        output_path = tmp_path / "matrix.csv"
        save_matrix_csv(matrix, output_path)

        content = output_path.read_text()
        lines = content.strip().split("\n")

        # Header + 3 agents = 4 lines
        assert len(lines) == 4
        # Header should have Agent + agent names
        assert "Agent" in lines[0]
        assert "alice" in lines[0]
        assert "bob" in lines[0]
        assert "charlie" in lines[0]

    def test_save_csv_diagonal_empty(self, tmp_path: Path) -> None:
        """Diagonal cells should be empty in CSV."""
        matrix = MatchupMatrix(game="test_game")
        matrix.add_result("alice", "bob", "win")

        output_path = tmp_path / "matrix.csv"
        save_matrix_csv(matrix, output_path)

        content = output_path.read_text()
        lines = content.strip().split("\n")

        # Find alice row and check diagonal is empty
        # Agents are sorted: alice, bob
        # alice row: alice, "", "1.0000" (diagonal empty, vs bob is 1.0)
        alice_row = lines[1].split(",")
        assert alice_row[1] == ""  # Diagonal is empty


# ---------------------------------------------------------------------------
# Test expand_glob_paths
# ---------------------------------------------------------------------------


class TestExpandGlobPaths:
    """Tests for expand_glob_paths function."""

    def test_no_expansion_needed(self, tmp_path: Path) -> None:
        """Non-glob paths should pass through."""
        file1 = tmp_path / "file1.csv"
        file2 = tmp_path / "file2.csv"
        file1.touch()
        file2.touch()

        result = expand_glob_paths([str(file1), str(file2)])
        assert len(result) == 2
        assert str(file1) in result
        assert str(file2) in result

    def test_glob_expansion(self, tmp_path: Path) -> None:
        """Glob patterns should be expanded."""
        file1 = tmp_path / "file1.csv"
        file2 = tmp_path / "file2.csv"
        file1.touch()
        file2.touch()

        pattern = str(tmp_path / "*.csv")
        result = expand_glob_paths([pattern])

        assert len(result) == 2


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------


class TestIntegration:
    """Integration tests for the full workflow."""

    def test_full_workflow(self, tmp_path: Path) -> None:
        """Test the full workflow from CSV to outputs."""
        # Create test data
        csv_content = """run_id,game,agent_a,agent_b,agent_a_side,agent_b_side,winner,is_draw,num_moves,seed,invalid_move_retries,agent_a_latency_ms,agent_b_latency_ms,termination_reason
id1,breakthrough,mcts-100,random,0,1,mcts-100,False,20,,0,100,0,normal
id2,breakthrough,mcts-100,random,0,1,mcts-100,False,15,,0,100,0,normal
id3,breakthrough,random,mcts-100,0,1,mcts-100,False,25,,0,0,100,normal
id4,breakthrough,mcts-500,random,0,1,mcts-500,False,18,,0,500,0,normal
id5,breakthrough,mcts-100,mcts-500,0,1,mcts-500,False,30,,0,100,500,normal
id6,breakthrough,mcts-500,mcts-100,0,1,mcts-500,False,28,,0,500,100,normal
"""
        csv_file = tmp_path / "results.csv"
        csv_file.write_text(csv_content)

        output_dir = tmp_path / "output"

        # Run the CLI
        result = main([str(csv_file), "--output-dir", str(output_dir), "--show-samples"])

        assert result == 0
        assert (output_dir / "matchup_heatmap.png").exists()
        assert (output_dir / "matchup_matrix.csv").exists()

        # Verify CSV content
        matrix_csv = (output_dir / "matchup_matrix.csv").read_text()
        assert "mcts-100" in matrix_csv
        assert "mcts-500" in matrix_csv
        assert "random" in matrix_csv
