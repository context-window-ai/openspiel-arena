"""
Tests for ratings.payoff_matrix module.
"""

import json
import tempfile
from pathlib import Path

import pytest

from ratings.payoff_matrix import (
    HeadToHeadRecord,
    PayoffMatrix,
    compute_payoff_matrix,
    compute_payoff_matrix_from_csv,
    load_results_from_csv,
    main,
    row_to_outcome,
)


class TestHeadToHeadRecord:
    """Tests for HeadToHeadRecord dataclass."""

    def test_empty_record(self) -> None:
        """Empty record should have zero games and win rate."""
        record = HeadToHeadRecord()
        assert record.games == 0
        assert record.win_rate == 0.0

    def test_wins_only(self) -> None:
        """Record with only wins."""
        record = HeadToHeadRecord(wins=10)
        assert record.games == 10
        assert record.win_rate == 1.0

    def test_losses_only(self) -> None:
        """Record with only losses."""
        record = HeadToHeadRecord(losses=10)
        assert record.games == 10
        assert record.win_rate == 0.0

    def test_draws_only(self) -> None:
        """Record with only draws should have 0.5 win rate."""
        record = HeadToHeadRecord(draws=10)
        assert record.games == 10
        assert record.win_rate == 0.5

    def test_mixed_record(self) -> None:
        """Mixed win/loss/draw record."""
        record = HeadToHeadRecord(wins=6, losses=2, draws=2)
        assert record.games == 10
        # Win rate: (6 + 0.5 * 2) / 10 = 0.7
        assert record.win_rate == 0.7

    def test_to_dict(self) -> None:
        """Serialization to dict."""
        record = HeadToHeadRecord(wins=3, losses=1, draws=1)
        d = record.to_dict()
        assert d["wins"] == 3
        assert d["losses"] == 1
        assert d["draws"] == 1
        assert d["games"] == 5
        assert d["win_rate"] == 0.7


class TestPayoffMatrix:
    """Tests for PayoffMatrix class."""

    def test_empty_matrix(self) -> None:
        """Empty matrix should have no agents."""
        matrix = PayoffMatrix()
        assert matrix.agents == []
        assert str(matrix) != ""

    def test_add_win_result(self) -> None:
        """Adding a win result."""
        matrix = PayoffMatrix()
        matrix.add_result("alice", "bob", "win")
        assert "alice" in matrix.agents
        assert "bob" in matrix.agents
        assert matrix.get_win_rate("alice", "bob") == 1.0
        assert matrix.get_win_rate("bob", "alice") == 0.0

    def test_add_loss_result(self) -> None:
        """Adding a loss result."""
        matrix = PayoffMatrix()
        matrix.add_result("alice", "bob", "loss")
        assert matrix.get_win_rate("alice", "bob") == 0.0
        assert matrix.get_win_rate("bob", "alice") == 1.0

    def test_add_draw_result(self) -> None:
        """Adding a draw result."""
        matrix = PayoffMatrix()
        matrix.add_result("alice", "bob", "draw")
        assert matrix.get_win_rate("alice", "bob") == 0.5
        assert matrix.get_win_rate("bob", "alice") == 0.5

    def test_multiple_results(self) -> None:
        """Multiple results between same agents."""
        matrix = PayoffMatrix()
        matrix.add_result("alice", "bob", "win")
        matrix.add_result("alice", "bob", "win")
        matrix.add_result("alice", "bob", "loss")
        # 2 wins, 1 loss = 2/3 win rate
        assert matrix.get_win_rate("alice", "bob") == pytest.approx(2 / 3)

    def test_self_play(self) -> None:
        """Self-play should return 0.5."""
        matrix = PayoffMatrix()
        assert matrix.get_win_rate("alice", "alice") == 0.5

    def test_unknown_matchup(self) -> None:
        """Unknown matchup with no data should return 0.5."""
        matrix = PayoffMatrix()
        matrix.add_result("alice", "bob", "win")
        # charlie vs dave has no data
        assert matrix.get_win_rate("charlie", "dave") == 0.5

    def test_reverse_lookup(self) -> None:
        """If only one direction recorded, reverse should be inferred."""
        matrix = PayoffMatrix()
        matrix.add_result("alice", "bob", "win")
        # bob vs alice not directly recorded, but should be 0 from alice's win
        assert matrix.get_win_rate("bob", "alice") == 0.0

    def test_to_matrix(self) -> None:
        """Conversion to 2D matrix."""
        matrix = PayoffMatrix()
        matrix.add_result("alice", "bob", "win")
        matrix.add_result("bob", "charlie", "win")
        matrix.add_result("charlie", "alice", "win")

        m = matrix.to_matrix()
        # Agents sorted alphabetically: alice, bob, charlie
        assert len(m) == 3
        assert len(m[0]) == 3
        # Diagonal should be 0.5
        assert m[0][0] == 0.5
        assert m[1][1] == 0.5
        assert m[2][2] == 0.5
        # alice beats bob
        assert m[0][1] == 1.0
        assert m[1][0] == 0.0

    def test_to_dict_and_json(self) -> None:
        """Serialization to dict and JSON."""
        matrix = PayoffMatrix(game="test_game")
        matrix.add_result("alice", "bob", "win")

        d = matrix.to_dict()
        assert d["game"] == "test_game"
        assert "alice" in d["agents"]
        assert "bob" in d["agents"]
        assert len(d["matrix"]) == 2

        # JSON should be valid
        j = matrix.to_json()
        parsed = json.loads(j)
        assert parsed["game"] == "test_game"

    def test_str_output(self) -> None:
        """String representation should be readable."""
        matrix = PayoffMatrix(game="tic_tac_toe")
        matrix.add_result("alice", "bob", "win")
        s = str(matrix)
        assert "Payoff Matrix" in s
        assert "tic_tac_toe" in s


class TestRowToOutcome:
    """Tests for row_to_outcome function."""

    def test_win_outcome(self) -> None:
        """Win when agent_a is the winner."""
        row = {"agent_a": "alice", "agent_b": "bob", "winner": "alice", "is_draw": "False"}
        agent_a, agent_b, outcome = row_to_outcome(row)
        assert agent_a == "alice"
        assert agent_b == "bob"
        assert outcome == "win"

    def test_loss_outcome(self) -> None:
        """Loss when agent_b is the winner."""
        row = {"agent_a": "alice", "agent_b": "bob", "winner": "bob", "is_draw": "False"}
        _, _, outcome = row_to_outcome(row)
        assert outcome == "loss"

    def test_draw_from_is_draw(self) -> None:
        """Draw when is_draw is True."""
        row = {"agent_a": "alice", "agent_b": "bob", "winner": "", "is_draw": "True"}
        _, _, outcome = row_to_outcome(row)
        assert outcome == "draw"

    def test_draw_from_empty_winner(self) -> None:
        """Draw when winner is empty."""
        row = {"agent_a": "alice", "agent_b": "bob", "winner": "", "is_draw": "False"}
        _, _, outcome = row_to_outcome(row)
        assert outcome == "draw"


class TestComputePayoffMatrix:
    """Tests for compute_payoff_matrix function."""

    def test_empty_results(self) -> None:
        """Empty results should produce empty matrix."""
        matrix = compute_payoff_matrix([])
        assert matrix.agents == []

    def test_single_result(self) -> None:
        """Single result should be recorded correctly."""
        results = [
            {"agent_a": "alice", "agent_b": "bob", "winner": "alice", "is_draw": "False"}
        ]
        matrix = compute_payoff_matrix(results)
        assert matrix.get_win_rate("alice", "bob") == 1.0

    def test_multiple_results(self) -> None:
        """Multiple results should aggregate correctly."""
        results = [
            {"agent_a": "alice", "agent_b": "bob", "winner": "alice", "is_draw": "False"},
            {"agent_a": "alice", "agent_b": "bob", "winner": "bob", "is_draw": "False"},
            {"agent_a": "bob", "agent_b": "charlie", "winner": "bob", "is_draw": "False"},
        ]
        matrix = compute_payoff_matrix(results)
        # alice vs bob: 1-1 = 0.5
        assert matrix.get_win_rate("alice", "bob") == 0.5
        # bob vs charlie: 1-0 = 1.0
        assert matrix.get_win_rate("bob", "charlie") == 1.0


class TestCSVLoading:
    """Tests for CSV loading functions."""

    def test_load_results_from_csv(self, tmp_path: Path) -> None:
        """Load results from a CSV file."""
        csv_content = """run_id,game,agent_a,agent_b,agent_a_side,agent_b_side,winner,is_draw,num_moves,seed,invalid_move_retries,agent_a_latency_ms,agent_b_latency_ms,termination_reason
id1,tic_tac_toe,alice,bob,0,1,alice,False,10,,0,0,0,normal
id2,tic_tac_toe,alice,bob,0,1,bob,False,12,,0,0,0,normal
"""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text(csv_content)

        results = load_results_from_csv(csv_file)
        assert len(results) == 2
        assert results[0]["agent_a"] == "alice"
        assert results[1]["winner"] == "bob"

    def test_load_nonexistent_file(self) -> None:
        """Loading a nonexistent file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_results_from_csv("/nonexistent/file.csv")

    def test_compute_from_csv(self, tmp_path: Path) -> None:
        """Compute payoff matrix directly from CSV file."""
        csv_content = """run_id,game,agent_a,agent_b,agent_a_side,agent_b_side,winner,is_draw,num_moves,seed,invalid_move_retries,agent_a_latency_ms,agent_b_latency_ms,termination_reason
id1,tic_tac_toe,alice,bob,0,1,alice,False,10,,0,0,0,normal
id2,tic_tac_toe,alice,bob,0,1,alice,False,12,,0,0,0,normal
id3,tic_tac_toe,alice,bob,0,1,bob,False,8,,0,0,0,normal
"""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text(csv_content)

        matrix = compute_payoff_matrix_from_csv(str(csv_file))
        # alice vs bob: 2 wins, 1 loss = 2/3
        assert matrix.get_win_rate("alice", "bob") == pytest.approx(2 / 3)
        assert matrix.game == "tic_tac_toe"


class TestCLI:
    """Tests for CLI functionality."""

    def test_main_help(self, capsys: pytest.CaptureFixture) -> None:
        """Help flag should print usage."""
        result = main(["--help"])
        assert result == 0
        captured = capsys.readouterr()
        assert "Usage" in captured.out

    def test_main_no_args(self, capsys: pytest.CaptureFixture) -> None:
        """No arguments should print usage."""
        result = main([])
        assert result == 0
        captured = capsys.readouterr()
        assert "Usage" in captured.out

    def test_main_nonexistent_file(self, capsys: pytest.CaptureFixture) -> None:
        """Nonexistent file should return error."""
        result = main(["/nonexistent/file.csv"])
        assert result == 1
        captured = capsys.readouterr()
        assert "Error" in captured.err

    def test_main_table_output(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        """Table output format."""
        csv_content = """run_id,game,agent_a,agent_b,agent_a_side,agent_b_side,winner,is_draw,num_moves,seed,invalid_move_retries,agent_a_latency_ms,agent_b_latency_ms,termination_reason
id1,test_game,alice,bob,0,1,alice,False,10,,0,0,0,normal
"""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text(csv_content)

        result = main([str(csv_file)])
        assert result == 0
        captured = capsys.readouterr()
        assert "Payoff Matrix" in captured.out
        assert "alice" in captured.out
        assert "bob" in captured.out

    def test_main_json_output(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        """JSON output format."""
        csv_content = """run_id,game,agent_a,agent_b,agent_a_side,agent_b_side,winner,is_draw,num_moves,seed,invalid_move_retries,agent_a_latency_ms,agent_b_latency_ms,termination_reason
id1,test_game,alice,bob,0,1,alice,False,10,,0,0,0,normal
"""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text(csv_content)

        result = main([str(csv_file), "--format", "json"])
        assert result == 0
        captured = capsys.readouterr()
        # Should be valid JSON
        parsed = json.loads(captured.out)
        assert "agents" in parsed
        assert "matrix" in parsed

    def test_main_output_to_file(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        """Output to file instead of stdout."""
        csv_content = """run_id,game,agent_a,agent_b,agent_a_side,agent_b_side,winner,is_draw,num_moves,seed,invalid_move_retries,agent_a_latency_ms,agent_b_latency_ms,termination_reason
id1,test_game,alice,bob,0,1,alice,False,10,,0,0,0,normal
"""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text(csv_content)
        output_file = tmp_path / "output.json"

        result = main([str(csv_file), "--format", "json", "--output", str(output_file)])
        assert result == 0
        captured = capsys.readouterr()
        assert "Wrote payoff matrix" in captured.out

        # Verify output file
        content = output_file.read_text()
        parsed = json.loads(content)
        assert "agents" in parsed

    def test_main_multiple_files(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        """Multiple CSV files should be combined."""
        csv1 = """run_id,game,agent_a,agent_b,agent_a_side,agent_b_side,winner,is_draw,num_moves,seed,invalid_move_retries,agent_a_latency_ms,agent_b_latency_ms,termination_reason
id1,test,alice,bob,0,1,alice,False,10,,0,0,0,normal
"""
        csv2 = """run_id,game,agent_a,agent_b,agent_a_side,agent_b_side,winner,is_draw,num_moves,seed,invalid_move_retries,agent_a_latency_ms,agent_b_latency_ms,termination_reason
id2,test,bob,charlie,0,1,bob,False,10,,0,0,0,normal
"""
        file1 = tmp_path / "test1.csv"
        file2 = tmp_path / "test2.csv"
        file1.write_text(csv1)
        file2.write_text(csv2)

        result = main([str(file1), str(file2), "--format", "json"])
        assert result == 0
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        # Should have 3 agents
        assert len(parsed["agents"]) == 3


class TestRockPaperScissors:
    """Tests using classic rock-paper-scissors scenario."""

    def test_rps_non_transitive(self) -> None:
        """Rock-Paper-Scissors should produce non-transitive win rates."""
        matrix = PayoffMatrix(game="rps")
        # Rock beats Scissors
        matrix.add_result("rock", "scissors", "win")
        matrix.add_result("rock", "scissors", "win")
        # Paper beats Rock
        matrix.add_result("paper", "rock", "win")
        matrix.add_result("paper", "rock", "win")
        # Scissors beats Paper
        matrix.add_result("scissors", "paper", "win")
        matrix.add_result("scissors", "paper", "win")

        # Verify non-transitivity
        assert matrix.get_win_rate("rock", "scissors") == 1.0
        assert matrix.get_win_rate("scissors", "paper") == 1.0
        assert matrix.get_win_rate("paper", "rock") == 1.0

        # Reverse directions
        assert matrix.get_win_rate("scissors", "rock") == 0.0
        assert matrix.get_win_rate("paper", "scissors") == 0.0
        assert matrix.get_win_rate("rock", "paper") == 0.0
