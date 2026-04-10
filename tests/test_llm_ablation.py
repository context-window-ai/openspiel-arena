"""
tests.test_llm_ablation — tests for scripts.plot_llm_ablation
==============================================================
Covers parsing, metric computation, chart generation, and CLI.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from scripts.plot_llm_ablation import (
    VariantMetrics,
    compute_metrics,
    compute_metrics_by_opponent,
    load_csvs,
    parse_memory_turns,
    parse_prompt_style,
    plot_invalid_rate_chart,
    plot_latency_chart,
    plot_winrate_chart,
    save_metrics_csv,
    main,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ablation_csv(tmp_path: Path) -> Path:
    """Create a sample CSV with LLM ablation variants."""
    csv_content = """run_id,game,agent_a,agent_b,agent_a_side,agent_b_side,winner,is_draw,num_moves,seed,invalid_move_retries,agent_a_latency_ms,agent_b_latency_ms,termination_reason
id1,tic_tac_toe,llm-gpt-5.4-mini-mem0,random,0,1,llm-gpt-5.4-mini-mem0,False,7,,0,500,0,normal
id2,tic_tac_toe,llm-gpt-5.4-mini-mem0,random,0,1,random,False,9,,1,800,0,normal
id3,tic_tac_toe,llm-gpt-5.4-mini-mem0,mcts-50,0,1,mcts-50,False,8,,0,600,50,normal
id4,tic_tac_toe,llm-gpt-5.4-mini-mem1,random,0,1,llm-gpt-5.4-mini-mem1,False,6,,0,450,0,normal
id5,tic_tac_toe,llm-gpt-5.4-mini-mem1,random,0,1,llm-gpt-5.4-mini-mem1,False,7,,0,470,0,normal
id6,tic_tac_toe,llm-gpt-5.4-mini-mem1,mcts-50,0,1,mcts-50,False,8,,1,520,55,normal
id7,tic_tac_toe,llm-gpt-5.4-mini-mem3,random,0,1,llm-gpt-5.4-mini-mem3,False,5,,0,430,0,normal
id8,tic_tac_toe,llm-gpt-5.4-mini-mem3,random,0,1,,True,9,,2,900,0,normal
id9,tic_tac_toe,llm-gpt-5.4-mini-mem3,mcts-50,0,1,mcts-50,False,7,,0,480,60,normal
id10,tic_tac_toe,random,mcts-50,0,1,mcts-50,False,10,,0,0,50,normal
"""
    csv_file = tmp_path / "results_ablation.csv"
    csv_file.write_text(csv_content)
    return csv_file


@pytest.fixture
def ablation_df(ablation_csv: Path) -> pd.DataFrame:
    """Load the ablation CSV into a DataFrame."""
    return pd.read_csv(ablation_csv)


@pytest.fixture
def sample_metrics() -> list[VariantMetrics]:
    """Create sample VariantMetrics for plotting tests."""
    return [
        VariantMetrics("llm-gpt-5.4-mini-mem0", 0, "board_summary_then_choice", 4, 0.25, 0.25, 633.33),
        VariantMetrics("llm-gpt-5.4-mini-mem1", 1, "board_summary_then_choice", 4, 0.625, 0.25, 480.0),
        VariantMetrics("llm-gpt-5.4-mini-mem3", 3, "board_summary_then_choice", 4, 0.375, 0.67, 603.33),
    ]


# ---------------------------------------------------------------------------
# Test parsing helpers
# ---------------------------------------------------------------------------


class TestParseMemoryTurns:
    """Tests for parse_memory_turns."""

    def test_mem0(self) -> None:
        assert parse_memory_turns("llm-gpt-5.4-mini-mem0") == 0

    def test_mem1(self) -> None:
        assert parse_memory_turns("llm-gpt-5.4-mini-mem1") == 1

    def test_mem3(self) -> None:
        assert parse_memory_turns("llm-gpt-5.4-mini-mem3") == 3

    def test_no_mem_token_defaults_to_1(self) -> None:
        assert parse_memory_turns("llm-gpt-5.4-mini") == 1

    def test_mem_token_at_end(self) -> None:
        assert parse_memory_turns("llm-gpt-5.4-mini-mem5") == 5

    def test_mem_token_in_middle(self) -> None:
        assert parse_memory_turns("llm-gpt-5.4-mini-mem3-reason") == 3


class TestParsePromptStyle:
    """Tests for parse_prompt_style."""

    def test_unknown_defaults(self) -> None:
        assert parse_prompt_style("llm-gpt-5.4-mini-mem0") == "board_summary_then_choice"

    def test_zero_shot(self) -> None:
        assert parse_prompt_style("llm-zero_shot-mem0") == "zero_shot"

    def test_reason_then_choice(self) -> None:
        assert parse_prompt_style("llm-reason_then_choice-mem1") == "reason_then_choice"


# ---------------------------------------------------------------------------
# Test metric computation
# ---------------------------------------------------------------------------


class TestComputeMetrics:
    """Tests for compute_metrics function."""

    def test_finds_llm_variants(self, ablation_df: pd.DataFrame) -> None:
        metrics = compute_metrics(ablation_df)
        variants = {m.variant for m in metrics}
        assert "llm-gpt-5.4-mini-mem0" in variants
        assert "llm-gpt-5.4-mini-mem1" in variants
        assert "llm-gpt-5.4-mini-mem3" in variants

    def test_game_counts(self, ablation_df: pd.DataFrame) -> None:
        metrics = compute_metrics(ablation_df)
        by_variant = {m.variant: m for m in metrics}
        # mem0: rows id1, id2 (agent_a), id3 (agent_a) = 3
        assert by_variant["llm-gpt-5.4-mini-mem0"].games == 3
        # mem1: rows id4, id5, id6 = 3
        assert by_variant["llm-gpt-5.4-mini-mem1"].games == 3
        # mem3: rows id7, id8, id9 = 3
        assert by_variant["llm-gpt-5.4-mini-mem3"].games == 3

    def test_win_rate_calculation(self, ablation_df: pd.DataFrame) -> None:
        metrics = compute_metrics(ablation_df)
        by_variant = {m.variant: m for m in metrics}
        # mem0: 1 win (id1) + 0 draws = 1/3 ≈ 0.333
        assert by_variant["llm-gpt-5.4-mini-mem0"].win_rate == pytest.approx(1 / 3)
        # mem1: 2 wins (id4, id5) = 2/3 ≈ 0.667
        assert by_variant["llm-gpt-5.4-mini-mem1"].win_rate == pytest.approx(2 / 3)
        # mem3: 1 win (id7) + 0.5 draw (id8) = 1.5/3 = 0.5
        assert by_variant["llm-gpt-5.4-mini-mem3"].win_rate == pytest.approx(0.5)

    def test_invalid_retries(self, ablation_df: pd.DataFrame) -> None:
        metrics = compute_metrics(ablation_df)
        by_variant = {m.variant: m for m in metrics}
        # mem0: retries [0, 1, 0] → avg ≈ 0.333
        assert by_variant["llm-gpt-5.4-mini-mem0"].avg_invalid_retries == pytest.approx(1 / 3)

    def test_latency(self, ablation_df: pd.DataFrame) -> None:
        metrics = compute_metrics(ablation_df)
        by_variant = {m.variant: m for m in metrics}
        # mem1: latencies [450, 470, 520] → avg = 480
        assert by_variant["llm-gpt-5.4-mini-mem1"].avg_latency_ms == pytest.approx(480.0)

    def test_excludes_non_llm(self, ablation_df: pd.DataFrame) -> None:
        metrics = compute_metrics(ablation_df)
        variants = {m.variant for m in metrics}
        assert "random" not in variants
        assert "mcts-50" not in variants


class TestComputeMetricsByOpponent:
    """Tests for compute_metrics_by_opponent."""

    def test_groups_by_opponent(self, ablation_df: pd.DataFrame) -> None:
        result = compute_metrics_by_opponent(ablation_df)
        assert "random" in result
        assert "mcts-50" in result

    def test_random_opponent_metrics(self, ablation_df: pd.DataFrame) -> None:
        result = compute_metrics_by_opponent(ablation_df)
        # Against random: mem0=2 games, mem1=2 games, mem3=2 games
        random_metrics = {m.variant: m for m in result["random"]}
        assert "llm-gpt-5.4-mini-mem0" in random_metrics
        assert random_metrics["llm-gpt-5.4-mini-mem0"].games == 2


# ---------------------------------------------------------------------------
# Test CSV loading
# ---------------------------------------------------------------------------


class TestLoadCSVs:
    """Tests for load_csvs."""

    def test_load_single_file(self, ablation_csv: Path) -> None:
        df = load_csvs([ablation_csv])
        assert len(df) == 10

    def test_load_directory(self, tmp_path: Path) -> None:
        csv_content = """run_id,game,agent_a,agent_b,agent_a_side,agent_b_side,winner,is_draw,num_moves,seed,invalid_move_retries,agent_a_latency_ms,agent_b_latency_ms,termination_reason
id1,tic_tac_toe,a,b,0,1,a,False,5,,0,0,0,normal
"""
        (tmp_path / "results_test.csv").write_text(csv_content)
        df = load_csvs([tmp_path])
        assert len(df) == 1

    def test_nonexistent_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_csvs(["/nonexistent/path.csv"])


# ---------------------------------------------------------------------------
# Test plotting
# ---------------------------------------------------------------------------


class TestPlotting:
    """Tests for plot generation (files are created)."""

    def test_plot_winrate_chart(self, tmp_path: Path, sample_metrics: list[VariantMetrics]) -> None:
        metrics_by_opp = {"random": sample_metrics, "mcts-50": sample_metrics}
        output = tmp_path / "winrate.png"
        plot_winrate_chart(metrics_by_opp, output)
        assert output.exists()
        assert output.stat().st_size > 0

    def test_plot_invalid_rate_chart(self, tmp_path: Path, sample_metrics: list[VariantMetrics]) -> None:
        output = tmp_path / "invalid.png"
        plot_invalid_rate_chart(sample_metrics, output)
        assert output.exists()
        assert output.stat().st_size > 0

    def test_plot_latency_chart(self, tmp_path: Path, sample_metrics: list[VariantMetrics]) -> None:
        output = tmp_path / "latency.png"
        plot_latency_chart(sample_metrics, output)
        assert output.exists()
        assert output.stat().st_size > 0

    def test_plot_empty_metrics(self, tmp_path: Path) -> None:
        output = tmp_path / "empty.png"
        plot_invalid_rate_chart([], output)
        assert output.exists()


# ---------------------------------------------------------------------------
# Test CSV export
# ---------------------------------------------------------------------------


class TestSaveMetricsCSV:
    """Tests for save_metrics_csv."""

    def test_csv_format(self, tmp_path: Path, sample_metrics: list[VariantMetrics]) -> None:
        output = tmp_path / "metrics.csv"
        save_metrics_csv(sample_metrics, output)
        assert output.exists()
        df = pd.read_csv(output)
        assert len(df) == 3
        assert list(df.columns) == [
            "variant", "memory_turns", "prompt_style", "games",
            "win_rate", "avg_invalid_retries", "avg_latency_ms",
        ]

    def test_csv_values(self, tmp_path: Path, sample_metrics: list[VariantMetrics]) -> None:
        output = tmp_path / "metrics.csv"
        save_metrics_csv(sample_metrics, output)
        df = pd.read_csv(output)
        row0 = df.iloc[0]
        assert row0["variant"] == "llm-gpt-5.4-mini-mem0"
        assert row0["memory_turns"] == 0
        assert row0["games"] == 4

    def test_creates_parent_dirs(self, tmp_path: Path, sample_metrics: list[VariantMetrics]) -> None:
        output = tmp_path / "nested" / "dir" / "metrics.csv"
        save_metrics_csv(sample_metrics, output)
        assert output.exists()


# ---------------------------------------------------------------------------
# Test CLI
# ---------------------------------------------------------------------------


class TestCLI:
    """Tests for the CLI entry-point."""

    def test_main_basic(self, ablation_csv: Path, tmp_path: Path) -> None:
        from click.testing import CliRunner

        output_dir = tmp_path / "output"
        runner = CliRunner()
        result = runner.invoke(main, [str(ablation_csv), "--output-dir", str(output_dir)])
        assert result.exit_code == 0
        assert (output_dir / "llm_ablation_winrate.png").exists()
        assert (output_dir / "llm_ablation_invalid_rate.png").exists()
        assert (output_dir / "llm_ablation_latency.png").exists()
        assert (output_dir / "llm_ablation_metrics.csv").exists()

    def test_main_no_llm_agents(self, tmp_path: Path) -> None:
        from click.testing import CliRunner

        csv_content = """run_id,game,agent_a,agent_b,agent_a_side,agent_b_side,winner,is_draw,num_moves,seed,invalid_move_retries,agent_a_latency_ms,agent_b_latency_ms,termination_reason
id1,tic_tac_toe,random,mcts-50,0,1,random,False,5,,0,0,50,normal
"""
        csv_file = tmp_path / "no_llm.csv"
        csv_file.write_text(csv_content)
        output_dir = tmp_path / "output"
        runner = CliRunner()
        result = runner.invoke(main, [str(csv_file), "--output-dir", str(output_dir)])
        assert result.exit_code == 1

    def test_main_nonexistent_file(self, tmp_path: Path) -> None:
        from click.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(main, ["/nonexistent/file.csv", "--output-dir", str(tmp_path / "out")])
        assert result.exit_code == 1

    def test_main_directory_input(self, ablation_csv: Path, tmp_path: Path) -> None:
        """Should accept a directory containing results CSVs."""
        from click.testing import CliRunner
        import shutil

        # Copy the csv into a subdirectory
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        shutil.copy(ablation_csv, results_dir / "results_ablation.csv")
        output_dir = tmp_path / "output"
        runner = CliRunner()
        result = runner.invoke(main, [str(results_dir), "--output-dir", str(output_dir)])
        assert result.exit_code == 0
