"""
Tests for analysis.loader — load_results.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from analysis.loader import load_results
from arena.result import MatchResult


def _write_result(directory: Path, result: MatchResult) -> None:
    (directory / f"{result.match_id}.json").write_text(
        json.dumps(result.to_dict())
    )


def test_load_results_returns_list(sample_match_result):
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_result(Path(tmpdir), sample_match_result)
        results = load_results(tmpdir)
        assert isinstance(results, list)
        assert len(results) == 1


def test_load_results_roundtrip(sample_match_result):
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_result(Path(tmpdir), sample_match_result)
        results = load_results(tmpdir)
        r = results[0]
        assert r.match_id == sample_match_result.match_id
        assert r.outcome == sample_match_result.outcome


def test_load_results_empty_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        results = load_results(tmpdir)
        assert results == []


def test_load_results_missing_dir():
    with pytest.raises(FileNotFoundError):
        load_results("/nonexistent/path/that/does/not/exist")


def test_load_results_sorted_by_timestamp():
    with tempfile.TemporaryDirectory() as tmpdir:
        r1 = MatchResult(
            agent_a="a", agent_b="b", game_name="g", outcome="win",
            timestamp="2024-01-01T00:00:00+00:00",
        )
        r2 = MatchResult(
            agent_a="a", agent_b="b", game_name="g", outcome="loss",
            timestamp="2024-06-01T00:00:00+00:00",
        )
        # Write in reverse order to confirm sort is by timestamp, not filename.
        _write_result(Path(tmpdir), r2)
        _write_result(Path(tmpdir), r1)
        results = load_results(tmpdir)
        assert results[0].match_id == r1.match_id
        assert results[1].match_id == r2.match_id
