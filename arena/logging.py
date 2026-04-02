"""
arena.logging — canonical results logging
==========================================
Provides CSV writing for match results with a canonical schema.

The canonical schema includes:
    - run_id, game, agent_a, agent_b, agent_a_side, agent_b_side
    - winner, is_draw, num_moves, seed
    - invalid_move_retries, agent_a_latency_ms, agent_b_latency_ms
    - termination_reason
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from arena.result import MatchResult


# Canonical CSV column order for results
CANONICAL_COLUMNS = [
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


def match_result_to_row(result: "MatchResult") -> dict:
    """Convert a MatchResult to a canonical row dict for CSV export.

    Parameters
    ----------
    result:
        A MatchResult object.

    Returns
    -------
    dict
        A dictionary with canonical column names.
    """
    return {
        "run_id": result.run_id,
        "game": result.game,
        "agent_a": result.agent_a,
        "agent_b": result.agent_b,
        "agent_a_side": result.agent_a_side if result.agent_a_side is not None else 0,
        "agent_b_side": result.agent_b_side if result.agent_b_side is not None else 1,
        "winner": result.winner or "",
        "is_draw": result.is_draw,
        "num_moves": result.num_moves,
        "seed": result.seed if result.seed is not None else "",
        "invalid_move_retries": result.invalid_move_retries,
        "agent_a_latency_ms": result.agent_a_latency_ms
        if result.agent_a_latency_ms is not None
        else 0.0,
        "agent_b_latency_ms": result.agent_b_latency_ms
        if result.agent_b_latency_ms is not None
        else 0.0,
        "termination_reason": result.termination_reason,
    }


class ResultsWriter:
    """CSV writer for match results.

    Writes results to a CSV file with the canonical schema.
    The file is created on first write with headers, then rows are appended.

    Parameters
    ----------
    path:
        Path to the CSV file. Parent directories are created if needed.
    columns:
        List of column names. Defaults to CANONICAL_COLUMNS.
    append:
        If True, append to existing file without writing headers.
        If False, create new file (or overwrite existing).

    Example
    -------
    >>> writer = ResultsWriter("results/matches.csv")
    >>> writer.write(result)
    >>> writer.close()
    """

    def __init__(
        self,
        path: str | Path,
        columns: list[str] | None = None,
        append: bool = False,
    ) -> None:
        self.path = Path(path)
        self.columns = columns or CANONICAL_COLUMNS
        self._file = None
        self._writer = None
        self._append = append
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Initialize the file and writer if needed."""
        if self._initialized:
            return

        # Create parent directories if needed
        self.path.parent.mkdir(parents=True, exist_ok=True)

        # Check if file exists and has content (for append mode)
        file_exists = self.path.exists() and self.path.stat().st_size > 0

        # Open file in appropriate mode
        mode = "a" if self._append or file_exists else "w"
        self._file = open(self.path, mode, newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self.columns)

        # Write header if file is new or not appending
        if not file_exists:
            self._writer.writeheader()

        self._initialized = True

    def write(self, result: "MatchResult") -> None:
        """Write a single MatchResult to the CSV file.

        Parameters
        ----------
        result:
            The MatchResult to write.
        """
        self._ensure_initialized()
        row = match_result_to_row(result)
        # Only include columns that are in our schema
        filtered_row = {k: v for k, v in row.items() if k in self.columns}
        self._writer.writerow(filtered_row)
        # Flush to ensure data is written
        self._file.flush()

    def write_many(self, results: list["MatchResult"]) -> None:
        """Write multiple MatchResults to the CSV file.

        Parameters
        ----------
        results:
            List of MatchResults to write.
        """
        for result in results:
            self.write(result)

    def close(self) -> None:
        """Close the underlying file."""
        if self._file is not None:
            self._file.close()
            self._file = None
            self._writer = None
            self._initialized = False

    def __enter__(self) -> "ResultsWriter":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore
        """Context manager exit."""
        self.close()


def write_results_csv(
    results: list["MatchResult"],
    path: str | Path,
    columns: list[str] | None = None,
) -> None:
    """Write a list of MatchResults to a CSV file.

    Convenience function for one-time writes.

    Parameters
    ----------
    results:
        List of MatchResults to write.
    path:
        Path to the CSV file.
    columns:
        Optional list of column names. Defaults to CANONICAL_COLUMNS.
    """
    with ResultsWriter(path, columns=columns) as writer:
        writer.write_many(results)


def load_results_csv(path: str | Path) -> list[dict]:
    """Load results from a CSV file.

    Parameters
    ----------
    path:
        Path to the CSV file.

    Returns
    -------
    list[dict]
        List of row dictionaries.
    """
    path = Path(path)
    if not path.exists():
        return []

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)
