"""
analysis.loader — load persisted match results from disk
=========================================================
Reads the JSON files written by the arena into ``results/`` and returns a
list of ``MatchResult`` objects ready for rating computation or plotting.
"""

from __future__ import annotations

import json
from pathlib import Path

from arena.result import MatchResult


def load_results(results_dir: str | Path = "results/") -> list[MatchResult]:
    """Load every ``*.json`` file in *results_dir* as a ``MatchResult``.

    Parameters
    ----------
    results_dir:
        Path to the directory produced by the arena (default: ``results/``).

    Returns
    -------
    list[MatchResult]
        Sorted chronologically by ``timestamp``.

    Raises
    ------
    FileNotFoundError
        If *results_dir* does not exist.
    """
    dir_path = Path(results_dir)
    if not dir_path.exists():
        raise FileNotFoundError(f"Results directory not found: {dir_path.resolve()}")

    matches: list[MatchResult] = []
    for json_file in sorted(dir_path.glob("*.json")):
        data = json.loads(json_file.read_text())
        matches.append(MatchResult.from_dict(data))

    # Sort by played_at string (ISO-8601 sorts lexicographically)
    matches.sort(key=lambda m: m.played_at)
    return matches
