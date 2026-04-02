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
        Raises ``FileNotFoundError`` if the directory does not exist.

    Returns
    -------
    list[MatchResult]
        Sorted chronologically by the ``timestamp`` field.

    Raises
    ------
    FileNotFoundError
        When *results_dir* does not exist on disk.
    """
    path = Path(results_dir)
    if not path.exists():
        raise FileNotFoundError(f"Results directory not found: {path}")

    matches: list[MatchResult] = []
    for json_file in sorted(path.glob("*.json")):
        data = json.loads(json_file.read_text(encoding="utf-8"))
        matches.append(MatchResult.from_dict(data))

    # Sort chronologically; fall back to alphabetical match_id for stable sort.
    matches.sort(key=lambda r: (r.timestamp, r.match_id))
    return matches
