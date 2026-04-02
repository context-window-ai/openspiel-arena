"""
tests.test_smoke — import-level sanity checks
===============================================
These tests confirm that every top-level package is importable and that the
core data model works, without requiring OpenSpiel or any API keys.
"""

from __future__ import annotations


def test_import_games() -> None:
    import games  # noqa: F401


def test_import_agents() -> None:
    import agents  # noqa: F401


def test_import_arena() -> None:
    import arena  # noqa: F401


def test_import_ratings() -> None:
    import ratings  # noqa: F401


def test_import_analysis() -> None:
    import analysis  # noqa: F401


def test_import_scripts() -> None:
    import scripts  # noqa: F401


def test_match_result_round_trip() -> None:
    """MatchResult serialises to dict and back without data loss."""
    from arena.result import MatchResult

    original = MatchResult(
        agent_a="agent-x",
        agent_b="agent-y",
        game_name="tic_tac_toe",
        outcome="draw",
        returns=[0.0, 0.0],
        moves=[0, 1, 2, 3, 4, 5, 6, 7, 8],
    )
    restored = MatchResult.from_dict(original.to_dict())

    assert restored.agent_a == original.agent_a
    assert restored.agent_b == original.agent_b
    assert restored.game_name == original.game_name
    assert restored.winner == original.winner
    assert restored.is_draw is True
    assert restored.returns == original.returns
    assert restored.moves == original.moves
    assert restored.match_id == original.match_id


def test_match_result_is_draw() -> None:
    from arena.result import MatchResult

    draw = MatchResult(
        agent_a="a",
        agent_b="b",
        game_name="tic_tac_toe",
        outcome="draw",
    )
    assert draw.is_draw is True
    assert draw.winner is None


def test_match_result_num_moves_auto() -> None:
    from arena.result import MatchResult

    r = MatchResult(
        agent_a="a",
        agent_b="b",
        game_name="tic_tac_toe",
        outcome="win",
        moves=[0, 1, 2],
    )
    assert r.num_moves == 3
