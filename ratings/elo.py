"""
ratings.elo — simple Elo rating computation
============================================
Processes a sequence of ``MatchResult`` objects and returns updated ratings
for every agent, starting from a configurable default.

References
----------
- Elo, A. E. (1978). *The Rating of Chessplayers, Past and Present*.
- https://en.wikipedia.org/wiki/Elo_rating_system
"""

from __future__ import annotations

from arena.result import MatchResult

DEFAULT_RATING: float = 1500.0
K_FACTOR: float = 32.0


def expected_score(rating_a: float, rating_b: float) -> float:
    """Return the expected score for player A against player B.

    The score is a probability in [0, 1] where 1 means certain win.

    Parameters
    ----------
    rating_a:
        Current Elo rating of player A.
    rating_b:
        Current Elo rating of player B.
    """
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def update_elo(
    results: list[MatchResult],
    initial_ratings: dict[str, float] | None = None,
    k: float = K_FACTOR,
    default: float = DEFAULT_RATING,
) -> dict[str, float]:
    """Compute Elo ratings after processing every result in order.

    Parameters
    ----------
    results:
        Match results in chronological order.
    initial_ratings:
        Seed ratings (agent name → float).  Agents not listed start at
        *default*.
    k:
        K-factor controlling how much each result shifts ratings.
    default:
        Starting rating for agents not present in *initial_ratings*.

    Returns
    -------
    dict[str, float]
        Current rating for every agent that appeared in *results*.
    """
    if not results:
        return {}

    ratings: dict[str, float] = dict(initial_ratings or {})

    def _get(name: str) -> float:
        return ratings.setdefault(name, default)

    for r in results:
        ra = _get(r.agent_a)
        rb = _get(r.agent_b)

        ea = expected_score(ra, rb)
        eb = expected_score(rb, ra)

        # Score from agent_a's perspective
        outcome = r.outcome
        if outcome == "win":
            sa, sb = 1.0, 0.0
        elif outcome == "loss":
            sa, sb = 0.0, 1.0
        else:  # draw
            sa, sb = 0.5, 0.5

        ratings[r.agent_a] = ra + k * (sa - ea)
        ratings[r.agent_b] = rb + k * (sb - eb)

    return ratings
