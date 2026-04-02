"""
arena.scheduler — match-up generators
=======================================
Produces ordered lists of ``(agent_a, agent_b)`` pairs for the tournament
runner to execute.

Available schedulers
--------------------
``round_robin(agents, repeat)``
    Every agent plays every other agent ``repeat`` times from each side.
    Total matches = ``n * (n - 1) * repeat`` where ``n = len(agents)``.

``round_robin_with_side_swap(agents, rounds_per_pairing)``
    Every unordered pair plays ``rounds_per_pairing`` matches with balanced sides.
    With ``rounds_per_pairing=2``, each pair plays once per side (side swap).
    Total matches = ``n * (n - 1) // 2 * rounds_per_pairing``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agents.base import BaseAgent


def round_robin(
    agents: "list[BaseAgent]",
    repeat: int = 1,
) -> "list[tuple[BaseAgent, BaseAgent]]":
    """Generate all ordered agent pairs for a round-robin tournament.

    Each pair ``(a, b)`` represents one match where *a* plays as player 0 and
    *b* plays as player 1.  With ``repeat=2`` every unordered pair is played
    twice in each direction (i.e. four total matches per pair).

    Parameters
    ----------
    agents:
        List of participating agents (must contain at least 2).
    repeat:
        How many times to repeat each ordered pair.

    Returns
    -------
    list[tuple[BaseAgent, BaseAgent]]
        Flat list of ``(agent_a, agent_b)`` pairs in round-robin order.

    Raises
    ------
    ValueError
        If fewer than 2 agents are provided.
    """
    if len(agents) < 2:
        raise ValueError("round_robin requires at least 2 agents")

    pairs: "list[tuple[BaseAgent, BaseAgent]]" = []
    for i, a in enumerate(agents):
        for j, b in enumerate(agents):
            if i != j:
                for _ in range(repeat):
                    pairs.append((a, b))
    return pairs


def round_robin_with_side_swap(
    agents: "list[BaseAgent]",
    rounds_per_pairing: int = 2,
) -> "list[tuple[BaseAgent, BaseAgent]]":
    """Generate balanced match-ups for a round-robin tournament with side swaps.

    Each unordered pair of agents plays ``rounds_per_pairing`` matches, with
    sides balanced as evenly as possible. For example:

    - ``rounds_per_pairing=2``: each pair plays once as (a, b) and once as (b, a)
    - ``rounds_per_pairing=4``: each pair plays twice as (a, b) and twice as (b, a)
    - ``rounds_per_pairing=3``: each pair plays twice in one direction, once in the other

    Parameters
    ----------
    agents:
        List of participating agents (must contain at least 2).
    rounds_per_pairing:
        Number of matches per unordered agent pair (default: 2 for side swap).

    Returns
    -------
    list[tuple[BaseAgent, BaseAgent]]
        Flat list of ``(agent_a, agent_b)`` pairs in round-robin order.

    Raises
    ------
    ValueError
        If fewer than 2 agents are provided.
    """
    if len(agents) < 2:
        raise ValueError("round_robin_with_side_swap requires at least 2 agents")
    if rounds_per_pairing < 1:
        raise ValueError("rounds_per_pairing must be at least 1")

    pairs: "list[tuple[BaseAgent, BaseAgent]]" = []
    n = len(agents)

    # Generate unordered pairs (i < j) and create balanced match-ups
    for i in range(n):
        for j in range(i + 1, n):
            a, b = agents[i], agents[j]

            # Calculate how many times each direction should be played
            half = rounds_per_pairing // 2
            remainder = rounds_per_pairing % 2

            # First half: (a, b) direction
            for _ in range(half + remainder):
                pairs.append((a, b))

            # Second half: (b, a) direction
            for _ in range(half):
                pairs.append((b, a))

    return pairs


def count_pairings(agents: "list[BaseAgent]") -> int:
    """Return the number of unique unordered agent pairs.

    Parameters
    ----------
    agents:
        List of participating agents.

    Returns
    -------
    int
        Number of unordered pairs: ``n * (n - 1) // 2`` where ``n = len(agents)``.
    """
    n = len(agents)
    return n * (n - 1) // 2
