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
"""

from __future__ import annotations

from agents.base import BaseAgent


def round_robin(
    agents: list[BaseAgent],
    repeat: int = 1,
) -> list[tuple[BaseAgent, BaseAgent]]:
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

    pairs: list[tuple[BaseAgent, BaseAgent]] = []
    for i, a in enumerate(agents):
        for j, b in enumerate(agents):
            if i != j:
                for _ in range(repeat):
                    pairs.append((a, b))
    return pairs
