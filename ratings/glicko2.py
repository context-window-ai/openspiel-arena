"""
ratings.glicko2 — Glicko-2 rating system
==========================================
Adds rating deviation (RD) and volatility on top of Elo for more
statistically meaningful uncertainty estimates.

References
----------
- Glickman, M. E. (2012). *Example of the Glicko-2 system*.
  http://www.glicko.net/glicko/glicko2.pdf
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from arena.result import MatchResult

# Glicko-2 system constants
_Q = math.log(10) / 400  # conversion constant ≈ 0.005756
_DEFAULT_MU = 1500.0       # initial rating (same scale as Elo)
_DEFAULT_PHI = 350.0       # initial rating deviation (RD)
_DEFAULT_SIGMA = 0.06      # initial volatility
_TAU = 0.5                  # system constant controlling volatility change
_EPSILON = 0.000001         # convergence tolerance for volatility update


@dataclass
class Glicko2Rating:
    """Rating record for a single agent under the Glicko-2 system.

    Attributes
    ----------
    mu:
        Current rating (same 1500-scale as Elo).
    phi:
        Rating deviation — smaller means more certainty.
    sigma:
        Rating volatility — reflects consistency of performance.
    """

    mu: float = _DEFAULT_MU
    phi: float = _DEFAULT_PHI
    sigma: float = _DEFAULT_SIGMA

    def __repr__(self) -> str:
        return (
            f"Glicko2Rating(mu={self.mu:.1f}, phi={self.phi:.1f},"
            f" sigma={self.sigma:.4f})"
        )


def update_glicko2(
    results: list[MatchResult],
    initial_ratings: dict[str, Glicko2Rating] | None = None,
) -> dict[str, Glicko2Rating]:
    """Compute Glicko-2 ratings after processing every result in order.

    All results are treated as a single rating period (the simplest
    possible approach).  A proper implementation would batch results into
    rating periods.

    Parameters
    ----------
    results:
        Match results in chronological order.
    initial_ratings:
        Seed ratings.  Agents not present start with default values.

    Returns
    -------
    dict[str, Glicko2Rating]
        Updated rating for every agent that appeared in *results*.
    """
    # TODO: implement Glicko-2 update algorithm.
    #
    # Outline (see http://www.glicko.net/glicko/glicko2.pdf):
    # 1. Convert ratings to Glicko-2 scale: mu' = (mu - 1500)/173.7178
    # 2. For each player, collect opponents' scaled ratings and outcomes.
    # 3. Compute the estimated variance v.
    # 4. Compute the estimated improvement Delta.
    # 5. Update volatility sigma' via iterative algorithm (Illinois method).
    # 6. Update phi* = sqrt(phi^2 + sigma'^2).
    # 7. Update phi' = 1 / sqrt(1/phi*^2 + 1/v).
    # 8. Update mu' = mu + phi'^2 * sum(g(phi_j) * (s_j - E_j)).
    # 9. Convert back to 1-400 scale.
    raise NotImplementedError("ratings.glicko2.update_glicko2 is not yet implemented")
