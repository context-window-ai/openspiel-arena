"""
ratings.glicko2 — Glicko-2 rating system
==========================================
Adds rating deviation (RD) and volatility on top of Elo for more
statistically meaningful uncertainty estimates.

All internal computation uses the Glicko-2 scale (μ, φ, σ).  The
``display_rating`` and ``display_rd`` properties convert back to the
familiar 1–3000 Elo-like scale for human consumption.

References
----------
- Glickman, M. E. (2012). *Example of the Glicko-2 system*.
  http://www.glicko.net/glicko/glicko2.pdf
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from arena.result import MatchResult

# ---------------------------------------------------------------------------
# Public constants (display scale, same as traditional Elo)
# ---------------------------------------------------------------------------
DEFAULT_DISPLAY_RATING: float = 1500.0
DEFAULT_DISPLAY_RD: float = 350.0

# ---------------------------------------------------------------------------
# Internal Glicko-2 constants
# ---------------------------------------------------------------------------
_SCALE: float = 173.7178   # converts between display and internal scale
_DEFAULT_MU: float = 0.0   # 1500 on display scale
_DEFAULT_PHI: float = DEFAULT_DISPLAY_RD / _SCALE   # ≈ 2.015
_DEFAULT_SIGMA: float = 0.06       # initial volatility
_TAU: float = 0.5                  # system constant (controls vol. change)
_EPSILON: float = 1e-6             # convergence tolerance


# ---------------------------------------------------------------------------
# Glicko-2 Rating record
# ---------------------------------------------------------------------------

@dataclass
class Glicko2Rating:
    """Internal Glicko-2 rating for a single agent.

    Attributes
    ----------
    mu:
        Rating on the Glicko-2 internal scale (centre = 0).
    phi:
        Rating deviation on the internal scale.
    sigma:
        Rating volatility.
    """

    mu: float = _DEFAULT_MU
    phi: float = _DEFAULT_PHI
    sigma: float = _DEFAULT_SIGMA

    # ------------------------------------------------------------------
    # Class factories
    # ------------------------------------------------------------------

    @classmethod
    def default(cls) -> "Glicko2Rating":
        """Return a fresh rating with default (new-player) values."""
        return cls(mu=_DEFAULT_MU, phi=_DEFAULT_PHI, sigma=_DEFAULT_SIGMA)

    # ------------------------------------------------------------------
    # Display-scale properties
    # ------------------------------------------------------------------

    @property
    def display_rating(self) -> float:
        """Rating in the traditional 1500-centre display scale."""
        return self.mu * _SCALE + 1500.0

    @property
    def display_rd(self) -> float:
        """Rating deviation in the display scale."""
        return self.phi * _SCALE

    def __repr__(self) -> str:
        return (
            f"Glicko2Rating(display_rating={self.display_rating:.1f},"
            f" display_rd={self.display_rd:.1f}, sigma={self.sigma:.4f})"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _g(phi: float) -> float:
    """Glicko-2 g function."""
    return 1.0 / math.sqrt(1.0 + 3.0 * phi ** 2 / math.pi ** 2)


def _E(mu: float, mu_j: float, phi_j: float) -> float:
    """Expected score for a player with rating *mu* against opponent (mu_j, phi_j)."""
    return 1.0 / (1.0 + math.exp(-_g(phi_j) * (mu - mu_j)))


def _update_sigma(
    phi: float,
    sigma: float,
    delta: float,
    v: float,
    tau: float = _TAU,
    epsilon: float = _EPSILON,
) -> float:
    """Iterative algorithm (Illinois method) to update volatility σ."""
    a = math.log(sigma ** 2)
    phi2 = phi ** 2

    def _f(x: float) -> float:
        ex = math.exp(x)
        tmp = phi2 + v + ex
        return (
            ex * (delta ** 2 - phi2 - v - ex) / (2 * tmp ** 2)
            - (x - a) / (tau ** 2)
        )

    # Bracket
    A = a
    if delta ** 2 > phi2 + v:
        B = math.log(delta ** 2 - phi2 - v)
    else:
        k = 1
        while _f(a - k * tau) < 0:
            k += 1
        B = a - k * tau

    fA, fB = _f(A), _f(B)
    while abs(B - A) > epsilon:
        C = A + (A - B) * fA / (fB - fA)
        fC = _f(C)
        if fC * fB < 0:
            A, fA = B, fB
        else:
            fA /= 2
        B, fB = C, fC

    return math.exp(A / 2)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def update_glicko2(
    results: list[MatchResult],
    initial_ratings: dict[str, Glicko2Rating] | None = None,
) -> dict[str, Glicko2Rating]:
    """Compute Glicko-2 ratings after processing every result in order.

    All results are treated as a single rating period.

    Parameters
    ----------
    results:
        Match results in chronological order.
    initial_ratings:
        Seed ratings (agent name → ``Glicko2Rating``).  Agents not listed
        start with default values.

    Returns
    -------
    dict[str, Glicko2Rating]
        Updated rating for every agent that appeared in *results*.
    """
    if not results:
        return {}

    ratings: dict[str, Glicko2Rating] = dict(initial_ratings or {})

    def _get(name: str) -> Glicko2Rating:
        if name not in ratings:
            ratings[name] = Glicko2Rating.default()
        return ratings[name]

    # Ensure every agent has an entry before computing.
    for r in results:
        _get(r.agent_a)
        _get(r.agent_b)

    # Collect outcomes per agent: list of (mu_j, phi_j, score)
    games: dict[str, list[tuple[float, float, float]]] = {name: [] for name in ratings}

    for r in results:
        ra = ratings[r.agent_a]
        rb = ratings[r.agent_b]
        if r.outcome == "win":
            sa, sb = 1.0, 0.0
        elif r.outcome == "loss":
            sa, sb = 0.0, 1.0
        else:
            sa, sb = 0.5, 0.5
        games[r.agent_a].append((rb.mu, rb.phi, sa))
        games[r.agent_b].append((ra.mu, ra.phi, sb))

    # Update each agent using the Glicko-2 algorithm.
    new_ratings: dict[str, Glicko2Rating] = {}
    for name, old in ratings.items():
        player_games = games.get(name, [])
        if not player_games:
            # No games this period: φ increases (uncertainty grows), μ unchanged.
            phi_star = math.sqrt(old.phi ** 2 + old.sigma ** 2)
            new_ratings[name] = Glicko2Rating(mu=old.mu, phi=phi_star, sigma=old.sigma)
            continue

        # Step 3: compute estimated variance v
        v_inv = sum(
            _g(phi_j) ** 2 * _E(old.mu, mu_j, phi_j) * (1 - _E(old.mu, mu_j, phi_j))
            for mu_j, phi_j, _ in player_games
        )
        v = 1.0 / v_inv

        # Step 4: compute improvement estimate Δ
        delta = v * sum(
            _g(phi_j) * (s_j - _E(old.mu, mu_j, phi_j))
            for mu_j, phi_j, s_j in player_games
        )

        # Step 5: update volatility
        new_sigma = _update_sigma(old.phi, old.sigma, delta, v)

        # Step 6: update φ* (pre-rating-period deviation)
        phi_star = math.sqrt(old.phi ** 2 + new_sigma ** 2)

        # Step 7: update φ'
        new_phi = 1.0 / math.sqrt(1.0 / phi_star ** 2 + 1.0 / v)

        # Step 8: update μ'
        new_mu = old.mu + new_phi ** 2 * sum(
            _g(phi_j) * (s_j - _E(old.mu, mu_j, phi_j))
            for mu_j, phi_j, s_j in player_games
        )

        new_ratings[name] = Glicko2Rating(mu=new_mu, phi=new_phi, sigma=new_sigma)

    return new_ratings
