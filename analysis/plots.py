"""
analysis.plots — visualisation helpers
=======================================
Produces charts from match results and rating histories.

Requires the ``analysis`` optional-dependency group::

    pip install -e ".[analysis]"

Functions
---------
plot_rating_history
    Line chart: Elo rating over cumulative matches, one line per agent.
plot_win_rate_heatmap
    Heat-map of head-to-head win rates between every agent pair.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import matplotlib.pyplot as plt


def plot_rating_history(
    results_dir: str | Path = "results/",
    output: str | Path | None = None,
) -> None:
    """Plot Elo rating over cumulative matches for each agent.

    Parameters
    ----------
    results_dir:
        Directory of match result JSON files.
    output:
        If given, save the figure to this path instead of showing it.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for plots. "
            "Install it with: pip install -e '.[analysis]'"
        ) from exc

    from analysis.loader import load_results
    from ratings.elo import DEFAULT_RATING, update_elo

    all_results = load_results(results_dir)
    agents: set[str] = {r.agent_a for r in all_results} | {r.agent_b for r in all_results}
    history: dict[str, list[float]] = {a: [DEFAULT_RATING] for a in agents}

    for i, _ in enumerate(all_results):
        ratings = update_elo(all_results[: i + 1])
        for agent in agents:
            history[agent].append(ratings.get(agent, DEFAULT_RATING))

    fig, ax = plt.subplots(figsize=(10, 5))
    for agent, values in history.items():
        ax.plot(values, label=agent)
    ax.set_xlabel("Cumulative matches")
    ax.set_ylabel("Elo rating")
    ax.set_title("Elo rating history")
    ax.legend()
    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=150)
    else:
        plt.show()


def plot_win_rate_heatmap(
    results_dir: str | Path = "results/",
    output: str | Path | None = None,
) -> None:
    """Plot a head-to-head win-rate heat-map for all agent pairs.

    Parameters
    ----------
    results_dir:
        Directory of match result JSON files.
    output:
        If given, save the figure to this path instead of showing it.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:
        raise ImportError(
            "matplotlib and numpy are required for plots. "
            "Install them with: pip install -e '.[analysis]'"
        ) from exc

    from analysis.loader import load_results

    all_results = load_results(results_dir)
    agents = sorted({r.agent_a for r in all_results} | {r.agent_b for r in all_results})
    n = len(agents)
    idx = {a: i for i, a in enumerate(agents)}

    wins = np.zeros((n, n))
    totals = np.zeros((n, n))

    for r in all_results:
        i, j = idx[r.agent_a], idx[r.agent_b]
        totals[i, j] += 1
        outcome = r.resolved_outcome()
        if outcome == "win":
            wins[i, j] += 1
        elif outcome == "draw":
            wins[i, j] += 0.5

    with np.errstate(invalid="ignore"):
        rates = np.where(totals > 0, wins / totals, np.nan)

    fig, ax = plt.subplots(figsize=(max(5, n), max(4, n - 1)))
    im = ax.imshow(rates, vmin=0, vmax=1, cmap="RdYlGn")
    fig.colorbar(im, ax=ax, label="Win rate (row vs column)")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(agents, rotation=45, ha="right")
    ax.set_yticklabels(agents)
    ax.set_title("Head-to-head win rates")
    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=150)
    else:
        plt.show()
