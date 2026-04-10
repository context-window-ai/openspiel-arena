"""
scripts.plot_llm_ablation — ablation charts for memory_turns and prompt_style
==============================================================================
Reads results CSV files from the arena, filters to LLM agent variants, and
produces charts comparing win-rate, invalid-move rate, and latency across
memory-turn and prompt-style ablations.

Usage
-----
::

    python scripts/plot_llm_ablation.py results/*.csv
    python scripts/plot_llm_ablation.py results/ --output-dir output/

Deliverables
------------
- output/llm_ablation_winrate.png
- output/llm_ablation_invalid_rate.png
- output/llm_ablation_latency.png
- output/llm_ablation_metrics.csv
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import click
import pandas as pd


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VariantMetrics:
    """Aggregated metrics for a single LLM variant.

    Attributes
    ----------
    variant:
        Agent name (e.g. ``"llm-gpt-5.4-mini-mem0"``).
    memory_turns:
        Parsed memory_turns value from the agent name.
    prompt_style:
        Parsed prompt_style string (or ``"unknown"``).
    games:
        Number of games this agent played.
    win_rate:
        Fraction of games won (draws count as 0.5).
    avg_invalid_retries:
        Mean ``invalid_move_retries`` across all games.
    avg_latency_ms:
        Mean latency in milliseconds for this agent across all games.
    """

    variant: str
    memory_turns: int
    prompt_style: str
    games: int
    win_rate: float
    avg_invalid_retries: float
    avg_latency_ms: float


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

# Pattern to extract memory_turns from agent name like "llm-gpt-5.4-mini-mem3"
_MEM_RE = re.compile(r"-mem(\d+)(?:-|$)")


def parse_memory_turns(agent_name: str) -> int:
    """Extract memory_turns integer from an LLM agent name.

    Falls back to 1 if no ``-memN`` token is found.

    Parameters
    ----------
    agent_name:
        The agent name string.

    Returns
    -------
    int
        Parsed memory_turns value.
    """
    m = _MEM_RE.search(agent_name)
    return int(m.group(1)) if m else 1


def parse_prompt_style(agent_name: str) -> str:
    """Best-effort extraction of prompt style from the agent name.

    Since the current naming convention doesn't encode prompt_style, this
    returns ``"unknown"`` unless the name contains a recognisable token.

    Parameters
    ----------
    agent_name:
        The agent name string.

    Returns
    -------
    str
        A prompt style label.
    """
    for style in (
        "zero_shot",
        "legal_moves_only",
        "board_summary_then_choice",
        "reason_then_choice",
        "critic_then_choice",
    ):
        if style in agent_name:
            return style
    return "board_summary_then_choice"


def is_llm_agent(name: str) -> bool:
    """Return ``True`` if *name* looks like an LLM agent variant.

    Parameters
    ----------
    name:
        Agent name string.

    Returns
    -------
    bool
    """
    return name.startswith("llm-")


def _is_llm_variant(name: str) -> bool:
    """Check if agent name indicates an LLM variant (including prompt-style markers)."""
    return name.startswith("llm-")


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------


def load_csvs(paths: Sequence[str | Path]) -> pd.DataFrame:
    """Load and concatenate one or more result CSV files.

    Parameters
    ----------
    paths:
        File paths or glob patterns.

    Returns
    -------
    pd.DataFrame
        Concatenated DataFrame with standardised columns.

    Raises
    ------
    FileNotFoundError
        If none of the paths resolve to existing files.
    """
    frames: list[pd.DataFrame] = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            frames.extend(_load_dir(path))
        elif path.is_file():
            frames.append(pd.read_csv(path))
        else:
            # Try glob
            parent = path.parent
            pattern = path.name
            matched = sorted(parent.glob(pattern))
            if not matched:
                raise FileNotFoundError(f"No files match: {p}")
            for f in matched:
                frames.append(pd.read_csv(f))

    if not frames:
        raise FileNotFoundError("No CSV files found")

    return pd.concat(frames, ignore_index=True)


def _load_dir(directory: Path) -> list[pd.DataFrame]:
    """Load all ``results_*.csv`` files in *directory*."""
    frames: list[pd.DataFrame] = []
    for csv_file in sorted(directory.glob("results_*.csv")):
        frames.append(pd.read_csv(csv_file))
    return frames


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------


def compute_metrics(df: pd.DataFrame) -> list[VariantMetrics]:
    """Compute per-variant ablation metrics from a results DataFrame.

    Each LLM agent variant is identified by its ``agent_a`` / ``agent_b``
    name.  The function looks at every row where the LLM agent participated
    (as either side) and aggregates wins, invalid retries, and latency.

    Parameters
    ----------
    df:
        Results DataFrame with at least the columns: ``agent_a``, ``agent_b``,
        ``winner``, ``is_draw``, ``invalid_move_retries``,
        ``agent_a_latency_ms``, ``agent_b_latency_ms``.

    Returns
    -------
    list[VariantMetrics]
        One entry per unique LLM agent variant, sorted by variant name.
    """
    # Collect all LLM agent names
    all_agents: set[str] = set()
    for col in ("agent_a", "agent_b"):
        for val in df[col].unique():
            if _is_llm_variant(str(val)):
                all_agents.add(str(val))

    metrics: list[VariantMetrics] = []

    for agent in sorted(all_agents):
        wins = 0
        games = 0
        invalid_retries: list[float] = []
        latencies: list[float] = []

        for _, row in df.iterrows():
            a = str(row["agent_a"])
            b = str(row["agent_b"])
            winner = str(row.get("winner", ""))
            is_draw = str(row.get("is_draw", "False")).lower() == "true"

            if a == agent:
                games += 1
                # Win from agent_a perspective
                if winner == agent:
                    wins += 1
                elif is_draw:
                    wins += 0.5
                invalid_retries.append(float(row.get("invalid_move_retries", 0)))
                lat = row.get("agent_a_latency_ms")
                if lat is not None and not pd.isna(lat):
                    latencies.append(float(lat))

            elif b == agent:
                games += 1
                # Win from agent_b perspective
                if winner == agent:
                    wins += 1
                elif is_draw:
                    wins += 0.5
                invalid_retries.append(float(row.get("invalid_move_retries", 0)))
                lat = row.get("agent_b_latency_ms")
                if lat is not None and not pd.isna(lat):
                    latencies.append(float(lat))

        win_rate = wins / games if games > 0 else 0.0
        avg_invalid = sum(invalid_retries) / len(invalid_retries) if invalid_retries else 0.0
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

        metrics.append(
            VariantMetrics(
                variant=agent,
                memory_turns=parse_memory_turns(agent),
                prompt_style=parse_prompt_style(agent),
                games=games,
                win_rate=win_rate,
                avg_invalid_retries=avg_invalid,
                avg_latency_ms=avg_latency,
            )
        )

    return metrics


def compute_metrics_by_opponent(
    df: pd.DataFrame,
) -> dict[str, list[VariantMetrics]]:
    """Compute per-variant metrics grouped by opponent type.

    Parameters
    ----------
    df:
        Results DataFrame.

    Returns
    -------
    dict[str, list[VariantMetrics]]
        Mapping from opponent name to list of variant metrics.
    """
    opponents: set[str] = set()
    for col in ("agent_a", "agent_b"):
        for val in df[col].unique():
            v = str(val)
            if not _is_llm_variant(v):
                opponents.add(v)

    result: dict[str, list[VariantMetrics]] = {}
    for opp in sorted(opponents):
        mask = (df["agent_a"] == opp) | (df["agent_b"] == opp)
        subset = df.loc[mask]
        if len(subset) > 0:
            result[opp] = compute_metrics(subset)

    return result


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _ensure_matplotlib() -> None:
    """Import matplotlib; raise with a helpful message if missing."""
    try:
        import matplotlib  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for plots. "
            "Install it with: pip install -e '.[analysis]'"
        ) from exc


def plot_winrate_chart(
    metrics_by_opp: dict[str, list[VariantMetrics]],
    output: Path,
) -> None:
    """Produce a grouped bar chart of win-rate by memory_turns, grouped by opponent.

    Parameters
    ----------
    metrics_by_opp:
        Per-opponent variant metrics.
    output:
        Path to write the PNG file.
    """
    _ensure_matplotlib()
    import matplotlib.pyplot as plt
    import numpy as np

    output.parent.mkdir(parents=True, exist_ok=True)

    # Collect unique variants across all opponents
    all_variants: list[str] = []
    seen: set[str] = set()
    for _, mlist in metrics_by_opp.items():
        for m in mlist:
            if m.variant not in seen:
                all_variants.append(m.variant)
                seen.add(m.variant)

    if not all_variants:
        return

    opponents = sorted(metrics_by_opp.keys())
    n_variants = len(all_variants)
    n_opponents = len(opponents)

    fig, ax = plt.subplots(figsize=(19.2, 10.8), dpi=100)

    x = np.arange(n_variants)
    width = 0.8 / max(n_opponents, 1)
    colours = plt.cm.Set2(np.linspace(0, 1, max(n_opponents, 1)))

    for i, opp in enumerate(opponents):
        opp_metrics = {m.variant: m for m in metrics_by_opp[opp]}
        rates = [opp_metrics[v].win_rate if v in opp_metrics else 0.0 for v in all_variants]
        offset = (i - n_opponents / 2 + 0.5) * width
        bars = ax.bar(x + offset, rates, width, label=opp, color=colours[i])
        for bar, rate in zip(bars, rates):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{rate:.0%}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xlabel("LLM Variant")
    ax.set_ylabel("Win Rate")
    ax.set_title("LLM Ablation: Win Rate by Variant and Opponent")
    ax.set_xticks(x)
    ax.set_xticklabels(all_variants, rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend(title="Opponent")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    fig.tight_layout()
    fig.savefig(output, dpi=100)
    plt.close(fig)


def plot_invalid_rate_chart(
    all_metrics: list[VariantMetrics],
    output: Path,
) -> None:
    """Produce a bar chart of average invalid-move retry rate by variant.

    Parameters
    ----------
    all_metrics:
        List of variant metrics (aggregated across all opponents).
    output:
        Path to write the PNG file.
    """
    _ensure_matplotlib()
    import matplotlib.pyplot as plt
    import numpy as np

    output.parent.mkdir(parents=True, exist_ok=True)

    variants = [m.variant for m in all_metrics]
    rates = [m.avg_invalid_retries for m in all_metrics]

    fig, ax = plt.subplots(figsize=(19.2, 10.8), dpi=100)

    x = np.arange(len(variants))
    bars = ax.bar(x, rates, color=plt.cm.Oranges(np.linspace(0.3, 0.8, len(variants))))

    for bar, rate in zip(bars, rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(rates) * 0.02 if rates else 0,
            f"{rate:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xlabel("LLM Variant")
    ax.set_ylabel("Avg Invalid Move Retries")
    ax.set_title("LLM Ablation: Invalid Move Retry Rate by Variant")
    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(output, dpi=100)
    plt.close(fig)


def plot_latency_chart(
    all_metrics: list[VariantMetrics],
    output: Path,
) -> None:
    """Produce a bar chart of average latency (ms) by variant.

    Parameters
    ----------
    all_metrics:
        List of variant metrics (aggregated across all opponents).
    output:
        Path to write the PNG file.
    """
    _ensure_matplotlib()
    import matplotlib.pyplot as plt
    import numpy as np

    output.parent.mkdir(parents=True, exist_ok=True)

    variants = [m.variant for m in all_metrics]
    latencies = [m.avg_latency_ms for m in all_metrics]

    fig, ax = plt.subplots(figsize=(19.2, 10.8), dpi=100)

    x = np.arange(len(variants))
    bars = ax.bar(x, latencies, color=plt.cm.Blues(np.linspace(0.3, 0.8, len(variants))))

    for bar, lat in zip(bars, latencies):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(latencies) * 0.02 if latencies else 0,
            f"{lat:.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xlabel("LLM Variant")
    ax.set_ylabel("Avg Latency (ms)")
    ax.set_title("LLM Ablation: Average Latency by Variant")
    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(output, dpi=100)
    plt.close(fig)


def save_metrics_csv(
    all_metrics: list[VariantMetrics],
    output: Path,
) -> None:
    """Write aggregated metrics to a CSV file.

    Parameters
    ----------
    all_metrics:
        List of variant metrics.
    output:
        Path to write the CSV file.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "variant": m.variant,
            "memory_turns": m.memory_turns,
            "prompt_style": m.prompt_style,
            "games": m.games,
            "win_rate": round(m.win_rate, 4),
            "avg_invalid_retries": round(m.avg_invalid_retries, 4),
            "avg_latency_ms": round(m.avg_latency_ms, 2),
        }
        for m in all_metrics
    ]
    df = pd.DataFrame(rows)
    df.to_csv(output, index=False)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command()
@click.argument("csv_paths", nargs=-1, required=True)
@click.option(
    "--output-dir",
    default="output",
    show_default=True,
    type=click.Path(),
    help="Directory for output charts and CSV.",
)
@click.pass_context
def main(ctx: click.Context, csv_paths: tuple[str, ...], output_dir: str) -> None:
    """Generate LLM ablation charts from tournament result CSVs.

    CSV_PATHS can be individual files, directories, or glob patterns.
    """
    try:
        df = load_csvs(csv_paths)
    except FileNotFoundError as exc:
        click.echo(f"Error: {exc}", err=True)
        ctx.exit(1)
        return

    if df.empty:
        click.echo("Error: no data rows found.", err=True)
        ctx.exit(1)
        return

    # Compute overall metrics (across all opponents)
    all_metrics = compute_metrics(df)

    if not all_metrics:
        click.echo("No LLM agent variants found in data.", err=True)
        ctx.exit(1)
        return

    # Compute metrics grouped by opponent for the win-rate chart
    metrics_by_opp = compute_metrics_by_opponent(df)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Generate charts
    plot_winrate_chart(metrics_by_opp, out / "llm_ablation_winrate.png")
    click.echo(f"  ✓ {out / 'llm_ablation_winrate.png'}")

    plot_invalid_rate_chart(all_metrics, out / "llm_ablation_invalid_rate.png")
    click.echo(f"  ✓ {out / 'llm_ablation_invalid_rate.png'}")

    plot_latency_chart(all_metrics, out / "llm_ablation_latency.png")
    click.echo(f"  ✓ {out / 'llm_ablation_latency.png'}")

    # Save metrics CSV
    save_metrics_csv(all_metrics, out / "llm_ablation_metrics.csv")
    click.echo(f"  ✓ {out / 'llm_ablation_metrics.csv'}")

    click.echo(f"\nDone — {len(all_metrics)} variant(s) analysed.")


if __name__ == "__main__":
    main()
