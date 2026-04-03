# openspiel-arena

**Experiment goal:** Pit traditional search-style game agents (e.g. MCTS, alpha-beta) against
LLM-backed agents on the same board game, measure their relative strength with an Elo/Glicko
rating system, and produce reproducible analysis artifacts.

---

## What this is

`openspiel-arena` is a lightweight tournament harness built on top of
[OpenSpiel](https://github.com/google-deepmind/open_spiel).  It provides:

| Layer | Package | Purpose |
|---|---|---|
| Game wrappers | `games/` | Thin adapters around OpenSpiel game objects |
| Agent library | `agents/` | OpenSpiel built-ins + LLM-backed agents sharing one interface |
| Tournament engine | `arena/` | Round-robin / Swiss scheduler, match executor, result store |
| Rating system | `ratings/` | Elo & Glicko-2 computation over match history |
| Analysis | `analysis/` | Notebooks & scripts that turn results into charts / tables |
| CLI | `scripts/run_tournament.py` | Single entry-point to kick off a tournament from the command line |

---

## Quick start

### 1 — Prerequisites

- Python 3.11+
- A virtual environment tool (`venv`, `uv`, or `conda`)

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 2 — Install dependencies

```bash
pip install -e ".[dev]"
```

> **OpenSpiel note:** OpenSpiel ships pre-built wheels for Linux/macOS on PyPI
> (`open_spiel`).  The `pyproject.toml` pulls it in automatically.  If you are
> on Windows or need a custom build, follow the
> [OpenSpiel build guide](https://github.com/google-deepmind/open_spiel/blob/master/docs/install.md)
> and then `pip install -e ".[dev]"`.

### 3 — Configure secrets

```bash
cp .env.example .env
# edit .env and fill in your API keys
```

### 4 — Run the smoke test

```bash
pytest tests/ -q
```

### 5 — Launch a tournament

```bash
python scripts/run_tournament.py --game tic_tac_toe --rounds 10
```

---

## Project layout

```
openspiel-arena/
├── games/          # OpenSpiel game adapters
├── agents/         # Agent implementations (MCTS, LLM, random, …)
├── arena/          # Tournament scheduler & match runner
├── ratings/        # Elo / Glicko-2 rating engine
├── analysis/       # Result analysis helpers & notebooks
├── tests/          # pytest test suite
├── scripts/        # CLI entry-points
├── pyproject.toml  # Package metadata & dependencies
├── .env.example    # Required environment variables (no secrets)
└── README.md
```

---

## Environment variables

See `.env.example` for the full list.  The most important ones:

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY` | Key for GPT-based LLM agents |
| `ANTHROPIC_API_KEY` | Key for Claude-based LLM agents |
| `ARENA_RESULTS_DIR` | Directory where match results are written (default: `results/`) |
| `ARENA_LOG_LEVEL` | Logging verbosity (`DEBUG`, `INFO`, `WARNING`) |

---

## Development

```bash
# Lint & format
ruff check . --fix
ruff format .

# Type-check
mypy .

# Tests with coverage
pytest --cov=. --cov-report=term-missing
```

---

## Rating Systems: Elo vs AlphaRank

This project supports two complementary approaches to rating agents:

### Elo Rating

**Best for:** Transitive games where skill is hierarchical.

Elo assumes that if agent A beats agent B, and agent B beats agent C, then agent A should beat agent C. This works well for:
- Chess, Go, and most board games
- Direct skill comparisons
- Simple leaderboard rankings

**Usage:**
```bash
python3 -m ratings.elo results/matches.csv
```

### AlphaRank

**Best for:** Non-transitive games with cyclic win patterns.

Some games exhibit "Rock-Paper-Scissors" dynamics where A beats B, B beats C, but C beats A. Elo struggles here because no clear hierarchy exists. AlphaRank:
- Uses evolutionary game theory to find equilibrium strategies
- Produces a population distribution over agents
- Handles cyclic win patterns naturally
- Identifies which strategies dominate in the meta-game

**Usage:**
```python
from ratings.payoff_matrix import compute_payoff_matrix_from_csv
from analysis.alpharank import compute_alpha_rank

matrix = compute_payoff_matrix_from_csv('results/matches.csv')
ranking = compute_alpha_rank(matrix.to_numpy())
```

See `analysis/alpharank.ipynb` for a full interactive example.

### Comparison

| Criterion | Elo | AlphaRank |
|-----------|-----|----------|
| Transitive games | ✅ Best | ✅ Works |
| Non-transitive games | ❌ Fails | ✅ Best |
| Simple interpretation | ✅ Easy | ⚠️ Moderate |
| Computational cost | ✅ O(n) | ⚠️ O(n²) |
| Single rating number | ✅ Yes | ❌ Distribution |
| Handles cycles | ❌ No | ✅ Yes |

**Recommendation:** Use Elo for quick leaderboards and when skill is clearly hierarchical. Use AlphaRank when analyzing complex meta-games with strategic diversity.

---

## Roadmap

- [x] Tic-tac-toe game wrapper (baseline game)
- [x] Random agent
- [x] OpenSpiel MCTS agent
- [x] LLM agent (OpenAI)
- [x] LLM agent (Anthropic)
- [x] Elo rating computation
- [x] Glicko-2 rating computation
- [x] Payoff matrix computation
- [x] AlphaRank analysis
- [ ] Analysis notebook: rating convergence curves
- [ ] Analysis notebook: move-quality comparison

---

## License

MIT
