"""
analysis — Result analysis helpers.

Provides functions and classes that consume TournamentResult / rating tables
and produce charts, summary statistics, and notebook-ready data frames.

Planned modules
---------------
- plots.py         : Matplotlib/seaborn plot helpers (rating curves, win-rate bars)
- summary.py       : Tabular summaries (win/loss/draw counts, head-to-head matrix)

Usage example (once implemented)::

    from analysis.summary import head_to_head_matrix
    from analysis.plots import plot_rating_history

    matrix = head_to_head_matrix(tournament_result.matches)
    plot_rating_history(rating_history)
"""
