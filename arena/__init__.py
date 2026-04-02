"""
arena — Tournament scheduler and match runner.

The arena is responsible for:
1. Scheduling match-ups (round-robin, Swiss, etc.)
2. Executing individual games between two agents
3. Collecting results and persisting them to disk

Planned modules
---------------
- match.py         : Run a single game between two agents; return a MatchResult
- tournament.py    : Schedule and run a full tournament; aggregate MatchResults
- result.py        : MatchResult / TournamentResult dataclasses

Usage example (once implemented)::

    from arena.tournament import Tournament
    from agents.random_agent import RandomAgent
    from agents.mcts_agent import MCTSAgent
    from games.tic_tac_toe import TicTacToeGame

    t = Tournament(
        game=TicTacToeGame(),
        agents=[RandomAgent(), MCTSAgent()],
        rounds=20,
    )
    results = t.run()

Modules are importable individually; this ``__init__`` stays import-free to
avoid circular dependencies.
"""
