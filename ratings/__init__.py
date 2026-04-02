"""
ratings — Elo and Glicko-2 rating computation.

Takes a sequence of MatchResult objects and produces an up-to-date rating
table for every agent.

Planned modules
---------------
- elo.py     : Classic Elo rating with configurable K-factor
- glicko2.py : Glicko-2 rating with rating deviation and volatility

Usage example (once implemented)::

    from ratings.elo import EloRatingSystem
    from arena.result import TournamentResult

    elo = EloRatingSystem(k_factor=32, initial_rating=1500)
    table = elo.compute(tournament_result.matches)
    print(table)  # agent_name -> elo_rating
"""
