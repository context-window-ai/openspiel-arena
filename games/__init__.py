"""
games — OpenSpiel game adapters.

Each module in this package wraps an OpenSpiel game with a thin, typed
interface that the arena and agents can depend on without touching the raw
pyspiel API directly.

Planned modules
---------------
- tic_tac_toe.py   : TicTacToeGame (first reference game)

Usage example (once implemented)::

    from games.tic_tac_toe import TicTacToeGame

    game = TicTacToeGame()
    state = game.new_initial_state()
"""
