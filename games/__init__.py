"""
games — OpenSpiel game adapters.

Each module in this package wraps an OpenSpiel game with a thin, typed
interface that the arena and agents can depend on without touching the raw
pyspiel API directly.

Available modules
-----------------
- tic_tac_toe.py   : TicTacToeGame
- breakthrough.py  : BreakthroughGame

Usage example::

    from games.tic_tac_toe import TicTacToeGame
    from games.breakthrough import BreakthroughGame

    game = BreakthroughGame()
    state = game.new_state()
"""

from games.breakthrough import BreakthroughAction, BreakthroughGame
from games.tic_tac_toe import TicTacToeGame

__all__ = [
    "BreakthroughAction",
    "BreakthroughGame",
    "TicTacToeGame",
]
