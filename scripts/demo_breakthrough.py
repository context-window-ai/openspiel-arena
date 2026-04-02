#!/usr/bin/env python3
"""
Demo script for the Breakthrough game wrapper.

This script demonstrates the basic usage of the BreakthroughGame class,
including creating games, enumerating legal actions, applying moves,
and detecting terminal states.
"""

from games.breakthrough import BreakthroughGame


def demo_basic_usage() -> None:
    """Demonstrate basic game operations."""
    print("=" * 60)
    print("Breakthrough Game Wrapper Demo")
    print("=" * 60)
    print()

    # Create a standard 8x8 game
    game = BreakthroughGame()
    print(f"Created game: {game.name} ({game.columns}x{game.rows})")
    print(f"Players: {game.num_players}")
    print()

    # Create initial state
    state = game.new_state()
    print("Initial board:")
    print(game.render_compact(state))
    print()

    # Show current player
    current = game.current_player(state)
    print(f"Current player: {game.player_name(current)} (player {current})")
    print()

    # Enumerate legal actions
    actions = game.legal_actions(state)
    print(f"Number of legal actions: {len(actions)}")
    print()

    # Show action metadata
    print("Sample actions (first 5):")
    for action_id in actions[:5]:
        metadata = game.get_action_metadata(state, action_id)
        capture_str = " (capture)" if metadata.is_capture else ""
        print(f"  {metadata.notation}: {metadata.from_square} -> {metadata.to_square}{capture_str}")
    print()


def demo_move_sequence() -> None:
    """Demonstrate applying a sequence of moves."""
    print("=" * 60)
    print("Move Sequence Demo (4x4 board)")
    print("=" * 60)
    print()

    game = BreakthroughGame(columns=4, rows=4)
    state = game.new_state()

    moves = [
        "a4b3",   # Black moves diagonal-right
        "a1a2",   # White moves forward
        "b4a3",   # Black moves diagonal-left
        "b1c2",   # White moves diagonal-right
    ]

    print("Initial state:")
    print(game.render_compact(state))
    print()

    for i, move in enumerate(moves):
        player = game.player_name(game.current_player(state))
        action = game.string_to_action(state, move)
        game.apply_action(state, action)
        print(f"Move {i + 1}: {player} plays {move}")
        print(game.render_compact(state))
        print()


def demo_contextual_rendering() -> None:
    """Demonstrate the contextual rendering for LLM prompts."""
    print("=" * 60)
    print("Contextual Rendering Demo (for LLM prompts)")
    print("=" * 60)
    print()

    game = BreakthroughGame(columns=4, rows=4)
    state = game.new_state()

    # Make a couple of moves
    game.apply_action(state, game.string_to_action(state, "a4b3"))
    game.apply_action(state, game.string_to_action(state, "a1a2"))

    print(game.render_with_context(state))
    print()


def demo_game_completion() -> None:
    """Demonstrate detecting game completion and winner."""
    print("=" * 60)
    print("Game Completion Demo")
    print("=" * 60)
    print()

    game = BreakthroughGame(columns=4, rows=4)
    state = game.new_state()

    # Play a short game where white wins
    moves = [
        "a4b3",   # Black
        "a1a2",   # White
        "d4c3",   # Black
        "b1c2",   # White
        "c4d3",   # Black
        "a2b3*",  # White captures
        "b4a3",   # Black
        "c1d2",   # White
        "d3c2*",  # Black captures
        "b3a4",   # White reaches row 4 - wins!
    ]

    print("Playing a game until white wins...")
    print()

    for i, move in enumerate(moves):
        if game.is_terminal(state):
            break
        player = game.player_name(game.current_player(state))
        action = game.string_to_action(state, move)
        game.apply_action(state, action)
        print(f"Move {i + 1}: {player} plays {move}")

    print()
    print("Final board:")
    print(game.render_compact(state))
    print()

    if game.is_terminal(state):
        winner = game.winner(state)
        if winner is not None:
            print(f"Winner: {game.player_name(winner)}")
        returns = game.returns(state)
        print(f"Returns: {returns}")
    print()


def demo_action_metadata() -> None:
    """Demonstrate action metadata for LLM agents."""
    print("=" * 60)
    print("Action Metadata Demo (for LLM agents)")
    print("=" * 60)
    print()

    game = BreakthroughGame(columns=4, rows=4)
    state = game.new_state()

    # Make a move to create capture opportunities
    game.apply_action(state, game.string_to_action(state, "a4b3"))
    game.apply_action(state, game.string_to_action(state, "a1a2"))
    game.apply_action(state, game.string_to_action(state, "b4a3"))

    print("Current board:")
    print(game.render_compact(state))
    print()

    # Show all legal actions with metadata
    all_actions = game.get_all_legal_actions_metadata(state)
    print(f"Legal actions for {game.player_name(game.current_player(state))}:")
    print()

    # Group by from_square
    by_from: dict[str, list] = {}
    for action in all_actions:
        if action.from_square not in by_from:
            by_from[action.from_square] = []
        by_from[action.from_square].append(action)

    for from_sq in sorted(by_from.keys()):
        moves = by_from[from_sq]
        for m in moves:
            capture_str = " [CAPTURE]" if m.is_capture else ""
            print(f"  {m.notation}: {m.from_square} -> {m.to_square}{capture_str}")
    print()


if __name__ == "__main__":
    demo_basic_usage()
    demo_move_sequence()
    demo_contextual_rendering()
    demo_game_completion()
    demo_action_metadata()

    print("=" * 60)
    print("Demo complete!")
    print("=" * 60)
