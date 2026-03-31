#!/usr/bin/env python3
"""
Interactive CLI to play Dots and Boxes against any AI agent.

Usage:
    python play_vs_agent.py                          # interactive menu
    python play_vs_agent.py --agent greedy           # play vs greedy
    python play_vs_agent.py --agent alphabeta:depth=4 --player 2 --size 3
"""

import sys
import os

_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)


AGENT_MENU = [
    ("random",              "Random Agent"),
    ("greedy",              "Greedy Agent"),
    ("minmax:depth=3",      "MinMax  (depth 3)"),
    ("minmax:depth=5",      "MinMax  (depth 5)"),
    ("alphabeta:depth=3",   "AlphaBeta (depth 3)"),
    ("alphabeta:depth=5",   "AlphaBeta (depth 5)"),
    ("mcts:iterations=500", "MCTS (500 iterations)"),
    ("alphazero_bit:n_simulations=400", "AlphaZero Bit (400 sims)"),
    ("alphazero_cpp:n_simulations=400", "AlphaZero C++ (400 sims)"),
]


def interactive_menu():
    """Show a menu and return (agent_spec, board_size, player)."""
    print()
    print("  ╔══════════════════════════════════════════════╗")
    print("  ║        DOTS AND BOXES — Human vs AI          ║")
    print("  ╠══════════════════════════════════════════════╣")

    for idx, (_, label) in enumerate(AGENT_MENU, 1):
        print(f"  ║   {idx}. {label:<40s} ║")

    print(f"  ║   {len(AGENT_MENU)+1}. Custom agent spec                        ║")
    print("  ╚══════════════════════════════════════════════╝")

    # Choose opponent
    while True:
        try:
            choice = input("\n  Choose opponent [1-{}]: ".format(len(AGENT_MENU) + 1)).strip()
            idx = int(choice)
            if 1 <= idx <= len(AGENT_MENU):
                agent_spec = AGENT_MENU[idx - 1][0]
                break
            elif idx == len(AGENT_MENU) + 1:
                agent_spec = input("  Enter agent spec (e.g. alphabeta:depth=4): ").strip()
                break
            else:
                print("  Invalid choice.")
        except (ValueError, EOFError, KeyboardInterrupt):
            print("\n  Goodbye!")
            sys.exit(0)

    # Board size
    size_str = input("  Board size [3]: ").strip()
    board_size = int(size_str) if size_str.isdigit() else 3

    # Player order
    order = input("  Go first? (y/n) [y]: ").strip().lower()
    player = 2 if order in ("n", "no", "2") else 1

    return agent_spec, board_size, player


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Play Dots and Boxes against an AI agent")
    parser.add_argument("--agent", "-a", type=str, default=None,
                        help="Agent spec to play against (e.g. greedy, alphabeta:depth=4)")
    parser.add_argument("--size", "-s", type=int, default=3,
                        help="Board size N (default: 3)")
    parser.add_argument("--player", "-p", type=int, default=1, choices=[1, 2],
                        help="Play as player 1 (first) or 2 (second)")
    parser.add_argument("--env", type=str, default="classic", choices=["bit", "classic"],
                        help="Environment type (default: classic)")
    parser.add_argument("--visual", "-v", action="store_true",
                        help="Launch the web-based visual interface instead of CLI")
    parser.add_argument("--port", type=int, default=5050,
                        help="Port for visual server (default: 5050)")

    args = parser.parse_args()

    if args.visual:
        # Launch web-based play server
        import subprocess
        ui_script = os.path.join(_script_dir, "ui", "play_server.py")
        subprocess.run([sys.executable, ui_script, "--port", str(args.port)])
        return

    if args.agent:
        agent_spec = args.agent
        board_size = args.size
        player = args.player
    else:
        agent_spec, board_size, player = interactive_menu()

    from gameSimulation.arena import Arena

    arena = Arena(board_size=board_size, env_type=args.env if args.agent else "classic")

    while True:
        result = arena.play_interactive(agent_spec, human_player=player)
        again = input("\n  Play again? (y/n) [y]: ").strip().lower()
        if again in ("n", "no"):
            break

    print("  Thanks for playing!")


if __name__ == "__main__":
    main()

