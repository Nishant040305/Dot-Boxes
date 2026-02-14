"""
Simulate a game and record all states to a JSON file for visualization.
Usage: python simulate_and_record.py [board_size] [num_games]
"""
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.BoardEnv import BaseBoardEnv
from agents.RandomAgent import RandomAgent
from gameSimulation.simulation import SimulateRandomVSrandom

def record_game(board_size=3):
    """Record a single game and return state history."""
    env = BaseBoardEnv(board_size)
    simulator = SimulateRandomVSrandom(env)
    
    # Run the simulation
    score = simulator.simulate()
    
    return {
        "board_size": board_size,
        "final_score": env.score,
        "total_moves": len(env.action_history),
        "action_history": env.action_history,
        "state_history": env.state_history,
    }

def record_multiple_games(board_size=3, num_games=5):
    """Record multiple games."""
    games = []
    for i in range(num_games):
        game_data = record_game(board_size)
        game_data["game_id"] = i + 1
        games.append(game_data)
        print(f"Game {i+1}: P1={game_data['final_score'][0]} P2={game_data['final_score'][1]} ({game_data['total_moves']} moves)")
    return games

if __name__ == "__main__":
    board_size = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    num_games = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    
    print(f"\n Recording {num_games} game(s) on a {board_size}x{board_size} board...\n")
    
    games = record_multiple_games(board_size, num_games)
    
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui", "game_data.json")
    
    with open(output_path, "w") as f:
        json.dump({"games": games, "board_size": board_size}, f, indent=2)
    
    print(f"\n Game data saved to: {output_path}")
    print(f" Open the visualizer: ui/simulation.html")
