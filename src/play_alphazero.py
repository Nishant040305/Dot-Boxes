import sys
import os
import torch
from agents.AlphaZeroAgent import AlphaZeroAgent
from agents.GreedyAgent import GreedyAgent
from env.BoardEnv import BaseBoardEnv
from models.Net import AlphaZeroNet

# Add src to path
script_dir = os.path.dirname(os.path.abspath(__file__))
# parent is src so we are good if we run from src or root with PYTHONPATH
# logic in main.py was:
# script_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.dirname(script_dir)
# sys.path.append(script_dir)

# Let's replicate
sys.path.append(script_dir)

def play(n_games=10, board_size=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    models_dir = os.path.join(os.path.dirname(script_dir), 'models')
    model_path = os.path.join(models_dir, f"alphazero_{board_size}x{board_size}.pth")
    
    env = BaseBoardEnv(board_size)
    
    # Initialize AlphaZero Agent
    agent_az = AlphaZeroAgent(env, model_path=model_path, n_simulations=400, device=device)
    
    # Initialize Greedy Agent
    agent_greedy = GreedyAgent(env)
    
    wins = {0: 0, 1: 0, 2: 0} # 0=Draw, 1=AZ, 2=Greedy
    
    print(f"\nStarting {n_games} games: AlphaZero vs Greedy on {board_size}x{board_size}")
    
    for i in range(n_games):
        env.reset()
        
        # Randomize start player
        # Actually BaseBoardEnv usually starts with player 1.
        # We can swap agents or just track who is who.
        # Let's keep AZ as Player 1 for simplicity in this script, or alternate?
        # Alternating is better for fair evaluation.
        
        az_player = 1 if i % 2 == 0 else 2
        greedy_player = 2 if az_player == 1 else 1
        
        print(f"Game {i+1}: AZ is Player {az_player}")
        
        while not env.done:
            if env.current_player == az_player:
                action = agent_az.act()
            else:
                action = agent_greedy.act()
            env.step(action)
            
        s1, s2 = env.score
        print(f"  Score: P1={s1} P2={s2}")
        
        if s1 > s2: 
            winner = 1
        elif s2 > s1: 
            winner = 2
        else: 
            winner = 0
            
        if winner == az_player:
            wins[1] += 1
            print("  Result: AlphaZero Wins")
        elif winner == greedy_player:
            wins[2] += 1
            print("  Result: Greedy Wins")
        else:
            wins[0] += 1
            print("  Result: Draw")
            
    print("\nFinal Results:")
    print(f"AlphaZero Wins: {wins[1]}")
    print(f"Greedy Wins:    {wins[2]}")
    print(f"Draws:          {wins[0]}")

if __name__ == "__main__":
    play()
