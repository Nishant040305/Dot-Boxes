import sys
import os

# Get the directory where the script is located (src/train)
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (src)
src_dir = os.path.dirname(script_dir)
# Add src to sys.path so we can import modules like 'agents'
sys.path.append(src_dir)

# Define path to models directory (src/../models -> RLAgent/models)
project_root = os.path.dirname(src_dir)
models_dir = os.path.join(project_root, 'models')
model_path = os.path.join(models_dir, "mcts_3x3.pkl")

from agents.MCTSTrainer import MCTSTrainer
from agents.GreedyAgent import GreedyAgent
from env.BoardEnv import BaseBoardEnv

trainer = MCTSTrainer(N=3, iterations=500)
# Ensure models directory exists
os.makedirs(models_dir, exist_ok=True)

for i in range(100):
    if os.path.exists(model_path):
        trainer.load(model_path)
    else:
        print(f"Model not found at {model_path}, starting fresh.")
    trainer.train(num_games=100)
    trainer.save(model_path)

    print("=== Trained MCTS vs Greedy (3x3, 5 games) ===")
    env = BaseBoardEnv(3)
    agent1 = trainer.get_trained_agent(env, iterations=500)
    agent2 = GreedyAgent(env)

    for i in range(5):
        env.reset()
        while not env.done:
            if env.current_player == 1:
                action = agent1.act()
            else:
                action = agent2.act()
            env.step(action)
        print(f"Game {i+1}: MCTS={env.score[0]} Greedy={env.score[1]}")