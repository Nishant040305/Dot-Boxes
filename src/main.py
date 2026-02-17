import os
import sys

# Get the directory where the script is located (src)
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the project root (RLAgent)
project_root = os.path.dirname(script_dir)
# Add src to sys.path so imports work
sys.path.append(script_dir)

# Define path to models directory
models_dir = os.path.join(project_root, 'models')
model_path = os.path.join(models_dir, "mcts_3x3.pkl")

from env.BoardEnv import BaseBoardEnv
from agents.MCTSTrainer import MCTSTrainer
from agents.MinMaxAgent import MinMaxAgent
from agents.GreedyAgent import GreedyAgent
if __name__ == "__main__":
    # ── Train MCTS via self-play ──
    # Note: Iterations here is just for initialization, load overrides policy table
    trainer = MCTSTrainer(N=3, iterations=500)
    
    if os.path.exists(model_path):
        trainer.load(model_path)
    else:
        print(f"Error: Model not found at {model_path}")
        sys.exit(1)
    print("\n" + "=" * 60)
    print("  Evaluation: Trained MCTS vs Greedy (3×3)")
    print("=" * 60 + "\n")

    env = BaseBoardEnv(3)
    env.reset()

    trained_agent = trainer.get_trained_agent(env, iterations=5000)
    greedy =GreedyAgent(env)

    while not env.done:
        if env.current_player == 2:
            action = trained_agent.act()
        else:
            action = greedy.act()
        env.step(action)

    print(f"  Trained MCTS: {env.score[0]}")
    print(f"  Greedy:       {env.score[1]}")
    if env.score[0] > env.score[1]:
        print("  Result: Trained MCTS wins!")
    elif env.score[0] < env.score[1]:
        print("  Result: Greedy wins")
    else:
        print("  Result: Draw")