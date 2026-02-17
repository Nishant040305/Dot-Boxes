from agents.RandomAgent import RandomAgent
from agents.GreedyAgent import GreedyAgent
from agents.MinMaxAgent import MinMaxAgent
from agents.AlphaBetaAgent import AlphaBetaAgent
from agents.AlphaZeroAgent import AlphaZeroAgent
from env.BoardEnv import BaseBoardEnv
import os
import torch

class BaseSimulation:
    def __init__(self, env):
        self.env = env
        self.agent1 = None
        self.agent2 = None
    def set_agents(self, agent1, agent2):
        self.agent1 = agent1
        self.agent2 = agent2
    def simulate(self):
        self.env.reset()
        if hasattr(self.agent1, 'reset'): self.agent1.reset()
        if hasattr(self.agent2, 'reset'): self.agent2.reset()
        if hasattr(self.agent1, 'env'): self.agent1.env = self.env
        if hasattr(self.agent2, 'env'): self.agent2.env = self.env
        
        while(self.env.done == False):
            if(self.env.current_player == 1):
                self.env.step(self.agent1.act())
            else:
                self.env.step(self.agent2.act())
        return self.env.score
    def simulate_n_games(self, n):
        scores = []
        for i in range(n):
            scores.append(self.simulate())
        return scores
    def simulate_actions_display(self):
        self.env.reset()
        if hasattr(self.agent1, 'reset'): self.agent1.reset()
        if hasattr(self.agent2, 'reset'): self.agent2.reset()
        if hasattr(self.agent1, 'env'): self.agent1.env = self.env
        if hasattr(self.agent2, 'env'): self.agent2.env = self.env

        while(self.env.done == False):
            if(self.env.current_player == 1):
                self.env.step(self.agent1.act())
            else:
                self.env.step(self.agent2.act())
            self.env.render()
        return self.env.score

class SimulateRandomVSrandom(BaseSimulation):
    def __init__(self, env):
        super().__init__(env)
    def simulate(self):
        self.set_agents(RandomAgent(self.env), RandomAgent(self.env))
        return super().simulate()
    def simulate_n_games(self, n):
        self.set_agents(RandomAgent(self.env), RandomAgent(self.env))
        return super().simulate_n_games(n)
    def simulate_actions_display(self):
        self.set_agents(RandomAgent(self.env), RandomAgent(self.env))
        return super().simulate_actions_display()

class SimulateRandomVSgreedy(BaseSimulation):
    def __init__(self, env):
        super().__init__(env)
    def simulate(self):
        self.set_agents(RandomAgent(self.env), GreedyAgent(self.env))
        return super().simulate()
    def simulate_n_games(self, n):
        self.set_agents(RandomAgent(self.env), GreedyAgent(self.env))
        return super().simulate_n_games(n)
    def simulate_actions_display(self):
        self.set_agents(RandomAgent(self.env), GreedyAgent(self.env))
        return super().simulate_actions_display()

class SimulateGreedyVSgreedy(BaseSimulation):
    def __init__(self, env):
        super().__init__(env)
    def simulate(self):
        self.set_agents(GreedyAgent(self.env), GreedyAgent(self.env))
        return super().simulate()
    def simulate_n_games(self, n):
        self.set_agents(GreedyAgent(self.env), GreedyAgent(self.env))
        return super().simulate_n_games(n)
    def simulate_actions_display(self):
        self.set_agents(GreedyAgent(self.env), GreedyAgent(self.env))
        return super().simulate_actions_display()

class SimulateMinMaxVSgreedy(BaseSimulation):
    def __init__(self, env, depth=1, parallel=True):
        super().__init__(env)
        self.depth = depth
        self.parallel = parallel
    def simulate(self):
        self.set_agents(MinMaxAgent(self.env, depth=self.depth, parallel=self.parallel), GreedyAgent(self.env))
        return super().simulate()
    def simulate_n_games(self, n):
        self.set_agents(MinMaxAgent(self.env, depth=self.depth, parallel=self.parallel), GreedyAgent(self.env))
        return super().simulate_n_games(n)
    def simulate_actions_display(self):
        self.set_agents(MinMaxAgent(self.env, depth=self.depth, parallel=self.parallel), GreedyAgent(self.env))
        return super().simulate_actions_display()

class SimulateMinMaxVSMinMax(BaseSimulation):
    def __init__(self, env, depth1=1, depth2=1, parallel=True):
        super().__init__(env)
        self.depth1 = depth1
        self.depth2 = depth2
        self.parallel = parallel
    def simulate(self):
        self.set_agents(MinMaxAgent(self.env, depth=self.depth1, parallel=self.parallel), MinMaxAgent(self.env, depth=self.depth2, parallel=self.parallel))
        return super().simulate()
    def simulate_n_games(self, n):
        self.set_agents(MinMaxAgent(self.env, depth=self.depth1, parallel=self.parallel), MinMaxAgent(self.env, depth=self.depth2, parallel=self.parallel))
        return super().simulate_n_games(n)
    def simulate_actions_display(self):
        self.set_agents(MinMaxAgent(self.env, depth=self.depth1, parallel=self.parallel), MinMaxAgent(self.env, depth=self.depth2, parallel=self.parallel))
        return super().simulate_actions_display()

class SimulateAlphaBetaVSgreedy(BaseSimulation):
    def __init__(self, env, depth=3, parallel=True):
        super().__init__(env)
        self.depth = depth
        self.parallel = parallel
    def simulate(self):
        self.set_agents(AlphaBetaAgent(self.env, depth=self.depth, parallel=self.parallel), GreedyAgent(self.env))
        return super().simulate()
    def simulate_n_games(self, n):
        self.set_agents(AlphaBetaAgent(self.env, depth=self.depth, parallel=self.parallel), GreedyAgent(self.env))
        return super().simulate_n_games(n)
    def simulate_actions_display(self):
        self.set_agents(AlphaBetaAgent(self.env, depth=self.depth, parallel=self.parallel), GreedyAgent(self.env))
        return super().simulate_actions_display()

class SimulateAlphaBetaVSMinMax(BaseSimulation):
    def __init__(self, env, depth_ab=3, depth_mm=3, parallel=True):
        super().__init__(env)
        self.depth_ab = depth_ab
        self.depth_mm = depth_mm
        self.parallel = parallel
    def simulate(self):
        self.set_agents(AlphaBetaAgent(self.env, depth=self.depth_ab, parallel=self.parallel), MinMaxAgent(self.env, depth=self.depth_mm, parallel=self.parallel))
        return super().simulate()
    def simulate_n_games(self, n):
        self.set_agents(AlphaBetaAgent(self.env, depth=self.depth_ab, parallel=self.parallel), MinMaxAgent(self.env, depth=self.depth_mm, parallel=self.parallel))
        return super().simulate_n_games(n)
    def simulate_actions_display(self):
        self.set_agents(AlphaBetaAgent(self.env, depth=self.depth_ab, parallel=self.parallel), MinMaxAgent(self.env, depth=self.depth_mm, parallel=self.parallel))
        return super().simulate_actions_display()

class SimulateAlphaBetaVSAlphaBeta(BaseSimulation):
    def __init__(self, env, depth1=3, depth2=3, parallel=True):
        super().__init__(env)
        self.depth1 = depth1
        self.depth2 = depth2
        self.parallel = parallel
    def simulate(self):
        self.set_agents(AlphaBetaAgent(self.env, depth=self.depth1, parallel=self.parallel), AlphaBetaAgent(self.env, depth=self.depth2, parallel=self.parallel))
        return super().simulate()
    def simulate_n_games(self, n):
        self.set_agents(AlphaBetaAgent(self.env, depth=self.depth1, parallel=self.parallel), AlphaBetaAgent(self.env, depth=self.depth2, parallel=self.parallel))
        return super().simulate_n_games(n)
    def simulate_actions_display(self):
        self.set_agents(AlphaBetaAgent(self.env, depth=self.depth1, parallel=self.parallel), AlphaBetaAgent(self.env, depth=self.depth2, parallel=self.parallel))
        return super().simulate_actions_display()

class BaseAlphaZeroSimulation(BaseSimulation):
    def __init__(self, env, model_path=None, n_simulations=100):
        super().__init__(env)
        self.n_simulations = n_simulations
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if model_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            root_dir = os.path.dirname(os.path.dirname(current_dir))
            models_dir = os.path.join(root_dir, 'models')
            if not os.path.exists(models_dir):
                models_dir = os.path.join(os.path.dirname(current_dir), 'models')
            self.model_path = os.path.join(models_dir, f"alphazero_{env.N}x{env.N}.pth")
        else:
            self.model_path = model_path

    def _get_az_agent(self):
        # We need to pass the environment for n=3 etc
        return AlphaZeroAgent(self.env, model_path=self.model_path, n_simulations=self.n_simulations, device=self.device)

class SimulateAlphaZeroVsRandom(BaseAlphaZeroSimulation):
    def simulate(self):
        self.set_agents(self._get_az_agent(), RandomAgent(self.env))
        return super().simulate()
    def simulate_n_games(self, n):
        self.set_agents(self._get_az_agent(), RandomAgent(self.env))
        return super().simulate_n_games(n)
    def simulate_actions_display(self):
        self.set_agents(self._get_az_agent(), RandomAgent(self.env))
        return super().simulate_actions_display()

class SimulateAlphaZeroVsGreedy(BaseAlphaZeroSimulation):
    def simulate(self):
        self.set_agents(self._get_az_agent(), GreedyAgent(self.env))
        return super().simulate()
    def simulate_n_games(self, n):
        self.set_agents(self._get_az_agent(), GreedyAgent(self.env))
        return super().simulate_n_games(n)
    def simulate_actions_display(self):
        self.set_agents(self._get_az_agent(), GreedyAgent(self.env))
        return super().simulate_actions_display()

class SimulateAlphaZeroVsAlphaBeta(BaseAlphaZeroSimulation):
    def __init__(self, env, model_path=None, n_simulations=100, ab_depth=3, parallel=True):
        super().__init__(env, model_path, n_simulations)
        self.ab_depth = ab_depth
        self.parallel = parallel
    def simulate(self):
        self.set_agents(self._get_az_agent(), AlphaBetaAgent(self.env, depth=self.ab_depth, parallel=self.parallel))
        return super().simulate()
    def simulate_n_games(self, n):
        self.set_agents(self._get_az_agent(), AlphaBetaAgent(self.env, depth=self.ab_depth, parallel=self.parallel))
        return super().simulate_n_games(n)
    def simulate_actions_display(self):
        self.set_agents(self._get_az_agent(), AlphaBetaAgent(self.env, depth=self.ab_depth, parallel=self.parallel))
        return super().simulate_actions_display()