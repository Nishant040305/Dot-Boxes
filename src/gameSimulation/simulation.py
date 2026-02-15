from agents.RandomAgent import RandomAgent
from agents.GreedyAgent import GreedyAgent
from agents.MinMaxAgent import MinMaxAgent
from agents.AlphaBetaAgent import AlphaBetaAgent
from env.BoardEnv import BaseBoardEnv
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
        while(self.env.done == False):
            if(self.env.current_player == 1):
                self.env.step(self.agent1.act())
            else:
                self.env.step(self.agent2.act())
            self.env.render()
        return self.env.score


# Simulate Random VS Random
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