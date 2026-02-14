from agents.RandomAgent import RandomAgent
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
            self.env.step(self.agent1.act())
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
            self.env.step(self.agent1.act())
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
