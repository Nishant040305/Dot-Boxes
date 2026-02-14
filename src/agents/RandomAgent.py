from agents.agent import Agent
import random
class RandomAgent(Agent):
    def __init__(self, env):
        super().__init__(env)
    
    def act(self):
        return self.env._get_available_actions()[random.randint(0, len(self.env._get_available_actions())-1)]