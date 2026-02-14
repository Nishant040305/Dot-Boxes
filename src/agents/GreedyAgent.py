from agents.agent import Agent
from rewards.reward import SimpleReward
import random
class GreedyAgent(Agent):
    def __init__(self, env):
        self.reward = SimpleReward(env)
        super().__init__(env)
    def act(self):
        actions = self.env._get_available_actions()
        return self._get_best_action(actions)
    def _get_best_action(self,actions):
        best_action = actions[0]
        best_actions = [best_action]
        for action in actions:
            if(self.reward.get_reward(action) > self.reward.get_reward(best_action)):
                best_action = action
                best_actions = [best_action]
            elif(self.reward.get_reward(action) == self.reward.get_reward(best_action)):
                best_actions.append(action)
        return random.choice(best_actions)