class Reward:
    def __init__(self, env):
        self.env = env
    def get_reward(self, action):
        raise NotImplementedError

class SimpleReward(Reward):
    def __init__(self, env):
        super().__init__(env)
    def get_reward(self, action):
        reward = 0
        if(action[0] == 0):
            if(action[1] > 0 and self.env.horizontal_edges[action[1]-1][action[2]] and self.env.vertical_edges[action[1]-1][action[2]] and self.env.vertical_edges[action[1]-1][action[2]+1]):
                reward +=1
            if(action[1] < self.env.N and self.env.horizontal_edges[action[1]+1][action[2]] and self.env.vertical_edges[action[1]][action[2]] and self.env.vertical_edges[action[1]][action[2]+1]):
                reward +=1
        else:
            if(action[2] > 0 and self.env.vertical_edges[action[1]][action[2]-1] and self.env.horizontal_edges[action[1]][action[2]-1] and self.env.horizontal_edges[action[1]+1][action[2]-1]):
                reward +=1
            if(action[2] < self.env.N and self.env.vertical_edges[action[1]][action[2]+1] and self.env.horizontal_edges[action[1]][action[2]] and self.env.horizontal_edges[action[1]+1][action[2]]):
                reward +=1
        return reward