class Reward:
    def __init__(self, env):
        self.env = env
    def get_reward(self, action):
        raise NotImplementedError

class SimpleReward(Reward):
    def __init__(self, env):
        super().__init__(env)

    @property
    def _rows(self):
        return self.env.rows

    @property
    def _cols(self):
        return self.env.cols

    def get_reward(self, action):
        reward = 0
        if(action[0] == 0):
            if(action[1] > 0 and self.env.horizontal_edges[action[1]-1][action[2]] and self.env.vertical_edges[action[1]-1][action[2]] and self.env.vertical_edges[action[1]-1][action[2]+1]):
                reward +=1
            if(action[1] < self._rows and self.env.horizontal_edges[action[1]+1][action[2]] and self.env.vertical_edges[action[1]][action[2]] and self.env.vertical_edges[action[1]][action[2]+1]):
                reward +=1
        else:
            if(action[2] > 0 and self.env.vertical_edges[action[1]][action[2]-1] and self.env.horizontal_edges[action[1]][action[2]-1] and self.env.horizontal_edges[action[1]+1][action[2]-1]):
                reward +=1
            if(action[2] < self._cols and self.env.vertical_edges[action[1]][action[2]+1] and self.env.horizontal_edges[action[1]][action[2]] and self.env.horizontal_edges[action[1]+1][action[2]]):
                reward +=1
        return reward
    def get_reward_state(self,action,state):
        reward = 0
        if(action[0] == 0):
            if(action[1] > 0 and state.horizontal_edges[action[1]-1][action[2]] and state.vertical_edges[action[1]-1][action[2]] and state.vertical_edges[action[1]-1][action[2]+1]):
                reward +=1
            if(action[1] < self._rows and state.horizontal_edges[action[1]+1][action[2]] and state.vertical_edges[action[1]][action[2]] and state.vertical_edges[action[1]][action[2]+1]):
                reward +=1
        else:
            if(action[2] > 0 and state.vertical_edges[action[1]][action[2]-1] and state.horizontal_edges[action[1]][action[2]-1] and state.horizontal_edges[action[1]+1][action[2]-1]):
                reward +=1
            if(action[2] < self._cols and state.vertical_edges[action[1]][action[2]+1] and state.horizontal_edges[action[1]][action[2]] and state.horizontal_edges[action[1]+1][action[2]]):
                reward +=1
        return reward

    def get_move_priority(self, action, state):
        """
        Move ordering heuristic for alpha-beta pruning.
        Returns a priority score — higher means search this move first.

        Priority levels:
          - Capture moves (completes a box):  100 + reward  (always first)
          - Safe moves (no danger created):   0
          - Dangerous moves (creates 3rd edge, gifting opponent a capture): -100 * dangers
        """
        # Captures are always top priority
        reward = self.get_reward_state(action, state)
        if reward > 0:
            return 100 + reward     # double captures ranked even higher

        # Check if this move creates a 3-sided box (giving opponent a free capture)
        dangers = self._count_dangers(action, state)
        if dangers > 0:
            return -100 * dangers   # more danger = lower priority

        return 0  # safe move

    def _count_dangers(self, action, state):
        """
        Count how many adjacent boxes would end up with exactly 3 edges
        after placing this action — giving the opponent a free capture
        on the very next move.

        For each adjacent box, count how many of the OTHER 3 edges are
        already placed.  If exactly 2 are present, placing this action
        creates the 3rd edge (danger).
        """
        danger = 0
        if action[0] == 0:                     # horizontal edge at row i, col j
            i, j = action[1], action[2]
            # Box ABOVE (i-1, j) — this edge is its bottom
            if i > 0:
                count = (int(state.horizontal_edges[i-1][j]) +
                         int(state.vertical_edges[i-1][j]) +
                         int(state.vertical_edges[i-1][j+1]))
                if count == 2:
                    danger += 1
            # Box BELOW (i, j) — this edge is its top
            if i < self._rows:
                count = (int(state.horizontal_edges[i+1][j]) +
                         int(state.vertical_edges[i][j]) +
                         int(state.vertical_edges[i][j+1]))
                if count == 2:
                    danger += 1
        else:                                   # vertical edge at row i, col j
            i, j = action[1], action[2]
            # Box to the LEFT (i, j-1) — this edge is its right side
            if j > 0:
                count = (int(state.vertical_edges[i][j-1]) +
                         int(state.horizontal_edges[i][j-1]) +
                         int(state.horizontal_edges[i+1][j-1]))
                if count == 2:
                    danger += 1
            # Box to the RIGHT (i, j) — this edge is its left side
            if j < self._cols:
                count = (int(state.vertical_edges[i][j+1]) +
                         int(state.horizontal_edges[i][j]) +
                         int(state.horizontal_edges[i+1][j]))
                if count == 2:
                    danger += 1
        return danger