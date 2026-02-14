class BaseBoardEnv:
    def __init__(self, N):
        self.N = N
        self.horizontal_edges = [[False for _ in range(N)] for _ in range(N+1)]
        self.vertical_edges = [[False for _ in range(N+1)] for _ in range(N)]
        self.boxes = [[0 for _ in range(N)] for _ in range(N)]
        self.current_player = 1
        self.done = False
        self.score = [0, 0]
        pass
    
    def reset(self):
        self.horizontal_edges = [[False for _ in range(self.N)] for _ in range(self.N+1)]
        self.vertical_edges = [[False for _ in range(self.N+1)] for _ in range(self.N)]
        self.boxes = [[0 for _ in range(self.N)] for _ in range(self.N)]
        self.current_player = 1
        self.done = False
        self.score = [0, 0]
    
    def step(self, action):
        # action is a tuple (type, i, j)
        # type is 0 for horizontal and 1 for vertical
        # i is the row index
        # j is the column index
        if(action[0] == 0):
            self.horizontal_edges[action[1]][action[2]] = True
        else:
            self.vertical_edges[action[1]][action[2]] = True
        
        # check if any box is made
        box_made = False
        reward = 0
        if(action[0] == 0):
            if(action[1] > 0 and self.horizontal_edges[action[1]-1][action[2]] and self.vertical_edges[action[1]-1][action[2]] and self.vertical_edges[action[1]-1][action[2]+1]):
                self.boxes[action[1]-1][action[2]] = self.current_player
                self.score[self.current_player-1] += 1
                reward +=1
                box_made = True
            if(action[1] < self.N and self.horizontal_edges[action[1]+1][action[2]] and self.vertical_edges[action[1]][action[2]] and self.vertical_edges[action[1]][action[2]+1]):
                self.boxes[action[1]][action[2]] = self.current_player
                self.score[self.current_player-1] += 1
                reward +=1
                box_made = True
        else:
            if(action[2] > 0 and self.vertical_edges[action[1]][action[2]-1] and self.horizontal_edges[action[1]][action[2]-1] and self.horizontal_edges[action[1]+1][action[2]-1]):
                self.boxes[action[1]][action[2]-1] = self.current_player
                self.score[self.current_player-1] += 1
                reward +=1
                box_made = True
            if(action[2] < self.N and self.vertical_edges[action[1]][action[2]+1] and self.horizontal_edges[action[1]][action[2]] and self.horizontal_edges[action[1]+1][action[2]]):
                self.boxes[action[1]][action[2]] = self.current_player
                self.score[self.current_player-1] += 1
                reward +=1
                box_made = True
        
        # check if game is done
        if(self.score[0] + self.score[1] == self.N * self.N):
            self.done = True
        
        # switch player
        if(box_made==False):
            self.current_player = 3 - self.current_player
        
        return self.boxes, self.score, self.done, reward
    def _get_available_actions(self):
        available_actions = []
        #horizontal edges
        for i in range(self.N+1):
            for j in range(self.N):
                if(self.horizontal_edges[i][j] == False):
                    available_actions.append((0, i, j))
        #vertical edges
        for i in range(self.N):
            for j in range(self.N+1):
                if(self.vertical_edges[i][j] == False):
                    available_actions.append((1, i, j))
        return available_actions
    def render(self):
        pass
    
    def close(self):
        pass
