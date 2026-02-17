"""
Bitboard-based Dots and Boxes environment.

Edge Layout (for NxN board):
  - Horizontal edges: (N+1) rows × N cols = (N+1)*N bits
    Bit index for h_edge(r, c) = r * N + c
  - Vertical edges: N rows × (N+1) cols = N*(N+1) bits
    Bit index for v_edge(r, c) = r * (N+1) + c
    
Box(r, c) needs 4 edges:
  - Top:    h_edge(r, c)
  - Bottom: h_edge(r+1, c)
  - Left:   v_edge(r, c)
  - Right:  v_edge(r, c+1)
"""

import copy


class BitBoardEnv:
    """
    Bitboard-based Dots and Boxes environment.
    
    All edges are stored as bits in integers for maximum performance.
    Cloning is a simple int copy, legal moves use bit tricks, and
    box completion checks are pure bitwise AND operations.
    """
    
    # Class-level cache for precomputed masks per board size
    _mask_cache = {}
    
    def __init__(self, N):
        self.N = N
        self.n_h_edges = (N + 1) * N      # horizontal edge count
        self.n_v_edges = N * (N + 1)       # vertical edge count
        self.total_edges = self.n_h_edges + self.n_v_edges
        self.total_boxes = N * N
        
        # Precompute masks for this board size
        if N not in BitBoardEnv._mask_cache:
            BitBoardEnv._precompute_masks(N)
        self._masks = BitBoardEnv._mask_cache[N]
        
        # State: all stored as integers
        self.h_edges = 0          # horizontal edges bitmask
        self.v_edges = 0          # vertical edges bitmask
        self.boxes_p1 = 0         # boxes owned by player 1
        self.boxes_p2 = 0         # boxes owned by player 2
        self.current_player = 1
        self.done = False
        self.score = [0, 0]
        
        # For compatibility with existing code
        self.action_history = []
        self.state_history = []
    
    @staticmethod
    def _precompute_masks(N):
        """Precompute bitmasks for box completion checks."""
        masks = {}
        
        # For each box (r, c), precompute which bits in h_edges and v_edges
        # need to be set for the box to be complete
        box_h_masks = {}  # box (r,c) -> h_edges mask (top + bottom)
        box_v_masks = {}  # box (r,c) -> v_edges mask (left + right)
        
        for r in range(N):
            for c in range(N):
                # Top: h_edge(r, c), Bottom: h_edge(r+1, c)
                h_top = r * N + c
                h_bot = (r + 1) * N + c
                box_h_masks[(r, c)] = (1 << h_top) | (1 << h_bot)
                
                # Left: v_edge(r, c), Right: v_edge(r, c+1)
                v_left = r * (N + 1) + c
                v_right = r * (N + 1) + (c + 1)
                box_v_masks[(r, c)] = (1 << v_left) | (1 << v_right)
        
        # For each edge, which boxes does it border?
        # Horizontal edge (r, c) borders box above (r-1, c) and below (r, c)
        h_edge_adj_boxes = {}
        for r in range(N + 1):
            for c in range(N):
                adj = []
                if r > 0:       # box above
                    adj.append((r - 1, c))
                if r < N:       # box below
                    adj.append((r, c))
                h_edge_adj_boxes[(r, c)] = adj
        
        # Vertical edge (r, c) borders box left (r, c-1) and right (r, c)
        v_edge_adj_boxes = {}
        for r in range(N):
            for c in range(N + 1):
                adj = []
                if c > 0:       # box left
                    adj.append((r, c - 1))
                if c < N:       # box right
                    adj.append((r, c))
                v_edge_adj_boxes[(r, c)] = adj
        
        # All-ones masks for iterating legal moves
        all_h = (1 << ((N + 1) * N)) - 1
        all_v = (1 << (N * (N + 1))) - 1
        
        masks['box_h_masks'] = box_h_masks
        masks['box_v_masks'] = box_v_masks
        masks['h_edge_adj_boxes'] = h_edge_adj_boxes
        masks['v_edge_adj_boxes'] = v_edge_adj_boxes
        masks['all_h'] = all_h
        masks['all_v'] = all_v
        
        BitBoardEnv._mask_cache[N] = masks
    
    def reset(self):
        self.h_edges = 0
        self.v_edges = 0
        self.boxes_p1 = 0
        self.boxes_p2 = 0
        self.current_player = 1
        self.done = False
        self.score = [0, 0]
        self.action_history = []
        self.state_history = []
        self.state_history.append(self.get_state_snapshot(None))
    
    def step(self, action):
        """
        action: (type, r, c) where type=0 is horizontal, type=1 is vertical
        Returns: (boxes_p1, boxes_p2, score, done, reward)
        """
        player_before = self.current_player
        edge_type, r, c = action
        
        # Place edge
        if edge_type == 0:
            bit_idx = r * self.N + c
            self.h_edges |= (1 << bit_idx)
            adj_boxes = self._masks['h_edge_adj_boxes'][(r, c)]
        else:
            bit_idx = r * (self.N + 1) + c
            self.v_edges |= (1 << bit_idx)
            adj_boxes = self._masks['v_edge_adj_boxes'][(r, c)]
        
        # Check if any adjacent box is completed
        box_made = False
        reward = 0
        box_h_masks = self._masks['box_h_masks']
        box_v_masks = self._masks['box_v_masks']
        
        for box_r, box_c in adj_boxes:
            h_mask = box_h_masks[(box_r, box_c)]
            v_mask = box_v_masks[(box_r, box_c)]
            
            # Box is complete if all 4 edges are set
            if (self.h_edges & h_mask) == h_mask and (self.v_edges & v_mask) == v_mask:
                # Check if box was already claimed
                box_bit = 1 << (box_r * self.N + box_c)
                if not (self.boxes_p1 & box_bit) and not (self.boxes_p2 & box_bit):
                    if self.current_player == 1:
                        self.boxes_p1 |= box_bit
                    else:
                        self.boxes_p2 |= box_bit
                    self.score[self.current_player - 1] += 1
                    reward += 1
                    box_made = True
        
        # Check if game is done
        if self.score[0] + self.score[1] == self.total_boxes:
            self.done = True
        
        # Switch player if no box was made
        if not box_made:
            self.current_player = 3 - self.current_player
        
        # Record history
        self.action_history.append({
            "action": list(action),
            "player": player_before,
            "box_made": box_made,
            "reward": reward,
        })
        self.state_history.append(self.get_state_snapshot(action))
        
        return self.boxes_p1, self.boxes_p2, self.score, self.done, reward
    
    def _get_available_actions(self):
        """Get list of all legal actions using bit tricks."""
        actions = []
        N = self.N
        
        # Horizontal edges: find unset bits
        free_h = self._masks['all_h'] & ~self.h_edges
        while free_h:
            bit = free_h & (-free_h)  # lowest set bit
            idx = bit.bit_length() - 1
            r, c = divmod(idx, N)
            actions.append((0, r, c))
            free_h ^= bit
        
        # Vertical edges: find unset bits
        free_v = self._masks['all_v'] & ~self.v_edges
        while free_v:
            bit = free_v & (-free_v)  # lowest set bit
            idx = bit.bit_length() - 1
            r, c = divmod(idx, N + 1)
            actions.append((1, r, c))
            free_v ^= bit
        
        return actions
    
    def get_legal_actions_mask(self):
        """Return a bitmask of all legal actions as a single integer.
        First n_h_edges bits are horizontal, next n_v_edges bits are vertical."""
        free_h = self._masks['all_h'] & ~self.h_edges
        free_v = self._masks['all_v'] & ~self.v_edges
        return free_h | (free_v << self.n_h_edges)
    
    def clone(self):
        """Ultra-fast clone — just copy integers."""
        new_env = BitBoardEnv.__new__(BitBoardEnv)
        new_env.N = self.N
        new_env.n_h_edges = self.n_h_edges
        new_env.n_v_edges = self.n_v_edges
        new_env.total_edges = self.total_edges
        new_env.total_boxes = self.total_boxes
        new_env._masks = self._masks  # shared reference (immutable)
        new_env.h_edges = self.h_edges
        new_env.v_edges = self.v_edges
        new_env.boxes_p1 = self.boxes_p1
        new_env.boxes_p2 = self.boxes_p2
        new_env.current_player = self.current_player
        new_env.done = self.done
        new_env.score = self.score[:]
        new_env.action_history = []
        new_env.state_history = []
        return new_env
    
    def get_state_snapshot(self, action):
        """Return a JSON-serializable snapshot."""
        return {
            "h_edges": self.h_edges,
            "v_edges": self.v_edges,
            "boxes_p1": self.boxes_p1,
            "boxes_p2": self.boxes_p2,
            "current_player": self.current_player,
            "score": list(self.score),
            "done": self.done,
            "action": list(action) if action else None,
        }
    
    def action_to_index(self, action):
        """Convert (type, r, c) to a flat index."""
        edge_type, r, c = action
        if edge_type == 0:
            return r * self.N + c
        else:
            return self.n_h_edges + r * (self.N + 1) + c
    
    def index_to_action(self, idx):
        """Convert flat index back to (type, r, c)."""
        if idx < self.n_h_edges:
            r, c = divmod(idx, self.N)
            return (0, r, c)
        else:
            idx -= self.n_h_edges
            r, c = divmod(idx, self.N + 1)
            return (1, r, c)
    
    def does_complete_box(self, action):
        """Check if placing this edge would complete at least one box."""
        edge_type, r, c = action
        
        if edge_type == 0:
            # Simulate placing horizontal edge
            new_h = self.h_edges | (1 << (r * self.N + c))
            adj_boxes = self._masks['h_edge_adj_boxes'][(r, c)]
        else:
            new_h = self.h_edges
            adj_boxes = self._masks['v_edge_adj_boxes'][(r, c)]
        
        new_v = self.v_edges
        if edge_type == 1:
            new_v = self.v_edges | (1 << (r * (self.N + 1) + c))
        
        box_h_masks = self._masks['box_h_masks']
        box_v_masks = self._masks['box_v_masks']
        
        for box_r, box_c in adj_boxes:
            h_mask = box_h_masks[(box_r, box_c)]
            v_mask = box_v_masks[(box_r, box_c)]
            box_bit = 1 << (box_r * self.N + box_c)
            
            if ((new_h & h_mask) == h_mask and 
                (new_v & v_mask) == v_mask and
                not (self.boxes_p1 & box_bit) and 
                not (self.boxes_p2 & box_bit)):
                return True
        return False
    
    def count_box_edges(self, box_r, box_c):
        """Count how many edges box (r,c) currently has placed."""
        h_mask = self._masks['box_h_masks'][(box_r, box_c)]
        v_mask = self._masks['box_v_masks'][(box_r, box_c)]
        return bin(self.h_edges & h_mask).count('1') + bin(self.v_edges & v_mask).count('1')
    
    def get_edge_count_for_adjacent_boxes(self, action):
        """
        For move ordering: returns how many edges each adjacent box has
        BEFORE placing this edge. If any box has 2 edges, placing this
        creates a dangerous 3rd edge.
        """
        edge_type, r, c = action
        if edge_type == 0:
            adj_boxes = self._masks['h_edge_adj_boxes'][(r, c)]
        else:
            adj_boxes = self._masks['v_edge_adj_boxes'][(r, c)]
        
        return [self.count_box_edges(br, bc) for br, bc in adj_boxes]
    
    # --- Compatibility properties to look like BaseBoardEnv ---
    
    @property
    def horizontal_edges(self):
        """Reconstruct 2D array from bitmask (for compatibility)."""
        N = self.N
        result = [[False] * N for _ in range(N + 1)]
        for r in range(N + 1):
            for c in range(N):
                if self.h_edges & (1 << (r * N + c)):
                    result[r][c] = True
        return result
    
    @property
    def vertical_edges(self):
        """Reconstruct 2D array from bitmask (for compatibility)."""
        N = self.N
        result = [[False] * (N + 1) for _ in range(N)]
        for r in range(N):
            for c in range(N + 1):
                if self.v_edges & (1 << (r * (N + 1) + c)):
                    result[r][c] = True
        return result
    
    @property
    def boxes(self):
        """Reconstruct 2D array from bitmasks (for compatibility)."""
        N = self.N
        result = [[0] * N for _ in range(N)]
        for r in range(N):
            for c in range(N):
                bit = 1 << (r * N + c)
                if self.boxes_p1 & bit:
                    result[r][c] = 1
                elif self.boxes_p2 & bit:
                    result[r][c] = 2
        return result
    
    def render(self):
        """Render the board in ASCII art."""
        N = self.N
        print(f"\n  Player {self.current_player}'s turn | Score: P1={self.score[0]} P2={self.score[1]}")
        print("  " + "-" * (4 * N + 1))
        
        for i in range(N + 1):
            row = "  "
            for j in range(N):
                row += "•"
                if self.h_edges & (1 << (i * N + j)):
                    row += "———"
                else:
                    row += "   "
            row += "•"
            print(row)
            
            if i < N:
                row = "  "
                for j in range(N + 1):
                    if self.v_edges & (1 << (i * (N + 1) + j)):
                        row += "│"
                    else:
                        row += " "
                    if j < N:
                        box_bit = 1 << (i * N + j)
                        if self.boxes_p1 & box_bit:
                            row += " 1 "
                        elif self.boxes_p2 & box_bit:
                            row += " 2 "
                        else:
                            row += "   "
                print(row)
        
        print("  " + "-" * (4 * N + 1))
        if self.done:
            if self.score[0] > self.score[1]:
                print("  Player 1 wins!")
            elif self.score[1] > self.score[0]:
                print("  Player 2 wins!")
            else:
                print("  It's a draw!")
        print()
    
    def close(self):
        pass
