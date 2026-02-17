"""
Neural network for bitboard-based AlphaZero.

Input: Flat feature vector from bitboard state rather than 2D grid.
This uses a fully-connected residual architecture (no convolutions needed)
since the bitboard representation is inherently flat.

Feature channels (concatenated into a single vector):
  - Horizontal edge bits:  (N+1)*N values  (0 or 1)
  - Vertical edge bits:    N*(N+1) values  (0 or 1)
  - Box ownership:         N*N values      (-1, 0, or 1 from current player perspective)
  - Current player flag:   1 value         (1 if player 1, -1 if player 2)

Total input size = (N+1)*N + N*(N+1) + N*N + 1 = 2*N*(N+1) + N*N + 1
For N=3: 12 + 12 + 9 + 1 = 34
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FCResidualBlock(nn.Module):
    """Fully-connected residual block."""
    def __init__(self, hidden_size, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.bn1 = nn.LayerNorm(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.fc1(x)))
        out = self.dropout(out)
        out = self.bn2(self.fc2(out))
        out += residual
        out = F.relu(out)
        return out


class AlphaZeroBitNet(nn.Module):
    def __init__(self, N, hidden_size=256, num_res_blocks=6, dropout=0.1):
        super().__init__()
        self.N = N
        self.n_h_edges = (N + 1) * N
        self.n_v_edges = N * (N + 1)
        self.action_size = self.n_h_edges + self.n_v_edges
        
        # Input: h_edges + v_edges + box_ownership + player_flag
        self.input_size = self.n_h_edges + self.n_v_edges + N * N + 1
        
        
        # Input projection
        self.input_fc = nn.Linear(self.input_size, hidden_size)
        self.input_bn = nn.LayerNorm(hidden_size)
        
        # Residual tower
        self.res_blocks = nn.ModuleList([
            FCResidualBlock(hidden_size, dropout) for _ in range(num_res_blocks)
        ])
        
        # Policy head
        self.p_fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.p_bn = nn.LayerNorm(hidden_size // 2)
        self.p_fc2 = nn.Linear(hidden_size // 2, self.action_size)
        
        # Value head
        self.v_fc1 = nn.Linear(hidden_size, hidden_size // 4)
        self.v_bn = nn.LayerNorm(hidden_size // 4)
        self.v_fc2 = nn.Linear(hidden_size // 4, 1)
    
    def forward(self, x):
        """
        x: (batch_size, input_size) flat feature vector
        Returns: (policy_logits, value)
        """
        # Input projection
        x = F.relu(self.input_bn(self.input_fc(x)))
        
        # Residual tower
        for block in self.res_blocks:
            x = block(x)
        
        # Policy head
        p = F.relu(self.p_bn(self.p_fc1(x)))
        policy_logits = self.p_fc2(p)
        
        # Value head
        v = F.relu(self.v_bn(self.v_fc1(x)))
        value = torch.tanh(self.v_fc2(v))
        
        return policy_logits, value
    
    def preprocess(self, state):
        """
        Convert a BitBoardEnv state to a flat feature tensor.
        
        Args:
            state: A BitBoardEnv instance (or clone)
            
        Returns:
            torch.Tensor of shape (input_size,)
        """
        N = self.N
        current_player = state.current_player
        
        # Extract horizontal edge bits
        h_features = np.zeros(self.n_h_edges, dtype=np.float32)
        for i in range(self.n_h_edges):
            if state.h_edges & (1 << i):
                h_features[i] = 1.0
        
        # Extract vertical edge bits
        v_features = np.zeros(self.n_v_edges, dtype=np.float32)
        for i in range(self.n_v_edges):
            if state.v_edges & (1 << i):
                v_features[i] = 1.0
        
        # Box ownership from current player's perspective
        box_features = np.zeros(N * N, dtype=np.float32)
        for r in range(N):
            for c in range(N):
                bit = 1 << (r * N + c)
                if state.boxes_p1 & bit:
                    box_features[r * N + c] = 1.0 if current_player == 1 else -1.0
                elif state.boxes_p2 & bit:
                    box_features[r * N + c] = 1.0 if current_player == 2 else -1.0
        
        # Current player flag
        player_flag = np.array([1.0 if current_player == 1 else -1.0], dtype=np.float32)
        
        # Concatenate all features
        features = np.concatenate([h_features, v_features, box_features, player_flag])
        return torch.from_numpy(features)
    
    def preprocess_batch_fast(self, h_edges_list, v_edges_list, 
                              boxes_p1_list, boxes_p2_list, players_list):
        """
        Vectorized preprocessing for a batch of states.
        Avoids Python loops for each state.
        
        Args:
            Lists of integer bitmasks and player values.
            
        Returns:
            torch.Tensor of shape (batch_size, input_size)
        """
        batch_size = len(h_edges_list)
        N = self.N
        features = np.zeros((batch_size, self.input_size), dtype=np.float32)
        
        for b in range(batch_size):
            h = h_edges_list[b]
            v = v_edges_list[b]
            p1 = boxes_p1_list[b]
            p2 = boxes_p2_list[b]
            player = players_list[b]
            
            # Horizontal edges
            for i in range(self.n_h_edges):
                if h & (1 << i):
                    features[b, i] = 1.0
            
            # Vertical edges
            offset = self.n_h_edges
            for i in range(self.n_v_edges):
                if v & (1 << i):
                    features[b, offset + i] = 1.0
            
            # Box ownership
            offset2 = offset + self.n_v_edges
            for i in range(N * N):
                if p1 & (1 << i):
                    features[b, offset2 + i] = 1.0 if player == 1 else -1.0
                elif p2 & (1 << i):
                    features[b, offset2 + i] = 1.0 if player == 2 else -1.0
            
            # Player flag
            features[b, -1] = 1.0 if player == 1 else -1.0
        
        return torch.from_numpy(features)
