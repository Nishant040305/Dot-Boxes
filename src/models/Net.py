import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AlphaZeroNet(nn.Module):
    def __init__(self, N, dropout=0.3):
        super(AlphaZeroNet, self).__init__()
        self.N = N
        self.grid_size = N + 1
        
        # Configuration
        self.channels = 64 
        
        # Convolutional Backbone
        self.conv1 = nn.Conv2d(3, self.channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(self.channels)
        
        # Residual Tower (4 blocks)
        self.res1 = ResidualBlock(self.channels)
        self.res2 = ResidualBlock(self.channels)
        self.res3 = ResidualBlock(self.channels)
        self.res4 = ResidualBlock(self.channels)
        
        # Policy Head
        self.p_conv = nn.Conv2d(self.channels, 2, kernel_size=1) 
        self.p_bn = nn.BatchNorm2d(2)
        self.p_fc = nn.Linear(2 * self.grid_size * self.grid_size, (N + 1) * N + N * (N + 1))
        
        # Value Head
        self.v_conv = nn.Conv2d(self.channels, 1, kernel_size=1)
        self.v_bn = nn.BatchNorm2d(1)
        self.v_fc1 = nn.Linear(self.grid_size * self.grid_size, 64)
        self.v_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # x shape: (batch_size, 3, N+1, N+1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        
        # Policy Head
        p = F.relu(self.p_bn(self.p_conv(x)))
        p = p.view(p.size(0), -1)
        policy_logits = self.p_fc(p)
        # policy = F.softmax(p_logits, dim=1) # REMOVED SOFTMAX - Return Logits
        
        # Value Head
        v = F.relu(self.v_bn(self.v_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.v_fc1(v))
        value = torch.tanh(self.v_fc2(v))
        
        return policy_logits, value


    def preprocess(self, state):
        """
        Convert State object to tensor.
        Perspective: Always from the view of the current player.
        Output Shape: (3, N+1, N+1)
        """
        current_player = state.current_player
        grid_size = self.N + 1
        
        # Channel 0: Horizontal Edges
        # shape (N+1, N). We pad to (N+1, N+1)
        # We can place them in the first N columns
        h_channel = np.zeros((grid_size, grid_size), dtype=np.float32)
        h_edges = np.array(state.horizontal_edges, dtype=np.float32) # (N+1, N)
        h_channel[:grid_size, :self.N] = h_edges

        # Channel 1: Vertical Edges
        # shape (N, N+1). We pad to (N+1, N+1)
        # We can place them in the first N rows
        v_channel = np.zeros((grid_size, grid_size), dtype=np.float32)
        v_edges = np.array(state.vertical_edges, dtype=np.float32) # (N, N+1)
        v_channel[:self.N, :grid_size] = v_edges
        
        # Channel 2: Boxes (ownership)
        # shape (N, N). We resize/embed to (N+1, N+1)
        # Place in top-left
        b_channel = np.zeros((grid_size, grid_size), dtype=np.float32)
        for r in range(self.N):
            for c in range(self.N):
                owner = state.boxes[r][c]
                if owner == current_player:
                    b_channel[r][c] = 1.0
                elif owner != 0:
                    b_channel[r][c] = -1.0
        
        # Stack channels
        input_tensor = np.stack([h_channel, v_channel, b_channel]) # (3, N+1, N+1)
        return torch.tensor(input_tensor)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out

