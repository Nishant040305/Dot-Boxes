"""
PatchNet — Python-side hierarchical patch-based network for inference.

Architecture (mirrors C++ PatchNet.h):
  1. Extract overlapping KxK patches from the NxN board
  2. Run a FROZEN pre-trained KxK model on each patch
  3. Aggregate local policy/value signals into a spatial map
  4. A lightweight global network combines local signals with
     global features to produce the final policy + value.

This module is used for:
  - Loading models trained by the C++ pipeline
  - Python-based inference (arena, simulation UI, etc.)
  - Integration with the AlphaZeroCppAgent system
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os


class FCResidualBlock(nn.Module):
    """Fully-connected residual block (matches C++ FCResidualBlockImpl)."""
    def __init__(self, hidden_size, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = F.relu(self.ln1(self.fc1(x)))
        out = self.dropout(out)
        out = self.ln2(self.fc2(out))
        out += residual
        out = F.relu(out)
        return out


class PatchDesc:
    """Describes a single patch: its position and local→global action mapping."""
    def __init__(self, r0, c0, local_to_global):
        self.r0 = r0
        self.c0 = c0
        self.local_to_global = local_to_global  # list[int]: local_idx → global_idx


def compute_feature_size(rows, cols):
    """Compute the feature vector size for a board of given dimensions.
    Matches AlphaZeroBitNetImpl::preprocess layout."""
    n_h = (rows + 1) * cols
    n_v = rows * (cols + 1)
    n_box = rows * cols
    return n_h + n_v + n_box + n_box * 5 + 2


class AlphaZeroBitNet(nn.Module):
    """Standalone local model (matches C++ AlphaZeroBitNetImpl).
    Used as the frozen local patch model inside PatchNet."""
    def __init__(self, rows, cols, hidden_size=256, num_res_blocks=6, dropout=0.1):
        super().__init__()
        self.rows = rows
        self.cols = cols
        self.n_h_edges = (rows + 1) * cols
        self.n_v_edges = rows * (cols + 1)
        self.action_size = self.n_h_edges + self.n_v_edges
        self.input_size = compute_feature_size(rows, cols)

        # Input projection
        self.input_fc = nn.Linear(self.input_size, hidden_size)
        self.input_ln = nn.LayerNorm(hidden_size)

        # Residual tower
        self.res_blocks = nn.ModuleList([
            FCResidualBlock(hidden_size, dropout) for _ in range(num_res_blocks)
        ])

        # Policy head
        self.p_fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.p_ln = nn.LayerNorm(hidden_size // 2)
        self.p_fc2 = nn.Linear(hidden_size // 2, self.action_size)

        # Value head (deep: hidden → h/2 → h/4 → 1)
        self.v_fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.v_ln = nn.LayerNorm(hidden_size // 2)
        self.v_fc2 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.v_ln2 = nn.LayerNorm(hidden_size // 4)
        self.v_fc3 = nn.Linear(hidden_size // 4, 1)

    def forward(self, x):
        x = F.relu(self.input_ln(self.input_fc(x)))
        for block in self.res_blocks:
            x = block(x)
        # Policy
        p = F.relu(self.p_ln(self.p_fc1(x)))
        policy_logits = self.p_fc2(p)
        # Value
        v = F.relu(self.v_ln(self.v_fc1(x)))
        v = F.relu(self.v_ln2(self.v_fc2(v)))
        value = torch.tanh(self.v_fc3(v))
        return policy_logits, value


class PatchNet(nn.Module):
    """
    Hierarchical Patch-Based Network for Dots and Boxes.

    Uses a frozen pre-trained local model to extract tactical features
    from overlapping patches, then a trainable global aggregator combines
    these with global features to produce the final policy and value.

    Args:
        big_rows, big_cols: Full board dimensions (boxes)
        patch_rows, patch_cols: Local patch dimensions (boxes)
        local_hidden: Hidden size of the frozen local model
        local_blocks: Number of residual blocks in the local model
        global_hidden: Hidden size of the global aggregator
        global_blocks: Number of residual blocks in the global aggregator
        dropout: Dropout rate
    """

    def __init__(self, big_rows, big_cols,
                 patch_rows=3, patch_cols=3,
                 local_hidden=128, local_blocks=6,
                 global_hidden=192, global_blocks=4,
                 dropout=0.1):
        super().__init__()

        self.big_rows = big_rows
        self.big_cols = big_cols
        self.patch_rows = patch_rows
        self.patch_cols = patch_cols

        self.n_patches_r = big_rows - patch_rows + 1
        self.n_patches_c = big_cols - patch_cols + 1
        self.n_patches = self.n_patches_r * self.n_patches_c

        self.global_action_size = (big_rows + 1) * big_cols + big_rows * (big_cols + 1)
        self.local_action_size = (patch_rows + 1) * patch_cols + patch_rows * (patch_cols + 1)

        self.global_feat_size = compute_feature_size(big_rows, big_cols)
        self.local_feat_size = compute_feature_size(patch_rows, patch_cols)

        # ── Local model (FROZEN) ──
        self.local_model = AlphaZeroBitNet(
            patch_rows, patch_cols, local_hidden, local_blocks, dropout)

        # ── Global feature processor ──
        self.input_fc = nn.Linear(self.global_feat_size, global_hidden)
        self.input_ln = nn.LayerNorm(global_hidden)

        self.global_res_blocks = nn.ModuleList([
            FCResidualBlock(global_hidden, dropout) for _ in range(global_blocks)
        ])

        # ── Fusion layer ──
        fuse_input = global_hidden + self.global_action_size + self.n_patches
        self.fuse_fc = nn.Linear(fuse_input, global_hidden)
        self.fuse_ln = nn.LayerNorm(global_hidden)

        # ── Policy head ──
        self.p_fc1 = nn.Linear(global_hidden, global_hidden // 2)
        self.p_ln = nn.LayerNorm(global_hidden // 2)
        self.p_fc2 = nn.Linear(global_hidden // 2, self.global_action_size)

        # ── Value head ──
        self.v_fc1 = nn.Linear(global_hidden, global_hidden // 2)
        self.v_ln1 = nn.LayerNorm(global_hidden // 2)
        self.v_fc2 = nn.Linear(global_hidden // 2, global_hidden // 4)
        self.v_ln2 = nn.LayerNorm(global_hidden // 4)
        self.v_fc3 = nn.Linear(global_hidden // 4, 1)

        # ── Precompute patch mappings ──
        self.patches = self._precompute_patches()

        # Precompute overlap count tensor (constant)
        overlap = torch.zeros(self.global_action_size)
        for pd in self.patches:
            for g in pd.local_to_global:
                overlap[g] += 1.0
        self.register_buffer('overlap_count', overlap.clamp_min(1.0))

    def load_and_freeze_local(self, path):
        """Load pre-trained local model weights and freeze them."""
        state_dict = torch.load(path, map_location='cpu')
        self.local_model.load_state_dict(state_dict)
        for param in self.local_model.parameters():
            param.requires_grad = False
        self.local_model.eval()
        print(f"[PatchNet] Loaded & froze local model: {path}")

    def freeze_local(self):
        """Freeze local model parameters."""
        for param in self.local_model.parameters():
            param.requires_grad = False
        self.local_model.eval()

    @property
    def action_size(self):
        return self.global_action_size

    def combined_input_size(self):
        return self.global_feat_size + self.n_patches * self.local_feat_size

    def forward(self, x):
        """
        x: (batch, combined_input_size) — global features + patch features concatenated

        Returns: (policy_logits, value)
        """
        B = x.size(0)

        # ── 1. Split global and patch features ──
        global_feats = x[:, :self.global_feat_size]
        patch_feats = x[:, self.global_feat_size:]

        # ── 2. Run local model on all patches (no grad) ──
        with torch.no_grad():
            flat = patch_feats.reshape(B * self.n_patches, self.local_feat_size)
            local_logits, local_values = self.local_model(flat)
            local_logits = local_logits.reshape(B, self.n_patches, self.local_action_size)
            local_values = local_values.reshape(B, self.n_patches)

        # ── 3. Scatter local policy to global action space ──
        policy_map = torch.zeros(B, self.global_action_size, device=x.device)
        for p_idx, pd in enumerate(self.patches):
            patch_logit = local_logits[:, p_idx, :]  # (B, local_action_size)
            for l_idx, g_idx in enumerate(pd.local_to_global):
                policy_map[:, g_idx] += patch_logit[:, l_idx]

        # Average overlapping signals
        policy_map = policy_map / self.overlap_count.unsqueeze(0)

        # ── 4. Process global features through residual tower ──
        h = F.relu(self.input_ln(self.input_fc(global_feats)))
        for block in self.global_res_blocks:
            h = block(h)

        # ── 5. Fuse global hidden + local policy + local values ──
        fused = torch.cat([h, policy_map, local_values], dim=1)
        f = F.relu(self.fuse_ln(self.fuse_fc(fused)))

        # ── 6. Policy head ──
        p = F.relu(self.p_ln(self.p_fc1(f)))
        policy_logits = self.p_fc2(p)

        # ── 7. Value head ──
        v = F.relu(self.v_ln1(self.v_fc1(f)))
        v = F.relu(self.v_ln2(self.v_fc2(v)))
        value = torch.tanh(self.v_fc3(v))

        return policy_logits, value

    def preprocess(self, state):
        """
        Convert a BitBoardEnv state to the combined input tensor.
        Returns: torch.Tensor of shape (combined_input_size,)
        """
        # Global features
        global_feat = self._preprocess_board(state, self.big_rows, self.big_cols)

        # Patch features
        parts = [global_feat]
        for pd in self.patches:
            local_state = self._extract_patch(state, pd)
            local_feat = self._preprocess_board(local_state, self.patch_rows, self.patch_cols)
            parts.append(local_feat)

        return torch.cat(parts)

    def _preprocess_board(self, state, rows, cols):
        """Preprocess a board state into a flat feature tensor.
        Matches AlphaZeroBitNetImpl::preprocess layout."""
        n_h = (rows + 1) * cols
        n_v = rows * (cols + 1)
        n_box = rows * cols
        size = n_h + n_v + n_box + n_box * 5 + 2

        features = np.zeros(size, dtype=np.float32)
        current_player = state.current_player

        # h-edge bits
        for i in range(n_h):
            if state.h_edges & (1 << i):
                features[i] = 1.0

        # v-edge bits
        for i in range(n_v):
            if state.v_edges & (1 << i):
                features[n_h + i] = 1.0

        # Box ownership
        box_off = n_h + n_v
        for i in range(n_box):
            bit = 1 << i
            if state.boxes_p1 & bit:
                features[box_off + i] = 1.0 if current_player == 1 else -1.0
            elif state.boxes_p2 & bit:
                features[box_off + i] = 1.0 if current_player == 2 else -1.0

        # Edge-count one-hot (5 channels per box)
        cnt_off = box_off + n_box
        for r in range(rows):
            for c in range(cols):
                box_idx = r * cols + c
                count = 0
                if state.h_edges & (1 << (r * cols + c)):
                    count += 1
                if state.h_edges & (1 << ((r + 1) * cols + c)):
                    count += 1
                if state.v_edges & (1 << (r * (cols + 1) + c)):
                    count += 1
                if state.v_edges & (1 << (r * (cols + 1) + (c + 1))):
                    count += 1
                features[cnt_off + box_idx * 5 + count] = 1.0

        # Game-progress scalars
        p1_count = bin(state.boxes_p1).count('1')
        p2_count = bin(state.boxes_p2).count('1')
        filled = p1_count + p2_count
        total = n_box
        my_score = state.score[current_player - 1] if hasattr(state, 'score') else (
            p1_count if current_player == 1 else p2_count)
        opp_score = state.score[2 - current_player] if hasattr(state, 'score') else (
            p2_count if current_player == 1 else p1_count)
        prog_off = cnt_off + n_box * 5
        features[prog_off] = filled / max(total, 1)
        features[prog_off + 1] = (my_score - opp_score) / max(total, 1)

        return torch.from_numpy(features)

    def _extract_patch(self, state, pd):
        """Extract a local sub-board state from the full state."""
        r0, c0 = pd.r0, pd.c0
        p_rows, p_cols = self.patch_rows, self.patch_cols
        big_cols = self.big_cols

        # Build a simple namespace-like object for the local state
        loc_h = 0
        loc_v = 0
        loc_bp1 = 0
        loc_bp2 = 0
        loc_score_p1 = 0
        loc_score_p2 = 0

        # Copy h-edges
        for lr in range(p_rows + 1):
            for lc in range(p_cols):
                gr, gc = lr + r0, lc + c0
                gidx = gr * big_cols + gc
                if state.h_edges & (1 << gidx):
                    lidx = lr * p_cols + lc
                    loc_h |= (1 << lidx)

        # Copy v-edges
        for lr in range(p_rows):
            for lc in range(p_cols + 1):
                gr, gc = lr + r0, lc + c0
                gidx = gr * (big_cols + 1) + gc
                if state.v_edges & (1 << gidx):
                    lidx = lr * (p_cols + 1) + lc
                    loc_v |= (1 << lidx)

        # Copy box ownership
        for lr in range(p_rows):
            for lc in range(p_cols):
                gr, gc = lr + r0, lc + c0
                gidx = gr * big_cols + gc
                lidx = lr * p_cols + lc
                if state.boxes_p1 & (1 << gidx):
                    loc_bp1 |= (1 << lidx)
                    loc_score_p1 += 1
                if state.boxes_p2 & (1 << gidx):
                    loc_bp2 |= (1 << lidx)
                    loc_score_p2 += 1

        # Return a simple state object
        class LocalState:
            pass
        ls = LocalState()
        ls.h_edges = loc_h
        ls.v_edges = loc_v
        ls.boxes_p1 = loc_bp1
        ls.boxes_p2 = loc_bp2
        ls.current_player = state.current_player
        ls.score = [loc_score_p1, loc_score_p2]
        return ls

    def _precompute_patches(self):
        """Precompute all patch positions and local→global action mappings."""
        patches = []
        big_n_h_cols = self.big_cols
        big_h_total = (self.big_rows + 1) * self.big_cols
        loc_n_h_cols = self.patch_cols
        loc_n_v_cols = self.patch_cols + 1
        loc_n_h = (self.patch_rows + 1) * self.patch_cols
        loc_n_v = self.patch_rows * (self.patch_cols + 1)

        for pr in range(self.n_patches_r):
            for pc in range(self.n_patches_c):
                local_to_global = [0] * self.local_action_size

                # H-edges mapping
                for i in range(loc_n_h):
                    lr = i // loc_n_h_cols
                    lc = i % loc_n_h_cols
                    gr, gc = lr + pr, lc + pc
                    global_idx = gr * big_n_h_cols + gc
                    local_to_global[i] = global_idx

                # V-edges mapping
                big_n_v_cols = self.big_cols + 1
                for i in range(loc_n_v):
                    lr = i // loc_n_v_cols
                    lc = i % loc_n_v_cols
                    gr, gc = lr + pr, lc + pc
                    global_idx = big_h_total + gr * big_n_v_cols + gc
                    local_to_global[loc_n_h + i] = global_idx

                patches.append(PatchDesc(pr, pc, local_to_global))

        print(f"[PatchNet] {len(patches)} patches "
              f"({self.patch_rows}x{self.patch_cols} on "
              f"{self.big_rows}x{self.big_cols})")
        return patches


def load_patch_model(model_dir, device='cpu'):
    """
    Load a PatchNet model from a model directory containing model_info.json.

    Args:
        model_dir: Path to the model directory
        device: torch device

    Returns:
        PatchNet instance with loaded weights
    """
    info_path = os.path.join(model_dir, 'model_info.json')
    with open(info_path) as f:
        info = json.load(f)

    if not info.get('use_patch_net', False):
        raise ValueError(f"Model at {model_dir} is not a PatchNet model")

    model = PatchNet(
        big_rows=info['rows'],
        big_cols=info['cols'],
        patch_rows=info['patch_rows'],
        patch_cols=info['patch_cols'],
        local_hidden=info['local_hidden_size'],
        local_blocks=info['local_num_res_blocks'],
        global_hidden=info['global_hidden_size'],
        global_blocks=info['global_num_res_blocks'],
    )

    # Load full model weights (includes frozen local model)
    model_path = os.path.join(
        model_dir,
        f"alphazero_{info['rows']}x{info['cols']}.pt"
    )
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.freeze_local()
        print(f"[PatchNet] Loaded model from {model_path}")

    return model.to(device)
