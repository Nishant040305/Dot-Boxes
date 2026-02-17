"""
AlphaZero Agent using Bitboard representation.

All game state manipulation uses integer bitmasks for speed.
No 2D array operations — cloning is int copy, legal moves use
bit tricks, box checks are bitwise AND.

Optimizations over base version:
  - FPU (First Play Urgency) in PUCT selection
  - Dirichlet noise at root for exploration
  - Adaptive heuristic boosting: capture > safe > sacrifice
  - Safe move detection via 3-edge box check (bitwise)
"""

import numpy as np
import torch
import math
import random
from agents.agent import Agent
from models.BitNet import AlphaZeroBitNet


class BitNode:
    """MCTS Node with minimal overhead."""
    __slots__ = ['h_edges', 'v_edges', 'boxes_p1', 'boxes_p2', 
                 'current_player', 'done', 'score_p1', 'score_p2',
                 'parent', 'children', 'visits', 'value_sum', 'prior']
    
    def __init__(self, state, parent=None, prior=0.0):
        # Store raw ints instead of env object
        self.h_edges = state.h_edges
        self.v_edges = state.v_edges
        self.boxes_p1 = state.boxes_p1
        self.boxes_p2 = state.boxes_p2
        self.current_player = state.current_player
        self.done = state.done
        self.score_p1 = state.score[0]
        self.score_p2 = state.score[1]
        
        self.parent = parent
        self.children = {}  # action -> BitNode
        self.visits = 0
        self.value_sum = 0.0
        self.prior = prior
    
    @property
    def score(self):
        return [self.score_p1, self.score_p2]
    
    def is_expanded(self):
        return len(self.children) > 0
    
    def value(self):
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits


class AlphaZeroBitAgent(Agent):
    """AlphaZero agent using bitboard environment."""
    
    def __init__(self, env, model_path=None, n_simulations=400, c_puct=1.5, 
                 device='cpu', dirichlet_alpha=0.3, dirichlet_epsilon=0.25,
                 fpu_reduction=0.25, checkpoint=False, profiler=None,add_noise=True):
        super().__init__(env)
        self.device = device
        self.profiler = profiler
        self.n = env.N
        self.n_simulations = n_simulations
        self.c_puct = c_puct
        #Drichlets noice
        self.add_noise = add_noise
        # FPU: First Play Urgency
        self.fpu_reduction = fpu_reduction
        
        # Dirichlet noise for root exploration
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        
        # Precomputed masks (shared with env)
        self._masks = env._masks
        
        # Precompute 3-edge masks for safe move detection
        self._precompute_edge_count_masks()
        
        # Initialize model
        self.model = AlphaZeroBitNet(self.n).to(self.device)
        if model_path:
            try:
                if(checkpoint):
                    model_checkpoint = torch.load(model_path,map_location=self.device)
                    self.model.load_state_dict(model_checkpoint['model_state_dict'])
                else:
                    self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"Loaded AlphaZero BitBoard model from {model_path}")
            except FileNotFoundError:
                print(f"Model {model_path} not found, using random weights.")
        
        self.model.eval()
    
    def _precompute_edge_count_masks(self):
        """Precompute individual edge bit positions for each box.
        Used for fast edge-count-around-box checks (safe move detection)."""
        N = self.n
        self._box_edge_bits = {}
        for r in range(N):
            for c in range(N):
                h_top = r * N + c
                h_bot = (r + 1) * N + c
                v_left = r * (N + 1) + c
                v_right = r * (N + 1) + (c + 1)
                self._box_edge_bits[(r, c)] = {
                    'h_bits': [h_top, h_bot],
                    'v_bits': [v_left, v_right],
                }
    
    def act(self, return_probs=False, temperature=1e-3, add_noise=True):
        if(not self.add_noise):
            add_noise = self.add_noise
        """
        Run MCTS and return action.
        If return_probs=True, returns (action, visit_count_dict).
        """
        root = BitNode(self.env, prior=1.0)
        
        # Expand root
        self._evaluate_and_expand(root, is_root=True, add_noise=add_noise)
        
        # Run simulations
        for _ in range(self.n_simulations):
            node = root
            
            # 1. Selection
            if self.profiler: self.profiler.start('mcts_select')
            while node.is_expanded():
                action, node = self._select_child(node)
            if self.profiler: self.profiler.stop('mcts_select')
            
            # 2. Expansion and Evaluation
            if self.profiler: self.profiler.start('mcts_eval_expand')
            value = self._evaluate_and_expand(node)
            if self.profiler: self.profiler.stop('mcts_eval_expand')
            
            # 3. Backpropagation
            if self.profiler: self.profiler.start('mcts_backprop')
            self._backpropagate(node, value)
            if self.profiler: self.profiler.stop('mcts_backprop')
        
        if not root.children:
            return None
        
        # Build visit counts
        visit_counts = {action: child.visits for action, child in root.children.items()}
        
        # Temperature sampling
        if temperature < 1e-3:
            best_action = self._get_best_action(root)
        else:
            actions = list(visit_counts.keys())
            visits = np.array([visit_counts[a] for a in actions], dtype=np.float64)
            log_visits = np.log(visits + 1e-10) / temperature
            log_visits -= np.max(log_visits)
            probs = np.exp(log_visits)
            probs /= np.sum(probs)
            best_action = actions[np.random.choice(len(actions), p=probs)]
        
        if return_probs:
            return best_action, visit_counts
        return best_action
    
    def _select_child(self, node):
        """PUCT selection with First Play Urgency (FPU)."""
        best_score = -float('inf')
        best_action = None
        best_child = None
        
        sum_visits = sum(child.visits for child in node.children.values())
        sqrt_sum = math.sqrt(sum_visits + 1)  # +1 for numerical stability
        
        # FPU: unvisited nodes get parent's value minus a reduction
        parent_q = node.value() if node.visits > 0 else 0.0
        fpu_value = parent_q - self.fpu_reduction
        
        for action, child in node.children.items():
            # Q-value with FPU for unvisited nodes
            if child.visits == 0:
                q_value = fpu_value
            else:
                q_value = child.value()
            
            # Perspective flip based on player change
            if child.current_player == node.current_player:
                action_value = q_value
            else:
                action_value = -q_value
            
            u_score = self.c_puct * child.prior * sqrt_sum / (1 + child.visits)
            score = action_value + u_score
            
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        
        return best_action, best_child
    
    def _evaluate_and_expand(self, node, is_root=False, add_noise=True):
        """Evaluate leaf node with NN and expand.
        
        Enhanced with:
          - Dirichlet noise at root for exploration
          - Adaptive heuristic: capture (5x), safe (2x when no captures),
            sacrifice (0.1x)
        """
        # Terminal check
        if node.done:
            my_score = node.score[node.current_player - 1]
            opp_score = node.score[2 - node.current_player]
            if my_score > opp_score:
                return 1.0
            elif my_score < opp_score:
                return -1.0
            return 0.0
        
        # Build feature vector directly from node ints
        features = self._node_to_tensor(node)
        
        if self.profiler: self.profiler.start('nn_inference_wait')
        with torch.no_grad():
            policy_logits, value = self.model(features.unsqueeze(0).to(self.device))
        if self.profiler: self.profiler.stop('nn_inference_wait')
        
        # Squeeze any extra dimensions — RemoteBitModel returns [1, 1, action_size]
        # due to double-batching; local model returns [1, action_size]
        policy_logits = policy_logits.squeeze()  # → (action_size,)
        value_scalar = value.squeeze().item()
        
        policy_probs = torch.softmax(policy_logits, dim=-1).cpu().numpy()
        # Ensure 1D
        if policy_probs.ndim > 1:
            policy_probs = policy_probs.reshape(-1)
        
        # Get legal actions
        legal_actions = self._get_legal_actions_from_node(node)
        
        # Classify moves: capture, safe, sacrifice
        capture_moves = set()
        sacrifice_moves = set()
        safe_moves = set()
        
        for action in legal_actions:
            n_completed = self._count_boxes_completed_node(node, action)
            if n_completed > 0:
                capture_moves.add(action)
            elif self._creates_capturable_box_node(node, action):
                sacrifice_moves.add(action)
            else:
                safe_moves.add(action)
        
        # Build children with adaptive priors
        n_h = self.model.n_h_edges
        N = self.n
        
        children_priors = {}
        total_prior = 0.0
        
        for action in legal_actions:
            edge_type, r, c = action
            if edge_type == 0:
                idx = r * N + c
            else:
                idx = n_h + r * (N + 1) + c
            
            prior = policy_probs[idx]
            
            # # Adaptive heuristic boosting
            # if action in capture_moves:
            #     prior *= 5.0  # Strong boost for captures
            # elif action in safe_moves and len(capture_moves) == 0:
            #     prior *= 2.0  # Boost safe when no captures available
            # elif action in sacrifice_moves:
            #     prior *= 0.1  # Heavily penalize creating capturable boxes
            
            children_priors[action] = prior
            total_prior += prior
        
        # Normalize priors
        if total_prior > 0:
            for action in legal_actions:
                children_priors[action] /= total_prior
        else:
            uniform = 1.0 / len(legal_actions)
            for action in legal_actions:
                children_priors[action] = uniform
        
        # Dirichlet noise at root for exploration
        if is_root and add_noise and len(legal_actions) > 0:
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(legal_actions))
            eps = self.dirichlet_epsilon
            for i, action in enumerate(legal_actions):
                children_priors[action] = (1 - eps) * children_priors[action] + eps * noise[i]
        
        # Create child nodes
        for action in legal_actions:
            child_state = self._apply_action_to_node(node, action)
            child = BitNode(child_state, parent=node, prior=children_priors[action])
            node.children[action] = child
        
        return value_scalar
    
    def _count_boxes_completed_node(self, node, action):
        """Count how many boxes placing this edge would complete."""
        edge_type, r, c = action
        N = self.n
        
        if edge_type == 0:
            new_h = node.h_edges | (1 << (r * N + c))
            new_v = node.v_edges
            adj_boxes = self._masks['h_edge_adj_boxes'][(r, c)]
        else:
            new_h = node.h_edges
            new_v = node.v_edges | (1 << (r * (N + 1) + c))
            adj_boxes = self._masks['v_edge_adj_boxes'][(r, c)]
        
        box_h_masks = self._masks['box_h_masks']
        box_v_masks = self._masks['box_v_masks']
        
        count = 0
        for box_r, box_c in adj_boxes:
            h_mask = box_h_masks[(box_r, box_c)]
            v_mask = box_v_masks[(box_r, box_c)]
            box_bit = 1 << (box_r * N + box_c)
            if ((new_h & h_mask) == h_mask and 
                (new_v & v_mask) == v_mask and
                not (node.boxes_p1 & box_bit) and 
                not (node.boxes_p2 & box_bit)):
                count += 1
        return count
    
    def _creates_capturable_box_node(self, node, action):
        """Check if placing this edge creates a box with exactly 3 edges
        (making it capturable by the opponent on their next move).
        
        All bitwise — no array operations."""
        edge_type, r, c = action
        N = self.n
        
        # Simulate placing the edge
        if edge_type == 0:
            new_h = node.h_edges | (1 << (r * N + c))
            new_v = node.v_edges
            adj_boxes = self._masks['h_edge_adj_boxes'][(r, c)]
        else:
            new_h = node.h_edges
            new_v = node.v_edges | (1 << (r * (N + 1) + c))
            adj_boxes = self._masks['v_edge_adj_boxes'][(r, c)]
        
        # Check ALL boxes (not just adjacent) for 3-edge count after action
        # But first, only adjacent boxes could have their edge count change
        for box_r, box_c in adj_boxes:
            box_bit = 1 << (box_r * N + box_c)
            # Skip already-claimed boxes
            if (node.boxes_p1 & box_bit) or (node.boxes_p2 & box_bit):
                continue
            
            # Count edges around this box after placing
            edge_info = self._box_edge_bits[(box_r, box_c)]
            count = 0
            for hb in edge_info['h_bits']:
                if new_h & (1 << hb):
                    count += 1
            for vb in edge_info['v_bits']:
                if new_v & (1 << vb):
                    count += 1
            
            if count == 3:
                return True
        
        return False
    
    def _node_to_tensor(self, node):
        """Convert node's raw ints to feature tensor (no env object needed)."""
        N = self.n
        n_h = (N + 1) * N
        n_v = N * (N + 1)
        player = node.current_player
        
        h_features = np.zeros(n_h, dtype=np.float32)
        for i in range(n_h):
            if node.h_edges & (1 << i):
                h_features[i] = 1.0
        
        v_features = np.zeros(n_v, dtype=np.float32)
        for i in range(n_v):
            if node.v_edges & (1 << i):
                v_features[i] = 1.0
        
        box_features = np.zeros(N * N, dtype=np.float32)
        for i in range(N * N):
            bit = 1 << i
            if node.boxes_p1 & bit:
                box_features[i] = 1.0 if player == 1 else -1.0
            elif node.boxes_p2 & bit:
                box_features[i] = 1.0 if player == 2 else -1.0
        
        player_flag = np.array([1.0 if player == 1 else -1.0], dtype=np.float32)
        features = np.concatenate([h_features, v_features, box_features, player_flag])
        return torch.from_numpy(features)
    
    def _get_legal_actions_from_node(self, node):
        """Get legal actions directly from node bitmasks."""
        actions = []
        N = self.n
        masks = self._masks
        
        free_h = masks['all_h'] & ~node.h_edges
        while free_h:
            bit = free_h & (-free_h)
            idx = bit.bit_length() - 1
            r, c = divmod(idx, N)
            actions.append((0, r, c))
            free_h ^= bit
        
        free_v = masks['all_v'] & ~node.v_edges
        while free_v:
            bit = free_v & (-free_v)
            idx = bit.bit_length() - 1
            r, c = divmod(idx, N + 1)
            actions.append((1, r, c))
            free_v ^= bit
        
        return actions
    
    def _apply_action_to_node(self, node, action):
        """
        Apply action and return a lightweight state object.
        Returns a _NodeState with the right attributes.
        """
        edge_type, r, c = action
        N = self.n
        
        new_h = node.h_edges
        new_v = node.v_edges
        new_p1 = node.boxes_p1
        new_p2 = node.boxes_p2
        new_s1 = node.score_p1
        new_s2 = node.score_p2
        player = node.current_player
        
        if edge_type == 0:
            new_h |= (1 << (r * N + c))
            adj_boxes = self._masks['h_edge_adj_boxes'][(r, c)]
        else:
            new_v |= (1 << (r * (N + 1) + c))
            adj_boxes = self._masks['v_edge_adj_boxes'][(r, c)]
        
        box_made = False
        box_h_masks = self._masks['box_h_masks']
        box_v_masks = self._masks['box_v_masks']
        
        for box_r, box_c in adj_boxes:
            h_mask = box_h_masks[(box_r, box_c)]
            v_mask = box_v_masks[(box_r, box_c)]
            
            if (new_h & h_mask) == h_mask and (new_v & v_mask) == v_mask:
                box_bit = 1 << (box_r * N + box_c)
                if not (new_p1 & box_bit) and not (new_p2 & box_bit):
                    if player == 1:
                        new_p1 |= box_bit
                        new_s1 += 1
                    else:
                        new_p2 |= box_bit
                        new_s2 += 1
                    box_made = True
        
        new_player = player if box_made else (3 - player)
        new_done = (new_s1 + new_s2) == N * N
        
        return _NodeState(new_h, new_v, new_p1, new_p2, new_player, 
                         new_done, [new_s1, new_s2])
    
    def _backpropagate(self, node, value):
        while node is not None:
            node.visits += 1
            node.value_sum += value
            
            if node.parent:
                if node.parent.current_player != node.current_player:
                    value = -value
            
            node = node.parent
    
    def _get_best_action(self, root):
        max_visits = max(child.visits for child in root.children.values())
        candidates = [a for a, c in root.children.items() if c.visits == max_visits]
        return random.choice(candidates)
    
    def _clone_state(self, env):
        """Clone for compatibility."""
        return env.clone()
    
    def _action_to_index(self, action):
        edge_type, r, c = action
        if edge_type == 0:
            return r * self.n + c
        else:
            return (self.n + 1) * self.n + r * (self.n + 1) + c


class _NodeState:
    """Lightweight state container for MCTS node creation.
    Avoids creating full BitBoardEnv objects."""
    __slots__ = ['h_edges', 'v_edges', 'boxes_p1', 'boxes_p2', 
                 'current_player', 'done', 'score']
    
    def __init__(self, h_edges, v_edges, boxes_p1, boxes_p2, 
                 current_player, done, score):
        self.h_edges = h_edges
        self.v_edges = v_edges
        self.boxes_p1 = boxes_p1
        self.boxes_p2 = boxes_p2
        self.current_player = current_player
        self.done = done
        self.score = score
