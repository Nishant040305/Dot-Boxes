import numpy as np
import torch
import copy
import math
import random
from agents.agent import Agent
from models.Net import AlphaZeroNet

class AlphaZeroNode:
    def __init__(self, state, parent=None, prior=0.0):
        self.state = state
        self.parent = parent
        self.children = {}  # action -> Node
        self.visits = 0
        self.value_sum = 0.0
        self.prior = prior  # P(s, a) from neural net
        
    def is_expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits

class AlphaZeroAgent(Agent):
    def __init__(self, env, model_path=None, n_simulations=100, c_puct=1.0, device='cpu'):
        super().__init__(env)
        self.device = device
        self.n = env.N
        self.n_simulations = n_simulations
        self.c_puct = c_puct
        self.chain_prior = 1.25
        # Initialize model
        self.model = AlphaZeroNet(self.n).to(self.device)
        if model_path:
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"Loaded AlphaZero model from {model_path}")
            except FileNotFoundError:
                print(f"Model {model_path} not found, using random weights.")
                
        self.model.eval()

    def act(self, return_probs=False, temperature=1e-3):
        """
        Perform MCTS simulations and return the best action.
        If return_probs is True, returns (action, policy_vector).
        """
        # Create root
        state = self._clone_state(self.env)
        root = AlphaZeroNode(state, prior=1.0)
        
        # Expand root immediately to get legal moves and priors
        self._evaluate_and_expand(root)
        
        # Run simulations
        for _ in range(self.n_simulations):
            node = root
            
            # 1. Selection
            while node.is_expanded():
                action, node = self._select_child(node)
                
            # 2. Expansion and Evaluation
            value = self._evaluate_and_expand(node)
            
            # 3. Backpropagation
            self._backpropagate(node, value)
            
        # Select best move based on visit counts
        if not root.children:
             return None

        # Create policy based on visits (for training)
        visit_counts = {action: child.visits for action, child in root.children.items()}
        
        # Temperature Sampling
        if temperature < 1e-3:
            best_action = self._get_best_action(root)
        else:
            actions = list(visit_counts.keys())
            visits = np.array([visit_counts[a] for a in actions], dtype=np.float64)
            # Use log-space to avoid overflow: log(v^(1/t)) = log(v)/t
            log_visits = np.log(visits + 1e-10) / temperature
            # Subtract max for numerical stability (log-sum-exp trick)
            log_visits -= np.max(log_visits)
            probs = np.exp(log_visits)
            probs /= np.sum(probs)
            best_action = actions[np.random.choice(len(actions), p=probs)]

        if return_probs:
            return best_action, visit_counts
        
        return best_action

    def _select_child(self, node):
        """Select child using PUCT formula."""
        # PUCT: Q(s,a) + c_puct * P(s,a) * sqrt(sum_b N(s,b)) / (1 + N(s,a))
        best_score = -float('inf')
        best_action = None
        best_child = None
        
        sum_visits = sum(child.visits for child in node.children.values())
        sqrt_sum_visits = math.sqrt(sum_visits)
        
        for action, child in node.children.items():
            q_value = child.value()
            
            # Perspective flip: If it's opponent's turn in child node, 
            # the value returned is from their perspective.
            # But Q(s,a) should be from current player's perspective.
            # Wait, usually value head returns value for the *current player* of the state.
            # So if child.state.current_player is different from node.state.current_player,
            # we simply take -q_value.
            # BUT: In this implementation, value is always relative to the player who just moved
            # or the player whose turn it is?
            # Standard AlphaZero: Value head returns expected outcome for state.current_player.
            # So parent should MINIMIZE child value? Or use -child.value().
            # Let's clarify:
            # v = expected return for state.current_player.
            # If I make move 'a' to reach state', and state'.current_player is opponent,
            # then v' is good for opponent. So for me, it's -v'.
            
            # However, Dots & Boxes has "move again" mechanic.
            # If move 'a' keeps the turn (completed box), child.state.current_player == node.state.current_player.
            # Then v' is good for me. So +v'.
            # If move 'a' passes turn, child.state.current_player != node.state.current_player.
            # Then v' is good for opponent. So -v'.
            
            if child.state.current_player == node.state.current_player:
                action_value = q_value
            else:
                action_value = -q_value
                
            u_score = self.c_puct * child.prior * sqrt_sum_visits / (1 + child.visits)
            score = action_value + u_score
            
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
                
        return best_action, best_child
        
    def _evaluate_and_expand(self, node):
        """
        Evaluate leaf node with NN and expand children.
        Returns value v for the current player of node.state.
        """
        # Check terminal
        if node.state.done:
            # Value is final score difference or win/loss
            score_diff = node.state.score[node.state.current_player - 1] - node.state.score[2 - node.state.current_player]
            if score_diff > 0: return 1.0
            elif score_diff < 0: return -1.0
            else: return 0.0

        # Prepare input
        # Use model's preprocess
        tensor_state = self.model.preprocess(node.state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            policy_logits, value = self.model(tensor_state)
            
        # Apply Softmax to get probabilities (since model now returns logits)
        policy_probs = torch.softmax(policy_logits, dim=1).cpu().numpy()[0]
        value = value.item()
        
        # Get legal moves
        legal_actions = self._get_legal_actions(node.state)
        
        # Filter and renormalize policy with Chain Heuristic
        # We need a mapping from action -> index and index -> action
        
        # 1. Identify moves that complete a box (start/continue a chain)
        winning_moves = set()
        for action in legal_actions:
            if self._does_complete_box(node.state, action):
                winning_moves.add(action)
        
        total_prior = 0.0
        children_priors = {} 
        
        for action in legal_actions:
            # Calculate index
            idx = self._action_to_index(action)
            prior = policy_probs[idx]
            
            # Heuristic Boost:
            # Multiplicative boost is safer than additive.
            if action in winning_moves:
                prior *= self.chain_prior # Boost by 25%
            
            children_priors[action] = prior
            total_prior += prior
            
        # 2. Create children
        for action in legal_actions:
            # Normalized prior
            prior = children_priors[action] / total_prior if total_prior > 0 else 0.0
            
            # Create child node
            next_state = self._get_next_state(node.state, action)
            child = AlphaZeroNode(next_state, parent=node, prior=prior)
            node.children[action] = child
            
        return value

    def _does_complete_box(self, state, action):
        """Check if placing an edge completes a box."""
        type_, r, c = action
        n = self.n
        
        if type_ == 0: # Horizontal (r, c)
            # Check box above (r-1, c)
            if r > 0:
                if (state.horizontal_edges[r-1][c] and 
                    state.vertical_edges[r-1][c] and state.vertical_edges[r-1][c+1]):
                    return True
            # Check box below (r, c)
            if r < n:
                if (state.horizontal_edges[r+1][c] and 
                    state.vertical_edges[r][c] and state.vertical_edges[r][c+1]):
                    return True
        else: # Vertical (r, c)
            # Check box left (r, c-1)
            if c > 0:
                if (state.vertical_edges[r][c-1] and 
                    state.horizontal_edges[r][c-1] and state.horizontal_edges[r+1][c-1]):
                    return True
            # Check box right (r, c)
            if c < n:
                if (state.vertical_edges[r][c+1] and 
                    state.horizontal_edges[r][c] and state.horizontal_edges[r+1][c]):
                    return True
        return False

    def _backpropagate(self, node, value):
        while node is not None:
            node.visits += 1
            node.value_sum += value
            
            # If we go up to parent:
            # If parent -> node was "pass turn": value flips
            # If parent -> node was "same turn": value stays
            if node.parent:
                if node.parent.state.current_player != node.state.current_player:
                    value = -value
                else:
                    value = value # Same player, same value perspective
            
            node = node.parent

    def _get_best_action(self, root):
        # Tie-break randomly
        max_visits = max(child.visits for child in root.children.values())
        candidates = [a for a, c in root.children.items() if c.visits == max_visits]
        return random.choice(candidates)
        
    # --- Helpers ---
    
    def _action_to_index(self, action):
        # action is (type, r, c)
        # type 0: horiz, r in 0..N, c in 0..N-1
        # type 1: vert, r in 0..N-1, c in 0..N
        type_, r, c = action
        if type_ == 0:
            # Horizontal edges come first
            # Index = r * N + c
            return r * self.n + c
        else:
            # Vertical edges start after H-edges
            # H-edges count = (N+1)*N
            offset = (self.n + 1) * self.n
            # Index = offset + r * (N+1) + c
            return offset + r * (self.n + 1) + c

    def _index_to_action(self, idx):
        h_limit = (self.n + 1) * self.n
        if idx < h_limit:
            # Horizontal
            r = idx // self.n
            c = idx % self.n
            return (0, r, c)
        else:
            # Vertical
            idx -= h_limit
            r = idx // (self.n + 1)
            c = idx % (self.n + 1)
            return (1, r, c)

    def _state_to_tensor(self, state):
        if hasattr(self, 'model') and self.model:
            return self.model.preprocess(state)
        return torch.tensor([])

    # Copied from MCTSAgent
    def _clone_state(self, state):
        if hasattr(state, 'clone'):
            return state.clone()
        return copy.deepcopy(state)
        
    def _get_legal_actions(self, state):
        legal_actions = []
        for i in range(self.n + 1):
            for j in range(self.n):
                if not state.horizontal_edges[i][j]:
                    legal_actions.append((0, i, j))
        for i in range(self.n):
            for j in range(self.n + 1):
                if not state.vertical_edges[i][j]:
                    legal_actions.append((1, i, j))
        return legal_actions

    def _get_next_state(self, state, action):
        # This needs access to game logic. Ideally we use the env's logic.
        # But the state object itself doesn't always have methods.
        # I'll replicate the logic from MCTSAgent._get_next_state
        new_state = self._clone_state(state)
        
        # Place edge
        if action[0] == 0:
            new_state.horizontal_edges[action[1]][action[2]] = True
        else:
            new_state.vertical_edges[action[1]][action[2]] = True
            
        # Check boxes
        reward = 0
        # If horizontal edge (r, c)
        # Check box above (r-1, c) and below (r, c)
        if action[0] == 0:
            r, c = action[1], action[2]
            # Box below: (r, c) - needs H(r,c), H(r+1,c), V(r,c), V(r,c+1)
            if r < self.n:
                if (new_state.horizontal_edges[r][c] and new_state.horizontal_edges[r+1][c] and 
                    new_state.vertical_edges[r][c] and new_state.vertical_edges[r][c+1]):
                    new_state.boxes[r][c] = state.current_player
                    reward += 1
            # Box above: (r-1, c)
            if r > 0:
                if (new_state.horizontal_edges[r-1][c] and new_state.horizontal_edges[r][c] and
                    new_state.vertical_edges[r-1][c] and new_state.vertical_edges[r-1][c+1]):
                    new_state.boxes[r-1][c] = state.current_player
                    reward += 1
        else: # Vertical edge (r, c)
            r, c = action[1], action[2]
            # Box right: (r, c) - needs H(r,c), H(r+1,c), V(r,c), V(r,c+1)
            if c < self.n:
                if (new_state.horizontal_edges[r][c] and new_state.horizontal_edges[r+1][c] and
                    new_state.vertical_edges[r][c] and new_state.vertical_edges[r][c+1]):
                    new_state.boxes[r][c] = state.current_player
                    reward += 1
            # Box left: (r, c-1)
            if c > 0:
                if (new_state.horizontal_edges[r][c-1] and new_state.horizontal_edges[r+1][c-1] and
                    new_state.vertical_edges[r][c-1] and new_state.vertical_edges[r][c]):
                    new_state.boxes[r][c-1] = state.current_player
                    reward += 1
                    
        if reward > 0:
            new_state.score[state.current_player - 1] += reward
        else:
            new_state.current_player = 3 - state.current_player
            
        if new_state.score[0] + new_state.score[1] == self.n * self.n:
            new_state.done = True
            
        return new_state

    def _expand_node(self, node):
        if node.is_expanded(): return
        # Initial expansion for root or leaf
        # We need to set up children. 
        # But wait, self._evaluate_and_expand already does this using NN priors.
        # This helper is just a wrapper if needed.
        pass
