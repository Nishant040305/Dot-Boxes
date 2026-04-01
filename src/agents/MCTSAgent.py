from agents.agent import Agent
from agents.MinMaxAgent import State
from rewards.reward import SimpleReward
import copy
import math
import random


class MCTSNode:
    """
    A node in the Monte Carlo Search Tree.

    Each node corresponds to a game state reached by taking `action`
    from the parent's state.  `wins` is always stored from the **root
    player's** perspective so that backpropagation is uniform.
    """

    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action                 # action taken from parent → this node

        self.children = []
        self.untried_actions = None          # lazily initialised
        self.visits = 0
        self.wins = 0.0                      # always from root player's perspective

    def is_fully_expanded(self):
        return self.untried_actions is not None and len(self.untried_actions) == 0

    def is_terminal(self):
        return self.state.done

    def ucb1(self, exploration_weight, root_player):
        """
        Upper Confidence Bound for Trees (UCT).

        `root_player` is needed because wins are stored from the root
        player's viewpoint.  When the *parent* of this node is the
        opponent, the parent wants to pick the child that is *worst*
        for root_player, so we flip the exploitation term.
        """
        if self.visits == 0:
            return float('inf')

        win_rate = self.wins / self.visits
        # If the parent's current player is NOT root_player,
        # the parent wants to *minimise* root_player's win rate.
        if self.parent and self.parent.state.current_player != root_player:
            win_rate = 1.0 - win_rate

        exploration = exploration_weight * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )
        return win_rate + exploration


class MCTSAgent(Agent):
    """
    Monte Carlo Tree Search agent for Dots and Boxes.

    The algorithm repeats four phases for `iterations` rounds:
        1. Selection    — walk down the tree via UCB1
        2. Expansion    — add one untried child
        3. Simulation   — random rollout to terminal state
        4. Backpropagation — update statistics back to root

    After all iterations, the child of the root with the most visits
    is chosen (robust child selection).
    """

    def __init__(self, env, iterations=1000, exploration_weight=1.41,
                 policy_table=None, guided_epsilon=0.1):
        super().__init__(env)
        self.reward_fn = SimpleReward(env)
        self.iterations = iterations
        self.exploration_weight = exploration_weight
        self.policy_table = policy_table     # {state_key: {action: visit_count}}
        self.guided_epsilon = guided_epsilon  # chance of random move during guided rollout
        self._last_root = None               # exposed for trainer stat extraction

    # ──────────────────────────────────────────────
    #  Public API
    # ──────────────────────────────────────────────

    def act(self):
        """Choose the best action from the current env state using MCTS."""
        state = self._get_state(self.env)
        self._root_player = self.env.current_player
        root = MCTSNode(state)
        root.untried_actions = self._get_legal_actions(state)

        for _ in range(self.iterations):
            # 1. Selection
            node = self._select(root)

            # 2. Expansion
            if not node.is_terminal():
                node = self._expand(node)

            # 3. Simulation (rollout)
            result = self._simulate(node.state, self._root_player)

            # 4. Backpropagation
            self._backpropagate(node, result)

        # Store root for trainer to extract stats
        self._last_root = root

        # Pick the child with the most visits (robust child)
        best_child = max(root.children, key=lambda c: c.visits)
        return best_child.action

    def get_last_move_stats(self):
        """
        Return {action: visit_count} from the last MCTS search root.
        Used by MCTSTrainer to build the policy table.
        """
        if self._last_root is None:
            return {}
        return {child.action: child.visits for child in self._last_root.children}

    # ──────────────────────────────────────────────
    #  MCTS Phases
    # ──────────────────────────────────────────────

    def _select(self, node):
        """
        Walk down the tree choosing the child with the highest UCB1
        until we reach a node that is not fully expanded or is terminal.
        """
        rp = self._root_player
        while node.is_fully_expanded() and not node.is_terminal():
            node = max(node.children,
                       key=lambda c: c.ucb1(self.exploration_weight, rp))
        return node

    def _expand(self, node):
        """
        Pick one untried action, create a new child node, and return it.
        """
        # Lazy init
        if node.untried_actions is None:
            node.untried_actions = self._get_legal_actions(node.state)

        if len(node.untried_actions) == 0:
            return node  # nothing to expand (terminal reached between select & expand)

        action = random.choice(node.untried_actions)
        node.untried_actions.remove(action)

        next_state = self._get_next_state(node.state, action)
        child = MCTSNode(
            state=next_state,
            parent=node,
            action=action,
        )
        child.untried_actions = self._get_legal_actions(next_state)
        node.children.append(child)
        return child

    def _simulate(self, state, root_player):
        """
        Rollout from `state` until the game ends.

        Uses heuristic-guided rollout policy (capture-first, then
        policy table, then safe moves, then random).

        Returns:
            A value in [0, 1] representing the outcome quality
            from root_player's perspective.  Uses continuous score
            difference (not just win/loss) for richer signal.
        """
        sim_state = self._clone_state(state)

        while not sim_state.done:
            legal = self._get_legal_actions(sim_state)
            if not legal:
                break
            action = self._rollout_policy(sim_state, legal)
            sim_state = self._get_next_state(sim_state, action)

        # Continuous evaluation: score difference normalised to [0, 1]
        my_score = sim_state.score[root_player - 1]
        opp_score = sim_state.score[2 - root_player]
        total = my_score + opp_score
        if total == 0:
            return 0.5
        # Maps from [-total, +total] → [0, 1]
        return (my_score - opp_score + total) / (2 * total)

    def _rollout_policy(self, state, legal_actions):
        """
        Choose a move during rollout.

        Priority order:
          1. Capture moves (always take a box if possible — strongest heuristic)
          2. Policy table lookup (if available and state is known)
          3. Safe random (avoid giving opponent a free capture)
          4. Pure random fallback
        """
        # ── 1. Always take captures first ──
        capture_moves = []
        for a in legal_actions:
            if self.reward_fn.get_reward_state(a, state) > 0:
                capture_moves.append(a)
        if capture_moves:
            return random.choice(capture_moves)

        # ── 2. Check policy table (if available) ──
        if self.policy_table is not None and random.random() >= self.guided_epsilon:
            key = self._state_key(state)
            if key in self.policy_table:
                action_visits = self.policy_table[key]
                candidates = []
                weights = []
                for a in legal_actions:
                    if a in action_visits:
                        candidates.append(a)
                        weights.append(action_visits[a])
                if candidates:
                    return random.choices(candidates, weights=weights, k=1)[0]

        # ── 3. Prefer safe moves (avoid creating 3rd edge on a box) ──
        safe_moves = []
        for a in legal_actions:
            if self.reward_fn._count_dangers(a, state) == 0:
                safe_moves.append(a)
        if safe_moves:
            return random.choice(safe_moves)

        # ── 4. Random fallback ──
        return random.choice(legal_actions)

    @staticmethod
    def _state_key(state):
        """Create a hashable key from a State for the policy table."""
        h = tuple(tuple(row) for row in state.horizontal_edges)
        v = tuple(tuple(row) for row in state.vertical_edges)
        return (h, v, state.current_player)

    def _backpropagate(self, node, result):
        """
        Walk back to the root, updating visits and wins.

        `result` is a value in [0, 1] from root_player's perspective
        (1.0 = best, 0.0 = worst).  All nodes store wins from
        root_player's viewpoint; UCB1 handles perspective flipping.
        """
        while node is not None:
            node.visits += 1
            node.wins += result
            node = node.parent

    # ──────────────────────────────────────────────
    #  State helpers (borrowed from MinMaxAgent pattern)
    # ──────────────────────────────────────────────

    @property
    def rows(self) -> int:
        r = getattr(self.env, 'rows', None)
        if r is not None: return r
        n = getattr(self.env, 'N', 3)
        return n if n is not None else 3

    @property
    def cols(self) -> int:
        c = getattr(self.env, 'cols', None)
        if c is not None: return c
        n = getattr(self.env, 'N', 3)
        return n if n is not None else 3

    def _get_state(self, env):
        """Build a State snapshot from the live environment (deep-copied)."""
        return State(
            horizontal_edges=copy.deepcopy(env.horizontal_edges),
            vertical_edges=copy.deepcopy(env.vertical_edges),
            boxes=copy.deepcopy(env.boxes),
            current_player=env.current_player,
            score=list(env.score),
            done=env.done,
        )

    def _clone_state(self, state):
        """Deep-copy a State object for simulation rollouts."""
        return State(
            horizontal_edges=copy.deepcopy(state.horizontal_edges),
            vertical_edges=copy.deepcopy(state.vertical_edges),
            boxes=copy.deepcopy(state.boxes),
            current_player=state.current_player,
            score=list(state.score),
            done=state.done,
        )

    def _get_legal_actions(self, state):
        """Return list of all unplaced edges as (type, i, j) tuples."""
        legal_actions = []
        for i in range(self.rows + 1):
            for j in range(self.cols):
                if not state.horizontal_edges[i][j]:
                    legal_actions.append((0, i, j))
        for i in range(self.rows):
            for j in range(self.cols + 1):
                if not state.vertical_edges[i][j]:
                    legal_actions.append((1, i, j))
        return legal_actions

    def _get_next_state(self, state, action):
        """
        Return a NEW State after applying `action`.
        Handles the Dots-and-Boxes rule: completing a box → same player's turn.
        """
        new_state = self._clone_state(state)

        # Place the edge
        if action[0] == 0:
            new_state.horizontal_edges[action[1]][action[2]] = True
        else:
            new_state.vertical_edges[action[1]][action[2]] = True

        # Check if this action completed any box(es)
        reward = self.reward_fn.get_reward_state(action, new_state)

        if reward > 0:
            new_state.score[state.current_player - 1] += reward
        else:
            new_state.current_player = 3 - new_state.current_player

        # Check if game is done
        if new_state.score[0] + new_state.score[1] == self.rows * self.cols:
            new_state.done = True

        return new_state
