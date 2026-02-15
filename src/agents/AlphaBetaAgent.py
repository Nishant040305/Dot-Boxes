from agents.MinMaxAgent import MinMaxAgent, State
from rewards.reward import SimpleReward
import random
import copy
import multiprocessing
import os


# ── Transposition table flag constants ─────────────────────────────
TT_EXACT      = 0   # stored value is the exact minimax value
TT_LOWERBOUND = 1   # stored value is a lower bound (beta cutoff)
TT_UPPERBOUND = 2   # stored value is an upper bound (alpha cutoff)


def _evaluate_action_ab(args):
    """
    Top-level worker function for multiprocessing.
    Evaluates a single action by running alpha-beta on the resulting state.

    Must be a top-level function (not a method) so it can be pickled
    by multiprocessing.Pool.
    """
    action, next_state, depth, player, env = args
    # Create a temporary agent just for the recursive search
    agent = AlphaBetaAgent.__new__(AlphaBetaAgent)
    agent.env = env
    agent.reward = SimpleReward(env)
    agent.depth = depth
    agent.tt = {}                  # each worker gets its own fresh TT
    agent.tt_max_size = 500_000
    _, score = agent._alpha_beta(
        depth, player, next_state,
        alpha=-float('inf'), beta=float('inf'),
    )
    return action, score


class AlphaBetaAgent(MinMaxAgent):
    """
    MinMax agent with Alpha-Beta pruning, enhanced with:

    1. **Move Ordering** — Captures first, safe moves next, dangerous
       moves last.  Good ordering lets alpha-beta prune far more branches,
       approaching the optimal O(b^{d/2}) complexity.

    2. **Transposition Table (TT)** — Caches evaluated positions so the
       same board state reached via different move orders is never
       re-searched.  Stores relative values (adjusted for accumulated
       score) to maximise cache hits.

    - alpha: best score the maximizing player can guarantee (lower bound)
    - beta:  best score the minimizing player can guarantee (upper bound)

    When alpha >= beta, the remaining branches are pruned.
    Supports root-level parallelism via multiprocessing (parallel=True).
    """

    def __init__(self, env, depth=3, parallel=True):
        super().__init__(env, depth, parallel)
        self.tt = {}                   # hash-map: tt_key → (depth, flag, relative_value, best_action)
        self.tt_max_size = 1_000_000   # cap entries to limit RAM (~100 MB)

    # ── public interface ───────────────────────────────────────────
    def act(self):
        """Choose the best action using alpha-beta pruning."""
        state = self._get_state(self.env)
        player = self.env.current_player

        # Clear TT each move to keep memory bounded.
        # (Old entries are still valid, but stale depth values
        #  reduce their usefulness — clearing is simpler.)
        self.tt.clear()

        if self.parallel:
            return self._act_parallel(state, player)

        action, _ = self._alpha_beta(
            self.depth, player, state,
            alpha=-float('inf'), beta=float('inf'),
        )
        return action

    # ── parallel root evaluation ───────────────────────────────────
    def _act_parallel(self, state, player):
        """
        Parallel root-level evaluation: each legal action is evaluated
        on a separate CPU core using multiprocessing.Pool.

        Note: alpha-beta pruning across branches is lost at the root level
        (each worker searches independently), but the speedup from parallel
        execution more than compensates on multi-core machines.
        """
        legal_actions = self._get_legal_actions(state)
        is_maximizing = (state.current_player == player)

        # Prepare arguments for each worker
        work_items = []
        for action in legal_actions:
            next_state = self._get_next_state(state, action)
            work_items.append((
                action, next_state, self.depth - 1, player, self.env,
            ))

        num_workers = min(len(work_items), os.cpu_count() or 4)
        with multiprocessing.Pool(processes=num_workers) as pool:
            results = pool.map(_evaluate_action_ab, work_items)

        # Collect best actions (with randomness among ties)
        best_actions = []
        best_score = -float('inf') if is_maximizing else float('inf')

        for action, score in results:
            if is_maximizing:
                if score > best_score:
                    best_score = score
                    best_actions = [action]
                elif score == best_score:
                    best_actions.append(action)
            else:
                if score < best_score:
                    best_score = score
                    best_actions = [action]
                elif score == best_score:
                    best_actions.append(action)

        return random.choice(best_actions)

    # ── transposition table helpers ────────────────────────────────
    def _tt_key(self, state):
        """
        Create a hashable key for the transposition table.

        Only edges + current_player are needed.  The accumulated score
        is factored out via relative values (see _alpha_beta).
        """
        h = tuple(tuple(row) for row in state.horizontal_edges)
        v = tuple(tuple(row) for row in state.vertical_edges)
        return (h, v, state.current_player)

    # ── move ordering ──────────────────────────────────────────────
    def _order_moves(self, legal_actions, state, tt_best_action=None):
        """
        Order moves for optimal alpha-beta pruning.

        Priority (highest first):
          1. TT best move from a previous / shallower search  (1000)
          2. Capture moves  (100+)
          3. Safe moves     (0)
          4. Dangerous moves — create a 3rd edge  (-100)
        """
        prioritized = []
        for action in legal_actions:
            if action == tt_best_action:
                priority = 1000          # always try TT move first
            else:
                priority = self.reward.get_move_priority(action, state)
            prioritized.append((priority, action))

        prioritized.sort(key=lambda x: x[0], reverse=True)
        return [action for _, action in prioritized]

    # ── core search ────────────────────────────────────────────────
    def _alpha_beta(self, depth, player, state, alpha, beta):
        """
        Alpha-Beta pruning search for Dots and Boxes, with
        transposition table lookups/stores and move ordering.

        Args:
            depth:  remaining search depth
            player: the player we are maximizing for (1 or 2)
            state:  current State object
            alpha:  best already-explored score for the maximizer
            beta:   best already-explored score for the minimizer

        Returns:
            (best_action, best_score) where score is from `player`'s perspective.
        """
        legal_actions = self._get_legal_actions(state)

        # ── terminal: game over ────────────────────────────────────
        if len(legal_actions) == 0 or state.done:
            return None, state.score[player - 1] - state.score[2 - player]

        # ── transposition table lookup ─────────────────────────────
        #
        # We store *relative* values in the TT:
        #     relative_value = value - score_diff
        #
        # Two paths to the same edge configuration (same edges,
        # same current_player) with different accumulated scores
        # have identical future game trees.  The terminal values
        # differ only by a constant offset (the accumulated score
        # difference).  Storing relative values lets us reuse TT
        # entries across those paths.
        #
        tt_key = self._tt_key(state)
        score_diff = state.score[player - 1] - state.score[2 - player]
        tt_best_action = None

        if tt_key in self.tt:
            tt_depth, tt_flag, tt_relative_value, tt_action = self.tt[tt_key]
            tt_value = tt_relative_value + score_diff
            tt_best_action = tt_action     # use for move ordering regardless

            if tt_depth >= depth:
                if tt_flag == TT_EXACT:
                    return tt_action, tt_value
                elif tt_flag == TT_LOWERBOUND and tt_value >= beta:
                    return tt_action, tt_value
                elif tt_flag == TT_UPPERBOUND and tt_value <= alpha:
                    return tt_action, tt_value

        # ── leaf: depth exhausted — greedy one-step evaluation ─────
        if depth == 0:
            best_actions = self._get_best_actions(legal_actions, state)
            action = random.choice(best_actions)
            reward = self.reward.get_reward_state(action, state)
            score_val = score_diff
            if state.current_player == player:
                score_val += reward
            else:
                score_val -= reward
            return action, score_val

        # ── recursive alpha-beta with move ordering ────────────────
        orig_alpha = alpha
        orig_beta  = beta
        is_maximizing = (state.current_player == player)
        best_actions = []
        best_score = -float('inf') if is_maximizing else float('inf')

        ordered_actions = self._order_moves(legal_actions, state, tt_best_action)

        for action in ordered_actions:
            next_state = self._get_next_state(state, action)
            _, score = self._alpha_beta(depth - 1, player, next_state, alpha, beta)

            if is_maximizing:
                if score > best_score:
                    best_score = score
                    best_actions = [action]
                elif score == best_score:
                    best_actions.append(action)
                alpha = max(alpha, best_score)
            else:
                if score < best_score:
                    best_score = score
                    best_actions = [action]
                elif score == best_score:
                    best_actions.append(action)
                beta = min(beta, best_score)

            # Prune: no need to explore further
            if alpha >= beta:
                break

        # ── transposition table store ──────────────────────────────
        best_action = random.choice(best_actions)
        relative_value = best_score - score_diff

        if best_score <= orig_alpha:
            tt_flag = TT_UPPERBOUND     # failed low — value is at most this
        elif best_score >= orig_beta:
            tt_flag = TT_LOWERBOUND     # failed high — value is at least this
        else:
            tt_flag = TT_EXACT          # exact minimax value

        # Store or replace (prefer deeper entries)
        if tt_key in self.tt:
            old_depth = self.tt[tt_key][0]
            if depth >= old_depth:
                self.tt[tt_key] = (depth, tt_flag, relative_value, best_action)
        elif len(self.tt) < self.tt_max_size:
            self.tt[tt_key] = (depth, tt_flag, relative_value, best_action)

        return best_action, best_score
