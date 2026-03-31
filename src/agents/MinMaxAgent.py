from agents.agent import Agent
from rewards.reward import SimpleReward
import random
import copy
import multiprocessing
import os


class State:
    """Lightweight state object for minimax search."""
    def __init__(self, horizontal_edges, vertical_edges, boxes, current_player, score, done):
        self.horizontal_edges = horizontal_edges
        self.vertical_edges = vertical_edges
        self.boxes = boxes
        self.current_player = current_player
        self.score = score
        self.done = done


def _evaluate_action(args):
    """
    Top-level worker function for multiprocessing.
    Evaluates a single action by running minimax on the resulting state.

    Must be a top-level function (not a method) so it can be pickled
    by multiprocessing.Pool.
    """
    action, next_state, depth, player, env_rows, env_cols, reward_env = args
    # Create a temporary agent just for the recursive search
    agent = MinMaxAgent.__new__(MinMaxAgent)
    agent.env = reward_env
    agent.reward = SimpleReward(reward_env)
    agent.depth = depth
    agent._rows = env_rows
    agent._cols = env_cols
    _, score = agent._minmax(depth, player, next_state)
    return action, score


class MinMaxAgent(Agent):
    def __init__(self, env, depth=3, parallel=True):
        super().__init__(env)
        self.reward = SimpleReward(env)
        self.depth = depth
        self.parallel = parallel
        self._rows = env.rows
        self._cols = env.cols

    @property
    def rows(self):
        return self._rows

    @property
    def cols(self):
        return self._cols

    def act(self):
        """Choose the best action from the current environment state using minimax."""
        state = self._get_state(self.env)
        player = self.env.current_player

        if self.parallel:
            return self._act_parallel(state, player)

        action, _ = self._minmax(self.depth, player, state)
        return action

    def _act_parallel(self, state, player):
        """
        Parallel root-level evaluation: each legal action is evaluated
        on a separate CPU core using multiprocessing.Pool.
        """
        legal_actions = self._get_legal_actions(state)
        is_maximizing = (state.current_player == player)

        # Prepare arguments for each worker
        work_items = []
        for action in legal_actions:
            next_state = self._get_next_state(state, action)
            work_items.append((
                action, next_state, self.depth - 1, player,
                self._rows, self._cols, self.env,
            ))

        num_workers = min(len(work_items), os.cpu_count() or 4)
        with multiprocessing.Pool(processes=num_workers) as pool:
            results = pool.map(_evaluate_action, work_items)

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

    def _minmax(self, depth, player, state):
        """
        Minimax search for Dots and Boxes.

        Args:
            depth:  remaining search depth
            player: the player we are maximizing for (1 or 2)
            state:  current State object

        Returns:
            (best_action, best_score) where score is from `player`'s perspective.

        Key rule: if an action completes a box, the *same* player moves again
        (no turn change), so the recursive call keeps the same current_player.
        """
        legal_actions = self._get_legal_actions(state)

        # ---- terminal: game over (no moves left) ----
        if len(legal_actions) == 0 or state.done:
            return None, state.score[player - 1] - state.score[2 - player]

        # ---- leaf: depth exhausted — use greedy one-step evaluation ----
        if depth == 0:
            best_actions = self._get_best_actions(legal_actions, state)
            action = random.choice(best_actions)
            reward = self.reward.get_reward_state(action, state)
            # evaluate from maximizing player's perspective
            score_diff = (state.score[player - 1] - state.score[2 - player])
            if state.current_player == player:
                score_diff += reward
            else:
                score_diff -= reward
            return action, score_diff

        # ---- recursive minimax ----
        is_maximizing = (state.current_player == player)
        best_actions = []
        best_score = -float('inf') if is_maximizing else float('inf')

        for action in legal_actions:
            next_state = self._get_next_state(state, action)
            _, score = self._minmax(depth - 1, player, next_state)

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

        return random.choice(best_actions), best_score

    def _get_best_actions(self, actions, state):
        best_action = actions[0]
        best_actions = [best_action]
        for i in range(1, len(actions)):
            if self.reward.get_reward_state(actions[i], state) > self.reward.get_reward_state(best_action, state):
                best_action = actions[i]
                best_actions = [best_action]
            elif self.reward.get_reward_state(actions[i], state) == self.reward.get_reward_state(best_action, state):
                best_actions.append(actions[i])
        return best_actions

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

    def _get_legal_actions(self, state):
        legal_actions = []
        for i in range(self._rows + 1):
            for j in range(self._cols):
                if state.horizontal_edges[i][j] == False:
                    legal_actions.append((0, i, j))
        for i in range(self._rows):
            for j in range(self._cols + 1):
                if state.vertical_edges[i][j] == False:
                    legal_actions.append((1, i, j))
        return legal_actions

    def _get_next_state(self, state, action):
        """
        Return a NEW State after applying `action`.
        Uses deep copy so the original state is never mutated.

        Dots-and-Boxes rule: if the action completes a box,
        the current player gets another turn (no player switch).
        """
        new_state = State(
            horizontal_edges=copy.deepcopy(state.horizontal_edges),
            vertical_edges=copy.deepcopy(state.vertical_edges),
            boxes=copy.deepcopy(state.boxes),
            current_player=state.current_player,
            score=list(state.score),
            done=state.done,
        )

        # Place the edge
        if action[0] == 0:
            new_state.horizontal_edges[action[1]][action[2]] = True
        else:
            new_state.vertical_edges[action[1]][action[2]] = True

        # Check if this action completed any box(es)
        reward = self.reward.get_reward_state(action, new_state)

        if reward > 0:
            # Player scored — gets another turn, no player switch
            new_state.score[state.current_player - 1] += reward
        else:
            # No box completed — switch player
            new_state.current_player = 3 - new_state.current_player

        # Check if game is done
        if new_state.score[0] + new_state.score[1] == self._rows * self._cols:
            new_state.done = True

        return new_state