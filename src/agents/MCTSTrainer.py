"""
MCTSTrainer — Self-play training loop for the MCTS agent.

Runs MCTS vs MCTS games, accumulates move statistics into a policy
table, and uses that table to guide future rollouts.  The trained
policy table can be saved/loaded from disk and plugged into any
MCTSAgent to boost its playing strength.

Usage:
    from agents.MCTSTrainer import MCTSTrainer

    trainer = MCTSTrainer(N=3, iterations=500)
    trainer.train(num_games=100)
    trainer.save("models/mcts_3x3.pkl")

    # Later — load and play:
    trainer2 = MCTSTrainer(N=3)
    trainer2.load("models/mcts_3x3.pkl")
    agent = trainer2.get_trained_agent(env)
    action = agent.act()
"""

import os
import pickle
import time

from env.BoardEnv import BaseBoardEnv
from agents.MCTSAgent import MCTSAgent


class MCTSTrainer:
    """
    Self-play training loop for MCTS.

    After each move in a self-play game, the MCTS root's search
    statistics (action → visit_count) are recorded, keyed by the
    board state.  On subsequent games these statistics act as a
    *guided rollout policy*, replacing pure random play with
    informed move sampling.

    The policy table is board-size-specific because state dimensions
    change with N.
    """

    def __init__(self, N, iterations=500, exploration_weight=1.41,
                 guided_epsilon=0.1):
        """
        Args:
            N:                   board size (N×N grid of boxes)
            iterations:          MCTS iterations per move during training
            exploration_weight:  UCB1 exploration constant
            guided_epsilon:      chance of random move during guided rollout
        """
        self.N = N
        self.iterations = iterations
        self.exploration_weight = exploration_weight
        self.guided_epsilon = guided_epsilon

        # {state_key: {action: cumulative_visit_count}}
        self.policy_table = {}

        # Training statistics
        self.training_stats = {
            "total_games": 0,
            "p1_wins": 0,
            "p2_wins": 0,
            "draws": 0,
            "table_size": 0,
        }

    # ──────────────────────────────────────────────
    #  Training
    # ──────────────────────────────────────────────

    def train(self, num_games=100, verbose=True):
        """
        Run `num_games` self-play games, accumulating statistics
        into the policy table.

        Args:
            num_games:  number of games to play
            verbose:    print progress every 10 games
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"  MCTS Self-Play Training  |  Board: {self.N}×{self.N}")
            print(f"  Iterations/move: {self.iterations}  |  Games: {num_games}")
            print(f"{'='*60}\n")

        start_time = time.time()

        for game_idx in range(1, num_games + 1):
            game_stats = self._play_one_game()
            self.training_stats["total_games"] += 1

            # Record outcome
            s = game_stats["score"]
            if s[0] > s[1]:
                self.training_stats["p1_wins"] += 1
            elif s[1] > s[0]:
                self.training_stats["p2_wins"] += 1
            else:
                self.training_stats["draws"] += 1

            self.training_stats["table_size"] = len(self.policy_table)

            if verbose and (game_idx % max(1, num_games // 10) == 0 or game_idx == 1):
                elapsed = time.time() - start_time
                avg_time = elapsed / game_idx
                remaining = avg_time * (num_games - game_idx)
                total = self.training_stats
                print(
                    f"  Game {game_idx:>4}/{num_games}  |  "
                    f"Score: {s[0]}-{s[1]}  |  "
                    f"Table: {total['table_size']:,} states  |  "
                    f"P1/P2/D: {total['p1_wins']}/{total['p2_wins']}/{total['draws']}  |  "
                    f"ETA: {remaining:.0f}s"
                )

        elapsed = time.time() - start_time
        if verbose:
            print(f"\n  Training complete in {elapsed:.1f}s")
            print(f"  Policy table: {len(self.policy_table):,} unique states")
            t = self.training_stats
            print(f"  Results: P1 wins {t['p1_wins']}, P2 wins {t['p2_wins']}, Draws {t['draws']}")
            print()

        return self.training_stats

    def _play_one_game(self):
        """Play a single self-play game, recording move statistics."""
        env = BaseBoardEnv(self.N)
        env.reset()

        # Both agents share the same (evolving) policy table
        agent1 = MCTSAgent(
            env,
            iterations=self.iterations,
            exploration_weight=self.exploration_weight,
            policy_table=self.policy_table,
            guided_epsilon=self.guided_epsilon,
        )
        agent2 = MCTSAgent(
            env,
            iterations=self.iterations,
            exploration_weight=self.exploration_weight,
            policy_table=self.policy_table,
            guided_epsilon=self.guided_epsilon,
        )

        game_records = []  # [(state_key, {action: visits})]

        while not env.done:
            agent = agent1 if env.current_player == 1 else agent2
            state_before = agent._get_state(env)
            state_key = MCTSAgent._state_key(state_before)

            action = agent.act()
            move_stats = agent.get_last_move_stats()

            game_records.append((state_key, move_stats))
            env.step(action)

        # Merge game records into global policy table
        self._merge_stats(game_records)

        return {"score": list(env.score), "moves": len(env.action_history)}

    def _merge_stats(self, game_records):
        """
        Merge per-move search statistics into the global policy table.

        Visit counts are accumulated (summed) across games so that
        frequently-seen states develop stronger priors.
        """
        for state_key, move_stats in game_records:
            if state_key not in self.policy_table:
                self.policy_table[state_key] = {}
            table_entry = self.policy_table[state_key]
            for action, visits in move_stats.items():
                table_entry[action] = table_entry.get(action, 0) + visits

    # ──────────────────────────────────────────────
    #  Progressive training (expanding N)
    # ──────────────────────────────────────────────

    def train_progressive(self, sizes, games_per_size=50, verbose=True):
        """
        Train on progressively larger board sizes.

        The policy table is specific to a board size, so it is reset
        between sizes.  However, training on smaller boards first
        warms up the MCTS tree-search behavior, meaning the agent
        starts each new board size with better-tuned hyperparameters
        and timing expectations.

        Args:
            sizes:          list of board sizes, e.g. [2, 3, 4]
            games_per_size: games to train per board size
            verbose:        print progress

        Returns:
            dict mapping N → {policy_table, training_stats}
        """
        results = {}

        for n in sizes:
            if verbose:
                print(f"\n{'#'*60}")
                print(f"  Progressive Training — Board Size: {n}×{n}")
                print(f"{'#'*60}")

            self.N = n
            self.policy_table = {}  # reset for new board size
            self.training_stats = {
                "total_games": 0,
                "p1_wins": 0,
                "p2_wins": 0,
                "draws": 0,
                "table_size": 0,
            }

            stats = self.train(num_games=games_per_size, verbose=verbose)
            results[n] = {
                "policy_table": dict(self.policy_table),
                "training_stats": dict(stats),
            }

        return results

    # ──────────────────────────────────────────────
    #  Save / Load
    # ──────────────────────────────────────────────

    def save(self, path):
        """
        Save the trained policy table and metadata to disk.

        Args:
            path: file path (e.g. 'models/mcts_3x3.pkl')
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

        data = {
            "N": self.N,
            "policy_table": self.policy_table,
            "training_stats": self.training_stats,
            "iterations": self.iterations,
            "exploration_weight": self.exploration_weight,
            "guided_epsilon": self.guided_epsilon,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

        print(f"  Model saved to {path}")
        print(f"  Board size: {self.N}×{self.N}")
        print(f"  Policy table: {len(self.policy_table):,} states")

    def load(self, path):
        """
        Load a previously trained policy table from disk.

        Args:
            path: file path to the saved model
        """
        with open(path, "rb") as f:
            data = pickle.load(f)

        self.N = data["N"]
        self.policy_table = data["policy_table"]
        self.training_stats = data["training_stats"]
        self.iterations = data["iterations"]
        self.exploration_weight = data["exploration_weight"]
        self.guided_epsilon = data["guided_epsilon"]

        print(f"  Model loaded from {path}")
        print(f"  Board size: {self.N}×{self.N}")
        print(f"  Policy table: {len(self.policy_table):,} states")
        print(f"  Training games: {self.training_stats['total_games']}")

    # ──────────────────────────────────────────────
    #  Agent factory
    # ──────────────────────────────────────────────

    def get_trained_agent(self, env, iterations=None):
        """
        Create an MCTSAgent preloaded with the learned policy table.

        Args:
            env:        the game environment
            iterations: MCTS iterations per move (defaults to training value)

        Returns:
            MCTSAgent with guided rollouts
        """
        return MCTSAgent(
            env,
            iterations=iterations or self.iterations,
            exploration_weight=self.exploration_weight,
            policy_table=self.policy_table,
            guided_epsilon=self.guided_epsilon,
        )
