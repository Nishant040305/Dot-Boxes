"""
Unified simulation interface for Dots and Boxes.

Works with both BaseBoardEnv and BitBoardEnv, and any combination of agents.
No more one-class-per-matchup — just specify agent names/configs.

Usage:
    from gameSimulation.arena import Arena

    # Quick match
    arena = Arena(board_size=3, env_type='bit')
    arena.play('alphazero_bit', 'greedy', n_games=10)

    # CLI
    python -m gameSimulation.arena --p1 alphazero_bit --p2 greedy --games 10 --size 3
"""

import os
import sys
import time
import json
import random
import copy
from collections import defaultdict

# Ensure src is in path
_script_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.dirname(_script_dir)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

import torch


# ─── Environment Factory ──────────────────────────────────────────

def make_env(board_size, env_type='bit'):
    """Create an environment.
    
    Args:
        board_size: N for NxN board
        env_type: 'bit' for BitBoardEnv, 'classic' for BaseBoardEnv
    """
    if env_type == 'bit':
        from env.BitBoardEnv import BitBoardEnv
        return BitBoardEnv(board_size)
    else:
        from env.BoardEnv import BaseBoardEnv
        return BaseBoardEnv(board_size)


# ─── Agent Factory ────────────────────────────────────────────────

def _find_model_path(filename):
    """Find model file in standard locations."""
    candidates = [
        os.path.join(_src_dir, '..', 'models', filename),
        os.path.join(_src_dir, 'models', filename),
        os.path.join(_src_dir, 'cpp', 'models', filename),
    ]
    for path in candidates:
        path = os.path.abspath(path)
        if os.path.exists(path):
            return path
    return os.path.abspath(candidates[0])  # return default even if doesn't exist


def make_agent(agent_type, env, **kwargs):
    """Create an agent by name.
    
    Args:
        agent_type: One of:
            'random'       - RandomAgent
            'greedy'       - GreedyAgent
            'minmax'       - MinMaxAgent (kwargs: depth, parallel)
            'alphabeta'    - AlphaBetaAgent (kwargs: depth, parallel)
            'alphazero'    - AlphaZeroAgent with conv net (kwargs: model_path, n_simulations)
            'alphazero_bit'- AlphaZeroBitAgent with FC net (kwargs: model_path, n_simulations)
            'mcts'         - MCTSAgent (kwargs: n_simulations)
        env: The environment instance
        **kwargs: Agent-specific parameters
        
    Returns:
        Agent instance
    """
    agent_type = agent_type.lower().strip()
    rows = getattr(env, 'rows', env.N)
    cols = getattr(env, 'cols', env.N)
    
    if agent_type == 'random':
        from agents.RandomAgent import RandomAgent
        return RandomAgent(env)
    
    elif agent_type == 'greedy':
        from agents.GreedyAgent import GreedyAgent
        return GreedyAgent(env)
    
    elif agent_type == 'minmax':
        from agents.MinMaxAgent import MinMaxAgent
        depth = kwargs.get('depth', 3)
        parallel = kwargs.get('parallel', True)
        return MinMaxAgent(env, depth=depth, parallel=parallel)
    
    elif agent_type == 'alphabeta':
        from agents.AlphaBetaAgent import AlphaBetaAgent
        depth = kwargs.get('depth', 3)
        parallel = kwargs.get('parallel', True)
        return AlphaBetaAgent(env, depth=depth, parallel=parallel)
    
    elif agent_type == 'alphazero':
        from agents.AlphaZeroAgent import AlphaZeroAgent
        n_sims = kwargs.get('n_simulations', 400)
        model_path = kwargs.get('model_path', _find_model_path(f"alphazero_{rows}x{cols}.pth"))
        device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        return AlphaZeroAgent(env, model_path=model_path, n_simulations=n_sims, device=device)
    
    elif agent_type == 'alphazero_bit':
        from agents.AlphaZeroBitAgent import AlphaZeroBitAgent
        n_sims = kwargs.get('n_simulations', 400)
        model_path = kwargs.get('model_path', _find_model_path(f"alphazero_bit_{rows}x{cols}_checkpoint.pth"))
        device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = kwargs.get('checkpoint', True)
        add_noise = kwargs.get('add_noise', False)
        return AlphaZeroBitAgent(env, model_path=model_path, n_simulations=n_sims, device=device, checkpoint=checkpoint, add_noise=add_noise)
    
    elif agent_type == 'alphazero_cpp':
        from agents.AlphaZeroCppAgent import AlphaZeroCppAgent
        n_sims = kwargs.get('n_simulations', 400)
        model_path = kwargs.get('model_path', _find_model_path(f"alphazero_{rows}x{cols}.pt"))
        hidden = kwargs.get('hidden_size', 256)
        blocks = kwargs.get('num_res_blocks', 6)
        return AlphaZeroCppAgent(env, model_path=model_path, n_simulations=n_sims,
                                 hidden_size=hidden, num_res_blocks=blocks)
    
    elif agent_type == 'mcts':
        from agents.MCTSAgent import MCTSAgent
        iters = kwargs.get('iterations', kwargs.get('n_simulations', 1000))
        return MCTSAgent(env, iterations=iters)
    
    elif agent_type == 'human':
        from agents.HumanAgent import HumanAgent
        return HumanAgent(env)
    
    else:
        raise ValueError(
            f"Unknown agent type: '{agent_type}'. "
            f"Available: random, greedy, minmax, alphabeta, alphazero, "
            f"alphazero_bit, alphazero_cpp, mcts, human"
        )


# ─── Agent config parser ──────────────────────────────────────────

def parse_agent_spec(spec_str):
    """Parse an agent specification string.
    
    Format: 'agent_type' or 'agent_type:key=val,key=val'
    
    Examples:
        'greedy'
        'alphabeta:depth=4'
        'alphazero_bit:n_simulations=800'
        'minmax:depth=3,parallel=false'
    
    Returns:
        (agent_type, kwargs_dict)
    """
    if ':' not in spec_str:
        return spec_str.strip(), {}
    
    agent_type, params_str = spec_str.split(':', 1)
    kwargs = {}
    
    for param in params_str.split(','):
        param = param.strip()
        if '=' not in param:
            continue
        key, val = param.split('=', 1)
        key = key.strip()
        val = val.strip()
        
        # Auto-cast types
        if val.lower() in ('true', 'false'):
            kwargs[key] = val.lower() == 'true'
        elif val.replace('.', '', 1).replace('-', '', 1).isdigit():
            kwargs[key] = float(val) if '.' in val else int(val)
        else:
            kwargs[key] = val
    
    return agent_type.strip(), kwargs


# ─── Arena ─────────────────────────────────────────────────────────

class Arena:
    """
    Unified simulation arena for Dots and Boxes.
    
    Supports any combination of agents and either environment type.
    """
    
    def __init__(self, board_size=3, env_type='bit'):
        """
        Args:
            board_size: N for NxN board
            env_type: 'bit' or 'classic'
        """
        self.board_size = board_size
        self.env_type = env_type
        self.env = make_env(board_size, env_type)
    
    def play(self, p1_spec, p2_spec, n_games=1, verbose=True, display=False,
             swap_sides=False):
        """
        Play games between two agents.
        
        Args:
            p1_spec: Agent specification string (e.g. 'greedy', 'alphabeta:depth=4')
            p2_spec: Agent specification string  
            n_games: Number of games to play
            verbose: Print per-game and summary results
            display: Render board after each move
            swap_sides: If True, play 2*n_games total (each side once)
            
        Returns:
            dict with keys: 'p1_wins', 'p2_wins', 'draws', 'scores', 'times',
                            'p1_name', 'p2_name', 'total_games'
        """
        p1_type, p1_kwargs = parse_agent_spec(p1_spec)
        p2_type, p2_kwargs = parse_agent_spec(p2_spec)
        
        results = {
            'p1_name': p1_spec,
            'p2_name': p2_spec,
            'p1_wins': 0,
            'p2_wins': 0,
            'draws': 0,
            'scores': [],
            'times': [],
            'total_games': 0,
        }
        
        configs = [(p1_type, p1_kwargs, p2_type, p2_kwargs, False)]
        total = n_games
        
        if swap_sides:
            configs.append((p2_type, p2_kwargs, p1_type, p1_kwargs, True))
            total = n_games * 2
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"  Arena: {p1_spec} vs {p2_spec}")
            print(f"  Board: {self.board_size}x{self.board_size} | "
                  f"Env: {self.env_type} | Games: {total}")
            print(f"{'='*60}\n")
        
        game_num = 0
        for a1_type, a1_kw, a2_type, a2_kw, swapped in configs:
            for _ in range(n_games):
                game_num += 1
                score, duration = self._play_single(
                    a1_type, a1_kw, a2_type, a2_kw, display
                )
                
                # Map back to original p1/p2
                if swapped:
                    s1, s2 = score[1], score[0]
                else:
                    s1, s2 = score[0], score[1]
                
                results['scores'].append((s1, s2))
                results['times'].append(duration)
                results['total_games'] += 1
                
                if s1 > s2:
                    results['p1_wins'] += 1
                    outcome = f"{p1_spec} wins"
                elif s2 > s1:
                    results['p2_wins'] += 1
                    outcome = f"{p2_spec} wins"
                else:
                    results['draws'] += 1
                    outcome = "Draw"
                
                if verbose:
                    side_note = " (swapped)" if swapped else ""
                    print(f"  Game {game_num:3d}/{total}: "
                          f"{s1}-{s2} | {outcome}{side_note} "
                          f"({duration:.2f}s)")
        
        if verbose:
            self._print_summary(results)
        
        return results
    
    def _play_single(self, a1_type, a1_kwargs, a2_type, a2_kwargs, display=False):
        """Play a single game. Returns (score, duration)."""
        self.env.reset()
        
        agent1 = make_agent(a1_type, self.env, **a1_kwargs)
        agent2 = make_agent(a2_type, self.env, **a2_kwargs)
        
        # Point agents to the shared env
        if hasattr(agent1, 'env'):
            agent1.env = self.env
        if hasattr(agent2, 'env'):
            agent2.env = self.env
        
        start = time.time()
        
        while not self.env.done:
            if self.env.current_player == 1:
                action = agent1.act()
            else:
                action = agent2.act()
            
            self.env.step(action)
            
            if display:
                self.env.render()
        
        duration = time.time() - start
        return list(self.env.score), duration
    
    def record_games(self, p1_spec, p2_spec, n_games=1):
        """
        Play and record games with full state history.
        Returns list of game data dicts suitable for JSON serialization.
        """
        p1_type, p1_kwargs = parse_agent_spec(p1_spec)
        p2_type, p2_kwargs = parse_agent_spec(p2_spec)
        
        games = []
        for i in range(n_games):
            self.env.reset()
            
            agent1 = make_agent(p1_type, self.env, **p1_kwargs)
            agent2 = make_agent(p2_type, self.env, **p2_kwargs)
            
            if hasattr(agent1, 'env'):
                agent1.env = self.env
            if hasattr(agent2, 'env'):
                agent2.env = self.env
            
            start = time.time()
            while not self.env.done:
                if self.env.current_player == 1:
                    action = agent1.act()
                else:
                    action = agent2.act()
                self.env.step(action)
            
            duration = time.time() - start
            
            game_data = {
                'game_id': i + 1,
                'board_size': self.board_size,
                'p1': p1_spec,
                'p2': p2_spec,
                'final_score': list(self.env.score),
                'total_moves': len(self.env.action_history),
                'duration': round(duration, 3),
                'action_history': self.env.action_history,
                'state_history': self._serialize_state_history(),
            }
            games.append(game_data)
            
            winner = "P1" if self.env.score[0] > self.env.score[1] else \
                     "P2" if self.env.score[1] > self.env.score[0] else "Draw"
            print(f"  Game {i+1}: P1={self.env.score[0]} P2={self.env.score[1]} "
                  f"({winner}, {game_data['total_moves']} moves, {duration:.2f}s)")
        
        return games
    
    def _serialize_state_history(self):
        """Convert state history to JSON-serializable format."""
        history = self.env.state_history
        
        # BitBoardEnv stores ints in snapshots, BaseBoardEnv stores arrays
        # Both are already JSON-serializable as-is
        if not history:
            return []
        
        # For BitBoardEnv, also include reconstructed arrays for the visualizer
        from env.BitBoardEnv import BitBoardEnv
        if isinstance(self.env, BitBoardEnv):
            serialized = []
            for snap in history:
                entry = dict(snap)  # copy
                # Add array forms for compatibility with existing visualizer
                entry['horizontal_edges'] = self._bits_to_2d(
                    snap['h_edges'], self.board_size + 1, self.board_size)
                entry['vertical_edges'] = self._bits_to_2d(
                    snap['v_edges'], self.board_size, self.board_size + 1)
                entry['boxes'] = self._boxes_to_2d(
                    snap.get('boxes_p1', 0), snap.get('boxes_p2', 0))
                serialized.append(entry)
            return serialized
        else:
            return history
    
    def _bits_to_2d(self, bitmask, rows, cols):
        """Convert bitmask to 2D list."""
        result = []
        for r in range(rows):
            row = []
            for c in range(cols):
                row.append(bool(bitmask & (1 << (r * cols + c))))
            result.append(row)
        return result
    
    def _boxes_to_2d(self, p1_mask, p2_mask):
        """Convert box ownership bitmasks to 2D list."""
        N = self.board_size
        result = []
        for r in range(N):
            row = []
            for c in range(N):
                bit = 1 << (r * N + c)
                if p1_mask & bit:
                    row.append(1)
                elif p2_mask & bit:
                    row.append(2)
                else:
                    row.append(0)
            result.append(row)
        return result
    
    @staticmethod  
    def _print_summary(results):
        """Print a formatted summary of results."""
        total = results['total_games']
        p1w = results['p1_wins']
        p2w = results['p2_wins']
        draws = results['draws']
        
        avg_time = sum(results['times']) / total if total else 0
        
        p1_scores = [s[0] for s in results['scores']]
        p2_scores = [s[1] for s in results['scores']]
        
        print(f"\n{'─'*60}")
        print(f"  Results ({total} games)")
        print(f"{'─'*60}")
        print(f"  {results['p1_name']:>25s}: "
              f"{p1w:3d} wins ({100*p1w/total:5.1f}%)  "
              f"Avg Score: {sum(p1_scores)/total:.1f}")
        print(f"  {results['p2_name']:>25s}: "
              f"{p2w:3d} wins ({100*p2w/total:5.1f}%)  "
              f"Avg Score: {sum(p2_scores)/total:.1f}")
        print(f"  {'Draws':>25s}: {draws:3d}      ({100*draws/total:5.1f}%)")
        print(f"  {'Avg game time':>25s}: {avg_time:.2f}s")
        print(f"{'─'*60}\n")

    def play_interactive(self, opponent_spec, human_player=1):
        """Play a single interactive game: human vs agent.

        Args:
            opponent_spec: Agent spec string for the AI opponent
            human_player:  1 if human goes first, 2 if second
        """
        self.env.reset()

        from agents.HumanAgent import HumanAgent
        human = HumanAgent(self.env)

        opp_type, opp_kwargs = parse_agent_spec(opponent_spec)
        opponent = make_agent(opp_type, self.env, **opp_kwargs)

        if human_player == 1:
            agents = {1: human, 2: opponent}
            names  = {1: 'You', 2: opponent_spec}
        else:
            agents = {1: opponent, 2: human}
            names  = {1: opponent_spec, 2: 'You'}

        print(f"\n{'═' * 50}")
        print(f"  You (Player {human_player}) vs {opponent_spec}")
        print(f"  Board: {self.board_size}x{self.board_size}")
        print(f"{'═' * 50}")

        try:
            while not self.env.done:
                cp = self.env.current_player
                action = agents[cp].act()

                if cp != human_player:
                    # Show what the AI played
                    rc = human.action_to_rc(action)
                    print(f"  ◆ {names[cp]} plays ({rc[0]},{rc[1]})")

                self.env.step(action)

        except KeyboardInterrupt:
            print("\n  Game aborted.")
            return None

        # Final board
        human._render_board()
        s1, s2 = self.env.score
        print(f"\n  Final Score: P1={s1}  P2={s2}")
        your_score = s1 if human_player == 1 else s2
        opp_score  = s2 if human_player == 1 else s1
        if your_score > opp_score:
            print("  🎉 You win!")
        elif opp_score > your_score:
            print(f"  {opponent_spec} wins. Better luck next time!")
        else:
            print("  It's a draw!")

        return {'your_score': your_score, 'opp_score': opp_score}


# ─── Round-Robin Tournament ───────────────────────────────────────

def tournament(agent_specs, board_size=3, env_type='bit', 
               games_per_match=10, verbose=True):
    """
    Run a round-robin tournament between multiple agents.
    
    Args:
        agent_specs: List of agent spec strings
        board_size, env_type: Arena configuration
        games_per_match: Games per matchup (played from both sides)
        verbose: Print detailed results
        
    Returns:
        dict with standings and detailed match results
    """
    arena = Arena(board_size, env_type)
    n = len(agent_specs)
    
    # Scores: wins are 3 points, draws are 1 point
    standings = {spec: {'points': 0, 'wins': 0, 'losses': 0, 'draws': 0,
                         'score_for': 0, 'score_against': 0} 
                 for spec in agent_specs}
    match_results = []
    
    if verbose:
        print(f"\n{'#'*60}")
        print(f"  TOURNAMENT: {n} agents, {board_size}x{board_size} board")
        print(f"  {games_per_match} games per matchup (×2 with side swap)")
        print(f"{'#'*60}")
    
    for i in range(n):
        for j in range(i + 1, n):
            p1 = agent_specs[i]
            p2 = agent_specs[j]
            
            result = arena.play(p1, p2, n_games=games_per_match, 
                              verbose=verbose, swap_sides=True)
            match_results.append(result)
            
            # Update standings
            standings[p1]['wins'] += result['p1_wins']
            standings[p1]['losses'] += result['p2_wins']
            standings[p1]['draws'] += result['draws']
            standings[p2]['wins'] += result['p2_wins']
            standings[p2]['losses'] += result['p1_wins']
            standings[p2]['draws'] += result['draws']
            
            standings[p1]['points'] += result['p1_wins'] * 3 + result['draws']
            standings[p2]['points'] += result['p2_wins'] * 3 + result['draws']
            
            for s1, s2 in result['scores']:
                standings[p1]['score_for'] += s1
                standings[p1]['score_against'] += s2
                standings[p2]['score_for'] += s2
                standings[p2]['score_against'] += s1
    
    # Sort by points
    ranking = sorted(standings.items(), key=lambda x: (-x[1]['points'], 
                     -(x[1]['score_for'] - x[1]['score_against'])))
    
    if verbose:
        print(f"\n{'#'*60}")
        print(f"  FINAL STANDINGS")
        print(f"{'#'*60}")
        print(f"  {'Rank':<5} {'Agent':<28} {'Pts':>4} {'W':>4} {'L':>4} {'D':>4} {'GF':>5} {'GA':>5} {'GD':>5}")
        print(f"  {'─'*85}")
        for rank, (agent, stats) in enumerate(ranking, 1):
            gd = stats['score_for'] - stats['score_against']
            gd_str = f"+{gd}" if gd > 0 else str(gd)
            print(f"  {rank:<5} {agent:<28} {stats['points']:>4} "
                  f"{stats['wins']:>4} {stats['losses']:>4} {stats['draws']:>4} "
                  f"{stats['score_for']:>5} {stats['score_against']:>5} {gd_str:>5}")
        print()
    
    return {
        'standings': ranking,
        'matches': match_results,
    }


# ─── CLI Interface ────────────────────────────────────────────────

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Dots and Boxes Arena — pit any agent against any other',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Agent types: random, greedy, minmax, alphabeta, alphazero, alphazero_bit, mcts, human

Agent spec format: 'type' or 'type:param=value,param=value'
  Examples:
    random
    greedy
    alphabeta:depth=4
    alphazero_bit:n_simulations=800
    minmax:depth=3,parallel=false

Interactive (human) mode:
    python arena.py --play alphabeta:depth=3
    python arena.py --play greedy --player 2     # go second

Tournament mode (--tournament):
    python arena.py --tournament random greedy alphabeta:depth=3 --games 20
        """
    )
    
    parser.add_argument('--p1', type=str, help='Player 1 agent spec')
    parser.add_argument('--p2', type=str, help='Player 2 agent spec')
    parser.add_argument('--games', '-n', type=int, default=10, 
                       help='Number of games per matchup (default: 10)')
    parser.add_argument('--size', '-s', type=int, default=3,
                       help='Board size N (default: 3)')
    parser.add_argument('--env', type=str, default='classic', choices=['bit', 'classic'],
                       help="Environment type (default: classic)")
    parser.add_argument('--display', '-d', action='store_true',
                       help='Display board after each move')
    parser.add_argument('--swap', action='store_true',
                       help='Play from both sides')
    parser.add_argument('--record', type=str, default=None,
                       help='Record games to JSON file')
    parser.add_argument('--tournament', nargs='+', type=str, default=None,
                       help='Run round-robin tournament with these agents')
    parser.add_argument('--play', type=str, default=None,
                       help='Play interactively against this agent')
    parser.add_argument('--player', type=int, default=1, choices=[1, 2],
                       help='Which player you are (1=first, 2=second, default: 1)')
    
    args = parser.parse_args()
    
    if args.play:
        arena = Arena(board_size=args.size, env_type=args.env)
        arena.play_interactive(args.play, human_player=args.player)
    elif args.tournament:
        tournament(args.tournament, board_size=args.size, 
                  env_type=args.env, games_per_match=args.games)
    elif args.p1 and args.p2:
        arena = Arena(board_size=args.size, env_type=args.env)
        
        if args.record:
            print(f"\nRecording {args.games} game(s)...\n")
            games = arena.record_games(args.p1, args.p2, args.games)
            
            output = {'games': games, 'board_size': args.size,
                      'p1': args.p1, 'p2': args.p2}
            with open(args.record, 'w') as f:
                json.dump(output, f, indent=2)
            print(f"\nSaved to: {args.record}")
        else:
            arena.play(args.p1, args.p2, n_games=args.games, 
                      display=args.display, swap_sides=args.swap)
    else:
        parser.print_help()
        print("\n  Quick examples:")
        print("    python arena.py --play greedy                           # play vs greedy")
        print("    python arena.py --play alphabeta:depth=3 --player 2    # go second")
        print("    python arena.py --p1 greedy --p2 random --games 20")
        print("    python arena.py --p1 alphabeta:depth=3 --p2 greedy --swap")
        print("    python arena.py --tournament random greedy alphabeta:depth=3")


if __name__ == '__main__':
    main()
