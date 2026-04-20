#!/usr/bin/env python3
"""
Evaluation script for the research paper.

Runs the trained AlphaZero C++ agent against baselines (Random, Greedy,
AlphaBeta) and reports win-rate, average score difference, and time/move.

Usage:
    # Evaluate 2x2 model against Random (200 games, 200 sims):
    python evaluate_for_paper.py --rows 2 --cols 2 --opponent random --sims 200 --games 200

    # Evaluate 2x2 model against Greedy:
    python evaluate_for_paper.py --rows 2 --cols 2 --opponent greedy --sims 200 --games 200

    # Evaluate 3x3 model against AlphaBeta depth=3:
    python evaluate_for_paper.py --rows 3 --cols 3 --opponent alphabeta --ab-depth 3 --sims 800 --games 200

    # Run all 2x2 evaluations at once:
    python evaluate_for_paper.py --rows 2 --cols 2 --all

    # Specify model path explicitly:
    python evaluate_for_paper.py --rows 2 --cols 2 --opponent random --model src/cpp/models/_2x2/alphazero_2x2.pt
"""

import sys
import os
import time
import argparse
import json
import logging

# Add src to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from env.BoardEnv import BaseBoardEnv
from agents.RandomAgent import RandomAgent
from agents.GreedyAgent import GreedyAgent
from agents.AlphaBetaAgent import AlphaBetaAgent
from agents.AlphaZeroCppAgent import AlphaZeroCppAgent


logger = logging.getLogger(__name__)

def write_results_txt(results: dict, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    az_wins = int(results.get("az_wins", 0))
    opp_wins = int(results.get("opp_wins", 0))
    draws = int(results.get("draws", 0))

    if az_wins > opp_wins:
        overall = "WIN"
    elif az_wins < opp_wins:
        overall = "LOSS"
    else:
        overall = "DRAW"

    lines = [
        f"Board: {results.get('board', '')}",
        f"Opponent: {results.get('opponent', '')}",
        f"Games: {results.get('games', '')}",
        f"Wins: {az_wins}",
        f"Losses: {opp_wins}",
        f"Draws: {draws}",
        f"Result: {overall}",
        f"Win-rate: {results.get('win_rate', 0):.1f}%",
        f"Avg score diff: {results.get('avg_score_diff', 0):+.2f}",
        f"Avg time/move: {results.get('avg_time_per_move_ms', 0):.1f} ms",
    ]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def configure_logging(log_level: str) -> None:
    level = getattr(logging, str(log_level).upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(processName)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def find_model_path(rows, cols):
    """Auto-detect the model path for the given board size."""
    candidates = [
        os.path.join(script_dir, '..', 'src', 'cpp', 'models', f'_{rows}x{cols}', f'alphazero_{rows}x{cols}.pt'),
        os.path.join(script_dir, 'cpp', 'models', f'_{rows}x{cols}', f'alphazero_{rows}x{cols}.pt'),
        os.path.join(script_dir, '..', 'models', f'alphazero_{rows}x{cols}.pt'),
        os.path.join(script_dir, '..', 'models', f'alphazero_bit_{rows}x{cols}.pth'),
    ]
    for path in candidates:
        resolved = os.path.abspath(path)
        if os.path.exists(resolved):
            return resolved
    return None


def read_model_info(rows, cols):
    """Read model_info.json to get hidden_size and num_res_blocks."""
    info_candidates = [
        os.path.join(script_dir, '..', 'src', 'cpp', 'models', f'_{rows}x{cols}', 'model_info.json'),
        os.path.join(script_dir, 'cpp', 'models', f'_{rows}x{cols}', 'model_info.json'),
    ]
    for path in info_candidates:
        resolved = os.path.abspath(path)
        if os.path.exists(resolved):
            with open(resolved) as f:
                return json.load(f)
    return None


def make_opponent(env, opponent_type, ab_depth=3,use_dag=False):
    """Create an opponent agent."""
    if opponent_type == 'random':
        return RandomAgent(env), 'Random'
    elif opponent_type == 'greedy':
        return GreedyAgent(env), 'Greedy'
    elif opponent_type == 'alphabeta':
        return AlphaBetaAgent(env, depth=ab_depth, parallel=True), f'AlphaBeta(d={ab_depth})'
    else:
        raise ValueError(f"Unknown opponent: {opponent_type}")


def _play_chunk(args_tuple):
    """
    Worker: play a chunk of games with its own C++ server process.
    Returns partial stats dict.
    """
    if len(args_tuple) == 12:
        args_tuple = (*args_tuple, False)
    (rows, cols, opponent_type, n_sims, game_indices,
     model_path, ab_depth, hidden_size, num_res_blocks, worker_id,
     log_every, log_level, use_dag) = args_tuple

    configure_logging(log_level)
    logger.info(
        "Worker %s starting: %sx%s vs %s, sims=%s, games=%s",
        worker_id,
        rows,
        cols,
        opponent_type,
        n_sims,
        len(game_indices),
    )

    # Important: BaseBoardEnv(rows) defaults to a square board (rows×rows).
    # For non-square boards (e.g., 4×3) this must pass both dimensions,
    # otherwise the C++ AlphaZero server/model will see the wrong input size.
    env = BaseBoardEnv(rows, cols)
    if env.rows != rows or env.cols != cols:
        raise RuntimeError(
            f"Environment size mismatch: got {env.rows}x{env.cols}, expected {rows}x{cols}"
        )
    opp_agent, opp_name = make_opponent(env, opponent_type, ab_depth,use_dag)

    az_agent = AlphaZeroCppAgent(
        env,
        model_path=model_path,
        n_simulations=n_sims,
        hidden_size=hidden_size,
        num_res_blocks=num_res_blocks,
        use_dag=use_dag
    )

    az_wins = 0
    opp_wins = 0
    draws = 0
    total_score_diff = 0
    total_az_move_time = 0.0
    total_az_moves = 0

    start_t = time.time()
    for i, game_idx in enumerate(game_indices, start=1):
        env.reset()
        az_player = 1 if game_idx % 2 == 0 else 2

        if hasattr(opp_agent, 'env'):
            opp_agent.env = env
        if hasattr(opp_agent, 'reset'):
            opp_agent.reset()

        game_az_time = 0.0
        game_az_moves = 0

        while not env.done:
            if env.current_player == az_player:
                t0 = time.time()
                action = az_agent.act()
                t1 = time.time()
                game_az_time += (t1 - t0)
                game_az_moves += 1
            else:
                action = opp_agent.act()
            env.step(action)

        s1, s2 = env.score
        az_score = s1 if az_player == 1 else s2
        opp_score = s2 if az_player == 1 else s1
        diff = az_score - opp_score
        total_score_diff += diff

        if diff > 0:
            az_wins += 1
        elif diff < 0:
            opp_wins += 1
        else:
            draws += 1

        total_az_move_time += game_az_time
        total_az_moves += game_az_moves

        if log_every and (i % log_every == 0):
            elapsed_s = time.time() - start_t
            logger.info(
                "Worker %s progress: %s/%s games (%.1fs elapsed)",
                worker_id,
                i,
                len(game_indices),
                elapsed_s,
            )

    az_agent.close()

    elapsed_s = time.time() - start_t
    logger.info(
        "Worker %s finished: %s games in %.1fs (%sW/%sL/%sD)",
        worker_id,
        len(game_indices),
        elapsed_s,
        az_wins,
        opp_wins,
        draws,
    )

    return {
        'worker_id': worker_id,
        'opp_name': opp_name,
        'games_played': len(game_indices),
        'az_wins': az_wins,
        'opp_wins': opp_wins,
        'draws': draws,
        'total_score_diff': total_score_diff,
        'total_az_move_time': total_az_move_time,
        'total_az_moves': total_az_moves,
    }


def evaluate(rows, cols, opponent_type, n_sims, n_games, model_path=None,
             ab_depth=3, hidden_size=128, num_res_blocks=6, n_workers=8,
             log_every=20, log_level="INFO", use_dag=False, out_txt_path: str | None = None):
    """
    Run evaluation: AlphaZero (C++) vs opponent.
    Games are split across n_workers parallel processes, each with its own
    C++ server. Returns dict with aggregated results.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    configure_logging(log_level)

    if model_path is None:
        model_path = find_model_path(rows, cols)

    if model_path is None:
        print(f"ERROR: No model found for {rows}x{cols}. Train one first!")
        print(f"  Run: cd src/cpp/build && ./alphazero_train --config {rows}x{cols}")
        return None

    # Try to read model info for architecture params
    info = read_model_info(rows, cols)
    if info:
        hidden_size = info.get('hidden_size', hidden_size)
        num_res_blocks = info.get('num_res_blocks', num_res_blocks)

    # Cap workers to game count
    actual_workers = min(n_workers, n_games)

    print(f"\n{'='*60}")
    print(f"  EVALUATION: AlphaZero vs {opponent_type.upper()}")
    print(f"  Board: {rows}x{cols}  |  Sims: {n_sims}  |  Games: {n_games}")
    print(f"  Workers: {actual_workers} parallel C++ servers")
    print(f"  Model: {model_path}")
    print(f"  Network: hidden={hidden_size}, blocks={num_res_blocks}")
    print(f"{'='*60}\n")

    # Split game indices into chunks for each worker
    all_indices = list(range(n_games))
    chunks = [[] for _ in range(actual_workers)]
    for i, idx in enumerate(all_indices):
        chunks[i % actual_workers].append(idx)

    # Build worker args
    worker_args = [
        (rows, cols, opponent_type, n_sims, chunk,
         model_path, ab_depth, hidden_size, num_res_blocks, w_id,
         log_every, log_level, use_dag)
        for w_id, chunk in enumerate(chunks) if chunk
    ]

    # Run in parallel
    opp_name = None
    partial_results = []

    if actual_workers > 1:
        with ProcessPoolExecutor(max_workers=actual_workers) as executor:
            futures = {executor.submit(_play_chunk, args): args for args in worker_args}
            for future in as_completed(futures):
                try:
                    partial = future.result()
                    partial_results.append(partial)
                    opp_name = partial['opp_name']
                    done = sum(p['games_played'] for p in partial_results)
                    wins = sum(p['az_wins'] for p in partial_results)
                    wr = wins / done * 100 if done > 0 else 0
                    print(f"  ⚡ Worker {partial['worker_id']} done: "
                          f"{partial['games_played']} games, "
                          f"{partial['az_wins']}W/{partial['opp_wins']}L/{partial['draws']}D  "
                          f"[total {done}/{n_games}, {wr:.1f}%]")
                except Exception as e:
                    print(f"  ERROR in worker: {e}")
    else:
        partial = _play_chunk(worker_args[0])
        partial_results.append(partial)
        opp_name = partial['opp_name']

    # Aggregate results
    az_wins = sum(p['az_wins'] for p in partial_results)
    opp_wins = sum(p['opp_wins'] for p in partial_results)
    draws_total = sum(p['draws'] for p in partial_results)
    total_score_diff = sum(p['total_score_diff'] for p in partial_results)
    total_az_move_time = sum(p['total_az_move_time'] for p in partial_results)
    total_az_moves = sum(p['total_az_moves'] for p in partial_results)
    games_played = sum(p['games_played'] for p in partial_results)

    win_rate = az_wins / games_played * 100 if games_played > 0 else 0
    avg_diff = total_score_diff / games_played if games_played > 0 else 0
    avg_time_per_move = (total_az_move_time / total_az_moves * 1000) if total_az_moves > 0 else 0

    results = {
        'board': f'{rows}x{cols}',
        'opponent': opp_name or opponent_type,
        'sims_per_move': n_sims,
        'games': games_played,
        'az_wins': az_wins,
        'opp_wins': opp_wins,
        'draws': draws_total,
        'win_rate': win_rate,
        'avg_score_diff': avg_diff,
        'avg_time_per_move_ms': avg_time_per_move,
    }

    print(f"\n{'─'*60}")
    print(f"  RESULTS: AlphaZero vs {opp_name}")
    print(f"  Board: {rows}x{cols}  |  Sims/move: {n_sims}")
    print(f"  Games: {games_played}  |  Workers: {actual_workers}")
    print(f"  AZ Wins: {az_wins}  |  Opp Wins: {opp_wins}  |  Draws: {draws_total}")
    print(f"  Win-rate:         {win_rate:.1f}%")
    print(f"  Avg score diff:   {avg_diff:+.2f}")
    print(f"  Avg time/move:    {avg_time_per_move:.1f} ms")
    print(f"{'─'*60}\n")

    if out_txt_path:
        write_results_txt(results, out_txt_path)
        print(f"  Txt saved to: {out_txt_path}\n")

    return results


def _eval_worker(args_tuple):
    """Worker function for parallel evaluation (must be top-level for pickling)."""
    if len(args_tuple) == 12:
        args_tuple = (*args_tuple, False)
    (rows, cols, opp, sims, n_games, model_path, ab_depth, hidden_size,
     num_res_blocks, n_workers, log_every, log_level, use_dag) = args_tuple
    return evaluate(rows, cols, opp, sims, n_games, model_path,
                    ab_depth, hidden_size, num_res_blocks, n_workers,
                    log_every=log_every, log_level=log_level, use_dag=use_dag)


def get_configs(rows, cols):
    """Return the list of (opponent, sims, ab_depth) configs for a board size."""
    if rows == 2 and cols == 2:
        return [
            ('random', 200, 3),
            ('greedy', 200, 3),
        ]
    elif rows == 3 and cols == 3:
        return [
            ('random', 400, 3),
            ('greedy', 400, 3),
            ('greedy', 800, 3),
            ('alphabeta', 800, 3),
        ]
    elif rows == 5 and cols == 5:
        return [
            ('random', 600, 3),
            ('greedy', 600, 3),
            ('alphabeta', 1000, 3),
        ]
    else:
        return [
            ('random', 200, 3),
            ('greedy', 200, 3),
        ]


def print_summary(board_key, all_results, out_path):
    """Print the results summary table and save to JSON."""
    print(f"\n{'='*70}")
    print(f"  SUMMARY TABLE — {board_key} Board")
    print(f"{'='*70}")
    print(f"  {'Opponent':<20s} {'Sims':>6s} {'Win%':>7s} {'Δ̄':>7s} {'ms/move':>8s}")
    print(f"  {'─'*50}")
    for r in all_results:
        print(f"  {r['opponent']:<20s} {r['sims_per_move']:>6d} "
              f"{r['win_rate']:>6.1f}% {r['avg_score_diff']:>+6.2f} "
              f"{r['avg_time_per_move_ms']:>7.1f}")
    print()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"  Results saved to: {out_path}")


def run_all_evaluations(rows, cols, n_games, model_path=None,
                        hidden_size=128, num_res_blocks=6, parallel=True, n_workers=8,
                        log_every=20, log_level="INFO", use_dag=False, out_txt_path: str | None = None):
    """Run all evaluation configurations for a given board size."""
    from concurrent.futures import ProcessPoolExecutor, as_completed

    board_key = f'{rows}x{cols}'
    configs = get_configs(rows, cols)

    # Build argument tuples for each config
    worker_args = [
        (rows, cols, opp, sims, n_games, model_path, ab_depth, hidden_size, num_res_blocks,
         n_workers, log_every, log_level, use_dag)
        for opp, sims, ab_depth in configs
    ]

    all_results = []

    if parallel and len(configs) > 1:
        n_workers = len(configs)
        print(f"\n  ⚡ Running {n_workers} evaluations in PARALLEL "
              f"(each gets its own C++ server process)\n")

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_eval_worker, args): args for args in worker_args}
            for future in as_completed(futures):
                args = futures[future]
                opp_name = args[2]
                try:
                    result = future.result()
                    if result:
                        all_results.append(result)
                except Exception as e:
                    print(f"  ERROR in {opp_name}: {e}")
    else:
        for args in worker_args:
            result = _eval_worker(args)
            if result:
                all_results.append(result)

    # Sort results to match config order (parallel may finish out of order)
    config_order = {(opp, sims): i for i, (opp, sims, _) in enumerate(configs)}
    all_results.sort(key=lambda r: config_order.get((r['opponent'].lower().split('(')[0].strip(),
                                                      r['sims_per_move']), 999))

    # Print & save
    out_dir = os.path.join(script_dir, 'training_logs')
    out_path = os.path.join(out_dir, f'eval_{board_key}.json')
    print_summary(board_key, all_results, out_path)

    if out_txt_path:
        os.makedirs(os.path.dirname(out_txt_path), exist_ok=True)
        with open(out_txt_path, "w", encoding="utf-8") as f:
            for r in all_results:
                az_wins = int(r.get("az_wins", 0))
                opp_wins = int(r.get("opp_wins", 0))
                draws = int(r.get("draws", 0))
                if az_wins > opp_wins:
                    overall = "WIN"
                elif az_wins < opp_wins:
                    overall = "LOSS"
                else:
                    overall = "DRAW"
                f.write(
                    f"{r.get('opponent','')}: "
                    f"W={az_wins} L={opp_wins} D={draws} "
                    f"Result={overall} Win%={r.get('win_rate', 0):.1f}\n"
                )
        print(f"  Txt saved to: {out_txt_path}\n")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate AlphaZero agent for the research paper',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test (10 games):
  python evaluate_for_paper.py --rows 2 --cols 2 --opponent random --games 10

  # Full 2x2 evaluation (parallel by default):
  python evaluate_for_paper.py --rows 2 --cols 2 --all --games 200

  # Control parallelism (8 games at once by default):
  python evaluate_for_paper.py --rows 2 --cols 2 --opponent greedy --games 200 --workers 10

  # Sequential mode:
  python evaluate_for_paper.py --rows 2 --cols 2 --all --games 200 --workers 1

  # Single opponent:
  python evaluate_for_paper.py --rows 3 --cols 3 --opponent greedy --sims 400 --games 200
        """)

    parser.add_argument('--rows', type=int, default=2, help='Board rows')
    parser.add_argument('--cols', type=int, default=None, help='Board cols (default: same as rows)')
    parser.add_argument('--opponent', type=str, default='random',
                       choices=['random', 'greedy', 'alphabeta'],
                       help='Opponent type')
    parser.add_argument('--use-dag', type=bool, default=False, help='Use DAG in MCTS')
    parser.add_argument('--sims', type=int, default=200, help='MCTS simulations per move')
    parser.add_argument('--games', type=int, default=200, help='Number of games')
    parser.add_argument('--model', type=str, default=None, help='Model path (auto-detected if not given)')
    parser.add_argument('--ab-depth', type=int, default=3, help='Alpha-beta depth')
    parser.add_argument('--hidden', type=int, default=128, help='Hidden size')
    parser.add_argument('--blocks', type=int, default=6, help='Residual blocks')
    parser.add_argument('--all', action='store_true', help='Run ALL evaluations for this board size')
    parser.add_argument('--workers', '-w', type=int, default=8,
                       help='Number of parallel game workers (default: 8)')
    parser.add_argument('--no-parallel', action='store_true',
                       help='Disable parallel across opponent types with --all')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level')
    parser.add_argument('--log-every', type=int, default=20,
                        help='Log progress every N games per worker (0 disables)')
    parser.add_argument('--out-txt', type=str, default=None,
                        help='Write final W/L/Result summary to a .txt file')

    args = parser.parse_args()
    configure_logging(args.log_level)
    
    if args.cols is None:
        args.cols = args.rows

    # Try to auto-detect hidden/blocks from model_info.json
    info = read_model_info(args.rows, args.cols)
    if info:
        if args.hidden == 128:  # not explicitly set
            args.hidden = info.get('hidden_size', args.hidden)
        if args.blocks == 6:  # not explicitly set
            args.blocks = info.get('num_res_blocks', args.blocks)

    if args.all:
        out_txt_path = args.out_txt
        if out_txt_path is None:
            out_txt_path = os.path.join(script_dir, 'training_logs', f'eval_{args.rows}x{args.cols}.txt')
        run_all_evaluations(args.rows, args.cols, args.games, args.model,
                           args.hidden, args.blocks,
                           parallel=not args.no_parallel, n_workers=args.workers,
                           log_every=args.log_every, log_level=args.log_level, use_dag=args.use_dag,
                           out_txt_path=out_txt_path)
    else:
        out_txt_path = args.out_txt
        if out_txt_path is None:
            out_txt_path = os.path.join(
                script_dir, 'training_logs', f'eval_{args.rows}x{args.cols}_{args.opponent}.txt'
            )
        evaluate(args.rows, args.cols, args.opponent, args.sims, args.games,
                args.model, args.ab_depth, args.hidden, args.blocks,
                n_workers=args.workers,
                log_every=args.log_every, log_level=args.log_level, use_dag=args.use_dag,
                out_txt_path=out_txt_path)


if __name__ == '__main__':
    main()
