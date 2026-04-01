#!/usr/bin/env python3
"""
Web-based interactive Dots and Boxes server.

Bridges the Flask/SocketIO frontend to the game engine and all available agents
(including AlphaZero C++).

Usage:
    python ui/play_server.py                    # default: port 5050
    python ui/play_server.py --port 8080        # custom port
    python ui/play_server.py --size 4           # 4x4 board
"""

import os
import sys
import argparse
import time
import threading
import json

# Ensure src is on the path
_script_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.dirname(_script_dir)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from flask import Flask, send_from_directory, jsonify, request
from flask_socketio import SocketIO, emit

# ─── Game engine imports ──────────────────────────────────────────
from gameSimulation.arena import make_env, make_agent, parse_agent_spec

# ─── App setup ────────────────────────────────────────────────────
app = Flask(__name__, static_folder='.', static_url_path='')
app.config['SECRET_KEY'] = 'dots-and-boxes-secret'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# ─── Game state (per-session, single-player for now) ──────────────
game_lock = threading.Lock()
game_state = {
    'env': None,
    'agent': None,
    'agent_spec': None,
    'board_size': 3,
    'env_type': 'bit',
    'human_player': 1,
    'game_active': False,
}


# ─── Available agents ────────────────────────────────────────────
AGENTS = [
    {'id': 'random', 'name': 'Random Agent', 'spec': 'random',
     'description': 'Makes completely random moves', 'difficulty': 1},
    {'id': 'greedy', 'name': 'Greedy Agent', 'spec': 'greedy',
     'description': 'Always captures boxes when possible', 'difficulty': 2},
    {'id': 'minmax3', 'name': 'MinMax (depth 3)', 'spec': 'minmax:depth=3',
     'description': 'Minimax search 3 moves deep', 'difficulty': 3},
    {'id': 'minmax5', 'name': 'MinMax (depth 5)', 'spec': 'minmax:depth=5',
     'description': 'Minimax search 5 moves deep', 'difficulty': 4},
    {'id': 'alphabeta3', 'name': 'AlphaBeta (depth 3)', 'spec': 'alphabeta:depth=3',
     'description': 'Alpha-beta pruning, depth 3', 'difficulty': 3},
    {'id': 'alphabeta5', 'name': 'AlphaBeta (depth 5)', 'spec': 'alphabeta:depth=5',
     'description': 'Alpha-beta pruning, depth 5', 'difficulty': 5},
    {'id': 'mcts500', 'name': 'MCTS (500 iter)', 'spec': 'mcts:iterations=500',
     'description': 'Monte Carlo Tree Search, 500 simulations', 'difficulty': 4},
    {'id': 'mcts1000', 'name': 'MCTS (1000 iter)', 'spec': 'mcts:iterations=1000',
     'description': 'Monte Carlo Tree Search, 1000 simulations', 'difficulty': 5},
    {'id': 'alphazero_bit', 'name': 'AlphaZero (Python)', 'spec': 'alphazero_bit:n_simulations=400',
     'description': 'Neural MCTS with bitboard (Python)', 'difficulty': 7},
    {'id': 'alphazero_cpp', 'name': 'AlphaZero (C++)', 'spec': 'alphazero_cpp:n_simulations=400',
     'description': 'Neural MCTS via C++ server — fastest engine', 'difficulty': 8},
]


def _env_to_state(env, board_size):
    """Convert the current env into a JSON-serializable dict for the frontend."""
    N = board_size
    # Build 2D arrays regardless of env type
    if hasattr(env, 'horizontal_edges') and callable(getattr(env, 'horizontal_edges', None)):
        h_edges = env.horizontal_edges
        v_edges = env.vertical_edges
        boxes = env.boxes
    elif hasattr(env, 'h_edges'):
        # BitBoardEnv — use compatibility properties
        h_edges = env.horizontal_edges  # property returns 2D
        v_edges = env.vertical_edges
        boxes = env.boxes
    else:
        h_edges = env.horizontal_edges
        v_edges = env.vertical_edges
        boxes = env.boxes

    # Ensure they are plain lists
    def _to_list(arr):
        if hasattr(arr, 'tolist'):
            return arr.tolist()
        return [list(row) for row in arr]

    return {
        'horizontal_edges': _to_list(h_edges),
        'vertical_edges': _to_list(v_edges),
        'boxes': _to_list(boxes),
        'current_player': env.current_player,
        'score': list(env.score),
        'done': env.done,
        'board_size': board_size,
    }


def _action_to_rc(action):
    """Convert internal action (type, i, j) → visual-grid (row, col)."""
    if action[0] == 0:
        return (2 * action[1], 2 * action[2] + 1)
    else:
        return (2 * action[1] + 1, 2 * action[2])


def _action_from_rc(row, col):
    """Convert visual-grid (row, col) → internal action (type, i, j)."""
    if row % 2 == 0 and col % 2 == 1:
        return (0, row // 2, col // 2)
    elif row % 2 == 1 and col % 2 == 0:
        return (1, row // 2, col // 2)
    else:
        raise ValueError(f"Invalid edge position ({row}, {col})")


# ─── HTTP routes ──────────────────────────────────────────────────

@app.route('/')
def index():
    return send_from_directory('.', 'play.html')


@app.route('/api/agents')
def get_agents():
    return jsonify(AGENTS)


@app.route('/api/archive')
def get_archive():
    archive_file = os.path.join(_script_dir, 'game_archive.json')
    if os.path.exists(archive_file):
        try:
            with open(archive_file, 'r') as f:
                return jsonify(json.load(f))
        except Exception:
            pass
    return jsonify([])


def _archive_game(env, game_state):
    from datetime import datetime
    
    archive_file = os.path.join(_script_dir, 'game_archive.json')
    try:
        if os.path.exists(archive_file):
            with open(archive_file, 'r') as f:
                archive = list(json.load(f))
        else:
            archive = []
    except Exception:
        archive = []

    s1, s2 = list(env.score)
    if s1 > s2:
        winner = 'Player 1'
    elif s2 > s1:
        winner = 'Player 2'
    else:
        winner = 'Draw'

    human_player = game_state['human_player']
    if human_player == 1:
        p1 = 'Human'
        p2 = game_state['agent_spec']
    elif human_player == 2:
        p1 = game_state['agent_spec']
        p2 = 'Human'
    else:
        p1 = game_state['agent_spec']
        p2 = game_state['agent2_spec'] if 'agent2_spec' in game_state and game_state['agent2_spec'] else p1

    record = {
        'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'board_size': f"{game_state['board_size'][0]}x{game_state['board_size'][1]}",
        'matchup': f"{p1} vs {p2}",
        'score': f"{s1} - {s2}",
        'winner': winner
    }
    
    archive.insert(0, record)
    while len(archive) > 50:
        archive.pop()
    
    try:
        with open(archive_file, 'w') as f:
            json.dump(archive, f, indent=4)
    except Exception as e:
        print(f"Failed to save archive: {e}")

# ─── SocketIO events ──────────────────────────────────────────────

@socketio.on('connect')
def on_connect():
    print(f'[server] Client connected')
    emit('agents_list', AGENTS)


@socketio.on('new_game')
def on_new_game(data):
    """Start a new game.
    data: { agent_spec: str, board_size: int, human_player: 1|2, env_type: 'bit'|'classic' }
    """
    with game_lock:
        agent_spec = data.get('agent_spec', 'greedy')
        agent2_spec = data.get('agent2_spec', agent_spec)
        board_size_in = data.get('board_size', [3, 3])
        if isinstance(board_size_in, int):
            rows, cols = board_size_in, board_size_in
        elif isinstance(board_size_in, list) and len(board_size_in) == 2:
            rows, cols = board_size_in
        else:
            rows, cols = 3, 3

        board_size = [rows, cols]
        human_player = data.get('human_player', 1)
        env_type = data.get('env_type', 'bit')

        # Determine correct env type for the agent
        agent_type_str, _ = parse_agent_spec(agent_spec)
        a2_type_str, _ = parse_agent_spec(agent2_spec)
        
        # if agent_type_str == 'alphazero_bit' or (human_player == -1 and a2_type_str == 'alphazero_bit'):
        #     if rows != cols:
        #         emit('error', {'message': f'AlphaZero (Python) strictly requires a square board!'})
        #         return
            
        if rows != cols:
            env_type = 'classic' # BitBoardEnv only supports square boards right now
        elif agent_type_str in ('alphazero_cpp', 'alphazero_bit') or (human_player == -1 and a2_type_str in ('alphazero_cpp', 'alphazero_bit')):
            env_type = 'bit'

        if env_type == 'bit':
            env = make_env(rows, 'bit')
        else:
            from env.BoardEnv import BaseBoardEnv
            env = BaseBoardEnv(rows, cols)
            
        env.reset()

        a_type, a_kwargs = parse_agent_spec(agent_spec)
        a2_type, a2_kwargs = parse_agent_spec(agent2_spec)
        try:
            agent = make_agent(a_type, env, **a_kwargs)
            if human_player == -1:
                agent2 = make_agent(a2_type, env, **a2_kwargs)
            else:
                agent2 = None
        except Exception as e:
            emit('error', {'message': f'Failed to create agent: {str(e)}'})
            return

        game_state['env'] = env
        game_state['agent'] = agent
        game_state['agent2'] = agent2
        game_state['agent_spec'] = agent_spec
        game_state['agent2_spec'] = agent2_spec
        game_state['board_size'] = board_size
        game_state['env_type'] = env_type
        game_state['human_player'] = human_player
        game_state['game_active'] = True

        state = _env_to_state(env, board_size)
        state['human_player'] = human_player
        state['agent_name'] = agent_spec
        state['agent2_name'] = agent2_spec
        emit('game_started', state)

    # If AI goes first, make its move
    if human_player in (2, -1):
        socketio.start_background_task(_ai_move)


@socketio.on('human_move')
def on_human_move(data):
    """Human makes a move.
    data: { row: int, col: int }  — visual-grid coordinates
    """
    with game_lock:
        env = game_state['env']
        if not env or not game_state['game_active']:
            emit('error', {'message': 'No active game'})
            return

        if env.done:
            emit('error', {'message': 'Game is already over'})
            return

        human_player = game_state['human_player']
        if env.current_player != human_player:
            emit('error', {'message': 'Not your turn!'})
            return

        row = data['row']
        col = data['col']

        try:
            action = _action_from_rc(row, col)
        except ValueError as e:
            emit('error', {'message': str(e)})
            return

        # Validate move
        available = env._get_available_actions()
        if action not in available:
            emit('error', {'message': f'Invalid move at ({row}, {col})'})
            return

        # Execute human move
        player_before = env.current_player
        env.step(action)

        state = _env_to_state(env, game_state['board_size'])
        state['last_move'] = {'row': row, 'col': col, 'player': player_before,
                              'action': list(action)}
        state['human_player'] = human_player
        emit('state_update', state)

        # If game over, stop
        if env.done:
            game_state['game_active'] = False
            _archive_game(env, game_state)
            emit('game_over', {
                'score': list(env.score),
                'human_player': human_player,
            })
            return

        # If it's still human's turn (box was made), wait for next move
        if env.current_player == human_player:
            return

    # Otherwise, let AI play
    socketio.start_background_task(_ai_move)


def _ai_move():
    """Let the AI agent make move(s) until it's the human's turn or game over."""
    while True:
        with game_lock:
            if not game_state['game_active']:
                break
                
            env = game_state['env']
            human_player = game_state['human_player']
            
            if env.done or env.current_player == human_player:
                break
                
            if human_player == -1:
                agent = game_state['agent'] if env.current_player == 1 else game_state['agent2']
            else:
                agent = game_state['agent']
                
            board_size = game_state['board_size']

            # Point agent to the current env
            agent.env = env

            try:
                start_t = time.time()
                action = agent.act()
                elapsed = time.time() - start_t
            except Exception as e:
                socketio.emit('error', {'message': f'Agent error: {str(e)}'})
                break

            if action is None:
                break

            player_before = env.current_player
            rc = _action_to_rc(action)
            env.step(action)

            state = _env_to_state(env, board_size)
            state['last_move'] = {
                'row': rc[0], 'col': rc[1],
                'player': player_before,
                'action': list(action),
                'think_time': round(elapsed, 3),
            }
            state['human_player'] = human_player

            socketio.emit('state_update', state)

            if env.done:
                game_state['game_active'] = False
                _archive_game(env, game_state)
                socketio.emit('game_over', {
                    'score': list(env.score),
                    'human_player': human_player,
                })
                break

        # Small delay outside the lock so the frontend can animate, and humans can interrupt
        time.sleep(0.5)


@socketio.on('get_available_moves')
def on_get_available():
    """Return available edge positions in visual-grid coordinates."""
    with game_lock:
        env = game_state['env']
        if not env or env.done:
            emit('available_moves', {'moves': []})
            return

        actions = env._get_available_actions()
        moves = []
        for a in actions:
            rc = _action_to_rc(a)
            moves.append({'row': rc[0], 'col': rc[1], 'type': 'h' if a[0] == 0 else 'v'})
        emit('available_moves', {'moves': moves})


@socketio.on('disconnect')
def on_disconnect():
    print('[server] Client disconnected')
    # Clean up C++ agent if needed
    with game_lock:
        agent = game_state.get('agent')
        if agent and hasattr(agent, 'close'):
            try:
                agent.close()
            except Exception:
                pass
        game_state['agent'] = None
        game_state['game_active'] = False


# ─── Main ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Dots and Boxes Visual Play Server')
    parser.add_argument('--port', '-p', type=int, default=5050)
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    print(f"\n  🎮  Dots & Boxes — Visual Play Server")
    print(f"  ────────────────────────────────────────")
    print(f"  Open http://localhost:{args.port} in your browser")
    print(f"  ────────────────────────────────────────\n")

    socketio.run(app, host=args.host, port=args.port,
                 debug=args.debug, allow_unsafe_werkzeug=True)


if __name__ == '__main__':
    main()
