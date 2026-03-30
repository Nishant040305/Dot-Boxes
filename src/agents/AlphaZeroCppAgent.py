"""
AlphaZero C++ Agent — Python wrapper that communicates with the C++ server.

Spawns the C++ `alphazero_server` process and communicates via stdin/stdout
using a JSON-line protocol. This agent plugs into the existing Python
simulation/arena framework as a drop-in replacement.

Usage:
    from agents.AlphaZeroCppAgent import AlphaZeroCppAgent
    agent = AlphaZeroCppAgent(env, model_path='models/alphazero_3x3.pt')
    action = agent.act()
"""

import json
import os
import subprocess
import sys
import atexit

from agents.agent import Agent


class AlphaZeroCppAgent(Agent):
    """AlphaZero agent backed by the C++ alphazero_server process."""

    def __init__(self, env, model_path=None, n_simulations=400,
                 hidden_size=256, num_res_blocks=6, server_binary=None):
        """
        Args:
            env: BitBoardEnv (Python) — used for reading state.
            model_path: Path to the .pt model trained by alphazero_train.
            n_simulations: MCTS simulations per move.
            hidden_size: Must match the model architecture.
            num_res_blocks: Must match the model architecture.
            server_binary: Path to alphazero_server binary.
                           Auto-detected if None.
        """
        super().__init__(env)
        self.n = env.N

        if model_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            root_dir = os.path.dirname(os.path.dirname(current_dir))
            models_dir = os.path.join(root_dir, 'models')
            model_path = os.path.join(models_dir, f"alphazero_{env.N}x{env.N}.pt")

        if server_binary is None:
            server_binary = self._find_server_binary()

        self._proc = self._start_server(
            server_binary, env.N, env.N, model_path,
            n_simulations, hidden_size, num_res_blocks
        )

        # Register cleanup
        atexit.register(self.close)

    def _find_server_binary(self):
        """Search for alphazero_server binary in common locations."""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        src_dir = os.path.dirname(script_dir)
        candidates = [
            os.path.join(src_dir, 'cpp', 'build', 'alphazero_server'),
            os.path.join(src_dir, '..', 'build', 'alphazero_server'),
        ]
        for path in candidates:
            path = os.path.abspath(path)
            if os.path.isfile(path) and os.access(path, os.X_OK):
                return path
        raise FileNotFoundError(
            f"alphazero_server not found. Searched:\n" +
            "\n".join(f"  - {os.path.abspath(c)}" for c in candidates) +
            "\nBuild it with: cd src/cpp/build && cmake .. && make -j$(nproc)"
        )

    def _start_server(self, binary, rows, cols, model_path,
                      n_sims, hidden, blocks):
        """Launch the C++ server process."""
        cmd = [
            binary,
            '--rows', str(rows),
            '--cols', str(cols),
            '--model', model_path,
            '--sims', str(n_sims),
            '--hidden', str(hidden),
            '--blocks', str(blocks),
        ]

        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # line-buffered
        )

        # Read the ready signal
        ready_line = proc.stdout.readline().strip()
        if not ready_line:
            stderr = proc.stderr.read()
            raise RuntimeError(
                f"Server failed to start.\nstderr: {stderr}"
            )

        ready = json.loads(ready_line)
        if ready.get('status') != 'ready':
            raise RuntimeError(f"Unexpected server response: {ready_line}")

        # Print any stderr diagnostics
        # (non-blocking read would be ideal, but stderr is usually small)
        print(f"[AlphaZeroCppAgent] Server ready "
              f"(rows={ready['rows']}, cols={ready['cols']}, "
              f"action_size={ready['action_size']})")

        return proc

    def act(self, return_probs=False, temperature=1e-3, add_noise=True):
        """Run MCTS via the C++ server and return an action."""
        env = self.env

        # Send current state to C++ server
        request = {
            'cmd': 'act',
            'h_edges': env.h_edges,
            'v_edges': env.v_edges,
            'boxes_p1': env.boxes_p1,
            'boxes_p2': env.boxes_p2,
            'current_player': env.current_player,
            'score_p1': env.score[0],
            'score_p2': env.score[1],
            'done': env.done,
        }

        self._send(request)
        response = self._recv()

        edge_type = response['edge_type']
        r = response['r']
        c = response['c']

        if edge_type == -1:
            return None  # Game is already done

        action = (edge_type, r, c)

        if return_probs:
            # C++ server doesn't return visit counts in this mode,
            # so return empty dict
            return action, {}
        return action

    def reset(self):
        """Reset notification (optional)."""
        self._send({'cmd': 'reset'})
        self._recv()

    def close(self):
        """Shut down the C++ server process."""
        if hasattr(self, '_proc') and self._proc and self._proc.poll() is None:
            try:
                self._send({'cmd': 'quit'})
                self._proc.wait(timeout=5)
            except Exception:
                self._proc.kill()
            self._proc = None

    def _send(self, obj):
        """Send a JSON object to the server."""
        line = json.dumps(obj, separators=(',', ':'))
        self._proc.stdin.write(line + '\n')
        self._proc.stdin.flush()

    def _recv(self):
        """Read a JSON response from the server."""
        line = self._proc.stdout.readline().strip()
        if not line:
            stderr = self._proc.stderr.read()
            raise RuntimeError(
                f"Server died unexpectedly.\nstderr: {stderr}"
            )
        return json.loads(line)

    def __del__(self):
        self.close()
