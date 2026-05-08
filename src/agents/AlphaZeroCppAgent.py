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
                 hidden_size=256, num_res_blocks=6, server_binary=None,
                 use_dag=True, value_eval='score_diff_scaled'):
        """
        Args:
            env: BitBoardEnv (Python) — used for reading state.
            model_path: Path to the .pt model trained by alphazero_train.
            n_simulations: MCTS simulations per move.
            hidden_size: Must match the model architecture.
            num_res_blocks: Must match the model architecture.
            server_binary: Path to alphazero_server binary.
                           Auto-detected if None.
            use_dag: Enable DAG transpositions in the C++ MCTS.
        """
        super().__init__(env)
        self.rows = getattr(env, 'rows', env.N)
        self.cols = getattr(env, 'cols', env.N)
        self.use_dag = use_dag

        if model_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            root_dir = os.path.dirname(os.path.dirname(current_dir))
            
            # List of candidate paths to check
            candidates = [
                os.path.join(root_dir, 'src', 'cpp', 'models', f"_{self.rows}x{self.cols}_cnn", f"alphazero_{self.rows}x{self.cols}.pt"),
                os.path.join(root_dir, 'src', 'cpp', 'models', f"_{self.rows}x{self.cols}", f"alphazero_{self.rows}x{self.cols}.pt"),
                os.path.join(root_dir, 'models', f"alphazero_{self.rows}x{self.cols}.pt")
            ]
            
            for cand in candidates:
                if os.path.exists(cand):
                    model_path = cand
                    break
                    
            if model_path is None:
                # Fallback to standard if none found, to let the error propagate later
                model_path = candidates[1]

        if server_binary is None:
            server_binary = self._find_server_binary()
            
        # Auto-detect architecture from model_info.json sidecar written by alphazero_train.
        # We cannot inspect the TorchScript .pt archive as a state_dict (torch::save writes
        # a full module archive, not a plain dict), so we use the sidecar instead.
        model_dir = os.path.dirname(model_path)
        info_path = os.path.join(model_dir, 'model_info.json')
        use_patch_net = False
        patch_rows = None
        patch_cols = None
        local_hidden = None
        local_blocks = None
        global_hidden = None
        global_blocks = None
        local_model_path = None
        use_cnn_net = False
        cnn_channels = None

        if os.path.exists(info_path):
            try:
                with open(info_path, 'r') as f:
                    info = json.load(f)
                print(info)
                hidden_size = info.get('hidden_size', hidden_size)
                num_res_blocks = info.get('num_res_blocks', num_res_blocks)
                value_eval = info.get('value_eval', value_eval)
                use_patch_net = info.get('use_patch_net', False)
                patch_rows = info.get('patch_rows')
                patch_cols = info.get('patch_cols')
                local_hidden = info.get('local_hidden_size')
                local_blocks = info.get('local_num_res_blocks')
                global_hidden = info.get('global_hidden_size')
                global_blocks = info.get('global_num_res_blocks')
                local_model_path = info.get('local_model_path')
                use_cnn_net = info.get('use_cnn_net', False)
                cnn_channels = info.get('cnn_channels')
                print(f"[AlphaZeroCppAgent] Detected architecture: "
                      f"hidden={hidden_size}, blocks={num_res_blocks}, "
                      f"value_eval={value_eval}")
            except Exception as e:
                print(f"[AlphaZeroCppAgent] Could not read model_info.json: {e}, using defaults.")

        self._proc = self._start_server(
            server_binary, self.rows, self.cols, model_path,
            n_simulations, hidden_size, num_res_blocks, use_dag,
            value_eval,
            use_patch_net=use_patch_net,
            patch_rows=patch_rows,
            patch_cols=patch_cols,
            local_hidden=local_hidden,
            local_blocks=local_blocks,
            global_hidden=global_hidden,
            global_blocks=global_blocks,
            local_model_path=local_model_path,
            use_cnn_net=use_cnn_net,
            cnn_channels=cnn_channels,
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
                      n_sims, hidden, blocks, use_dag, value_eval,
                      use_patch_net=False, patch_rows=None, patch_cols=None,
                      local_hidden=None, local_blocks=None,
                      global_hidden=None, global_blocks=None,
                      local_model_path=None,
                      use_cnn_net=False, cnn_channels=None):
        """Launch the C++ server process."""
        cmd = [
            binary,
            '--rows', str(rows),
            '--cols', str(cols),
            '--model', model_path,
            '--sims', str(n_sims),
            '--hidden', str(hidden),
            '--blocks', str(blocks),
            '--value-eval', str(value_eval),
        ]
        if use_patch_net:
            cmd.append('--patch')
            if patch_rows is not None:
                cmd.extend(['--patch-rows', str(patch_rows)])
            if patch_cols is not None:
                cmd.extend(['--patch-cols', str(patch_cols)])
            if local_hidden is not None:
                cmd.extend(['--local-hidden', str(local_hidden)])
            if local_blocks is not None:
                cmd.extend(['--local-blocks', str(local_blocks)])
            if global_hidden is not None:
                cmd.extend(['--global-hidden', str(global_hidden)])
            if global_blocks is not None:
                cmd.extend(['--global-blocks', str(global_blocks)])
            if local_model_path:
                resolved = None
                if os.path.isabs(local_model_path):
                    resolved = local_model_path
                else:
                    model_dir = os.path.dirname(model_path)
                    repo_root = os.path.abspath(
                        os.path.join(os.path.dirname(__file__), '..', '..')
                    )
                    candidates = [
                        os.path.abspath(os.path.join(model_dir, local_model_path)),
                        os.path.abspath(os.path.join(repo_root, local_model_path)),
                    ]
                    for cand in candidates:
                        if os.path.exists(cand):
                            resolved = cand
                            break
                if resolved:
                    cmd.extend(['--local-model', resolved])
                else:
                    print(f"[AlphaZeroCppAgent] WARNING: local_model_path not found: "
                          f"{local_model_path}. Continuing without it.")
        elif use_cnn_net:
            cmd.append('--cnn')
            if cnn_channels is not None:
                cmd.extend(['--cnn-channels', str(cnn_channels)])
                
        if not use_dag:
            cmd.append('--no-dag')

        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,
            text=True,
            bufsize=1,  # line-buffered
        )

        # Read the ready signal
        ready_line = proc.stdout.readline().strip()
        if not ready_line:
            raise RuntimeError(
                "Server failed to start. See terminal for stderr output."
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
        if hasattr(env, 'h_edges'):
            h_edges = env.h_edges
            v_edges = env.v_edges
            boxes_p1 = env.boxes_p1
            boxes_p2 = env.boxes_p2
        else:
            h_edges = 0
            for r in range(self.rows + 1):
                for c in range(self.cols):
                    if env.horizontal_edges[r][c]:
                        h_edges |= (1 << (r * self.cols + c))
            v_edges = 0
            for r in range(self.rows):
                for c in range(self.cols + 1):
                    if env.vertical_edges[r][c]:
                        v_edges |= (1 << (r * (self.cols + 1) + c))
            boxes_p1 = 0
            boxes_p2 = 0
            for r in range(self.rows):
                for c in range(self.cols):
                    if env.boxes[r][c] == 1:
                        boxes_p1 |= (1 << (r * self.cols + c))
                    elif env.boxes[r][c] == 2:
                        boxes_p2 |= (1 << (r * self.cols + c))

        request = {
            'cmd': 'act',
            'h_edges': h_edges,
            'v_edges': v_edges,
            'boxes_p1': boxes_p1,
            'boxes_p2': boxes_p2,
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
            raise RuntimeError(
                "Server died unexpectedly. See terminal for stderr output."
            )
        return json.loads(line)

    def __del__(self):
        self.close()
