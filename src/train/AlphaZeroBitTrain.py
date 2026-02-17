"""
AlphaZero Training with Bitboard representation — Optimized.

Key improvements over base trainer:
  - Symmetry augmentation (8x data for free)
  - Phased training schedule (bootstrap → refinement → mastery)
  - LR schedule with cosine annealing + weight decay
  - Temperature schedule (explore early, exploit late)
  - Gradient clipping
  - Separate loss tracking (policy vs value)
"""

import sys
import os
import random
import time
import queue
import traceback
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp
from collections import deque

# Add src to path
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(script_dir)
sys.path.append(src_dir)

from env.BitBoardEnv import BitBoardEnv
from agents.AlphaZeroBitAgent import AlphaZeroBitAgent, BitNode, _NodeState
from models.BitNet import AlphaZeroBitNet
from util.TimeProfile import TimeProfile

# ════════════════════════════════════════════════════════════════════
# Configuration
# ════════════════════════════════════════════════════════════════════
N = 3
NUM_WORKERS = 16
BUFFER_CAPACITY = 100000
BATCH_SIZE = 256
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
VALUE_LOSS_WEIGHT = 1.0       # weight on MSE value loss vs CE policy loss

# Phased training schedule
#   Phase 1 (Bootstrap):   explore aggressively, lower sims
#   Phase 2 (Refinement):  balance explore/exploit, medium sims
#   Phase 3 (Mastery):     mostly exploit, high sims
PHASES = [
    {
        'name': 'Bootstrap',
        'iterations': 15,
        'mcts_sims': 200,
        'episodes_per_iter': 100,
        'epochs': 10,
        'lr': 0.002,
        'temp_threshold': 15,    # moves before switching to exploit
        'temp_explore': 1.0,
        'temp_exploit': 0.3,
    },
    {
        'name': 'Refinement',
        'iterations': 20,
        'mcts_sims': 400,
        'episodes_per_iter': 100,
        'epochs': 15,
        'lr': 0.001,
        'temp_threshold': 10,
        'temp_explore': 0.8,
        'temp_exploit': 0.1,
    },
    {
        'name': 'Mastery',
        'iterations': 15,
        'mcts_sims': 600,
        'episodes_per_iter': 100,
        'epochs': 20,
        'lr': 0.0003,
        'temp_threshold': 6,
        'temp_explore': 0.8,
        'temp_exploit': 0.0,
    },
]


# ════════════════════════════════════════════════════════════════════
# Symmetry Augmentation
# ════════════════════════════════════════════════════════════════════

class SymmetryAugmenter:
    """
    Precompute permutation tables for all 8 symmetries of the NxN board.

    Dots form an (N+1)×(N+1) grid. Under rotation/flip, dots permute,
    and edges between them change type (h↔v) and position.
    
    For each symmetry, we compute:
      - feature_perm: rearranges the feature vector
      - policy_perm: rearranges the policy vector
    """
    
    def __init__(self, N):
        self.N = N
        self.n_h = (N + 1) * N
        self.n_v = N * (N + 1)
        self.action_size = self.n_h + self.n_v
        self.input_size = self.n_h + self.n_v + N * N + 1
        
        # Precompute all 8 symmetry permutations
        self.feature_perms, self.policy_perms = self._compute_all_symmetries()
    
    def _compute_all_symmetries(self):
        """Compute feature and policy permutations for all 8 symmetries."""
        N = self.N
        
        # All 8 symmetries as (rotation_count, flip_before_rotate)
        # rot: 0°, 90°CW, 180°, 270°CW; flip: horizontal mirror
        symmetry_ops = []
        for flip in [False, True]:
            for rot in range(4):
                symmetry_ops.append((rot, flip))
        
        feature_perms = []
        policy_perms = []
        
        for rot, flip in symmetry_ops:
            fp, pp = self._compute_symmetry(rot, flip)
            feature_perms.append(fp)
            policy_perms.append(pp)
        
        return feature_perms, policy_perms
    
    def _transform_dot(self, r, c, rot, flip, N):
        """Transform dot position under rotation and flip.
        Dots are on an (N+1)×(N+1) grid."""
        # Apply horizontal flip first
        if flip:
            c = N - c  # flip left-right
        
        # Apply rotation (clockwise)
        for _ in range(rot):
            r, c = c, N - r
        
        return r, c
    
    def _compute_symmetry(self, rot, flip):
        """Compute permutation arrays for one symmetry."""
        N = self.N
        n_h = self.n_h
        n_v = self.n_v
        
        # Feature permutation (h_edges + v_edges + boxes + player_flag)
        feature_perm = np.zeros(self.input_size, dtype=np.int32)
        # Policy permutation (h_edges + v_edges)
        policy_perm = np.zeros(self.action_size, dtype=np.int32)
        
        # --- Map horizontal edges ---
        # h_edge(r, c) connects dot(r, c) — dot(r, c+1)
        for r in range(N + 1):
            for c in range(N):
                src_idx = r * N + c  # original h_edge index
                
                # Transform the two dots
                d1 = self._transform_dot(r, c, rot, flip, N)
                d2 = self._transform_dot(r, c + 1, rot, flip, N)
                
                # Determine where this edge goes
                dst_idx = self._edge_between_dots(d1, d2, N)
                
                feature_perm[src_idx] = dst_idx
                policy_perm[src_idx] = dst_idx
        
        # --- Map vertical edges ---
        # v_edge(r, c) connects dot(r, c) — dot(r+1, c)
        for r in range(N):
            for c in range(N + 1):
                src_idx = n_h + r * (N + 1) + c  # original v_edge index (offset by n_h)
                
                d1 = self._transform_dot(r, c, rot, flip, N)
                d2 = self._transform_dot(r + 1, c, rot, flip, N)
                
                dst_idx = self._edge_between_dots(d1, d2, N)
                
                feature_perm[src_idx] = dst_idx
                policy_perm[src_idx] = dst_idx
        
        # --- Map boxes ---
        # box(r, c) has its top-left dot at (r, c)
        box_offset = n_h + n_v
        for r in range(N):
            for c in range(N):
                src_idx = box_offset + r * N + c
                
                # Top-left dot of box
                d_tl = self._transform_dot(r, c, rot, flip, N)
                d_tr = self._transform_dot(r, c + 1, rot, flip, N)
                d_bl = self._transform_dot(r + 1, c, rot, flip, N)
                
                # Find the transformed box — it's the box whose top-left is
                # the minimum (row, col) among the 4 transformed corners
                corners = [d_tl, d_tr, d_bl, self._transform_dot(r + 1, c + 1, rot, flip, N)]
                min_r = min(p[0] for p in corners)
                min_c = min(p[1] for p in corners)
                
                dst_idx = box_offset + min_r * N + min_c
                feature_perm[src_idx] = dst_idx
        
        # Player flag stays in place
        feature_perm[-1] = self.input_size - 1
        
        return feature_perm, policy_perm
    
    def _edge_between_dots(self, d1, d2, N):
        """Given two adjacent dots, return the flat index of the edge between them."""
        r1, c1 = d1
        r2, c2 = d2
        
        # Ensure d1 < d2 lexicographically
        if (r1, c1) > (r2, c2):
            r1, c1, r2, c2 = r2, c2, r1, c1
        
        n_h = (N + 1) * N
        
        if r1 == r2:
            # Horizontal edge: same row, cols differ by 1
            assert abs(c1 - c2) == 1
            c_min = min(c1, c2)
            return r1 * N + c_min  # h_edge index
        else:
            # Vertical edge: same col, rows differ by 1
            assert c1 == c2 and abs(r1 - r2) == 1
            r_min = min(r1, r2)
            return n_h + r_min * (N + 1) + c1  # v_edge index (offset by n_h)
    
    def augment(self, features_np, policy_np, value):
        """
        Generate all 8 symmetric versions of a training sample.
        
        Args:
            features_np: np.array of shape (input_size,)
            policy_np: np.array of shape (action_size,)
            value: float
            
        Returns:
            List of (features_np, policy_np, value) tuples
        """
        augmented = []
        for feat_perm, pol_perm in zip(self.feature_perms, self.policy_perms):
            new_features = features_np[feat_perm]
            new_policy = policy_np[pol_perm]
            augmented.append((new_features, new_policy, value))
        return augmented


# ════════════════════════════════════════════════════════════════════ 
# Exceptions
# ════════════════════════════════════════════════════════════════════

class StopTraining(Exception):
    """Raised when training is complete and workers should exit cleanly."""
    pass


# ════════════════════════════════════════════════════════════════════
# Replay Buffer
# ════════════════════════════════════════════════════════════════════

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, features, policy, value):
        self.buffer.append((features, policy, value))
        
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)


# ════════════════════════════════════════════════════════════════════
# Remote Model (proxy for workers)
# ════════════════════════════════════════════════════════════════════

class RemoteBitModel:
    """
    Proxy model for workers. Sends inference requests to the prediction server
    via multiprocessing queues.
    """
    def __init__(self, N, request_queue, result_queue, worker_id, stop_event, profiler=None):
        self.N = N
        self.n_h_edges = (N + 1) * N
        self.n_v_edges = N * (N + 1)
        self.action_size = self.n_h_edges + self.n_v_edges
        self.input_size = self.n_h_edges + self.n_v_edges + N * N + 1
        
        self.request_queue = request_queue
        self.result_queue = result_queue
        self.worker_id = worker_id
        self.stop_event = stop_event
        self.device = 'cpu'
        self.profiler = profiler
    
    def preprocess(self, state):
        """Convert state to flat feature tensor."""
        N = self.N
        n_h = self.n_h_edges
        n_v = self.n_v_edges
        player = state.current_player
        
        h_features = np.zeros(n_h, dtype=np.float32)
        for i in range(n_h):
            if state.h_edges & (1 << i):
                h_features[i] = 1.0
        
        v_features = np.zeros(n_v, dtype=np.float32)
        for i in range(n_v):
            if state.v_edges & (1 << i):
                v_features[i] = 1.0
        
        box_features = np.zeros(N * N, dtype=np.float32)
        for i in range(N * N):
            bit = 1 << i
            if state.boxes_p1 & bit:
                box_features[i] = 1.0 if player == 1 else -1.0
            elif state.boxes_p2 & bit:
                box_features[i] = 1.0 if player == 2 else -1.0
        
        player_flag = np.array([1.0 if player == 1 else -1.0], dtype=np.float32)
        features = np.concatenate([h_features, v_features, box_features, player_flag])
        return torch.from_numpy(features)
    
    def __call__(self, state_tensor):
        """Send request and wait for response."""
        if self.stop_event.is_set():
            raise StopTraining()
        
        # Squeeze to 1D — the prediction server will batch multiple requests
        # via torch.stack, so we need 1D tensors here to avoid 3D stacking
        state_tensor = state_tensor.squeeze()
        state_tensor.share_memory_()
        if self.profiler: self.profiler.start('queue_put')
        self.request_queue.put((self.worker_id, state_tensor))
        if self.profiler: self.profiler.stop('queue_put')
        
        if self.profiler: self.profiler.start('queue_wait')
        while not self.stop_event.is_set():
            try:
                resp = self.result_queue.get(timeout=0.2)
                if isinstance(resp, Exception):
                    raise resp
                logits, value = resp
                if self.profiler: self.profiler.stop('queue_wait')
                return logits, value
            except queue.Empty:
                continue
        
        if self.profiler: self.profiler.stop('queue_wait')
        raise StopTraining()
    
    def to(self, device):
        return self
    
    def eval(self):
        pass


# ════════════════════════════════════════════════════════════════════
# Worker Processes
# ════════════════════════════════════════════════════════════════════

def prediction_worker(N, model_path, request_queue, result_queues, stop_event, device_str, max_batch_size):
    """Central inference server."""
    try:
        torch.set_num_threads(1)
        device = torch.device(device_str)
        model = AlphaZeroBitNet(N).to(device)
        
        if os.path.exists(model_path):
            try:
                model.load_state_dict(torch.load(model_path, map_location=device))
                print(f"[Prediction] Loaded model from {model_path}")
            except:
                print(f"[Prediction] Failed to load model, starting fresh.")
        
        model.eval()
        
        profiler = TimeProfile()
        batch_count = 0
        last_report_time = time.time()
        
        while not stop_event.is_set():
            batch_reqs = []
            
            try:
                req = request_queue.get(timeout=0.1)
                batch_reqs.append(req)
            except queue.Empty:
                continue
            
            # Adaptive batching — gather up to max_batch_size requests
            start_wait = time.time()
            while len(batch_reqs) < max_batch_size and (time.time() - start_wait) < 0.001:
                try:
                    req = request_queue.get_nowait()
                    batch_reqs.append(req)
                except queue.Empty:
                    break
            
            indices = []
            tensors = []
            
            for req in batch_reqs:
                worker_id, data = req
                
                if worker_id == 'COMMAND':
                    if data == 'RELOAD':
                        print("[Prediction] Reloading model weights...")
                        try:
                            model.load_state_dict(torch.load(model_path, map_location=device))
                            model.eval()
                            print("[Prediction] Model reloaded.")
                        except Exception as e:
                            print(f"[Prediction] Error reloading: {e}")
                    continue
                
                indices.append(worker_id)
                tensors.append(data)
            
            if not tensors:
                continue
            
            input_batch = torch.stack(tensors).to(device)
            
            profiler.start('inference')
            with torch.no_grad():
                logits, values = model(input_batch)
            profiler.stop('inference')
            
            logits = logits.cpu()
            values = values.cpu()
            
            for i, worker_id in enumerate(indices):
                l = logits[i:i+1].clone()
                v = values[i:i+1].clone()
                l.share_memory_()
                v.share_memory_()
                result_queues[worker_id].put((l, v))
                
            batch_count += 1
            if time.time() - last_report_time > 60:  # Report every minute
                print(f"[Prediction] Stats:\n{profiler.report()}")
                profiler.reset()
                last_report_time = time.time()
        
        # Graceful shutdown
        drain_deadline = time.time() + 2.0
        while time.time() < drain_deadline:
            try:
                req = request_queue.get_nowait()
                worker_id, data = req
                if worker_id == 'COMMAND':
                    continue
                result_queues[worker_id].put(StopTraining())
            except queue.Empty:
                break
    
    except Exception as e:
        print(f"[Prediction] CRASHED: {e}")
        traceback.print_exc()


def self_play_worker(worker_id, N, request_queue, result_queue, game_queue, 
                     stop_event, mcts_sims, temp_threshold, temp_explore, temp_exploit):
    """Self-play worker with temperature schedule."""
    try:
        np.random.seed(int(time.time()) + worker_id * 1000)
        random.seed(int(time.time()) + worker_id * 1000)
        
        env = BitBoardEnv(N)
        
        # Build agent with remote model
        profiler = TimeProfile()
        remote_model = RemoteBitModel(N, request_queue, result_queue, worker_id, stop_event, profiler=profiler)
        agent = AlphaZeroBitAgent(env, n_simulations=mcts_sims, device='cpu', profiler=profiler)
        agent.model = remote_model
        
        n_h = remote_model.n_h_edges
        action_size = remote_model.action_size
        
        while not stop_event.is_set():
            env.reset()
            game_history = []
            move_count = 0
            
            while not env.done:
                if stop_event.is_set():
                    break
                
                # Snapshot as raw ints (ultra-fast clone)
                state_snapshot = _NodeState(
                    env.h_edges, env.v_edges, env.boxes_p1, env.boxes_p2,
                    env.current_player, env.done, list(env.score)
                )
                
                # Temperature schedule: explore early, exploit late
                temp = temp_explore if move_count < temp_threshold else temp_exploit
                
                action, policy_dict = agent.act(return_probs=True, temperature=temp,
                                                 add_noise=(move_count < temp_threshold))
                
                game_history.append([state_snapshot, env.current_player, policy_dict])
                env.step(action)
                move_count += 1
            
            if stop_event.is_set():
                break
            
            if move_count > 0 and (move_count % 10 == 0) and worker_id == 0:
                 print(f"[Worker 0] Profile:\n{profiler.report()}")
                 profiler.reset()
            
            # Game over — compute result
            s1, s2 = env.score
            if s1 > s2:
                result = 1
            elif s2 > s1:
                result = -1
            else:
                result = 0
            
            # Process game data
            processed_data = []
            for state, player, policy_dict in game_history:
                # Build policy vector from visit counts
                policy_vec = np.zeros(action_size, dtype=np.float32)
                total_visits = sum(policy_dict.values())
                
                for action_key, visits in policy_dict.items():
                    edge_type, r, c = action_key
                    if edge_type == 0:
                        idx = r * N + c
                    else:
                        idx = n_h + r * (N + 1) + c
                    policy_vec[idx] = visits / total_visits
                
                z = result if player == 1 else -result
                
                # Preprocess state to features
                features = remote_model.preprocess(state).numpy()
                processed_data.append((features, policy_vec, z))
            
            game_queue.put(processed_data)
    
    except StopTraining:
        pass
    except (ConnectionResetError, FileNotFoundError, EOFError, BrokenPipeError):
        if not stop_event.is_set():
            print(f"[Worker {worker_id}] Connection error (unexpected)")
    except Exception as e:
        if not stop_event.is_set():
            print(f"[Worker {worker_id}] CRASHED: {e}")
            traceback.print_exc()


# ════════════════════════════════════════════════════════════════════
# Main Trainer
# ════════════════════════════════════════════════════════════════════

def train_alphazero_bit():
    mp.set_start_method('spawn', force=True)
    
    # ════════════════════════════════════════════════════════════════════
    # Resume Configuration
    # ════════════════════════════════════════════════════════════════════
    # To manually resume from a specific point (e.g., if you lost the checkpoint),
    # set these values. Otherwise, leave them as 0 to start fresh or resume
    # from the last saved checkpoint file.
    MANUAL_START_PHASE = 0   # 0-indexed (e.g. 0 is Bootstrap, 1 is Refinement)
    MANUAL_START_ITER  = 0   # 0-indexed
    # ════════════════════════════════════════════════════════════════════

    models_dir = os.path.join(src_dir, '../models')
    os.makedirs(models_dir, exist_ok=True)
    
    # We maintain two files:
    # 1. Raw model weights (compatible with inference/GUI)
    model_path = os.path.join(models_dir, f"alphazero_bit_{N}x{N}.pth")
    # 2. Full training checkpoint (includes phase/iter/optimizer state)
    checkpoint_path = os.path.join(models_dir, f"alphazero_bit_{N}x{N}_checkpoint.pth")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Main Process: Using device: {device}")
    
    # Symmetry augmenter
    augmenter = SymmetryAugmenter(N)
    print(f"Symmetry augmenter ready: {len(augmenter.feature_perms)} symmetries")
    
    # Training model
    replay_buffer = ReplayBuffer(capacity=BUFFER_CAPACITY)
    train_net = AlphaZeroBitNet(N).to(device)
    
    # Initialize tracking variables
    start_phase_idx = MANUAL_START_PHASE
    start_iter_idx = MANUAL_START_ITER
    total_games = 0
    total_iteration = 0
    
    # Load Checkpoint if exists
    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            train_net.load_state_dict(checkpoint['model_state_dict'])
            
            # Only use checkpoint metadata if manual override is NOT set (is 0)
            if MANUAL_START_PHASE == 0 and MANUAL_START_ITER == 0:
                start_phase_idx = checkpoint.get('phase_idx', 0)
                start_iter_idx = checkpoint.get('iter_idx', 0) + 1  # Resume next iter
                total_games = checkpoint.get('total_games', 0)
                total_iteration = checkpoint.get('total_iteration', 0)
                print(f"Main Process: Resuming from Checkpoint (Phase {start_phase_idx}, Iter {start_iter_idx})")
            else:
                print(f"Main Process: Checkpoint found but MANUAL OVERRIDE active.")
                print(f"              Resuming from Phase {start_phase_idx}, Iter {start_iter_idx}")
                
        except Exception as e:
            print(f"Main Process: Error loading checkpoint: {e}")
            print("Main Process: Starting fresh.")
    elif os.path.exists(model_path):
        # Fallback to raw weights if no checkpoint
        try:
            train_net.load_state_dict(torch.load(model_path, map_location=device))
            print("Main Process: Loaded raw model weights (no training metadata).")
        except:
            print("Main Process: Starting fresh model.")
            
    
    try:
        for phase_idx, phase in enumerate(PHASES):
            # Skip completed phases
            if phase_idx < start_phase_idx:
                continue

            print(f"\n{'#'*60}")
            print(f"  PHASE {phase_idx+1}/{len(PHASES)}: {phase['name']}")
            print(f"  MCTS Sims: {phase['mcts_sims']}, LR: {phase['lr']}, "
                  f"Epochs: {phase['epochs']}")
            print(f"  Temp: {phase['temp_explore']}→{phase['temp_exploit']} "
                  f"(switch at move {phase['temp_threshold']})")
            print(f"{'#'*60}")
            
            # Optimizer with weight decay + cosine LR schedule per phase
            optimizer = optim.AdamW(train_net.parameters(), lr=phase['lr'],
                                    weight_decay=WEIGHT_DECAY)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=phase['iterations'], eta_min=phase['lr'] * 0.1)
                
            # If we are resuming in the middle of a phase, we might want to load optimizer state
            # For now, we just start fresh optimizer for the phase, which is generally fine 
            # as long as we fast-forward the scheduler.
            
            # Determine success criteria
            current_start_iter = start_iter_idx if phase_idx == start_phase_idx else 0

            # Shared queues
            request_queue = mp.Queue(maxsize=1000)
            result_queues = [mp.Queue(maxsize=10) for _ in range(NUM_WORKERS)]
            game_queue = mp.Queue(maxsize=200)
            stop_event = mp.Event()
            
            # Start Prediction Server
            # Ensure model file is up to date before worker starts
            torch.save(train_net.state_dict(), model_path)
            print("Starting Prediction Server...")
            pred_process = mp.Process(
                target=prediction_worker,
                args=(N, model_path, request_queue, result_queues, stop_event, str(device), NUM_WORKERS)
            )
            pred_process.start()
            
            # Start Workers
            print(f"Starting {NUM_WORKERS} Self-Play Workers...")
            workers = []
            for i in range(NUM_WORKERS):
                p = mp.Process(
                    target=self_play_worker,
                    args=(i, N, request_queue, result_queues[i], game_queue, 
                          stop_event, phase['mcts_sims'],
                          phase['temp_threshold'], phase['temp_explore'], 
                          phase['temp_exploit'])
                )
                p.start()
                workers.append(p)
            
            # Phase training loop
            for iter_in_phase in range(phase['iterations']):
                # Fast-forward if resuming
                if iter_in_phase < current_start_iter:
                    scheduler.step()
                    # We approximate total_iteration count
                    if phase_idx == start_phase_idx:
                         pass # total_iteration likely loaded or manually set
                    else:
                         total_iteration += 1
                    continue

                total_iteration += 1
                eps = phase['episodes_per_iter']
                print(f"\n=== Phase {phase['name']} | "
                      f"Iter {iter_in_phase+1}/{phase['iterations']} "
                      f"(Global #{total_iteration}) ===")
                
                start_time = time.time()
                games_collected = 0
                samples_this_iter = 0
                
                print(f"  Collecting {eps} games...")
                
                while games_collected < eps:
                    try:
                        game_data = game_queue.get(timeout=1.0)
                        
                        for step in game_data:
                            features_np, policy_np, value = step
                            
                            # Symmetry augmentation: 8x training data
                            augmented = augmenter.augment(features_np, policy_np, value)
                            for aug_feat, aug_pol, aug_val in augmented:
                                replay_buffer.push(
                                    torch.from_numpy(aug_feat),
                                    aug_pol,
                                    aug_val
                                )
                                samples_this_iter += 1
                        
                        games_collected += 1
                        total_games += 1
                        if games_collected % 20 == 0:
                            print(f"    Collected {games_collected}/{eps} games")
                    except queue.Empty:
                        if not pred_process.is_alive():
                            print("Prediction process died!")
                            stop_event.set()
                            break
                        continue
                
                duration = time.time() - start_time
                print(f"  Collection: {duration:.1f}s ({duration/eps:.2f}s/game) | "
                      f"+{samples_this_iter} samples (8x sym) | "
                      f"Buffer: {len(replay_buffer)} | "
                      f"Total games: {total_games}")
                
                # Training phase
                if len(replay_buffer) < BATCH_SIZE:
                    print("  Not enough samples to train.")
                    continue
                
                print(f"  Training (lr={optimizer.param_groups[0]['lr']:.6f})...")
                train_net.train()
                total_loss = 0.0
                total_p_loss = 0.0
                total_v_loss = 0.0
                
                for _ in range(phase['epochs']):
                    batch = replay_buffer.sample(BATCH_SIZE)
                    features_batch, policies, values = zip(*batch)
                    
                    features_v = torch.stack(features_batch).to(device)
                    policies_v = torch.tensor(np.array(policies), 
                                              dtype=torch.float32).to(device)
                    values_v = torch.tensor(np.array(values), 
                                            dtype=torch.float32).unsqueeze(1).to(device)
                    
                    optimizer.zero_grad()
                    out_logits, out_value = train_net(features_v)
                    
                    # Value loss: MSE
                    v_loss = F.mse_loss(out_value, values_v)
                    
                    # Policy loss: cross-entropy with target distribution
                    log_probs = F.log_softmax(out_logits, dim=1)
                    p_loss = -torch.sum(policies_v * log_probs) / BATCH_SIZE
                    
                    # Combined loss
                    loss = p_loss + VALUE_LOSS_WEIGHT * v_loss
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(train_net.parameters(), GRAD_CLIP)
                    
                    optimizer.step()
                    
                    total_loss += loss.item()
                    total_p_loss += p_loss.item()
                    total_v_loss += v_loss.item()
                
                epochs = phase['epochs']
                print(f"  Loss: {total_loss/epochs:.4f} "
                      f"(policy: {total_p_loss/epochs:.4f}, "
                      f"value: {total_v_loss/epochs:.4f})")
                
                scheduler.step()
                
                # Save Raw Model (for inference)
                torch.save(train_net.state_dict(), model_path)
                
                # Save Full Checkpoint (for resuming)
                torch.save({
                    'phase_idx': phase_idx,
                    'iter_idx': iter_in_phase,
                    'total_games': total_games,
                    'total_iteration': total_iteration,
                    'model_state_dict': train_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, checkpoint_path)
                
                print(f"  Model & Checkpoint saved.")
                request_queue.put(('COMMAND', 'RELOAD'))
            
            # Shutdown workers for this phase
            print(f"\nShutting down Phase {phase['name']} workers...")
            stop_event.set()
            
            pred_process.join(timeout=10)
            if pred_process.is_alive():
                pred_process.terminate()
                pred_process.join(timeout=5)
            
            for p in workers:
                p.join(timeout=5)
                if p.is_alive():
                    p.terminate()
                    p.join(timeout=2)
            
            print(f"Phase {phase['name']} complete. Total games: {total_games}")
    
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        print("Final cleanup...")
        # Save final model
        try:
            torch.save(train_net.state_dict(), model_path)
            # We don't necessarily update checkpoint on crash, rely on last iter save
            print(f"Final model saved to {model_path}")
        except:
            pass
        
        # Kill any leftover processes
        try:
            stop_event.set()
            if pred_process.is_alive():
                pred_process.terminate()
            for p in workers:
                if p.is_alive():
                    p.terminate()
        except:
            pass
        
        print(f"Training complete. {total_games} total games played.")


if __name__ == '__main__':
    train_alphazero_bit()
