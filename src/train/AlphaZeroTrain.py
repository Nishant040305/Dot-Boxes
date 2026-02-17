
import sys
import os
import random
import time
import queue
import traceback
import numpy as np
import torch
import torch.optim as optim
import torch.multiprocessing as mp
from collections import deque

# Add src to path
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(script_dir)
sys.path.append(src_dir)

from env.BoardEnv import BaseBoardEnv
from agents.AlphaZeroAgent import AlphaZeroAgent
from models.Net import AlphaZeroNet

# --- Configuration ---
N = 3
NUM_WORKERS = 16  # Number of self-play workers
MCTS_SIMS = 400
EPISODES_PER_ITER = 100
BATCH_SIZE = 128
EPOCHS = 10
LR = 0.001
BUFFER_CAPACITY = 50000
NUM_ITERS = 10

# --- Helper Classes ---

class StopTraining(Exception):
    """Raised when training is complete and workers should exit cleanly."""
    pass

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, policy, value):
        self.buffer.append((state, policy, value))
        
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class RemoteModel:
    """
    A wrapper that acts like the model but sends requests to a prediction queue.
    Used by workers to get predictions from the central GPU/inference process.
    """
    def __init__(self, N, request_queue, result_queue, worker_id, stop_event):
        self.dummy_net = AlphaZeroNet(N) # Lightweight, for preprocess only
        self.request_queue = request_queue
        self.result_queue = result_queue
        self.worker_id = worker_id
        self.stop_event = stop_event
        self.device = 'cpu' # Workers run on CPU
        
    def preprocess(self, state):
        # preprocess is a static-like method in AlphaZeroNet
        return self.dummy_net.preprocess(state)
        
    def __call__(self, state_tensor):
        # state_tensor shape: (1, 3, N+1, N+1)
        if self.stop_event.is_set():
            raise StopTraining()
        
        # Send request
        state_tensor.share_memory_()
        self.request_queue.put((self.worker_id, state_tensor))
        
        # Wait for valid response
        while not self.stop_event.is_set():
            try:
                resp = self.result_queue.get(timeout=0.2)
                if isinstance(resp, Exception):
                    raise resp
                logits, value = resp
                return logits, value
            except queue.Empty:
                continue
        
        # If we reach here, stop_event is set — clean exit
        raise StopTraining()
        
    def to(self, device):
        # Ignore device movement, we stay on CPU/Queue
        return self
        
    def eval(self):
        pass

# --- Worker Processes ---

def prediction_worker(N, model_path, request_queue, result_queues, stop_event, device_str):
    """
    Central Inference Server.
    Batches requests from workers and runs model on GPU.
    """
    try:
        # Set threads to 1 to avoid contention with workers and overhead for small batches
        torch.set_num_threads(1)
        device = torch.device(device_str)
        model = AlphaZeroNet(N).to(device)
        
        if os.path.exists(model_path):
            try:
                model.load_state_dict(torch.load(model_path, map_location=device))
                print(f"[Prediction] Loaded model from {model_path}")
            except:
                print(f"[Prediction] Failed to load model, starting fresh.")
        
        model.eval()
        
        while not stop_event.is_set():
            # Collect batch logic
            batch_reqs = []
            
            # Simple adaptive batching
            try:
                # Blocking get for first item
                req = request_queue.get(timeout=0.1)
                batch_reqs.append(req)
            except queue.Empty:
                continue
                
            # Try to get more items to fill batch (without waiting too long)
            # This is the "Batch" part of Batch MCTS
            start_wait = time.time()
            while len(batch_reqs) < 16 and (time.time() - start_wait) < 0.01:
                try:
                    req = request_queue.get_nowait()
                    batch_reqs.append(req)
                except queue.Empty:
                    break
            
            # Process batch
            indices = []
            tensors = []
            reload_cmd = False
            
            for req in batch_reqs:
                worker_id, data = req
                
                # Check for special commands
                if worker_id == 'COMMAND':
                    if data == 'RELOAD':
                        print("[Prediction] Reloading model weights...")
                        try:
                            model.load_state_dict(torch.load(model_path, map_location=device))
                            model.eval()
                            print("[Prediction] Model reloaded.")
                        except Exception as e:
                            print(f"[Prediction] Error reloading model: {e}")
                        reload_cmd = True
                    continue
                
                indices.append(worker_id)
                tensors.append(data)
            
            if not tensors:
                continue
                
            if reload_cmd:
                # If we reloaded, the pending requests might be against old model, 
                # but currently we just process them with new model.
                pass
                
            # Inference
            # Stack: tensors are (1, C, H, W), so stack -> (B, C, H, W)
            # We need to squeeze the dim 0 from each tensor before stack? 
            # AlphaZeroAgent sends .unsqueeze(0).
            # So standard torch.cat is enough
            input_batch = torch.cat(tensors).to(device)
            
            with torch.no_grad():
                logits, values = model(input_batch)
                
            logits = logits.cpu()
            values = values.cpu()
            
            # Distribute results
            for i, worker_id in enumerate(indices):
                # result_queues is a list of queues
                # We return separate tensors (slice them)
                l = logits[i:i+1].clone() # Keep (1, ...) shape, clone to detach from batch storage
                v = values[i:i+1].clone()
                l.share_memory_()
                v.share_memory_()
                result_queues[worker_id].put((l, v))
        
        # --- Graceful shutdown: drain any remaining requests ---
        # Workers may still have pending requests in the queue.
        # Respond with StopTraining so they unblock cleanly.
        _drain_deadline = time.time() + 2.0  # spend at most 2 seconds draining
        while time.time() < _drain_deadline:
            try:
                req = request_queue.get_nowait()
                worker_id, data = req
                if worker_id == 'COMMAND':
                    continue
                # Send an exception so the worker unblocks
                result_queues[worker_id].put(StopTraining())
            except queue.Empty:
                break
                
    except Exception as e:
        print(f"[Prediction] CRASHED: {e}")
        traceback.print_exc()

def self_play_worker(worker_id, N, request_queue, result_queue, game_queue, stop_event, mcts_sims):
    """
    Self-Play Worker.
    Plays games continuously and sends results to game_queue.
    """
    try:
        # Pseudo-random seed for each worker
        np.random.seed(int(time.time()) + worker_id * 1000)
        random.seed(int(time.time()) + worker_id * 1000)
        
        env = BaseBoardEnv(N)
        
        # Initialize Agent with RemoteModel
        remote_model = RemoteModel(N, request_queue, result_queue, worker_id, stop_event)
        agent = AlphaZeroAgent(env, n_simulations=mcts_sims, device='cpu')
        agent.model = remote_model
        
        while not stop_event.is_set():
            env.reset()
            game_history = []
            
            while not env.done:
                # If stop requested inside game, break
                if stop_event.is_set(): break
                
                state_snapshot = agent._clone_state(env)
                
                # Temp schedule
                temp = 1.0 if len(game_history) < 10 else 0.0
                
                action, policy_dict = agent.act(return_probs=True, temperature=temp)
                
                game_history.append([state_snapshot, env.current_player, policy_dict])
                env.step(action)
            
            if stop_event.is_set(): break

            # Game Over
            s1, s2 = env.score
            if s1 > s2: result = 1
            elif s2 > s1: result = -1
            else: result = 0
            
            # Process Game History
            processed_data = []
            for state, player, policy_dict in game_history:
                policy_vec = np.zeros(agent.model.dummy_net.action_size if hasattr(agent.model.dummy_net, 'action_size') else (N+1)*N + N*(N+1), dtype=np.float32)
                # Note: AlphaZeroNet does not expose action_size directly usually, 
                # but we know it for N.
                # Let's fix action_size logic.
                # AlphaZeroTrainer.py used net.action_size? 
                # Checking Net.py: p_fc output size is (N+1)*N + N*(N+1).
                # But Net does not have self.action_size attribute visible in code.
                # AlphaZeroTrainer line 100 used 'net.action_size'.
                # Wait, did I miss that in Net.py?
                # The file I read didn't show self.action_size = ...
                # Let's calculate it manually to be safe.
                action_size = (N+1)*N + N*(N+1)
                
                policy_vec = np.zeros(action_size, dtype=np.float32)
                total_visits = sum(policy_dict.values())
                
                for action, visits in policy_dict.items():
                    idx = agent._action_to_index(action)
                    policy_vec[idx] = visits / total_visits
                    
                z = result if player == 1 else -result
                
                # Optimization: Send numpy array instead of tensor to avoid 
                # "Too many open files" error with shared memory tensors.
                state_numpy = remote_model.preprocess(state).numpy()
                processed_data.append((state_numpy, policy_vec, z))
            
            # Send game data
            game_queue.put(processed_data)
    
    except StopTraining:
        # Clean exit — training is complete
        pass
    except (ConnectionResetError, FileNotFoundError, EOFError, BrokenPipeError):
        # Expected during shutdown — the prediction server closed its end
        if not stop_event.is_set():
            print(f"[Worker {worker_id}] Connection error (unexpected, stop not set)")
    except Exception as e:
        # Only print truly unexpected crashes
        if not stop_event.is_set():
            print(f"[Worker {worker_id}] CRASHED: {e}")
            traceback.print_exc()

# --- Main Trainer ---

def train_alpha_zero_mp():
    mp.set_start_method('spawn', force=True)
    
    models_dir = os.path.join(src_dir, '../models')
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, f"alphazero_{N}x{N}.pth")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Main Process: Using device: {device}")
    
    # Shared Queues
    # request_queue: (worker_id, data)
    request_queue = mp.Queue(maxsize=1000)
    # result_queues: list of queues, one per worker
    result_queues = [mp.Queue(maxsize=10) for _ in range(NUM_WORKERS)]
    # game_queue: buffer for finished games
    game_queue = mp.Queue(maxsize=100)
    
    stop_event = mp.Event()
    
    # Start Prediction Server
    print("Starting Prediction Server...")
    pred_process = mp.Process(
        target=prediction_worker,
        args=(N, model_path, request_queue, result_queues, stop_event, str(device))
    )
    pred_process.start()
    
    # Start Self-Play Workers
    print(f"Starting {NUM_WORKERS} Self-Play Workers...")
    workers = []
    for i in range(NUM_WORKERS):
        p = mp.Process(
            target=self_play_worker,
            args=(i, N, request_queue, result_queues[i], game_queue, stop_event, MCTS_SIMS)
        )
        p.start()
        workers.append(p)
        
    # Replay Buffer
    replay_buffer = ReplayBuffer(capacity=BUFFER_CAPACITY)
    
    # Load Model for Training (Main process also needs a model instance to train)
    train_net = AlphaZeroNet(N).to(device)
    optimizer = optim.Adam(train_net.parameters(), lr=LR)
    
    if os.path.exists(model_path):
        try:
            train_net.load_state_dict(torch.load(model_path, map_location=device))
            print("Main Process: Loaded existing model.")
        except:
            print("Main Process: Starting fresh model.")
            
    # Main Loop
    try:
        for iteration in range(NUM_ITERS):
            print(f"\n=== Iteration {iteration+1}/{NUM_ITERS} ===")
            start_time = time.time()
            
            # Collection Phase
            games_collected = 0
            print(f"  Collecting {EPISODES_PER_ITER} games...")
            
            while games_collected < EPISODES_PER_ITER:
                try:
                    game_data = game_queue.get(timeout=1.0) # List of steps
                    for step in game_data:
                        state_np, policy, value = step
                        state_tensor = torch.from_numpy(state_np)
                        replay_buffer.push(state_tensor, policy, value)
                    
                    games_collected += 1
                    if games_collected % 10 == 0:
                        print(f"    Collected {games_collected}/{EPISODES_PER_ITER} games")
                except queue.Empty:
                    # Check if processes are alive
                    if not pred_process.is_alive():
                        print("Prediction process died!")
                        stop_event.set()
                        break
                    continue
            
            duration = time.time() - start_time
            print(f"  Collection Time: {duration:.2f}s ({duration/EPISODES_PER_ITER:.2f}s/game)")
            print(f"  Replay Buffer Size: {len(replay_buffer)}")
            
            # Training Phase
            if len(replay_buffer) < BATCH_SIZE:
                print("  Not enough samples to train.")
                continue
            
            print(f"  Training Neural Net...")
            train_net.train()
            total_loss = 0
            
            for _ in range(EPOCHS):
                batch = replay_buffer.sample(BATCH_SIZE)
                states, policies, values = zip(*batch)
                
                states_v = torch.stack(states).to(device)
                policies_v = torch.tensor(np.array(policies), dtype=torch.float32).to(device)
                values_v = torch.tensor(np.array(values), dtype=torch.float32).unsqueeze(1).to(device)
                
                optimizer.zero_grad()
                out_logits, out_value = train_net(states_v)
                
                v_loss = torch.nn.functional.mse_loss(out_value, values_v)
                log_probs = torch.nn.functional.log_softmax(out_logits, dim=1)
                p_loss = -torch.sum(policies_v * log_probs) / BATCH_SIZE
                
                loss = v_loss + p_loss
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
            print(f"  Avg Loss: {total_loss / EPOCHS:.4f}")
            
            # Save and Reload
            torch.save(train_net.state_dict(), model_path)
            print(f"  Model saved to {model_path}")
            
            # Trigger Reload in Prediction Server
            request_queue.put(('COMMAND', 'RELOAD'))
            
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        print("Shutting down workers...")
        stop_event.set()
        
        # Give prediction server time to drain pending requests
        pred_process.join(timeout=10)
        if pred_process.is_alive():
            print("  Prediction server didn't stop, terminating...")
            pred_process.terminate()
            pred_process.join(timeout=5)
        
        # Wait for workers with timeout
        for i, p in enumerate(workers):
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
                p.join(timeout=2)
        
        print("Done.")

if __name__ == '__main__':
    train_alpha_zero_mp()
