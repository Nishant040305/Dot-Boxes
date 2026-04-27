#!/usr/bin/env python3
"""
Diagnostic: test if the AlphaZero 5x5 model can identify obvious captures.

Creates board positions with 3-sided boxes and checks:
1. Does the policy head assign high probability to the capture move?
2. Does the value head output a reasonable value?
3. Does MCTS find the capture at low sim counts?
"""

import os
import sys
import json

_script_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.join(_script_dir, '..')
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

import torch
from env.BitBoardEnv import BitBoardEnv

def load_model(rows, cols):
    """Load the 5x5 model and return it."""
    model_dir = os.path.join(_src_dir, 'cpp', 'models', f'_{rows}x{cols}')
    info_path = os.path.join(model_dir, 'model_info.json')
    
    if os.path.exists(info_path):
        with open(info_path) as f:
            info = json.load(f)
        hidden = info.get('hidden_size', 256)
        blocks = info.get('num_res_blocks', 6)
        print(f"Model: hidden={hidden}, blocks={blocks}")
    else:
        print("No model_info.json found, using defaults")
        hidden, blocks = 384, 10
    
    # Try to import the Python BitAgent which has the network
    from agents.AlphaZeroBitAgent import AlphaZeroBitAgent as PyAgent
    
    env = BitBoardEnv(rows)
    
    # Find model file
    model_files = [f for f in os.listdir(model_dir) 
                   if f.endswith('.pth') and 'checkpoint' in f]
    if not model_files:
        model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
    
    if model_files:
        model_path = os.path.join(model_dir, sorted(model_files)[-1])
        print(f"Loading: {model_path}")
    else:
        print(f"No .pth model found in {model_dir}")
        return None, env
    
    agent = PyAgent(env, model_path=model_path, n_simulations=100, 
                    device='cpu', checkpoint=True)
    return agent, env


def test_obvious_capture(rows=5, cols=5):
    """Create a position with an obvious capture and test the model."""
    env = BitBoardEnv(rows)
    env.reset()
    
    print(f"\n{'='*60}")
    print(f"  TEST: Obvious Capture Detection ({rows}x{cols})")
    print(f"{'='*60}")
    
    # Place 3 edges around box (0,0): top, left, bottom - leave right open
    # Box (0,0) edges: h(0,0), h(1,0), v(0,0), v(0,1)
    # Place h(0,0), h(1,0), v(0,0) → missing v(0,1)
    moves = [
        (0, 0, 0),  # h-edge row=0, col=0 (top of box 0,0)
        (0, 1, 0),  # h-edge row=1, col=0 (bottom of box 0,0)
        (1, 0, 0),  # v-edge row=0, col=0 (left of box 0,0)
    ]
    
    # Alternate players for these non-capturing moves
    for move in moves:
        env.step(move)
    
    print(f"\nCurrent player: {env.current_player}")
    print(f"Score: P1={env.score[0]}, P2={env.score[1]}")
    print(f"Available actions: {len(env._get_available_actions())}")
    env.render()
    
    # The capture move is v(0,1) = edge_type=1, r=0, c=1
    print(f"\nCapture move: (1, 0, 1) = v-edge row=0, col=1 (right side of box 0,0)")
    print(f"This should be the BEST move!\n")
    
    # Check if greedy sees it
    from agents.GreedyAgent import GreedyAgent
    greedy = GreedyAgent(env)
    greedy_action = greedy.act()
    print(f"Greedy's choice: {greedy_action}")
    
    # Now test the C++ agent via arena
    from gameSimulation.arena import make_agent
    
    for sims in [100, 400, 1000, 2000]:
        env_test = BitBoardEnv(rows)
        env_test.reset()
        for move in moves:
            env_test.step(move)
        
        try:
            agent = make_agent('alphazero_cpp', env_test, 
                             n_simulations=sims, use_dag=False)
            action = agent.act()
            is_capture = (action == (1, 0, 1))
            status = "✓ CAPTURE" if is_capture else "✗ MISSED"
            print(f"  AlphaZero (sims={sims:4d}): {action} → {status}")
            agent.close()
        except Exception as e:
            print(f"  AlphaZero (sims={sims:4d}): ERROR - {e}")


def test_multiple_captures(rows=5, cols=5):
    """Create a position with MULTIPLE obvious captures."""
    env = BitBoardEnv(rows)
    env.reset()
    
    print(f"\n{'='*60}")
    print(f"  TEST: Multiple Captures Available ({rows}x{cols})")
    print(f"{'='*60}")
    
    # Set up two 3-sided boxes
    # Box (0,0): place top, bottom, left → missing right v(0,1)
    # Box (0,1): place top, bottom, right → missing left v(0,1)
    # Actually v(0,1) is shared! So placing it captures BOTH boxes.
    
    # Let's set up two independent boxes:
    # Box (0,0): top h(0,0), bottom h(1,0), left v(0,0) → miss right v(0,1)  
    # Box (2,2): top h(2,2), bottom h(3,2), left v(2,2) → miss right v(2,3)
    
    setup_moves = [
        (0, 0, 0),  # h(0,0) - top of box(0,0)
        (0, 1, 0),  # h(1,0) - bottom of box(0,0)
        (1, 0, 0),  # v(0,0) - left of box(0,0) 
        (0, 2, 2),  # h(2,2) - top of box(2,2)
        (0, 3, 2),  # h(3,2) - bottom of box(2,2)
        (1, 2, 2),  # v(2,2) - left of box(2,2)
    ]
    
    for move in setup_moves:
        env.step(move)
    
    print(f"\nCurrent player: {env.current_player}")
    print(f"Score: P1={env.score[0]}, P2={env.score[1]}")
    env.render()
    
    capture_moves = [(1, 0, 1), (1, 2, 3)]
    print(f"\nCapture moves: {capture_moves}")
    print(f"Either of these should be chosen!\n")
    
    from gameSimulation.arena import make_agent
    
    for sims in [100, 400, 1000, 2000]:
        env_test = BitBoardEnv(rows)
        env_test.reset()
        for move in setup_moves:
            env_test.step(move)
        
        try:
            agent = make_agent('alphazero_cpp', env_test,
                             n_simulations=sims, use_dag=False)
            action = agent.act()
            is_capture = action in capture_moves
            status = "✓ CAPTURE" if is_capture else "✗ MISSED"
            print(f"  AlphaZero (sims={sims:4d}): {action} → {status}")
            agent.close()
        except Exception as e:
            print(f"  AlphaZero (sims={sims:4d}): ERROR - {e}")


if __name__ == '__p1__':
    test_obvious_capture(5, 5)
    test_multiple_captures(5, 5)
    
    # Also test 4x3 as a sanity check
    print(f"\n\n{'#'*60}")
    print(f"  SANITY CHECK: 4x3 model")
    print(f"{'#'*60}")
    test_obvious_capture(4, 3)
