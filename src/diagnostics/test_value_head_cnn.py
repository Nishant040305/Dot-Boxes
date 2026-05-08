#!/usr/bin/env python3
"""
Diagnostics for the *CNN* (grid-based) 5x5 AlphaZero model value head.

Checks:
1) Forced-win / forced-loss sanity: value should be near +1 / -1.
2) Perspective flip: flipping only current_player should roughly negate value.
3) Relabel symmetry: swapping player labels + current_player should preserve value.
"""

import os
import sys
import argparse

_script_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.join(_script_dir, "..")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

import torch

from env.BoardEnv import BaseBoardEnv
from models.Net import AlphaZeroNet


def _load_cnn_model(model_path: str, n: int, device: str):
    """
    Loads either:
    - a regular PyTorch checkpoint/state_dict (preferred), or
    - a TorchScript archive (via torch.jit.load).

    Notes:
    - In PyTorch 2.6+, torch.load defaults weights_only=True which is incompatible
      with TorchScript archives; those must be loaded with torch.jit.load.
    """
    net = AlphaZeroNet(n).to(device)
    try:
        ckpt = torch.load(model_path, map_location=device, weights_only=True)
    except RuntimeError as e:
        msg = str(e)
        if "weights_only=True" in msg and "TorchScript archives" in msg:
            scripted = torch.jit.load(model_path, map_location=device)
            scripted.eval()
            return scripted.to(device) if hasattr(scripted, "to") else scripted
        raise

    # Support either raw state_dict or a checkpoint dict with model_state_dict.
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt

    net.load_state_dict(state_dict)
    net.eval()
    return net


def _infer_value(net, env: BaseBoardEnv, device: str) -> float:
    if not hasattr(net, "preprocess"):
        raise TypeError(
            "Loaded model does not have `preprocess(env)`; "
            "please provide a non-TorchScript checkpoint/state_dict, "
            "or export a TorchScript model that includes `preprocess`."
        )
    x = net.preprocess(env).unsqueeze(0).to(device)
    with torch.no_grad():
        _, v = net(x)
    return float(v.squeeze().item())


def _set_boxes(env: BaseBoardEnv, p1_boxes: set[tuple[int, int]], p2_boxes: set[tuple[int, int]]):
    # Clear
    for r in range(env.rows):
        for c in range(env.cols):
            env.boxes[r][c] = 0

    # Assign
    for r, c in p1_boxes:
        env.boxes[r][c] = 1
    for r, c in p2_boxes:
        env.boxes[r][c] = 2

    env.score = [len(p1_boxes), len(p2_boxes)]
    env.done = (env.score[0] + env.score[1]) == (env.rows * env.cols)


def _clone_env(env: BaseBoardEnv) -> BaseBoardEnv:
    # BaseBoardEnv.clone intentionally avoids copying histories; good for diagnostics.
    cloned = env.clone()
    return cloned


def _swap_player_labels(env: BaseBoardEnv) -> BaseBoardEnv:
    """Swap players (1<->2) in boxes + score, but keep edges unchanged."""
    out = _clone_env(env)
    for r in range(out.rows):
        for c in range(out.cols):
            if out.boxes[r][c] == 1:
                out.boxes[r][c] = 2
            elif out.boxes[r][c] == 2:
                out.boxes[r][c] = 1
    out.score = [env.score[1], env.score[0]]
    out.current_player = 3 - env.current_player
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to CNN model .pt/.pth file")
    parser.add_argument("--n", type=int, default=5, help="Board size N (NxN)")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    net = _load_cnn_model(args.model, args.n, args.device)

    # --- Test 1: forced win / forced loss ---
    env_win = BaseBoardEnv(args.n)
    env_win.reset()
    env_win.current_player = 1

    # Give player 1 an uncatchable lead: 20-0 on 5x5 (25 boxes total).
    p1_boxes = {(r, c) for r in range(args.n) for c in range(args.n) if (r * args.n + c) < 20}
    p2_boxes = set()
    _set_boxes(env_win, p1_boxes, p2_boxes)

    v_win = _infer_value(net, env_win, args.device)

    env_loss = BaseBoardEnv(args.n)
    env_loss.reset()
    env_loss.current_player = 1
    _set_boxes(env_loss, p1_boxes=set(), p2_boxes=p1_boxes)
    v_loss = _infer_value(net, env_loss, args.device)

    # --- Test 2: perspective flip (only current_player) ---
    env_mid = BaseBoardEnv(args.n)
    env_mid.reset()
    env_mid.current_player = 1

    mid_p1 = {(0, 0), (0, 1), (1, 0), (2, 2)}
    mid_p2 = {(0, 2), (1, 2), (3, 3)}
    _set_boxes(env_mid, mid_p1, mid_p2)

    v_p1_turn = _infer_value(net, env_mid, args.device)
    env_mid_flip = _clone_env(env_mid)
    env_mid_flip.current_player = 2
    v_p2_turn_same_labels = _infer_value(net, env_mid_flip, args.device)

    # --- Test 3: relabel symmetry (swap player labels + current_player) ---
    env_mid_relabel = _swap_player_labels(env_mid)
    v_relabel = _infer_value(net, env_mid_relabel, args.device)

    print("=== CNN Value Head Diagnostics ===")
    print(f"Model: {args.model}")
    print(f"N: {args.n}  device: {args.device}")
    print()
    print("[Forced outcomes]")
    print(f"  forced win (P1 lead 20-0, P1 to move): value={v_win:+.4f} (expect near +1)")
    print(f"  forced loss (P2 lead 20-0, P1 to move): value={v_loss:+.4f} (expect near -1)")
    print()
    print("[Perspective flip]")
    print(f"  same boxes, P1 to move: value={v_p1_turn:+.4f}")
    print(f"  same boxes, P2 to move: value={v_p2_turn_same_labels:+.4f} (expect roughly {-v_p1_turn:+.4f})")
    print()
    print("[Relabel symmetry]")
    print(f"  swapped labels + turn: value={v_relabel:+.4f} (expect roughly {v_p1_turn:+.4f})")


if __name__ == "__main__":
    main()
