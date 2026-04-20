# Experiments & Reporting TODO

All experiments required to complete the research paper. Each item includes
the exact task, expected output (plot/table/metric), and where it goes in
the paper.

---

## Legend

- `[ ]` Not started
- `[/]` In progress  
- `[x]` Completed

---

## 1. Training Runs (Must Complete First)

### 1.1 Train Models

- [ ] **2×2 board** — full training with config `_2x2.cpp`
  - Expected: ~30 min on CPU
  - Save model to `models/_2x2/alphazero_2x2.pt`
  - Log training loss per iteration to a CSV or stdout capture

- [ ] **3×3 board** — full training with config `_3x3.cpp`
  - Expected: ~2-4 hours on CPU
  - Save model to `models/_3x3/alphazero_3x3.pt`
  - Log training loss per iteration

- [ ] **5×5 board (monolithic)** — full training with config `_5x5.cpp`
  - Expected: ~4-6 hours on CPU
  - Save model to `models/_5x5/alphazero_5x5.pt`
  - Log training loss per iteration

- [ ] **5×5 board (PatchNet)** — full training with config `_5x5_patch.cpp`
  - **Prerequisite**: 3×3 model must be trained first (used as frozen local model)
  - Expected: ~2-3 hours on CPU  
  - Save model to `models/_5x5_patch/alphazero_5x5_patch.pt`
  - Log training loss per iteration

### 1.2 Capture Training Logs

For each training run, capture:
- [ ] Iteration number, policy loss, value loss, combined loss (every iteration)
- [ ] Number of games played per iteration
- [ ] Replay buffer size over time
- [ ] Phase transition points (iteration numbers)
- [ ] Total wall-clock training time
- [ ] Hardware info: CPU model, RAM, GPU (if used)

**Format**: Save as CSV files in `docs/training_logs/`:
```
docs/training_logs/
  train_2x2.csv      # columns: iter, phase, p_loss, v_loss, total_loss, buffer_size, games
  train_3x3.csv
  train_5x5.csv
  train_5x5_patch.csv
```

---

## 2. Plots to Generate

### 2.1 Training Loss Curves

→ **Paper location**: Section 7.1 (Training Dynamics)

- [ ] **Plot A**: Policy loss vs iteration for 3×3 board
  - X-axis: iteration number
  - Y-axis: policy cross-entropy loss
  - Mark phase transitions with vertical dashed lines
  
- [ ] **Plot B**: Value loss vs iteration for 3×3 board
  - Same format as Plot A

- [ ] **Plot C**: Combined loss vs iteration for 3×3 board

- [ ] **Plot D**: Same as A-C but for 5×5 board (both monolithic and PatchNet on same plot)
  - Two lines per subplot: monolithic (solid) vs PatchNet (dashed)
  - **This is the key plot for the PatchNet contribution**

**Instructions for generating these plots**:
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load training log
df = pd.read_csv('docs/training_logs/train_3x3.csv')

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Policy loss
axes[0].plot(df['iter'], df['p_loss'])
axes[0].set_xlabel('Iteration')
axes[0].set_ylabel('Policy Loss')
axes[0].set_title('Policy Loss (3×3)')

# Value loss
axes[1].plot(df['iter'], df['v_loss'])
axes[1].set_xlabel('Iteration')
axes[1].set_ylabel('Value Loss')
axes[1].set_title('Value Loss (3×3)')

# Combined
axes[2].plot(df['iter'], df['total_loss'])
axes[2].set_xlabel('Iteration')
axes[2].set_ylabel('Total Loss')
axes[2].set_title('Combined Loss (3×3)')

# Add phase transition lines
for phase_start in phase_transitions:
    for ax in axes:
        ax.axvline(x=phase_start, color='gray', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('paper/figures/training_curves_3x3.pdf', dpi=300, bbox_inches='tight')
```

### 2.2 Win-Rate Over Training

→ **Paper location**: Section 7.1 (Training Dynamics)

- [ ] **Plot E**: Win-rate vs Random over training (3×3)
  - Evaluate every 10-20 iterations: play 50 games vs Random
  - X-axis: iteration, Y-axis: win-rate %

- [ ] **Plot F**: Win-rate vs Greedy over training (3×3)

- [ ] **Plot G**: Win-rate vs Random over training (5×5, mono vs patch)

**How to evaluate during training**: Add evaluation callbacks or run
post-hoc evaluation using saved checkpoints:
```bash
# For each checkpoint:
./dots_boxes --mode=eval --model=models/_3x3/checkpoint_iter10.pt \
  --opponent=greedy --games=50 --rows=3 --cols=3
```

### 2.3 Value Target Comparison

→ **Paper location**: Section 4.5.2 (Value Target Design) — Figure

- [ ] **Plot H**: Value target functions across score differences
  - X-axis: Δ = score difference (-5 to +5)
  - Y-axis: z (value target)
  - 5 lines: WinLoss, ScoreDiff, ScoreDiffSqrt, ScoreDiffTanh, ScoreDiffScaled
  - Show for B=9 (3×3) and B=25 (5×5) side by side
  - **Key visual**: show how ScoreDiffTanh crushes signal on 5×5

```python
import numpy as np
import matplotlib.pyplot as plt

deltas = np.arange(-5, 6)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, B, title in zip(axes, [9, 25], ['3×3 (B=9)', '5×5 (B=25)']):
    # WinLoss
    wl = np.sign(deltas)
    ax.plot(deltas, wl, 'o-', label='WinLoss')
    
    # ScoreDiff
    sd = deltas / B
    ax.plot(deltas, sd, 's-', label='ScoreDiff')
    
    # ScoreDiffSqrt
    sds = np.sign(deltas) * np.sqrt(np.abs(deltas) / B)
    ax.plot(deltas, sds, '^-', label='ScoreDiffSqrt')
    
    # ScoreDiffTanh
    sdt = np.tanh(2 * deltas / B)
    ax.plot(deltas, sdt, 'D-', label='ScoreDiffTanh')
    
    # ScoreDiffScaled (OUR TARGET)
    ssc = np.tanh(0.5 * deltas)
    ax.plot(deltas, ssc, 'P-', linewidth=2, label='ScoreDiffScaled (ours)')
    
    ax.set_xlabel('Score Difference (Δ)')
    ax.set_ylabel('Value Target (z)')
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1.1, 1.1)

plt.tight_layout()
plt.savefig('paper/figures/value_targets.pdf', dpi=300, bbox_inches='tight')
```

### 2.4 MCTS Simulation Budget vs Win-Rate

→ **Paper location**: Section 8 (Ablation Studies) or Section 7

- [ ] **Plot I**: Win-rate vs sims/move (3×3 vs Greedy)
  - Test at: 50, 100, 200, 400, 600, 800, 1200 sims
  - Shows diminishing returns / where investment plateaus

---

## 3. Tables to Fill

### 3.1 Win-Rate Table (Table 3 in paper)

→ **Paper location**: Section 7.2

Run these evaluation games:

| Entry | Command | Status |
|-------|---------|--------|
| 2×2 vs Random (200 sims) | `./dots_boxes --eval --opponent=random --sims=200 --rows=2 --cols=2 --games=200` | [ ] |
| 2×2 vs Greedy (200 sims) | `./dots_boxes --eval --opponent=greedy --sims=200 --rows=2 --cols=2 --games=200` | [ ] |
| 3×3 vs Random (400 sims) | `./dots_boxes --eval --opponent=random --sims=400 --rows=3 --cols=3 --games=200` | [ ] |
| 3×3 vs Greedy (400 sims) | `./dots_boxes --eval --opponent=greedy --sims=400 --rows=3 --cols=3 --games=200` | [ ] |
| 3×3 vs Greedy (800 sims) | `./dots_boxes --eval --opponent=greedy --sims=800 --rows=3 --cols=3 --games=200` | [ ] |
| 3×3 vs AB(d=3) (800 sims) | `./dots_boxes --eval --opponent=alphabeta --depth=3 --sims=800 --rows=3 --cols=3 --games=200` | [ ] |
| 5×5 vs Random (600 sims) | `./dots_boxes --eval --opponent=random --sims=600 --rows=5 --cols=5 --games=200` | [ ] |
| 5×5 vs Greedy (600 sims) | `./dots_boxes --eval --opponent=greedy --sims=600 --rows=5 --cols=5 --games=200` | [ ] |
| 5×5 vs AB(d=3) (1000 sims) | `./dots_boxes --eval --opponent=alphabeta --depth=3 --sims=1000 --rows=5 --cols=5 --games=100` | [ ] |

For each evaluation, record:
- Win count, Draw count, Loss count → Win-rate %
- Average score difference (Δ̄)
- Average time per move (ms)

### 3.2 PatchNet Comparison Table (Table 6 in paper)

→ **Paper location**: Section 7.3

- [ ] Count trainable parameters for monolithic 5×5 model  
  Command: Add param counting code or compute from architecture:
  ```
  Monolithic: input_fc + 10 res_blocks + policy_head + value_head
  PatchNet:   global_fc + 6 res_blocks + fuse + policy_head + value_head (local is frozen)
  ```
- [ ] Record total training wall-clock time for monolithic 5×5
- [ ] Record total training wall-clock time for PatchNet 5×5
- [ ] Win-rate of each against Greedy (600 sims, 200 games)
- [ ] Win-rate of each against AB(d=3) (1000 sims, 100 games)

### 3.3 Scalability Table (Table 5 in paper)

→ **Paper location**: Section 7.3

- [ ] Measure MCTS time/move for 5×5 with 600 sims (average over 50 moves)
- [ ] Measure MCTS time/move for 5×5 with 1000 sims

### 3.4 Ablation Table (Table 7 in paper)

→ **Paper location**: Section 8

Each ablation requires:
1. Modify the config
2. Train from scratch (or use saved checkpoints where possible)
3. Evaluate against Greedy (200 games)

| Ablation | Config Change | Status |
|----------|--------------|--------|
| No Dirichlet noise | Set `dirichlet_epsilon = 0` | [ ] |
| No edge-count features | Remove one-hot from preprocess | [ ] |
| WinLoss value target | Set `value_eval = kWinLoss` | [ ] |
| No symmetry augmentation | Set `use_augmentation = false` | [ ] |
| FPU reduction = 0 | Set `fpu_reduction = 0` | [ ] |
| Half MCTS sims (200) | Set sims to 200 in eval | [ ] |
| Double MCTS sims (800) | Set sims to 800 in eval | [ ] |

**Note**: The "half/double MCTS" ablations don't require retraining—just
change sims at evaluation time. All others require retraining.

---

## 4. Figures to Create

### 4.1 Architecture Diagrams

→ **Paper location**: Figures throughout Section 4

- [ ] **Figure 1**: System architecture diagram (replace text box)
  - Use draw.io, TikZ, or similar
  - Show: Workers → InferenceServer → ReplayBuffer → Trainer → Model loop
  - Save as `paper/figures/system_architecture.pdf`

- [ ] **Figure 2**: AlphaZeroBitNet architecture diagram
  - Block diagram: Input → Projection → Residual Tower → Policy/Value heads
  - Show skip connections, LayerNorm, dimensions
  - Save as `paper/figures/network_architecture.pdf`

- [ ] **Figure 3**: PatchNet architecture diagram
  - Show: 5×5 board → 9 overlapping 3×3 patches → Local model → Scatter → Fuse → Global → Output
  - This is the most important figure for the novel contribution
  - Save as `paper/figures/patchnet_architecture.pdf`

- [ ] **Figure 4**: Input feature encoding visualization
  - Show a sample board state and how it maps to the feature vector
  - Color-code: edges (blue), ownership (red/green), one-hot (gray), scalars
  - Save as `paper/figures/feature_encoding.pdf`

### 4.2 Game Diagrams

- [ ] **Figure 5**: Dots-and-Boxes board with indexing
  - Show dots, edges (horizontal/vertical), boxes with index labels
  - Show a chain sequence example
  - Save as `paper/figures/board_diagram.pdf`

---

## 5. Compute Budget Reporting

→ **Paper location**: Section 5 (Implementation) and throughout Results

Record for the final paper:

- [ ] **Hardware specification**:
  - CPU: (e.g., Intel i7-13700H)
  - RAM: 
  - GPU: (if used — CUDA version)
  - OS:

- [ ] **Training compute**:
  - 3×3: total training time, total games, total positions
  - 5×5 monolithic: same
  - 5×5 PatchNet: same

- [ ] **Inference speed**:
  - Average time per MCTS simulation at different sim counts
  - Batch utilization rate (avg batch size / max batch size)

---

## 6. Paper Finalization Checklist

After all experiments are done:

- [ ] Fill in all `--` entries in tables
- [ ] Remove all `% TODO:` comments from `main.tex`
- [ ] Write Section 7.1 (Training Dynamics) analysis
- [ ] Write Section 9 (Analysis and Discussion) based on actual results
- [ ] Write conclusion with actual numbers
- [ ] Fill acknowledgments (advisor name, funding)
- [ ] Add GitHub commit hash for reproducibility
- [ ] Compile paper and check for LaTeX warnings
- [ ] Verify all references resolve (no `[?]` in PDF)
- [ ] Proofread entire paper
- [ ] Generate final PDF

---

## 7. Optional but Recommended

- [ ] **Self-play game visualization**: Record a full game (MCTS agent vs itself)
  and show the sequence of moves as a figure
- [ ] **MCTS tree visualization**: Show the search tree at a critical decision
  point (e.g., chain sacrifice or double-dealing)
- [ ] **Value prediction accuracy**: Scatter plot of predicted value vs actual
  game outcome across test positions
- [ ] **Policy entropy over training**: Show how the policy becomes more
  confident (lower entropy) as training progresses
- [ ] **Head-to-head: monolithic vs PatchNet**: Play 200 games between the two
  5×5 agents directly (not just against baselines)

---

## Quick Start

To begin generating results, run in order:

```bash
# 1. Build the project
cd src/cpp/build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# 2. Train 3×3 model (captures training logs to stdout)
./dots_boxes --config=3x3 2>&1 | tee docs/training_logs/train_3x3_raw.log

# 3. Train 5×5 models (after 3×3 is done for PatchNet)
./dots_boxes --config=5x5 2>&1 | tee docs/training_logs/train_5x5_raw.log
./dots_boxes --config=5x5_patch 2>&1 | tee docs/training_logs/train_5x5_patch_raw.log

# 4. Run evaluations
./dots_boxes --mode=eval --model=../models/_3x3/alphazero_3x3.pt \
  --opponent=greedy --sims=400 --rows=3 --cols=3 --games=200

# 5. Generate plots (using Python scripts — create in docs/scripts/)
python docs/scripts/plot_training_curves.py
python docs/scripts/plot_value_targets.py
python docs/scripts/plot_winrate_over_training.py
```
