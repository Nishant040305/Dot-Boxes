# C++ AlphaZero Dots-and-Boxes

Complete C++ port of the AlphaZero training pipeline for Dots-and-Boxes,
using LibTorch for neural network training and inference.

## Features

- **Rectangular board support**: Train on any `rows × cols` board (e.g., 3×3, 3×4, 5×5)
- **Multi-threaded self-play**: Configurable worker threads with batched inference
- **LibTorch neural network**: FC residual architecture with policy + value heads
- **Modular design**: Separate files for env, agent, model, training, and config

## Project Structure

```
src/cpp/
├── CMakeLists.txt              # Build system
├── main.cpp                    # CLI entry point
├── include/
│   ├── BitBoardEnv.h           # Board environment (bitmask-based)
│   └── FastBitSet.h            # Bitset utility
├── env/
│   └── BitBoardEnv.cpp         # Environment implementation
├── agents/
│   ├── AlphaZeroBitAgent.h     # MCTS agent header
│   └── AlphaZeroBitAgent.cpp   # MCTS agent implementation
├── models/
│   └── AlphaZeroBitNet.h       # LibTorch neural network (header-only)
└── train/
    ├── TrainConfig.h            # Configuration struct
    ├── ReplayBuffer.h           # Thread-safe replay buffer
    ├── InferenceServer.h        # Batched inference server
    ├── SelfPlayWorker.h         # Self-play worker
    ├── AlphaZeroTrain.h         # Trainer header
    └── AlphaZeroTrain.cpp       # Trainer implementation
```

## Build Instructions

### Prerequisites

- C++17 compiler (g++ 9+ or clang++ 10+)
- CMake 3.18+
- LibTorch (already included in `include/libtorch/`)

### Compile

```bash
cd src/cpp
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

### Run

```bash
# Default: 3x3 board, 16 workers, 400 MCTS sims, 10 iterations
./alphazero_train

# Custom square board
./alphazero_train --rows 4

# Custom rectangular board (3 rows, 4 cols)
./alphazero_train --rows 3 --cols 4

# Quick smoke test
./alphazero_train --rows 2 --iters 1 --episodes 4 --workers 2 --sims 10

# Full options
./alphazero_train --rows 3 --cols 3 --workers 8 --sims 200 --iters 20 \
    --episodes 50 --batch 64 --epochs 5 --lr 0.001 --hidden 256 --blocks 6

# See all options
./alphazero_train --help
```

### Output

Models are saved to `../models/` (relative to build dir) as:
```
alphazero_<rows>x<cols>.pt
```
