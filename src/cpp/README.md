# C++ AlphaZeroBit Port

This folder is the start of a C++ port of the bitboard-based AlphaZero stack.

Current status:
- `BitBoardEnv` ported with bitmask state, legal move mask, clone, and render.
- `AlphaZeroBitAgent` ported with MCTS core and pluggable policy/value interface.
- `FastBitset` modernized as a header-only utility.

Next steps:
- Implement a C++ model wrapper for inference (ONNX Runtime / libtorch).
- Port training loop pieces if you intend to train in C++.

