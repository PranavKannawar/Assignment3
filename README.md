# Multi-GPU Competitive Connect-4 Championship

An enterprise-grade, high-performance CUDA implementation of Connect-4 featuring two autonomous GPU-accelerated competitor engines running concurrently:
1. **Competitor 1 (Red / Player 1)**: Massive Parallel Monte Carlo Tree Search (MCTS) / Random Rollouts kernel.
2. **Competitor 2 (Yellow / Player 2)**: Parallel Depth-Limited Minimax Search with Alpha-Beta Pruning and positional evaluation heuristics.

The system supports true multi-GPU architectures (Device 0 vs Device 1) as well as single-GPU environments (such as the Coursera cloud VM) via multi-threaded host workers and lock-file based turn synchronization (`output/turn.lock`).

---



---

## Key Features

- **True Dual-GPU or Lock-File Single-GPU Operation**: Detects available devices at runtime via `cudaGetDeviceCount()`. If 2 or more GPUs are present, each competitor runs exclusively on its own dedicated device. If only 1 GPU is detected, dual host threads schedule independent GPU work using lock files to prevent race conditions.
- **Competitor 1: Parallel Monte Carlo Rollouts (`src/competitor1_mcts.cu`)**:
  - Distributes hundreds of thousands of simulated games across GPU blocks and warps.
  - Generates fast pseudo-random paths using inline device `xorshift32` state generators.
  - Employs lock-free atomic accumulators (`atomicAdd`) to aggregate candidate column win rates.
- **Competitor 2: Parallel Depth-Limited Minimax (`src/competitor2_minimax.cu`)**:
  - Evaluates root game tree branches simultaneously across parallel GPU threads.
  - Implements alpha-beta pruning and comprehensive window scoring heuristics (center bias, 2-in-a-row, 3-in-a-row traps, instant threat detection).
- **Zero Third-Party Dependency**: Pure standard C++17 and CUDA Runtime API.
- **Turn-by-Turn Visualization**: Includes both an ASCII CLI terminal viewer and an interactive HTML5/CSS3 web visualizer (`output/game_visualization.html`).

---

## 🛠️ Build & Execution Instructions

### Prerequisites
- NVIDIA CUDA Toolkit 11.0+ (`nvcc`)
- GCC / G++ 9+ with C++17 support
- POSIX threads (`-lpthread`)

### Compiling
```bash
make clean
make all
```

### Running the Championship
Run with default parameters (100,000 MCTS rollouts per turn, Minimax depth 5):
```bash
make run
```

Or pass custom parameters:
```bash
./bin/connect4_gpu <num_simulations> <minimax_depth>
# Example:
./bin/connect4_gpu 250000 6
```

### Visualizing the Match
Generate interactive HTML replay and view steps:
```bash
python3 scripts/visualize_game.py
# Open output/game_visualization.html in any web browser!
```

---

## Project Structure

```
├── Makefile                      # Build recipes for nvcc and g++
├── include/
│   └── connect4_common.h         # Board state representation, win logic, and lock helpers
├── src/
│   ├── competitor1_mcts.cu       # GPU Monte Carlo simulation kernel
│   ├── competitor2_minimax.cu    # GPU Minimax search with alpha-beta pruning
│   ├── game_engine.cpp           # Board CLI renderer and JSON logging
│   └── main.cpp                  # Multi-threaded game loop & lock-file turn coordinator
├── scripts/
│   └── visualize_game.py         # Generates interactive HTML replay visualization
└── output/
    ├── game_log.json             # Serialized turn-by-turn match log
    └── game_visualization.html   # Standalone interactive graphical replay
```
