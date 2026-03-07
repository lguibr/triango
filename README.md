# Triango

State-of-the-Art (SOTA) high-performance Go engine for **Tricrack**, powered by native Root-Parallelized Monte Carlo Tree Search (MCTS) and zero-allocation immutable bitboards.

Triango is engineered to solve the complex 96-cell triangular interlocking pieces of Tricrack at native limits, simulating tens of thousands of complete game instances per second without triggering the Go Garbage Collector.

## Getting Started (The Triango Command Center)

Triango operates as a 3-node localized Reinforcement Learning ecosystem. To train your AI to become superhuman, we built a beautiful, unified Command Line Interface to orchestrate everything natively.

To begin the AlphaZero loop, open 3-4 terminal tabs and run these pure commands from the root directory:

### 1. Start the PyTorch Inference Server
This backend hosts the Dual-Headed ResNet, analyzing `[96]board` states and returning `$P$` and `$V$` predictions to Go.
```bash
./triango.py server
```

### 2. Stream Self-Play Data (The Engine)
Run the core Go engine. It connects to the PyTorch server via loopback, generates games natively across 8 cores using PUCT, and streams the absolute geometric observations into `.jsonl`.
```bash
./triango.py play
```

### 3. Run the Deep Learning Trainer
Execute the continuous gradient descent loop. It perfectly reads the `games.jsonl`, calculates Divergence, updates the PyTorch Weights, and hot-swaps them to disk for the inference server to immediately use!
```bash
./triango.py train
```

### 4. Live Telemetry (TensorBoard)
To visualize exactly how well the Triango Model is learning to predict the `Score` and `P` move values in real-time:
```bash
./triango.py dashboard
# -> View live graphs at http://localhost:6006
```

### 5. Deterministic A/B Duel
Automatically pit the live neural network against a purely blind Monte-Carlo heuristic agent on 30 identical seeds to prove the AlphaZero engine is Superhuman:
```bash
./triango.py duel
```

### 6. The "Walk Away" Auto-Orchestrator
To completely saturate the Apple Silicon hardware, we built a single command to automate all terminals concurrently. By running:
```bash
./triango.py auto --hours 1
```
The Command Center will boot the Inference Server, build a background Go Worker pool to generate games infinitely, and spin up TensorBoard. Every 5 minutes, it will automatically wake the Trainer to minimize Loss, Hot-Swap new weights into the active server, and repeat completely unsupervised.

## Internal Geometry and Rule Specs
Instead of standard visual constraints (which inflate coordinates), the physical bounds are flattened mathematically into a 128-bit structure (a `[2]uint64` byte mask). Look inside purely `core/README.md` to see exactly how scores geometrically escalate via Triango's non-standard line-collapses.

## Core Engineering

Triango achieves extreme performance through four foundational pillars:

1. **128-bit Bitboard State:** The entire hexagonal 96-triangle grid is packed into a `[2]uint64` byte-aligned Bitboard. All collision detection, geometric bounds checking, and line-clearing validation are executed via raw bitwise instructions.
2. **Lock-Free Root Parallelization:** The MCTS `ParallelSearch` provisions isolated RNG-seeded execution threads. Trees expand entirely in parallel without mutexes, aggregating value networks safely at the root node post-computation.
3. **Immutable Transistions:** The `GameState` struct relies exclusively on value-copying (`< 48 bytes`). Game trees advance through `ApplyMove` generating an entirely autonomous state fork, bypassing heap allocation overhead.
4. **Precompiled Axial Masks:** Triangular piece configurations are projected via 3D cartesian mappings in `init()`, storing all 96 absolute position intersections in memory for Instant Valid Move Verification.

## Quickstart

Witness the Monte Carlo Engine playing perfectly against itself in real-time on your console:

```bash
go run cmd/visualizer/main.go
```

To run the full suite of integration tests and performance benchmarks (measuring operations per nanosecond):

```bash
go test -bench . ./...
```

## Module Taxonomy

- [`/core`](core/README.md) - The core geometric constraints, bit-level intersections, grid mappings (FlatIndex), the immutable transition state logic, and visual standard piece layouts.
- [`/mcts`](mcts/README.md) - The Intelligence. Handles Node definitions, UCT formulas, rollout policies, and scalable wait-free Thread distribution.
- `/cmd/visualizer` - The CLI harness driving self-play interactions and debug stringification.
