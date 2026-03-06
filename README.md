# Triango

State-of-the-Art (SOTA) high-performance Go engine for **Tricrack**, powered by native Root-Parallelized Monte Carlo Tree Search (MCTS) and zero-allocation immutable bitboards.

Triango is engineered to solve the complex 96-cell triangular interlocking pieces of Tricrack at native limits, simulating tens of thousands of complete game instances per second without triggering the Go Garbage Collector.

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
