# Triango MCTS Intelligence (`/mcts`)

The `mcts` engine utilizes the pure Go concurrency architecture to parallelize highly capable intelligence agents optimally, leveraging Wait-Free Root Parallelization to sidestep all mutex bottlenecking during deep exploratory Monte Carlo Rollouts.

## Core Capabilities

- **`node.go`:** Controls the Game Tree Node expansion mechanics. Dynamically filters out impossible structural placements dynamically via deterministic collision caching. Exposes `SelectChild`, enforcing balancing of exploration/exploitation through $UCT$ (Upper Confidence bounds applied to Trees).
- **`search.go`:** The primary operational command. Implements `ParallelSearch` using independent root amalgamation. By dispatching radically decoupled tree expansions downward within dedicated Goroutines—each initialized by a mathematically isolated `rand.Rand` context—Triango guarantees perfectly linear multi-processor scaling $O(n)$ lacking state corruption or lock contention.
