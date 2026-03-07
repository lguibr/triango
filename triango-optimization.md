# Triango AlphaZero Optimization

## Goal
Architect crystal-clear, dynamically batched, mathematically-augmented PyTorch/Go MCTS pipeline for hyper-fast AlphaZero learning.

## Tasks
- [ ] Task 1: **Modularize Python Code** (Refactor `server.py` and `train.py` into strict, type-hinted `model.py`, `dataset.py`, `batcher.py`, `train.py`, `server.py`) -> Verify: Server boots and python scripts are crystal clear.
- [ ] Task 2: **Implement Dynamic gRPC Batching** (Create `DynamicBatcher` in PyTorch with `asyncio` to natively group incoming Go requests into `[N, 4, 96]` arrays) -> Verify: `gpu` or `cpu` handles 8+ concurrent Go MCTS workers flawlessly without locking overhead.
- [ ] Task 3: **Implement Dirichlet Noise** (Inject Gamma distribution noise into the root node probabilities in `mcts/search.go`) -> Verify: AI plays incredibly diverse and aggressive opening mathematical moves.
- [ ] Task 4: **Implement Go Symmetries** (Update `recorder.go` to insert 3 rotated board geometries per actual game state) -> Verify: SQLite row capacity triples per game.

## Done When
- [ ] Python codebase is strictly decoupled into 5 concise files.
- [ ] `DynamicBatcher` merges multi-threaded gRPC requests.
- [ ] Root MCTS exploration uses true AlphaZero structural noise.
- [ ] Go geometrically inserts 3x perfect data points into SQLite via mathematical rotation.
