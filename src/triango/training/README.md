# Distributed Execution & Memory Optimization (`src/triango/training`)

The `training` module provides the robust infrastructure to organically scale the Triango self-play loops and optimize PyTorch backend compute capabilities via distributed worker clustering.

## Execution Loops

### `buffer.py`
To prevent massive reinforcement learning loops from blindly overfitting over newer (potentially worse) iterations, AlphaZero models require immense historical memory.
- The `ReplayBuffer` implements a unified `PyTorch Dataset` structure capable of holding `250,000` total geometric states.
- **Elite Architecture**: Data pushed with a final game score beneath the `elite_threshold` enters the standard queue. Games scoring exceptionally well dynamically update the threshold and are permanently partitioned into a separate 10% protected vector memory, preventing random eviction. This permanently cures AI "catastrophic forgetting" of successful geometric patterns.

### `self_play.py`
To fully leverage symmetric hardware threads effectively, we bypass native Python GIL limits for local reinforcement trajectories.
- `PythonMCTS` traverses independently inside an asynchronous `mp.get_context("spawn").Pool()`. 
- Each concurrent worker process generates localized game trajectories containing raw 96-tile boolean masks, `(-1, 0, 1)` reward scoring, and exact `(3, 50)` MCTS target policy probabilities observed.
- The processes then funnel trajectory metadata asynchronously back into the centralized RAM Elite Replay Buffer.

### `trainer.py`
The backpropagation engine. Consumes `DataLoader` batches natively from the `ReplayBuffer` and maps the categorical optimization curve onto `AlphaZeroNet`.
- Utilizes standard Mean-Squared Error `F.mse_loss` against the `value` branch to accurately guide long-term piece placement rewards.
- Utilizes Binary Cross-Entropy `F.bce_loss` explicitly towards the `line_clear` prediction.
- Computes sophisticated KL-Divergence distributions `$H(p, q) = -\sum P(x) \log Q(x)$` explicitly against the `AlphaZeroNet > Policy Head`, directly constraining the heuristic Softmax branch to map 1:1 against the raw MCTS topological decisions observed during simulations.
