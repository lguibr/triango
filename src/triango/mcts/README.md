# Monte-Carlo Tree Search (`src/triango/mcts`)

This directory houses the algorithms responsible for generating robust gameplay trajectories and establishing the probabilistic bounds of intelligent action selection. 

## Search Optimization Architecture

### `node.py`
The foundational component of the search tree. A `Node` captures an absolute instance of `GameState`.
- Tracks exploration metrics including `visits`, `total_score`, and structural priors.
- Automatically caches expanded legal moves as its topological children.
- Implements the **PUCT Algorithm** (Predictor + Upper Confidence Tree). Balances Exploitation (trusting high-value predictions) vs. Exploration (traveling under-visited paths with high prior probability).

### `search.py`
The orchestrator that runs iterative simulations against the `AlphaZeroNet` to resolve ambiguous game states.
- **Batched GPU Evaluation**: Standard MCTS executes single neural net inferences sequentially at every leaf. `PythonMCTS` traverses down $N$ independent paths simultaneously until it discovers $N$ separate leaf configurations. These are grouped into a singular monolithic PyTorch Batch tensor, fundamentally dissolving the multi-threading GPU bottleneck.
- **Learned Policy Injection**: During root expansion, `AlphaZeroNet` outputs a custom $P(s, a)$ tensor dictating the relative likelihood of every geometric move constraint. These probabilities are injected instantly into child node `priors`, allowing the tree to instantly focus computational depth towards high-confidence placements.
- **Virtual Loss**: While independent workers asynchronously traverse the tree searching for batch targets, paths apply temporary "virtual loss" to node metadata. This intelligently funnels concurrent workers away from identical branches so they explore parallel topologies efficiently.
- **Dirichlet Noise**: To maintain tactical diversity during Self-Play operations, AlphaZero injects raw mathematical noise directly into the root branch probabilities. This forcefully incentivizes the system to continually test suboptimal/novel edge cases rather than collapsing into deterministic gameplay loops.

### `features.py`
The neural interpretation framework. Converts logical `GameState` data (like score or remaining pieces) into rigid 7-dimensional image tensors perfectly shaped for the `AlphaZeroNet`.
- Constructs a $7 \times 96$ categorical Boolean stack encapsulating:
  1. **Occupancy**: Current `1`s on the 96-grid.
  2. **Valid Geometry Bounds**: Which of the 96 anchors points "up" vs "down".
  3. **Tray Availability**: Hot-encoded vectors declaring which shapes (1-5) are physically available to drop this turn.
