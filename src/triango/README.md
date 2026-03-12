# Triango Core Pipeline (`src/triango`)

This directory houses the highest-level orchestrators of the Triango ecosystem. It defines the global configuration abstractions and the main execution loop for distributed AI learning.

## Files and Responsibilities

### `config.py`
The absolute source of truth for runtime hyperparameter constraints, environment variables, and compute hardware detection.
- **`get_hardware_config()`**: Dynamically interrogates the host system to determine the most performant ML backend (NVIDIA CUDA, Apple MPS Framework, or local multi-core CPU inference via multiprocessing pools).
- Returns unified dictionaries containing critical topology parameters such as `d_model`, `batch_sizes`, and MCTS `simulations`.

### `main.py`
The CLI entry point for initiating a limitless AlphaZero self-play and training sequence. Executable globally via `python -m triango.main`.
- **Bootstrapping**: Instantiates the initial `AlphaZeroNet` state and constructs an empty `ReplayBuffer`.
- **The AlphaZero Loop**: Cycles continuously between three asynchronous components:
  1. **Self-Play Generation**: Instructs the `PythonMCTS` to generate novel gameplay trajectories on localized compute nodes, returning enriched MCTS tree policy histories. (`play_one_game_worker()`)
  2. **Model Training**: Extracts `mini_batches` from the ReplayBuffer and applies temporal and categorical loss curves upon `AlphaZeroNet` weights. (`train()`)
  3. **Evaluation Arena**: Pits the newly tuned network against the previously best-saved champion checkpoint. If the median performance overtakes the champion threshold over an arena match block, the new model writes its tensors strictly to `triango_model.pth`.

## Development Operations
If scaling the training pool or diagnosing bottleneck lags, configurations should exclusively be overridden inside `config.py`. Never hardcode `device=cuda` parameters within subclasses, as `main.py` distributes the `hw_config` strictly down the pipeline chain.
