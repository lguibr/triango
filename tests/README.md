# Triango Test Suite (`tests/`)

The tests directory validates the functional integrity of all modular environments, math, search traversals, and Python networking within the system using the `pytest` runner.

## Core File Components
- **`test_config.py`**: Ensures the config abstraction appropriately manages hardware backpressure dynamically depending on system GPU/CPU load bounds relative to available cores.
- **`test_env.py`** & **`test_state*.py`**: Granular 50+ assertion scenarios strictly testing `GameState` edge boundary cases, exact geometric layout rotations, and simulated map placements.
- **`test_search.py`** & **`test_search_coverage.py`**: Interrogates `PythonMCTS` and PUCT mathematical coverage constraints, guaranteeing valid batch-tensors structure without collisions natively.
- **`test_main.py`** & **`test_self_play*.py`**: Spawns isolated multiprocessing Pools using `unittest.mock` to ensure self-play evaluation mechanisms seamlessly construct and populate `ReplayBuffer` target sequences flawlessly without deadlocking.

## Diagnostics
### `test_features.py` (AI Visualization)
We've isolated an exact test explicitly meant to reverse-engineer and print the internal categorical representation variables that `AlphaZeroNet` expects.
This natively generates a 2D ascii breakdown of any 96-tile grid sequence to visually inspect raw boolean bitmasks the AI natively reads during `features.py` execution!

To run this visual diagnostic standalone:
```bash
python tests/test_features.py
```

## Running Verification Commands
To execute the total 27+ integration sequence mathematically and print organic structural coverage statements > 80%, execute:
```bash
python -m pytest tests --cov=triango
```
