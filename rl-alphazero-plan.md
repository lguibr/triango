# Triango: AlphaZero RL Integration Plan

## Goal
Build a high-performance Python-Go bridge to train a PyTorch Neural Network using the native `triango` MCTS engine for self-play data generation.

## Tasks
- [x] Task 1: **Dataset Recorder (Go)**: Create a system to write MCTS states (the bitboards), normalized visit counts (the target Policy), and final game score (the target Value) into `.jsonl` or `.pb` files during self-play.
  - *Verify*: Run MCTS self-play and check that a mathematically valid `games.jsonl` file is written to disk containing at least 1 turn's data.
- [x] Task 2: **PyTorch Model (Python)**: Stand up a simple ResNet or Transformer architecture accepting the 96-triange binary planes as input and outputting `Value` (scalar) and `Policy` (96-len array).
  - *Verify*: Pass a dummy Tensor payload through the model and print the parameter shape.
- [x] Task 3: **Inference Bridge (Go $\leftrightarrow$ Python)**: Implement the communication protocol (either a fast lightweight `zeromq`/`gRPC` server in Python that Go calls, or bind Go via `pybind11`).
  - *Verify*: Go MCTS pauses on expansion, calls Python for $P$ and $V$, and successfully updates a node.
- [x] Task 4: **PUCT Implementation (Go)**: Modify `SelectChild` from standard UCT to PUCT, weighting node exploration by the Neural Network's prior probability.
  - *Verify*: Run Go MCTS and observe that nodes with high NN probability are explored vastly more often.
- [x] Task 5: **Training Loop (Python)**: Write the PyTorch continuous loop script `train.py` that reads the `games.jsonl`, takes a gradient step, and hot-swaps the model weights to the running inference bridge.
  - *Verify*: Loss goes down over 10 epochs on dummy data.

## Done When
- [x] We have a `train.py` actively reducing loss.
- [x] We have a `simulate.go` script streaming high-throughput self-play games leveraging a live PyTorch model instead of random rollouts.
- [x] The Neural Network agent demonstrably outperforms the pure multi-core UCT random-rollout agent.
