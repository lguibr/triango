# AlphaZero Evaluation & Progress Dashboard

## Goal
Implement a completely automated verifiable evaluation phase where the live Neural Network Agent plays objectively against a pure Random/Heuristic Agent, sending exact Win Rates and Average Score metrics to both the console and TensorBoard.

## Tasks
- [ ] Task 1: **Evaluation Script (`python/evaluate.py`)**: Create an independent script that loads `best_model.pth` and pits it against a purely random simulated Go engine for 100 fast games via gRPC.
  - *Verify*: Script outputs `Neural Win Rate: X% | Neural Avg Score: Y | Random Avg Score: Z`.
- [ ] Task 2: **Continuous Eval Hook in Trainer**: Modify `train.py` to automatically trigger `evaluate.py` at the end of every 3 epochs or on completion.
  - *Verify*: The training loop natively halts, runs the combat evaluation, and prints the Elo/Score difference before resuming.
- [ ] Task 3: **TensorBoard Metric Injection**: Pipe the `Neural_Avg_Score` and `Win_Rate_vs_Random` metrics directly into `SummaryWriter`.
  - *Verify*: The `localhost:6006` dashboard populates with two new charts exclusively tracking the live Agent's actual tabletop Performance capability.
- [ ] Task 4: **Go Engine Duel Mode (`cmd/evaluator/main.go`)**: Create a specific Go binary that handles two distinct MCTS engines (One Neural, One Random) playing on the exact same starting boards to eliminate RNG luck.
  - *Verify*: The Go engine returns deterministic combat data to Python.

## Done When
- [ ] The user can actively sit on TensorBoard and clearly watch the "Average AlphaZero Game Score" graph climb linearly as the network learns geometry.
- [ ] We have a mathematically fair baseline (the pure randomized actor) to definitively prove the AI is superhuman.
