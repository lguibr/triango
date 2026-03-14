from typing import Any

import numpy as np
import torch
import torch.multiprocessing as mp

from triango.env.state import GameState
from triango.mcts.features import extract_feature
from triango.mcts.search import PythonMCTS
from triango.model.network import AlphaZeroNet
from triango.training.buffer import ReplayBuffer


def play_one_game(
    game_idx: int, mcts: PythonMCTS, simulations: int, num_games: int
) -> tuple[list[tuple[np.ndarray[Any, Any], float, np.ndarray[Any, Any]]], float]:
    state = GameState()
    game_history: list[tuple[np.ndarray[Any, Any], float, np.ndarray[Any, Any]]] = []

    step = 0
    # Hard cap 10,000 steps to prevent practically infinite games from halting the epoch
    for step in range(10000):
        if state.pieces_left == 0:
            state.refill_tray()
            
        if state.terminal:
            break

        best_move, visits = mcts.search(state, simulations=simulations)

        if best_move is None:
            break

        temp = 1.0 if step < 15 else (0.5 if step < 30 else 0.1)

        moves = list(visits.keys())
        counts = np.array([visits[m] for m in moves], dtype=np.float64)

        probs = counts ** (1.0 / temp)
        probs_sum = np.sum(probs)
        if probs_sum == 0:
            probs = np.ones_like(probs) / len(probs)
        else:
            probs = probs / probs_sum
        
        # Build full target policy matrix [3, 50]
        # Map each legal (slot, orientation_idx) to its MCTS visitation probability
        target_policy_mat = np.zeros((3, 50), dtype=np.float32)
        for idx_m, m in enumerate(moves):
            slot, o_idx = m
            # clip to bounds just in case
            target_policy_mat[slot, min(o_idx, 49)] = probs[idx_m]

        chosen_idx = np.random.choice(len(moves), p=probs)
        chosen_move = moves[chosen_idx]

        slot, idx = chosen_move

        next_state = state.apply_move(slot, idx)
        if next_state is None:
            break
        state = next_state

        feat = extract_feature(state)
        game_history.append((feat.cpu().numpy(), float(state.score), target_policy_mat))

        step += 1
    else:
        print(f"Warning: Game {game_idx} hit maximum depth cutoff (10000 steps). Terminating early.")
        
    return game_history, float(state.score)


def play_one_game_worker(
    args: tuple[int, dict[str, Any] | None, dict[str, Any]],
) -> tuple[list[tuple[np.ndarray[Any, Any], float, np.ndarray[Any, Any]]], float]:
    try:
        import torch

        torch.set_num_threads(1)
        game_idx, state_dict, hw_config = args

        worker_device = hw_config["worker_device"]

        model = AlphaZeroNet(
            d_model=hw_config["d_model"],
            nhead=hw_config["nhead"],
            num_layers=hw_config["num_layers"],
        ).to(worker_device)

        if state_dict is not None:
            # Reconstruct safely from state_dict to bypass multiprocessing deadlocks
            model.load_state_dict(state_dict)
        model.eval()

        mcts = PythonMCTS(model, worker_device, batch_size=hw_config["self_play_batch_size"])
        return play_one_game(game_idx, mcts, hw_config["simulations"], hw_config["num_games"])
    except Exception as e:
        import traceback

        print(f"Worker {args[0]} failed: {e}")
        traceback.print_exc()
        return [], 0.0


def self_play(
    model: AlphaZeroNet, buffer: ReplayBuffer, hw_config: dict[str, Any]
) -> tuple[ReplayBuffer, list[float]]:
    import sys
    
    context = mp.get_context("spawn")
    num_games = hw_config["num_games"]

    state_dict = None
    if hw_config["device"].type != "cpu":
        state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
    else:
        state_dict = model.state_dict()

    args = [(i, state_dict, hw_config) for i in range(num_games)]

    results = []

    num_processes = hw_config["num_processes"]
    print(
        f"Spawning {num_processes} concurrent workers targeting '{hw_config['worker_device'].type}' for {num_games} games..."
    )
    
    completed_games = 0
    running_scores = []
    
    try:
        with context.Pool(processes=num_processes) as pool:
            # We use imap_unordered to get results as they finish to update the progress bar live
            for history, final_score in pool.imap_unordered(play_one_game_worker, args):
                results.append((history, final_score))
                running_scores.append(final_score)
                completed_games += 1
                
                # Live Analytics
                curr_med = np.median(running_scores)
                curr_max = max(running_scores)
                curr_min = min(running_scores)
                curr_mean = np.mean(running_scores)
                
                # Build Progress Bar string
                pct = int((completed_games / num_games) * 20)
                try:
                    bar = "█" * pct + "-" * (20 - pct)
                    sys.stdout.write(
                        f"\r[{bar}] {completed_games}/{num_games} | "
                        f"Med: {curr_med:.1f} | Avg: {curr_mean:.1f} | Max: {curr_max:.1f} | Min: {curr_min:.1f}   "
                    )
                    sys.stdout.flush()
                except UnicodeEncodeError:
                    bar = "#" * pct + "-" * (20 - pct)
                    sys.stdout.write(
                        f"\r[{bar}] {completed_games}/{num_games} | "
                        f"Med: {curr_med:.1f} | Avg: {curr_mean:.1f} | Max: {curr_max:.1f} | Min: {curr_min:.1f}   "
                    )
                    sys.stdout.flush()
                
            print() # Newline after progress bar finishes
    except RuntimeError as e:
        print(f"\nMultiprocessing error: {e}")
        return buffer, []

    scores = [res[1] for res in results]
    
    for history_np, final_score in results:
        history_tensor = [
            (torch.from_numpy(f), s, torch.from_numpy(p))
            for f, s, p in history_np
        ]
        buffer.push_game(history_tensor, final_score)

    return buffer, scores
