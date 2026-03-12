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
) -> tuple[list[tuple[torch.Tensor, float, float, torch.Tensor]], float]:
    state = GameState()
    game_history: list[tuple[torch.Tensor, float, float, torch.Tensor]] = []

    print(f"--- Game {game_idx+1}/{num_games} Started ---")

    step = 0
    while not state.terminal:
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
        probs = probs / np.sum(probs)
        
        # Build full target policy matrix [3, 50]
        # Map each legal (slot, orientation_idx) to its MCTS visitation probability
        target_policy_mat = torch.zeros(3, 50, dtype=torch.float32)
        for idx_m, m in enumerate(moves):
            slot, o_idx = m
            # clip to bounds just in case
            target_policy_mat[slot, min(o_idx, 49)] = probs[idx_m]

        chosen_idx = np.random.choice(len(moves), p=probs)
        chosen_move = moves[chosen_idx]

        slot, idx = chosen_move

        board_before = state.board

        next_state = state.apply_move(slot, idx)
        if next_state is None:
            break
        state = next_state

        pop_before = bin(board_before).count("1")
        pop_after = bin(state.board).count("1")
        cleared_line = 1.0 if pop_after <= pop_before else 0.0

        feat = extract_feature(state)
        game_history.append((feat.clone().detach(), cleared_line, float(state.score), target_policy_mat))

        step += 1

    print(f"Game {game_idx+1} Finished. Steps: {step}, Final Score: {state.score}")
    return game_history, float(state.score)


def play_one_game_worker(
    args: tuple[int, dict[str, Any] | None, dict[str, Any]],
) -> tuple[list[tuple[torch.Tensor, float, float, torch.Tensor]], float]:
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
        f"Spawning {num_processes} concurrent workers targeting '{hw_config['worker_device'].type}' for self-play generation..."
    )
    # Add a fallback just incase MacOS hangs gracefully
    try:
        with context.Pool(processes=num_processes) as pool:
            results = pool.map(play_one_game_worker, args)
    except RuntimeError as e:
        print(f"Multiprocessing error (likely MPS collision): {e}")
        return buffer, []

    scores = [res[1] for res in results]
    median_score = float(np.median(scores)) if scores else 0.0
    print(
        f"Self-Play Median Score: {median_score:.1f}, Max Score: {max(scores) if scores else 0.0}"
    )
    top_quartile = float(np.percentile(scores, 75)) if scores else 0.0
    max_score = max(scores) if scores else 0.0

    for history, final_score in results:
        buffer.push_game(history, final_score)

    return buffer, scores
