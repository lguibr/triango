import json
import os
import time

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.optim as optim

from triango.config import get_hardware_config
from triango.model.network import AlphaZeroNet
from triango.training.buffer import ReplayBuffer
from triango.training.self_play import self_play
from triango.training.trainer import train


def main() -> None:
    torch.set_num_threads(1)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    hw_config = get_hardware_config()
    device = hw_config["device"]

    print(f"Booting Next-State AlphaZero ecosystem on: {device}")

    model = AlphaZeroNet(
        d_model=hw_config["d_model"], nhead=hw_config["nhead"], num_layers=hw_config["num_layers"]
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.8)
    buffer = ReplayBuffer(capacity=hw_config["capacity"])

    checkpoint = hw_config["model_checkpoint"]
    if os.path.exists(str(checkpoint)):
        try:
            model.load_state_dict(torch.load(str(checkpoint), map_location=device))
            print("Loaded checkpoint.")
        except Exception as e:
            print(f"Failed to load checkpoint (likely architecture mismatch from upgrade): {e}")
            print("=> Training from Tabula Rasa.")

    ITERATIONS = 50 if device.type == "cuda" else 250
    for i in range(ITERATIONS):
        print(f"\n================ Iteration {i+1}/{ITERATIONS} ================")
        model.eval()

        start = time.time()

        buffer, scores = self_play(model, buffer, hw_config)
        print(f"Self-play generated {len(buffer)} states in {time.time() - start:.2f}s")

        if scores:
            metrics_file = hw_config["metrics_file"]
            metrics = {}
            if os.path.exists(str(metrics_file)):
                with open(str(metrics_file)) as f:
                    try:
                        metrics = json.load(f)
                    except Exception:
                        pass

            iter_key = f"iteration_{i+1}"
            best_score = max(scores)
            median_score = float(np.median(scores))
            metrics[iter_key] = {"best": best_score, "median": median_score, "distribution": scores}
            metrics_dir = os.path.dirname(str(metrics_file))
            if metrics_dir:
                os.makedirs(metrics_dir, exist_ok=True)
            with open(str(metrics_file), "w") as f:
                json.dump(metrics, f, indent=2)

            print("\n--- Score Distribution ---")
            bins = np.linspace(min(scores), max(scores) + 1, 10)
            hist, bin_edges = np.histogram(scores, bins=bins)
            max_count = max(hist) if len(hist) > 0 else 1
            for b_idx in range(len(hist)):
                bar = "█" * int(20 * hist[b_idx] / max_count)
                print(
                    f"{bin_edges[b_idx]:6.1f} - {bin_edges[b_idx+1]:6.1f} | {bar} ({hist[b_idx]})"
                )
            print("--------------------------")

        if len(buffer) > 0:
            train(model, buffer, optimizer, scheduler, hw_config)

            # --- Evaluation Arena ---
            print("\nEntering Evaluation Arena...")
            arena_config = hw_config.copy()
            arena_config["num_games"] = 10
            
            model.eval()
            _, arena_scores = self_play(model, ReplayBuffer(capacity=10), arena_config)
            
            arena_median = float(np.median(arena_scores)) if arena_scores else 0.0
            print(f"Arena Challenger achieved Median Score: {arena_median:.1f}")
            
            best_arena_score = metrics.get("best_arena_score", 0.0)
            
            if arena_median >= best_arena_score:
                print(f"Challenger successfully defeated the Champion! ({arena_median:.1f} >= {best_arena_score:.1f})")
                metrics["best_arena_score"] = arena_median
                with open(str(metrics_file), "w") as f:
                    json.dump(metrics, f, indent=2)
                    
                ckpt_dir = os.path.dirname(str(checkpoint))
                if ckpt_dir:
                    os.makedirs(ckpt_dir, exist_ok=True)
                torch.save(model.state_dict(), str(checkpoint))
                print("=> Saved SOTA PyTorch Model!")
            else:
                print(f"Challenger failed to defeat the Champion ({arena_median:.1f} < {best_arena_score:.1f}). Discarding weights.")
                # Reload champion weights to prevent catastrophic forgetting
                if os.path.exists(str(checkpoint)):
                    model.load_state_dict(torch.load(str(checkpoint), map_location=device))
                    print("=> Restored Champion PyTorch Model.")


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
