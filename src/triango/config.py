import multiprocessing
from typing import Any

import torch


def get_hardware_config() -> dict[str, Any]:
    """
    Dynamically determines the maximum safe hyper-parameters based on the executing environment.
    Optimizes for RTX 3080 Ti locally while retaining safe limits for Apple Silicon (MPS).
    """
    if torch.cuda.is_available():
        # Windows / Linux - NVIDIA RTX Ecosystem
        # Expected baseline: RTX 3080 Ti Laptop GPU (16GB VRAM), 20 CPU threads.
        return {
            "device": torch.device("cuda"),
            "model_checkpoint": "models/best_model_python.pth",
            "metrics_file": "models/metrics.json",
            "d_model": 512,  # Large Transformer
            "nhead": 8,
            "num_layers": 8,
            "capacity": 250000,
            "num_games": 32,
            "simulations": 800,
            "self_play_batch_size": 256,
            "train_batch_size": 1024,
            "train_epochs": 10,
            "num_processes": min(4, multiprocessing.cpu_count()),  # Cap at 4 to prevent CUDA OOM
            "worker_device": torch.device("cuda"),  # Run MCTS heavily on GPU
        }
    elif torch.backends.mps.is_available():
        # MacOS - Apple Silicon Ecosystem
        # MPS unified memory easily OOMs or hangs from multi-processed heavy model access.
        return {
            "device": torch.device("mps"),
            "model_checkpoint": "models/best_model_mac.pth",
            "metrics_file": "models/metrics_mac.json",
            "d_model": 128,  # Smaller Transformer
            "nhead": 8,
            "num_layers": 8,
            "capacity": 100000,
            "num_games": 100,
            "simulations": 500,
            "self_play_batch_size": 64,
            "train_batch_size": 256,
            "train_epochs": 20,
            # Maximize CPU usage since MPS workers deadlock when sharing memory
            "num_processes": min(14, multiprocessing.cpu_count()),
            "worker_device": torch.device("cpu"),  # Force CPU to bypass MPS deadlock
        }
    else:
        # Fallback - Basic CPU
        return {
            "device": torch.device("cpu"),
            "model_checkpoint": "models/best_model_cpu.pth",
            "metrics_file": "models/metrics_cpu.json",
            "d_model": 64,
            "nhead": 4,
            "num_layers": 4,
            "capacity": 50000,
            "num_games": 16,
            "simulations": 100,
            "self_play_batch_size": 32,
            "train_batch_size": 64,
            "train_epochs": 5,
            "num_processes": max(1, multiprocessing.cpu_count() - 2),
            "worker_device": torch.device("cpu"),
        }
