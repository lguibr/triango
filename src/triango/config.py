import multiprocessing
from typing import Any

import torch


def get_hardware_config() -> dict[str, Any]:
    """
    Dynamically determines the maximum safe hyper-parameters based on the executing environment.
    Optimizes for RTX 3080 Ti locally while retaining safe limits for Apple Silicon (MPS).
    
    *** TINY TESTING MODEL (For quick architecture validation and debugging) ***
    To quickly verify if the AI is learning without waiting hours, use this ultra-light config:
        "d_model": 32, "nhead": 2, "num_layers": 2
        "num_games": 32, "simulations": 50, "train_epochs": 1
    This speeds up iteration times drastically, allowing you to observe rapid loss convergence.
    """
    if torch.cuda.is_available():
        # Windows / Linux - NVIDIA RTX Ecosystem
        # Expected baseline: RTX 3080 Ti Laptop GPU (16GB VRAM), 20 CPU threads.
        return {
            "device": torch.device("cuda"),
            "model_checkpoint": "models/best_model_v2_python.pth",
            "metrics_file": "models/metrics_v2.json",
            "d_model": 64,  # Small Fast Transformer
            "nhead": 4,
            "num_layers": 4,
            "capacity": 250000,
            "num_games": 2048,          # Scale up Game Generation locally 
            "simulations": 64,
            "self_play_batch_size": 1024,
            "train_batch_size": 1024,  # Scale up PyTorch Batch optimization targets
            "train_epochs": 4,      
            # Squeeze max gradient loss out of the Elite sequences
            "num_processes": max(4, multiprocessing.cpu_count() - 2),  # Unleash full CPU cores!
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
