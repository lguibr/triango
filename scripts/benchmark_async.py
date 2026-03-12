import time
import torch
from triango.env.state import GameState
from triango.model.network import AlphaZeroNet
from triango.mcts.search import PythonMCTS
import triango_ext

def run_benchmark():
    print("Initializing Environment...")
    triango_ext.initialize_env()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = AlphaZeroNet().to(device)
    model.eval()
    
    batch_size = 64
    simulations = 800
    
    print(f"\n--- Benchmarking AsyncMCTS (Native C++ Threads + PyTorch Batch={batch_size}) ---")
    
    mcts = PythonMCTS(model, device, batch_size=batch_size)
    state = GameState()
    
    # Warmup
    print("Warming up GPU...")
    mcts.search(state, simulations=100)
    
    # Actual Benchmark
    num_moves = 10
    start_time = time.time()
    
    total_sims = 0
    for i in range(num_moves):
        t0 = time.time()
        move, visits = mcts.search(state, simulations=simulations)
        t1 = time.time()
        
        sims_per_sec = simulations / (t1 - t0)
        print(f"Move {i+1}: {simulations} simulations in {t1-t0:.4f}s ({sims_per_sec:.2f} sims/sec)")
        total_sims += simulations
        
        # Apply the move to keep the tree state evolving
        next_state = state.apply_move(move[0], move[1])
        if next_state is None:
            print("Terminal reached early.")
            break
        state = next_state
        
        if state.pieces_left == 0:
            state.refill_tray()
            if state.terminal:
                print("Game Over naturally.")
                break
        
    total_time = time.time() - start_time
    avg_sims_per_sec = total_sims / total_time
    
    print(f"\n--- RESULTS ---")
    print(f"Total Time for {num_moves} moves ({total_sims} simulations): {total_time:.4f}s")
    print(f"Average Throughput: {avg_sims_per_sec:.2f} simulations/sec")
    print(f"Previous Pure Python sync speeds were ~30-50 sims/sec on CPU.")

if __name__ == "__main__":
    run_benchmark()
