import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import time
import os
import json
import torch.multiprocessing as mp

from env import GameState, TOTAL_TRIANGLES
from mcts import PythonMCTS, extract_feature
from model import AlphaZeroNet

class ReplayBuffer(Dataset):
    def __init__(self, capacity=250000):
        self.capacity = capacity
        self.buffer = []
        
    def __len__(self):
        return len(self.buffer)
        
    def __getitem__(self, idx):
        # state is [4, 96]
        # target_value is scalar True Game Score
        # target_line_clear is scalar probability [0.0 or 1.0] if this action cleared a line
        state, target_value, target_line_clear = self.buffer[idx]
        return state, torch.tensor([target_value], dtype=torch.float32), torch.tensor([target_line_clear], dtype=torch.float32)
        
    def push(self, state_tensor, target_value, target_line_clear):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append((state_tensor, target_value, target_line_clear))


def play_one_game(game_idx, mcts, simulations, num_games):
    state = GameState()
    game_history = []
    
    print(f"--- Game {game_idx+1}/{num_games} Started ---")
    
    step = 0
    while not state.terminal:
        if state.pieces_left == 0:
            state.refill_tray()
            if state.terminal:
                break
                
        # The MCTS returns the best move and visit distribution
        best_move, visits = mcts.search(state, simulations=simulations)
        
        if best_move is None:
            break
            
        # Dynamic Temperature Exploration
        # Early game (step 0-15): temp = 1.0 (Highly exploratory)
        # Mid game (15-30): temp = 0.5 (Starts favoring strong moves)
        # Late game (>30): temp = 0.1 (Almost entirely greedy / exploitation)
        temp = 1.0 if step < 15 else (0.5 if step < 30 else 0.1)
        
        moves = list(visits.keys())
        counts = np.array([visits[m] for m in moves], dtype=np.float64)
        
        # Apply temperature scaling to visit counts
        probs = counts ** (1.0 / temp)
        probs = probs / np.sum(probs)
        
        chosen_idx = np.random.choice(len(moves), p=probs)
        chosen_move = moves[chosen_idx]
            
        # Extract features for THIS specifically chosen move's after-state 
        # (Since we are doing Next-State evaluation, the neural net looks at the board AFTER the place)
        slot, idx = chosen_move
        
        # We need to know if this specific move caused a line clear.
        # Track the score before
        score_before = state.score
        board_before = state.board
        
        # Apply move
        state = state.apply_move(slot, idx)
        if state is None: # Should never happen with MCTS legal logic
            break
            
        # To determine if a line was cleared: 
        # normally placing a piece adds points == pure bits added.
        # But if the total score grew by more than the piece size + bonuses, or the board shrank...
        # Safer: just check if the new board has fewer bits than before + piece bits. 
        # Or even simpler, if the score jumped significantly. Since line clears give 2x bits, we check score diff.
        
        # A standard piece is max 6 blocks. So a jump > 10 is definitely a line clear.
        # Let's cleanly compute if any lines were destroyed (population count dropped)
        pop_before = bin(board_before).count('1')
        pop_after = bin(state.board).count('1')
        # If the number of triangles on the board went DOWN or stayed perfectly equal despite adding a piece, a line was cleared
        cleared_line = 1.0 if pop_after <= pop_before else 0.0
        
        # Save the *Next-State* (the state we just transitioned to)
        feat = extract_feature(state)
        game_history.append((feat.clone().detach(), cleared_line, state.score))
        
        # RENDERING MUTED: To prevent terminal flooding and I/O lockups when the model survives for hundreds of steps
        # if game_idx == 0:
        #     state.render()
            
        step += 1
        
    print(f"Game {game_idx+1} Finished. Steps: {step}, Final Score: {state.score}")
    return game_history, state.score

def play_one_game_worker(args):
    try:
        import torch
        # Critical. Force thread count to 1 inside the worker to prevent nested CPU thread thrashing
        torch.set_num_threads(1)
        game_idx, state_dict, simulations, num_games, batch_size = args
        
        # Explicitly instantiate the localized model on the CPU for generation.
        # This completely bypasses CUDA/MPS memory/lock constraints and allows us to scale 
        # to 16+ workers easily without OOMing the GPU.
        cpu_device = torch.device('cpu')
        # MAX POWER Scaling
        model = AlphaZeroNet(d_model=512, nhead=16, num_layers=16).to(cpu_device)
        if state_dict is not None:
            model.load_state_dict(state_dict)
        model.eval()
        
        mcts = PythonMCTS(model, cpu_device, batch_size=batch_size)
        return play_one_game(game_idx, mcts, simulations, num_games)
    except Exception as e:
        import traceback
        print(f"Worker {args[0]} failed: {e}")
        traceback.print_exc()
        return [], 0

def self_play(model, buffer, device, num_games=128, simulations=2400, batch_size=256):
    
    context = mp.get_context('spawn')
    
    # Extract state_dict to pass safely instead of the full model object
    # Crucially, we MUST move it to CPU first, otherwise MPS/CUDA tensors will crash pickler
    state_dict = None
    if device.type != 'cpu':
        state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
    else:
        state_dict = model.state_dict()
    
    args = [(i, state_dict, simulations, num_games, batch_size) for i in range(num_games)]
    
    results = []
    # Aggressively saturate the CPU to feed the RTX 3080 Ti
    num_processes = min(16, num_games)
    print(f"Spawning {num_processes} concurrent CPU workers for self-play generation...")
    with context.Pool(processes=num_processes) as pool:
        results = pool.map(play_one_game_worker, args)
        
    scores = [res[1] for res in results]
    median_score = np.median(scores) if scores else 0
    print(f"Self-Play Median Score: {median_score:.1f}, Max Score: {max(scores) if scores else 0}")
    top_quartile = np.percentile(scores, 75) if scores else 0
    max_score = max(scores) if scores else 0
    
    for history, final_score in results:
        multiplier = 1
        if final_score >= max_score and final_score > 0:
            multiplier = 5 # 5x representation for the absolute best game
        elif final_score >= top_quartile and final_score > 0:
            multiplier = 3 # 3x representation for the top 25% of games
        elif final_score >= median_score and final_score > 0:
            multiplier = 2 # 2x representation for above average games
            
        for _ in range(multiplier):
            for (feat, cleared_line, state_score) in history:
                # The target is the remaining score to be achieved FROM this state
                rem_score = max(0.0, float(final_score - state_score))
                buffer.push(feat, rem_score, cleared_line)
                
    return buffer, scores

def train(model, buffer, optimizer, scheduler, device, epochs=5, batch_size=32):
    model.train()
    
    dataloader = DataLoader(buffer, batch_size=batch_size, shuffle=True)
    
    # We use Binary Cross Entropy for the auxiliary Line Clear target since it's a probability [0, 1]
    bce_loss_fn = nn.BCELoss()
    
    for epoch in range(epochs):
        total_loss = 0
        value_loss_sum = 0
        line_clear_loss_sum = 0
        
        for states, targets_value, targets_line in dataloader:
            states = states.to(device)
            
            # Normalize target values to ~[0.0, 3.0] scale so MSE isn't astronomically overpowering
            targets_value = targets_value.to(device) / 100.0
            targets_line = targets_line.to(device)
            
            optimizer.zero_grad()
            
            # Predict
            pred_values, pred_line_clears = model(states)
            
            # MSE for the core Value expectation
            value_loss = F.mse_loss(pred_values, targets_value)
            
            # BCE for the explicit line clear event prediction
            line_clear_loss = bce_loss_fn(pred_line_clears, targets_line)
            
            # Total Loss Balancing (Prioritize Value, but guide strongly with Line mapping)
            loss = value_loss + (line_clear_loss * 0.5)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            value_loss_sum += value_loss.item()
            line_clear_loss_sum += line_clear_loss.item()
            
        scheduler.step()
            
        print(f"Epoch {epoch+1}/{epochs} | LR: {scheduler.get_last_lr()[0]:.6f} | "
              f"Total Loss: {total_loss:.4f} (Value MSE: {value_loss_sum:.4f}, Line Clear BCE: {line_clear_loss_sum:.4f})")
              
def main():
    torch.set_num_threads(1)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    print(f"Booting Next-State AlphaZero ecosystem on: {device}")
    
    # MAX POWER Scaling
    model = AlphaZeroNet(d_model=512, nhead=16, num_layers=16).to(device)
    # MPS does not support share_memory(), only CPU/CUDA. Removing to fix the crash.
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    # The scheduler decays the LR per *Epoch*. With 10-20 epochs per iteration, 
    # step_size=2 was collapsing the LR to near-zero before the first iteration finished.
    # We increase step_size to 15 to allow the network time to learn on the new architectures.
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.8)
    buffer = ReplayBuffer(capacity=250000)

    checkpoint = "models/best_model_python.pth"
    if os.path.exists(checkpoint):
        try:
            model.load_state_dict(torch.load(checkpoint, map_location=device))
            print("Loaded checkpoint.")
        except Exception as e:
            print(f"Failed to load checkpoint (likely architecture mismatch from upgrade): {e}")
            print("=> Training from Tabula Rasa.")
        
    ITERATIONS = 50
    for i in range(ITERATIONS):
        print(f"\n================ Iteration {i+1}/{ITERATIONS} ================")
        model.eval()
        
        start = time.time()
        
        # Generate experience with Next-State Evaluation via MCTS
        # MAX POWER Scaling: 128 concurrent games, 2400 MCTS sims, 256 batch
        buffer, scores = self_play(model, buffer, device, num_games=128, simulations=2400, batch_size=256) 
        print(f"Self-play generated {len(buffer)} states in {time.time() - start:.2f}s")
        
        if scores:
            # Metrics Tracking
            metrics_file = "models/metrics.json"
            metrics = {}
            if os.path.exists(metrics_file):
                with open(metrics_file, "r") as f:
                    try:
                        metrics = json.load(f)
                    except:
                        pass
            
            iter_key = f"iteration_{i+1}"
            best_score = max(scores)
            median_score = float(np.median(scores))
            metrics[iter_key] = {
                "best": best_score,
                "median": median_score,
                "distribution": scores
            }
            with open(metrics_file, "w") as f:
                json.dump(metrics, f, indent=2)
                
            # ASCII Histogram (Bell Curve)
            print("\n--- Score Distribution ---")
            bins = np.linspace(min(scores), max(scores) + 1, 10)
            hist, bin_edges = np.histogram(scores, bins=bins)
            max_count = max(hist) if len(hist) > 0 else 1
            for b in range(len(hist)):
                bar = "█" * int(20 * hist[b] / max_count)
                print(f"{bin_edges[b]:6.1f} - {bin_edges[b+1]:6.1f} | {bar} ({hist[b]})")
            print("--------------------------")
        
        if len(buffer) > 0:
            # Scale training batch size enormously to leverage RTX 3080 Ti 12GB
            train(model, buffer, optimizer, scheduler, device, epochs=10, batch_size=1024)
            
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), checkpoint)
            print("=> Saved SOTA PyTorch Model!")

if __name__ == "__main__":
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()
