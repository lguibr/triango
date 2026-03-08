import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import time
import os
import torch.multiprocessing as mp

from env import GameState, TOTAL_TRIANGLES
from mcts import PythonMCTS, extract_feature
from model import AlphaZeroNet

class ReplayBuffer(Dataset):
    def __init__(self, capacity=50000):
        self.capacity = capacity
        self.buffer = []
        
    def __len__(self):
        return len(self.buffer)
        
    def __getitem__(self, idx):
        # state is [16, 96], policy is [96], value is scalar
        state, policy, value = self.buffer[idx]
        return state, torch.tensor(policy, dtype=torch.float32), torch.tensor([value], dtype=torch.float32)
        
    def push(self, state_tensor, policy_array, value):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append((state_tensor, policy_array, value))


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
                
        best_move, visits = mcts.search(state, simulations=simulations)
        
        # Policy Target Distribution
        policy = np.zeros(TOTAL_TRIANGLES, dtype=np.float32)
        total_visits = sum(visits.values())
        for move, v in visits.items():
            slot, idx = move
            policy[idx] += v / total_visits
            
        feat = extract_feature(state)
        game_history.append((feat.clone().detach(), policy))
        
        # Temperature Exploration (tau) for more diverse experience
        if step < 15:
            moves = list(visits.keys())
            probs = np.array([visits[m] for m in moves], dtype=np.float64)
            probs = probs / np.sum(probs)
            chosen_idx = np.random.choice(len(moves), p=probs)
            chosen_move = moves[chosen_idx]
        else:
            chosen_move = best_move
            
        # Apply move
        slot, idx = chosen_move
        state = state.apply_move(slot, idx)
        
        # Only render Game 1 to prevent interleaved multithreading terminal spam
        if game_idx == 0:
            state.render()
            
        step += 1
        
    print(f"Game {game_idx+1} Finished. Steps: {step}, Final Score: {state.score}")
    return game_history, state.score

def play_one_game_worker(args):
    game_idx, model, device, simulations, num_games, batch_size = args
    mcts = PythonMCTS(model, device, batch_size=batch_size)
    return play_one_game(game_idx, mcts, simulations, num_games)

def self_play(model, device, num_games=10, simulations=100, batch_size=32):
    buffer = ReplayBuffer()
    
    # Needs mp spawn context for sharing CUDA models
    context = mp.get_context('spawn')
    
    args = [(i, model, device, simulations, num_games, batch_size) for i in range(num_games)]
    
    # Multi-processing Self-Play utilizing all CPU cores natively
    results = []
    # Windows has a hard limit of 63 handles for WaitForMultipleObjects in multiprocessing.Pool
    # We cap the active workers at 48 to avoid starving Windows OS threads.
    with context.Pool(processes=min(16, num_games)) as pool:
        results = pool.map(play_one_game_worker, args)
        
    scores = [res[1] for res in results]
    median_score = np.median(scores) if scores else 0
    print(f"Self-Play Median Score: {median_score:.1f}, Max Score: {max(scores) if scores else 0}")
    
    for history, final_score in results:
        # Better Experience Selection: Double weight high-scoring games
        multiplier = 2 if final_score > median_score and final_score > 0 else 1
        for _ in range(multiplier):
            for (feat, policy) in history:
                buffer.push(feat, policy, float(final_score))
                
    return buffer

def train(model, buffer, device, epochs=5, batch_size=32):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.8)
    
    dataloader = DataLoader(buffer, batch_size=batch_size, shuffle=True)
    
    for epoch in range(epochs):
        total_loss = 0
        policy_loss_sum = 0
        value_loss_sum = 0
        
        for states, policies, values in dataloader:
            states = states.to(device)
            # Add small epsilon to target policies to prevent log(0) in KL Div
            policies = policies.to(device) + 1e-8
            policies = policies / policies.sum(dim=1, keepdim=True)
            
            # Normalize target values to ~[0.0, 3.0] scale so MSE isn't astronomically overpowering
            values_norm = values.to(device) / 100.0
            
            optimizer.zero_grad()
            
            # Predict
            pred_policies, pred_values = model(states)
            
            # Loss Function
            policy_loss = F.kl_div(pred_policies, policies, reduction='batchmean')
            
            # Value loss: We amplify it slightly to match the Policy gradient magnitude
            value_loss = F.mse_loss(pred_values, values_norm)
            
            # Total Loss Balancing
            loss = policy_loss + (value_loss * 0.5)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            policy_loss_sum += policy_loss.item()
            value_loss_sum += value_loss.item()
            
        scheduler.step()
            
        print(f"Epoch {epoch+1}/{epochs} | LR: {scheduler.get_last_lr()[0]:.6f} | "
              f"Total Loss: {total_loss:.4f} (Policy: {policy_loss_sum:.4f}, Value: {value_loss_sum:.4f})")
              
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    print(f"Booting pure Python AlphaZero ecosystem on: {device}")
    
    model = AlphaZeroNet().to(device)
    model.share_memory() # Crucial for multi-processing CUDA sharing
    
    # Optional load
    checkpoint = "models/best_model_python.pth"
    if os.path.exists(checkpoint):
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        print("Loaded checkpoint.")
        
    ITERATIONS = 50
    for i in range(ITERATIONS):
        print(f"\n================ Iteration {i+1}/{ITERATIONS} ================")
        model.eval()
        
        start = time.time()
        
        # Increased games and multithreaded tree searching. 
        # Using native GPU batch_size per agent = 16 (lowered to save RAM).
        buffer = self_play(model, device, num_games=16, simulations=200, batch_size=64) 
        print(f"Self-play generated {len(buffer)} states in {time.time() - start:.2f}s")
        
        # Larger batch size and epochs for faster, deeper learning
        train(model, buffer, device, epochs=10, batch_size=128)
        
        # Save SOTA
        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), checkpoint)
        print("=> Saved SOTA PyTorch Model!")

if __name__ == "__main__":
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()
