import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import time
import os

from env import GameState, TOTAL_TRIANGLES
from mcts import PythonMCTS, extract_feature
from model import AlphaZeroNet

class ReplayBuffer(Dataset):
    def __init__(self, capacity=10000):
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


def self_play(mcts, num_games=10, simulations=100):
    buffer = ReplayBuffer()
    device = mcts.device
    
    for game in range(num_games):
        state = GameState()
        game_history = []
        
        print(f"--- Game {game+1}/{num_games} Started ---")
        
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
                
            # Store state snapshot for replay buffer
            # Note: For strict identicality to AZ, we should store all symmetries (Rotate120/Rotate240) here. 
            # Skipping augmentation for pure simplicity in V1 Python port.
            feat = extract_feature(state)
            
            # Send feature numpy array equivalent directly back to MPS device
            game_history.append((feat.clone().detach(), policy))
            
            # Apply move
            slot, idx = best_move
            state = state.apply_move(slot, idx)
            step += 1
            
        print(f"Game {game+1} Finished. Steps: {step}, Final Score: {state.score}")
        
        # Retroactively assign the final score as the Value to all states in the history
        # (Standard AlphaZero often uses +1/-1, but Triango uses pure Score)
        for (feat, policy) in game_history:
            buffer.push(feat, policy, float(state.score))
            
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
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Booting pure Python AlphaZero ecosystem on: {device}")
    
    model = AlphaZeroNet().to(device)
    
    # Optional load
    checkpoint = "models/best_model_python.pth"
    if os.path.exists(checkpoint):
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        print("Loaded checkpoint.")
        
    mcts = PythonMCTS(model, device)
    
    ITERATIONS = 50
    for i in range(ITERATIONS):
        print(f"\n================ Iteration {i+1}/{ITERATIONS} ================")
        model.eval()
        
        start = time.time()
        buffer = self_play(mcts, num_games=5, simulations=50) # Extremely fast Python logic
        print(f"Self-play generated {len(buffer)} states in {time.time() - start:.2f}s")
        
        train(model, buffer, device, epochs=5, batch_size=32)
        
        # Save SOTA
        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), checkpoint)
        print("=> Saved SOTA PyTorch Model!")

if __name__ == "__main__":
    main()
