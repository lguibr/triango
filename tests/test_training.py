import torch
import torch.optim as optim
from triango.training.buffer import ReplayBuffer
from triango.model.network import AlphaZeroNet
from triango.training.self_play import self_play
from triango.training.trainer import train

def test_buffer():
    buf = ReplayBuffer(capacity=10, elite_ratio=0.5)
    
    # 5 standard capacity, 5 elite capacity.
    dummy_policy = torch.zeros(3, 50)
    for i in range(15):
        # Push standard games (score 0 doesn't trigger elite)
        buf.push_game([(torch.zeros(7, 96), 1.0, 0.0, dummy_policy)], 0.0)
        
    for i in range(6):
        # Push elite games
        buf.push_game([(torch.zeros(7, 96), 1.0, 100.0, dummy_policy)], 100.0)
        
    assert len(buf) == 10 # 5 standard + 5 elite
    state, val, lc, p = buf[0]
    assert state.shape == (7, 96)
    assert p.shape == (3, 50)

def test_training_loop():
    model = AlphaZeroNet(d_model=64, nhead=2, num_layers=2)
    buffer = ReplayBuffer(capacity=100)
    
    # Push dummy data
    dummy_policy = torch.zeros(3, 50)
    for _ in range(5):
        buffer.push_game([(torch.zeros(7, 96), 1.0, 10.0, dummy_policy)], 10.0)
        
    hw_config = {
        'device': torch.device('cpu'),
        'train_epochs': 1,
        'train_batch_size': 2,
    }
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1)
    
    train(model, buffer, optimizer, scheduler, hw_config)
    # If no exception, it passed.

def test_self_play_integration():
    model = AlphaZeroNet(d_model=32, nhead=2, num_layers=1)
    buffer = ReplayBuffer(capacity=100)
    
    hw_config = {
        'num_games': 1,
        'simulations': 2, 
        'device': torch.device('cpu'),
        'worker_device': torch.device('cpu'),
        'num_processes': 1,
        'self_play_batch_size': 2,
        'd_model': 32,
        'nhead': 2,
        'num_layers': 1
    }
    
    buffer, scores = self_play(model, buffer, hw_config)
    assert len(scores) == 1
    assert len(buffer) > 0
