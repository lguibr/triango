import torch
from unittest.mock import patch, MagicMock
from triango.training.self_play import play_one_game, play_one_game_worker, self_play
from triango.env.state import GameState
from triango.mcts.search import PythonMCTS
from triango.training.buffer import ReplayBuffer
import numpy as np

def test_play_one_game():
    model = MagicMock()
    mcts = PythonMCTS(model, torch.device('cpu'), batch_size=2)
    
    with patch('triango.env.state.GameState.apply_move') as mock_apply:
        # Give it a safe dummy state with score to record history
        dummy_state = GameState()
        dummy_state.score = 50
        mock_apply.return_value = dummy_state
        
        with patch.object(PythonMCTS, 'search', side_effect=[
            ((0, 0), {(0, 0): 10, (0, 1): 5}), # Play 1 move
            (None, {}) # Terminate
        ]):
            history, score = play_one_game(0, mcts, 2, 1)
            assert len(history) >= 1
            assert score >= 0

def test_play_one_game_worker():
    hw_config = {
        'd_model': 16, 'nhead': 1, 'num_layers': 1,
        'simulations': 2, 'num_games': 1,
        'self_play_batch_size': 2,
        'worker_device': torch.device('cpu')
    }
    
    # Needs a valid state_dict
    from triango.model.network import AlphaZeroNet
    net = AlphaZeroNet(d_model=16, nhead=1, num_layers=1)
    
    with patch('triango.training.self_play.play_one_game') as mock_play:
        mock_play.return_value = ([], 0.0)
        res = play_one_game_worker((0, net.state_dict(), hw_config))
        assert res == ([], 0.0)

def test_self_play():
    model = MagicMock()
    # Mock state dict to bypass CPU tensor clone
    model.state_dict.return_value = {}
    
    buffer = ReplayBuffer(10)
    hw_config = {
        'device': torch.device('cpu'),
        'num_games': 2,
        'num_processes': 1,
        'worker_device': torch.device('cpu')
    }
    
    with patch('torch.multiprocessing.get_context') as mock_ctx:
        # Mock Pool
        mock_pool = MagicMock()
        mock_ctx.return_value.Pool.return_value.__enter__.return_value = mock_pool
        # Give mock results
        dummy_policy = torch.zeros(3, 50)
        mock_pool.map.return_value = [
            ([ (torch.zeros(7, 96), 0.0, 5.0, dummy_policy) ], 5.0),
            ([ (torch.zeros(7, 96), 0.0, 1.0, dummy_policy) ], 1.0),
        ]
        
        buf, scores = self_play(model, buffer, hw_config)
        assert len(scores) == 2
        assert len(buf) > 0
