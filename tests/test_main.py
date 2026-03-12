import os
from unittest.mock import patch, MagicMock
from triango.main import main

@patch('triango.main.self_play')
@patch('triango.main.train')
@patch('torch.load')
@patch('os.path.exists')
def test_main_execution(mock_exists, mock_load, mock_train, mock_self_play):
    mock_exists.return_value = False
    
    from triango.training.buffer import ReplayBuffer
    dummy_buffer = ReplayBuffer(10)
    import torch
    dummy_buffer.push(torch.zeros(7, 96), 1.0, 1.0)
    
    mock_self_play.side_effect = [(dummy_buffer, [10.0, 15.0]), Exception("Break Loop")]
    
    with patch('triango.main.get_hardware_config') as mock_hw:
        mock_hw.return_value = {
            'device': torch.device('cpu'),
            'd_model': 16,
            'nhead': 1,
            'num_layers': 1,
            'capacity': 100,
            'model_checkpoint': 'dummy.pth',
            'metrics_file': 'dummy.json',
        }
            
        try:
            main()
        except Exception as e:
            if str(e) != "Break Loop":
                raise e

@patch('triango.main.self_play')
@patch('triango.main.train')
def test_main_checkpoint(mock_train, mock_self_play):
    with patch('triango.main.get_hardware_config') as mock_hw:
        import torch
        from triango.training.buffer import ReplayBuffer
        
        # Raise an exception on the second call to break the loop early 
        mock_self_play.side_effect = [(ReplayBuffer(1), [1.0]), Exception("Break Loop")]
        
        mock_hw.return_value = {
            'device': torch.device('cpu'),
            'd_model': 16,
            'nhead': 1,
            'num_layers': 1,
            'capacity': 100,
            'model_checkpoint': 'dummy.pth',
            'metrics_file': 'dummy.json',
        }
        with patch('os.path.exists', return_value=True):
            with patch('torch.load') as mock_load:
                try:
                    main()
                except Exception as e:
                    if str(e) != "Break Loop":
                        raise e
                mock_load.assert_called_once()
