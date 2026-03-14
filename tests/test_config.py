from unittest.mock import patch

from triango.config import get_hardware_config


def test_get_hardware_config_cuda() -> None:
    with patch('torch.cuda.is_available', return_value=True):
        config = get_hardware_config()
        assert config['device'].type == 'cuda'

def test_get_hardware_config_mps() -> None:
    with patch('torch.cuda.is_available', return_value=False):
        with patch('torch.backends.mps.is_available', return_value=True):
            config = get_hardware_config()
            assert config['device'].type == 'mps'

def test_get_hardware_config_cpu() -> None:
    with patch('torch.cuda.is_available', return_value=False):
        with patch('torch.backends.mps.is_available', return_value=False):
            config = get_hardware_config()
            assert config['device'].type == 'cpu'
