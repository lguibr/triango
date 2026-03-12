import torch
from triango.model.network import AlphaZeroNet

def test_network_forward():
    model = AlphaZeroNet(d_model=64, nhead=4, num_layers=2)
    # Batch size 2, 7 channels, 96 triangles
    dummy_input = torch.zeros(2, 7, 96)
    
    val, line_prob = model(dummy_input)
    assert val.shape == (2, 1)
    assert line_prob.shape == (2, 1)
    assert torch.all(line_prob >= 0.0) and torch.all(line_prob <= 1.0)
