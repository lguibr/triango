import torch

from triango.model.network import AlphaZeroNet


def test_network_forward():
    model = AlphaZeroNet(d_model=64, nhead=4, num_layers=2)
    # Batch size 2, 7 channels, 96 triangles
    dummy_input = torch.zeros(2, 7, 96)
    
    val, policy_prob = model(dummy_input)
    assert val.shape == (2, 1)
    assert policy_prob.shape == (2, 3, 50)
    
    # Verify policy probs sum to 1 over the last dimension logically (viewed as flattened)
    assert torch.allclose(policy_prob.sum(dim=(1,2)), torch.ones(2))
