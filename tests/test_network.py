import math

import torch

from triango.model.network import AlphaZeroNet


def test_geometric_embeddings_frozen() -> None:
    # Instantiate net
    net = AlphaZeroNet(d_model=64, nhead=4, num_layers=2)
    
    # The pos_emb should be a registered buffer, not a Parameter
    assert not isinstance(net.pos_emb, torch.nn.Parameter)
    assert not net.pos_emb.requires_grad
    
    # Shape should be [1, 96, d_model]
    assert net.pos_emb.shape == (1, 96, 64)

def test_geometric_embeddings_values() -> None:
    from triango.env.coords import INDEX_TO_COORD
    
    net = AlphaZeroNet(d_model=64)
    pe = net.pos_emb
    
    # Test origin
    if 0 in INDEX_TO_COORD:
        x, y, z = INDEX_TO_COORD[0] # (0, 0, 1) usually based on env code
        
        # Test the injected sine logic
        # For d_model=64, div_term[0] = 1.0
        # channel 0 should be sin(x)
        # channel 2 should be sin(y)
        # channel 4 should be sin(z)
        
        expected_sin_x = math.sin(x * 1.0)
        expected_sin_y = math.sin(y * 1.0)
        expected_sin_z = math.sin(z * 1.0)
        
        assert math.isclose(pe[0, 0, 0].item(), expected_sin_x, abs_tol=1e-5)
        assert math.isclose(pe[0, 0, 2].item(), expected_sin_y, abs_tol=1e-5)
        assert math.isclose(pe[0, 0, 4].item(), expected_sin_z, abs_tol=1e-5)

def test_transformer_forward_pass_with_new_features() -> None:
    from triango.mcts.features import extract_feature
    from triango_ext import GameState, initialize_env
    
    initialize_env()
    state = GameState()
    
    # Get feature map [7, 96]
    feat = extract_feature(state)
    
    # Ensure it unpacked bytes correctly without erroring
    assert feat.shape == (7, 96)
    assert not torch.isnan(feat).any()
    
    # Run through the network
    net = AlphaZeroNet(d_model=64, nhead=4, num_layers=2)
    batch = feat.unsqueeze(0) # [1, 7, 96]
    
    with torch.no_grad():
        val, policy = net(batch)
        
    assert val.shape == (1, 1)
    assert policy.shape == (1, 3, 50)
    
    # Policy outputs softmax
    sum_probs = policy.sum().item()
    # 3 slots, each with 50 orientations, sum of all probs = 1.0 because softmax is flattened across (3*50)=150
    # The model flattens [3, 50] to 150, softmaxes, and reshapes back to [3, 50]. Total sum should be 1.0.
    assert math.isclose(sum_probs, 1.0, abs_tol=1e-4)

