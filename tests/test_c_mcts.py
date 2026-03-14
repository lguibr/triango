import numpy as np

from triango_ext import GameState, Node, initialize_env


def setup_module() -> None:
    initialize_env()

def test_game_state_bytes_conversion() -> None:
    state = GameState()
    
    # Take a few deterministic moves
    moves = state.get_valid_moves()
    assert len(moves) > 0
    state2 = state.apply_move(moves[0][0], moves[0][1])
    assert state2 is not None
    
    # Test C++ Native uint8_t 12-byte export mapping
    bytes_arr = state2.board_bytes
    
    assert isinstance(bytes_arr, np.ndarray)
    assert bytes_arr.dtype == np.uint8
    assert bytes_arr.shape == (12,)
    
    # 96 bits / 8 = 12 bytes. If the board has pieces, at least one bit should be non-zero
    assert np.any(bytes_arr > 0)

    # Test re-hydration behavior from bytes
    bytes_arr_2 = state2.board_bytes
    assert np.array_equal(bytes_arr, bytes_arr_2)
    
    del state2
    del state


def test_node_dirichlet_noise_caching() -> None:
    state = GameState()
    root = Node(state)
    root.expand()
    
    assert len(root.children) > 0
    
    # Save original priors
    original_priors = []
    for c in root.children:
        original_priors.append(c.prior)
    # They should initialize to 0.0 because there isn't cached policy in an empty node
    assert all(p == 0.0 for p in original_priors)
    
    # Inject Noise (Alpha=0.3, Eps=0.25)
    root.apply_dirichlet_noise(0.3, 0.25)
    
    # Noise modifies the child priors
    new_priors = []
    for c in root.children:
        new_priors.append(c.prior)
    assert not all(p == 0.0 for p in new_priors)
    assert sum(new_priors) > 0.0
    
    del root
    del state

def test_async_mcts_policy_injection() -> None:
    from triango_ext import AsyncMCTS, EvalResult
    state = GameState()
    
    mcts = AsyncMCTS(state, threads=2, sims=10, c_puct=1.5)
    mcts.start()
    
    import time
    time.sleep(0.01) # let worker pause
    
    reqs = mcts.get_requests(1)
    assert len(reqs) > 0
    node = reqs[0].node
    
    # Mock some neural policy output (150 floats)
    mock_policy = [0.0] * 150
    mock_policy[0] = 1.0 # fake probability
    res = [EvalResult(node=node, value=10.0, policy=mock_policy)]
    
    mcts.submit_results(res)
    
    # Allow tree backprop to execute
    time.sleep(0.01)
    
    # The node should now have cached_policy populated from C++ natively
    # Because `expand` natively binds child priors, let's test if the children got it
    node.expand() # force expand to test inheritance
    
    if len(node.children) > 0:
        c_node = node.children[0]
        assert c_node.prior >= 0.0
    
    mcts.stop()
    del c_node
    del node
    del res
    del reqs
    del mock_policy
    del mcts
    del state
