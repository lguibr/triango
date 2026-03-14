import triango_ext
from triango.env.state import GameState as PyGameState

# Ensure the C++ backend is statically initialized once exactly
triango_ext.initialize_env()

def test_c_plus_plus_parity_initialization() -> None:
    """Verify that the C++ environment mounts and parses identical to Python."""
    py_state = PyGameState(pieces=[1, 5, 11], board=0, score=0)
    
    # Python integer '0' converted into a 96-length binary string
    cpp_state = triango_ext.GameState([1, 5, 11], py_state.board_bytes, 0)
    
    assert py_state.score == cpp_state.score
    assert py_state.pieces_left == cpp_state.pieces_left
    assert py_state.terminal == cpp_state.terminal
    import numpy as np
    assert np.array_equal(py_state.board_bytes, cpp_state.board_bytes)

def test_c_plus_plus_parity_placement() -> None:
    """Verify that C++ bitwise overlays evaluate precisely identically to Python."""
    py_state = PyGameState(pieces=[2, 4, 7], board=0, score=0)
    
    cpp_state = triango_ext.GameState([2, 4, 7], py_state.board_bytes, 0)
    
    # We will pick a valid placement manually for piece '2' at a specific index
    # We use trial and error mapping here simply to find a valid coordinate
    valid_index = -1
    from triango.env.pieces import STANDARD_PIECES
    for idx, mask in enumerate(STANDARD_PIECES[py_state.available[0]]):
        if mask != 0:
            valid_index = idx
            break
            
    assert valid_index != -1, "Could not find a valid piece mask"
    
    next_py = py_state.apply_move(0, valid_index)
    next_cpp = cpp_state.apply_move(0, valid_index)
    
    assert next_py is not None
    assert next_cpp is not None
    
    assert next_py.score == next_cpp.score
    assert next_py.pieces_left == next_cpp.pieces_left
    assert next_py.available == next_cpp.available
    import numpy as np
    assert np.array_equal(next_py.board_bytes, next_cpp.board_bytes)
    assert next_py.terminal == next_cpp.terminal
