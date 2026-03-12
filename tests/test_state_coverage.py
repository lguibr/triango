from triango.env.state import GameState
from triango.env.pieces import STANDARD_PIECES_DEFS

def test_state_render(capsys):
    state = GameState(pieces=[0, 1, 2], score=42, board=1)
    # Just render it to cover the lines
    state.render()
    
    captured = capsys.readouterr()
    assert "Score: 42" in captured.out
    assert "Available Pieces:" in captured.out

def test_state_terminal_paths():
    # Force a totally filled board to test terminal
    state = GameState(pieces=[0, 0, 0])
    state.board = -1  # All bits 1 regardless of size
    state._check_terminal()
    assert state.terminal

    # Test pieces left = 0
    state2 = GameState(pieces=[-1, -1, -1])
    assert state2.terminal

def test_apply_move_invalid():
    state = GameState(pieces=[0, 1, 2], board=(1 << 96) - 1)
    # Board is full, piece 0 cannot be placed
    res = state.apply_move(0, 50)
    assert res is None
    
    # Piece -1 cannot be placed
    state2 = GameState(pieces=[-1, -1, -1])
    res2 = state2.apply_move(0, 0)
    assert res2 is None
