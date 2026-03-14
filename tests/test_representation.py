from triango.env.constants import ROW_LENGTHS
from triango.env.coords import flat_index, is_up
from triango.env.state import GameState
from triango.mcts.features import extract_feature


def test_visualize_representation():
    """
    User requested an easy way to inspect and understand exactly what the 
    AI representation generates for debugging.
    """
    # Create an empty state but manually force 3 diverse pieces
    # Piece 2: Hexagonal block
    # Piece 8: Line block
    # Piece 10: Point down
    state = GameState(pieces=[2, 8, 10], board=0, score=0)
    
    # We add 1 random block to the board
    state.board |= (1 << 45) # Centerish
    
    feature = extract_feature(state)
    assert feature.shape == (7, 96)
    
    print("\n\n==== AI REPRESENTATION INSPECTOR ====")
    print("This is exactly what the Transformer 'sees' when it evaluates the board.")
    
    channels = [
        "Channel 0: Current Board Layout",
        "Channel 1: Slot 0 Piece Geometry Overlay",
        "Channel 2: Slot 0 Valid Placement Mask",
        "Channel 3: Slot 1 Piece Geometry Overlay",
        "Channel 4: Slot 1 Valid Placement Mask",
        "Channel 5: Slot 2 Piece Geometry Overlay",
        "Channel 6: Slot 2 Valid Placement Mask",
    ]
    
    for ch in range(7):
        print(f"\n>> {channels[ch]}")
        max_len = 15
        for r in range(8):
            row_str = ""
            pad = (max_len - ROW_LENGTHS[r]) // 2
            row_str += " " * pad
            for c in range(ROW_LENGTHS[r]):
                idx = flat_index(r, c)
                val = feature[ch, idx].item()
                if is_up(r, c):
                    row_str += "▲" if val > 0.5 else "△"
                else:
                    row_str += "▼" if val > 0.5 else "▽"
            print("  " + row_str)
            
    print("=====================================\n")
    
if __name__ == "__main__":
    test_visualize_representation()
