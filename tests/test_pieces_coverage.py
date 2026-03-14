from triango.env.pieces import get_piece_overlay, get_valid_placement_mask


def test_piece_overlay_invalid():
    res = get_piece_overlay(-1)
    assert all(x == 0 for x in res)

def test_valid_placement_mask_invalid():
    res = get_valid_placement_mask(-1, 0)
    assert all(x == 0 for x in res)
