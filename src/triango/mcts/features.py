from typing import TYPE_CHECKING, Any, Union

import torch

from triango.env.constants import TOTAL_TRIANGLES
from triango.env.pieces import get_piece_overlay, get_valid_placement_mask
from triango.env.state import GameState

if TYPE_CHECKING:
    from triango_ext import GameState as CppGameState
else:
    CppGameState = Any

def extract_feature(state: Union[GameState, 'CppGameState', Any]) -> torch.Tensor:
    # Build a [7, 96] spatial tensor
    # Channel 0: Current Board
    # Channel 1: Geometry of piece in slot 0
    # Channel 2: Valid Mask of piece in slot 0
    # Channel 3: Geometry of piece in slot 1
    # Channel 4: Valid Mask of piece in slot 1
    # Channel 5: Geometry of piece in slot 2
    # Channel 6: Valid Mask of piece in slot 2

    feature = torch.zeros(7, 96, dtype=torch.float32)
    # Channel 0: Board - unpack the 12 bytes
    if hasattr(state, "board_bytes"):
        # C++ returns a 12-byte numpy array representing 96 bits.
        # unpacking gives [bit0, bit1, bit2..] matching the little/big endian structure.
        # Note: the C++ byte extraction shifts logically `(board.lo >> (i*8)) & 0xFF`,
        # meaning byte 0 is lowest bits. unpackbits gives MSB first.
        # To match EXACT structure without error, we do this natively in pure torch/numpy
        # Actually it's simpler and much faster to do bitwise against the bytes natively
        bytes_arr = state.board_bytes
        for byte_idx, b in enumerate(bytes_arr):
            for bit_idx in range(8):
                if (b & (1 << bit_idx)) != 0:
                    global_idx = byte_idx * 8 + bit_idx
                    if global_idx < 96:
                        feature[0, global_idx] = 1.0
    else:
        # Fallback for pure python states
        bin_str = bin(state.board)[2:].zfill(TOTAL_TRIANGLES)[::-1]
        for i in range(TOTAL_TRIANGLES):
            if i < len(bin_str) and bin_str[i] == "1":
                feature[0, i] = 1.0

    # Channels 1-6: Piece Overlays and Valid Masks
    for slot in range(3):
        p_id = state.available[slot]
        overlay = get_piece_overlay(p_id)
        valid_mask = get_valid_placement_mask(p_id, state.board)
        for i in range(TOTAL_TRIANGLES):
            if overlay[i] == 1:
                feature[(slot * 2) + 1, i] = 1.0
            if valid_mask[i] == 1:
                feature[(slot * 2) + 2, i] = 1.0

    return feature
