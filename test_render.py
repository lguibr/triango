import sys
import os

# Create a mock env to test render
# We just need the constants
ROW_LENGTHS = [9, 11, 13, 15, 15, 13, 11, 9]

def is_up(r: int, c: int) -> bool:
    if r < 4:
        return c % 2 == 0
    return c % 2 == 1

def render(board=0):
    max_len = 15
    for r in range(8):
        row_str = ""
        # The offset is max_len - row_len. For row 0 it's 6. So 3 spaces padding.
        pad = (max_len - ROW_LENGTHS[r]) // 2
        row_str += " " * pad
        for c in range(ROW_LENGTHS[r]):
            # mock filled = False
            filled = False
            if is_up(r, c):
                row_str += "▲" if filled else "△"
            else:
                row_str += "▼" if filled else "▽"
        print(row_str)

render()
