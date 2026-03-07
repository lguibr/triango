import torch
import numpy as np
import random

TOTAL_TRIANGLES = 96
ROW_LENGTHS = [9, 11, 13, 15, 15, 13, 11, 9]

def flat_index(r: int, c: int) -> int:
    idx = 0
    for i in range(r):
        idx += ROW_LENGTHS[i]
    return idx + c

def get_row_col(idx: int):
    rem = idx
    for r in range(8):
        if rem < ROW_LENGTHS[r]:
            return r, rem
        rem -= ROW_LENGTHS[r]
    return -1, -1

def is_up(r: int, c: int) -> bool:
    if r < 4:
        return c % 2 == 0
    return c % 2 == 1

def is_up_flat(idx: int) -> bool:
    r, c = get_row_col(idx)
    return is_up(r, c)

def vertical_neighbor(r: int, c: int):
    if is_up(r, c):
        if r == 7: return -1, -1
        if r < 3: return r + 1, c + 1
        elif r == 3: return r + 1, c
        else: return r + 1, c - 1
    else:
        if r == 0: return -1, -1
        if r < 4: return r - 1, c - 1
        elif r == 4: return r - 1, c
        else: return r - 1, c + 1

def generate_masks():
    masks = []
    
    # 1. Horizontal
    for r in range(8):
        m = 0
        for c in range(ROW_LENGTHS[r]):
            m |= (1 << flat_index(r, c))
        masks.append(m)

    def extract_lines(next_fn):
        visited = [False] * TOTAL_TRIANGLES
        lines = []
        for r in range(8):
            for c in range(ROW_LENGTHS[r]):
                idx = flat_index(r, c)
                if visited[idx]:
                    continue
                
                # Traverse backward to find absolute start
                start_r, start_c = r, c
                while True:
                    prev_r, prev_c = -1, -1
                    for pr in range(8):
                        for pc in range(ROW_LENGTHS[pr]):
                            nr, nc = next_fn(pr, pc)
                            if nr == start_r and nc == start_c:
                                prev_r, prev_c = pr, pc
                    if prev_r == -1:
                        break
                    start_r, start_c = prev_r, prev_c
                
                # Traverse forward
                m = 0
                curr_r, curr_c = start_r, start_c
                length = 0
                while curr_r != -1 and curr_c != -1:
                    i = flat_index(curr_r, curr_c)
                    m |= (1 << i)
                    visited[i] = True
                    curr_r, curr_c = next_fn(curr_r, curr_c)
                    length += 1
                lines.append(m)
        return lines

    def next_red(r, c):
        if is_up(r, c):
            if c + 1 < ROW_LENGTHS[r]: return r, c + 1
        else:
            nr, nc = vertical_neighbor(r, c)
            if nr != -1: return nr, nc
        return -1, -1

    def next_black(r, c):
        if is_up(r, c):
            if c - 1 >= 0: return r, c - 1
        else:
            nr, nc = vertical_neighbor(r, c)
            if nr != -1: return nr, nc
        return -1, -1

    masks.extend(extract_lines(next_red))
    masks.extend(extract_lines(next_black))
    assert len(masks) == 24
    return masks

ALL_MASKS = generate_masks()

# Coordinates Logic
INDEX_TO_COORD = {}
COORD_TO_INDEX = {}

def build_coords():
    visited = [False] * TOTAL_TRIANGLES
    queue = [0]
    INDEX_TO_COORD[0] = (0, 0, 1)
    visited[0] = True
    
    while queue:
        curr = queue.pop(0)
        x, y, z = INDEX_TO_COORD[curr]
        r, c = get_row_col(curr)
        
        def assign(n_idx, nx, ny, nz):
            if not visited[n_idx]:
                INDEX_TO_COORD[n_idx] = (nx, ny, nz)
                visited[n_idx] = True
                queue.append(n_idx)

        if is_up(r, c):
            if c + 1 < ROW_LENGTHS[r]: assign(flat_index(r, c + 1), x, y, z - 1)
            if c - 1 >= 0: assign(flat_index(r, c - 1), x - 1, y, z)
            nr, nc = vertical_neighbor(r, c)
            if nr != -1: assign(flat_index(nr, nc), x, y - 1, z)
        else:
            if c + 1 < ROW_LENGTHS[r]: assign(flat_index(r, c + 1), x + 1, y, z)
            if c - 1 >= 0: assign(flat_index(r, c - 1), x, y, z + 1)
            nr, nc = vertical_neighbor(r, c)
            if nr != -1: assign(flat_index(nr, nc), x, y + 1, z)
            
    for i in range(TOTAL_TRIANGLES):
        COORD_TO_INDEX[INDEX_TO_COORD[i]] = i

build_coords()

class PieceDef:
    def __init__(self, req_up, req_down, offsets):
        self.require_up = req_up
        self.require_down = req_down
        self.offsets = offsets

# Precompile pieces into array of masks
def compile_pieces(defs):
    pieces = []
    for def_obj in defs:
        masks = []
        for i in range(TOTAL_TRIANGLES):
            is_up_idx = is_up_flat(i)
            if def_obj.require_up and not is_up_idx:
                masks.append(0)
                continue
            if def_obj.require_down and is_up_idx:
                masks.append(0)
                continue
            
            origin_x, origin_y, origin_z = INDEX_TO_COORD[i]
            m = 0
            valid = True
            for off_x, off_y, off_z in def_obj.offsets:
                tx, ty, tz = origin_x + off_x, origin_y + off_y, origin_z + off_z
                if (tx, ty, tz) not in COORD_TO_INDEX:
                    valid = False
                    break
                m |= (1 << COORD_TO_INDEX[(tx, ty, tz)])
            
            masks.append(m if valid else 0)
        pieces.append(masks)
    return pieces

STANDARD_PIECES_DEFS = [
    PieceDef(True, False, [(0,0,0)]),
    PieceDef(False, True, [(0,0,0)]),
    PieceDef(True, False, [(0,0,0), (0,0,-1), (1,0,-1), (0,-1,0), (1,-1,0), (1,-1,-1)]),
    PieceDef(True, False, [(0,0,0), (0,0,-1), (0,-1,0), (1,-1,0), (1,-1,-1)]),
    PieceDef(True, False, [(0,0,0), (0,0,-1), (1,0,-1), (0,-1,0)]),
    PieceDef(True, False, [(0,0,0), (0,-1,1), (0,-1,0), (1,-1,0)]),
    PieceDef(False, True, [(0,0,0), (1,0,0), (1,0,-1), (1,-1,0)]),
    PieceDef(True, False, [(0,0,0), (0,0,-1), (1,0,-1)]),
    PieceDef(False, True, [(0,0,0), (1,0,0), (1,0,-1)]),
    PieceDef(True, False, [(0,0,0), (0,0,-1)]),
    PieceDef(True, False, [(0,0,0), (0,-1,0)]),
    PieceDef(True, False, [(0,0,0), (0,-1,1), (0,-1,0)]),
]

STANDARD_PIECES = compile_pieces(STANDARD_PIECES_DEFS)

class GameState:
    __slots__ = ['board', 'score', 'available', 'pieces_left', 'terminal']
    
    def __init__(self, pieces=None, board=0, score=0):
        self.board = board
        self.score = score
        if pieces is None:
            pieces = [random.randint(0, 11) for _ in range(3)]
        self.available = list(pieces)
        self.pieces_left = sum(1 for p in self.available if p != -1)
        self.terminal = False
        self._check_terminal()

    def _check_terminal(self):
        if self.pieces_left > 0:
            has_move = False
            for p_id in self.available:
                if p_id == -1: continue
                for m in STANDARD_PIECES[p_id]:
                    if m != 0 and (self.board & m) == 0:
                        has_move = True
                        break
                if has_move: break
            self.terminal = not has_move
            
    def apply_move(self, slot: int, index: int):
        p_id = self.available[slot]
        if p_id == -1: return None
        
        mask = STANDARD_PIECES[p_id][index]
        if mask == 0 or (self.board & mask) != 0:
            return None # invalid move
            
        next_state = GameState(self.available, self.board, self.score)
        next_state.available[slot] = -1
        next_state.pieces_left -= 1
        
        # Place piece
        next_state.board |= mask
        next_state.score += bin(mask).count('1')
        
        # Line clears
        cleared_mask = 0
        lines_cleared = 0
        for line in ALL_MASKS:
            if (next_state.board & line) == line:
                cleared_mask |= line
                lines_cleared += 1
                
        if lines_cleared > 0:
            next_state.board &= ~cleared_mask
            next_state.score += bin(cleared_mask).count('1') * 2
            
        next_state._check_terminal()
        return next_state
        
    def refill_tray(self):
        self.available = [random.randint(0, 11) for _ in range(3)]
        self.pieces_left = 3
        self._check_terminal()
