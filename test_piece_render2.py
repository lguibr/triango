import sys
from env import TOTAL_TRIANGLES, ROW_LENGTHS, flat_index, get_row_col, is_up, is_up_flat, INDEX_TO_COORD, COORD_TO_INDEX, STANDARD_PIECES_DEFS

def render_piece_test(p_id):
    p_def = STANDARD_PIECES_DEFS[p_id]
    
    # 1. Find a valid origin on the board where this piece could theoretically be placed
    # Prefer somewhere in the middle of the board so all offsets fit (e.g. row 4, which has len 15)
    origin_idx = -1
    for r in range(2, 6):
        for c in range(2, ROW_LENGTHS[r]-2):
            idx = flat_index(r, c)
            if p_def.require_up and is_up(r, c): 
                origin_idx = idx
                break
            if p_def.require_down and not is_up(r, c): 
                origin_idx = idx
                break
        if origin_idx != -1:
            break
            
    if origin_idx == -1:
        # Fallback
        for i in range(TOTAL_TRIANGLES):
            if p_def.require_up and is_up_flat(i): origin_idx = i; break
            if p_def.require_down and not is_up_flat(i): origin_idx = i; break

    origin_x, origin_y, origin_z = INDEX_TO_COORD[origin_idx]
    
    # 2. Map all offsets to board indices
    piece_indices = []
    for off_x, off_y, off_z in p_def.offsets:
        tx, ty, tz = origin_x + off_x, origin_y + off_y, origin_z + off_z
        if (tx, ty, tz) not in COORD_TO_INDEX:
            print(f"Piece {p_id} fell off the board with origin {origin_idx}. Finding another origin...")
            return False # Need better origin finding for test, but for the actual pieces, they are small enough
        piece_indices.append(COORD_TO_INDEX[(tx, ty, tz)])
        
    # 3. Convert indices to visual coordinates
    visual_coords = []
    max_len = 15
    for idx in piece_indices:
        r, c = get_row_col(idx)
        pad = (max_len - ROW_LENGTHS[r]) // 2
        visual_x = pad + c
        visual_y = r
        visual_coords.append({
            'vx': visual_x,
            'vy': visual_y,
            'is_up': is_up(r, c)
        })
        
    # 4. Find bounding box in visual space
    min_vx = min(vc['vx'] for vc in visual_coords)
    max_vx = max(vc['vx'] for vc in visual_coords)
    min_vy = min(vc['vy'] for vc in visual_coords)
    max_vy = max(vc['vy'] for vc in visual_coords)
    
    width = max_vx - min_vx + 1
    height = max_vy - min_vy + 1
    
    print(f"Piece {p_id}:")
    for y in range(height):
        row_str = ""
        for x in range(width):
            # Check if there is a triangle at this relative position
            match = next((vc for vc in visual_coords if vc['vx'] - min_vx == x and vc['vy'] - min_vy == y), None)
            if match:
                row_str += "▲" if match['is_up'] else "▼"
            else:
                row_str += " "
        print(row_str)
    print()
    return True

for i in range(len(STANDARD_PIECES_DEFS)):
    success = False
    while not success:
        success = render_piece_test(i)
