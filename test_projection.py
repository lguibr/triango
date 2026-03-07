import sys
from env import TOTAL_TRIANGLES, ROW_LENGTHS, flat_index, get_row_col, is_up, is_up_flat, INDEX_TO_COORD, COORD_TO_INDEX, STANDARD_PIECES_DEFS

def get_preview_lines(p_id):
    p_def = STANDARD_PIECES_DEFS[p_id]
            
    origin_idx = -1
    for r in range(3, 5):
        for c in range(5, ROW_LENGTHS[r]-5):
            idx = flat_index(r, c)
            if p_def.require_up and is_up(r, c): origin_idx = idx; break
            if p_def.require_down and not is_up(r, c): origin_idx = idx; break
        if origin_idx != -1: break
            
    if origin_idx == -1: # Fallbacks for safety
        for i in range(TOTAL_TRIANGLES):
            if p_def.require_up and is_up_flat(i): origin_idx = i; break
            if p_def.require_down and not is_up_flat(i): origin_idx = i; break

    origin_x, origin_y, origin_z = INDEX_TO_COORD[origin_idx]
            
    visual_coords = []
    for off_x, off_y, off_z in p_def.offsets:
        tx, ty, tz = origin_x + off_x, origin_y + off_y, origin_z + off_z
        idx = COORD_TO_INDEX[(tx, ty, tz)]
        r, c = get_row_col(idx)
        pad = (15 - ROW_LENGTHS[r]) // 2
        visual_coords.append({'vx': c + pad, 'vy': r, 'is_up': is_up(r, c)})
                
    min_vx = min(vc['vx'] for vc in visual_coords)
    max_vx = max(vc['vx'] for vc in visual_coords)
    min_vy = min(vc['vy'] for vc in visual_coords)
    max_vy = max(vc['vy'] for vc in visual_coords)
            
    lines = []
    for y in range(max_vy - min_vy + 1):
        row_str = ""
        for x in range(max_vx - min_vx + 1):
            match = next((vc for vc in visual_coords if vc['vx'] - min_vx == x and vc['vy'] - min_vy == y), None)
            if match:
                row_str += "▲" if match['is_up'] else "▼"
            else:
                row_str += " "
        lines.append(row_str)
                
    return lines

for i in range(12):
    print(f"Piece {i}:")
    for l in get_preview_lines(i):
        print(l)
    print()
