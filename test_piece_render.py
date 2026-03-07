import sys

# Test script to verify the piece rendering UI
def render_piece_test(req_up, offsets):
    # Find bounding box
    min_y = min(off[1] for off in offsets)
    max_y = max(off[1] for off in offsets)
    min_x = min(off[0] for off in offsets)
    max_x = max(off[0] for off in offsets)
    
    # center offsets around 0,0 locally
    local_coords = [(off[0]-min_x, off[1]-min_y) for off in offsets]
    
    print(f"req_up: {req_up}, offsets: {offsets}")
    
    for y_loc in range(max_y - min_y + 1):
        row_str = ""
        for x_loc in range(max_x - min_x + 1):
            if (x_loc, y_loc) in local_coords:
                # Find the actual original coordinate
                orig = next(o for o in offsets if o[0]-min_x == x_loc and o[1]-min_y == y_loc)
                
                # In hex grid with coordinates x, y, z:
                # Origin piece has orientation `req_up`.
                # Distance/parity dictates the orientation of neighbors.
                # A move in x (same row, next triangle) flips orientation.
                # A move in z (same row, next triangle) flips orientation.
                # A move in y (moving up/down a row) flips orientation.
                # Total distance = abs(orig[0]) + abs(orig[1]) + abs(orig[2])
                
                dist = abs(orig[0]) + abs(orig[1]) + abs(orig[2])
                
                # If distance is odd, it's flipped from the origin.
                # Actually, let's test purely based on parity of sum(coords)
                coord_sum = orig[0] + orig[1] + orig[2]
                
                is_piece_up = req_up
                if coord_sum % 2 != 0:
                    is_piece_up = not is_piece_up
                    
                row_str += "▲" if is_piece_up else "▼"
            else:
                row_str += " "
        print(row_str)
    print()

STANDARD_PIECES_DEFS = [
    (True, [(0,0,0)]),
    (False, [(0,0,0)]),
    (True, [(0,0,0), (0,0,-1), (1,0,-1), (0,-1,0), (1,-1,0), (1,-1,-1)]),
    (True, [(0,0,0), (0,0,-1), (0,-1,0), (1,-1,0), (1,-1,-1)]),
    (True, [(0,0,0), (0,0,-1), (1,0,-1), (0,-1,0)]),
    (True, [(0,0,0), (0,-1,1), (0,-1,0), (1,-1,0)]),
    (False, [(0,0,0), (1,0,0), (1,0,-1), (1,-1,0)]),
    (True, [(0,0,0), (0,0,-1), (1,0,-1)]),
    (False, [(0,0,0), (1,0,0), (1,0,-1)]),
    (True, [(0,0,0), (0,0,-1)]),
    (True, [(0,0,0), (0,-1,0)]),
    (True, [(0,0,0), (0,-1,1), (0,-1,0)]),
]

for p in STANDARD_PIECES_DEFS:
    render_piece_test(p[0], p[1])
