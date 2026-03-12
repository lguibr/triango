import sys
from triango.env.pieces import STANDARD_PIECES_DEFS, STANDARD_PIECES, get_piece_overlay, get_valid_placement_mask
from triango.env.state import GameState
from triango.env.constants import ROW_LENGTHS
from triango.env.coords import is_up, flat_index
from triango.mcts.features import extract_feature

def visualize_all_shapes_and_features():
    output_lines = []
    output_lines.append("=========================================================")
    output_lines.append("      TRIANGO AI REPRESENTATION VISUALIZER: ALL SHAPES   ")
    output_lines.append("=========================================================\n")
    
    output_lines.append("The system checks every possible orientation and shape interaction.")
    output_lines.append("For each of the 12 shapes, we attempt to find the first valid placement")
    output_lines.append("near the center of the board, and dump exactly how the Neural Network")
    output_lines.append("interprets this feature structurally across its 7 Channels.\n")
    
    for p_id in range(12):
        output_lines.append(f"\n#########################################################")
        output_lines.append(f"                SHAPE ID {p_id}")
        output_lines.append(f"#########################################################")
        
        state = GameState(pieces=[p_id, -1, -1], board=0, score=0)
        
        masks = STANDARD_PIECES[p_id]
        chosen_idx = -1
        
        for cand_idx in range(40, 70):
            if masks[cand_idx] != 0:
                chosen_idx = cand_idx
                break
                
        if chosen_idx == -1:
            for cand_idx in range(len(masks)):
                if masks[cand_idx] != 0:
                    chosen_idx = cand_idx
                    break
                    
        if chosen_idx == -1:
            output_lines.append(f"ERROR: No valid placement found for Piece {p_id}! Something is fundamentally wrong.")
            continue
            
        output_lines.append(f"--- Placing Shape {p_id} at target index {chosen_idx} ---")
        
        next_state = state.apply_move(0, chosen_idx)
        if next_state is None:
            output_lines.append("Failed to apply valid move.")
            continue
            
        feature = extract_feature(next_state)
        pre_feat = extract_feature(state)
        
        output_lines.append(f"\n--- 1. PRE-ACTION KNOWLEDGE (Before placing Shape {p_id}) ---")
        output_lines.append(">> The Shape's Natural Geometry Overlay (What the shape looks like)")
        output_lines.extend(render_channel(pre_feat[1]))
        
        output_lines.append("\n>> The Action Mask (Legal Placements on the current empty board)")
        output_lines.append("This teaches us where this shape can physically fit.")
        output_lines.extend(render_channel(pre_feat[2]))

        output_lines.append(f"\n--- 2. ACTION TAKEN (We place Shape {p_id} at index {chosen_idx}) ---")
        output_lines.append("We now generate the NEXT-STATE. *This* is what the AlphaZero Transformer actually evaluates!\n")
        
        output_lines.append(">> NEXT-STATE CHANNEL 0: The New Board")
        output_lines.append("Notice the shape is now embedded into the board structure.")
        output_lines.extend(render_channel(feature[0]))

        output_lines.append(f"\n>> NEXT-STATE CHANNEL 1 & 2: Slot 0 is now Piece {next_state.available[0]}")
        output_lines.append("The Valid Placements mask (Channel 2) updates dynamically because the board changed!")
        output_lines.append("Channel 1 (Geometry):")
        output_lines.extend(render_channel(feature[1]))
        output_lines.append("Channel 2 (Valid Placements Action Mask):")
        output_lines.extend(render_channel(feature[2]))
        
        output_lines.append("\n" + "="*57)

    with open(r"C:\Users\lgui_\.gemini\antigravity\brain\df6a4141-bc22-4081-bb44-a9852c20aff5\feature_representation_guide.md", "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))
        
def render_channel(tensor_channel):
    lines = []
    max_len = 15
    for r in range(8):
        row_str = ""
        pad = (max_len - ROW_LENGTHS[r]) // 2
        row_str += " " * pad
        for c in range(ROW_LENGTHS[r]):
            idx = flat_index(r, c)
            val = tensor_channel[idx].item()
            if is_up(r, c):
                row_str += "▲" if val > 0.5 else "△"
            else:
                row_str += "▼" if val > 0.5 else "▽"
        lines.append("  " + row_str)
    return lines

if __name__ == "__main__":
    visualize_all_shapes_and_features()
