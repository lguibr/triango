import triango_ext
from triango.mcts.features import extract_feature


def run_test():
    print("1", flush=True)
    state = triango_ext.GameState()
    print("2", flush=True)
    
    board_before = state.board
    print("3", flush=True)
    
    feat = extract_feature(state)
    print("4", flush=True)
    
    state.apply_move(0, 0)
    print("5", flush=True)

if __name__ == "__main__":
    run_test()
    print("Done")
