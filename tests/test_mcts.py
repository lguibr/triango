from triango.env.state import GameState
from triango.mcts.node import Node


def test_node_initialization():
    state = GameState()
    node = Node(state)
    assert node.visits == 0
    assert node.value_sum == 0.0
    if not state.terminal:
        assert len(node.untried) > 0
        assert not node.expanded
        
def test_node_expansion():
    state = GameState(pieces=[0, -1, -1])  # 1 small piece
    node = Node(state)
    
    # Expand one
    if not node.expanded:
        child = node.expand()
        assert child.parent == node
        assert len(node.children) == 1
        
def test_node_backprop():
    state = GameState()
    root = Node(state)
    root.backpropagate(1.5)
    
    assert root.visits == 1
    assert root.value_sum == 1.5
