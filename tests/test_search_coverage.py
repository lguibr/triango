import torch
import numpy as np
from triango.mcts.node import Node
from triango.mcts.search import PythonMCTS
from triango.env.state import GameState
from unittest.mock import patch, MagicMock

def test_puct():
    state = GameState()
    parent = Node(state)
    parent.visits = 10
    parent.value_sum = 5.0
    
    # Dummy child
    child = Node(state, parent=parent)
    child.visits = 2
    child.value_sum = 2.0
    child.prior = 0.5
    
    score = child.puct()
    assert score > 0

def test_select_child():
    state = GameState()
    node = Node(state)
    child1 = Node(state, parent=node)
    child1.visits = 10
    child1.value_sum = 10.0
    child1.prior = 0.5
    
    child2 = Node(state, parent=node)
    child2.visits = 1
    child2.value_sum = 0.0
    child2.prior = 0.5
    
    node.children = [child1, child2]
    best = node.select_child()
    assert best is not None

def test_mcts_search():
    # Mock model
    class MockModel(torch.nn.Module):
        def forward(self, x):
            batch_size = x.size(0)
            return torch.zeros(batch_size, 1), torch.zeros(batch_size, 1), torch.zeros(batch_size, 3, 50)
            
    model = MockModel()
    mcts = PythonMCTS(model, torch.device('cpu'), batch_size=2)
    
    state = GameState()
    
    # Small search
    best_move, visits = mcts.search(state, simulations=4)
    assert best_move is not None
    assert len(visits) > 0

    # Test dirichlet
    node = Node(state)
    node.expand()
    node.expand()
    mcts.add_dirichlet_noise(node)

