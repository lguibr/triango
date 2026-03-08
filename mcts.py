import math
import random
import torch
import numpy as np
import threading
from queue import Queue, Empty

from env import GameState, STANDARD_PIECES, TOTAL_TRIANGLES

class Node:
    __slots__ = ['state', 'move', 'visits', 'value_sum', 'prior', 'parent', 'children', 'untried', 'expanded', 'virtual_loss']
    
    def __init__(self, state: GameState, parent=None, move=None):
        self.state = state
        self.move = move
        self.visits = 0
        self.value_sum = 0.0
        self.prior = 0.0
        self.parent = parent
        self.children = []
        self.untried = []
        self.expanded = False
        self.virtual_loss = 0
        
        if not self.state.terminal:
            for slot in range(3):
                p_id = self.state.available[slot]
                if p_id == -1: continue
                # We reverse iter so pop() is O(1) from the end
                for idx in range(TOTAL_TRIANGLES - 1, -1, -1):
                    m = STANDARD_PIECES[p_id][idx]
                    if m != 0 and (self.state.board & m) == 0:
                        self.untried.append((slot, idx))
                        
        if len(self.untried) == 0:
            self.expanded = True

    def puct(self, c_puct=1.25):
        exploit = (self.value_sum - self.virtual_loss) / (self.visits + self.virtual_loss) if (self.visits + self.virtual_loss) > 0 else 0.0
        parent_visits = self.parent.visits + self.parent.virtual_loss if self.parent else 1
        explore = c_puct * self.prior * math.sqrt(parent_visits) / (1.0 + self.visits + self.virtual_loss)
        return exploit + explore
        
    def select_child(self):
        best_child = None
        best_score = -float('inf')
        for child in self.children:
            score = child.puct()
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def expand(self):
        if len(self.untried) == 0:
            return self
            
        slot, idx = self.untried.pop()
        next_state = self.state.apply_move(slot, idx)
        
        if len(self.untried) == 0:
            self.expanded = True
            
        child = Node(next_state, self, (slot, idx))
        self.children.append(child)
        return child
        
    def backpropagate(self, reward):
        curr = self
        while curr is not None:
            curr.visits += 1
            curr.value_sum += reward
            curr = curr.parent

            
def extract_feature(state: GameState) -> torch.Tensor:
    # Build 1 slice of the Tensor spanning right now (T=0). 
    # For full AlphaZero we would track history, but for simplicity of this pure port, let's map [4, 96]
    # Where [0, :] is current board, and [1/2/3, :3] is tray
    
    feature = torch.zeros(16, 96, dtype=torch.float32)
    # Binary unpacking of integer bitboard
    bin_str = bin(state.board)[2:].zfill(TOTAL_TRIANGLES)[::-1] # reverse so bit 0 is index 0
    for i in range(TOTAL_TRIANGLES):
        if i < len(bin_str) and bin_str[i] == '1':
            feature[0, i] = 1.0
            
    # Tray
    feature[1, 0] = state.available[0]
    feature[2, 0] = state.available[1]
    feature[3, 0] = state.available[2]
    return feature

class PythonMCTS:
    def __init__(self, model, device, batch_size=8):
        self.model = model
        self.device = device
        self.batch_size = batch_size
        
    def add_dirichlet_noise(self, node: Node):
        if not node.children: return
        alpha = 0.3
        epsilon = 0.25
        noise = np.random.dirichlet([alpha] * len(node.children))
        for i, child in enumerate(node.children):
            child.prior = (1.0 - epsilon) * child.prior + epsilon * noise[i]
            
    def search(self, root_state: GameState, simulations: int = 100):
        # We need to completely expand the root instantly
        root = Node(root_state)
        
        # Expand all untried blindly at root
        while not root.expanded:
            root.expand()
            
        # Neural Evaluation for root natively
        features = extract_feature(root.state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            policy, value = self.model(features)
            policy = torch.exp(policy[0]).cpu().numpy()
            
        for child in root.children:
            slot, idx = child.move
            child.prior = float(policy[idx])
            
        self.add_dirichlet_noise(root)
        
        # Batched evaluation iterations
        iters = max(1, simulations // self.batch_size)
        
        for _ in range(iters):
            leaves = []
            search_paths = []
            
            # Phase 1: Select B leaves and apply Virtual Loss (GIL free logical traversal)
            for _ in range(self.batch_size):
                node = root
                path = []
                
                # Select
                while node.expanded and len(node.children) > 0:
                    node = node.select_child()
                    path.append(node)
                    node.virtual_loss += 1
                    
                # Expand
                if not node.state.terminal and not node.expanded:
                    node = node.expand()
                    path.append(node)
                    node.virtual_loss += 1
                    
                leaves.append(node)
                search_paths.append(path)
                
            # Phase 2: Native Batch GPU Evaluation
            feats = [extract_feature(leaf.state) for leaf in leaves]
            batch_tensor = torch.stack(feats).to(self.device)
            
            with torch.no_grad():
                policies, values = self.model(batch_tensor)
                policies = torch.exp(policies).cpu().numpy()
                values = values.cpu().numpy()
                
            # Phase 3: Backprop & remove virtual loss
            for i in range(self.batch_size):
                leaf = leaves[i]
                path = search_paths[i]
                val = float(values[i][0])
                poli = policies[i]
                
                # Assign priors to this newly expanded leaf
                if leaf != root and leaf.move is not None:
                    slot, idx = leaf.move
                    leaf.prior = float(poli[idx])
                    
                # Untrack virtual losses safely tracking back up
                for n in path:
                    n.virtual_loss -= 1
                    
                leaf.backpropagate(val)
            
        # Compute visits
        visits = {}
        for child in root.children:
            visits[child.move] = child.visits
            
        best_move = max(visits.keys(), key=lambda k: visits[k])
        return best_move, visits
