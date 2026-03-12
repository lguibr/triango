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

    def puct(self, c_puct=1.5): # MAC OPTIMIZATION: Increased c_puct to value prior more heavily over raw brute force
        # First Play Urgency (FPU): If node is unvisited, assume it's roughly as good as its parent
        if self.visits + self.virtual_loss == 0:
            exploit = self.parent.value_sum / max(1, self.parent.visits) if self.parent else 0.0
        else:
            exploit = (self.value_sum - self.virtual_loss) / (self.visits + self.virtual_loss)
            
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
    # Build a [7, 96] spatial tensor
    # Channel 0: Current Board
    # Channel 1: Geometry of piece in slot 0
    # Channel 2: Valid Mask of piece in slot 0 
    # Channel 3: Geometry of piece in slot 1
    # Channel 4: Valid Mask of piece in slot 1
    # Channel 5: Geometry of piece in slot 2
    # Channel 6: Valid Mask of piece in slot 2
    
    from env import get_piece_overlay, get_valid_placement_mask
    
    feature = torch.zeros(7, 96, dtype=torch.float32)
    # Channel 0: Board
    bin_str = bin(state.board)[2:].zfill(TOTAL_TRIANGLES)[::-1]
    for i in range(TOTAL_TRIANGLES):
        if i < len(bin_str) and bin_str[i] == '1':
            feature[0, i] = 1.0
            
    # Channels 1-6: Piece Overlays and Valid Masks
    for slot in range(3):
        p_id = state.available[slot]
        overlay = get_piece_overlay(p_id)
        valid_mask = get_valid_placement_mask(p_id, state.board)
        for i in range(TOTAL_TRIANGLES):
            if overlay[i] == 1:
                feature[(slot * 2) + 1, i] = 1.0
            if valid_mask[i] == 1:
                feature[(slot * 2) + 2, i] = 1.0
                
    return feature

class PythonMCTS:
    def __init__(self, model, device, batch_size=32):
        self.model = model
        self.device = device
        self.batch_size = batch_size
        
    def add_dirichlet_noise(self, node: Node):
        if not node.children: return
        # MAC OPTIMIZATION: Dynamic Alpha. More moves -> sharper noise to avoid uniformly flat priors
        alpha = 10.0 / len(node.children) if len(node.children) > 0 else 0.3
        epsilon = 0.25
        noise = np.random.dirichlet([alpha] * len(node.children))
        for i, child in enumerate(node.children):
            child.prior = (1.0 - epsilon) * child.prior + epsilon * noise[i]
            
    def search(self, root_state: GameState, simulations: int = 100):
        root = Node(root_state)
        
        # Expand all untried blindly at root
        while not root.expanded:
            root.expand()
            
        # NEXT-STATE EVALUATION: We must evaluate all *children* (after-states) natively to build the Policy prior
        child_feats = [extract_feature(child.state) for child in root.children]
        
        if len(child_feats) > 0:
            batch_tensor = torch.stack(child_feats).to(self.device)
            with torch.no_grad():
                # Value head predicts Remaining Potential (normalized, so we scale it back)
                rem_scores, line_clears = self.model(batch_tensor)
                rem_scores = rem_scores.cpu().numpy().flatten() * 100.0
                line_clears = line_clears.cpu().numpy().flatten()
                
            # MCTS Expected Final Score = Immediate Score + Predicted Remaining Score
            values = np.array([child.state.score + rem_scores[i] for i, child in enumerate(root.children)])
            
            # Formulate prior: higher Expected Final Score = much higher prior
            # Incorporate line_clears as a heuristic boost (+20 points equivalent heuristically to encourage exploring line clears)
            heuristic_values = values + (line_clears * 20.0)
            
            # Temperature scaling (5.0) to smooth the priors so we don't solely explore the greedy line initially
            exp_v = np.exp((heuristic_values - np.max(heuristic_values)) / 5.0) 
            priors = exp_v / np.sum(exp_v)
            
            for i, child in enumerate(root.children):
                child.prior = float(priors[i])
                
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
            # Evaluate the AFTER-STATE of the leaf
            feats = [extract_feature(leaf.state) for leaf in leaves]
            batch_tensor = torch.stack(feats).to(self.device)
            
            with torch.no_grad():
                pred_rem_scores, _ = self.model(batch_tensor)
                pred_rem_scores = pred_rem_scores.cpu().numpy().flatten() * 100.0
                
            # Phase 3: Backprop & remove virtual loss
            for i in range(self.batch_size):
                leaf = leaves[i]
                path = search_paths[i]
                
                if leaf.state.terminal:
                    val = float(leaf.state.score)
                else:
                    val = float(leaf.state.score + pred_rem_scores[i])
                
                # If this leaf was just expanded, formulate priors for its explicitly-evaluated children
                # (To optimize computationally, we don't deeply evaluate children until they are selected,
                #  so we assign uniform priors for deeper nodes here)
                if leaf != root and not leaf.expanded:
                    if len(leaf.children) > 0:
                        uniform_p = 1.0 / len(leaf.children)
                        for c in leaf.children:
                            c.prior = uniform_p
                    
                # Untrack virtual losses safely tracking back up
                for n in path:
                    n.virtual_loss -= 1
                    
                leaf.backpropagate(val)
            
        # Compute visits
        visits = {}
        for child in root.children:
            visits[child.move] = child.visits
            
        if not visits:
            return None, {}
            
        best_move = max(visits.keys(), key=lambda k: visits[k])
        return best_move, visits
