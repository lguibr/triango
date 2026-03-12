
import numpy as np
import torch

from triango.env.state import GameState
from triango.mcts.features import extract_feature
from triango.mcts.node import Node


class PythonMCTS:
    def __init__(self, model: torch.nn.Module, device: torch.device, batch_size: int = 32):
        self.model = model
        self.device = device
        self.batch_size = batch_size

    def add_dirichlet_noise(self, node: Node) -> None:
        if not node.children:
            return
        alpha = 10.0 / len(node.children) if len(node.children) > 0 else 0.3
        epsilon = 0.25
        noise = np.random.dirichlet([alpha] * len(node.children))
        for i, child in enumerate(node.children):
            child.prior = (1.0 - epsilon) * child.prior + epsilon * noise[i]

    def search(
        self, root_state: GameState, simulations: int = 100
    ) -> tuple[tuple[int, int] | None, dict[tuple[int, int], int]]:
        root = Node(root_state)

        while not root.expanded:
            root.expand()

        # Evaluate the root node directly to extract Action Policy P(s,a)
        root_feat = extract_feature(root_state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, _, policy_probs = self.model(root_feat)
            
        policy_probs = policy_probs.squeeze(0).cpu().numpy()  # [3, 50]

        # Assign true learned priors from the Network Head
        for child in root.children:
            if child.move is not None:
                slot, orientation_idx = child.move
                # Safety fallback just incase orientation_idx > 50 in future expansions
                prob = policy_probs[slot, min(orientation_idx, 49)]
                child.prior = float(prob)
                
        # Normalize priors if sum > 0
        total_p = sum(child.prior for child in root.children)
        if total_p > 0:
            for child in root.children:
                child.prior /= total_p
        else:
            uniform = 1.0 / len(root.children) if len(root.children) > 0 else 1.0
            for child in root.children:
                child.prior = uniform

        self.add_dirichlet_noise(root)

        iters = max(1, simulations // self.batch_size)

        for _ in range(iters):
            leaves: list[Node] = []
            search_paths: list[list[Node]] = []

            for _ in range(self.batch_size):
                node = root
                path = []

                while node.expanded and len(node.children) > 0:
                    next_node = node.select_child()
                    if next_node is None:
                        break
                    node = next_node
                    path.append(node)
                    node.virtual_loss += 1

                if not node.state.terminal and not node.expanded:
                    node = node.expand()
                    path.append(node)
                    node.virtual_loss += 1

                leaves.append(node)
                search_paths.append(path)

            feats = [extract_feature(leaf.state) for leaf in leaves]
            batch_tensor = torch.stack(feats).to(self.device)

            with torch.no_grad():
                pred_rem_scores, _, _ = self.model(batch_tensor)
                pred_rem_scores = pred_rem_scores.cpu().numpy().flatten() * 100.0

            for i in range(self.batch_size):
                leaf = leaves[i]
                path = search_paths[i]

                if leaf.state.terminal:
                    val = float(leaf.state.score)
                else:
                    val = float(leaf.state.score + pred_rem_scores[i])

                if leaf != root and not leaf.expanded:
                    if len(leaf.children) > 0:
                        uniform_p = 1.0 / len(leaf.children)
                        for c in leaf.children:
                            c.prior = uniform_p

                for n in path:
                    n.virtual_loss -= 1

                leaf.backpropagate(val)

        visits: dict[tuple[int, int], int] = {}
        for child in root.children:
            if child.move is not None:
                visits[child.move] = child.visits

        if not visits:
            return None, {}

        best_move = max(visits.keys(), key=lambda k: visits[k])
        return best_move, visits
