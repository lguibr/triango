
import numpy as np
import torch

from triango.env.state import GameState
from triango.mcts.features import extract_feature
from triango_ext import Node


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
        """
        Drives the deeply-threaded Native C++ Tree asynchronously.
        Python's sole responsibility is routing `extract_feature` matrices heavily batched to the GPU.
        """
        import triango_ext
        from triango_ext import AsyncMCTS, EvalResult
        from triango_ext import GameState as CppGameState
        
        # Initialize the global C++ Native environment masks if not already initialized
        # (It's safe to call multiple times if the implementation has a guard, or just call it here)
        try:
            triango_ext.initialize_env()
        except Exception:
            pass

        if not isinstance(root_state, CppGameState):
            board_str = bin(root_state.board)[2:].zfill(96)
            root_state = CppGameState(root_state.available, board_str, root_state.score)

        # 1. Initialize C++ Thread Manager
        # We use slightly fewer threads than batch_size to keep the queue healthy and GPU saturated
        manager = AsyncMCTS(root_state, threads=32, sims=simulations, c_puct=1.5)
        manager.start()

        # 2. Polling Loop
        import time
        try:
            while not manager.is_done():
                # Attempt to pull up to batch_size states simultaneously
                requests = manager.get_requests(self.batch_size)
                if not requests:
                    time.sleep(0.001)
                    continue

                # Batch them immediately together
                batch_tensors = []
                for req in requests:
                    batch_tensors.append(extract_feature(req.state))
                
                x = torch.stack(batch_tensors).to(self.device)
                
                with torch.no_grad():
                    pred_rem_scores, policy_probs = self.model(x)
                    pred_rem_scores = pred_rem_scores.cpu().numpy().flatten() * 100.0
                    policy_probs = policy_probs.cpu().numpy()
                
                # Repackage the evaluations back into C++ Result payloads
                results = []
                for i, req in enumerate(requests):
                    # The neural net predicts *remaining* score, so total value = current score + predicted remaining
                    v_scalar = float(req.state.score + pred_rem_scores[i])
                    p_array = policy_probs[i].flatten().tolist()
                    results.append(EvalResult(node=req.node, value=v_scalar, policy=p_array))

                manager.submit_results(results)
        finally:
            # 3. Clean Tree shutdown
            manager.stop()
        
        # 4. Same output parsing as classic synchronous MCTS
        root = manager.root
        
        visits: dict[tuple[int, int], int] = {}
        for child in root.children:
            if child.move is not None:
                visits[child.move] = int(child.visits)

        if not visits:
            return None, {}

        best_move = max(visits.keys(), key=lambda k: visits[k])
        return best_move, visits
