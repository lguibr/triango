import random
import torch
from torch.utils.data import Dataset


class ReplayBuffer(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(self, capacity: int = 250000, elite_ratio: float = 0.1):
        self.capacity = capacity
        self.elite_capacity = int(capacity * elite_ratio)
        self.standard_capacity = self.capacity - self.elite_capacity
        
        self.buffer: list[tuple[torch.Tensor, float, float, torch.Tensor]] = []
        self.elite_buffer: list[tuple[torch.Tensor, float, float, torch.Tensor]] = []
        
        self.elite_threshold = 0.0

    def __len__(self) -> int:
        # To enforce a 50/50 Elite/Standard split during PyTorch DataLoader randomization,
        # we dynamically define the dataset epoch length to simply be twice the size of the smaller buffer,
        # or just 2x the standard buffer size if we interpolate Elite samples cleanly.
        return len(self.buffer) * 2 if len(self.elite_buffer) > 0 else len(self.buffer)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(self.elite_buffer) > 0 and idx % 2 == 1:
            # Over-Sample Elite Data (force 50% ratio mathematically inside the dataset sampler)
            elite_idx = random.randint(0, len(self.elite_buffer) - 1)
            state, target_value, target_line_clear, target_policy = self.elite_buffer[elite_idx]
        else:
            std_idx = (idx // 2) if len(self.elite_buffer) > 0 else idx
            state, target_value, target_line_clear, target_policy = self.buffer[std_idx]
            
        return (
            state,
            torch.tensor([target_value], dtype=torch.float32),
            torch.tensor([target_line_clear], dtype=torch.float32),
            target_policy,
        )

    def push(
        self, state_tensor: torch.Tensor, target_value: float, target_line_clear: float, target_policy: torch.Tensor
    ) -> None:
        if len(self.buffer) >= self.standard_capacity:
            self.buffer.pop(0)
        self.buffer.append((state_tensor, target_value, target_line_clear, target_policy))

    def push_game(
        self, game_history: list[tuple[torch.Tensor, float, float, torch.Tensor]], final_score: float
    ) -> None:
        """
        Pushes an entire game into the buffer.
        If the game score is higher than the elite threshold, it goes into the elite buffer.
        """
        if final_score >= self.elite_threshold and final_score > 0:
            # Add to elite buffer
            for state, cleared_line, state_score, policy in game_history:
                rem_score = max(0.0, float(final_score - state_score))
                if len(self.elite_buffer) >= self.elite_capacity:
                    # We pop random elements from elite to make room, rather than strictly oldest
                    # to maintain a diverse elite pool.
                    pop_idx = random.randint(0, len(self.elite_buffer) - 1)
                    self.elite_buffer.pop(pop_idx)
                self.elite_buffer.append((state, rem_score, cleared_line, policy))
                
            # Update threshold if we are full
            if len(self.elite_buffer) >= self.elite_capacity:
                self.elite_threshold = final_score
        else:
            # Add to standard buffer
            for state, cleared_line, state_score, policy in game_history:
                rem_score = max(0.0, float(final_score - state_score))
                self.push(state, rem_score, cleared_line, policy)

