import torch
from triango.training.buffer import ReplayBuffer

b = ReplayBuffer(capacity=100)
history_elite = [(torch.zeros(1), 1.0, torch.zeros(1)) for _ in range(10)]
b.push_game(history_elite, 100.0)

history_std = [(torch.zeros(1), 0.5, torch.zeros(1)) for _ in range(5)]
b.push_game(history_std, 10.0)

print(f"Len elite: {len(b.elite_buffer)}")
print(f"Len std: {len(b.buffer)}")
print(f"Total Len: {len(b)}")

# try to get
for i in range(len(b)):
    try:
        b[i]
    except Exception as e:
        print(f"Failed at {i}: {e}")
        break
print("Done getting")
