import traceback

from triango.training.warmup import reconstruct_elite_tensors

try:
    print("Testing reconstruct_elite_tensors with fake data...")
    fake_elite = [([ (0,0), (1,1) ], 12.0)]
    tensors = reconstruct_elite_tensors(fake_elite)
    print(f"Success, produced {len(tensors)} elements.")
except Exception:
    traceback.print_exc()
