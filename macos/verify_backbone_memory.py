import argparse
import ctypes
import gc
import hashlib
import json
from pathlib import Path

import torch

from backbone import empty_model, load_backbone
from metrics import MemoryGuard, snapshot


def fingerprint(tensors):
    digest = hashlib.sha256()
    for tensor in tensors:
        values = tensor.reshape(-1)[:8].float().cpu().numpy()
        digest.update(values.tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--memory-limit-gib', type=float, default=32)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh output file')
    guard = MemoryGuard(args.memory_limit_gib)
    torch.set_num_threads(8)
    model = empty_model(args.model, lambda _: torch.nn.Identity())
    head = load_backbone(args.model, model, memory_check=guard.check)
    tensors = [*model.state_dict().values(), head]
    before_hash = fingerprint(tensors)
    torch.mps.synchronize()
    before = snapshot()
    torch.mps.empty_cache()
    after = snapshot()
    gc.collect()
    allocator = ctypes.CDLL(None)
    allocator.malloc_zone_pressure_relief.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    allocator.malloc_zone_pressure_relief.restype = ctypes.c_size_t
    released = allocator.malloc_zone_pressure_relief(None, 0)
    after_allocator = snapshot()
    assert fingerprint(tensors) == before_hash
    assert after['mps_allocated_bytes'] == before['mps_allocated_bytes']
    guard.check(force=True)
    report = dict(status='passed', before_cache_release=before, after_cache_release=after,
                  after_allocator_release=after_allocator, allocator_reported_released_bytes=released,
                  tensor_count=len(tensors), tensor_bytes=sum(t.numel() * t.element_size() for t in tensors),
                  sampled_tensor_digest=before_hash,
                  scope='Text backbone only; first eight values of each live tensor unchanged after releasing unused MPS cache; no expert loading or generation')
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
