"""Opt-in device parity test and isolated-kernel timing. Never launches a server.

Run only after obtaining the GPU lease. For memory/race debugging run this
under compute-sanitizer. Performance here is NOT end-to-end serving TPS.
"""

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import fp4_kv_reference as ref
import nq_flash_fp4_kv as kv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--heads", type=int, default=64)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 2, 3, 4, 17, 65])
    args = parser.parse_args()
    torch.cuda.set_device(args.device)
    rng = np.random.default_rng(543)
    page, blocks = 256, 1024
    page_stride = page * 320 + 256
    backing = torch.full(
        (blocks * page_stride,), 0xCD, dtype=torch.uint8, device=args.device
    )
    cache = torch.as_strided(backing, (blocks, page, 320), (page_stride, 320, 1))
    x = torch.tensor(rng.normal(size=(2176, 512)), dtype=torch.bfloat16)
    # Includes the final slot of a 262K cache and non-monotonic physical pages.
    slots = np.r_[np.arange(2160), np.arange(blocks * page - 16, blocks * page)].astype(
        np.int64
    )
    slots_gpu = torch.from_numpy(slots).to(args.device)
    kv.pack(x.to(args.device), cache, slots_gpu)
    torch.cuda.synchronize()
    actual = cache[slots_gpu // page, slots_gpu % page].cpu().numpy()
    encoded = ref.encode(x.float().numpy())
    np.testing.assert_array_equal(actual, encoded)
    # Guard bytes prove no write crosses a physical page's payload.
    assert torch.all(backing.view(blocks, page_stride)[:, page * 320 :] == 0xCD).item()
    for dtype in (torch.float16, torch.bfloat16):
        for batch in args.batches:
            for width in (2048, 2176):
                q_cpu = torch.tensor(
                    rng.normal(size=(batch, args.heads, 512)), dtype=dtype
                )
                q = q_cpu.to(args.device)
                logical = np.tile(np.arange(width), (batch, 1))
                physical = slots[logical]
                # Invalid tail and an empty query exercise masking and splits.
                physical[0, -13:] = -1
                logical[0, -13:] = -1
                if batch > 1:
                    physical[-1] = -1
                    logical[-1] = -1
                indices = torch.tensor(physical, dtype=torch.int32, device=args.device)
                out = kv.attention(q, cache, indices, 0.04)
                torch.cuda.synchronize()
                # FP4 vectors are converted to query precision before the MMA.
                # Compare selected heads/rows with independent FP64 softmax;
                # tolerances include MMA operand and probability rounding.
                rows = sorted(set([0, batch // 2, batch - 1]))
                heads = sorted(set([0, args.heads // 2, args.heads - 1]))
                expected = ref.sparse_attention(
                    q_cpu[rows][:, heads].float().numpy(), encoded, logical[rows], 0.04
                )
                observed = out[rows][:, heads].float().cpu().numpy()
                diff = observed.astype(np.float64) - expected
                rel = float(np.linalg.norm(diff) / max(np.linalg.norm(expected), 1e-30))
                maxabs = float(np.abs(diff).max())
                assert rel < (0.02 if dtype == torch.bfloat16 else 0.005), (
                    dtype,
                    batch,
                    width,
                    rel,
                )
                assert maxabs < (0.035 if dtype == torch.bfloat16 else 0.009), (
                    dtype,
                    batch,
                    width,
                    maxabs,
                )
                start, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                start.record()
                for _ in range(10):
                    kv.attention(q, cache, indices, 0.04)
                end.record()
                end.synchronize()
                print(
                    json.dumps(
                        {
                            "dtype": str(dtype),
                            "batch": batch,
                            "width": width,
                            "max_abs": maxabs,
                            "relative_l2": rel,
                            "attention_ms": start.elapsed_time(end) / 10,
                        }
                    )
                )
    # Repeated request replaces a previously written high physical slot.
    fresh = torch.zeros((1, 512), dtype=torch.bfloat16, device=args.device)
    kv.pack(fresh, cache, slots_gpu[-1:])
    np.testing.assert_array_equal(
        cache[-1, -1].cpu().numpy(), ref.encode(np.zeros((1, 512)))[0]
    )
    print("GPU parity, masks, high positions, page guards and reuse: PASS")


if __name__ == "__main__":
    main()
