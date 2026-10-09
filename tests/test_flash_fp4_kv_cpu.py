"""Independent wire-format tests; optional actual-kernel CPU interpretation.

TRITON_INTERPRET=1 executes the packing/addressing/attention kernel Python on
CPU arrays. It does not verify CUDA codegen, device races or performance.
"""

import os
import pathlib
import struct
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "spark/flash"))
import fp4_kv_reference as ref
import nq_flash_fp4_kv as kv


class Format(unittest.TestCase):
    def test_every_nibble_and_little_endian_scales(self):
        row = np.zeros((1, 320), np.uint8)
        row[0, :256] = np.tile(np.arange(0x10, 0x100, 0x22, dtype=np.uint8), 32)
        scales = [2**-24, 0.125, 1.0, 65504.0] * 8
        for g, s in enumerate(scales):
            row[0, 256 + 2 * g : 258 + 2 * g] = list(struct.pack("<e", s))
        out = ref.decode(row)[0].reshape(32, 16)
        expected = np.r_[ref.VALUES, -ref.VALUES]
        np.testing.assert_array_equal(out, np.array(scales)[:, None] * expected)

    def test_midpoints_use_even_codes(self):
        vals = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0]
        row = np.tile(np.array(vals + [-x for x in vals]), 32)[None, :]
        encoded = ref.encode(row)
        codes = np.empty(512, np.uint8)
        codes[::2] = encoded[0, :256] & 15
        codes[1::2] = encoded[0, :256] >> 4
        np.testing.assert_array_equal(
            codes[:16], [0, 2, 2, 4, 4, 6, 6, 7, 8, 10, 10, 12, 12, 14, 14, 15]
        )

    def test_zero_and_representable_extremes(self):
        x = np.tile(np.r_[ref.VALUES, -ref.VALUES], (3, 32))
        x *= np.array([2**-24, 1.0, 65504.0])[:, None]
        np.testing.assert_array_equal(ref.decode(ref.encode(x)), x)
        np.testing.assert_array_equal(ref.decode(ref.encode(np.zeros((1, 512)))), 0)
        for value in (np.inf, np.nan, 393025.0):
            with self.assertRaises(ValueError):
                ref.encode(np.full((1, 512), value))

    def test_mapped_pages_and_rounding_accounting(self):
        # Logical pages [0,1,2] map to non-monotonic physical pages [2,0,3].
        table = [2, 0, 3]
        page = 4
        stride = 4 * 320 + 64
        buf = np.full(4 * stride, 0xCD, np.uint8)
        x = np.arange(12, dtype=np.float32)[:, None] * np.ones((12, 512))
        records = ref.encode(x)
        for logical in range(12):
            slot = table[logical // page] * page + logical % page
            offset = ref.address(slot, page, stride)
            buf[offset : offset + 320] = records[logical]
        for logical in range(12):
            offset = ref.address(
                table[logical // page] * page + logical % page, page, stride
            )
            np.testing.assert_array_equal(buf[offset : offset + 320], records[logical])
        for b in range(4):
            self.assertTrue(
                np.all(buf[b * stride + page * 320 : (b + 1) * stride] == 0xCD)
            )
        self.assertEqual(ref.memory_bytes(9, 2, page, stride), 3 * 2 * stride)
        self.assertEqual(ref.memory_bytes(262144, 15, 256), 1258291200)

    def test_sparse_empty_masks_and_causal_draft_rows(self):
        x = np.arange(6)[:, None] * np.ones((6, 512))
        cache = ref.encode(x)
        q = np.zeros((3, 2, 512))
        indices = np.array([[0, 1, -1, -1], [0, 1, 2, -1], [-1, -1, -1, -1]])
        result = ref.sparse_attention(q, cache, indices, 0.1)
        np.testing.assert_allclose(result[0], 0.5, atol=2e-4)
        np.testing.assert_allclose(result[1], 1.0, atol=4e-4)
        np.testing.assert_array_equal(result[2], 0.0)
        # Rejected later draft rows are still stored, but cannot participate in
        # an earlier token's softmax unless the caller supplies their indices.
        cache[3:] = ref.encode(np.full((3, 512), 100.0))
        np.testing.assert_array_equal(
            ref.sparse_attention(q, cache, indices, 0.1), result
        )

    def test_byte_view_preserves_padded_page_strides(self):
        data = torch.zeros(3 * (4 * 320 + 64), dtype=torch.uint8)
        cache = torch.as_strided(data, (3, 4, 320), (4 * 320 + 64, 320, 1))
        b = kv._cache_bytes(cache.view(torch.float8_e4m3fn))
        self.assertEqual(b.stride(), cache.stride())
        with self.assertRaises(ValueError):
            kv._cache_bytes(torch.empty((2, 4, 656), dtype=torch.uint8))


@unittest.skipUnless(
    os.environ.get("TRITON_INTERPRET") == "1", "Requires Triton CPU interpreter"
)
class KernelInterpreter(unittest.TestCase):
    def make_cache(self, blocks=4, page=8):
        stride = page * 320 + 64
        backing = torch.full((blocks * stride,), 0xCD, dtype=torch.uint8)
        return backing, torch.as_strided(backing, (blocks, page, 320), (stride, 320, 1))

    def write(self, x, cache, slots):
        kv._pack[(len(x),)](
            x,
            cache,
            slots,
            x.stride(0),
            slots.stride(0),
            cache.shape[1],
            cache.stride(0),
            cache.stride(1),
            cache.shape[0] * cache.shape[1],
            num_warps=4,
        )

    def test_pack_actual_kernel_masks_pages_extremes_and_reuse(self):
        backing, cache = self.make_cache()
        x = torch.randn((8, 512), generator=torch.Generator().manual_seed(715))
        vals = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0])
        x[0] = torch.cat((vals, -vals)).repeat(32)
        # Non-unit dyadic scales catch hardware reciprocal approximation at
        # exact FP4 midpoints (which the CPU interpreter alone cannot reveal).
        x[4] = x[0] * 0.46875
        x[6] = x[0] * 0.234375
        x[7] = x[0] * 0.2890625
        x[1].zero_()
        x[2] = torch.tensor(np.r_[ref.VALUES, -ref.VALUES].copy()).repeat(32) * 2**-24
        x[3] = torch.tensor(np.r_[ref.VALUES, -ref.VALUES].copy()).repeat(32) * 65504.0
        slots = torch.tensor([25, 0, 16, 7, 8, -1, 31, 2], dtype=torch.int64)
        self.write(x, cache, slots)
        expected = ref.encode(x.numpy())
        for i, slot in enumerate(slots.tolist()):
            if slot >= 0:
                np.testing.assert_array_equal(
                    cache[slot // 8, slot % 8].numpy(), expected[i]
                )
        for b in range(4):
            self.assertTrue(
                torch.all(
                    backing[b * cache.stride(0) + 8 * 320 : (b + 1) * cache.stride(0)]
                    == 0xCD
                )
            )
        # Reuse a physical page for a later request; the entire record replaces
        # old values and old group scales, rather than retaining stale tails.
        fresh = torch.full((1, 512), 3.0)
        self.write(fresh, cache, torch.tensor([25]))
        np.testing.assert_array_equal(cache[3, 1].numpy(), ref.encode(fresh.numpy())[0])

    def test_attention_actual_kernel_split_merge_masks_and_prefill(self):
        _, cache = self.make_cache()
        rng = np.random.default_rng(98)
        x = torch.tensor(rng.normal(size=(32, 512)), dtype=torch.float32)
        self.write(x, cache, torch.arange(32))
        # Include empty split partitions, an entirely masked row, mixed signs,
        # strided queries, and widths spanning both DSA and DSA+SWA dispatch.
        for width in (1, 33, 2048, 2176):
            n, heads, splits = 3, 3, 8
            q = torch.tensor(rng.normal(size=(n, heads, 1024)), dtype=torch.float16)[
                ..., ::2
            ].contiguous()
            indices = torch.full((n, width), -1, dtype=torch.int32)
            indices[0, : min(17, width)] = torch.arange(min(17, width))
            indices[1, : min(32, width)] = torch.randperm(
                32, generator=torch.Generator().manual_seed(91)
            )[: min(32, width)]
            p = torch.empty((n, splits, heads, 512), dtype=torch.float32)
            lognorm = torch.empty((n, splits, heads), dtype=torch.float32)
            out = torch.empty_like(q)
            kv._attention[(n, 1, splits)](
                q,
                cache,
                indices,
                p,
                lognorm,
                heads,
                width,
                q.stride(0),
                q.stride(1),
                indices.stride(0),
                cache.shape[1],
                cache.stride(0),
                cache.stride(1),
                32,
                0.04,
                splits,
                num_warps=8,
            )
            kv._merge[(n, heads)](
                p,
                lognorm,
                out,
                heads,
                splits,
                out.stride(0),
                out.stride(1),
                num_warps=4,
            )
            expected = ref.sparse_attention(
                q.numpy(), ref.encode(x.numpy()), indices.numpy(), 0.04
            )
            np.testing.assert_allclose(out.numpy(), expected, atol=0.004, rtol=0.015)
            np.testing.assert_array_equal(out[2].numpy(), 0)


if __name__ == "__main__":
    unittest.main()
