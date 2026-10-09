"""CPU-only specification/oracle for the experimental Flash FP4 MLA cache.

No CUDA, Triton, vLLM or checkpoint imports. Intended for tests and accounting,
not serving. This implementation enumerates codewords rather than copying the
GPU kernel's threshold tree.
"""

import math
import struct

import numpy as np

VALUES = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=np.float64)
ROW_BYTES = 320


def encode(rows):
    x = np.asarray(rows, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 512:
        raise ValueError("Expected [tokens, 512]")
    if not np.isfinite(x).all() or np.abs(x).max(initial=0) > 393024:
        raise ValueError("Input outside finite FP4/FP16-scale range")
    result = np.zeros((len(x), ROW_BYTES), dtype=np.uint8)
    for token, row in enumerate(x):
        for group in range(32):
            values = row[group * 16 : (group + 1) * 16]
            scale = np.float16(max(np.max(np.abs(values)) / 6, 2**-24))
            encoded = []
            for value in values:
                distances = np.abs(VALUES - abs(value) / float(scale))
                candidates = np.flatnonzero(distances == distances.min())
                # Tie: choose the code with even low mantissa bit.
                even = candidates[candidates % 2 == 0]
                code = int(even[0] if len(even) else candidates[0])
                encoded.append(code | (8 if value < 0 else 0))
            result[token, group * 8 : (group + 1) * 8] = [
                encoded[j] | encoded[j + 1] << 4 for j in range(0, 16, 2)
            ]
            result[token, 256 + group * 2 : 258 + group * 2] = list(
                struct.pack("<e", scale)
            )
    return result


def decode(rows):
    b = np.asarray(rows)
    if b.ndim != 2 or b.shape[1] != ROW_BYTES or b.dtype != np.uint8:
        raise ValueError("Expected uint8 [tokens, 320]")
    result = np.empty((len(b), 512), dtype=np.float64)
    for token, row in enumerate(b):
        for group in range(32):
            scale = struct.unpack(
                "<e", row[256 + group * 2 : 258 + group * 2].tobytes()
            )[0]
            for j in range(16):
                code = (int(row[group * 8 + j // 2]) >> (4 * (j % 2))) & 15
                result[token, group * 16 + j] = (
                    VALUES[code & 7] * (-1 if code & 8 else 1) * scale
                )
    return result


def sparse_attention(query, cache_rows, indices, scale):
    q = np.asarray(query, dtype=np.float64)
    values = decode(cache_rows)
    result = np.zeros_like(q)
    for t, slots in enumerate(np.asarray(indices)):
        if np.any(slots >= len(values)):
            raise ValueError("Cache index out of bounds")
        v = values[slots[slots >= 0]]
        if not len(v):
            continue
        score = q[t] @ v.T * scale
        p = np.exp(score - score.max(axis=1, keepdims=True))
        result[t] = p @ v / p.sum(axis=1, keepdims=True)
    return result


def address(slot, page_tokens, page_stride, token_stride=ROW_BYTES):
    if slot < 0 or page_tokens <= 0 or page_stride < page_tokens * token_stride:
        raise ValueError("Invalid physical page address")
    return slot // page_tokens * page_stride + slot % page_tokens * token_stride


def memory_bytes(tokens, mla_layers, page_tokens, page_stride=None):
    """Only MLA rows, excluding indexer/KDA, null blocks and allocator reserve."""
    if min(tokens, mla_layers) < 0 or page_tokens <= 0:
        raise ValueError("Invalid cache dimensions")
    minimum_page = page_tokens * ROW_BYTES
    if page_stride is None:
        page_stride = minimum_page
    if page_stride < minimum_page:
        raise ValueError("Page is smaller than its token storage")
    return math.ceil(tokens / page_tokens) * page_stride * mla_layers
