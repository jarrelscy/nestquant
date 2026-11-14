"""Metal kernels for the GLM-5.3-Flash sparse-attention indexer on short blocks (1-8 query tokens with a cache).

index_select replaces everything mlx-vlm's Glm5NextIndexer.__call__ does after the pool-score matmul (about 35
small kernels per layer: ReLU, head weighting and head sum, candidate mask, argpartition, argsort back to
chronological order, pool expansion, padding, and the always-selected partial tail with a cumsum and argmax over
the whole key history) with two kernels: a parallel score kernel and one threadgroup per query row that finds
the top-k threshold by a 4-pass radix select and writes the selected pools in chronological order, expanded to
token indices, followed by the tail.

Equality with the reference: scores use the reference's float32 operations in its order (max(s, 0) * (w * scale),
then MLX's pairwise sum over heads, no fused multiply-add); test_sparse_kernels checks them bitwise. The selected
set is the reference's except when pools tie exactly at the k-th score, where argpartition picks an unspecified
subset and this kernel the lowest pool indices. The output layout is the reference's; with trim=True the columns
that are -1 for every row because fewer than 512 pools exist (contexts under 2,048 tokens) are omitted, which
changes attention only in float32 summation order (masked slots contribute exact zeros).
"""

from functools import lru_cache

import mlx.core as mx


_TG = 1024
# Products and sums rounded separately, as in the reference's separate MLX kernels (no fused multiply-add).
_NO_CONTRACT = '#pragma METAL fp contract(off)\n'

_SCORE_SOURCE = """
    // One thread per (pool, row): max(s, 0) * (w * scale), then MLX's pairwise sum over heads (H is a
    // power of two), as an order-preserving unsigned key.
    const int p = thread_position_in_grid.x;
    const int t = thread_position_in_grid.y;
    const int P = raw_shape[2];
    if (p >= P) return;
    const device float* r = raw + (t * H) * P + p;
    const device float* wt = w + t * H;
    const float sc = scale[0];
    float v[H];
    for (int h = 0; h < H; ++h) v[h] = metal::max(r[h * P], 0.0f) * (wt[h] * sc);
    for (int width = H / 2; width >= 1; width /= 2) {
        for (int h = 0; h < width; ++h) v[h] = v[2 * h] + v[2 * h + 1];
    }
    float s = v[0] + 0.0f;  // -0 -> +0: equal scores compare equal
    uint u = as_type<uint>(s);
    keys[t * P + p] = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
"""

_INDEX_SELECT_SOURCE = """
    constexpr uint TG = 1024;
    const uint tid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const int t = (int)threadgroup_position_in_grid.y;
    const int P = params[0];          // pools
    const int N = params[1];          // keys (cache length including this block)
    const int PV = params[2];         // valid pools
    const int SK = params[3];         // pools to select (== P: all candidates)
    const int L = params[4];          // query rows
    const int OUTW = params[5];       // output width: 4 * min(512, P) + TAIL
    const int qpos = N - L + t;
    const int sq = metal::clamp(qpos, 0, metal::max(N - 1, 0));
    device int* out = topk + t * OUTW;
    device uint* keys = scratch + t * P;

    threadgroup int red_a[32];
    threadgroup int red_b[32];
    threadgroup atomic_uint hist[8 * 256];
    threadgroup uint shared_u[4];

    // First valid key (argmax of the validity row, N if none) and visible keys up to this query.
    int fk = N;
    for (int i = (int)tid; i < N; i += TG) {
        if (key_valid[i]) { fk = i; break; }
    }
    int cnt = 0;
    for (int i = (int)tid; i <= sq && i < N; i += TG) cnt += key_valid[i] ? 1 : 0;
    fk = simd_min(fk);
    cnt = simd_sum(cnt);
    if (lane == 0) { red_a[sg] = fk; red_b[sg] = cnt; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        int a = red_a[lane];
        int b = red_b[lane];
        a = simd_min(a);
        b = simd_sum(b);
        if (lane == 0) { red_a[0] = a; red_b[0] = b; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    fk = red_a[0];
    const int visible = red_b[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Order-preserving keys; 0 marks a non-candidate pool (below every real score).
    const bool select_all = SK >= P;
    for (int p = (int)tid; p < P; p += TG) {
        int end = metal::clamp(fk + 4 * p + 3, 0, metal::max(N - 1, 0));
        bool candidate = p < PV && end <= qpos;
        uint key = 0;
        if (candidate) {
            if (select_all) {
                key = 1;
            } else {
                key = prekeys[t * P + p];
                key = metal::max(key, 1u);
            }
        }
        keys[p] = key;
    }
    threadgroup_barrier(mem_flags::mem_device);

    // Radix select: threshold key T (the SK-th largest) and how many keys equal to T to take.
    uint prefix = 0, mask = 0;
    int need = SK;
    if (!select_all) {
        for (int pass = 0; pass < 4; ++pass) {
            const uint shift = 24 - 8 * pass;
            // Eight histogram copies (simdgroup % 8) spread the atomic traffic: scores share few top bytes.
            for (uint b = tid; b < 8 * 256; b += TG) atomic_store_explicit(&hist[b], 0u, memory_order_relaxed);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            threadgroup atomic_uint* mine = hist + (sg % 8) * 256;
            for (int p = (int)tid; p < P; p += TG) {
                uint key = keys[p];
                if ((key & mask) == prefix) {
                    atomic_fetch_add_explicit(&mine[(key >> shift) & 255u], 1u, memory_order_relaxed);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            // Bin counts from the top (bin 255 first): thread i < 256 holds bin 255 - i; suffix sums by
            // simdgroup prefix sums plus the totals of earlier simdgroups.
            int count = 0;
            if (tid < 256) {
                uint bin = 255 - tid;
                for (int c = 0; c < 8; ++c) count += (int)atomic_load_explicit(&hist[c * 256 + bin], memory_order_relaxed);
            }
            int incl = simd_prefix_inclusive_sum(count);
            if (tid < 256 && lane == 31) red_a[sg] = incl;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid < 256) {
                int before = 0;
                for (uint g2 = 0; g2 < sg; ++g2) before += red_a[g2];
                int above = before + incl - count;   // keys in higher bins
                if (above < need && above + count >= need) {
                    shared_u[0] = 255 - tid;
                    shared_u[1] = (uint)(need - above);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            prefix |= shared_u[0] << shift;
            mask |= 255u << shift;
            need = (int)shared_u[1];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    } else {
        prefix = 0;   // every candidate (key 1) is above the threshold
        need = 0;
    }
    const uint T = prefix;

    // Emit selected pools in chronological order: contiguous chunks per thread, two prefix sums.
    const int chunk = (P + (int)TG - 1) / (int)TG;
    const int lo = metal::min((int)tid * chunk, P);
    const int hi = metal::min(lo + chunk, P);
    int g = 0, e = 0;
    for (int p = lo; p < hi; ++p) {
        uint key = keys[p];
        if (key == 0) continue;
        g += key > T ? 1 : 0;
        e += key == T ? 1 : 0;
    }
    // exclusive prefix of e over threads
    int e_incl = simd_prefix_inclusive_sum(e);
    if (lane == 31) red_a[sg] = e_incl;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        int v = red_a[lane];
        red_a[lane] = simd_prefix_exclusive_sum(v);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int tie_before = red_a[sg] + e_incl - e;
    const int ties_taken = metal::clamp(need - tie_before, 0, e);
    const int mine = g + ties_taken;
    int m_incl = simd_prefix_inclusive_sum(mine);
    if (lane == 31) red_b[sg] = m_incl;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        int v = red_b[lane];
        int total = simd_sum(v);
        red_b[lane] = simd_prefix_exclusive_sum(v);
        if (lane == 0) shared_u[2] = (uint)total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    int j = red_b[sg] + m_incl - mine;
    const int selected = (int)shared_u[2];
    int ties_left = ties_taken;
    for (int p = lo; p < hi; ++p) {
        uint key = keys[p];
        if (key == 0) continue;
        bool take = key > T;
        if (!take && key == T && ties_left > 0) { take = true; --ties_left; }
        if (take) {
            int base = fk + 4 * p;
            out[4 * j + 0] = base;
            out[4 * j + 1] = base + 1;
            out[4 * j + 2] = base + 2;
            out[4 * j + 3] = base + 3;
            ++j;
        }
    }
    const int region = OUTW - TAIL;
    for (int c = 4 * selected + (int)tid; c < region; c += TG) out[c] = -1;

    // Always-selected partial tail.
    if (tid < (uint)TAIL) {
        int tail_count = visible % 4;
        int r = (int)tid;
        int tail = fk + visible - tail_count + r;
        bool valid = r < tail_count && tail < N;
        if (valid) valid = key_valid[metal::clamp(tail, 0, metal::max(N - 1, 0))];
        out[region + r] = valid ? tail : -1;
    }
"""


@lru_cache(maxsize=None)
def _score_kernel():
    return mx.fast.metal_kernel(
        name='nq_index_scores', input_names=['raw', 'w', 'scale'], output_names=['keys'], source=_SCORE_SOURCE,
        header=_NO_CONTRACT, ensure_row_contiguous=True)


@lru_cache(maxsize=None)
def _index_select_kernel(with_scores):
    inputs = ['prekeys', 'key_valid', 'params'] if with_scores else ['key_valid', 'params']
    source = _INDEX_SELECT_SOURCE
    if not with_scores:
        source = source.replace('key = prekeys[t * P + p];', 'key = 1;')
    return mx.fast.metal_kernel(
        name='nq_index_select' if with_scores else 'nq_index_select_all', input_names=inputs,
        output_names=['topk', 'scratch'], source=source, ensure_row_contiguous=True)


def index_select(raw, weights, scale, key_valid, pools, valid_pools, select_k, rows, kpool=4, tail=3, index_topk=2048,
                 return_keys=False, trim=True):
    """Top-k token indices for each query row of a block.

    raw: [rows, H, pools] float32 pool scores before ReLU (None when every candidate pool is selected),
    weights: [rows, H] float32 head weights, scale: float32 [1], key_valid: [N] bool.
    Returns [rows, 4 * min(index_topk // kpool, pools) + tail] int32 (see module docstring); with trim=False the
    reference's full [rows, index_topk + tail] layout."""
    if kpool != 4:
        raise ValueError('index_select supports 4-token pools')
    n = key_valid.shape[-1]
    region = kpool * min(index_topk // kpool, pools) if trim else index_topk
    width = region + tail
    params = mx.array([pools, n, valid_pools, select_k, rows, width], dtype=mx.int32)
    with_scores = select_k < pools
    heads = raw.shape[-2] if with_scores else 1
    if heads & (heads - 1):
        raise ValueError('index_select needs a power-of-two head count')
    inputs = [key_valid, params]
    if with_scores:
        prekeys = _score_kernel()(
            inputs=[raw, weights, scale], template=[('H', heads)], grid=(pools, rows, 1),
            threadgroup=(min(256, pools), 1, 1), output_shapes=[(rows, pools)], output_dtypes=[mx.uint32])[0]
        inputs = [prekeys] + inputs
    topk, keys = _index_select_kernel(with_scores)(
        inputs=inputs, template=[('TAIL', tail)],
        grid=(_TG, rows, 1), threadgroup=(_TG, 1, 1),
        output_shapes=[(rows, width), (rows, max(pools, 1))], output_dtypes=[mx.int32, mx.uint32])
    return (topk, keys) if return_keys else topk
