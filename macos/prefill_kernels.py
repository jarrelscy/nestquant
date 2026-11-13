"""Batched prefill kernels: decode an expert's rotated weights to a dense fp16 matrix once, then use matmuls.

The banked decode kernels (decoder.projection_fused) compute one matvec per (token, expert) row and so
decode every weight again for every routed token. During prefill each expert serves many tokens, so here
an expert's 4-bit weights are decoded once into a dense rotated matrix and all its tokens go through a
GPU matrix multiply. The rotation, scales and low-rank terms around that matmul follow the prep and post
kernels in decoder.py, row by row:

    rotated = half(H(x * su))                    (same butterflies and rounding as decoder._prep_kernel)
    partial = rotated @ W                        (W holds exactly the fp16 weights the matvec kernels use)
    result  = H(partial) * sv + (x . lrV) lrU2 + (x . lrV) lrU4

The only numerical differences from the matvec path are the matmul's accumulation order and the fp16
rounding of `partial`. `partial` is scaled by 2^-PARTIAL_SHIFT before rounding so that it cannot
overflow fp16; the scale is a power of two, so it is exact apart from subnormal underflow.
"""

from functools import lru_cache

import mlx.core as mx

from decoder import HEADER, PATTERNS, WINDOW_HEADER, _matrix_indices

PARTIAL_SHIFT = 6
COLD = 0xffffffff

FWHT_HEADER = r'''
// 128-point Hadamard transform with the butterfly order and scale of decoder._FWHT. Each lane of a
// simdgroup holds four consecutive elements of one 128-element block.
inline float4 nq_fwht128(float4 v, uint lane) {
    float a0 = v.x + v.y, a1 = v.x - v.y, a2 = v.z + v.w, a3 = v.z - v.w;
    v = float4(a0 + a2, a1 + a3, a0 - a2, a1 - a3);
    for (uint m = 1; m < 32; m <<= 1) {
        float4 u = simd_shuffle_xor(v, ushort(m));
        v = (lane & m) ? (u - v) : (v + u);
    }
    return v * 0.08838834764831845f;
}
inline float4 nq_load4(const device half* p, uint i) {
    return float4(*reinterpret_cast<const device packed_half4*>(p + i));
}
inline float4 nq_load4(const constant half* p, uint i) {
    return float4(p[i], p[i + 1], p[i + 2], p[i + 3]);
}
inline void nq_store4(device half* p, uint i, float4 v) {
    *reinterpret_cast<device packed_half4*>(p + i) = packed_half4(half4(v));
}
'''


@lru_cache(maxsize=None)
def _dense_kernel(base_k: float, residual_k: float):
    """Write W^T[n, k] in fp16 for each expert of a group, with the values of decoder._window_kernel.

    One threadgroup per (packed unit, expert); simdgroup g decodes ring g of the unit and each lane
    eight consecutive ring positions, exactly as in the window matvec, but stores the weights instead
    of multiplying them by an input.
    """
    bka, bmask = PATTERNS[base_k]
    rka, rmask = PATTERNS[residual_k]

    def bit(ka, mask, position):
        if mask == 0xAAAA:
            return f'({ka}u * ({position}) + ({position}) / 2u)'
        return (f'({ka}u * ({position}) + (({position}) / 16u) * {mask.bit_count()}u'
                f' + popcount({mask}u & ((1u << (({position}) % 16u)) - 1u)))')

    base_bytes = (bka * 256 + bmask.bit_count() * 16) // 8
    residual_bytes = (rka * 256 + rmask.bit_count() * 16) // 8

    def delta(ka, mask, j):
        low = ka * j + (mask & ((1 << j) - 1)).bit_count()
        high = ka * j + (mask & ((1 << (8 + j)) - 1)).bit_count() - (mask & 0xFF).bit_count()
        return str(low) + 'u' if low == high else f'((lane & 1u) ? {high}u : {low}u)'

    def load(buffer, ring, offset, size):
        if size % 4:
            return f'nq_window({buffer}, {ring} * {size}u, {offset}, {size}u)'
        return f'nq_shuffle_window({buffer}, {ring} * {size}u, {offset}, {size // 4}u, lane)'

    def state(window, shift, ka, mask, j):
        low = ka * j + (mask & ((1 << j) - 1)).bit_count()
        high = ka * j + (mask & ((1 << (8 + j)) - 1)).bit_count() - (mask & 0xFF).bit_count()
        for start in (0, 8, 16):
            if all(start <= d and d + 7 + 16 - start <= 32 for d in (low, high)):
                return f'(({window}{start} >> ({shift} + {delta(ka, mask, j)} - {start}u)) & 65535u)'
        return f'(uint({window} >> ({shift} + {delta(ka, mask, j)})) & 65535u)'

    steps = []
    for j in range(8):
        steps.append(f"""
        {{
            uint code = nq_sum_fast({state('base_window', 'base_shift', bka, bmask, j)});
            if (refined) {{
                uint sr = nq_sum_fast({state('residual_window', 'residual_shift', rka, rmask, j)});
                code = (mb * code + n * sr + 128u) >> 8;
            }}
            w[{j}] = half(fma(a, float(code), b1024));
        }}""")
    source = f"""
    uint unit = threadgroup_position_in_grid.x;
    uint gi = threadgroup_position_in_grid.y;
    uint g = simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    uint depth = uint(variants_shape[0]);
    uint hot_rows = uint(words_shape[0]);
    uint expert = min(experts[gi], depth - 1u);
    uint hot = slots[gi];
    bool refined = hot < hot_rows;
    uint flat = order[unit];
    uint chunk = flat / TN;
    uint strip = flat % TN;
    uint p0 = lane * 8u;
    uint base_bit = {bit(bka, bmask, 'p0')};
    uint base_shift = base_bit % 8u;
    uint base_offset = base_bit / 8u;
    uint residual_bit = {bit(rka, rmask, 'p0')};
    uint residual_shift = residual_bit % 8u;
    uint residual_offset = residual_bit / 8u;
    uint ring = (expert * TK * TN + unit) * 8u + g;
    uint variant = PACKED_VAR ? ((uint(variants[ring >> 3]) >> (ring & 7u)) & 1u) : uint(variants[ring]);
    float sign = variant ? -1.0f : 1.0f;
    float a = float(as_type<half>(ushort(0x1eee))) * sign;
    float b = float(as_type<half>(ushort(0xc931))) * sign;
    uint mb = 0, n = 0;
    ulong residual_window = 0;
    if (refined) {{
        uint refined_unit = hot * TK * TN + unit;
        uint refined_ring = refined_unit * 8u + g;
        mb = words[refined_unit] & 255;
        n = (words[refined_unit] >> 8) & 255;
        volatile float rcp = 1.0f / float(mb);
        half ah = half(1.732421875f * rcp);
        volatile float nr = float(n) * rcp;
        volatile float nk = nr * -3.453125f;
        volatile float t = nk + -3.453125f;
        half ch = half(t - 1024.0f * float(ah));
        a = float(ah) * sign;
        b = float(ch) * sign;
        residual_window = {load('residual', 'refined_ring', 'residual_offset', residual_bytes)};
    }}
    ulong base_window = {load('base', 'ring', 'base_offset', base_bytes)};
    uint base_window0 = uint(base_window), base_window8 = uint(base_window >> 8),
         base_window16 = uint(base_window >> 16);
    uint residual_window0 = uint(residual_window), residual_window8 = uint(residual_window >> 8),
         residual_window16 = uint(residual_window >> 16);
    float b1024 = fma(a, 1024.0f, b);
    half w[8];
{''.join(steps)}
    uint k0 = chunk * 128u + 16u * (lane % 8u) + 2u * (lane / 8u);
    uint n0 = strip * 16u + g;
    device half2* out = reinterpret_cast<device half2*>(W + ulong(gi) * (TK * 128u) * (TN * 16u));
    const uint K2 = TK * 64u;
    out[(n0 * K2) + k0 / 2u] = half2(w[0], w[1]);
    out[((n0 + 8u) * K2) + k0 / 2u] = half2(w[2], w[3]);
    out[(n0 * K2) + k0 / 2u + 4u] = half2(w[4], w[5]);
    out[((n0 + 8u) * K2) + k0 / 2u + 4u] = half2(w[6], w[7]);
    """
    return mx.fast.metal_kernel(
        name=f'nq_dense_{int(base_k * 10000)}_{int(residual_k * 10000)}',
        input_names=['base', 'residual', 'words', 'variants', 'order', 'experts', 'slots'],
        output_names=['W'], source=source, header=HEADER + WINDOW_HEADER,
        compile_options={'math_mode': 'safe'},
    )


def dense_rotated(planes: dict, meta: dict, experts: mx.array, slots: mx.array | None = None,
                  refinements: dict | None = None) -> mx.array:
    """Dense rotated weights W^T with shape (experts, n, k) in fp16.

    `slots` index the refinement planes (p4, word) for each expert; COLD slots or no refinements use the
    1.5-bit base alone.
    """
    tk, tn = meta['tk'], meta['tn']
    count = experts.size
    if refinements is None:
        residual, words = planes['p4'], planes['word']
        slots = mx.full((count,), COLD, dtype=mx.uint32)
    else:
        residual, words = refinements['p4'], refinements['word']
    order = _matrix_indices(tk, tn, meta['shard_axis'])[2]
    return _dense_kernel(meta['base_K'], meta['res_rule']['K'])(
        inputs=[planes['base'], residual, words, planes['var'], order, experts, slots],
        template=[('TK', tk), ('TN', tn), ('PACKED_VAR', meta.get('var_bits', 8) == 1)],
        grid=(tk * tn * 256, count, 1), threadgroup=(256, 1, 1),
        output_shapes=[(count, meta['n'], meta['k'])], output_dtypes=[mx.float16])[0]


def _low_rank_reduce(name, rank, groups):
    """Reduce per-thread low-rank partial dot products over a threadgroup of `groups` simdgroups."""
    if rank == 0:
        return ''
    return f"""
    for (uint r = 0; r < {rank}u; ++r) {{
        float s = simd_sum({name}_part[r]);
        if (lane == 0) {name}_shared[r * {groups}u + sg] = s;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t < {rank}u) {{
        float s = 0.0f;
        for (uint i = 0; i < {groups}u; ++i) s += {name}_shared[t * {groups}u + i];
        {name}_low[row * {rank}u + t] = s;
    }}
"""


@lru_cache(maxsize=None)
def _prep_up_gate_kernel(k: int, rank_gate: int, rank_up: int):
    """Gather each row's token, rotate it for the gate and up projections, and form the low-rank dots."""
    threads = k // 4
    groups = threads // 32
    low_g = '' if rank_gate == 0 else f"""
    threadgroup float g_shared[{max(rank_gate, 1) * groups}];
    float g_part[{max(rank_gate, 1)}];
    for (uint r = 0; r < {rank_gate}u; ++r)
        g_part[r] = dot(x, nq_load4(lrv_gate, (le * {rank_gate}u + r) * K + i));"""
    low_u = '' if rank_up == 0 else f"""
    threadgroup float u_shared[{max(rank_up, 1) * groups}];
    float u_part[{max(rank_up, 1)}];
    for (uint r = 0; r < {rank_up}u; ++r)
        u_part[r] = dot(x, nq_load4(lrv_up, (le * {rank_up}u + r) * K + i));"""
    source = f"""
    const uint K = {k}u;
    uint row = threadgroup_position_in_grid.x;
    uint t = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    uint token = tokens[row];
    uint le = locals[row];
    uint i = t * 4u;
    // The reference rounds the bf16 MoE input to fp16 before the projections.
    uint o = token * K + i;
    float4 x = float4(half4(float4(float(X[o]), float(X[o + 1]), float(X[o + 2]), float(X[o + 3]))));
    float scale = float(SCALE_NUM) / float(SCALE_DEN);
    float4 vg = nq_fwht128(x * nq_load4(su_gate, le * K + i), lane);
    nq_store4(xr_gate, row * K + i, float4(half4(vg)) * scale);
    float4 vu = nq_fwht128(x * nq_load4(su_up, le * K + i), lane);
    nq_store4(xr_up, row * K + i, float4(half4(vu)) * scale);
    {low_g}
    {low_u}
    {_low_rank_reduce('g', rank_gate, groups)}
    {_low_rank_reduce('u', rank_up, groups)}
    """
    return mx.fast.metal_kernel(
        name=f'nq_prefill_prep_gu_{k}_{rank_gate}_{rank_up}',
        input_names=['X', 'tokens', 'locals', 'su_gate', 'su_up', 'lrv_gate', 'lrv_up'],
        output_names=['xr_gate', 'xr_up', 'g_low', 'u_low'], source=source, header=FWHT_HEADER,
        compile_options={'math_mode': 'safe'}), threads


def _apply_low_rank(target, low, rank, u2, u4, width):
    if rank == 0:
        return ''
    code = f"""
    for (uint r = 0; r < {rank}u; ++r)
        {target} += {low}[row * {rank}u + r] * nq_load4({u2}, (le * {rank}u + r) * {width} + i);"""
    if u4 is not None:
        code += f"""
    if (REFINED) {{
        for (uint r = 0; r < {rank}u; ++r)
            {target} += {low}[row * {rank}u + r] * nq_load4({u4}, (le * {rank}u + r) * {width} + i);
    }}"""
    return code


@lru_cache(maxsize=None)
def _post_up_gate_prep_down_kernel(n: int, rank_gate: int, rank_up: int, rank_down: int, limit: float):
    """Finish the gate and up projections, apply the clamped SwiGLU and rotate the result for down."""
    threads = n // 4
    groups = threads // 32
    low_d = '' if rank_down == 0 else f"""
    threadgroup float d_shared[{max(rank_down, 1) * groups}];
    float d_part[{max(rank_down, 1)}];
    for (uint r = 0; r < {rank_down}u; ++r)
        d_part[r] = dot(act, nq_load4(lrv_down, (le * {rank_down}u + r) * N + i));"""
    source = f"""
    const uint N = {n}u;
    uint row = threadgroup_position_in_grid.x;
    uint t = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    uint le = locals[row];
    uint i = t * 4u;
    float inverse = float(SCALE_DEN) / float(SCALE_NUM);
    float scale = float(SCALE_NUM) / float(SCALE_DEN);
    float4 g = nq_fwht128(nq_load4(part_gate, row * N + i) * inverse, lane) * nq_load4(sv_gate, le * N + i);
    {_apply_low_rank('g', 'g_low', rank_gate, 'u2_gate', 'u4_gate', 'N')}
    float4 u = nq_fwht128(nq_load4(part_up, row * N + i) * inverse, lane) * nq_load4(sv_up, le * N + i);
    {_apply_low_rank('u', 'u_low', rank_up, 'u2_up', 'u4_up', 'N')}
    g = min(g, {limit}f);
    u = clamp(u, -{limit}f, {limit}f);
    float4 act = g * (1.0f / (1.0f + exp(-g))) * u;
    float4 vd = nq_fwht128(act * nq_load4(su_down, le * N + i), lane);
    nq_store4(xr_down, row * N + i, float4(half4(vd)) * scale);
    {low_d}
    {_low_rank_reduce('d', rank_down, groups)}
    """
    return mx.fast.metal_kernel(
        name=f'nq_prefill_post_gu_{n}_{rank_gate}_{rank_up}_{rank_down}',
        input_names=['part_gate', 'part_up', 'locals', 'sv_gate', 'sv_up', 'g_low', 'u_low',
                     'u2_gate', 'u4_gate', 'u2_up', 'u4_up', 'su_down', 'lrv_down'],
        output_names=['xr_down', 'd_low'], source=source, header=FWHT_HEADER,
        compile_options={'math_mode': 'safe'}), threads


@lru_cache(maxsize=None)
def _post_down_kernel(n: int, rank: int, accumulate: bool = False):
    """Finish the down projection and weight each row by its fp16-rounded routing weight.

    With `accumulate`, each weighted row is added into accumulator[tokens[row]] with relaxed atomic float
    adds instead of being written out. The accumulator is an input that this kernel writes behind MLX's
    back: MLX's functional scatter cannot reuse the accumulator's buffer across asynchronous evaluations
    and would copy it once per expert group. Only read the accumulator after synchronizing.
    """
    threads = n // 4
    store = ("""
    device atomic_float* target = (device atomic_float*)(accumulator) + ulong(tokens[row]) * N + i;
    float4 value = d * weight;
    atomic_fetch_add_explicit(target, value.x, memory_order_relaxed);
    atomic_fetch_add_explicit(target + 1, value.y, memory_order_relaxed);
    atomic_fetch_add_explicit(target + 2, value.z, memory_order_relaxed);
    atomic_fetch_add_explicit(target + 3, value.w, memory_order_relaxed);
    if (row == 0 && t == 0) done[0] = 1u;""" if accumulate else
             "\n    *reinterpret_cast<device float4*>(out + row * N + i) = d * weight;")
    source = f"""
    const uint N = {n}u;
    uint row = threadgroup_position_in_grid.x;
    uint t = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint le = locals[row];
    uint i = t * 4u;
    float inverse = float(SCALE_DEN) / float(SCALE_NUM);
    float4 d = nq_fwht128(nq_load4(part_down, row * N + i) * inverse, lane) * nq_load4(sv_down, le * N + i);
    {_apply_low_rank('d', 'd_low', rank, 'u2_down', 'u4_down', 'N')}
    float weight = float(half(route_weights[row]));
    {store}
    """
    inputs = ['part_down', 'locals', 'sv_down', 'd_low', 'u2_down', 'u4_down', 'route_weights']
    return mx.fast.metal_kernel(
        name=f'nq_prefill_post_d_{n}_{rank}' + ('_acc' if accumulate else ''),
        input_names=inputs + (['tokens', 'accumulator'] if accumulate else []),
        output_names=['done' if accumulate else 'out'], source=source, header=FWHT_HEADER,
        compile_options={'math_mode': 'safe'}), threads


def _rank(planes):
    return planes['lrV'].shape[1] if 'lrV' in planes else 0


def _dummy():
    return mx.zeros((4,), dtype=mx.float16)


def group_parameters(base: dict, experts: mx.array, slots: mx.array | None, refinements: dict | None):
    """Gather the per-expert scale and low-rank planes of one expert group into fresh arrays.

    Gathering copies them out of the refinement source, so a staging slot can be reused as soon as
    these arrays and the dense weights have been evaluated.
    """
    params = {}
    for name, (planes, meta) in base.items():
        rank = _rank(planes)
        entry = dict(rank=rank)
        if refinements is None:
            entry['su'] = planes['suh2'][experts]
            entry['sv'] = planes['svh2'][experts]
            entry['u4'] = None
        else:
            ref = refinements[name]
            entry['su'] = ref['suh4'][slots]
            entry['sv'] = ref['svh4'][slots]
            entry['u4'] = ref['lrU4'][slots][:, :rank] if rank else None
        if rank:
            entry['lrv'] = planes['lrV'][experts]
            entry['u2'] = planes['lrU2'][experts]
        params[name] = entry
    return params


def expert_group_forward(X: mx.array, tokens: mx.array, locals_: mx.array, route_weights: mx.array,
                         dense: dict, params: dict, limit: float = 10.0, accumulator: mx.array | None = None) -> mx.array:
    """Routed outputs for the rows of one expert group, weighted by their routing weights.

    X is the (tokens, hidden) MoE input in bf16 or fp16; it is rounded to fp16 as in PagedExperts. Rows are (token, expert) pairs sorted by local expert index;
    `dense` maps each projection to its dense W^T (experts, n, k) and `params` comes from group_parameters.
    Returns float32 rows of shape (rows, hidden), or, with a float32 (tokens, hidden) `accumulator`,
    adds the rows into it in place and returns a one-element marker array to evaluate instead.
    """
    rows = tokens.size
    k = X.shape[1]
    gate, up, down = params['gate'], params['up'], params['down']
    refined = gate['u4'] is not None
    scale = [('SCALE_NUM', 1), ('SCALE_DEN', 1 << PARTIAL_SHIFT)]
    kernel, threads = _prep_up_gate_kernel(k, gate['rank'], up['rank'])
    xr_gate, xr_up, g_low, u_low = kernel(
        inputs=[X, tokens, locals_, gate['su'], up['su'], gate.get('lrv', _dummy()), up.get('lrv', _dummy())],
        template=scale, grid=(rows * threads, 1, 1), threadgroup=(threads, 1, 1),
        output_shapes=[(rows, k), (rows, k), (rows, max(gate['rank'], 1)), (rows, max(up['rank'], 1))],
        output_dtypes=[mx.float16, mx.float16, mx.float32, mx.float32])
    part_gate = mx.gather_mm(xr_gate[:, None, :], dense['gate'].swapaxes(1, 2), rhs_indices=locals_,
                             sorted_indices=True).reshape(rows, -1)
    part_up = mx.gather_mm(xr_up[:, None, :], dense['up'].swapaxes(1, 2), rhs_indices=locals_,
                           sorted_indices=True).reshape(rows, -1)
    n = part_gate.shape[1]
    kernel, threads = _post_up_gate_prep_down_kernel(n, gate['rank'], up['rank'], down['rank'], float(limit))
    xr_down, d_low = kernel(
        inputs=[part_gate, part_up, locals_, gate['sv'], up['sv'], g_low, u_low,
                gate.get('u2', _dummy()), gate['u4'] if refined and gate['rank'] else _dummy(),
                up.get('u2', _dummy()), up['u4'] if refined and up['rank'] else _dummy(),
                down['su'], down.get('lrv', _dummy())],
        template=scale + [('REFINED', refined)], grid=(rows * threads, 1, 1), threadgroup=(threads, 1, 1),
        output_shapes=[(rows, n), (rows, max(down['rank'], 1))], output_dtypes=[mx.float16, mx.float32])
    part_down = mx.gather_mm(xr_down[:, None, :], dense['down'].swapaxes(1, 2), rhs_indices=locals_,
                             sorted_indices=True).reshape(rows, -1)
    accumulate = accumulator is not None
    kernel, threads = _post_down_kernel(k, down['rank'], accumulate)
    inputs = [part_down, locals_, down['sv'], d_low, down.get('u2', _dummy()),
              down['u4'] if refined and down['rank'] else _dummy(), route_weights]
    if accumulate:
        inputs += [tokens, accumulator]
    return kernel(
        inputs=inputs, template=scale + [('REFINED', refined)], grid=(rows * threads, 1, 1),
        threadgroup=(threads, 1, 1), output_shapes=[(1,) if accumulate else (rows, k)],
        output_dtypes=[mx.uint32 if accumulate else mx.float32])[0]
