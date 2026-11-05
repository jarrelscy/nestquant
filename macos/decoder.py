from functools import lru_cache

import mlx.core as mx

PATTERNS = {1.5: (1, 0xAAAA), 2.5: (2, 0xAAAA), 2.8125: (2, 0xFBDE)}

HEADER = r'''
template <typename Buffer>
uint nq_state(Buffer buf, uint ring, uint pos, uint ka, uint mask) {
    uint ring_bits = ka * 256 + popcount(mask) * 16;
    uint bit = ka * pos + (mask == 0xAAAA ? pos / 2 :
        (pos / 16) * popcount(mask) + popcount(mask & ((1u << (pos % 16)) - 1)));
    uint bytes = ring_bits / 8;
    uint off = bit / 8;
    uint start = ring * bytes;
    uint packed = uint(buf[start + off])
        | (uint(buf[start + (off + 1) % bytes]) << 8)
        | (uint(buf[start + (off + 2) % bytes]) << 16);
    return (packed >> (bit % 8)) & 65535;
}
uint nq_sum(uint state) {
    uint h = state * 0x83DCD12Du;
    return (h & 255) + ((h >> 8) & 255) + ((h >> 16) & 255) + (h >> 24);
}
template <typename Buffer>
uint nq_state_wide(Buffer buf, uint ring, uint pos, uint ka, uint mask) {
    uint ring_bits = ka * 256 + popcount(mask) * 16;
    uint bit = ka * pos + (mask == 0xAAAA ? pos / 2 :
        (pos / 16) * popcount(mask) + popcount(mask & ((1u << (pos % 16)) - 1)));
    uint bytes = ring_bits / 8;
    uint off = bit / 8;
    if (off + 2 >= bytes) return nq_state(buf, ring, pos, ka, mask);
    uint absolute = ring * bytes + off;
    uint shift = (absolute % 4) * 8 + bit % 8;
    uint end = (ring + 1) * bytes;
    if ((absolute / 4 + 1) * 4 > end || (shift > 16 && (absolute / 4 + 2) * 4 > end))
        return nq_state(buf, ring, pos, ka, mask);
    // Imported byte views can start at an address with only byte alignment.
    device const packed_uchar4* words = reinterpret_cast<device const packed_uchar4*>(buf);
    uint value = as_type<uint>(uchar4(words[absolute / 4])) >> shift;
    if (shift > 16) value |= as_type<uint>(uchar4(words[absolute / 4 + 1])) << (32 - shift);
    return value & 65535;
}
template <typename Buffer>
uint nq_ring_word(Buffer buf, uint ring, uint lane, uint ka, uint mask) {
    uint bytes = (ka * 256 + popcount(mask) * 16) / 8;
    uint value = 0;
    if (lane < (bytes + 3) / 4) {
        for (uint i = 0; i < 4; ++i)
            value |= uint(buf[ring * bytes + (lane * 4 + i) % bytes]) << (i * 8);
    }
    return value;
}
uint nq_state_shuffled(uint lane_word, uint pos, uint ka, uint mask) {
    uint bytes = (ka * 256 + popcount(mask) * 16) / 8;
    uint bit = ka * pos + (mask == 0xAAAA ? pos / 2 :
        (pos / 16) * popcount(mask) + popcount(mask & ((1u << (pos % 16)) - 1)));
    uint shift = bit % 32;
    uint low = simd_shuffle(lane_word, bit / 32);
    uint high = simd_shuffle(lane_word, (bit / 32 + 1) % ((bytes + 3) / 4));
    return ((low >> shift) | (shift > 16 ? (high << ((32 - shift) & 31)) : 0)) & 65535;
}
'''


@lru_cache(maxsize=1)
def _sum_table():
    values = mx.arange(65536, dtype=mx.uint32) * mx.array(0x83DCD12D, dtype=mx.uint32)
    result = ((values & 255) + ((values >> 8) & 255) + ((values >> 16) & 255) + (values >> 24)).astype(mx.uint16)
    mx.eval(result)
    return result


@lru_cache(maxsize=None)
def _kernel(base_k: float, residual_k: float):
    bka, bmask = PATTERNS[base_k]
    rka, rmask = PATTERNS[residual_k]
    source = f'''
    uint i = thread_position_in_grid.x;
    if (i >= base_shape[0] / {int(base_k * 32)} * 256) return;
    uint ring = i / 256;
    uint unit = ring / 8;
    uint p = i % 256;
    uint sb = nq_sum(nq_state(base, ring, p, {bka}, {bmask}));
    uint sr = nq_sum(nq_state(residual, ring, p, {rka}, {rmask}));
    uint mb = words[unit] & 255;
    uint n = (words[unit] >> 8) & 255;
    float sign = variants[ring] ? -1.0f : 1.0f;
    float a = float(as_type<half>(ushort(0x1eee))) * sign;
    float b = float(as_type<half>(ushort(0xc931))) * sign;
    qbase[i] = half(fma(a, float(1024 + sb), b));
    volatile float rcp = 1.0f / float(mb);
    half ah = half(1.732421875f * rcp);
    volatile float nr = float(n) * rcp;
    volatile float nk = nr * -3.453125f;
    volatile float t = nk + -3.453125f;
    half ch = half(t - 1024.0f * float(ah));
    uint f = (mb * sb + n * sr + 128) >> 8;
    qhot[i] = half(fma(float(ah) * sign, float(1024 + f), float(ch) * sign));
    sums_base[i] = ushort(sb);
    sums_hot[i] = ushort(sr);
    '''
    return mx.fast.metal_kernel(
        name=f'nq_decode_{int(base_k * 10000)}_{int(residual_k * 10000)}',
        input_names=['base', 'residual', 'words', 'variants'],
        output_names=['qbase', 'qhot', 'sums_base', 'sums_hot'],
        source=source,
        header=HEADER,
        compile_options={'math_mode': 'safe'},
    )


def decode_rings(base: mx.array, residual: mx.array, words: mx.array,
                 variants: mx.array, base_k: float, residual_k: float):
    if base_k not in PATTERNS or residual_k not in PATTERNS:
        raise ValueError('Unsupported NestQuant bit pattern')
    units = words.size
    if base.size != units * 8 * int(base_k * 32):
        raise ValueError('Base stream length does not match block words')
    if residual.size != units * 8 * int(residual_k * 32):
        raise ValueError('Residual stream length does not match block words')
    if variants.size != units * 8:
        raise ValueError('Sign variant count does not match block words')
    if base.dtype != mx.uint8 or residual.dtype != mx.uint8:
        raise ValueError('Packed streams must be uint8')
    count = units * 2048
    return _kernel(base_k, residual_k)(
        inputs=[base, residual, words, variants],
        grid=(count, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(units, 8, 256)] * 4,
        output_dtypes=[mx.float16, mx.float16, mx.uint16, mx.uint16],
    )


@lru_cache(maxsize=None)
def _matrix_indices(tk: int, tn: int, shard_axis: str):
    import numpy as np

    chunks = np.repeat(np.arange(tk), tn)
    strips = np.tile(np.arange(tn), tk)
    shards = strips // 16 if shard_axis == 'n' else chunks // 2
    order = np.argsort((shards * tn + strips) * tk + chunks)
    ring_index = []
    for ring in range(8):
        for position in range(256):
            lane, j = divmod(position, 64)
            pair, element = divmod(j, 2)
            tile, row = divmod(pair, 4)
            k = 16 * tile + 2 * lane + 8 * (row >> 1) + element
            ring_index.append(k * 16 + ring + 8 * (row & 1))
    return (mx.array(np.argsort(order).astype(np.int32)),
            mx.array(np.argsort(ring_index).astype(np.int32)),
            mx.array(order.astype(np.uint32)))


def rotated_matrix(rings: mx.array, meta: dict) -> mx.array:
    tk, tn = meta['tk'], meta['tn']
    units, local, _ = _matrix_indices(tk, tn, meta['shard_axis'])
    flat = rings.reshape(tk * tn, 2048)[units][:, local]
    return flat.reshape(tk, tn, 128, 16).transpose(0, 2, 1, 3).reshape(meta['k'], meta['n'])


def dense_projection(rings: mx.array, meta: dict, su: mx.array, sv: mx.array,
                     v: mx.array | None = None, u2: mx.array | None = None,
                     u4: mx.array | None = None) -> mx.array:
    k, n = meta['k'], meta['n']
    rotated = rotated_matrix(rings, meta)
    left = mx.hadamard_transform(rotated.T.reshape(n, -1, 128).astype(mx.float32))
    left = left.reshape(n, k).T.astype(mx.float16)
    left = (left * su[:, None]).astype(mx.float16)
    right = mx.hadamard_transform(left.reshape(k, -1, 128).astype(mx.float32))
    weight = (right.reshape(k, n).astype(mx.float16) * sv[None, :]).astype(mx.float16).T.astype(mx.float32)
    if v is not None:
        weight = weight + u2.astype(mx.float32).T @ v.astype(mx.float32)
        if u4 is not None:
            weight = weight + u4.astype(mx.float32).T @ v.astype(mx.float32)
    return weight


@lru_cache(maxsize=None)
def _matvec_kernel(base_k: float, residual_k: float, threadgroup_size=256, state_load='bytes'):
    if type(threadgroup_size) is not int or threadgroup_size not in (64, 128, 256, 512):
        raise ValueError('Use 64, 128, 256, or 512 threads per banked matvec group')
    if state_load not in ('bytes', 'u32', 'shuffle'):
        raise ValueError('Use bytes, u32, or shuffle packed state reads')
    bka, bmask = PATTERNS[base_k]
    rka, rmask = PATTERNS[residual_k]
    if state_load == 'shuffle':
        for ka, mask in ((bka, bmask), (rka, rmask)):
            bits = ka * 256 + mask.bit_count() * 16
            last = ka * 255 + 15 * mask.bit_count() + (mask & 0x7fff).bit_count()
            if bits > 1024 or (bits % 32 and last % 32 > 16):
                raise ValueError('Packed pattern exceeds the cooperative ring layout')
    unit = ('threadgroup_position_in_grid.x' if threadgroup_size == 256 else
            f'(threadgroup_position_in_grid.x * {threadgroup_size // 32} + simdgroup_index_in_threadgroup) / 8')
    group = ('simdgroup_index_in_threadgroup' if threadgroup_size == 256 else
             f'(threadgroup_position_in_grid.x * {threadgroup_size // 32} + simdgroup_index_in_threadgroup) % 8')
    state = 'nq_state_wide' if state_load == 'u32' else 'nq_state'
    preload = (f'uint base_word = nq_ring_word(base, ring, lane, {bka}, {bmask});\n'
               f'    uint residual_word = refined ? nq_ring_word(residual, refined_ring, lane, {rka}, {rmask}) : 0;'
               if state_load == 'shuffle' else '')
    base_state = (f'nq_state_shuffled(base_word, p, {bka}, {bmask})' if state_load == 'shuffle' else
                  f'{state}(base, ring, p, {bka}, {bmask})')
    residual_state = (f'nq_state_shuffled(residual_word, p, {rka}, {rmask})' if state_load == 'shuffle' else
                      f'{state}(residual, refined_ring, p, {rka}, {rmask})')
    source = f"""
    uint unit = {unit};
    uint row = threadgroup_position_in_grid.y;
    uint expert = BANKED ? expert_ids[row] : 0;
    uint hot = MIXED ? refined_ids[row] : expert;
    bool refined = LEVEL == 4 || (MIXED && hot != 0xffffffffu);
    uint g = {group};
    uint lane = thread_index_in_simdgroup;
    uint flat = order[unit];
    uint chunk = flat / TN;
    uint strip = flat % TN;
    uint ring = unit * 8 + g;
    uint expert_units = expert * TK * TN;
    ring += expert_units * 8;
    uint variant = PACKED_VAR ? ((uint(variants[ring >> 3]) >> (ring & 7)) & 1u) : uint(variants[ring]);
    float sign = variant ? -1.0f : 1.0f;
    float a = float(as_type<half>(ushort(0x1eee))) * sign;
    float b = float(as_type<half>(ushort(0xc931))) * sign;
    uint mb = 0, n = 0;
    uint refined_ring = 0;
    if (refined) {{
        uint refined_unit = hot * TK * TN + unit;
        refined_ring = refined_unit * 8 + g;
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
    }}
    {preload}
    float s0 = 0.0f, s1 = 0.0f;
    for (uint j = 0; j < 8; ++j) {{
        uint p = lane + j * 32;
        uint bs = {base_state};
        uint sb = SUM_LOOKUP ? uint(sum_table[bs]) : nq_sum(bs);
        uint code = sb;
        if (refined) {{
            uint rs = {residual_state};
            uint sr = SUM_LOOKUP ? uint(sum_table[rs]) : nq_sum(rs);
            code = (mb * sb + n * sr + 128) >> 8;
        }}
        float weight = float(half(fma(a, float(1024 + code), b)));
        uint local_lane = p / 64;
        uint pair = (p % 64) / 2;
        uint r = pair % 4;
        uint k = 16 * (pair / 4) + 2 * local_lane + 8 * (r >> 1) + (p % 2);
        float value = weight * x[(row * TK + chunk) * 128 + k];
        if (r & 1) s1 += value; else s0 += value;
    }}
    s0 = simd_sum(s0);
    s1 = simd_sum(s1);
    if (lane == 0) {{
        uint offset = ((row * TN + strip) * TK + chunk) * 16 + g;
        partial[offset] = s0;
        partial[offset + 8] = s1;
    }}
    """
    return mx.fast.metal_kernel(
        name=f'nq_matvec_{int(base_k * 10000)}_{int(residual_k * 10000)}_g{threadgroup_size}_{state_load}',
        input_names=['base', 'residual', 'words', 'variants', 'order', 'x', 'expert_ids', 'refined_ids', 'sum_table'],
        output_names=['partial'], source=source, header=HEADER,
        compile_options={'math_mode': 'safe'},
    )


def projection_matvec(planes: dict, meta: dict, x: mx.array, level: int) -> mx.array:
    if meta.get('var_bits', 8) != 8:
        raise ValueError('Packed signs require the banked projection')
    if level not in (2, 4) or x.shape != (meta['k'],):
        raise ValueError('Expected one input vector and a NestQuant level of 2 or 4')
    tk, tn = meta['tk'], meta['tn']
    _, _, order = _matrix_indices(tk, tn, meta['shard_axis'])
    rotated = mx.hadamard_transform((x.astype(mx.float32) * planes[f'suh{level}'].astype(mx.float32)).reshape(-1, 128)).astype(mx.float16).astype(mx.float32).reshape(-1)
    partial = _matvec_kernel(meta['base_K'], meta['res_rule']['K'])(
        inputs=[planes['base'], planes['p4'], planes['word'], planes['var'], order, rotated, order, order, order],
        template=[('LEVEL', level), ('TK', tk), ('TN', tn), ('BANKED', False), ('MIXED', False),
                  ('PACKED_VAR', False), ('SUM_LOOKUP', False)],
        grid=(tk * tn * 256, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(tn, tk, 16)], output_dtypes=[mx.float32],
    )[0]
    projected = partial.sum(axis=1).reshape(-1)
    result = mx.hadamard_transform(projected.reshape(-1, 128)).reshape(-1) * planes[f'svh{level}'].astype(mx.float32)
    if 'lrV' in planes:
        low = x.astype(mx.float32) @ planes['lrV'].astype(mx.float32).T
        result = result + low @ planes['lrU2'].astype(mx.float32)
        if level == 4:
            result = result + low @ planes['lrU4'].astype(mx.float32)
    return result


def projection_matvec_banked(planes: dict, meta: dict, x: mx.array,
                             route: tuple[int, ...] | mx.array, level: int, *,
                             refinements: dict | None = None, refined_route: mx.array | None = None,
                             threadgroup_size=256, state_load='bytes', sum_lookup=False) -> mx.array:
    if type(threadgroup_size) is not int or threadgroup_size not in (64, 128, 256, 512):
        raise ValueError('Use 64, 128, 256, or 512 threads per banked matvec group')
    if state_load not in ('bytes', 'u32', 'shuffle') or type(sum_lookup) is not bool:
        raise ValueError('Unsupported packed decoder variant')
    if isinstance(route, mx.array) and (route.ndim != 1 or route.dtype != mx.uint32):
        raise ValueError('Device routes require a vector of uint32 expert IDs')
    rows = len(route)
    tk, tn = meta['tk'], meta['tn']
    if (level not in (2, 4) or rows == 0 or x.shape != (rows, tk * 128)
            or meta['k'] != tk * 128):
        raise ValueError('Expected one input vector per routed expert')
    depth = planes['base'].shape[0]
    if depth == 0:
        raise ValueError('Resident bank is empty')
    valid = None
    if isinstance(route, mx.array):
        valid = route < depth
        expert_ids = mx.minimum(route, depth - 1)
    else:
        if any(type(index) is not int or not 0 <= index < depth for index in route):
            raise ValueError('Expert route is outside the resident bank')
        expert_ids = mx.array(route, dtype=mx.uint32)
    units = tk * tn
    var_bits = meta.get('var_bits', 8)
    if type(var_bits) is not int or var_bits not in (1, 8) or planes['var'].dtype != mx.uint8:
        raise ValueError('Unsupported sign packing')
    mixed = refinements is not None
    if mixed != (refined_route is not None) or (mixed and level != 2):
        raise ValueError('Mixed precision requires a base level and a refinement route')
    if (planes['base'].shape != (depth, units * int(meta['base_K'] * 256))
            or planes['var'].shape != (depth, units * var_bits)
            or (level == 4 and (
                planes['p4'].shape != (depth, units * int(meta['res_rule']['K'] * 256))
                or planes['word'].shape != (depth, units)))):
        raise ValueError('Packed bank dimensions do not match the projection')
    _, _, order = _matrix_indices(tk, tn, meta['shard_axis'])
    x = x.astype(mx.float32)
    su = planes[f'suh{level}'][expert_ids]
    sv = planes[f'svh{level}'][expert_ids]
    residual, words = planes['p4'], planes['word']
    refinement_ids = expert_ids
    if mixed:
        if refined_route.shape != (rows,) or refined_route.dtype != mx.uint32:
            raise ValueError('Expected one uint32 refinement slot per row')
        residual, words = refinements['p4'], refinements['word']
        hot = residual.shape[0]
        if (hot == 0 or residual.shape != (hot, units * int(meta['res_rule']['K'] * 256))
                or words.shape != (hot, units) or refinements['suh4'].shape != (hot, meta['k'])
                or refinements['svh4'].shape != (hot, meta['n'])):
            raise ValueError('Refinement bank dimensions do not match the projection')
        cold = refined_route == 0xffffffff
        ready = refined_route < hot
        bounded = mx.minimum(refined_route, hot - 1)
        refinement_ids = mx.where(ready, bounded, mx.array(0xffffffff, dtype=mx.uint32))
        su = mx.where(ready[:, None], refinements['suh4'][bounded], su)
        sv = mx.where(ready[:, None], refinements['svh4'][bounded], sv)
        valid = (cold | ready) if valid is None else valid & (cold | ready)
    scaled = x * su.astype(mx.float32)
    rotated = mx.hadamard_transform(scaled.reshape(rows, -1, 128)).astype(mx.float16).astype(mx.float32)
    partial = _matvec_kernel(meta['base_K'], meta['res_rule']['K'], threadgroup_size, state_load)(
        inputs=[planes['base'], residual, words, planes['var'], order, rotated, expert_ids, refinement_ids,
                _sum_table() if sum_lookup else order],
        template=[('LEVEL', level), ('TK', tk), ('TN', tn), ('BANKED', True), ('MIXED', mixed),
                  ('PACKED_VAR', var_bits == 1), ('SUM_LOOKUP', sum_lookup)],
        grid=(tk * tn * 256, rows, 1), threadgroup=(threadgroup_size, 1, 1),
        output_shapes=[(rows, tn, tk, 16)], output_dtypes=[mx.float32],
    )[0]
    projected = partial.sum(axis=2).reshape(rows, -1, 128)
    result = mx.hadamard_transform(projected).reshape(rows, -1) * sv.astype(mx.float32)
    if 'lrV' in planes:
        v = planes['lrV'][expert_ids].astype(mx.float32).transpose(0, 2, 1)
        low = x[:, None, :] @ v
        result = result + (low @ planes['lrU2'][expert_ids].astype(mx.float32))[:, 0, :]
        if level == 4:
            result = result + (low @ planes['lrU4'][expert_ids].astype(mx.float32))[:, 0, :]
        if mixed:
            if refinements['lrU4'].shape != (hot, v.shape[-1], meta['n']):
                raise ValueError('Refinement bank low-rank dimensions do not match the base')
            update = (low @ refinements['lrU4'][bounded].astype(mx.float32))[:, 0, :]
            result = result + mx.where(ready[:, None], update, 0)
    return mx.where(valid[:, None], result, mx.nan) if valid is not None else result


def projection_matmul(planes: dict, meta: dict, x: mx.array, level: int) -> mx.array:
    if meta.get('var_bits', 8) != 8:
        raise ValueError('Packed signs require the banked projection')
    if x.ndim != 2 or x.shape[1] != meta['k'] or level not in (2, 4):
        raise ValueError('Expected a token matrix and a NestQuant level of 2 or 4')
    if level == 2 and x.shape[0] > 1:
        return mx.stack([projection_matvec(planes, meta, row, level) for row in x])
    if x.shape[0] == 1:
        return projection_matvec(planes, meta, x[0], level)[None]
    rings = decode_rings(planes['base'], planes['p4'], planes['word'], planes['var'],
                         meta['base_K'], meta['res_rule']['K'])[0 if level == 2 else 1]
    weights = rotated_matrix(rings, meta).astype(mx.float32)
    scaled = x.astype(mx.float32) * planes[f'suh{level}'].astype(mx.float32)
    rotated = mx.hadamard_transform(scaled.reshape(x.shape[0], -1, 128)).astype(mx.float16).astype(mx.float32)
    product = rotated.reshape(x.shape[0], meta['k']) @ weights
    result = mx.hadamard_transform(product.reshape(x.shape[0], -1, 128)).reshape(x.shape[0], meta['n'])
    result = result * planes[f'svh{level}'].astype(mx.float32)
    if 'lrV' in planes:
        low = x.astype(mx.float32) @ planes['lrV'].astype(mx.float32).T
        result = result + low @ planes['lrU2'].astype(mx.float32)
        if level == 4:
            result = result + low @ planes['lrU4'].astype(mx.float32)
    return result
