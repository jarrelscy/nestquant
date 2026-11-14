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
def _matvec_kernel(base_k: float, residual_k: float, threadgroup_size=256, state_load='bytes', loop_mode='inline'):
    if type(threadgroup_size) is not int or threadgroup_size not in (64, 128, 256, 512):
        raise ValueError('Use 64, 128, 256, or 512 threads per banked matvec group')
    if state_load not in ('bytes', 'u32', 'shuffle'):
        raise ValueError('Use bytes, u32, or shuffle packed state reads')
    if loop_mode not in ('inline', 'split'):
        raise ValueError('Use inline or split mixed decoder loops')
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
    refinement = f"""            uint rs = {residual_state};
            uint sr = SUM_LOOKUP ? uint(sum_table[rs]) : nq_sum(rs);
            code = (mb * sb + n * sr + 128) >> 8;
"""

    def loop(refinement):
        return f"""    for (uint j = 0; j < 8; ++j) {{
        uint p = lane + j * 32;
        uint bs = {base_state};
        uint sb = SUM_LOOKUP ? uint(sum_table[bs]) : nq_sum(bs);
        uint code = sb;
{refinement}        float weight = float(half(fma(a, float(1024 + code), b)));
        uint local_lane = p / 64;
        uint pair = (p % 64) / 2;
        uint r = pair % 4;
        uint k = 16 * (pair / 4) + 2 * local_lane + 8 * (r >> 1) + (p % 2);
        float value = weight * x[(row * TK + chunk) * 128 + k];
        if (r & 1) s1 += value; else s0 += value;
    }}
"""

    iterations = (loop('        if (refined) {\n' + refinement + '        }\n')
                  if loop_mode == 'inline' else
                  '    if (refined) {\n' + loop(refinement) + '    } else {\n' + loop('') + '    }\n')
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
{iterations}    s0 = simd_sum(s0);
    s1 = simd_sum(s1);
    if (lane == 0) {{
        uint offset = ((row * TN + strip) * TK + chunk) * 16 + g;
        partial[offset] = s0;
        partial[offset + 8] = s1;
    }}
    """
    return mx.fast.metal_kernel(
        name=(f'nq_matvec_{int(base_k * 10000)}_{int(residual_k * 10000)}_g{threadgroup_size}_{state_load}'
              + ('_split' if loop_mode == 'split' else '')),
        input_names=['base', 'residual', 'words', 'variants', 'order', 'x', 'expert_ids', 'refined_ids', 'sum_table'],
        output_names=['partial'], source=source, header=HEADER,
        compile_options={'math_mode': 'safe'},
    )


DECODE_STYLE = 'spark'
# Consecutive ring positions each lane decodes in the fused path. 16 (lanes 0-15 on even chunks,
# 16-31 on odd) makes every bit offset a compile-time constant and halves the per-chunk setup per
# weight; it changes the float32 summation order (relative RMS ~2e-7 versus 8).
LANE_POSITIONS = 16
LOAD_MODE = 'direct_u32'


# Integer arithmetic variants of the spark decode. All produce identical integer results (checked in
# test_window_decoder); they differ only in which instructions the compiler emits. 'mul32' is the original.
SUM_MODE = 'mul'      # 'mul': (p * 0x10001) >> 16 sums the two 16-bit lanes; 'add': (p & 0xFFFF) + (p >> 16)
HASH_MODE = 'mul32'   # 'mul32': s * 0x83DCD12D; 'mul24': two 24-bit multiplies (s < 2^16)
XV_FLOAT = False      # keep each lane's inputs in float registers (converted once at load) instead of fp16
MAD_MODE = 'mul32'    # refined combination mb * sb + n * sr + c: 'mul32' or 'mad24' (all operands < 2^24)
# Refined pair packing: t = mb * sb + n * sr + 0x640080 < 2^23 for any 8-bit mb, n and sb, sr <= 1020, so t >> 8
# already fits 16 bits ('nomask'); 'mask' keeps the previous explicit & 0xFFFF. Identical bits.
PACK_MODE = 'nomask'


def _hash(s):
    if HASH_MODE == 'mul24':
        return f'(mul24({s}, 0xDCD12Du) + (mul24({s}, 0x83u) << 24))'
    return f'{s} * 0x83DCD12Du'


def _sum_lo(p):
    if SUM_MODE == 'add':
        return f'(({p} & 0xFFFFu) + ({p} >> 16))'
    return f'(({p} * 0x00010001u) >> 16)'


def _sum_hi(p):
    if SUM_MODE == 'add':
        return f'((({p} & 0xFFFFu) + ({p} >> 16)) << 16)'
    return f'(({p} * 0x00010001u) & 0xFFFF0000u)'


def _int_suffix():
    variant = (SUM_MODE, HASH_MODE, MAD_MODE)
    suffix = '' if variant == ('mul', 'mul32', 'mul32') else '_' + '_'.join(variant)
    return suffix + ('_xf' if XV_FLOAT else '') + ('_nomask' if PACK_MODE == 'nomask' else '')


def _mad(a, b, c):
    if MAD_MODE == 'mad24':
        return f'mad24({a}, {b}, {c})'
    return f'{a} * {b} + {c}'


@lru_cache(maxsize=None)
def _window_kernel(base_k: float, residual_k: float, style: str = 'window', positions: int = 8,
                   specialized_members=(1, 1), pair: bool = False, only: int = 0, refined_group_positions: int = 8):
    """Banked matvec where each lane decodes `positions` (8 or 16) consecutive ring positions from one bit window.

    The inline kernel gives each lane positions lane + 32 * j, so every weight needs three byte
    loads, a ring-offset calculation and its own input load. Consecutive positions share one
    64-bit window, so a lane reads at most eight bytes per stream and four inputs per chunk.
    Each threadgroup also accumulates CHUNKS chunks of one output strip before writing, which
    shrinks the partial buffer and the reduction that follows it.
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

    def build(positions, lane_setup_in_section=True):
        # Kernel text for a lane mapping: the shared preamble and the accumulation section.
        if positions not in (8, 16) or (positions == 16 and style != 'spark'):
            raise ValueError('Use 8 positions per lane, or 16 with the spark decode')

        def offsets(ka, mask, j):
            # Bit offset of position p0 + j relative to p0. With 8 positions per lane p0 % 16 is 0 or 8,
            # and the offset depends only on that parity; with 16 p0 % 16 == 0 and it is a constant.
            low = ka * j + (mask & ((1 << j) - 1)).bit_count()
            if positions == 16:
                return low, low
            high = ka * j + (mask & ((1 << (8 + j)) - 1)).bit_count() - (mask & 0xFF).bit_count()
            return low, high

        def delta(ka, mask, j):
            low, high = offsets(ka, mask, j)
            return str(low) + 'u' if low == high else f'((lane & 1u) ? {high}u : {low}u)'

        def stream_bits(ka, mask, p):
            return ka * p + (p // 16) * mask.bit_count() + (mask & ((1 << (p % 16)) - 1)).bit_count()

        def max_shift(ka, mask, start=0):
            # Largest bit % 8 of a lane's position p0 + start: 0 when every lane starts on a byte boundary.
            return max(stream_bits(ka, mask, lane * positions + start) % 8 for lane in range(256 // positions))

        def split(ka, mask):
            # 16 positions of a stream may not fit one 64-bit window (2.8125 bits at bit shift 7); then
            # positions 8-15 come from a second window that starts at position p0 + 8.
            return max_shift(ka, mask) + max(offsets(ka, mask, positions - 1)) + 16 > 64

        assert not split(bka, bmask), 'base window too short'
        residual_split = split(rka, rmask)
        if residual_split:
            assert positions == 16 and max_shift(rka, rmask, 8) + (stream_bits(rka, rmask, 15) - stream_bits(rka, rmask, 8)) + 16 <= 64

        def load(buffer, ring, offset, size):
            if size % 4:
                if LOAD_MODE == 'direct_u32':
                    return f'nq_direct_window_unaligned({buffer}, {ring} * {size}u, {offset}, {size}u, uint({buffer}_shape[0]) * uint({buffer}_shape[1]))'
                return f'nq_window({buffer}, {ring} * {size}u, {offset}, {size}u)'
            if LOAD_MODE == 'direct_u32':
                return f'nq_direct_window({buffer}, {ring} * {size // 4}u, {offset}, {size // 4}u)'
            if LOAD_MODE == 'shuffle_u32':
                return f'nq_shuffle_window_u32({buffer}, {ring} * {size // 4}u, {offset}, {size // 4}u, lane)'
            return f'nq_shuffle_window({buffer}, {ring} * {size}u, {offset}, {size // 4}u, lane)'

        base_load = load('base', 'ring', 'base_offset', base_bytes)
        residual_load = load('residual', 'refined_ring', 'residual_offset', residual_bytes)
        # Rings that are not word-aligned (2.8125 bits: 90 bytes) and need more than one 64-bit window per lane:
        # take a 96-bit window from three aligned words (clamped to the buffer) instead of a second window, and
        # fill the bytes of the lanes whose states run past the ring end from the ring's first bytes, which
        # the half-warp's first lane holds in its own window (one shuffle, no divergent byte-load fallback).
        residual_wrap = residual_split and WRAP_SHUFFLE and LOAD_MODE == 'direct_u32' and residual_bytes % 4 != 0
        wrap_lanes = {}
        if residual_wrap:
            need = max_shift(rka, rmask) + max(offsets(rka, rmask, positions - 1)) + 16   # bits of a lane's window
            assert need <= 72, 'window wider than 72 bits'
            for l in range(256 // positions):
                bits = stream_bits(rka, rmask, l * positions)
                valid = residual_bytes - bits // 8
                if (bits % 8) + max(offsets(rka, rmask, positions - 1)) + 16 > 8 * valid:
                    assert l > 0 and valid >= 1
                    wrap_lanes[l] = valid
            assert stream_bits(rka, rmask, 0) == 0
        if residual_wrap:
            patches = ''.join(
                f'''                if ((lane & 15u) == {l}u) {{
                    residual_high = uint(head >> {64 - 8 * v}u);
                    residual_window = (residual_window & 0x{(1 << (8 * v)) - 1:X}ul) | (head << {8 * v}u);
                }}
''' for l, v in wrap_lanes.items())
            residual_load_code = f'''residual_window = nq_direct_window_wide(residual, refined_ring * {residual_bytes}u, residual_offset,
                    uint(residual_shape[0]) * uint(residual_shape[1]), residual_high);
                {{
                    // Lane 0 of each half-warp starts at its ring's first byte.
                    ulong head = ulong(simd_shuffle(uint(residual_window), lane & 16u))
                        | (ulong(simd_shuffle(uint(residual_window >> 32), lane & 16u)) << 32);
{patches}                }}
'''
        else:
            residual_load_code = f'residual_window = {residual_load};\n' + (
                '            residual_windowb = ' + load('residual', 'refined_ring', 'residual_offsetb', residual_bytes) + ';\n'
                if residual_split else '')
        def state(window, shift, ka, mask, j):
            # A 16-bit state needs at most shift + delta + 16 bits. Take it from a 32-bit slice of the
            # window starting at a byte chosen at compile time, instead of shifting the 64-bit window.
            low, high = offsets(ka, mask, j)
            top = max_shift(ka, mask)
            d = delta(ka, mask, j)
            if window == 'residual_window' and residual_wrap:
                for start in (0, 8, 16, 24, 32, 40):
                    if all(start <= v and v + top + 16 - start <= 32 for v in (low, high)):
                        return f'(({window}{start} >> ({shift} + {d} - {start}u)) & 65535u)'
                raise AssertionError('no 32-bit slice covers the state')
            if window == 'residual_window' and residual_split and j >= 8:
                low = high = stream_bits(ka, mask, j) - stream_bits(ka, mask, 8)
                top, d = max_shift(ka, mask, 8), f'{low}u'
                window, shift = 'residual_windowb', 'residual_shiftb'
            if top == 0:
                return f'(uint({window} >> {low}u) & 65535u)'
            for start in (0, 8, 16, 24, 32):
                if all(start <= v and v + top + 16 - start <= 32 for v in (low, high)):
                    return f'(({window}{start} >> ({shift} + {d} - {start}u)) & 65535u)'
            return f'(uint({window} >> ({shift} + {d})) & 65535u)'

        if style == 'spark':
            # Spark's decode (sm120/nqmoe.cu dec_pair) on Metal: two weights per pass. The code lands
            # directly in fp16 bits as 0x6400 + code == 1024 + code, and one half2 fma applies A, B.
            def pair(j):
                # Positions j, j + 1 of a run of eight take inputs kb + {0, 1} (j % 8 < 4) or kb + {8, 9},
                # the second run of a 16-position lane the same plus 16 (xv slots 4 to 7).
                xi = (0 if j % 8 < 4 else 2) + (4 if j >= 8 else 0)
                target = 's1' if j & 2 else 's0'
                sb0 = state('base_window', 'base_shift', bka, bmask, j)
                sb1 = state('base_window', 'base_shift', bka, bmask, j + 1)
                sr0 = state('residual_window', 'residual_shift', rka, rmask, j)
                sr1 = state('residual_window', 'residual_shift', rka, rmask, j + 1)
                return f"""
            {{
                uint h0 = {_hash(sb0)}, h1 = {_hash(sb1)};
                uint p0 = (h0 & 0x00FF00FFu) + ((h0 >> 8) & 0x00FF00FFu);
                uint p1 = (h1 & 0x00FF00FFu) + ((h1 >> 8) & 0x00FF00FFu);
                half2 w;
                if (refined) {{
                    uint g0 = {_hash(sr0)}, g1 = {_hash(sr1)};
                    uint q0 = (g0 & 0x00FF00FFu) + ((g0 >> 8) & 0x00FF00FFu);
                    uint q1 = (g1 & 0x00FF00FFu) + ((g1 >> 8) & 0x00FF00FFu);
                    uint t0 = {_mad('mb', _sum_lo('p0'), _mad('n', _sum_lo('q0'), '0x640080u'))};
                    uint t1 = {_mad('mb', _sum_lo('p1'), _mad('n', _sum_lo('q1'), '0x640080u'))};
                    w = fma(as_type<half2>({'(t0 >> 8)' if PACK_MODE == 'nomask' else '((t0 >> 8) & 0xFFFFu)'} | ((t1 << 8) & 0xFFFF0000u)), a2, b2);
                }} else {{
                    uint bits = ({_sum_lo('p0')} | {_sum_hi('p1')}) + 0x64006400u;
                    w = fma(as_type<half2>(bits), a2, b2);
                }}
                float wa = float(w.x), wb = float(w.y);
                for (uint m = 0; m < MAXR; ++m)
                    if (m < count) {target}[m] = fma(wb, float(xv[m][{xi + 1}]), fma(wa, float(xv[m][{xi}]), {target}[m]));
            }}"""
        steps = []
        for j in range(8 if style != 'spark' else 0):
            x_value = ('x1' if j & 4 == 0 else 'x9') if j & 1 else ('x0' if j & 4 == 0 else 'x8')
            target = 's1' if j & 2 else 's0'
            steps.append(f"""
            {{
                uint code = nq_sum_fast({state('base_window', 'base_shift', bka, bmask, j)});
                if (refined) {{
                    uint sr = nq_sum_fast({state('residual_window', 'residual_shift', rka, rmask, j)});
                    code = (mb * code + n * sr + 128u) >> 8;
                }}
                float w = float(half(fma(a, float(code), b1024)));
                for (uint m = 0; m < MAXR; ++m)
                    if (m < count) {target}[m] += w * float(xv[m][{ {'x0': 0, 'x1': 1, 'x8': 2, 'x9': 3}[x_value] }]);
            }}""")
        source = f"""
        uint group = threadgroup_position_in_grid.x;
        uint row = threadgroup_position_in_grid.y;
        uint g = simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;
        uint strip = group % TN;
        uint chunk_group = group / TN;
        uint depth = uint(variants_shape[0]);
        uint hot_rows = uint(words_shape[0]);
        uint expert = BANKED ? min(expert_ids[row], depth - 1) : 0;
        uint hot = expert;
        if (MIXED) {{
            hot = refined_ids[row];
            hot = hot < hot_rows ? hot : 0xffffffffu;
        }}
        bool refined = LEVEL == 4 || (MIXED && hot != 0xffffffffu);
        // Rows that route to the same expert and refinement slot share one decode of the weights.
        // groups[row] holds the member count (0 for rows another row leads) and up to MAXR member rows.
        uint gbase = row * (MAXR + 1u);
        uint count = MAXR > 1 ? groups[gbase] : 1u;
        if (count == 0) return;
        /*ONLY_SPLIT*/
        // {positions} consecutive ring positions per lane{'' if positions == 8 else '; lanes 0-15 take even chunks, 16-31 odd'}.
        uint p0 = {'lane * 8u' if positions == 8 else '(lane % 16u) * 16u'};
        uint kb = {'16u * (lane % 8u) + 2u * (lane / 8u)' if positions == 8 else '32u * (lane % 4u) + 2u * ((lane % 16u) / 4u)'};
        uint base_bit = {bit(bka, bmask, 'p0')};
        uint base_shift = base_bit % 8u;
        uint base_offset = base_bit / 8u;
        uint residual_bit = {bit(rka, rmask, 'p0')};
        uint residual_shift = residual_bit % 8u;
        uint residual_offset = residual_bit / 8u;
    {'''    uint residual_bitb = ''' + bit(rka, rmask, 'p0 + 8u') + ''';
        uint residual_shiftb = residual_bitb % 8u;
        uint residual_offsetb = residual_bitb / 8u;
    ''' if residual_split else ''}    float s0[MAXR], s1[MAXR];
        for (uint m = 0; m < MAXR; ++m) {{ s0[m] = 0.0f; s1[m] = 0.0f; }}
        for (uint c = 0; c < CHUNKS{'' if positions == 8 else ' / 2u'}; ++c) {{
            uint chunk = chunk_group * CHUNKS + {'c' if positions == 8 else '2u * c + lane / 16u'};
            // Same as inverse[chunk * TN + strip] for both shard layouts (checked in the tests).
            uint unit = SHARD_N ? strip * TK + chunk : (chunk >> 1) * (TN * 2u) + strip * 2u + (chunk & 1u);
            uint ring = (expert * TK * TN + unit) * 8u + g;
            uint variant = PACKED_VAR ? ((uint(variants[ring >> 3]) >> (ring & 7u)) & 1u) : uint(variants[ring]);
            float sign = variant ? -1.0f : 1.0f;
            float a = float(as_type<half>(ushort(0x1eee))) * sign;
            float b = float(as_type<half>(ushort(0xc931))) * sign;
            uint mb = 0, n = 0;
            ulong residual_window = 0, residual_windowb = 0;
            uint residual_high = 0;
            if (refined) {{
                uint refined_unit = hot * TK * TN + unit;
                uint refined_ring = refined_unit * 8u + g;
                uint word = uint(words[refined_unit]);
                mb = word & 255;
                n = (word >> 8) & 255;
                half ah, ch;
                if (AC_TABLE) {{
                    // (A', C) for this (Mb, N), precomputed by the same arithmetic in nq_ac_table.
                    half2 ac = as_type<half2>(ac_table[word & 0xFFFFu]);
                    ah = ac.x; ch = ac.y;
                }} else {{
                    volatile float rcp = 1.0f / float(mb);
                    ah = half(1.732421875f * rcp);
                    volatile float nr = float(n) * rcp;
                    volatile float nk = nr * -3.453125f;
                    volatile float t = nk + -3.453125f;
                    ch = half(t - 1024.0f * float(ah));
                }}
                a = float(ah) * sign;
                b = float(ch) * sign;
                {residual_load_code}
            }}
            ulong base_window = {base_load};
            uint base_window0 = uint(base_window), base_window8 = uint(base_window >> 8),
                 base_window16 = uint(base_window >> 16), base_window24 = uint(base_window >> 24),
                 base_window32 = uint(base_window >> 32);
            uint residual_window0 = uint(residual_window), residual_window8 = uint(residual_window >> 8),
                 residual_window16 = uint(residual_window >> 16), residual_window24 = uint(residual_window >> 24),
                 residual_window32 = uint(residual_window >> 32),
                 residual_window40 = uint(residual_window >> 40) | (residual_high << 24);
            uint residual_windowb0 = uint(residual_windowb), residual_windowb8 = uint(residual_windowb >> 8),
                 residual_windowb16 = uint(residual_windowb >> 16), residual_windowb24 = uint(residual_windowb >> 24),
                 residual_windowb32 = uint(residual_windowb >> 32);
            // a * (1024 + code) + b == a * code + (1024 * a + b) exactly: a has 11 significant bits, so
            // both products and the folded constant are exact in float32 and fma still rounds once.
            float b1024 = fma(a, 1024.0f, b);
            half2 a2 = half2(half(a)), b2 = half2(half(b));
            // Inputs stay in their storage precision (fp16 for XHALF) until use: half the registers.
            {'float' if XV_FLOAT else 'metal::conditional_t<XHALF, half, float>'} xv[MAXR][{positions // 2}];
            for (uint m = 0; m < MAXR; ++m) {{
                if (m < count) {{
                    uint member = MAXR > 1 ? groups[gbase + 1u + m] : row;
                    uint xi = (member * TK + chunk) * 128u + kb;
                    for (uint q = 0; q < {positions // 8}u; ++q) {{
                        uint xq = xi + 16u * q;
                        if (XHALF) {{
                            half2 lo = nq_half2(x, xq >> 1), hi = nq_half2(x, (xq + 8u) >> 1);
                            xv[m][4 * q] = lo.x; xv[m][4 * q + 1] = lo.y; xv[m][4 * q + 2] = hi.x; xv[m][4 * q + 3] = hi.y;
                        }} else {{
                            xv[m][4 * q] = x[xq]; xv[m][4 * q + 1] = x[xq + 1u];
                            xv[m][4 * q + 2] = x[xq + 8u]; xv[m][4 * q + 3] = x[xq + 9u];
                        }}
                    }}
                }}
            }}
    {''.join(steps) if style != 'spark' else ''.join(pair(j) for j in range(0, positions, 2))}
        }}
        for (uint m = 0; m < MAXR; ++m) {{
            if (m < count) {{
                float a0 = simd_sum(s0[m]);
                float a1 = simd_sum(s1[m]);
                if (lane == 0) {{
                    uint member = MAXR > 1 ? groups[gbase + 1u + m] : row;
                    uint offset = ((member * TN + strip) * (TK / CHUNKS) + chunk_group) * 16u + g;
                    partial[offset] = a0;
                    partial[offset + 8] = a1;
                }}
            }}
        }}
        """
        # Refined and base-only rows take separate copies of the accumulation, so the chunk loop has
        # no per-pair branch on the row's precision. Within each, member counts up to SPECIALIZED
        # get a copy with the count fixed at compile time: register arrays sized to the count and no
        # per-member predicates. Rows without partners take the one-row copy, so grouping costs nothing
        # when experts do not repeat. The per-member arithmetic is identical in every copy, so results
        # do not depend on the copy taken.
        section_start = source.index('    // {positions} consecutive'.format(positions=positions)
                                     if lane_setup_in_section else '    float s0[MAXR], s1[MAXR];')
        section = source[section_start:]
        assert section.count('(refined)') >= 2
        return source[:section_start], section


    def specialized(body, limit):
        if limit == 0:
            return body
        cases = ''.join(
            f'    {"if" if c == 1 else "else if"} (MAXR >= {c} && count == {c}) {{\n'
            + body.replace('m < count', f'm < {c}u').replace('MAXR', str(c)) + '    }\n'
            for c in range(1, limit + 1))
        return cases + f'    else if (MAXR > {limit}) {{\n' + body + '    }\n'

    spec_base, spec_refined = specialized_members
    # The lane setup stays ahead of the copies: computed inside each one, the kernel ran up to 30%
    # slower on multi-row calls.
    preamble, section = build(positions, False)
    if positions != 8 and refined_group_positions == 8:
        # Previous layout: refined rows with partners at eight positions per lane, in a copy that redefines
        # the lane setup in its own scope. It sums in a different order than a refined row without partners.
        _, section8 = build(8, True)
        single = section.replace('(refined)', '(true)').replace('m < count', 'm < 1u').replace('MAXR', '1')
        refined_cases = ('    if (count == 1) {\n' + single + '    } else {\n'
                         + specialized(section8.replace('(refined)', '(true)'), spec_refined) + '    }\n')
    else:
        # Refined rows with and without partners at the main lane mapping, by default (spec_refined == 0)
        # in one predicated copy: the refined decode is long, and each further unrolled copy measured
        # slower (code size), while a refined launch holding only this copy runs grouped rows at the
        # cost of one decode. Every row then sums in the same order whether or not it has partners.
        refined_cases = specialized(section.replace('(refined)', '(true)'), spec_refined)
    source = (preamble + '    if (refined) {\n' + refined_cases + '    } else {\n'
              + specialized(section.replace('(refined)', '(false)'), spec_base) + '    }\n')
    name = (f'nq_matvec_{style}_{LOAD_MODE}_{int(base_k * 10000)}_{int(residual_k * 10000)}_m{spec_base}{spec_refined}'
            f'_p{positions}' + ('' if refined_group_positions == 8 else f'r{refined_group_positions}')
            + ('' if WRAP_SHUFFLE else '_fb') + _int_suffix())
    inputs = ['base', 'residual', 'words', 'variants', 'inverse', 'x', 'expert_ids', 'refined_ids', 'groups', 'ac_table']
    outputs = ['partial']
    # Precision-split launches (only=1: refined rows, only=2: base rows; the other rows' partial sums are
    # left unwritten and the output kernels read each row from its own launch). Each launch then carries
    # only its own path's registers and code.
    split = {0: '', 1: '        if (!refined) return;\n', 2: '        if (refined) return;\n'}[only]
    source = source.replace('        /*ONLY_SPLIT*/\n', split)
    if only:
        name += f'_only{only}'
    if pair:
        # Two projections of the same rows (gate and up) in one launch: grid z selects the planes, the input
        # and the output; everything else (routing, groups, geometry) is shared. Same arithmetic per row.
        own = ('base', 'residual', 'words', 'variants', 'x', 'partial')
        prefix = '    uint pair_z = threadgroup_position_in_grid.z;\n'
        for item in own:
            prefix += f'    auto {item} = pair_z ? {item}_u : {item}_g;\n'
            if f'{item}_shape' in source:
                prefix += f'    auto {item}_shape = {item}_g_shape;\n'
        source = prefix + source
        inputs = [f'{item}_{suffix}' if item in own else item for item in inputs
                  for suffix in (('g', 'u') if item in own else ('',))]
        outputs = ['partial_g', 'partial_u']
        name += '_pair'
    return mx.fast.metal_kernel(
        name=name, input_names=inputs, output_names=outputs, source=source, header=HEADER + WINDOW_HEADER,
        compile_options={'math_mode': 'safe'},
    )


# Optionally run mixed-precision verify blocks as a refined-only and a base-only launch. That wins when layers
# run independently (throughput: verify-8 half-4-bit -26%), but in the model each layer waits for the previous
# one, and there the single launch with the one-copy refined path below is as fast or faster (latency: verify-4
# -16% vs -13% split, verify-8 -11% both; full-model verify agrees), so it is off by default.
SPLIT_PRECISION = __import__('os').environ.get('NQ_SPLIT_PRECISION', '0') != '0'
SPLIT_MIN_ROWS = 9   # split only verify blocks; single-token decode (8 rows, no groups) measured slower split
# Lane positions of refined rows with partners: 16 (the main mapping, same summation order as rows without
# partners, so grouping changes no value) or 8 (the previous layout: 16 measured slower there only because the
# kernel also held specialized refined copies; with one copy 16 is faster).
REFINED_GROUP_POSITIONS = 16
# 2.8125-bit (90-byte) rings: one 96-bit window per lane, ring-wrapping lanes filled by a shuffle (True), or a
# second window with a divergent byte-load fallback (False, the previous code). Identical results.
WRAP_SHUFFLE = True


def _window(base_k, residual_k, positions, pair, inputs, mixed, **call):
    """Run the window matvec. Returns (outputs for refined rows, outputs for base rows): the same launch's
    outputs unless SPLIT_PRECISION splits a mixed call into a refined-only and a base-only launch
    (identical per-row arithmetic)."""
    kernel = lambda only: _window_kernel(base_k, residual_k, DECODE_STYLE, positions, tuple(SPECIALIZED_MEMBERS),
                                         pair, only, REFINED_GROUP_POSITIONS)
    if not (SPLIT_PRECISION and mixed and call['grid'][1] >= SPLIT_MIN_ROWS):
        out = kernel(0)(inputs=inputs, **call)
        return out, out
    return kernel(1)(inputs=inputs, **call), kernel(2)(inputs=inputs, **call)


WINDOW_HEADER = r'''
template <typename Buffer>
ulong nq_window(Buffer buf, uint start, uint offset, uint bytes) {
    ulong value = 0;
    if (offset + 8u <= bytes) {
        for (uint i = 0; i < 8u; ++i)
            value |= ulong(buf[start + offset + i]) << (8u * i);
    } else {
        for (uint i = 0; i < 8u; ++i) {
            uint o = offset + i;
            value |= ulong(buf[start + (o >= bytes ? o - bytes : o)]) << (8u * i);
        }
    }
    return value;
}
inline uint nq_word(const device uchar* b, uint i) { return reinterpret_cast<const device uint*>(b)[i]; }
inline half2 nq_half2(const device half* b, uint i) { return reinterpret_cast<const device half2*>(b)[i]; }
inline half2 nq_half2(const constant half* b, uint i) { return reinterpret_cast<const constant half2*>(b)[i]; }
inline half2 nq_half2(const device float* b, uint i) { return half2(b[2 * i], b[2 * i + 1]); }
inline half2 nq_half2(const constant float* b, uint i) { return half2(b[2 * i], b[2 * i + 1]); }
inline uint nq_word(const constant uchar* b, uint i) { return reinterpret_cast<const constant uint*>(b)[i]; }
template <typename Buffer>
ulong nq_shuffle_window_u32(Buffer buf, uint start_word, uint offset, uint ring_words, uint lane) {
    // Lanes below ring_words load one aligned word of the ring; the others take theirs by shuffle.
    uint mine = lane < ring_words ? nq_word(buf, start_word + lane) : 0u;
    uint q = offset >> 2;
    uint q1 = q + 1u >= ring_words ? q + 1u - ring_words : q + 1u;
    uint q2 = q + 2u >= ring_words ? q + 2u - ring_words : q + 2u;
    uint w0 = simd_shuffle(mine, q), w1 = simd_shuffle(mine, q1), w2 = simd_shuffle(mine, q2);
    uint shift = (offset & 3u) * 8u;
    ulong low = ulong(w0) | (ulong(w1) << 32);
    return shift ? (low >> shift) | (ulong(w2) << (64u - shift)) : low;
}
template <typename Buffer>
ulong nq_direct_window(Buffer buf, uint start_word, uint offset, uint ring_words) {
    // Each lane loads the aligned words it needs; neighbouring lanes hit the same cache lines.
    uint q = offset >> 2;
    uint q1 = q + 1u >= ring_words ? q + 1u - ring_words : q + 1u;
    uint q2 = q + 2u >= ring_words ? q + 2u - ring_words : q + 2u;
    uint w0 = nq_word(buf, start_word + q), w1 = nq_word(buf, start_word + q1), w2 = nq_word(buf, start_word + q2);
    uint shift = (offset & 3u) * 8u;
    ulong low = ulong(w0) | (ulong(w1) << 32);
    return shift ? (low >> shift) | (ulong(w2) << (64u - shift)) : low;
}
template <typename Buffer>
ulong nq_direct_window_unaligned(Buffer buf, uint start, uint offset, uint bytes, uint total) {
    // Rings whose length is not a multiple of four: read the three aligned words that cover the
    // eight bytes when they lie inside the ring and the buffer, else fall back to byte loads.
    uint a = start + offset;
    uint q = a >> 2;
    if (offset + 8u <= bytes && (q + 3u) * 4u <= total) {
        uint w0 = nq_word(buf, q), w1 = nq_word(buf, q + 1u), w2 = nq_word(buf, q + 2u);
        uint shift = (a & 3u) * 8u;
        ulong low = ulong(w0) | (ulong(w1) << 32);
        return shift ? (low >> shift) | (ulong(w2) << (64u - shift)) : low;
    }
    return nq_window(buf, start, offset, bytes);
}
template <typename Buffer>
ulong nq_direct_window_wide(Buffer buf, uint start, uint offset, uint total, thread uint& high) {
    // Bytes [offset, offset + 8) of a ring as the window and bytes [offset + 8, offset + 12 - (start + offset) % 4)
    // in `high`, from three aligned words. Words past the buffer end are clamped to its last word: they lie past
    // the ring end, so the caller replaces those bytes.
    uint a = start + offset;
    uint q = a >> 2, last = total / 4u - 1u;
    uint w0 = nq_word(buf, q), w1 = nq_word(buf, min(q + 1u, last)), w2 = nq_word(buf, min(q + 2u, last));
    uint shift = (a & 3u) * 8u;
    ulong low = ulong(w0) | (ulong(w1) << 32);
    high = shift ? (w2 >> shift) : w2;
    return shift ? (low >> shift) | (ulong(w2) << (64u - shift)) : low;
}
template <typename Buffer>
ulong nq_shuffle_window(Buffer buf, uint start, uint offset, uint ring_words, uint lane) {
    // Lanes below ring_words each load one word of a word-aligned ring; every lane then assembles its
    // eight-byte window from three neighbouring words. Word indices wrap around the ring.
    uint mine = 0u;
    if (lane < ring_words) {
        uint at = start + lane * 4u;
        mine = uint(buf[at]) | (uint(buf[at + 1u]) << 8) | (uint(buf[at + 2u]) << 16) | (uint(buf[at + 3u]) << 24);
    }
    uint q = offset >> 2;
    uint q1 = q + 1u >= ring_words ? q + 1u - ring_words : q + 1u;
    uint q2 = q + 2u >= ring_words ? q + 2u - ring_words : q + 2u;
    uint w0 = simd_shuffle(mine, q);
    uint w1 = simd_shuffle(mine, q1);
    uint w2 = simd_shuffle(mine, q2);
    uint shift = (offset & 3u) * 8u;
    ulong low = ulong(w0) | (ulong(w1) << 32);
    return shift ? (low >> shift) | (ulong(w2) << (64u - shift)) : low;
}
uint nq_sum_fast(uint state) {
    uint h = state * 0x83DCD12Du;
    uint pairs = (h & 0x00FF00FFu) + ((h >> 8) & 0x00FF00FFu);
    return (pairs & 0xFFFFu) + (pairs >> 16);
}
'''

WINDOW_CHUNKS = 16
# Member counts up to these (base-only rows, refined rows) get a compile-time specialized
# accumulation in the window kernel; larger groups use the predicated MAXR loop (refined rows: only it).
SPECIALIZED_MEMBERS = (4, 0)
USE_AC_TABLE = True


@lru_cache(maxsize=1)
def _ac_table():
    """(A', C) as half2 bits for every 16-bit refinement word, from the kernel's own arithmetic."""
    kernel = mx.fast.metal_kernel(name='nq_ac_table', input_names=['dummy'], output_names=['table'], source='''
    uint word = thread_position_in_grid.x;
    uint mb = word & 255u, n = (word >> 8) & 255u;
    if (mb == 0u) { table[word] = 0u; return; }
    volatile float rcp = 1.0f / float(mb);
    half ah = half(1.732421875f * rcp);
    volatile float nr = float(n) * rcp;
    volatile float nk = nr * -3.453125f;
    volatile float t = nk + -3.453125f;
    half ch = half(t - 1024.0f * float(ah));
    table[word] = as_type<uint>(half2(ah, ch));
    ''')
    table = kernel(inputs=[mx.zeros((1,), dtype=mx.uint32)], grid=(65536, 1, 1), threadgroup=(256, 1, 1),
                   output_shapes=[(65536,)], output_dtypes=[mx.uint32])[0]
    mx.eval(table)
    return table
GROUP_SHARED_EXPERTS = True
GROUP_MAXR_OVERRIDE = None


_GROUPS = """
    // For this row: how many rows it leads (0 if another row leads it) and which, in runs of MAXR.
    uint total = uint(route_shape[0]);
    uint depth = uint(variants_shape[0]);
    uint hot_rows = uint(words_shape[0]);
    uint expert = min(route[row], depth - 1);
    uint hot = MIXED ? refined[row] : expert;
    if (MIXED) hot = hot < hot_rows ? hot : 0xffffffffu;
    uint before = 0;
    uint count = 1;
    uint base = row * (MAXR + 1);
    groups[base + 1] = row;
    for (uint r = 0; r < total; ++r) {
        if (r == row) continue;
        uint other = min(route[r], depth - 1);
        uint other_hot = MIXED ? refined[r] : other;
        if (MIXED) other_hot = other_hot < hot_rows ? other_hot : 0xffffffffu;
        if (other != expert || other_hot != hot) continue;
        if (r < row) before++;
        else if (count < MAXR) groups[base + 1 + count++] = r;
    }
    groups[base] = before % MAXR == 0 ? count : 0u;
"""


_FWHT = """
    threadgroup float buf[128];
    buf[t] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint h = 1; h < 128; h <<= 1) {
        float v = buf[t];
        float u = buf[t ^ h];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf[t] = (t & h) ? (u - v) : (v + u);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    value = buf[t] * 0.08838834764831845f;
"""


@lru_cache(maxsize=None)
def _prep_kernel(grouped: bool = False):
    """Select each row's input scale, scale, apply the 128-point Hadamard transform and round to fp16.

    With grouped=True the first threadgroup of each row also writes the row's expert group (see
    _GROUPS), which saves the window kernel's grouping launch.
    """
    source = """
    uint t = thread_position_in_threadgroup.x;
    uint block = threadgroup_position_in_grid.x;
    uint row = threadgroup_position_in_grid.y;
    uint e = min(route[row], uint(base_su_shape[0]) - 1u);
    uint slot = MIXED ? refined[row] : 0xffffffffu;
    bool ready = MIXED && slot < uint(ref_su_shape[0]);
    uint k = block * 128u + t;
    float su = ready ? float(ref_su[slot * K + k]) : float(base_su[e * K + k]);
    float value = x[row * K + k] * su;
""" + _FWHT + """
    rotated[row * K + k] = half(value);
"""
    names, outputs = ['x', 'base_su', 'ref_su', 'route', 'refined'], ['rotated']
    if grouped:
        source += "    if (block == 0 && t == 0) {\n" + _GROUPS + "    }\n"
        names += ['variants', 'words']
        outputs += ['groups']
    return mx.fast.metal_kernel(name='nq_prep' + ('_grouped' if grouped else ''), input_names=names,
                                output_names=outputs, source=source)


@lru_cache(maxsize=None)
def _post_kernel(rank: int):
    """Reduce matvec partials, apply the inverse rotation and output scale, add low-rank terms, mask bad slots."""
    low_rank = '' if rank == 0 else """
    threadgroup float lows[RANK][4];
    float low[RANK];
    for (uint r = 0; r < RANK; ++r) {
        float dot = 0.0f;
        for (uint k = t; k < K; k += 128u)
            dot += x[row * K + k] * float(lrV[(e * RANK + r) * K + k]);
        dot = simd_sum(dot);
        if ((t & 31u) == 0) lows[r][t / 32u] = dot;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint r = 0; r < RANK; ++r) {
        low[r] = lows[r][0] + lows[r][1] + lows[r][2] + lows[r][3];
        result += low[r] * float(lrU2[(e * RANK + r) * N + n]);
    }
    if (ready) {
        for (uint r = 0; r < RANK; ++r)
            result += low[r] * float(lrU4[(slot * RANK + r) * N + n]);
    }
"""
    source = """
    uint t = thread_position_in_threadgroup.x;
    uint block = threadgroup_position_in_grid.x;
    uint row = threadgroup_position_in_grid.y;
    uint raw = route[row];
    uint depth = uint(base_sv_shape[0]);
    uint e = min(raw, depth - 1u);
    uint slot = MIXED ? refined[row] : 0xffffffffu;
    bool ready = MIXED && slot < uint(ref_sv_shape[0]);
    bool valid = raw < depth && (!MIXED || slot == 0xffffffffu || ready);
    uint n = block * 128u + t;
    uint strip = n / 16u, j = n % 16u;
    float value = 0.0f;
    // Refined rows' partial sums come from the refined launch (the same buffer unless split by precision).
    auto source = ready ? partial_r : partial;
    for (uint g = 0; g < G; ++g)
        value += source[((row * TN + strip) * G + g) * 16u + j];
""" + _FWHT + """
    float sv = ready ? float(ref_sv[slot * N + n]) : float(base_sv[e * N + n]);
    float result = value * sv;
""" + low_rank + """
    out[row * N + n] = valid ? result : NAN;
    """
    names = ['partial', 'base_sv', 'ref_sv', 'route', 'refined', 'x', 'partial_r']
    if rank:
        names += ['lrV', 'lrU2', 'lrU4']
    return mx.fast.metal_kernel(name=f'nq_post_r{rank}', input_names=names, output_names=['out'], source=source)


def projection_fused(planes: dict, meta: dict, x: mx.array, route: mx.array, *,
                     refinements: dict | None = None, refined_route: mx.array | None = None) -> mx.array:
    """Level-2 banked projection in three kernels: input rotation, window matvec and output rotation.

    Matches projection_matvec_banked(..., loop_mode='window') up to float32 rounding, with far fewer
    MLX operations to encode per call.
    """
    rows = route.size
    tk, tn, k, n = meta['tk'], meta['tn'], meta['k'], meta['n']
    depth = planes['base'].shape[0]
    mixed = refinements is not None
    hot = refinements['p4'].shape[0] if mixed else 1
    x = x.astype(mx.float32)
    refined = refined_route if mixed else route
    ref_su = refinements['suh4'] if mixed else planes['suh2']
    ref_sv = refinements['svh4'] if mixed else planes['svh2']
    common = [('MIXED', mixed)]
    chunks = WINDOW_CHUNKS if tk % WINDOW_CHUNKS == 0 else 1
    # With several tokens (speculative verification, prefill chunks) rows can share an expert.
    # Up to 64 rows (8-token verification) groups of at most four decode fastest: the member
    # arrays stay small, and longer runs of one expert are rare.
    max_rows = max(1, min(4 if rows <= 64 else 8, rows // 8)) if GROUP_SHARED_EXPERTS else 1
    if GROUP_MAXR_OVERRIDE is not None and GROUP_SHARED_EXPERTS:
        max_rows = GROUP_MAXR_OVERRIDE
    residual, words = (refinements['p4'], refinements['word']) if mixed else (planes['p4'], planes['word'])
    var_bits = meta.get('var_bits', 8)
    grouped = max_rows > 1
    prep_inputs = [x, planes['suh2'], ref_su, route, refined] + ([planes['var'], words] if grouped else [])
    prepared = _prep_kernel(grouped)(
        inputs=prep_inputs, template=[('K', k), ('MAXR', max_rows), *common],
        grid=(k, rows, 1), threadgroup=(128, 1, 1),
        output_shapes=[(rows, k)] + ([(rows * (max_rows + 1),)] if grouped else []),
        output_dtypes=[mx.float16] + ([mx.uint32] if grouped else []))
    rotated = prepared[0]
    groups = prepared[1] if grouped else route
    positions = LANE_POSITIONS if chunks % 2 == 0 else 8
    (partial_r,), (partial,) = _window(meta['base_K'], meta['res_rule']['K'], positions, False,
        [planes['base'], residual, words, planes['var'], _matrix_indices(tk, tn, meta['shard_axis'])[0],
         rotated, route, refined, groups, _ac_table()], mixed,
        template=[('LEVEL', 2), ('TK', tk), ('TN', tn), ('BANKED', True), ('MIXED', mixed),
                  ('PACKED_VAR', var_bits == 1), ('CHUNKS', chunks), ('MAXR', max_rows),
                  ('SHARD_N', meta['shard_axis'] == 'n'), ('XHALF', True), ('AC_TABLE', USE_AC_TABLE)],
        grid=(tn * (tk // chunks) * 256, rows, 1), threadgroup=(256, 1, 1),
        output_shapes=[(rows, tn, tk // chunks, 16)], output_dtypes=[mx.float32])
    rank = planes['lrV'].shape[1] if 'lrV' in planes else 0
    inputs = [partial, planes['svh2'], ref_sv, route, refined, x, partial_r]
    if rank:
        lr_u4 = refinements['lrU4'] if mixed else planes['lrU2']
        inputs += [planes['lrV'], planes['lrU2'], lr_u4]
    return _post_kernel(rank)(
        inputs=inputs, template=[('K', k), ('N', n), ('TN', tn), ('G', tk // chunks), ('RANK', max(rank, 1)), *common],
        grid=(n, rows, 1), threadgroup=(128, 1, 1), output_shapes=[(rows, n)], output_dtypes=[mx.float32])[0]


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
                             threadgroup_size=256, state_load='bytes', sum_lookup=False, loop_mode='inline') -> mx.array:
    if type(threadgroup_size) is not int or threadgroup_size not in (64, 128, 256, 512):
        raise ValueError('Use 64, 128, 256, or 512 threads per banked matvec group')
    if (state_load not in ('bytes', 'u32', 'shuffle') or type(sum_lookup) is not bool
            or loop_mode not in ('inline', 'split', 'window')
            or (loop_mode == 'window' and (state_load != 'bytes' or sum_lookup or threadgroup_size != 256))):
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
    if loop_mode == 'window':
        chunks = WINDOW_CHUNKS if tk % WINDOW_CHUNKS == 0 else 1
        inverse = _matrix_indices(tk, tn, meta['shard_axis'])[0]
        partial = _window_kernel(meta['base_K'], meta['res_rule']['K'])(
            inputs=[planes['base'], residual, words, planes['var'], inverse, rotated, expert_ids, refinement_ids, expert_ids,
                    _ac_table()],
            template=[('LEVEL', level), ('TK', tk), ('TN', tn), ('BANKED', True), ('MIXED', mixed),
                      ('PACKED_VAR', var_bits == 1), ('CHUNKS', chunks), ('MAXR', 1),
                      ('SHARD_N', meta['shard_axis'] == 'n'), ('XHALF', False), ('AC_TABLE', False)],
            grid=(tn * (tk // chunks) * 256, rows, 1), threadgroup=(256, 1, 1),
            output_shapes=[(rows, tn, tk // chunks, 16)], output_dtypes=[mx.float32],
        )[0]
    else:
        partial = _matvec_kernel(meta['base_K'], meta['res_rule']['K'], threadgroup_size, state_load, loop_mode)(
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


# ---------------------------------------------------------------------------------------------------------------
# Fused routed-expert MLP: gate and up in shared launches, SwiGLU fused into the down projection's input rotation.
# Five launches per MoE layer instead of fifteen; every value is computed by the same arithmetic as
# projection_fused + PagedExperts' SwiGLU (checked bit-for-bit in test_window_decoder).

_FWHT_NODECL = _FWHT.replace('    threadgroup float buf[128];\n', '', 1)
assert _FWHT_NODECL != _FWHT


def _swiglu_source(gate, up, out):
    # PagedExperts: g = minimum(gate, 10); u = clip(up, -10, 10); out = g * sigmoid(g) * u, with MLX's NaN
    # propagation (Minimum/Maximum return x when x is NaN) and MLX's sigmoid.
    return f"""
    float swi_g = metal::isnan({gate}) ? {gate} : ({gate} < 10.0f ? {gate} : 10.0f);
    float swi_u = metal::isnan({up}) ? {up} : ({up} > -10.0f ? {up} : -10.0f);
    swi_u = metal::isnan(swi_u) ? swi_u : (swi_u < 10.0f ? swi_u : 10.0f);
    float swi_y = 1 / (1 + metal::precise::exp(metal::abs(swi_g)));
    float swi_s = (swi_g < 0) ? swi_y : 1 - swi_y;
    float {out} = (swi_g * swi_s) * swi_u;
"""


@lru_cache(maxsize=None)
def _prep_pair_kernel():
    """_prep_kernel(grouped=True) for gate and up at once (grid z), reading each row's token directly."""
    source = """
    uint t = thread_position_in_threadgroup.x;
    uint block = threadgroup_position_in_grid.x;
    uint row = threadgroup_position_in_grid.y;
    uint z = threadgroup_position_in_grid.z;
    auto base_su = z ? base_su_u : base_su_g;
    auto ref_su = z ? ref_su_u : ref_su_g;
    uint e = min(route[row], uint(base_su_g_shape[0]) - 1u);
    uint slot = MIXED ? refined[row] : 0xffffffffu;
    bool ready = MIXED && slot < uint(ref_su_g_shape[0]);
    uint k = block * 128u + t;
    float su = ready ? float(ref_su[slot * K + k]) : float(base_su[e * K + k]);
    float value = x[(row / E) * K + k] * su;
""" + _FWHT + """
    (z ? rotated_u : rotated_g)[row * K + k] = half(value);
    if (GROUPED && z == 0 && block == 0 && t == 0) {
""" + _GROUPS + """
    }
"""
    return mx.fast.metal_kernel(
        name='nq_prep_pair', input_names=['x', 'base_su_g', 'ref_su_g', 'base_su_u', 'ref_su_u', 'route', 'refined',
                                          'variants', 'words'],
        output_names=['rotated_g', 'rotated_u', 'groups'], source=source)


def _post_body(tag, rank):
    """The _post_kernel computation for one projection, leaving its masked output in `result_{tag}`."""
    low_rank = '' if rank == 0 else f"""
    float low_{tag}[{rank}];
    for (uint r = 0; r < {rank}u; ++r) {{
        float dot = 0.0f;
        for (uint k = t; k < K; k += 128u)
            dot += x[token * K + k] * float(lrV_{tag}[(e * {rank}u + r) * K + k]);
        dot = simd_sum(dot);
        if ((t & 31u) == 0) lows_{tag}[r][t / 32u] = dot;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint r = 0; r < {rank}u; ++r) {{
        low_{tag}[r] = lows_{tag}[r][0] + lows_{tag}[r][1] + lows_{tag}[r][2] + lows_{tag}[r][3];
        result += low_{tag}[r] * float(lrU2_{tag}[(e * {rank}u + r) * N + n]);
    }}
    if (ready) {{
        for (uint r = 0; r < {rank}u; ++r)
            result += low_{tag}[r] * float(lrU4_{tag}[(slot * {rank}u + r) * N + n]);
    }}
"""
    return f"""
    {{
    float value = 0.0f;
    auto source = ready ? partial_r{tag} : partial_{tag};
    for (uint g = 0; g < G; ++g)
        value += source[((row * TN + strip) * G + g) * 16u + j];
""" + _FWHT_NODECL + f"""
    float sv = ready ? float(ref_sv_{tag}[slot * N + n]) : float(base_sv_{tag}[e * N + n]);
    float result = value * sv;
""" + low_rank + f"""
    result_{tag} = valid ? result : NAN;
    }}
"""


@lru_cache(maxsize=None)
def _post_pair_kernel(rank_g: int, rank_u: int):
    """_post_kernel for gate and up, the SwiGLU, and _prep_kernel's input rotation for down, per 128 outputs.

    Gate/up output n is down input k = n, so one threadgroup finishes a 128-wide block of all three."""
    source = """
    uint t = thread_position_in_threadgroup.x;
    uint block = threadgroup_position_in_grid.x;
    uint row = threadgroup_position_in_grid.y;
    uint token = row / E;
    uint raw = route[row];
    uint depth = uint(base_sv_g_shape[0]);
    uint e = min(raw, depth - 1u);
    uint slot = MIXED ? refined[row] : 0xffffffffu;
    bool ready = MIXED && slot < uint(ref_sv_g_shape[0]);
    bool valid = raw < depth && (!MIXED || slot == 0xffffffffu || ready);
    uint n = block * 128u + t;
    uint strip = n / 16u, j = n % 16u;
    threadgroup float buf[128];
""" + (f"    threadgroup float lows_g[{rank_g}][4];\n" if rank_g else '') + (
        f"    threadgroup float lows_u[{rank_u}][4];\n" if rank_u else '') + """
    float result_g, result_u;
""" + _post_body('g', rank_g) + _post_body('u', rank_u) + _swiglu_source('result_g', 'result_u', 'act') + """
    act_out[row * N + n] = act;
    uint ed = min(raw, uint(base_su_d_shape[0]) - 1u);
    bool ready_d = MIXED && slot < uint(ref_su_d_shape[0]);
    float su = ready_d ? float(ref_su_d[slot * N + n]) : float(base_su_d[ed * N + n]);
    float value = act * su;
    threadgroup_barrier(mem_flags::mem_threadgroup);
""" + _FWHT_NODECL + """
    rotated_d[row * N + n] = half(value);
"""
    names = ['partial_g', 'partial_u', 'base_sv_g', 'ref_sv_g', 'base_sv_u', 'ref_sv_u', 'base_su_d', 'ref_su_d',
             'route', 'refined', 'x', 'partial_rg', 'partial_ru']
    if rank_g:
        names += ['lrV_g', 'lrU2_g', 'lrU4_g']
    if rank_u:
        names += ['lrV_u', 'lrU2_u', 'lrU4_u']
    return mx.fast.metal_kernel(name=f'nq_post_pair_r{rank_g}_{rank_u}', input_names=names,
                                output_names=['act_out', 'rotated_d'], source=source)


def expert_mlp_fused(planes: dict, tokens: mx.array, route: mx.array, experts_per_token: int, *,
                     refinements: dict | None = None, refined_route: mx.array | None = None) -> mx.array:
    """Routed-expert MLP for `route` (experts_per_token rows per token): down(g * sigmoid(g) * u) with
    g = min(gate(x), 10), u = clip(up(x), -10, 10), as PagedExperts computes it from projection_fused.

    planes: name -> (base planes, meta) for 'gate', 'up', 'down'; refinements: name -> refinement planes.
    tokens: (tokens, K) float32 inputs already rounded to fp16. Returns (rows, n_down) float32."""
    rows = route.size
    gate_planes, gate_meta = planes['gate']
    up_planes, up_meta = planes['up']
    down_planes, down_meta = planes['down']
    if (gate_meta['k'], gate_meta['n'], gate_meta['tk'], gate_meta['tn'], gate_meta['shard_axis'], gate_meta['base_K'],
            gate_meta['res_rule']['K'], gate_meta.get('var_bits', 8)) != (
            up_meta['k'], up_meta['n'], up_meta['tk'], up_meta['tn'], up_meta['shard_axis'], up_meta['base_K'],
            up_meta['res_rule']['K'], up_meta.get('var_bits', 8)) or down_meta['k'] != gate_meta['n']:
        raise ValueError('Gate and up projections must share a geometry that feeds down')
    mixed = refinements is not None
    refined = refined_route if mixed else route
    tk, tn, k, n = gate_meta['tk'], gate_meta['tn'], gate_meta['k'], gate_meta['n']
    common = [('MIXED', mixed)]
    ref = (lambda name, field, fallback: refinements[name][field] if mixed else planes[name][0][fallback])
    max_rows = max(1, min(4 if rows <= 64 else 8, rows // 8)) if GROUP_SHARED_EXPERTS else 1
    if GROUP_MAXR_OVERRIDE is not None and GROUP_SHARED_EXPERTS:
        max_rows = GROUP_MAXR_OVERRIDE
    grouped = max_rows > 1
    words_g = refinements['gate']['word'] if mixed else gate_planes['word']
    rot_g, rot_u, groups = _prep_pair_kernel()(
        inputs=[tokens, gate_planes['suh2'], ref('gate', 'suh4', 'suh2'), up_planes['suh2'], ref('up', 'suh4', 'suh2'),
                route, refined, gate_planes['var'], words_g],
        template=[('K', k), ('E', experts_per_token), ('MAXR', max_rows), ('GROUPED', grouped), *common],
        grid=(k, rows, 2), threadgroup=(128, 1, 1),
        output_shapes=[(rows, k), (rows, k), (rows * (max_rows + 1),)],
        output_dtypes=[mx.float16, mx.float16, mx.uint32])
    if not grouped:
        groups = route
    chunks = WINDOW_CHUNKS if tk % WINDOW_CHUNKS == 0 else 1
    positions = LANE_POSITIONS if chunks % 2 == 0 else 8
    window_template = [('LEVEL', 2), ('TK', tk), ('TN', tn), ('BANKED', True), ('MIXED', mixed),
                       ('PACKED_VAR', gate_meta.get('var_bits', 8) == 1), ('CHUNKS', chunks), ('MAXR', max_rows),
                       ('SHARD_N', gate_meta['shard_axis'] == 'n'), ('XHALF', True), ('AC_TABLE', USE_AC_TABLE)]
    residual = (lambda name: (refinements[name]['p4'], refinements[name]['word']) if mixed
                else (planes[name][0]['p4'], planes[name][0]['word']))
    (res_g, word_g), (res_u, word_u) = residual('gate'), residual('up')
    (partial_rg, partial_ru), (partial_g, partial_u) = _window(gate_meta['base_K'], gate_meta['res_rule']['K'], positions, True,
        [gate_planes['base'], up_planes['base'], res_g, res_u, word_g, word_u, gate_planes['var'],
         up_planes['var'], _matrix_indices(tk, tn, gate_meta['shard_axis'])[0], rot_g, rot_u, route, refined,
         groups, _ac_table()], mixed,
        template=window_template, grid=(tn * (tk // chunks) * 256, rows, 2), threadgroup=(256, 1, 1),
        output_shapes=[(rows, tn, tk // chunks, 16)] * 2, output_dtypes=[mx.float32] * 2)
    rank_g = gate_planes['lrV'].shape[1] if 'lrV' in gate_planes else 0
    rank_u = up_planes['lrV'].shape[1] if 'lrV' in up_planes else 0
    inputs = [partial_g, partial_u, gate_planes['svh2'], ref('gate', 'svh4', 'svh2'), up_planes['svh2'],
              ref('up', 'svh4', 'svh2'), down_planes['suh2'], ref('down', 'suh4', 'suh2'), route, refined, tokens,
              partial_rg, partial_ru]
    if rank_g:
        inputs += [gate_planes['lrV'], gate_planes['lrU2'], ref('gate', 'lrU4', 'lrU2')]
    if rank_u:
        inputs += [up_planes['lrV'], up_planes['lrU2'], ref('up', 'lrU4', 'lrU2')]
    act, rot_d = _post_pair_kernel(rank_g, rank_u)(
        inputs=inputs, template=[('K', k), ('N', n), ('TN', tn), ('G', tk // chunks), ('E', experts_per_token), *common],
        grid=(n, rows, 1), threadgroup=(128, 1, 1), output_shapes=[(rows, n), (rows, n)],
        output_dtypes=[mx.float32, mx.float16])
    # Down: the existing window and output kernels on the rotated SwiGLU (groups depend only on the routing).
    tk, tn, k, n = down_meta['tk'], down_meta['tn'], down_meta['k'], down_meta['n']
    chunks = WINDOW_CHUNKS if tk % WINDOW_CHUNKS == 0 else 1
    positions = LANE_POSITIONS if chunks % 2 == 0 else 8
    res_d, word_d = residual('down')
    (partial_r,), (partial,) = _window(down_meta['base_K'], down_meta['res_rule']['K'], positions, False,
        [down_planes['base'], res_d, word_d, down_planes['var'], _matrix_indices(tk, tn, down_meta['shard_axis'])[0],
         rot_d, route, refined, groups, _ac_table()], mixed,
        template=[('LEVEL', 2), ('TK', tk), ('TN', tn), ('BANKED', True), ('MIXED', mixed),
                  ('PACKED_VAR', down_meta.get('var_bits', 8) == 1), ('CHUNKS', chunks), ('MAXR', max_rows),
                  ('SHARD_N', down_meta['shard_axis'] == 'n'), ('XHALF', True), ('AC_TABLE', USE_AC_TABLE)],
        grid=(tn * (tk // chunks) * 256, rows, 1), threadgroup=(256, 1, 1),
        output_shapes=[(rows, tn, tk // chunks, 16)], output_dtypes=[mx.float32])
    rank = down_planes['lrV'].shape[1] if 'lrV' in down_planes else 0
    inputs = [partial, down_planes['svh2'], ref('down', 'svh4', 'svh2'), route, refined, act, partial_r]
    if rank:
        inputs += [down_planes['lrV'], down_planes['lrU2'], ref('down', 'lrU4', 'lrU2')]
    return _post_kernel(rank)(
        inputs=inputs, template=[('K', k), ('N', n), ('TN', tn), ('G', tk // chunks), ('RANK', max(rank, 1)), *common],
        grid=(n, rows, 1), threadgroup=(128, 1, 1), output_shapes=[(rows, n)], output_dtypes=[mx.float32])[0]
