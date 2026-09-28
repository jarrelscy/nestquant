"""Thread 15: standalone numpy reference decoder (bit-exact with nqk15.cu) for the level-4 int-fold codes.

Works on thread-12 style units: one 16x128 unit = 8 tail-biting rings of 256 weights (G=4 lanes / ring).
Ring g, position p = t4*64 + j  <->  lane 4g+t4, lane weight j  <->  row g + 8*(r&1), k 16t + 2*t4 + 8*(r>>1) + e,
with pp = j>>1, e = j&1, t = pp>>2, r = pp&3   (== thread-12 nq_decode.ring_index).

Streams are LSB-first bit streams per ring (thread-12 nq_decode.pack_bits format: uint8 [R, nbytes]); lane t4's
kernel record = ring bits [t4*BITS_lane, (t4+1)*BITS_lane).

  step_off(p; KA, MASK) = (p>>4)*(16*KA + popc(MASK)) + sum_{i < p%16} (KA + ((MASK>>i)&1))
      K=2: KA=2,MASK=0 (offset 2p).  1.5: 1,0xAAAA  1.75: 1,0xEEEE  2.25: 2,0x8888  2.5: 2,0xAAAA  2.75: 2,0xEEEE  3: 3,0
  state(p)  = ring bits [step_off(p), step_off(p)+16), bit step_off(p) = state LSB, ring wraps (tail-biting)
  hash      x = state * 0x83DCD12D mod 2^32, bytes b0..b3 (b0 = x & 255)
  S(state)  = b0+b1+b2+b3                      (0..1020)
  S2(state) = 510 + b0-b1+b2-b3                 (0..1020)      [V2 only]

  LEVEL 2 (base):  Q2 = fp16(A*(1024 + S(sb)) + B),  A = fp16 0x1eee = 1774/2^18, B = fp16 0xc931   (== harness mul1 LUT)

  LEVEL 4 (RM_P, V1):  per unit block word u16 = Mb | N<<8,  1 <= Mb <= 255, 0 <= N, Mb + N <= 257
      F   = (Mb*S(sb) + N*S(sr) + 128) >> 8          (integer, 0..1019)
      A'  = fp16(1.732421875f * (1.0f/Mb))           (fp32 ops, then fp16 round; 1.732421875 = 256*A)
      C   = fp16(fp32(N * (1.0f/Mb)) * K0 + K0 - 1024*A')     K0 = 1024*A + B = -3.453125
      Q4  = fp16(A' * (1024 + F) + C)                (single rounding = HFMA2)
      ~= Q2 + delta * (A*S(sr) + K0),  delta = N/Mb   (the residual uses the SAME mul1 codebook, scaled by delta)

  LEVEL 4 (RM_P2, V2 "2D mul1" residual): one residual state per weight PAIR q (ring pair position q = t4*32 + P,
      lane pair P covers lane weights 2P, 2P+1); state(q) at step_off(q; KA, MASK) of the residual ring stream
      (32 pair-steps per lane => lane record = 32*Kpair bits, Kpair = bits per pair).
      Sr(2P) = S(state(q)),  Sr(2P+1) = S2(state(q));  then the V1 formula, with the extra constraint N <= 127.
"""
import numpy as np

A = float(np.array([0x1eee], np.uint16).view(np.float16)[0])
B = float(np.array([0xc931], np.uint16).view(np.float16)[0])
K0 = 1024 * A + B                                       # -3.453125 exactly
PATTERNS = {1.5: (1, 0xAAAA), 1.75: (1, 0xEEEE), 2: (2, 0), 2.25: (2, 0x8888), 2.5: (2, 0xAAAA),
            2.75: (2, 0xEEEE), 3: (3, 0), 4: (4, 0)}


def popc(m): return bin(m).count("1")


def step_off(p, KA, MASK):
    p = np.asarray(p)
    per = np.array([sum(KA + ((MASK >> i) & 1) for i in range(r)) for r in range(16)])
    return (p >> 4) * (16 * KA + popc(MASK)) + per[p & 15]


def ring_bits(stream):
    """uint8 [R, nbytes] LSB-first -> {0,1} int64 [R, nbits]."""
    return ((stream[..., None].astype(np.int64) >> np.arange(8)) & 1).reshape(stream.shape[0], -1)


def states(stream, npos, K):
    """16-bit trellis states at ring positions 0..npos-1 (tail-biting)."""
    KA, MASK = K if isinstance(K, tuple) else PATTERNS[K]
    bits = ring_bits(stream); nb = bits.shape[1]
    off = step_off(np.arange(npos), KA, MASK)
    assert step_off(npos, KA, MASK) == nb, (nb, step_off(npos, KA, MASK))
    idx = (off[:, None] + np.arange(16)) % nb
    return (bits[:, idx] << np.arange(16)).sum(-1)       # [R, npos]


def hbytes(st):
    x = (st.astype(np.int64) * 0x83DCD12D) & 0xFFFFFFFF
    return [(x >> (8 * i)) & 255 for i in range(4)]


def S(st): b = hbytes(st); return b[0] + b[1] + b[2] + b[3]
def S2(st): b = hbytes(st); return 510 + b[0] - b[1] + b[2] - b[3]


f16 = lambda v: np.asarray(v, np.float64).astype(np.float16).astype(np.float64)


def q2(sb): return f16(A * (1024 + S(sb)) + B)


def consts(Mb, N):
    Mb = np.asarray(Mb); N = np.asarray(N)
    rcp = np.float32(1) / Mb.astype(np.float32)
    Ah = f16(np.float32(1.732421875) * rcp)
    Ch = f16((N.astype(np.float32) * rcp) * np.float32(K0) + np.float32(K0) - 1024 * Ah)
    return Ah, Ch


def fold(Ssb, Ssr, Mb, N):
    Ah, Ch = consts(Mb, N)
    F = (Mb * Ssb + N * Ssr + 128) >> 8
    return f16(Ah * (1024 + F) + Ch)


def delta_to_MbN(delta, v2=False):
    """Largest-Mb rational N/Mb nearest to delta (0 <= delta), Mb + N <= 257, Mb <= 255 (V2: N <= 127).
    Largest Mb => finest fold rounding (step 256/Mb units of A)."""
    d = np.asarray(delta, np.float64)
    Mb = np.minimum(255, np.floor(257 / (1 + d))).astype(np.int64)
    N = np.rint(d * Mb).astype(np.int64)
    N = np.minimum(N, 257 - Mb)
    if v2: N = np.minimum(N, 127)
    return Mb, N


def decode_unit(base_stream, res_stream, Mb, N, Kb=2, Kr=2, v2=False):
    """One unit: base/res ring streams uint8 [8, nbytes] -> (Q2, Q4) as ring-order [8, 256] fp16-valued float64.
    Mb, N: ints (per unit). K = key of PATTERNS or an explicit (KA, MASK). For V2, Kr is the per-PAIR-step pattern
    (e.g. (4, 0) = 4 bits/pair = 2 bpw; (5, 0xAAAA) = 5.5 bits/pair = 2.75 bpw)."""
    sb = states(base_stream, 256, Kb)
    Q2 = q2(sb)
    if not v2:
        Sr = S(states(res_stream, 256, Kr))
    else:
        sr = states(res_stream, 128, Kr)                     # ring pair positions q = t4*32 + P
        Sr = np.empty((8, 256), np.int64); Sr[:, 0::2] = S(sr); Sr[:, 1::2] = S2(sr)
    return Q2, fold(S(sb), Sr, Mb, N)


def ring_index():
    RI = np.empty((8, 256), np.int64)
    for g in range(8):
        for t4 in range(4):
            for j in range(64):
                pp, e = j >> 1, j & 1; t, r = pp >> 2, pp & 3
                RI[g, t4 * 64 + j] = (16 * t + 2 * t4 + 8 * (r >> 1) + e) * 16 + g + 8 * (r & 1)
    return RI                                                 # local index k_local*16 + row_local, unit [128, 16]


def to_unit(ring_vals):
    out = np.empty(2048); out[ring_index().ravel()] = ring_vals.ravel(); return out.reshape(128, 16)


def rings_from_lane_words(words, bits):
    """Kernel lane records (uint32 [32, NW], lane-major within a unit, 'bits' valid bits each) -> ring streams
    uint8 [8, 4*bits/8]; ring g = lanes 4g..4g+3 concatenated (G=4)."""
    b = ((words[..., None].astype(np.int64) >> np.arange(32)) & 1).reshape(32, -1)[:, :bits]
    b = b.reshape(8, 4 * bits)
    return (b.reshape(8, -1, 8) << np.arange(8)).sum(-1).astype(np.uint8)
