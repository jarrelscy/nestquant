// NestQuant grouped MoE decode layer (thread 13).
//
// Two launches per MoE layer, both graph-capturable, both reading a device-side per-expert table at replay:
//   K1 (gate|up): grid (2I/16/SB, H/kslice, S=B*topk). Every block redundantly dedups the routing (one warp,
//       __match_any_sync), takes run z (= one distinct expert + the <=8 tokens that picked it), rotates those
//       tokens' x (x*su -> WHT128) into smem, runs the tensor-core GEMV with the tokens as MMA N, split-K atomics
//       into acc_gu[slot]. The last-arriving block of a (run, 128-col group) does WHT, *sv, SwiGLU, *su_d, WHT and
//       stores h[slot] (fp16), zeroing acc_gu (self-cleaning).
//   K2 (down):   grid (H/16/SB, I/kslice, S). Same dedup; h[slot] -> smem; GEMV; atomics into acc_d[slot]. The
//       last-arriving block of a 128-row output group (counter over all runs) does the fused combine:
//       out[b] = sum_k rw[b,k] * sv_o[e] (.) WHT128(acc_d[b*topk+k]), zeroing acc_d.
// Levels are a block-uniform branch on table[e].level (0 = expert not resident on this GPU -> contributes 0; 2; 4).
//
// Plane layout (per expert, per projection; N rows x K cols; strip = 16 rows, chunk = 128 cols; C = K/128;
// record = one lane's share of one 16x128 unit, record index ((strip*C + c)*32 + lane)):
//   base : uint4 per record: 64 weights x 2 bit (K=2 ring stream, 128 bits)                     [nrec = S*C*32]
//   P4   : residual records of RBITS = 4*(16*KA + popc(MASK)) bits (K = RBITS/64 per weight), stored as
//          sub-arrays uint4[n4] | uint2 | uint | ushort (n4 = RBITS/128, then the 64/32/16-bit remainder, each
//          sub-array present iff that bit of RBITS%128 is set), each sub-array contiguous over all nrec_r records,
//          then (if RBITS % 16 != 0) a tail sub-array of T = RBITS%16 bits per record, bit-packed (record r at bit r*T)
//   d4   : uint32 block word Mb | N << 8 per 16x128 unit (1 <= Mb <= 255, Mb + N <= 257; delta = N/Mb) -> strip*C + c
//   level 2 = base; level 4 = base + P4 (int-fold, see nqdec).  Optional mask mode (flags != null): uint64 per strip,
//   bit c = chunk c is refined; P4/d4 are then compact over the nm flagged chunks of each strip
//   (-> (strip*nm + rank)*32 + lane, nrec_r = S*nm*32). Flags must be uniform over each 128-row group (8 strips).
//   Ring streams are tail-biting over G lanes (G = 4: lanes 4g..4g+3 = one ring of 256 weights; G = 2: lane pairs).
// Device table: int64 [E][16]
//   [0] level (0, 2, 4)  [1..4] gate|up: base, p4, d4, flags(0 = dense)  [5..8] down: same
//   [9] scales: half[H su_g | I sv_g | I sv_u | I su_d | H sv_o | H su_u] (EXL3 suh/svh; gate and up have separate
//        input scales; the level's own scale vector: T12 stores separate level-2 / level-4 scales)
//   [10] gate|up residual K code, [11] down residual K code (0: K=2, 1: 1.75, 2: 2.5, 3: 2.25, 4: 3, 5: 1.5, 6: 1.9375, 7: 2.3125, 8: 1.875; codes
//        outside NQ_RK_CODES / NQ_RK_GU / NQ_RK_DN fall back to code 0)
//   [12] gate|up base-variant plane, [13] down (0 = none): uint8 per unit (strip*C + c, dense), bit g = sign of ring g
//        (T12 base_var 'sign': a = -1 negates the ring's level-2 and level-4 values exactly). G = 4 only.
//   [14] low-rank plane, resident with the base (0 = none): half[V_gu r_gu x H | U2_g r_gu x I | U2_u r_gu x I |
//        V_dn r_dn x I | U2_d r_dn x H]   [15] level-4 part, lives with P4 (0 = none): half[U4_g | U4_u | U4_d]
//   [16] r_gu, [17] r_dn (<= 4)   [18] in_had_down: width of the down projection's input Hadamard (0 or 128 = Had128
//        blocks; 512 = one sign + Sylvester Hadamard-512 block per 512 SwiGLU columns, threads/29; must divide I)  [19] reserved
//   in_had_down w > 128: K1's finisher stores WHT128(sw * su_d) in fp32 into the (already consumed) gate columns of
//   acc_gu[slot]; the last-arriving of the w/128 column groups of a (run, w-block) applies the cross-block Sylvester
//   H_{w/128} / sqrt(w/128) (WHT_w = H_{w/128} (x) WHT128), rounds to fp16 once into h[slot] and re-zeroes those columns.
//   T12 lr plane (nq_decode.lr_term): W = Wq + U2^T V (+ U4^T V at level 4), V in the unrotated input basis, U in the
//   unrotated output basis. Gate|up: z = x V_gu^T in K1's finisher, g/u += z (U2 [+ U4]) before SwiGLU (gate and up share
//   V). Down: K1's finisher writes z partials of its 128 SwiGLU columns to zd [S][I/128][4]; K2's combine sums them and
//   adds z (U2_d [+ U4_d]) per slot. Under TP each rank adds its own partial (down U is replicated), the all-reduce of
//   the MoE output sums them.
#include <cuda_fp16.h>
#include <stdint.h>
#include <set>
#include <map>
#include <tuple>
#include <torch/extension.h>
// Residual K codes compiled in (bit c = code c; code 0 always), per kernel: NQ_RK_GU (K1 gate|up), NQ_RK_DN (K2 down).
// T13 default = gate|up {K 2, 1.9375}, down {K 2, 2.3125}; the SM120 port compiles the per-expert split set. Fewer codes
// per kernel = less stack in the CPW=2 / persistent variants and faster 4p (abk.py). If NQ_RK_CODES is given
// (e.g. 0x7 for the older 1.75 / 2.5 config, 0xff = all), both kernels get all of it unless NQ_RK_GU / NQ_RK_DN are
// also given. Codes a kernel lacks decode as code 0; MoELayer.set asserts against rk_codes().
#ifndef NQ_RK_CODES
#define NQ_RK_CODES 0x1C9   // SM120: per-expert split set K = 2, 2.25, 1.9375, 2.3125, 1.875 (codes 0, 3, 6, 7, 8) in both kernels
#endif
#ifndef NQ_RK_GU
#define NQ_RK_GU NQ_RK_CODES
#endif
#ifndef NQ_RK_DN
#define NQ_RK_DN NQ_RK_CODES
#endif
#include <ATen/cuda/CUDAContext.h>

// ============================== decoder (swappable; thread 15 nqk15.cu: greedy-funnel mul1 base + RM_P int-fold residual)
// Level 2 (base, K=2):   Q2 = fp16(A (1024 + S(sb)) + B)                       (standard mul1, == harness LUT)
// Level 4 (RM_P, V1):    block word u32 = Mb | N << 8 (1 <= Mb <= 255, Mb + N <= 257), delta = N / Mb
//     F = (Mb S(sb) + N S(sr) + 128) >> 8;  A' = fp16(256A / Mb);  C = fp16((N/Mb) K0 + K0 - 1024 A');  Q4 = fp16(A'(1024+F) + C)
// Residual window pattern per projection (runtime code in the table, compile-time per instantiation):
//     code 0: K=2   1: K=1.75 (1,0xEEEE)   2: K=2.5 (2,0xAAAA)   3: K=2.25 (2,0x8888)   4: K=3 (3,0)   5: K=1.5 (1,0xAAAA)
//     6: K=1.9375 (1,0xFFFE)   7: K=2.3125 (2,0x9248)       (step p has KA + ((MASK >> (p%16)) & 1) bits == nq_decode.PATTERNS)
// Bit-level spec: ref15_spec.py (copied from thread 15, LSB-first ring streams, tail-biting over G lanes).
namespace nqdec {
#define MUL1_A 0x1eee1eeeu
#define MUL1_B 0xc931c931u
#define HC 0x83DCD12Du
__device__ __constant__ float c_rcp[256] = {0.0f, 0x1.0000000000000p+0f, 0x1.0000000000000p-1f, 0x1.5555560000000p-2f, 0x1.0000000000000p-2f, 0x1.99999a0000000p-3f, 0x1.5555560000000p-3f, 0x1.24924a0000000p-3f, 0x1.0000000000000p-3f, 0x1.c71c720000000p-4f, 0x1.99999a0000000p-4f, 0x1.745d180000000p-4f, 0x1.5555560000000p-4f, 0x1.3b13b20000000p-4f, 0x1.24924a0000000p-4f, 0x1.1111120000000p-4f, 0x1.0000000000000p-4f, 0x1.e1e1e20000000p-5f, 0x1.c71c720000000p-5f, 0x1.af286c0000000p-5f, 0x1.99999a0000000p-5f, 0x1.8618620000000p-5f, 0x1.745d180000000p-5f, 0x1.642c860000000p-5f, 0x1.5555560000000p-5f, 0x1.47ae140000000p-5f, 0x1.3b13b20000000p-5f, 0x1.2f684c0000000p-5f, 0x1.24924a0000000p-5f, 0x1.1a7b960000000p-5f, 0x1.1111120000000p-5f, 0x1.0842100000000p-5f, 0x1.0000000000000p-5f, 0x1.f07c200000000p-6f, 0x1.e1e1e20000000p-6f, 0x1.d41d420000000p-6f, 0x1.c71c720000000p-6f, 0x1.bacf920000000p-6f, 0x1.af286c0000000p-6f, 0x1.a41a420000000p-6f, 0x1.99999a0000000p-6f, 0x1.8f9c180000000p-6f, 0x1.8618620000000p-6f, 0x1.7d05f40000000p-6f, 0x1.745d180000000p-6f, 0x1.6c16c20000000p-6f, 0x1.642c860000000p-6f, 0x1.5c98820000000p-6f, 0x1.5555560000000p-6f, 0x1.4e5e0a0000000p-6f, 0x1.47ae140000000p-6f, 0x1.4141420000000p-6f, 0x1.3b13b20000000p-6f, 0x1.3521d00000000p-6f, 0x1.2f684c0000000p-6f, 0x1.29e4120000000p-6f, 0x1.24924a0000000p-6f, 0x1.1f70480000000p-6f, 0x1.1a7b960000000p-6f, 0x1.15b1e60000000p-6f, 0x1.1111120000000p-6f, 0x1.0c97140000000p-6f, 0x1.0842100000000p-6f, 0x1.0410420000000p-6f, 0x1.0000000000000p-6f, 0x1.f81f820000000p-7f, 0x1.f07c200000000p-7f, 0x1.e9131a0000000p-7f, 0x1.e1e1e20000000p-7f, 0x1.dae6080000000p-7f, 0x1.d41d420000000p-7f, 0x1.cd85680000000p-7f, 0x1.c71c720000000p-7f, 0x1.c0e0700000000p-7f, 0x1.bacf920000000p-7f, 0x1.b4e81c0000000p-7f, 0x1.af286c0000000p-7f, 0x1.a98ef60000000p-7f, 0x1.a41a420000000p-7f, 0x1.9ec8ea0000000p-7f, 0x1.99999a0000000p-7f, 0x1.948b100000000p-7f, 0x1.8f9c180000000p-7f, 0x1.8acb900000000p-7f, 0x1.8618620000000p-7f, 0x1.8181820000000p-7f, 0x1.7d05f40000000p-7f, 0x1.78a4c80000000p-7f, 0x1.745d180000000p-7f, 0x1.702e060000000p-7f, 0x1.6c16c20000000p-7f, 0x1.6816820000000p-7f, 0x1.642c860000000p-7f, 0x1.6058160000000p-7f, 0x1.5c98820000000p-7f, 0x1.58ed240000000p-7f, 0x1.5555560000000p-7f, 0x1.51d07e0000000p-7f, 0x1.4e5e0a0000000p-7f, 0x1.4afd6a0000000p-7f, 0x1.47ae140000000p-7f, 0x1.446f860000000p-7f, 0x1.4141420000000p-7f, 0x1.3e22cc0000000p-7f, 0x1.3b13b20000000p-7f, 0x1.3813820000000p-7f, 0x1.3521d00000000p-7f, 0x1.323e340000000p-7f, 0x1.2f684c0000000p-7f, 0x1.2c9fb40000000p-7f, 0x1.29e4120000000p-7f, 0x1.27350c0000000p-7f, 0x1.24924a0000000p-7f, 0x1.21fb780000000p-7f, 0x1.1f70480000000p-7f, 0x1.1cf06a0000000p-7f, 0x1.1a7b960000000p-7f, 0x1.1811820000000p-7f, 0x1.15b1e60000000p-7f, 0x1.135c820000000p-7f, 0x1.1111120000000p-7f, 0x1.0ecf560000000p-7f, 0x1.0c97140000000p-7f, 0x1.0a68100000000p-7f, 0x1.0842100000000p-7f, 0x1.0624de0000000p-7f, 0x1.0410420000000p-7f, 0x1.0204080000000p-7f, 0x1.0000000000000p-7f, 0x1.fc07f00000000p-8f, 0x1.f81f820000000p-8f, 0x1.f4465a0000000p-8f, 0x1.f07c200000000p-8f, 0x1.ecc07c0000000p-8f, 0x1.e9131a0000000p-8f, 0x1.e573ac0000000p-8f, 0x1.e1e1e20000000p-8f, 0x1.de5d6e0000000p-8f, 0x1.dae6080000000p-8f, 0x1.d77b660000000p-8f, 0x1.d41d420000000p-8f, 0x1.d0cb580000000p-8f, 0x1.cd85680000000p-8f, 0x1.ca4b300000000p-8f, 0x1.c71c720000000p-8f, 0x1.c3f8f00000000p-8f, 0x1.c0e0700000000p-8f, 0x1.bdd2b80000000p-8f, 0x1.bacf920000000p-8f, 0x1.b7d6c40000000p-8f, 0x1.b4e81c0000000p-8f, 0x1.b203640000000p-8f, 0x1.af286c0000000p-8f, 0x1.ac57020000000p-8f, 0x1.a98ef60000000p-8f, 0x1.a6d01a0000000p-8f, 0x1.a41a420000000p-8f, 0x1.a16d400000000p-8f, 0x1.9ec8ea0000000p-8f, 0x1.9c2d140000000p-8f, 0x1.99999a0000000p-8f, 0x1.970e500000000p-8f, 0x1.948b100000000p-8f, 0x1.920fb40000000p-8f, 0x1.8f9c180000000p-8f, 0x1.8d30180000000p-8f, 0x1.8acb900000000p-8f, 0x1.886e600000000p-8f, 0x1.8618620000000p-8f, 0x1.83c9780000000p-8f, 0x1.8181820000000p-8f, 0x1.7f40600000000p-8f, 0x1.7d05f40000000p-8f, 0x1.7ad2200000000p-8f, 0x1.78a4c80000000p-8f, 0x1.767dce0000000p-8f, 0x1.745d180000000p-8f, 0x1.7242880000000p-8f, 0x1.702e060000000p-8f, 0x1.6e1f760000000p-8f, 0x1.6c16c20000000p-8f, 0x1.6a13ce0000000p-8f, 0x1.6816820000000p-8f, 0x1.661ec60000000p-8f, 0x1.642c860000000p-8f, 0x1.623fa80000000p-8f, 0x1.6058160000000p-8f, 0x1.5e75bc0000000p-8f, 0x1.5c98820000000p-8f, 0x1.5ac0560000000p-8f, 0x1.58ed240000000p-8f, 0x1.571ed40000000p-8f, 0x1.5555560000000p-8f, 0x1.5390940000000p-8f, 0x1.51d07e0000000p-8f, 0x1.5015020000000p-8f, 0x1.4e5e0a0000000p-8f, 0x1.4cab880000000p-8f, 0x1.4afd6a0000000p-8f, 0x1.49539e0000000p-8f, 0x1.47ae140000000p-8f, 0x1.460cbc0000000p-8f, 0x1.446f860000000p-8f, 0x1.42d6620000000p-8f, 0x1.4141420000000p-8f, 0x1.3fb0140000000p-8f, 0x1.3e22cc0000000p-8f, 0x1.3c995a0000000p-8f, 0x1.3b13b20000000p-8f, 0x1.3991c20000000p-8f, 0x1.3813820000000p-8f, 0x1.3698e00000000p-8f, 0x1.3521d00000000p-8f, 0x1.33ae460000000p-8f, 0x1.323e340000000p-8f, 0x1.30d1900000000p-8f, 0x1.2f684c0000000p-8f, 0x1.2e025c0000000p-8f, 0x1.2c9fb40000000p-8f, 0x1.2b404a0000000p-8f, 0x1.29e4120000000p-8f, 0x1.288b020000000p-8f, 0x1.27350c0000000p-8f, 0x1.25e2280000000p-8f, 0x1.24924a0000000p-8f, 0x1.2345680000000p-8f, 0x1.21fb780000000p-8f, 0x1.20b4700000000p-8f, 0x1.1f70480000000p-8f, 0x1.1e2ef40000000p-8f, 0x1.1cf06a0000000p-8f, 0x1.1bb4a40000000p-8f, 0x1.1a7b960000000p-8f, 0x1.1945380000000p-8f, 0x1.1811820000000p-8f, 0x1.16e0680000000p-8f, 0x1.15b1e60000000p-8f, 0x1.1485f00000000p-8f, 0x1.135c820000000p-8f, 0x1.12358e0000000p-8f, 0x1.1111120000000p-8f, 0x1.0fef020000000p-8f, 0x1.0ecf560000000p-8f, 0x1.0db20a0000000p-8f, 0x1.0c97140000000p-8f, 0x1.0b7e6e0000000p-8f, 0x1.0a68100000000p-8f, 0x1.0953f40000000p-8f, 0x1.0842100000000p-8f, 0x1.0732600000000p-8f, 0x1.0624de0000000p-8f, 0x1.0519800000000p-8f, 0x1.0410420000000p-8f, 0x1.03091c0000000p-8f, 0x1.0204080000000p-8f, 0x1.0101020000000p-8f};   // IEEE 1.0f / i

__host__ __device__ constexpr int popc16(int m) { int c = 0; for (int i = 0; i < 16; ++i) c += (m >> i) & 1; return c; }
__host__ __device__ constexpr int step_off(int j, int KA, int MASK)
{
    int s = (j >> 4) * (16 * KA + popc16(MASK));
    for (int i = 0; i < (j & 15); ++i) s += KA + ((MASK >> i) & 1);
    return s;
}
__host__ __device__ constexpr bool direct_win(int o) { return (o & 31) == 0 || (o & 31) == 8 || (o & 31) == 16; }
// greedy funnel grouping per residue class (mod 8) over the plane's window offsets; returns the funnel start serving O
__host__ __device__ constexpr int funnel_greedy(int O, int KA, int MASK, int NS)
{
    int cov = -1, F = O;
    for (int j = 0; j < NS; ++j)
    {
        int o = step_off(j, KA, MASK);
        if ((o & 7) != (O & 7) || direct_win(o)) continue;
        if (o > cov) { F = o; cov = o + 16; }
        if (o == O) return F;
    }
    return O;
}
template <int O, int KA, int MASK>
__device__ __forceinline__ uint32_t wv(const uint32_t* w)
{
    constexpr int i = O >> 5, s = O & 31;
    if constexpr (s == 0) return w[i] & 0xFFFFu;
    else if constexpr (s == 16) return w[i] >> 16;
    else if constexpr (s == 8) return __byte_perm(w[i], 0u, 0x4421);
    else
    {
        constexpr int F = funnel_greedy(O, KA, MASK, 64), t = (O - F) / 8, fi = F >> 5, fs = F & 31;
        uint32_t y;
        if constexpr (fs == 0) y = w[fi]; else y = __funnelshift_r(w[fi], w[fi + 1], fs);
        if constexpr (t == 0) return y & 0xFFFFu;
        else if constexpr (t == 1) return __byte_perm(y, 0u, 0x4421);
        else return y >> 16;
    }
}
__device__ __forceinline__ uint32_t dp4u(uint32_t a, uint32_t b, uint32_t c) { uint32_t d; asm("dp4a.u32.u32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c)); return d; }
__device__ __forceinline__ uint32_t hfma2u(uint32_t a, uint32_t s, uint32_t b)
{
    half2 r = __hfma2(*(half2*)&a, *(half2*)&s, *(half2*)&b);
    return *(uint32_t*)&r;
}

// residual code -> window pattern
template <int RC> struct RK;
template <> struct RK<0> { static constexpr int KA = 2, M = 0x0000; };
template <> struct RK<1> { static constexpr int KA = 1, M = 0xEEEE; };
template <> struct RK<2> { static constexpr int KA = 2, M = 0xAAAA; };
template <> struct RK<3> { static constexpr int KA = 2, M = 0x8888; };
template <> struct RK<4> { static constexpr int KA = 3, M = 0x0000; };
template <> struct RK<5> { static constexpr int KA = 1, M = 0xAAAA; };
template <> struct RK<6> { static constexpr int KA = 1, M = 0xFFFE; };   // K = 1.9375 (T14 pattern-rate, gate|up)
template <> struct RK<7> { static constexpr int KA = 2, M = 0x9248; };   // K = 2.3125 (T14 pattern-rate, down)
template <> struct RK<8> { static constexpr int KA = 1, M = 0xFEFE; };   // K = 1.875 (T14 pattern-rate)
template <int RC> struct RKB { static constexpr int BITS = 4 * (16 * RK<RC>::KA + popc16(RK<RC>::M)), NW = (BITS + 31) / 32; };

// Planes of one projection of one expert. p4/d4 are indexed densely (every chunk refined) when flags == nullptr,
// otherwise compactly over the flagged chunks of each strip (mask mode, nm flagged chunks per strip).
// base: uint4 per record. p4: sub-arrays (uint4 x n4 | uint2 | uint | ushort) each contiguous over the nrec records.
struct Planes { const uint4* base; const uint8_t* p4; const uint32_t* d4; const uint64_t* flags; const uint8_t* var; };
struct Ctx { int strip, C, nm; uint64_t fl; size_t nrec_r; };

template <int BITS>
__device__ __forceinline__ void load_plane(uint32_t* w, const uint8_t* __restrict__ p, size_t rec, size_t nrec)
{
    constexpr int n4 = BITS / 128, r1 = BITS % 128, n2 = r1 / 64, r2 = r1 % 64, n1 = r2 / 32, nh = (r2 % 32) / 16;
    int k = 0;
    #pragma unroll
    for (int i = 0; i < n4; ++i) { uint4 v = __ldg((const uint4*)p + rec * n4 + i); w[k++] = v.x; w[k++] = v.y; w[k++] = v.z; w[k++] = v.w; }
    p += nrec * 16 * n4;
    if constexpr (n2) { uint2 v = __ldg((const uint2*)p + rec); w[k++] = v.x; w[k++] = v.y; p += nrec * 8; }
    if constexpr (n1) { w[k++] = __ldg((const uint32_t*)p + rec); p += nrec * 4; }
    if constexpr (nh) { w[k++] = __ldg((const unsigned short*)p + rec); p += nrec * 2; }
    // tail: BITS % 16 bits (multiple of 4) per record, bit-packed over records (record rec at bit rec*T)
    constexpr int T = BITS % 16;
    if constexpr (T)
    {
        const size_t b = rec * T; const uint32_t* q = (const uint32_t*)p + (b >> 5); const int sh = b & 31;
#ifdef NQ_TAIL_BR
        uint32_t v = __ldg(q) >> sh;
        if constexpr (32 % T) { if (sh + T > 32) v |= __ldg(q + 1) << (32 - sh); }
#else
        uint32_t v;
        if constexpr (32 % T) v = __funnelshift_r(__ldg(q), __ldg(q + 1), sh);   // branch-free; q + 1 stays inside the 4 B pad
        else v = __ldg(q) >> sh;
#endif
        v &= (1u << T) - 1u;
        if constexpr (((BITS - T) & 31) == 16) w[k - 1] |= v << 16; else w[k++] = v;
    }
}
// ring wrap: stream bits [BITS, BITS+32) = ring neighbour's first word
template <int BITS, int NW>
__device__ __forceinline__ void ext_words(uint32_t* w, uint32_t nb)
{
    constexpr int T = BITS % 32;
    if constexpr (T == 0) { w[NW] = nb; w[NW + 1] = 0; }
    else { w[NW - 1] = (w[NW - 1] & ((1u << T) - 1u)) | (nb << T); w[NW] = nb >> (32 - T); w[NW + 1] = 0; }
}

// LV: 2 = base only, 4 = base + P4 on every chunk, 5 = base + P4 on flagged chunks (mask mode)
template <int LV, int CPW, int RC>
struct Stage { uint32_t wb[CPW][4]; uint32_t wr[CPW][LV >= 4 ? RKB<RC>::NW : 1]; uint32_t dl[CPW]; uint32_t on; };

template <int LV, int CPW, int RC>
__device__ __forceinline__ void load_stage(Stage<LV, CPW, RC>& S, const Planes& P, const Ctx& X, int ch0, int lane)
{
    S.on = 0;
    #pragma unroll
    for (int c = 0; c < CPW; ++c)
    {
        const int ch = ch0 + c;
        const size_t rec = (size_t)X.strip * X.C + ch;
        uint4 v = P.base[rec * 32 + lane];
        S.wb[c][0] = v.x; S.wb[c][1] = v.y; S.wb[c][2] = v.z; S.wb[c][3] = v.w;
        if (P.var) S.on |= ((uint32_t)(__ldg(P.var + rec) >> (lane >> 2)) & 1u) << (16 + c);   // base variant sign of this lane's ring
        if constexpr (LV >= 4)
        {
            size_t ri = rec; bool on = true;
            if constexpr (LV == 5)
            {
                on = (X.fl >> ch) & 1;
                ri = (size_t)X.strip * X.nm + __popcll(X.fl & ((1ull << ch) - 1ull));
            }
            if (on)
            {
                load_plane<RKB<RC>::BITS>(S.wr[c], P.p4, ri * 32 + lane, X.nrec_r);
                S.dl[c] = __ldg(P.d4 + ri);
                S.on |= 1u << c;
            }
        }
    }
}

__device__ __forceinline__ void mma16816(float* c, const uint32_t* a, uint32_t b0, uint32_t b1)
{
    asm volatile ("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
template <int G> __device__ __forceinline__ int ring_src(int lane)
{
    if constexpr (G == 1) return lane;
    else if constexpr (G == 2) return lane ^ 1;
    else return (lane & ~3) | ((lane + 1) & 3);
}
struct Consts { uint32_t Mrep, Nrep, Ah, Ch; };
// lane pair P (lane weights 2P, 2P+1) -> fp16x2 A-fragment register
template <bool RES, int RC, int P>
__device__ __forceinline__ uint32_t dec_pair(const uint32_t* w, const uint32_t* r, const Consts& k)
{
    const uint32_t xb0 = wv<4 * P, 2, 0>(w) * HC, xb1 = wv<4 * P + 2, 2, 0>(w) * HC;
    if constexpr (!RES)
        return hfma2u(__byte_perm(dp4u(xb0, 0x01010101u, 0x6400u), dp4u(xb1, 0x01010101u, 0x6400u), 0x5410), k.Ah, k.Ch);
    else
    {
        constexpr int KA = RK<RC>::KA, M = RK<RC>::M;
        constexpr int o0 = step_off(2 * P, KA, M), o1 = step_off(2 * P + 1, KA, M);
        uint32_t t0 = dp4u(xb0, k.Mrep, 0x640080u), t1 = dp4u(xb1, k.Mrep, 0x640080u);
        t0 = dp4u(wv<o0, KA, M>(r) * HC, k.Nrep, t0); t1 = dp4u(wv<o1, KA, M>(r) * HC, k.Nrep, t1);
        return hfma2u(__byte_perm(t0, t1, 0x6521), k.Ah, k.Ch);
    }
}
#ifdef NQ_WDUMP
struct WDump { half* p; int K; };   // debug: decoded weights of one projection, row-major [N][K]
#endif
// one 16x128 chunk: decode into A fragments, 8 MMAs; xs holds ntok rows of kslice halves
template <bool RES, int RC>
__device__ __forceinline__ void chunk_mma(const uint32_t* w, const uint32_t* r, uint32_t dl, uint32_t sgm, float* acc, const half* xs,
                                          int kslice, int kc, int lane, int ntok
#ifdef NQ_WDUMP
                                          , half* wd, int WK
#endif
                                          )
{
    const int g = lane >> 2, t4 = lane & 3;
    Consts k{};
    if constexpr (RES)
    {
        // exactly ref15_spec.consts (fp32 ops in spec order, no contraction)
        const uint32_t Mb = dl & 0xFF, N = (dl >> 8) & 0xFF;
        k.Mrep = Mb * 0x01010101u; k.Nrep = N * 0x01010101u;
        const float rc = c_rcp[Mb];
        const half Ah = __float2half_rn(__fmul_rn(1.732421875f, rc));
        const float C = __fsub_rn(__fadd_rn(__fmul_rn(__fmul_rn((float)N, rc), -3.453125f), -3.453125f), __fmul_rn(1024.f, __half2float(Ah)));
        half2 a2 = __half2half2(Ah), c2 = __half2half2(__float2half_rn(C));
        k.Ah = *(uint32_t*)&a2 ^ sgm; k.Ch = *(uint32_t*)&c2 ^ sgm;
    }
    else { k.Ah = MUL1_A ^ sgm; k.Ch = MUL1_B ^ sgm; }   // base variant a = -1: fp16(-A), fp16(-B) (exact negation)
    #pragma unroll
    for (int t = 0; t < 8; ++t)
    {
        uint32_t a[4];
        switch (t)
        {
#define KT(T) case T: a[0] = dec_pair<RES, RC, 4 * T>(w, r, k); a[1] = dec_pair<RES, RC, 4 * T + 1>(w, r, k); \
                      a[2] = dec_pair<RES, RC, 4 * T + 2>(w, r, k); a[3] = dec_pair<RES, RC, 4 * T + 3>(w, r, k); break;
            KT(0) KT(1) KT(2) KT(3) KT(4) KT(5) KT(6) default: KT(7)
#undef KT
        }
#ifdef NQ_WDUMP
        if (wd)
            #pragma unroll
            for (int q = 0; q < 4; ++q)
                *(uint32_t*)(wd + (size_t)(g + (q & 1) * 8) * WK + t * 16 + t4 * 2 + (q >> 1) * 8) = a[q];
#endif
        uint32_t b0 = 0, b1 = 0;
        if (g < ntok) { const half* xr = xs + g * kslice + kc + t * 16 + t4 * 2; b0 = *(const uint32_t*)xr; b1 = *(const uint32_t*)(xr + 8); }
        mma16816(acc, a, b0, b1);
    }
}
template <int LV, int G, int CPW, int RC>
__device__ __forceinline__ void compute_stage(const Stage<LV, CPW, RC>& S, float* acc, const half* xs, int kslice, int kc0, int lane, int ntok
#ifdef NQ_WDUMP
                                              , half* wd, int WK
#endif
                                              )
{
    const int src = ring_src<G>(lane);
    constexpr int NWR = RKB<RC>::NW;
    #pragma unroll
    for (int c = 0; c < CPW; ++c)
    {
        uint32_t w[6], r[NWR + 2];
        #pragma unroll
        for (int i = 0; i < 4; ++i) w[i] = S.wb[c][i];
        ext_words<128, 4>(w, __shfl_sync(0xffffffffu, w[0], src));
        const int kc = kc0 + c * 128;
        const uint32_t sgm = ((S.on >> (16 + c)) & 1u) * 0x80008000u;
#ifdef NQ_WDUMP
#define WDA , wd ? wd + kc : nullptr, WK
#else
#define WDA
#endif
        if constexpr (LV == 2) chunk_mma<false, RC>(w, r, 0, sgm, acc, xs, kslice, kc, lane, ntok WDA);
        else
        {
            if (LV == 4 || ((S.on >> c) & 1))
            {
                #pragma unroll
                for (int i = 0; i < NWR; ++i) r[i] = S.wr[c][i];
                ext_words<RKB<RC>::BITS, NWR>(r, __shfl_sync(0xffffffffu, r[0], src));
                chunk_mma<true, RC>(w, r, S.dl[c], sgm, acc, xs, kslice, kc, lane, ntok WDA);
            }
            else chunk_mma<false, RC>(w, r, 0, sgm, acc, xs, kslice, kc, lane, ntok WDA);
        }
#undef WDA
    }
}
}  // namespace nqdec
// ============================== end decoder ======================================================================

using namespace nqdec;

__device__ __forceinline__ void wht128_warp(float* v, int lane)
{
    float a0 = v[0] + v[1], a1 = v[0] - v[1], a2 = v[2] + v[3], a3 = v[2] - v[3];
    v[0] = a0 + a2; v[1] = a1 + a3; v[2] = a0 - a2; v[3] = a1 - a3;
    #pragma unroll
    for (int m = 1; m < 32; m <<= 1)
    {
        #pragma unroll
        for (int i = 0; i < 4; ++i) { float o = __shfl_xor_sync(0xffffffffu, v[i], m); v[i] = (lane & m) ? (o - v[i]) : (v[i] + o); }
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i) v[i] *= 0.08838834764f;
}
__device__ __forceinline__ void ld4h(const half* p, float* f)
{
    uint2 u = *(const uint2*)p; half2 a = *(half2*)&u.x, b = *(half2*)&u.y;
    f[0] = __low2float(a); f[1] = __high2float(a); f[2] = __low2float(b); f[3] = __high2float(b);
}
__device__ __forceinline__ void st4h(half* p, const float* v)
{
    half2 h0 = __floats2half2_rn(v[0], v[1]), h1 = __floats2half2_rn(v[2], v[3]);
    uint2 u; u.x = *(uint32_t*)&h0; u.y = *(uint32_t*)&h1; *(uint2*)p = u;
}

#define TBL_W 20
struct MoeArgs
{
    const half* x; const int64_t* sel; const half* rw; int B, topk;
    const int64_t* table; int H, I; int nm_gu, nm_dn;
    float* acc_gu; half* h; float* acc_d; float* out;
    int* cnt_gu; int* cnt_d; int NST;
    int force_level;   // debug: >0 overrides table level
    half* wdump[2]; int dump_e;   // NQ_WDUMP debug builds only: decoded fp16 weights of expert dump_e (gu [2I][H], dn [H][I])
    int* hits;         // optional [E] int32 pick counters (may be host-mapped pinned memory); nullptr = off
    float* zd;         // [S][I/128][4] down lr partials (written by K1, read by K2; no zeroing needed)
    int* cnt_h;        // [S*I/128] i32 down-input Hadamard block counters (in_had_down > 128; zero on entry/exit)
};

struct RunInfo { int e, level, ntok, nruns; int slot[8]; float w[8]; const int64_t* ent; };

// one warp: dedup the routing (S = B*topk <= 64 entries, entry s = lane + 32*h), fill run z (z < 0: all runs).
// Runs are numbered by first occurrence; slots within a run are in entry order (deterministic).
__device__ __forceinline__ void route(const MoeArgs& a, int z, RunInfo* R, int lane)
{
    const int S = a.B * a.topk;
    int64_t e[2]; bool ok[2]; unsigned m[2];
    #pragma unroll
    for (int h = 0; h < 2; ++h)
    {
        const int s = lane + 32 * h;
        e[h] = -1 - s; ok[h] = false;
        if (s < S)
        {
            int64_t ee = a.sel[s];
            float w = __half2float(a.rw[s]);
            int lv = (int)__ldg(a.table + ee * TBL_W);
            if (w != 0.f && lv > 0) { e[h] = ee; ok[h] = true; }
        }
        m[h] = __match_any_sync(0xffffffffu, (unsigned long long)e[h]);
    }
    // first occurrence + number of earlier matches in half 0, for half-1 entries
    int first1 = 32 + __ffs(m[1]) - 1, pre1 = 0;
    if (S > 32)
        for (int i = 0; i < 32; ++i)
        {
            const int64_t v = __shfl_sync(0xffffffffu, e[0], i);
            if (v == e[1]) { if (first1 >= 32) first1 = i; ++pre1; }
        }
    const unsigned below = (1u << lane) - 1u;
    const bool lead0 = ok[0] && lane == __ffs(m[0]) - 1, lead1 = ok[1] && first1 == lane + 32;
    const unsigned lm0 = __ballot_sync(0xffffffffu, lead0), lm1 = __ballot_sync(0xffffffffu, lead1);
    const int nr0 = __popc(lm0);
    const int rank0 = __popc(lm0 & below), rank1 = nr0 + __popc(lm1 & below);
    if (lane == 0) R->nruns = nr0 + __popc(lm1);
    // run index of each entry = rank of its leader
    const int first0 = __ffs(m[0]) - 1;
    const int r0 = __shfl_sync(0xffffffffu, rank0, first0);
    const int rA = __shfl_sync(0xffffffffu, rank0, first1 & 31), rB = __shfl_sync(0xffffffffu, rank1, first1 & 31);
    const int r1 = first1 < 32 ? rA : rB;
    const int pos0 = __popc(m[0] & below), pos1 = pre1 + __popc(m[1] & below);
    const int cnt0 = __popc(m[0]);
    // leaders: header; total count = half-0 members + half-1 members (a half-1 leader has no half-0 members)
    int n1 = 0;   // half-1 members of a half-0 leader's expert
    if (S > 32)
        for (int i = 0; i < 32; ++i)
        {
            const int64_t v = __shfl_sync(0xffffffffu, e[1], i);
            const bool okv = __shfl_sync(0xffffffffu, (int)ok[1], i);
            if (okv && v == e[0]) ++n1;
        }
    #pragma unroll
    for (int h = 0; h < 2; ++h)
    {
        const bool ld = h ? lead1 : lead0; const int rk = h ? rank1 : rank0;
        if (ld && (z < 0 || rk == z))
        {
            RunInfo* Q = z < 0 ? R + rk : R;
            Q->e = (int)e[h]; Q->ent = a.table + e[h] * TBL_W;
            int lv = (int)Q->ent[0]; if (a.force_level) lv = a.force_level; Q->level = lv;
            Q->ntok = h ? __popc(m[1]) : cnt0 + n1;
        }
    }
    #pragma unroll
    for (int h = 0; h < 2; ++h)
    {
        const int rk = h ? r1 : r0, ps = h ? pos1 : pos0, s = lane + 32 * h;
        if (ok[h] && (z < 0 || rk == z))
        {
            RunInfo* Q = z < 0 ? R + rk : R;
            Q->slot[ps] = s; Q->w[ps] = __half2float(a.rw[s]);
        }
    }
}

#ifdef NQ_WDUMP
#define WDP , half* wd, int WK
#define WDC(kc) , wd ? wd + (size_t)strip * 16 * WK + (kc) : nullptr, WK
#else
#define WDP
#define WDC(kc)
#endif
template <int LV, int G, int CPW, int RC>
__device__ __forceinline__ void gemv_body(const Planes& P, int strip, int C, int nm, int nstrips, int chunk0, int NST, float* acc,
                                          const half* xs, int kslice, int lane, int ntok WDP)
{
    Ctx X; X.strip = strip; X.C = C; X.nm = nm; X.fl = 0;
    X.nrec_r = (size_t)nstrips * (LV == 5 ? nm : C) * 32;
    if constexpr (LV == 5) X.fl = __ldg((const unsigned long long*)P.flags + strip);
    Stage<LV, CPW, RC> SA, SB_;
    load_stage<LV, CPW, RC>(SA, P, X, chunk0, lane);
    if (NST > 1) load_stage<LV, CPW, RC>(SB_, P, X, chunk0 + CPW, lane);
    const int kabs = chunk0 * 128;
    for (int s = 0; s < NST; s += 2)
    {
        compute_stage<LV, G, CPW, RC>(SA, acc, xs, kslice, s * CPW * 128, lane, ntok WDC(kabs));
        if (s + 2 < NST) load_stage<LV, CPW, RC>(SA, P, X, chunk0 + (s + 2) * CPW, lane);
        if (s + 1 < NST)
        {
            compute_stage<LV, G, CPW, RC>(SB_, acc, xs, kslice, (s + 1) * CPW * 128, lane, ntok WDC(kabs));
            if (s + 3 < NST) load_stage<LV, CPW, RC>(SB_, P, X, chunk0 + (s + 3) * CPW, lane);
        }
    }
}

__device__ __forceinline__ Planes planes_of(const int64_t* ent, int off)
{
    Planes P;
    P.base = (const uint4*)ent[off]; P.p4 = (const uint8_t*)ent[off + 1];
    P.d4 = (const uint32_t*)ent[off + 2]; P.flags = (const uint64_t*)ent[off + 3];
    P.var = (const uint8_t*)ent[off == 1 ? 12 : 13];
    return P;
}

// MODE 0: gate|up (N = 2I, K = H). MODE 1: down (N = H, K = I).
// One work item = (run z, row block bx, k-slice by). NY = number of k-slices. Must be called by the whole block.
template <int G, int CPW, int MODE>
__device__ __forceinline__ void do_item(const MoeArgs& a, const RunInfo& R, int nruns, int z, int bx, int by, int NY,
                                        uint32_t* smem, int* last_s)
{
    int& last = *last_s;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, SB = blockDim.x >> 5;
    const int N = MODE == 0 ? 2 * a.I : a.H, K = MODE == 0 ? a.H : a.I;
    const int strip = bx * SB + warp;
    const int C = K / 128, kslice = CPW * a.NST * 128, k0 = by * kslice;
    const int ntok = R.ntok, lv = R.level;
    const half* signs = (const half*)R.ent[9];
    half* xs = (half*)smem;

    // prologue: stage the run's token inputs for this k-slice
    if constexpr (MODE == 0)
    {
        const int ng = kslice / 128;
        const half* su = bx * SB * 16 >= a.I ? signs + 2 * a.H + 3 * a.I : signs;   // gate rows: su_g, up rows: su_u
        for (int task = warp; task < ntok * ng; task += SB)
        {
            int j = task / ng, gg = task % ng, k = k0 + gg * 128 + lane * 4;
            int tok = R.slot[j] / a.topk;
            float f[4], s[4]; ld4h(a.x + (size_t)tok * a.H + k, f); ld4h(su + k, s);
            float v[4] = {f[0] * s[0], f[1] * s[1], f[2] * s[2], f[3] * s[3]};
            wht128_warp(v, lane);
            st4h(xs + j * kslice + gg * 128 + lane * 4, v);
        }
    }
    else
    {
        for (int i = threadIdx.x; i < ntok * kslice / 8; i += blockDim.x)
        {
            int j = i / (kslice / 8), kk = i % (kslice / 8);
            ((uint4*)(xs + j * kslice))[kk] = ((const uint4*)(a.h + (size_t)R.slot[j] * a.I + k0))[kk];
        }
    }
    __syncthreads();

    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const Planes P = planes_of(R.ent, MODE == 0 ? 1 : 5);
    const int nm = MODE == 0 ? a.nm_gu : a.nm_dn;
    const int chunk0 = by * CPW * a.NST;
#ifdef NQ_WDUMP
    half* wd = (a.wdump[MODE] && R.e == a.dump_e) ? a.wdump[MODE] : nullptr; const int WK = K;
#define WDX , wd, WK
#else
#define WDX
#endif
    const int NS = N / 16, rc = (int)R.ent[10 + MODE];
#define BODY(LVv, RCv) gemv_body<LVv, G, CPW, RCv>(P, strip, C, nm, NS, chunk0, a.NST, acc, xs, kslice, lane, ntok WDX)
#ifdef NQ_NO_MASK
#define LV45(RCv) BODY(4, RCv)
#else
#define LV45(RCv) { if (P.flags) BODY(5, RCv); else BODY(4, RCv); }
#endif
    constexpr unsigned RKM = ((MODE == 0 ? NQ_RK_GU : NQ_RK_DN) & NQ_RK_CODES) | 1;
    const int rcm = (RKM >> rc) & 1 ? rc : 0;
    if (lv == 2) BODY(2, 0);
    else switch (rcm)
    {
#if NQ_RK_CODES & 2
        case 1: if constexpr (RKM & 2) LV45(1); break;
#endif
#if NQ_RK_CODES & 4
        case 2: if constexpr (RKM & 4) LV45(2); break;
#endif
#if NQ_RK_CODES & 8
        case 3: if constexpr (RKM & 8) LV45(3); break;
#endif
#if NQ_RK_CODES & 16
        case 4: if constexpr (RKM & 16) LV45(4); break;
#endif
#if NQ_RK_CODES & 32
        case 5: if constexpr (RKM & 32) LV45(5); break;
#endif
#if NQ_RK_CODES & 64
        case 6: if constexpr (RKM & 64) LV45(6); break;
#endif
#if NQ_RK_CODES & 128
        case 7: if constexpr (RKM & 128) LV45(7); break;
#endif
#if NQ_RK_CODES & 256
        case 8: if constexpr (RKM & 256) LV45(8); break;
#endif
        default: LV45(0); break;
    }
#undef BODY
#undef LV45
#undef WDX

    float* accb = MODE == 0 ? a.acc_gu : a.acc_d;
    {
        const int g = lane >> 2, t4 = lane & 3;
        #pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            int bb = t4 * 2 + (i & 1), row = strip * 16 + g + (i >> 1) * 8;
            if (bb < ntok) atomicAdd(accb + (size_t)R.slot[bb] * N + row, acc[i]);
        }
    }
    // arrival counters
    const int rows0 = bx * SB * 16;
    const int grp = MODE == 0 ? (rows0 % a.I) / 128 : rows0 / 128;
    int* cnt = MODE == 0 ? a.cnt_gu + z * (a.I / 128) + grp : a.cnt_d + grp;
    const int expect = (MODE == 0 ? 2 : nruns) * NY * (128 / (SB * 16));
    __syncthreads();
    if (threadIdx.x == 0)
    {
        int prev;
        asm volatile ("atom.add.acq_rel.gpu.global.s32 %0, [%1], 1;" : "=r"(prev) : "l"(cnt) : "memory");
        last = (prev == expect - 1);
    }
    __syncthreads();
    if (!last) return;   // block-uniform
    if constexpr (MODE == 0)
    {
        const half* sv_g = signs + a.H; const half* sv_u = sv_g + a.I; const half* su_d = sv_u + a.I;
        float* sg = (float*)smem;   // [ntok][2][128]
        __shared__ float zs[8 * 4];
        const half* lrp = (const half*)R.ent[14];
        const int rg = lrp ? (int)R.ent[16] : 0, rd = lrp ? (int)R.ent[17] : 0;
        const half* U4 = lv == 4 ? (const half*)R.ent[15] : nullptr;
        const int hw = (int)R.ent[18];
        for (int task = warp; task < ntok * rg; task += SB)   // z_gu = x V_gu^T (unrotated x)
        {
            int j = task / rg, r = task % rg, tok = R.slot[j] / a.topk; float d = 0.f;
            for (int k = lane * 4; k < a.H; k += 128)
            {
                float f[4], v[4]; ld4h(a.x + (size_t)tok * a.H + k, f); ld4h(lrp + (size_t)r * a.H + k, v);
                d += f[0] * v[0] + f[1] * v[1] + f[2] * v[2] + f[3] * v[3];
            }
            #pragma unroll
            for (int o = 16; o; o >>= 1) d += __shfl_xor_sync(0xffffffffu, d, o);
            if (lane == 0) zs[j * 4 + r] = d;
        }
        if (rg) __syncthreads();
        for (int task = warp; task < 2 * ntok; task += SB)
        {
            int j = task >> 1, up = task & 1, col = grp * 128 + lane * 4;
            float* p = a.acc_gu + (size_t)R.slot[j] * N + up * a.I + col;
            float4 f = __ldcg((const float4*)p); __stcg((float4*)p, make_float4(0, 0, 0, 0));
            float v[4] = {f.x, f.y, f.z, f.w}, s[4];
            wht128_warp(v, lane);
            ld4h((up ? sv_u : sv_g) + col, s);
            #pragma unroll
            for (int i = 0; i < 4; ++i) v[i] *= s[i];
            for (int r = 0; r < rg; ++r)
            {
                const float z = zs[j * 4 + r]; float u2[4], u4[4] = {0, 0, 0, 0};
                const size_t o = (size_t)rg * a.H + (size_t)(up * rg + r) * a.I + col;
                ld4h(lrp + o, u2); if (U4) ld4h(U4 + (o - (size_t)rg * a.H), u4);
                #pragma unroll
                for (int i = 0; i < 4; ++i) v[i] += z * u2[i] + z * u4[i];
            }
            #pragma unroll
            for (int i = 0; i < 4; ++i) sg[(j * 2 + up) * 128 + lane * 4 + i] = v[i];
        }
        __syncthreads();
        for (int j = warp; j < ntok; j += SB)
        {
            int col = grp * 128 + lane * 4; float s[4], v[4]; ld4h(su_d + col, s);
            #pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                float gg = sg[(j * 2) * 128 + lane * 4 + i], uu = sg[(j * 2 + 1) * 128 + lane * 4 + i];
                v[i] = gg / (1.f + __expf(-gg)) * uu;
            }
            for (int r = 0; r < rd; ++r)   // down lr partial over these 128 SwiGLU columns
            {
                float vd[4]; ld4h(lrp + (size_t)rg * a.H + 2 * (size_t)rg * a.I + (size_t)r * a.I + col, vd);
                float d = v[0] * vd[0] + v[1] * vd[1] + v[2] * vd[2] + v[3] * vd[3];
                #pragma unroll
                for (int o = 16; o; o >>= 1) d += __shfl_xor_sync(0xffffffffu, d, o);
                if (lane == 0) a.zd[((size_t)R.slot[j] * (a.I / 128) + grp) * 4 + r] = d;
            }
            #pragma unroll
            for (int i = 0; i < 4; ++i) v[i] *= s[i];
            wht128_warp(v, lane);
            if (hw <= 128) st4h(a.h + (size_t)R.slot[j] * a.I + col, v);
            else __stcg((float4*)(a.acc_gu + (size_t)R.slot[j] * N + col), make_float4(v[0], v[1], v[2], v[3]));
        }
        if (hw > 128)   // cross-block stage of WHT_hw (block-uniform branch)
        {
            const int nb = hw / 128, b0 = grp / nb * nb;
            __shared__ int last_h;
            __threadfence();
            __syncthreads();
            if (threadIdx.x == 0)
            {
                int prev; int* ch = a.cnt_h + z * (a.I / 128) + b0;
                asm volatile ("atom.add.acq_rel.gpu.global.s32 %0, [%1], 1;" : "=r"(prev) : "l"(ch) : "memory");
                last_h = (prev == nb - 1);
                if (last_h) *ch = 0;
            }
            __syncthreads();
            if (last_h)
            {
                const float rs = rsqrtf((float)nb);
                for (int task = warp; task < ntok * nb; task += SB)
                {
                    const int j = task / nb, ob = task % nb;
                    const float* p = a.acc_gu + (size_t)R.slot[j] * N + b0 * 128 + lane * 4;
                    float o[4] = {0, 0, 0, 0};
                    for (int b = 0; b < nb; ++b)
                    {
                        const float4 f = __ldcg((const float4*)(p + b * 128));
                        const float sgn = (__popc(ob & b) & 1) ? -1.f : 1.f;
                        o[0] += sgn * f.x; o[1] += sgn * f.y; o[2] += sgn * f.z; o[3] += sgn * f.w;
                    }
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) o[i] *= rs;
                    st4h(a.h + (size_t)R.slot[j] * a.I + (b0 + ob) * 128 + lane * 4, o);
                }
                __syncthreads();   // every read of the fp32 stage is done before it is re-zeroed
                for (int task = warp; task < ntok * nb; task += SB)
                {
                    const int j = task / nb, ob = task % nb;
                    __stcg((float4*)(a.acc_gu + (size_t)R.slot[j] * N + (b0 + ob) * 128 + lane * 4), make_float4(0, 0, 0, 0));
                }
            }
        }
    }
    else
    {
        // fused combine for output rows [grp*128, +128): every token, every active slot
        const int S = a.B * a.topk;
        float* part = (float*)smem;   // [SB][4 tokens max?] -> per (warp, token) 128 floats
        for (int b = 0; b < a.B; ++b)
        {
            float o[4] = {0, 0, 0, 0};
            for (int k = warp; k < a.topk; k += SB)
            {
                int s = b * a.topk + k; if (s >= S) break;
                int64_t e = a.sel[s]; float w = __half2float(a.rw[s]);
                const int64_t* ent = a.table + e * TBL_W;
                if (w == 0.f || ent[0] <= 0) continue;
                const half* sv_o = (const half*)ent[9] + a.H + 3 * a.I;
                float* p = a.acc_d + (size_t)s * a.H + grp * 128 + lane * 4;
                float4 f = __ldcg((const float4*)p); __stcg((float4*)p, make_float4(0, 0, 0, 0));
                float v[4] = {f.x, f.y, f.z, f.w}, sg[4];
                wht128_warp(v, lane);
                ld4h(sv_o + grp * 128 + lane * 4, sg);
                #pragma unroll
                for (int i = 0; i < 4; ++i) v[i] *= sg[i];
                const half* lrp = (const half*)ent[14];
                const int rd = lrp ? (int)ent[17] : 0;
                if (rd)
                {
                    const int rg = (int)ent[16], lvk = a.force_level ? a.force_level : (int)ent[0], ng = a.I / 128;
                    const size_t ou = (size_t)rg * a.H + 2 * (size_t)rg * a.I + (size_t)rd * a.I;
                    const half* U4 = lvk == 4 ? (const half*)ent[15] : nullptr;
                    for (int r = 0; r < rd; ++r)
                    {
                        float z = 0.f;
                        for (int gq = 0; gq < ng; ++gq) z += a.zd[((size_t)s * ng + gq) * 4 + r];
                        float u2[4], u4[4] = {0, 0, 0, 0}; const size_t o = (size_t)r * a.H + grp * 128 + lane * 4;
                        ld4h(lrp + ou + o, u2); if (U4) ld4h(U4 + 2 * (size_t)rg * a.I + o, u4);
                        #pragma unroll
                        for (int i = 0; i < 4; ++i) v[i] += z * u2[i] + z * u4[i];
                    }
                }
                #pragma unroll
                for (int i = 0; i < 4; ++i) o[i] += w * v[i];
            }
            #pragma unroll
            for (int i = 0; i < 4; ++i) part[warp * 128 + lane * 4 + i] = o[i];
            __syncthreads();
            if (warp == 0)
            {
                float r[4] = {0, 0, 0, 0};
                for (int w2 = 0; w2 < SB; ++w2)
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) r[i] += part[w2 * 128 + lane * 4 + i];
                *(float4*)(a.out + (size_t)b * a.H + grp * 128 + lane * 4) = make_float4(r[0], r[1], r[2], r[3]);
            }
            __syncthreads();
        }
    }
    if (threadIdx.x == 0) *cnt = 0;
}

#ifndef NQ_MINB
#define NQ_MINB 4
#endif
// grid-mapped kernel: grid (N/16/SB, K/kslice, B*topk); blocks with z >= nruns exit
template <int G, int CPW, int MODE>
__global__ void __launch_bounds__(256, NQ_MINB) nq_moe(MoeArgs a)
{
    extern __shared__ uint32_t smem[];
    __shared__ RunInfo R;
    __shared__ int nr, last;
    if (threadIdx.x < 32) { route(a, blockIdx.z, &R, threadIdx.x); if (threadIdx.x == 0) nr = R.nruns; }
    // routing-hit export for the CPU scheduler: one warp of one K1 block counts every (token, expert) pick,
    // whatever its level (the host owns the levels, so misses/upgrade candidates = hits where level < 4)
    if (MODE == 0 && a.hits && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x < a.B * a.topk)
        if (__half2float(a.rw[threadIdx.x]) != 0.f) atomicAdd(a.hits + a.sel[threadIdx.x], 1);
    __syncthreads();
    if ((int)blockIdx.z >= nr) return;
    do_item<G, CPW, MODE>(a, R, nr, blockIdx.z, blockIdx.x, blockIdx.y, gridDim.y, smem, &last);
}

// persistent kernel: gridDim.x resident blocks pull items from a device work counter (wq[0]); the last block to leave
// resets the counters (wq[0], wq[1]) so the kernel is graph-replayable. Items are run-major.
template <int G, int CPW, int MODE>
__global__ void __launch_bounds__(256, NQ_MINB) nq_moe_p(MoeArgs a, int NX, int NY, int* wq)
{
    extern __shared__ uint32_t smem[];
    __shared__ RunInfo R[64];
    __shared__ int nr, last, item;
    if (threadIdx.x < 32)
    {
        route(a, -1, R, threadIdx.x);
        __syncwarp();
        if (threadIdx.x == 0) nr = R[0].nruns;
    }
    __syncthreads();
    const int total = nr * NX * NY;
    while (true)
    {
        if (threadIdx.x == 0) item = atomicAdd(wq, 1);
        __syncthreads();
        const int it = item;
        if (it >= total) break;
        const int z = it / (NX * NY), r = it % (NX * NY);
        do_item<G, CPW, MODE>(a, R[z], nr, z, r % NX, r / NX, NY, smem, &last);
        __syncthreads();
    }
    if (threadIdx.x == 0)
    {
        __threadfence();
        if (atomicAdd(wq + 1, 1) == (int)gridDim.x - 1) { wq[0] = 0; wq[1] = 0; __threadfence(); }
    }
}

typedef void (*kfn)(MoeArgs);
typedef void (*kfp)(MoeArgs, int, int, int*);
template <int MODE> kfn pick(int g, int cpw)
{
    if (g == 2) { if (cpw == 1) return nq_moe<2, 1, MODE>; if (cpw == 2) return nq_moe<2, 2, MODE>; }
    if (g == 4) { if (cpw == 1) return nq_moe<4, 1, MODE>; if (cpw == 2) return nq_moe<4, 2, MODE>; }
    return nullptr;
}
template <int MODE> kfp pickp(int g, int cpw)
{
    if (g == 2) { if (cpw == 1) return nq_moe_p<2, 1, MODE>; if (cpw == 2) return nq_moe_p<2, 2, MODE>; }
    if (g == 4) { if (cpw == 1) return nq_moe_p<4, 1, MODE>; if (cpw == 2) return nq_moe_p<4, 2, MODE>; }
    return nullptr;
}
static int resident_blocks(const void* f, int threads, int shm)
{
    static std::map<std::tuple<const void*, int, int>, int> cache;
    auto k = std::make_tuple(f, threads, shm);
    auto it = cache.find(k); if (it != cache.end()) return it->second;
    { cudaFuncAttributes fa; cudaFuncGetAttributes(&fa, f);   // SM120: 99 KiB opt-in includes static smem
      cudaError_t e = cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024 - (int)fa.sharedSizeBytes); TORCH_CHECK(e == cudaSuccess, "setattr: ", cudaGetErrorString(e)); }
    int nb = 0, dev = 0, sms = 0; cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, f, threads, shm);
    cudaGetDevice(&dev); cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    return cache[k] = nb * sms;
}

static int64_t g_wdump[2] = {0, 0}; static int g_dump_e = -1;
void set_wdump(int64_t gu, int64_t dn, int64_t e) { g_wdump[0] = gu; g_wdump[1] = dn; g_dump_e = (int)e; }
// One MoE layer forward (graph-capturable; all routing/level/pointer decisions are read on device).
// cfg_gu/cfg_dn: (cpw, sb, nst[, persistent]).  ws: acc_gu [S,2I] f32, h [S,I] f16, acc_d [S,H] f32,
// cnt_gu [S*I/128] i32, cnt_d [H/128] i32, wq [4] i32 (persistent work counters) -- all zero on entry, zero on exit;
// zd [S*I/128*4] f32 (lr down partials, fully rewritten by K1 before K2 reads it); cnt_h [S*I/128] i32 (zero on entry/exit).
void moe_forward(torch::Tensor x, torch::Tensor sel, torch::Tensor rw, torch::Tensor table, torch::Tensor out,
                 torch::Tensor acc_gu, torch::Tensor h, torch::Tensor acc_d, torch::Tensor cnt_gu, torch::Tensor cnt_d,
                 torch::Tensor wq, int64_t I, int64_t nm_gu, int64_t nm_dn, std::vector<int64_t> cfg_gu,
                 std::vector<int64_t> cfg_dn, int64_t G, int64_t force_level, int64_t which, int64_t hits_ptr, torch::Tensor zd,
                 torch::Tensor cnt_h)
{
    MoeArgs a{};
    a.x = (const half*)x.data_ptr(); a.sel = (const int64_t*)sel.data_ptr(); a.rw = (const half*)rw.data_ptr();
    a.B = x.size(0); a.topk = sel.size(1); a.table = (const int64_t*)table.data_ptr(); a.H = x.size(1); a.I = I;
    a.nm_gu = nm_gu; a.nm_dn = nm_dn; a.acc_gu = (float*)acc_gu.data_ptr(); a.h = (half*)h.data_ptr();
    a.acc_d = (float*)acc_d.data_ptr(); a.out = (float*)out.data_ptr(); a.cnt_gu = (int*)cnt_gu.data_ptr(); a.cnt_d = (int*)cnt_d.data_ptr();
    a.force_level = force_level; a.hits = (int*)hits_ptr; a.zd = (float*)zd.data_ptr();
    a.cnt_h = (int*)cnt_h.data_ptr();
    TORCH_CHECK(cnt_h.scalar_type() == at::kInt && cnt_h.numel() >= (int64_t)x.size(0) * sel.size(1) * (I / 128), "cnt_h [S*I/128] i32");
    a.wdump[0] = (half*)g_wdump[0]; a.wdump[1] = (half*)g_wdump[1]; a.dump_e = g_dump_e;
    const int S = a.B * a.topk;
    TORCH_CHECK(S <= 64 && a.B <= 8, "B*topk <= 64, B <= 8");
    TORCH_CHECK(x.scalar_type() == at::kHalf && rw.scalar_type() == at::kHalf && sel.scalar_type() == at::kLong && table.scalar_type() == at::kLong, "x/rw fp16, sel/table int64");
    TORCH_CHECK(out.scalar_type() == at::kFloat && acc_gu.scalar_type() == at::kFloat && acc_d.scalar_type() == at::kFloat && zd.scalar_type() == at::kFloat && h.scalar_type() == at::kHalf, "workspace dtypes: out/acc_gu/acc_d/zd fp32, h fp16");
    TORCH_CHECK(acc_gu.numel() >= (int64_t)S * 2 * I && acc_d.numel() >= (int64_t)S * a.H && h.numel() >= (int64_t)S * I && out.numel() >= (int64_t)a.B * a.H, "workspace too small");
    TORCH_CHECK(a.H % 128 == 0 && I % 128 == 0 && a.H / 128 <= 64 && I / 128 <= 64);
    auto st = at::cuda::getCurrentCUDAStream();
    for (int mode = 0; mode < 2; ++mode)
    {
        if (!((which >> mode) & 1)) continue;
        auto& cf = mode == 0 ? cfg_gu : cfg_dn;
        int cpw = cf[0], sb = cf[1], nst = cf[2], pers = cf.size() > 3 ? cf[3] : 0;
        int N = mode == 0 ? 2 * I : a.H, K = mode == 0 ? a.H : I;
        int kslice = cpw * nst * 128;
        TORCH_CHECK(K % kslice == 0 && (N / 16) % sb == 0 && sb <= 8 && 128 % (sb * 16) == 0, "bad cfg");
        a.NST = nst;
        int shm = std::max(8 * kslice * 2, std::max(8 * 2 * 128 * 4, sb * 128 * 4));
        int NX = N / 16 / sb, NY = K / kslice;
        if (!pers)
        {
            kfn f = mode == 0 ? pick<0>(G, cpw) : pick<1>(G, cpw);
            TORCH_CHECK(f, "no kernel");
            resident_blocks((const void*)f, 32 * sb, shm);
            f<<<dim3(NX, NY, S), 32 * sb, shm, st>>>(a);
            { cudaError_t e = cudaGetLastError(); TORCH_CHECK(e == cudaSuccess, "launch: ", cudaGetErrorString(e), " grid ", NX, " ", NY, " ", S, " shm ", shm); }
        }
        else
        {
            kfp f = mode == 0 ? pickp<0>(G, cpw) : pickp<1>(G, cpw);
            TORCH_CHECK(f, "no kernel");
            int nb = std::min(resident_blocks((const void*)f, 32 * sb, shm) * (int)pers, NX * NY * S);
            f<<<nb, 32 * sb, shm, st>>>(a, NX, NY, (int*)wq.data_ptr() + 2 * mode);
        }
    }
}
// Table mailbox (graph-safe level switching without host access to the compute stream).
// Captured once at the start of each layer. The scheduler, on its own side stream and in stream order, (1) copies P4/d4
// into a free slot (upgrades only), (2) writes the new table row into stage[e], (3) bumps seq[e]. This kernel copies
// stage[e] -> table[e] whenever seq[e] != applied[e] and publishes applied[e] (device copy + optional host-mapped
// mirror). After the host sees applied == seq for a downgrade, no later kernel can reference the old slot.
// Rule: at most one outstanding op per expert (the host waits for applied == seq before re-staging e).
__global__ void nq_mailbox(int64_t* table, const int64_t* stage, const int* seq, int* applied, int* applied_host, int E)
{
    const int e = blockIdx.x * blockDim.x + threadIdx.x; if (e >= E) return;
    const int sq = __ldcv(seq + e);
    if (sq == applied[e]) return;
    #pragma unroll
    for (int i = 0; i < TBL_W; ++i) table[(size_t)e * TBL_W + i] = __ldcv((const long long*)stage + (size_t)e * TBL_W + i);
    __threadfence();
    applied[e] = sq; if (applied_host) applied_host[e] = sq;
}
void mailbox(torch::Tensor table, torch::Tensor stage, torch::Tensor seq, torch::Tensor applied, int64_t applied_host)
{
    const int E = table.size(0);
    nq_mailbox<<<(E + 127) / 128, 128, 0, at::cuda::getCurrentCUDAStream()>>>((int64_t*)table.data_ptr(), (const int64_t*)stage.data_ptr(),
        (const int*)seq.data_ptr(), (int*)applied.data_ptr(), (int*)applied_host, E);
}
int64_t occ(int64_t mode, int64_t G, int64_t cpw, int64_t sb, int64_t shm)
{
    kfn f = mode == 0 ? pick<0>(G, cpw) : pick<1>(G, cpw);
    { cudaFuncAttributes fa; cudaFuncGetAttributes(&fa, f); cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024 - (int)fa.sharedSizeBytes); }
    int nb = 0; cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, f, 32 * sb, shm);
    cudaFuncAttributes at; cudaFuncGetAttributes(&at, f);
    return nb * 1000000 + at.numRegs * 1000 + at.localSizeBytes;
}
// compiled residual K codes per kernel (bit c = code c): {gate|up, down}. Codes outside a mask silently decode as code 0,
// so hosts must check (MoELayer.set does).
std::vector<int64_t> rk_codes() { return {(NQ_RK_GU & NQ_RK_CODES) | 1, (NQ_RK_DN & NQ_RK_CODES) | 1}; }
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("moe_forward", &moe_forward); m.def("rk_codes", &rk_codes); m.def("occ", &occ); m.def("mailbox", &mailbox); m.def("set_wdump", &set_wdump); }
