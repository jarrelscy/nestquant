// Thread 15: cheaper level-4 refinement decode on top of thread 04's nqk2 (A4 additive, 128-weight rings, fused-B).
// Decoder = base plane (mul1 V=1, pattern KA_b/MASK_b, EXL3-style fractional steps) + optional residual plane.
// Residual modes:
//   RM_NONE  base only (2-bit path)
//   RM_A     mul1 V=1 residual, fp16: base = HFMA2(hb, A, B(1+d)); out = HFMA2(hr, dA, base)       (thread 04 A4)
//   RM_F     mul1 V=1 residual folded into the base dp4a in fp32: t = dp4a(xr, N, dp4a(xb, 128, 2^23)),
//            f = FFMA(t, A/128, C(N)), pack F2FP.  delta = N/128 (N = 1..255, u8 per 16x128 block)
//   RM_A2    2-D mul1 residual (V=2: one 16-bit window per PAIR, w0 = sum bytes, w1 = alternating sum), fp16 path
//   RM_F2    2-D mul1 residual, folded fp32 path (N <= 127)
//   RM_UNI4  uniform 4-bit (speed ceiling / SASS reference, base plane = 256 bits)
// Window extraction WOPT: 0 = thread-04 per-window shift+mask; 1 = shared funnels (one funnel serves windows at o, o+8, o+16).
// Planes: per (strip, chunk, lane) record of BITS = 64*K bits, stored as sub-arrays (uint4 x n4 | uint2 | uint | ushort),
// each sub-array contiguous over all records => every 16x128 block and TP8 shard is whole and aligned.
#include <cuda_fp16.h>
#include <stdint.h>
#include <map>
#include <set>
#include <tuple>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

enum { RM_NONE = 0, RM_A, RM_F, RM_A2, RM_F2, RM_UNI4, RM_P, RM_P2 };
#define MUL1_A 0x1eee1eeeu
#define MUL1_B 0xc931c931u
#define HC 0x83DCD12Du
#define AF 5.286931991577148e-05f   // 1774 * 2^-25 exactly

__host__ __device__ constexpr int popc16(int m) { int c = 0; for (int i = 0; i < 16; ++i) c += (m >> i) & 1; return c; }
// start offset of step j for pattern (KA, MASK) with period 16
__host__ __device__ constexpr int step_off(int j, int KA, int MASK)
{
    int s = (j >> 4) * (16 * KA + popc16(MASK));
    for (int i = 0; i < (j & 15); ++i) s += KA + ((MASK >> i) & 1);
    return s;
}

__host__ __device__ constexpr bool direct_win(int o) { return (o & 31) == 0 || (o & 31) == 8 || (o & 31) == 16; }
// WOPT 2: greedy funnel grouping per residue class (mod 8) over the plane's actual window offsets; returns funnel start for O
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
// 16-bit window at compile-time bit offset O of the lane stream (w has ext words)
template <int O, int WOPT, int KA = 0, int MASK = 0, int NS = 0>
__device__ __forceinline__ uint32_t wv(const uint32_t* w)
{
    constexpr int i = O >> 5, s = O & 31;
    if constexpr (s == 0) return w[i] & 0xFFFFu;
    else if constexpr (s == 16) return w[i] >> 16;
    else if constexpr (s == 8) return __byte_perm(w[i], 0u, 0x4421);
    else if constexpr (WOPT == 0)
    {
        if constexpr (s < 16) return (w[i] >> s) & 0xFFFFu;
        else return __funnelshift_r(w[i], w[i + 1], s) & 0xFFFFu;
    }
    else
    {
        // shared funnel at F = O - 8t, t = (O % 24) / 8  (F mod 24 < 8): one funnel serves O, O+8, O+16
        constexpr int F = WOPT == 2 ? funnel_greedy(O, KA, MASK, NS) : O - 8 * ((O % 24) / 8), t = (O - F) / 8, fi = F >> 5, fs = F & 31;
        uint32_t y;
        if constexpr (fs == 0) y = w[fi]; else y = __funnelshift_r(w[fi], w[fi + 1], fs);
        if constexpr (t == 0) return y & 0xFFFFu;
        else if constexpr (t == 1) return __byte_perm(y, 0u, 0x4421);
        else return y >> 16;
    }
}
__device__ __forceinline__ uint32_t dp4u(uint32_t a, uint32_t b, uint32_t c) { uint32_t d; asm("dp4a.u32.u32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c)); return d; }
__device__ __forceinline__ uint32_t dp4us(uint32_t a, uint32_t b, uint32_t c) { uint32_t d; asm("dp4a.u32.s32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c)); return d; }
__device__ __forceinline__ uint32_t hfma2u(uint32_t a, uint32_t s, uint32_t b)
{
    half2 r = __hfma2(*(half2*)&a, *(half2*)&s, *(half2*)&b);
    return *(uint32_t*)&r;
}

template <int BKA_, int BM_, int RMODE_, int RKA_, int RM_, int WOPT_, int RAW_ = 0>
struct Dec
{
    static constexpr int RAW = RAW_;
    static constexpr int BKA = BKA_, BM = BM_, RMODE = RMODE_, RKA = RKA_, RM = RM_, WOPT = WOPT_;
    static constexpr bool V2 = (RMODE == RM_A2 || RMODE == RM_F2 || RMODE == RM_P2);
    static constexpr bool PF = (RMODE == RM_P || RMODE == RM_P2);
    static constexpr bool HASR = (RMODE != RM_NONE && RMODE != RM_UNI4);
    static constexpr int BBITS = RMODE == RM_UNI4 ? 256 : 4 * (16 * BKA + popc16(BM));
    // residual: V1 -> 64 steps of pattern (RKA, RM); V2 -> 32 pair-steps
    static constexpr int RBITS = !HASR ? 0 : (V2 ? 2 : 4) * (16 * RKA + popc16(RM));
    static constexpr int NWB = (BBITS + 31) / 32, NWR = RBITS ? (RBITS + 31) / 32 : 1;
};

struct Plane { const uint8_t* p; int bits; };
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
    if constexpr (nh) { w[k++] = __ldg((const unsigned short*)p + rec); }
}

template <class D, int CPW>
struct Stage { uint32_t wb[CPW][D::NWB], wr[CPW][D::NWR], dl[CPW]; };

struct Args
{
    const void* x; const uint8_t* base; const uint8_t* res; const uint32_t* delta;
    float* acc; int B, N, K, NST; float* wdbg;
    const half* su_in;
    const half* sv_g; const half* sv_u; const half* su_d;
    const half* sv_o; half* out;
    int* cnt; int dbg;
    const float* acc_in; float* acc_zero; int nzero;
};

template <class D, int CPW>
__device__ __forceinline__ void load_stage(Stage<D, CPW>& S, const Args& a, size_t rec0, size_t blk0, size_t nrec)
{
    #pragma unroll
    for (int c = 0; c < CPW; ++c)
    {
        load_plane<D::BBITS>(S.wb[c], a.base, rec0 + c * 32, nrec);
        if constexpr (D::HASR) { load_plane<D::RBITS>(S.wr[c], a.res, rec0 + c * 32, nrec); S.dl[c] = __ldg(a.delta + blk0 + c); }
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
// build ext words for ring wrap: stream bits [BITS, BITS+32) = neighbour's first word
template <int BITS, int NW>
__device__ __forceinline__ void ext_words(uint32_t* w, uint32_t nb)
{
    if constexpr (BITS % 32 == 0) { w[NW] = nb; w[NW + 1] = 0; }
    else { w[NW - 1] = (w[NW - 1] & 0xFFFFu) | (nb << 16); w[NW] = nb >> 16; w[NW + 1] = 0; }
}

struct Consts { uint32_t Bp, dA, Nrep, Nsgn, c0, c1, Mrep, Ah, Ch; float Cf; };
__constant__ float c_rcp[256];

template <class D, int P>
__device__ __forceinline__ uint32_t dec_pair(const uint32_t* w, const uint32_t* r, const Consts& k)
{
    constexpr int WO = D::WOPT;
    if constexpr (D::RMODE == RM_UNI4)
    {
        uint32_t x = w[P >> 2]; constexpr int s = (P & 3) * 4;
        uint32_t q = ((x >> s) & 0x000F000Fu) | 0x64006400u;
        return hfma2u(q, 0x2c002c00u, 0xd408d408u);   // (q - 1032) / 16
    }
    else
    {
        constexpr int ob0 = step_off(2 * P, D::BKA, D::BM), ob1 = step_off(2 * P + 1, D::BKA, D::BM);
        const uint32_t xb0 = wv<ob0, WO, D::BKA, D::BM, 64>(w) * HC, xb1 = wv<ob1, WO, D::BKA, D::BM, 64>(w) * HC;
        if constexpr (D::PF)
        {
            // integer fold with fp16 byte-select: t = 0x640000 + 128 + Mb*Sb + N*Sr; fp16 bits = t[8:24) = 0x6400 + round((Mb Sb + N Sr)/256)
            uint32_t t0 = dp4u(xb0, k.Mrep, k.c0), t1 = dp4u(xb1, k.Mrep, k.c1);
            if constexpr (D::RMODE == RM_P)
            {
                constexpr int o0 = step_off(2 * P, D::RKA, D::RM), o1 = step_off(2 * P + 1, D::RKA, D::RM);
                t0 = dp4u(wv<o0, WO, D::RKA, D::RM, (D::V2 ? 32 : 64)>(r) * HC, k.Nrep, t0); t1 = dp4u(wv<o1, WO, D::RKA, D::RM, (D::V2 ? 32 : 64)>(r) * HC, k.Nrep, t1);
            }
            else
            {
                constexpr int o = step_off(P, D::RKA, D::RM);
                const uint32_t xr = wv<o, WO, D::RKA, D::RM, (D::V2 ? 32 : 64)>(r) * HC;
                t0 = dp4u(xr, k.Nrep, t0); t1 = dp4us(xr, k.Nsgn, t1);
            }
            if constexpr (D::RAW) return __byte_perm(t0, t1, 0x6521);
            else return hfma2u(__byte_perm(t0, t1, 0x6521), k.Ah, k.Ch);
        }
        else if constexpr (D::RMODE == RM_F || D::RMODE == RM_F2)
        {
            uint32_t t0 = dp4u(xb0, 0x80808080u, 0x4B000000u), t1 = dp4u(xb1, 0x80808080u, k.c1);
            if constexpr (D::RMODE == RM_F)
            {
                constexpr int o0 = step_off(2 * P, D::RKA, D::RM), o1 = step_off(2 * P + 1, D::RKA, D::RM);
                t0 = dp4u(wv<o0, WO, D::RKA, D::RM, (D::V2 ? 32 : 64)>(r) * HC, k.Nrep, t0); t1 = dp4u(wv<o1, WO, D::RKA, D::RM, (D::V2 ? 32 : 64)>(r) * HC, k.Nrep, t1);
            }
            else
            {
                constexpr int o = step_off(P, D::RKA, D::RM);
                const uint32_t xr = wv<o, WO, D::RKA, D::RM, (D::V2 ? 32 : 64)>(r) * HC;
                t0 = dp4u(xr, k.Nrep, t0); t1 = dp4us(xr, k.Nsgn, t1);
            }
            const float f0 = fmaf(__uint_as_float(t0), AF, k.Cf), f1 = fmaf(__uint_as_float(t1), AF, k.Cf);
            half2 h = __floats2half2_rn(f0, f1);
            return *(uint32_t*)&h;
        }
        else
        {
            const uint32_t hb = __byte_perm(dp4u(xb0, 0x01010101u, 0x6400u), dp4u(xb1, 0x01010101u, 0x6400u), 0x5410);
            if constexpr (D::RMODE == RM_NONE) { if constexpr (D::RAW) return hb; else return hfma2u(hb, MUL1_A, MUL1_B); }
            else
            {
                const uint32_t base = hfma2u(hb, MUL1_A, k.Bp);
                uint32_t hr;
                if constexpr (D::RMODE == RM_A)
                {
                    constexpr int o0 = step_off(2 * P, D::RKA, D::RM), o1 = step_off(2 * P + 1, D::RKA, D::RM);
                    hr = __byte_perm(dp4u(wv<o0, WO, D::RKA, D::RM, (D::V2 ? 32 : 64)>(r) * HC, 0x01010101u, 0x6400u), dp4u(wv<o1, WO, D::RKA, D::RM, (D::V2 ? 32 : 64)>(r) * HC, 0x01010101u, 0x6400u), 0x5410);
                }
                else
                {
                    constexpr int o = step_off(P, D::RKA, D::RM);
                    const uint32_t xr = wv<o, WO, D::RKA, D::RM, (D::V2 ? 32 : 64)>(r) * HC;
                    hr = __byte_perm(dp4u(xr, 0x01010101u, 0x6400u), dp4us(xr, 0xFF01FF01u, 0x6400u + 510u), 0x5410);
                }
                return hfma2u(hr, k.dA, base);
            }
        }
    }
}

template <class D, int T>
__device__ __forceinline__ void ktile(const uint32_t* w, const uint32_t* r, const Consts& k, uint32_t* a)
{
    a[0] = dec_pair<D, 4 * T + 0>(w, r, k); a[1] = dec_pair<D, 4 * T + 1>(w, r, k);
    a[2] = dec_pair<D, 4 * T + 2>(w, r, k); a[3] = dec_pair<D, 4 * T + 3>(w, r, k);
}

template <class D, int G>
__device__ __forceinline__ void chunk_mma(const uint32_t* w, const uint32_t* r, uint32_t dl, float* acc, const half* xs, const float* xsm,
                                          int kslice, int kc, int lane, int B, float* wdbg, int strip, int kabs, int K)
{
    const int g = lane >> 2, t4 = lane & 3;
    Consts k{};
    if constexpr (D::RMODE == RM_A || D::RMODE == RM_A2)
    {
        uint32_t aa = MUL1_A, bb = MUL1_B;
        half2 d = *(half2*)&dl, A = *(half2*)&aa, Bc = *(half2*)&bb;
        half2 t = __hmul2(d, A); k.dA = *(uint32_t*)&t;
        half2 u = __hfma2(d, Bc, Bc); k.Bp = *(uint32_t*)&u;
    }
    if constexpr (D::RMODE == RM_F || D::RMODE == RM_F2)
    {
        // f4 = AF * (2^23 + 128 sb + N sr) + Cf,  AF = A/128 = 1774*2^-25,  Cf = (1 + N/128)(1024A + B) - AF*2^23,
        // 1024A + B = -3.453125 (fp16 A = 0x1eee, B = 0xc931).  delta = N/128.
        const uint32_t N = dl & 0xFF;
        k.Nrep = N * 0x01010101u;
        k.Nsgn = N * 0x00010001u + ((256u - N) & 0xFF) * 0x01000100u;
        k.c1 = 0x4B000000u + (D::V2 ? 510u * N : 0u);
        k.Cf = fmaf((float)N, -3.453125f / 128.f, -3.453125f - 443.5f);
    }
    if constexpr (D::PF)
    {
        // block word: Mb | N << 8.  delta = N / Mb;  w = A'(1024 + F) + C,  A' = fp16(256 A / Mb),  C = fp16((1 + delta) K0 - 1024 A')
        const uint32_t Mb = dl & 0xFF, N = (dl >> 8) & 0xFF;
        k.Mrep = Mb * 0x01010101u; k.Nrep = N * 0x01010101u;
        k.Nsgn = N * 0x00010001u + ((256u - N) & 0xFF) * 0x01000100u;
        k.c0 = 0x640080u; k.c1 = 0x640080u + (D::V2 ? 510u * N : 0u);
        const float rc = c_rcp[Mb];
        const half Ah = __float2half_rn(1.732421875f * rc);   // 256 A / Mb, A = 1774 * 2^-18
        const float C = fmaf((float)N * rc, -3.453125f, -3.453125f) - 1024.f * __half2float(Ah);
        half2 a2 = __half2half2(Ah), c2 = __half2half2(__float2half_rn(C));
        k.Ah = *(uint32_t*)&a2; k.Ch = *(uint32_t*)&c2;
    }
    // RAW: mma on the fp16 codes h (= 1024 + F), per-chunk affine applied to the fp32 chunk accumulator: acc += fA * sum(h x) + fC * sum(x)
    float fA = 0.f, fC = 0.f, accc[4] = {0.f, 0.f, 0.f, 0.f};
    if constexpr (D::RAW)
    {
        if constexpr (D::PF)
        {
            const uint32_t Mb = dl & 0xFF, N = (dl >> 8) & 0xFF; const float rc = c_rcp[Mb];
            fA = 1.732421875f * rc; fC = fmaf((float)N * rc, -3.453125f, -3.453125f) - 1024.f * fA;
        }
        else { fA = 0.0067672729492f; fC = -10.3828125f; }
    }
    float* const accm = D::RAW ? accc : acc;
    #pragma unroll
    for (int t = 0; t < 8; ++t)
    {
        uint32_t a[4];
        switch (t)
        {
            case 0: ktile<D, 0>(w, r, k, a); break; case 1: ktile<D, 1>(w, r, k, a); break;
            case 2: ktile<D, 2>(w, r, k, a); break; case 3: ktile<D, 3>(w, r, k, a); break;
            case 4: ktile<D, 4>(w, r, k, a); break; case 5: ktile<D, 5>(w, r, k, a); break;
            case 6: ktile<D, 6>(w, r, k, a); break; default: ktile<D, 7>(w, r, k, a); break;
        }
#ifndef NO_WDBG
        if (wdbg)
        {
            #pragma unroll
            for (int q = 0; q < 4; ++q)
            {
                int row = strip * 16 + g + (q & 1) * 8, kk = kabs + kc + t * 16 + t4 * 2 + (q >> 1) * 8;
                half2 h = *(half2*)&a[q];
                float v0 = __low2float(h), v1 = __high2float(h);
                if constexpr (D::RAW) { v0 = fmaf(fA, v0, fC); v1 = fmaf(fA, v1, fC); }
                wdbg[(size_t)row * K + kk] = v0; wdbg[(size_t)row * K + kk + 1] = v1;
            }
        }
#endif
        uint32_t b0 = 0, b1 = 0;
        if (g < B) { const half* xr = xs + g * kslice + kc + t * 16 + t4 * 2; b0 = *(const uint32_t*)xr; b1 = *(const uint32_t*)(xr + 8); }
        mma16816(accm, a, b0, b1);
    }
    if constexpr (D::RAW)
    {
        const float2 s = *(const float2*)(xsm + (kc >> 7) * 8 + t4 * 2);
        acc[0] = fmaf(fA, accc[0], fmaf(fC, s.x, acc[0])); acc[1] = fmaf(fA, accc[1], fmaf(fC, s.y, acc[1]));
        acc[2] = fmaf(fA, accc[2], fmaf(fC, s.x, acc[2])); acc[3] = fmaf(fA, accc[3], fmaf(fC, s.y, acc[3]));
    }
}

template <class D, int G, int CPW>
__device__ __forceinline__ void compute_stage(const Stage<D, CPW>& S, float* acc, const half* xs, const float* xsm, int kslice, int kc0,
                                              int lane, int B, float* wdbg, int strip, int kabs, int K)
{
    const int src = ring_src<G>(lane);
    #pragma unroll
    for (int c = 0; c < CPW; ++c)
    {
        uint32_t w[D::NWB + 2], r[D::NWR + 2];
        #pragma unroll
        for (int i = 0; i < D::NWB; ++i) w[i] = S.wb[c][i];
        if constexpr (D::RMODE != RM_UNI4) ext_words<D::BBITS, D::NWB>(w, __shfl_sync(0xffffffffu, w[0], src));
        if constexpr (D::HASR)
        {
            #pragma unroll
            for (int i = 0; i < D::NWR; ++i) r[i] = S.wr[c][i];
            ext_words<D::RBITS, D::NWR>(r, __shfl_sync(0xffffffffu, r[0], src));
        }
        chunk_mma<D, G>(w, r, S.dl[c], acc, xs, xsm, kslice, kc0 + c * 128, lane, B, wdbg, strip, kabs, K);
    }
}

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

// MODE 0: x fp16, atomics into acc (unfused).  MODE 3: x fp32 + input sign/WHT prologue (fused-B gate/up).
// MODE 4: SwiGLU prologue from acc_in, zero acc_zero, output sign/WHT epilogue via arrival counter (fused-B down).
template <class D, int G, int CPW, int MODE>
__global__ void __launch_bounds__(256) nq15_gemv(Args a)
{
    extern __shared__ uint32_t smem[];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, SB = blockDim.x >> 5;
    const int strip = blockIdx.x * SB + warp;
    const int nchunks = a.K / 128, kslice = CPW * a.NST * 128, k0 = blockIdx.y * kslice, B = a.B;
    half* xs = (half*)smem;
    const int chunk0 = blockIdx.y * CPW * a.NST;
    const size_t nrec = (size_t)(a.N / 16) * nchunks * 32;
    const size_t rec0 = ((size_t)strip * nchunks + chunk0) * 32 + lane;
    const size_t blk0 = (size_t)strip * nchunks + chunk0;
    Stage<D, CPW> SA, SB_;
    load_stage<D, CPW>(SA, a, rec0, blk0, nrec);
    if (a.NST > 1) load_stage<D, CPW>(SB_, a, rec0 + CPW * 32, blk0 + CPW, nrec);

    if constexpr (MODE == 3)
    {
        const float* x = (const float*)a.x;
        const int ng = kslice / 128;
        for (int task = warp; task < B * ng; task += SB)
        {
            int bb = task / ng, gg = task % ng, k = k0 + gg * 128 + lane * 4;
            float4 f = *(const float4*)(x + (size_t)bb * a.K + k); float s[4]; ld4h(a.su_in + k, s);
            float v[4] = {f.x * s[0], f.y * s[1], f.z * s[2], f.w * s[3]};
            wht128_warp(v, lane);
            half2 h0 = __floats2half2_rn(v[0], v[1]), h1 = __floats2half2_rn(v[2], v[3]);
            uint2 u; u.x = *(uint32_t*)&h0; u.y = *(uint32_t*)&h1;
            *(uint2*)(xs + bb * kslice + gg * 128 + lane * 4) = u;
        }
    }
    else if constexpr (MODE == 4)
    {
        const int ng = kslice / 128, I = a.K;
        for (int task = warp; task < B * ng; task += SB)
        {
            int bb = task / ng, col = k0 + (task % ng) * 128 + lane * 4;
            float4 fg = __ldcg((const float4*)(a.acc_in + (size_t)bb * 2 * I + col));
            float4 fu = __ldcg((const float4*)(a.acc_in + (size_t)bb * 2 * I + I + col));
            float vg[4] = {fg.x, fg.y, fg.z, fg.w}, vu[4] = {fu.x, fu.y, fu.z, fu.w}, s1[4], s2[4], s3[4];
            wht128_warp(vg, lane); wht128_warp(vu, lane);
            ld4h(a.sv_g + col, s1); ld4h(a.sv_u + col, s2); ld4h(a.su_d + col, s3);
            #pragma unroll
            for (int i = 0; i < 4; ++i) { float gg = vg[i] * s1[i]; vg[i] = gg / (1.f + __expf(-gg)) * vu[i] * s2[i] * s3[i]; }
            wht128_warp(vg, lane);
            half2 h0 = __floats2half2_rn(vg[0], vg[1]), h1 = __floats2half2_rn(vg[2], vg[3]);
            uint2 u; u.x = *(uint32_t*)&h0; u.y = *(uint32_t*)&h1;
            *(uint2*)(xs + bb * kslice + (task % ng) * 128 + lane * 4) = u;
        }
        const int nb = gridDim.x * gridDim.y, bid = blockIdx.y * gridDim.x + blockIdx.x;
        for (int i = bid * blockDim.x + threadIdx.x; i < a.nzero / 4; i += nb * blockDim.x) ((float4*)a.acc_zero)[i] = make_float4(0, 0, 0, 0);
    }
    else
    {
        const half* x = (const half*)a.x;
        for (int i = threadIdx.x; i < B * kslice / 2; i += blockDim.x)
        {
            int bb = i / (kslice / 2), kk = i % (kslice / 2);
            ((half2*)xs)[i] = ((const half2*)(x + (size_t)bb * a.K + k0))[kk];
        }
    }
    __syncthreads();
    float* xsm = (float*)(smem + kslice * 2);   // after xs (4 * kslice halves): per local chunk, 8 column sums of x
    if constexpr (D::RAW)
    {
        const int nch = kslice / 128;
        for (int task = warp; task < nch * 8; task += SB)
        {
            const int c = task >> 3, bb = task & 7;
            float v = 0.f;
            if (bb < B) { float f[4]; ld4h(xs + bb * kslice + c * 128 + lane * 4, f); v = (f[0] + f[1]) + (f[2] + f[3]); }
            #pragma unroll
            for (int m = 16; m; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m);
            if (lane == 0) xsm[task] = v;
        }
        __syncthreads();
    }

    float acc[4] = {0, 0, 0, 0};
    for (int s = 0; s < a.NST; s += 2)
    {
        compute_stage<D, G, CPW>(SA, acc, xs, xsm, kslice, s * CPW * 128, lane, B, a.wdbg, strip, k0, a.K);
        if (s + 2 < a.NST) load_stage<D, CPW>(SA, a, rec0 + (s + 2) * CPW * 32, blk0 + (s + 2) * CPW, nrec);
        if (s + 1 < a.NST)
        {
            compute_stage<D, G, CPW>(SB_, acc, xs, xsm, kslice, (s + 1) * CPW * 128, lane, B, a.wdbg, strip, k0, a.K);
            if (s + 3 < a.NST) load_stage<D, CPW>(SB_, a, rec0 + (s + 3) * CPW * 32, blk0 + (s + 3) * CPW, nrec);
        }
    }
    const int g = lane >> 2, t4 = lane & 3;
    #pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        int bb = t4 * 2 + (i & 1), row = strip * 16 + g + (i >> 1) * 8;
        if (bb < B) atomicAdd(a.acc + (size_t)bb * a.N + row, acc[i]);
    }
    if constexpr (MODE == 4)
    {
        __shared__ int last;
        const int rows0 = blockIdx.x * SB * 16;
        const int grp = rows0 / 128;
        const int expect = gridDim.y * (128 / (SB * 16));
        __syncthreads();
        if (threadIdx.x == 0)
        {
            int prev;
            asm volatile ("atom.add.acq_rel.gpu.global.s32 %0, [%1], 1;" : "=r"(prev) : "l"(a.cnt + grp) : "memory");
            last = (prev == expect - 1);
        }
        __syncthreads();
        if (!last) return;
        for (int bb = warp; bb < B; bb += SB)
        {
            float* p = a.acc + (size_t)bb * a.N + grp * 128 + lane * 4;
            float4 f = __ldcg((const float4*)p); __stcg((float4*)p, make_float4(0, 0, 0, 0));
            float s[4]; ld4h(a.sv_o + grp * 128 + lane * 4, s);
            float v[4] = {f.x * s[0], f.y * s[1], f.z * s[2], f.w * s[3]};
            wht128_warp(v, lane);
            half2 h0 = __floats2half2_rn(v[0], v[1]), h1 = __floats2half2_rn(v[2], v[3]);
            uint2 u; u.x = *(uint32_t*)&h0; u.y = *(uint32_t*)&h1;
            *(uint2*)(a.out + (size_t)bb * a.N + grp * 128 + lane * 4) = u;
        }
        if (threadIdx.x == 0) a.cnt[grp] = 0;
    }
}

// ------------- unfused chain helpers (copied from thread 04 nqk.cu)
__device__ void had128_blk(float* v)
{
    int t = threadIdx.x;
    for (int h = 1; h < 128; h <<= 1)
    {
        __syncthreads();
        float a = v[t]; float b = v[t ^ h];
        __syncthreads();
        v[t] = (t & h) ? (b - a) : (a + b);
    }
    __syncthreads();
}
__global__ void k_swiglu(float* acc, half* h, const half* sv_g, const half* sv_u, const half* su_d, float* yzero, int B, int I, int Nout)
{
    __shared__ float vg[128], vu[128];
    int bb = blockIdx.y, grp = blockIdx.x, t = threadIdx.x;
    int col = grp * 128 + t;
    vg[t] = acc[(size_t)bb * 2 * I + col]; vu[t] = acc[(size_t)bb * 2 * I + I + col];
    acc[(size_t)bb * 2 * I + col] = 0.f; acc[(size_t)bb * 2 * I + I + col] = 0.f;
    had128_blk(vg); had128_blk(vu);
    float gg = vg[t] * __half2float(sv_g[col]) * 0.08838834764f, uu = vu[t] * __half2float(sv_u[col]) * 0.08838834764f;
    float s = gg / (1.f + __expf(-gg)) * uu;
    __syncthreads();
    vg[t] = s * __half2float(su_d[col]);
    had128_blk(vg);
    h[(size_t)bb * I + col] = __float2half(vg[t] * 0.08838834764f);
    int nb = gridDim.x * gridDim.y, bid = bb * gridDim.x + grp;
    for (int i = bid * 128 + t; i < B * Nout; i += nb * 128) yzero[i] = 0.f;
}
__global__ void k_had_in(const float* xin, half* xout, const half* su, int K)
{
    __shared__ float v[128];
    int bb = blockIdx.y, grp = blockIdx.x, t = threadIdx.x, col = grp * 128 + t;
    v[t] = xin[(size_t)bb * K + col] * __half2float(su[col]);
    had128_blk(v);
    xout[(size_t)bb * K + col] = __float2half(v[t] * 0.08838834764f);
}

// ------------- variant registry
typedef void (*kfn)(Args);
// id: (BKA, BM, RMODE, RKA, RM, WOPT)
#define VARIANTS(X) \
    X(0,  2, 0x0000, RM_NONE, 0, 0x0000, 0)   /* B2 (thread 04)            */ \
    X(1,  2, 0x0000, RM_NONE, 0, 0x0000, 1)   /* B2 shared funnels         */ \
    X(2,  4, 0x0000, RM_NONE, 0, 0x0000, 0)   /* T4 native, not progressive*/ \
    X(3,  2, 0x0000, RM_A,    2, 0x0000, 0)   /* A4 (thread 04)            */ \
    X(4,  2, 0x0000, RM_A,    2, 0x0000, 1)   /* A4 + funnels              */ \
    X(5,  2, 0x0000, RM_F,    2, 0x0000, 0)   /* fold                      */ \
    X(6,  2, 0x0000, RM_F,    2, 0x0000, 1)   /* fold + funnels            */ \
    X(7,  2, 0x0000, RM_A2,   4, 0x0000, 1)   /* 2D residual + funnels     */ \
    X(8,  2, 0x0000, RM_F2,   4, 0x0000, 1)   /* 2D residual fold + funnels*/ \
    X(9,  0, 0x0000, RM_UNI4, 0, 0x0000, 0)   /* UNI4 reference            */ \
    X(10, 2, 0x0000, RM_F,    1, 0xEEEE, 1)   /* fold, residual K=1.75     */ \
    X(11, 2, 0x0000, RM_F,    2, 0xAAAA, 1)   /* fold, residual K=2.5      */ \
    X(12, 2, 0x0000, RM_A,    1, 0xEEEE, 1)   /* A, residual K=1.75        */ \
    X(13, 2, 0x0000, RM_A,    2, 0xAAAA, 1)   /* A, residual K=2.5         */ \
    X(14, 2, 0x0000, RM_F,    1, 0xAAAA, 1)   /* fold, residual K=1.5      */ \
    X(15, 2, 0x0000, RM_F,    3, 0x0000, 1)   /* fold, residual K=3        */ \
    X(16, 2, 0x8888, RM_NONE, 0, 0x0000, 1)   /* base K=2.25               */ \
    X(17, 2, 0xAAAA, RM_NONE, 0, 0x0000, 1)   /* base K=2.5                */ \
    X(18, 2, 0x0000, RM_F2,   5, 0xAAAA, 1)   /* 2D fold, residual K=2.75 (5.5 b/pair) */ \
    X(19, 2, 0x0000, RM_F2,   3, 0xAAAA, 1)   /* 2D fold, residual K=1.75 (3.5 b/pair) */ \
    X(20, 2, 0x0000, RM_F2,   5, 0x0000, 1)   /* 2D fold, residual K=2.5 (5 b/pair)    */ \
    X(21, 2, 0x0000, RM_P,    2, 0x0000, 1)   /* int fold + PRMT select, K=2              */ \
    X(22, 2, 0x0000, RM_P2,   4, 0x0000, 1)   /* 2D int fold, K=2                         */ \
    X(23, 2, 0x0000, RM_P,    1, 0xEEEE, 1)   /* int fold, residual K=1.75               */ \
    X(24, 2, 0x0000, RM_P,    2, 0xAAAA, 1)   /* int fold, residual K=2.5                */ \
    X(25, 2, 0x0000, RM_P2,   5, 0x0000, 1)   /* 2D int fold, residual K=2.5             */ \
    X(26, 2, 0x0000, RM_P2,   3, 0xAAAA, 1)   /* 2D int fold, residual K=1.75            */ \
    X(27, 2, 0x0000, RM_P,    2, 0x0000, 0)   /* int fold, thread-04 windows             */ \
    X(28, 2, 0xAAAA, RM_P,    2, 0xAAAA, 1)   /* base 2.5 + residual 2.5 int fold        */ \
    X(29, 2, 0x0000, RM_P,    1, 0xEEEE, 2)   /* int fold, residual K=1.75, greedy funnels */ \
    X(30, 2, 0x0000, RM_P,    2, 0xAAAA, 2)   /* int fold, residual K=2.5, greedy        */ \
    X(31, 2, 0x0000, RM_P2,   5, 0x0000, 2)   /* 2D int fold, residual K=2.5, greedy     */ \
    X(32, 2, 0x0000, RM_P2,   3, 0xAAAA, 2)   /* 2D int fold, residual K=1.75, greedy    */ \
    X(33, 2, 0x0000, RM_P,    2, 0x0000, 2)   /* int fold K=2 greedy                     */ \
    X(34, 2, 0x0000, RM_NONE, 0, 0x0000, 2)   /* B2 greedy                               */ \
    X(35, 2, 0xAAAA, RM_NONE, 0, 0x0000, 2)   /* base K=2.5 greedy                       */ \
    X(36, 2, 0x8888, RM_NONE, 0, 0x0000, 2)   /* base K=2.25 greedy                      */ \
    X(37, 2, 0x0000, RM_P2,   4, 0x0000, 2)   /* 2D int fold K=2 greedy                  */ \
    X(38, 2, 0x0000, RM_P,    3, 0x0000, 2)   /* int fold residual K=3 greedy            */ \
    X(39, 2, 0x0000, RM_P,    1, 0xAAAA, 2)   /* int fold residual K=1.5 greedy          */ \
    X(40, 2, 0x0000, RM_NONE, 0, 0x0000, 2, 1)   /* RAW: B2                              */ \
    X(41, 2, 0x0000, RM_P,    2, 0x0000, 2, 1)   /* RAW: int fold K=2                    */ \
    X(42, 2, 0x0000, RM_P2,   4, 0x0000, 2, 1)   /* RAW: 2D int fold K=2                 */ \
    X(43, 2, 0x0000, RM_P,    1, 0xEEEE, 2, 1)   /* RAW: int fold residual 1.75          */ \
    X(44, 2, 0x0000, RM_P,    2, 0xAAAA, 2, 1)   /* RAW: int fold residual 2.5           */ \
    X(45, 2, 0x0000, RM_P2,   5, 0x0000, 2, 1)   /* RAW: 2D residual 2.5                 */ \
    X(46, 2, 0x0000, RM_P2,   3, 0xAAAA, 2, 1)   /* RAW: 2D residual 1.75                */ \
    X(47, 2, 0xAAAA, RM_NONE, 0, 0x0000, 2, 1)   /* RAW: base 2.5                        */ \
    X(48, 2, 0x8888, RM_NONE, 0, 0x0000, 2, 1)   /* RAW: base 2.25                       */ \
    X(49, 2, 0x0000, RM_P,    3, 0x0000, 2, 1)   /* RAW: int fold residual 3             */ \
    X(50, 2, 0x0000, RM_P,    1, 0xAAAA, 2, 1)   /* RAW: int fold residual 1.5           */ \
    X(51, 2, 0xAAAA, RM_P,    2, 0xAAAA, 2, 1)   /* RAW: base 2.5 + residual 2.5         */ \
    X(52, 2, 0xEEEE, RM_NONE, 0, 0x0000, 2, 1)   /* RAW: base 1.75 (1.75-bit base)       */

#define DEC_T(id, a, b, c, d, e, f, ...) using D##id = Dec<a, b, c, d, e, f, ##__VA_ARGS__>;
VARIANTS(DEC_T)

#ifndef NQ_G
#define NQ_G 2
#endif
static std::map<std::tuple<int, int, int>, kfn>& reg()
{
    static std::map<std::tuple<int, int, int>, kfn> m;
    if (m.empty())
    {
#define REG(id, a, b, c, d, e, f, ...) \
        m[{id, 1, 0}] = nq15_gemv<D##id, NQ_G, 1, 0>; m[{id, 2, 0}] = nq15_gemv<D##id, NQ_G, 2, 0>; \
        m[{id, 1, 3}] = nq15_gemv<D##id, NQ_G, 1, 3>; m[{id, 2, 3}] = nq15_gemv<D##id, NQ_G, 2, 3>; \
        m[{id, 1, 4}] = nq15_gemv<D##id, NQ_G, 1, 4>; m[{id, 2, 4}] = nq15_gemv<D##id, NQ_G, 2, 4>;
        VARIANTS(REG)
    }
    return m;
}
std::vector<int64_t> info(int64_t id)
{
    switch (id)
    {
#define INFO(i, a, b, c, d, e, f, ...) case i: return {D##i::BBITS, D##i::RBITS, D##i::HASR, D##i::RMODE, D##i::V2, a, b, d, e, f, D##i::RAW};
        VARIANTS(INFO)
    }
    return {};
}
template <typename T> const T* P(const torch::Tensor& t) { return t.numel() ? (const T*)t.data_ptr() : nullptr; }

void gemv(torch::Tensor x, torch::Tensor base, torch::Tensor res, torch::Tensor delta, torch::Tensor acc,
          int64_t id, int64_t cpw, int64_t sb, int64_t nst, int64_t N, int64_t K, int64_t mode,
          torch::Tensor wdbg, std::vector<torch::Tensor> ex)
{
    Args a{};
    a.x = x.data_ptr(); a.base = P<uint8_t>(base); a.res = P<uint8_t>(res); a.delta = P<uint32_t>(delta);
    a.acc = (float*)acc.data_ptr(); a.B = x.size(0); a.N = N; a.K = K; a.NST = nst;
    a.wdbg = wdbg.numel() ? (float*)wdbg.data_ptr() : nullptr;
    if (mode == 3) { a.su_in = P<half>(ex[0]); }
    if (mode == 4) { a.acc_in = (const float*)ex[0].data_ptr(); a.acc_zero = (float*)ex[1].data_ptr(); a.nzero = ex[1].numel();
                     a.sv_g = P<half>(ex[2]); a.sv_u = P<half>(ex[3]); a.su_d = P<half>(ex[4]);
                     a.sv_o = P<half>(ex[5]); a.out = (half*)ex[6].data_ptr(); a.cnt = (int*)ex[7].data_ptr(); }
    static bool rcp_done = false;
    if (!rcp_done) { float t[256]; t[0] = 0; for (int i = 1; i < 256; ++i) t[i] = 1.f / i; cudaMemcpyToSymbol(c_rcp, t, sizeof(t)); rcp_done = true; }
    auto it = reg().find({(int)id, (int)cpw, (int)mode});
    TORCH_CHECK(it != reg().end(), "no kernel");
    kfn f = it->second;
    int kslice = cpw * nst * 128;
    TORCH_CHECK(K % kslice == 0 && (N / 16) % sb == 0 && sb <= 8 && (mode != 4 || 128 % (sb * 16) == 0));
    int shm = 4 * kslice * 2 + (kslice / 128) * 8 * 4;
    static std::set<kfn> done;
    if (!done.count(f)) { cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, 100 * 1024); done.insert(f); }
    dim3 grid(N / 16 / sb, K / kslice);
    f<<<grid, 32 * sb, shm, at::cuda::getCurrentCUDAStream()>>>(a);
}
void swiglu(torch::Tensor acc, torch::Tensor h, torch::Tensor svg, torch::Tensor svu, torch::Tensor sud, torch::Tensor yzero)
{
    int B = h.size(0), I = h.size(1);
    k_swiglu<<<dim3(I / 128, B), 128, 0, at::cuda::getCurrentCUDAStream()>>>((float*)acc.data_ptr(), (half*)h.data_ptr(),
        (const half*)svg.data_ptr(), (const half*)svu.data_ptr(), (const half*)sud.data_ptr(), (float*)yzero.data_ptr(), B, I, yzero.size(1));
}
void had_in(torch::Tensor xin, torch::Tensor xout, torch::Tensor su)
{
    int B = xin.size(0), K = xin.size(1);
    k_had_in<<<dim3(K / 128, B), 128, 0, at::cuda::getCurrentCUDAStream()>>>((const float*)xin.data_ptr(), (half*)xout.data_ptr(), (const half*)su.data_ptr(), K);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("gemv", &gemv); m.def("swiglu", &swiglu); m.def("had_in", &had_in); m.def("info", &info); }
