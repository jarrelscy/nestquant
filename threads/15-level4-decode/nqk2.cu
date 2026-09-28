// NestQuant decode kernels v2: additive progressive trellis (f = f2(base) + delta * g(residual)),
// tail-biting rings shared across G lanes (ring = 64*G weights) via one shuffle per plane per chunk,
// optional fused prologue/epilogue (input Hadamard, SwiGLU+Hadamards, output Hadamard) via arrival counters.
//
// Tiling (identical at every level): strip = 16 rows, chunk = 128 k, lane owns 64 weights per (strip, chunk).
// Planes per (strip, chunk, lane): base uint4 (2b), P3 uint2 (1b residual trellis), P4 uint4 (2b residual trellis).
// Ring order inside a group of G lanes (consecutive t4 lanes): lane record gi occupies stream bits [gi*R, (gi+1)*R).
#include <cuda_fp16.h>
#include <stdint.h>
#include <set>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

enum { B2 = 0, T4, A3, A4, MIX, T2H, NDEC };
__device__ __forceinline__ uint32_t* lut_smem() { __shared__ uint32_t s_lut[512]; return s_lut; }

__device__ __forceinline__ uint32_t wn(const uint32_t* w, int o)   // 16-bit window at bit offset o (no wrap; w has 1 ext word)
{
    const int i = o >> 5, s = o & 31;
    if (s == 0) return w[i] & 0xFFFFu;
    if (s == 16) return w[i] >> 16;
    if (s == 8) return __byte_perm(w[i], 0u, 0x4421);
    if (s < 16) return (w[i] >> s) & 0xFFFFu;
    return __funnelshift_r(w[i], w[i + 1], s) & 0xFFFFu;
}
__device__ __forceinline__ uint32_t mul1_raw(uint32_t v0, uint32_t v1)   // two fp16 codes (1024 + bytesum) packed
{
    uint32_t x0 = v0 * 0x83DCD12Du, x1 = v1 * 0x83DCD12Du;
    uint32_t s0 = __dp4a(x0, 0x01010101u, 0x6400u), s1 = __dp4a(x1, 0x01010101u, 0x6400u);
    return __byte_perm(s0, s1, 0x5410);
}
__device__ __forceinline__ uint32_t hfma2u(uint32_t a, uint32_t s, uint32_t b)
{
    half2 r = __hfma2(*(half2*)&a, *(half2*)&s, *(half2*)&b);
    return *(uint32_t*)&r;
}
#define MUL1_A 0x1eee1eeeu
#define MUL1_B 0xc931c931u

template <int DEC> struct Cfg { static constexpr int NB = 4, N3 = 0, N4 = 0; };
template <> struct Cfg<T4>  { static constexpr int NB = 8, N3 = 0, N4 = 0; };
template <> struct Cfg<A3>  { static constexpr int NB = 4, N3 = 2, N4 = 0; };
template <> struct Cfg<A4>  { static constexpr int NB = 4, N3 = 0, N4 = 4; };
template <> struct Cfg<MIX> { static constexpr int NB = 4, N3 = 0, N4 = 4; };
template <> struct Cfg<T2H> { static constexpr int NB = 4, N3 = 0, N4 = 4; };

template <int DEC, int CPW>
struct Stage
{
    using C = Cfg<DEC>;
    uint32_t wb[CPW][C::NB], w3[CPW][C::N3 ? C::N3 : 1], w4[CPW][C::N4 ? C::N4 : 1], dl[CPW];
    uint32_t fl;
};

struct Args
{
    const void* x; const uint4* base; const uint2* p3; const uint4* p4; const uint32_t* delta; const uint32_t* flags; const uint32_t* lut;
    float* acc; int B, N, K, NST; float* wdbg;
    const half* su_in;                                          // MODE 1 input sign
    const half* sv_g; const half* sv_u; const half* su_d; half* hout;   // MODE 1 epilogue
    const half* sv_o; half* out;                                // MODE 2 epilogue
    int* cnt; int dbg;
    const float* acc_in; float* acc_zero; int nzero;   // MODE 4: SwiGLU prologue source, buffer to zero
};

template <int DEC, int CPW>
__device__ __forceinline__ void load_stage(Stage<DEC, CPW>& S, const Args& a, size_t rec0, size_t blk0, uint32_t fmask, int cbit)
{
    using C = Cfg<DEC>;
    #pragma unroll
    for (int c = 0; c < CPW; ++c)
    {
        size_t rec = rec0 + c * 32;
        if constexpr (C::NB == 8) { uint4 v = a.base[rec * 2], u = a.base[rec * 2 + 1];
            S.wb[c][0] = v.x; S.wb[c][1] = v.y; S.wb[c][2] = v.z; S.wb[c][3] = v.w; S.wb[c][4] = u.x; S.wb[c][5] = u.y; S.wb[c][6] = u.z; S.wb[c][7] = u.w; }
        else { uint4 v = a.base[rec]; S.wb[c][0] = v.x; S.wb[c][1] = v.y; S.wb[c][2] = v.z; S.wb[c][3] = v.w; }
        if constexpr (C::N3) { uint2 v = a.p3[rec]; S.w3[c][0] = v.x; S.w3[c][1] = v.y; }
        if constexpr (DEC == A3 || DEC == A4 || DEC == MIX || DEC == T2H)
        {
            bool on = DEC != MIX || ((fmask >> (cbit + c)) & 1);
            if (on)
            {
                if constexpr (C::N4) { uint4 v = a.p4[rec]; S.w4[c][0] = v.x; S.w4[c][1] = v.y; S.w4[c][2] = v.z; S.w4[c][3] = v.w; }
                S.dl[c] = __ldg(a.delta + blk0 + c);
            }
        }
    }
    if constexpr (DEC == MIX) S.fl = fmask >> cbit;
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

template <int DEC, bool RES>
__device__ __forceinline__ uint32_t dec_pair(const uint32_t* w, const uint32_t* w3, const uint32_t* w4, int p, uint32_t Bp, uint32_t dA)
{
    if constexpr (DEC == T4)
        return hfma2u(mul1_raw(wn(w, 8 * p), wn(w, 8 * p + 4)), MUL1_A, MUL1_B);
    else if constexpr (DEC == T2H)   // base mul1 + delta * HYB V=2 (shared-LUT Q=9) residual from 2b plane, 4 bits/pair
    {
        uint32_t base = hfma2u(mul1_raw(wn(w, 4 * p), wn(w, 4 * p + 2)), MUL1_A, MUL1_B);
        uint32_t v = wn(w4, 4 * p), h = v * v + v;
        return hfma2u(lut_smem()[(h >> 7) & 0x1FF], dA, base);
    }
    else
    {
        uint32_t hb = mul1_raw(wn(w, 4 * p), wn(w, 4 * p + 2));
        if constexpr (!RES) return hfma2u(hb, MUL1_A, MUL1_B);
        else
        {
            uint32_t base = hfma2u(hb, MUL1_A, Bp);          // A*hb + B*(1+delta)
            uint32_t hr = (DEC == A3) ? mul1_raw(wn(w3, 2 * p), wn(w3, 2 * p + 1)) : mul1_raw(wn(w4, 4 * p), wn(w4, 4 * p + 2));
            return hfma2u(hr, dA, base);                     // + delta*A*hr  (one HFMA2 per pair for the residual)
        }
    }
}

template <int DEC, int G, bool RES>
__device__ __forceinline__ void chunk_mma(uint32_t* w, uint32_t* w3, uint32_t* w4, uint32_t dl, float* acc, const half* xs,
                                          int kslice, int kc, int lane, int B, float* wdbg, int strip, int kabs, int K)
{
    const int g = lane >> 2, t4 = lane & 3;
    uint32_t Bp = MUL1_B, dA = 0;
    if constexpr (RES)
    {
        uint32_t aa = MUL1_A, bb = MUL1_B;
        half2 d = *(half2*)&dl, A = *(half2*)&aa, Bc = *(half2*)&bb;
        half2 t = __hmul2(d, A); dA = *(uint32_t*)&t;
        half2 u = __hfma2(d, Bc, Bc); Bp = *(uint32_t*)&u;
        if constexpr (DEC == T2H) { dA = dl; Bp = MUL1_B; }
    }
    #pragma unroll
    for (int t = 0; t < 8; ++t)
    {
        uint32_t a[4];
        #pragma unroll
        for (int r = 0; r < 4; ++r) a[r] = dec_pair<DEC, RES>(w, w3, w4, t * 4 + r, Bp, dA);
        if (wdbg)
        {
            #pragma unroll
            for (int r = 0; r < 4; ++r)
            {
                int row = strip * 16 + g + (r & 1) * 8, k = kabs + kc + t * 16 + t4 * 2 + (r >> 1) * 8;
                half2 h = *(half2*)&a[r];
                wdbg[(size_t)row * K + k] = __low2float(h); wdbg[(size_t)row * K + k + 1] = __high2float(h);
            }
        }
        uint32_t b0 = 0, b1 = 0;
        if (g < B) { const half* xr = xs + g * kslice + kc + t * 16 + t4 * 2; b0 = *(const uint32_t*)xr; b1 = *(const uint32_t*)(xr + 8); }
        mma16816(acc, a, b0, b1);
    }
}

template <int DEC, int G, int CPW>
__device__ __forceinline__ void compute_stage(const Stage<DEC, CPW>& S, float* acc, const half* xs, int kslice, int kc0,
                                              int lane, int B, float* wdbg, int strip, int kabs, int K)
{
    using C = Cfg<DEC>;
    const int src = ring_src<G>(lane);
    #pragma unroll
    for (int c = 0; c < CPW; ++c)
    {
        uint32_t w[C::NB + 1], w3[C::N3 + 1], w4[C::N4 + 1];
        #pragma unroll
        for (int i = 0; i < C::NB; ++i) w[i] = S.wb[c][i];
        w[C::NB] = G == 1 ? w[0] : __shfl_sync(0xffffffffu, w[0], src);
        if constexpr (C::N3) { w3[0] = S.w3[c][0]; w3[1] = S.w3[c][1]; w3[2] = G == 1 ? w3[0] : __shfl_sync(0xffffffffu, w3[0], src); }
        const int kc = kc0 + c * 128;
        if constexpr (DEC == B2 || DEC == T4) chunk_mma<DEC, G, false>(w, w3, w4, 0, acc, xs, kslice, kc, lane, B, wdbg, strip, kabs, K);
        else if constexpr (DEC == A3) chunk_mma<DEC, G, true>(w, w3, w4, S.dl[c], acc, xs, kslice, kc, lane, B, wdbg, strip, kabs, K);
        else
        {
            const bool on = DEC != MIX || ((S.fl >> c) & 1);
            if (on)
            {
                #pragma unroll
                for (int i = 0; i < 4; ++i) w4[i] = S.w4[c][i];
                w4[4] = G == 1 ? w4[0] : __shfl_sync(0xffffffffu, w4[0], src);
                chunk_mma<DEC == T2H ? T2H : A4, G, true>(w, w3, w4, S.dl[c], acc, xs, kslice, kc, lane, B, wdbg, strip, kabs, K);
            }
            else chunk_mma<B2, G, false>(w, w3, w4, 0, acc, xs, kslice, kc, lane, B, wdbg, strip, kabs, K);
        }
    }
}

// Butterfly WHT-128 held by one warp: lane holds v[0..3] = elements lane*4 .. lane*4+3.
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

// MODE 0: x fp16, atomics into acc. MODE 1: x fp32 (+ input sign/WHT prologue), gate|up rows, SwiGLU epilogue -> hout.
// MODE 2: x fp16 (already transformed), output sign/WHT epilogue -> out. grid (N/16/SB, K/kslice), block 32*SB.
template <int DEC, int G, int CPW, int MODE>
__global__ void __launch_bounds__(256) nq2_gemv(Args a)
{
    extern __shared__ uint32_t smem[];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, SB = blockDim.x >> 5;
    const int strip = blockIdx.x * SB + warp;
    const int nchunks = a.K / 128, kslice = CPW * a.NST * 128, k0 = blockIdx.y * kslice, B = a.B;
    half* xs = (half*)smem;
    if constexpr (DEC == T2H) for (int i = threadIdx.x; i < 512; i += blockDim.x) lut_smem()[i] = __ldg(a.lut + i);
    const int chunk0 = blockIdx.y * CPW * a.NST;
    const size_t rec0 = ((size_t)strip * nchunks + chunk0) * 32 + lane;
    const size_t blk0 = (size_t)strip * nchunks + chunk0;
    uint32_t fm = 0;
    if constexpr (DEC == MIX) fm = __ldg(a.flags + (size_t)strip * ((nchunks + 31) / 32) + chunk0 / 32) >> (chunk0 & 31);
    Stage<DEC, CPW> SA, SB_;
    load_stage<DEC, CPW>(SA, a, rec0, blk0, fm, 0);
    if (a.NST > 1) load_stage<DEC, CPW>(SB_, a, rec0 + CPW * 32, blk0 + CPW, fm, CPW);

    if constexpr (MODE == 1 || MODE == 3)
    {
        const float* x = (const float*)a.x;
        const int ng = kslice / 128;
        for (int task = warp; task < B * ng; task += SB)
        {
            int bb = task / ng, gg = task % ng, k = k0 + gg * 128 + lane * 4;
            float4 f = *(const float4*)(x + (size_t)bb * a.K + k); float s[4]; ld4h(a.su_in + k, s);
            float v[4] = {f.x * s[0], f.y * s[1], f.z * s[2], f.w * s[3]};
            if (!(a.dbg & 2)) wht128_warp(v, lane);
            half2 h0 = __floats2half2_rn(v[0], v[1]), h1 = __floats2half2_rn(v[2], v[3]);
            uint2 u; u.x = *(uint32_t*)&h0; u.y = *(uint32_t*)&h1;
            *(uint2*)(xs + bb * kslice + gg * 128 + lane * 4) = u;
        }
    }
    else if constexpr (MODE == 4)
    {
        // SwiGLU prologue: h[b, k0 : k0+kslice] from complete gate/up accumulators (previous launch), then zero the other buffer
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

    float acc[4] = {0, 0, 0, 0};
    for (int s = 0; s < a.NST; s += 2)
    {
        compute_stage<DEC, G, CPW>(SA, acc, xs, kslice, s * CPW * 128, lane, B, a.wdbg, strip, k0, a.K);
        if (s + 2 < a.NST) load_stage<DEC, CPW>(SA, a, rec0 + (s + 2) * CPW * 32, blk0 + (s + 2) * CPW, fm, (s + 2) * CPW);
        if (s + 1 < a.NST)
        {
            compute_stage<DEC, G, CPW>(SB_, acc, xs, kslice, (s + 1) * CPW * 128, lane, B, a.wdbg, strip, k0, a.K);
            if (s + 3 < a.NST) load_stage<DEC, CPW>(SB_, a, rec0 + (s + 3) * CPW * 32, blk0 + (s + 3) * CPW, fm, (s + 3) * CPW);
        }
    }
    const int g = lane >> 2, t4 = lane & 3;
    #pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        int bb = t4 * 2 + (i & 1), row = strip * 16 + g + (i >> 1) * 8;
        if (bb < B) atomicAdd(a.acc + (size_t)bb * a.N + row, acc[i]);
    }
    if constexpr (MODE == 0 || MODE == 3) return;
    else
    {
        if (a.dbg & 1) return;
        // arrival counter per 128-row group (MODE 1: gate group j and up group j share counter j)
        __shared__ int last;
        const int rows0 = blockIdx.x * SB * 16;
        const int half_n = a.N / 2;
        constexpr int EM = MODE == 4 ? 2 : MODE;
        const int grp = EM == 1 ? (rows0 % half_n) / 128 : rows0 / 128;
        const int expect = (EM == 1 ? 2 : 1) * gridDim.y * (128 / (SB * 16));
        __syncthreads();
        if (threadIdx.x == 0)
        {
            int prev;   // CUTLASS-semaphore style: CTA barrier, then one gpu-scope acq_rel RMW (cumulative release)
            asm volatile ("atom.add.acq_rel.gpu.global.s32 %0, [%1], 1;" : "=r"(prev) : "l"(a.cnt + grp) : "memory");
            last = (prev == expect - 1);
        }
        __syncthreads();
        if (!last) return;
        if constexpr (EM == 2)
        {
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
        }
        else
        {
            float* sg = (float*)smem;   // [B][2][128], reuse x smem (all warps done with xs after __syncthreads above)
            for (int task = warp; task < 2 * B; task += SB)
            {
                int bb = task >> 1, up = task & 1, col = grp * 128 + lane * 4;
                float* p = a.acc + (size_t)bb * a.N + up * half_n + col;
                float4 f = __ldcg((const float4*)p); __stcg((float4*)p, make_float4(0, 0, 0, 0));
                float v[4] = {f.x, f.y, f.z, f.w}, s[4];
                wht128_warp(v, lane);
                ld4h((up ? a.sv_u : a.sv_g) + col, s);
                #pragma unroll
                for (int i = 0; i < 4; ++i) sg[(bb * 2 + up) * 128 + lane * 4 + i] = v[i] * s[i];
            }
            __syncthreads();
            for (int bb = warp; bb < B; bb += SB)
            {
                int col = grp * 128 + lane * 4; float s[4], v[4]; ld4h(a.su_d + col, s);
                #pragma unroll
                for (int i = 0; i < 4; ++i)
                {
                    float gg = sg[(bb * 2) * 128 + lane * 4 + i], uu = sg[(bb * 2 + 1) * 128 + lane * 4 + i];
                    v[i] = gg / (1.f + __expf(-gg)) * uu * s[i];
                }
                wht128_warp(v, lane);
                half2 h0 = __floats2half2_rn(v[0], v[1]), h1 = __floats2half2_rn(v[2], v[3]);
                uint2 u; u.x = *(uint32_t*)&h0; u.y = *(uint32_t*)&h1;
                *(uint2*)(a.hout + (size_t)bb * half_n + col) = u;
            }
        }
        if (threadIdx.x == 0) a.cnt[grp] = 0;
    }
}

typedef void (*kfn)(Args);
template <int DEC, int G, int MODE> kfn pk_cpw(int cpw)
{
    if (cpw == 1) return nq2_gemv<DEC, G, 1, MODE>;
    if (cpw == 2) return nq2_gemv<DEC, G, 2, MODE>;
    return nullptr;
}
template <int DEC, int MODE> kfn pk_g(int g, int cpw)
{
    if (g == 1) return pk_cpw<DEC, 1, MODE>(cpw);
    if (g == 2) return pk_cpw<DEC, 2, MODE>(cpw);
    if (g == 4) return pk_cpw<DEC, 4, MODE>(cpw);
    return nullptr;
}
template <int MODE> kfn pk_dec(int dec, int g, int cpw)
{
    switch (dec) { case B2: return pk_g<B2, MODE>(g, cpw); case T4: return pk_g<T4, MODE>(g, cpw); case A3: return pk_g<A3, MODE>(g, cpw);
                   case A4: return pk_g<A4, MODE>(g, cpw); case MIX: return pk_g<MIX, MODE>(g, cpw); case T2H: return pk_g<T2H, MODE>(g, cpw); }
    return nullptr;
}
template <typename T> const T* P(const torch::Tensor& t) { return t.numel() ? (const T*)t.data_ptr() : nullptr; }

// mode 0: x fp16 [B,K] -> acc fp32 [B,N] (+=).  mode 1: x fp32, extras = (su_in, sv_g, sv_u, su_d, hout, cnt).
// mode 2: x fp16, extras = (sv_o, out, cnt).
void gemv2(torch::Tensor x, torch::Tensor base, torch::Tensor p3, torch::Tensor p4, torch::Tensor delta, torch::Tensor flags, torch::Tensor lut,
           torch::Tensor acc, int64_t dec, int64_t g, int64_t cpw, int64_t sb, int64_t nst, int64_t N, int64_t K, int64_t mode,
           torch::Tensor wdbg, std::vector<torch::Tensor> ex, int64_t dbg)
{
    Args a{};
    a.x = x.data_ptr(); a.base = P<uint4>(base); a.p3 = P<uint2>(p3); a.p4 = P<uint4>(p4); a.delta = P<uint32_t>(delta);
    a.flags = P<uint32_t>(flags); a.lut = P<uint32_t>(lut); a.acc = (float*)acc.data_ptr(); a.B = x.size(0); a.N = N; a.K = K; a.NST = nst;
    a.wdbg = wdbg.numel() ? (float*)wdbg.data_ptr() : nullptr; a.dbg = dbg;
    if (mode == 1) { a.su_in = P<half>(ex[0]); a.sv_g = P<half>(ex[1]); a.sv_u = P<half>(ex[2]); a.su_d = P<half>(ex[3]);
                     a.hout = (half*)ex[4].data_ptr(); a.cnt = (int*)ex[5].data_ptr(); }
    if (mode == 2) { a.sv_o = P<half>(ex[0]); a.out = (half*)ex[1].data_ptr(); a.cnt = (int*)ex[2].data_ptr(); }
    if (mode == 3) { a.su_in = P<half>(ex[0]); }
    if (mode == 4) { a.acc_in = (const float*)ex[0].data_ptr(); a.acc_zero = (float*)ex[1].data_ptr(); a.nzero = ex[1].numel();
                     a.sv_g = P<half>(ex[2]); a.sv_u = P<half>(ex[3]); a.su_d = P<half>(ex[4]);
                     a.sv_o = P<half>(ex[5]); a.out = (half*)ex[6].data_ptr(); a.cnt = (int*)ex[7].data_ptr(); }
    kfn f = mode == 0 ? pk_dec<0>(dec, g, cpw) : mode == 1 ? pk_dec<1>(dec, g, cpw) : mode == 2 ? pk_dec<2>(dec, g, cpw)
          : mode == 3 ? pk_dec<3>(dec, g, cpw) : pk_dec<4>(dec, g, cpw);
    TORCH_CHECK(f, "no kernel");
    int kslice = cpw * nst * 128;
    TORCH_CHECK(K % kslice == 0 && (N / 16) % sb == 0 && sb <= 8 && (mode == 0 || 128 % (sb * 16) == 0));
    int shm = 4 * kslice * 2; if (mode == 1) shm = std::max(shm, (int)(a.B * 2 * 128 * 4));
    static std::set<kfn> done;
    if (!done.count(f)) { cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, 100 * 1024); done.insert(f); }
    dim3 grid(N / 16 / sb, K / kslice);
    f<<<grid, 32 * sb, shm, at::cuda::getCurrentCUDAStream()>>>(a);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("gemv2", &gemv2); }
