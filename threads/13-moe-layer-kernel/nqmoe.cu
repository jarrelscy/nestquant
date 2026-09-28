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
// Plane layout (per expert, per projection; N rows x K cols; strip = 16 rows, chunk = 128 cols; C = K/128):
//   base : uint4 per (strip, chunk, lane): 64 weights x 2 bit            -> ((strip*C + c)*32 + lane)
//   P4   : residual records, same layout as base (dense)                 -> ((strip*C + c)*32 + lane)
//   d4   : one fp16 delta per 16x128 block, stored as a duplicated half2 (uint32) -> strip*C + c
//   level 2 = base; level 4 = base + P4.  Optional mask mode (flags != null): uint64 per strip, bit c = chunk c is
//   refined; P4/d4 are then compact over the nm flagged chunks of each strip (-> (strip*nm + rank)*32 + lane).
//   Flags must be uniform over each 128-row group (8 strips) for thread-block-uniform branching.
// Device table: int64 [E][16]
//   [0] level (0, 2, 4)  [1..4] gate|up: base, p4, d4, flags(0 = dense)  [5..8] down: same
//   [9] signs: half[H su_in | I sv_g | I sv_u | I su_d | H sv_o]      [10..15] reserved
#include <cuda_fp16.h>
#include <stdint.h>
#include <set>
#include <map>
#include <tuple>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

// ============================== decoder (swappable; from thread 04 nqk2.cu, T2/A4 additive mul1) ==============
namespace nqdec {
__device__ __forceinline__ uint32_t wn(const uint32_t* w, int o)
{
    const int i = o >> 5, s = o & 31;
    if (s == 0) return w[i] & 0xFFFFu;
    if (s == 16) return w[i] >> 16;
    if (s == 8) return __byte_perm(w[i], 0u, 0x4421);
    if (s < 16) return (w[i] >> s) & 0xFFFFu;
    return __funnelshift_r(w[i], w[i + 1], s) & 0xFFFFu;
}
__device__ __forceinline__ uint32_t mul1_raw(uint32_t v0, uint32_t v1)
{
#ifdef NQ_FAKE_HASH
    return __byte_perm(v0, v1, 0x5410);   // diagnostic only: no hash
#endif
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

// Planes of one projection of one expert. p4/d4 are indexed densely (every chunk refined) when flags == nullptr,
// otherwise compactly over the flagged chunks of each strip (mask mode, nm flagged chunks per strip).
struct Planes { const uint4* base; const uint4* p4; const uint32_t* d4; const uint64_t* flags; };

struct Ctx { int strip, C, nm; uint64_t fl; };

// LV: 2 = base only, 4 = base + P4 on every chunk, 5 = base + P4 on flagged chunks (mask mode)
template <int LV, int CPW>
struct Stage { uint32_t wb[CPW][4]; uint32_t wr[CPW][LV >= 4 ? 4 : 1]; uint32_t dl[CPW]; uint32_t on; };

template <int LV, int CPW>
__device__ __forceinline__ void load_stage(Stage<LV, CPW>& S, const Planes& P, const Ctx& X, int ch0, int lane)
{
    S.on = 0;
    #pragma unroll
    for (int c = 0; c < CPW; ++c)
    {
        const int ch = ch0 + c;
        const size_t rec = (size_t)X.strip * X.C + ch;
#ifdef NQ_FAKE_LOAD
        uint4 v = make_uint4(rec * 0x9E3779B9u, rec ^ 0x1234567u, rec * 7u, lane * 13u + ch);
#else
        uint4 v = P.base[rec * 32 + lane];
#endif
        S.wb[c][0] = v.x; S.wb[c][1] = v.y; S.wb[c][2] = v.z; S.wb[c][3] = v.w;
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
                uint4 u = P.p4[ri * 32 + lane]; S.wr[c][0] = u.x; S.wr[c][1] = u.y; S.wr[c][2] = u.z; S.wr[c][3] = u.w;
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
template <bool RES>
__device__ __forceinline__ uint32_t dec_pair(const uint32_t* w, const uint32_t* wr, int p, uint32_t Bp, uint32_t dA)
{
    uint32_t hb = mul1_raw(wn(w, 4 * p), wn(w, 4 * p + 2));
    if constexpr (!RES) return hfma2u(hb, MUL1_A, MUL1_B);
    else
    {
        uint32_t base = hfma2u(hb, MUL1_A, Bp);
        uint32_t hr = mul1_raw(wn(wr, 4 * p), wn(wr, 4 * p + 2));
        return hfma2u(hr, dA, base);
    }
}
// one 16x128 chunk: decode into A fragments, 8 MMAs; xs holds ntok rows of kslice halves
template <bool RES>
__device__ __forceinline__ void chunk_mma(const uint32_t* w, const uint32_t* wr, uint32_t dl, float* acc, const half* xs,
                                          int kslice, int kc, int lane, int ntok)
{
    const int g = lane >> 2, t4 = lane & 3;
    uint32_t Bp = MUL1_B, dA = 0;
    if constexpr (RES)
    {
        uint32_t aa = MUL1_A, bb = MUL1_B;
        half2 d = *(half2*)&dl, A = *(half2*)&aa, Bc = *(half2*)&bb;
        half2 t = __hmul2(d, A); dA = *(uint32_t*)&t;
        half2 u = __hfma2(d, Bc, Bc); Bp = *(uint32_t*)&u;
    }
    #pragma unroll
    for (int t = 0; t < 8; ++t)
    {
        uint32_t a[4];
        #pragma unroll
        for (int r = 0; r < 4; ++r) a[r] = dec_pair<RES>(w, wr, t * 4 + r, Bp, dA);
        uint32_t b0 = 0, b1 = 0;
        if (g < ntok) { const half* xr = xs + g * kslice + kc + t * 16 + t4 * 2; b0 = *(const uint32_t*)xr; b1 = *(const uint32_t*)(xr + 8); }
#ifdef NQ_ACC2
        mma16816(acc + (t & 1) * 4, a, b0, b1);   // two independent accumulator chains
#else
        mma16816(acc, a, b0, b1);
#endif
    }
}
template <int LV, int G, int CPW>
__device__ __forceinline__ void compute_stage(const Stage<LV, CPW>& S, float* acc, const half* xs, int kslice, int kc0, int lane, int ntok)
{
    const int src = ring_src<G>(lane);
    #pragma unroll
    for (int c = 0; c < CPW; ++c)
    {
        uint32_t w[5], wr[5];
        #pragma unroll
        for (int i = 0; i < 4; ++i) w[i] = S.wb[c][i];
        w[4] = G == 1 ? w[0] : __shfl_sync(0xffffffffu, w[0], src);
        const int kc = kc0 + c * 128;
        if constexpr (LV == 2) chunk_mma<false>(w, wr, 0, acc, xs, kslice, kc, lane, ntok);
        else
        {
            if (LV == 4 || ((S.on >> c) & 1))
            {
                #pragma unroll
                for (int i = 0; i < 4; ++i) wr[i] = S.wr[c][i];
                wr[4] = G == 1 ? wr[0] : __shfl_sync(0xffffffffu, wr[0], src);
                chunk_mma<true>(w, wr, S.dl[c], acc, xs, kslice, kc, lane, ntok);
            }
            else chunk_mma<false>(w, wr, 0, acc, xs, kslice, kc, lane, ntok);
        }
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

#define TBL_W 16
struct MoeArgs
{
    const half* x; const int64_t* sel; const half* rw; int B, topk;
    const int64_t* table; int H, I; int nm_gu, nm_dn;
    float* acc_gu; half* h; float* acc_d; float* out;
    int* cnt_gu; int* cnt_d; int NST;
    int force_level;   // debug: >0 overrides table level
    int* hits;         // optional [E] int32 pick counters (may be host-mapped pinned memory); nullptr = off
};

struct RunInfo { int e, level, ntok, nruns; int slot[8]; float w[8]; const int64_t* ent; };

// one warp: dedup the routing, fill run z. Returns nruns in smem.
__device__ __forceinline__ void route(const MoeArgs& a, int z, RunInfo* R, int lane)
{
    const int S = a.B * a.topk;
    int64_t e = -1 - lane; bool ok = false;
    if (lane < S)
    {
        int64_t ee = a.sel[lane];
        float w = __half2float(a.rw[lane]);
        int lv = (int)__ldg(a.table + ee * TBL_W);
        if (w != 0.f && lv > 0) { e = ee; ok = true; }
    }
    const unsigned m = __match_any_sync(0xffffffffu, (unsigned long long)e);
    const bool leader = ok && lane == __ffs(m) - 1;
    const unsigned lm = __ballot_sync(0xffffffffu, leader);
    if (lane == 0) R->nruns = __popc(lm);
    const int rank = __popc(lm & ((1u << lane) - 1u));
    if (leader && (z < 0 || rank == z))
    {
        if (z < 0) R += rank;
        R->e = (int)e; R->ent = a.table + e * TBL_W;
        int lv = (int)R->ent[0]; if (a.force_level) lv = a.force_level; R->level = lv;
        int n = 0; unsigned mm = m;
        while (mm) { int s = __ffs(mm) - 1; mm &= mm - 1; R->slot[n] = s; R->w[n] = __half2float(a.rw[s]); ++n; }
        R->ntok = n;
    }
}

template <int LV, int G, int CPW>
__device__ __forceinline__ void gemv_body(const Planes& P, int strip, int C, int nm, int chunk0, int NST, float* acc,
                                          const half* xs, int kslice, int lane, int ntok)
{
    Ctx X; X.strip = strip; X.C = C; X.nm = nm; X.fl = 0;
    if constexpr (LV == 5) X.fl = __ldg((const unsigned long long*)P.flags + strip);
    Stage<LV, CPW> SA, SB_;
    load_stage<LV, CPW>(SA, P, X, chunk0, lane);
    if (NST > 1) load_stage<LV, CPW>(SB_, P, X, chunk0 + CPW, lane);
    for (int s = 0; s < NST; s += 2)
    {
        compute_stage<LV, G, CPW>(SA, acc, xs, kslice, s * CPW * 128, lane, ntok);
        if (s + 2 < NST) load_stage<LV, CPW>(SA, P, X, chunk0 + (s + 2) * CPW, lane);
        if (s + 1 < NST)
        {
            compute_stage<LV, G, CPW>(SB_, acc, xs, kslice, (s + 1) * CPW * 128, lane, ntok);
            if (s + 3 < NST) load_stage<LV, CPW>(SB_, P, X, chunk0 + (s + 3) * CPW, lane);
        }
    }
}

__device__ __forceinline__ Planes planes_of(const int64_t* ent, int off)
{
    Planes P;
    P.base = (const uint4*)ent[off]; P.p4 = (const uint4*)ent[off + 1];
    P.d4 = (const uint32_t*)ent[off + 2]; P.flags = (const uint64_t*)ent[off + 3];
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
        const half* su = signs;
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
#ifndef NQ_SKIP_GEMV
    if (lv == 2) gemv_body<2, G, CPW>(P, strip, C, nm, chunk0, a.NST, acc, xs, kslice, lane, ntok);
#ifndef NQ_NO_MASK
    else if (P.flags) gemv_body<5, G, CPW>(P, strip, C, nm, chunk0, a.NST, acc, xs, kslice, lane, ntok);
#endif
    else gemv_body<4, G, CPW>(P, strip, C, nm, chunk0, a.NST, acc, xs, kslice, lane, ntok);
#endif

#ifdef NQ_ACC2
    #pragma unroll
    for (int i = 0; i < 4; ++i) acc[i] += acc[4 + i];
#endif
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
        for (int task = warp; task < 2 * ntok; task += SB)
        {
            int j = task >> 1, up = task & 1, col = grp * 128 + lane * 4;
            float* p = a.acc_gu + (size_t)R.slot[j] * N + up * a.I + col;
            float4 f = __ldcg((const float4*)p); __stcg((float4*)p, make_float4(0, 0, 0, 0));
            float v[4] = {f.x, f.y, f.z, f.w}, s[4];
            wht128_warp(v, lane);
            ld4h((up ? sv_u : sv_g) + col, s);
            #pragma unroll
            for (int i = 0; i < 4; ++i) sg[(j * 2 + up) * 128 + lane * 4 + i] = v[i] * s[i];
        }
        __syncthreads();
        for (int j = warp; j < ntok; j += SB)
        {
            int col = grp * 128 + lane * 4; float s[4], v[4]; ld4h(su_d + col, s);
            #pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                float gg = sg[(j * 2) * 128 + lane * 4 + i], uu = sg[(j * 2 + 1) * 128 + lane * 4 + i];
                v[i] = gg / (1.f + __expf(-gg)) * uu * s[i];
            }
            wht128_warp(v, lane);
            st4h(a.h + (size_t)R.slot[j] * a.I + col, v);
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
                for (int i = 0; i < 4; ++i) o[i] += w * v[i] * sg[i];
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
    __shared__ RunInfo R[32];
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
    cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, 100 * 1024);
    int nb = 0, dev = 0, sms = 0; cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, f, threads, shm);
    cudaGetDevice(&dev); cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    return cache[k] = nb * sms;
}

// One MoE layer forward (graph-capturable; all routing/level/pointer decisions are read on device).
// cfg_gu/cfg_dn: (cpw, sb, nst[, persistent]).  ws: acc_gu [S,2I] f32, h [S,I] f16, acc_d [S,H] f32,
// cnt_gu [S*I/128] i32, cnt_d [H/128] i32, wq [4] i32 (persistent work counters) -- all zero on entry, zero on exit.
void moe_forward(torch::Tensor x, torch::Tensor sel, torch::Tensor rw, torch::Tensor table, torch::Tensor out,
                 torch::Tensor acc_gu, torch::Tensor h, torch::Tensor acc_d, torch::Tensor cnt_gu, torch::Tensor cnt_d,
                 torch::Tensor wq, int64_t I, int64_t nm_gu, int64_t nm_dn, std::vector<int64_t> cfg_gu,
                 std::vector<int64_t> cfg_dn, int64_t G, int64_t force_level, int64_t which, int64_t hits_ptr)
{
    MoeArgs a{};
    a.x = (const half*)x.data_ptr(); a.sel = (const int64_t*)sel.data_ptr(); a.rw = (const half*)rw.data_ptr();
    a.B = x.size(0); a.topk = sel.size(1); a.table = (const int64_t*)table.data_ptr(); a.H = x.size(1); a.I = I;
    a.nm_gu = nm_gu; a.nm_dn = nm_dn; a.acc_gu = (float*)acc_gu.data_ptr(); a.h = (half*)h.data_ptr();
    a.acc_d = (float*)acc_d.data_ptr(); a.out = (float*)out.data_ptr(); a.cnt_gu = (int*)cnt_gu.data_ptr(); a.cnt_d = (int*)cnt_d.data_ptr();
    a.force_level = force_level; a.hits = (int*)hits_ptr;
    const int S = a.B * a.topk;
    TORCH_CHECK(S <= 32 && a.B <= 8, "B*topk <= 32");
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
    cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, 100 * 1024);
    int nb = 0; cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, f, 32 * sb, shm);
    cudaFuncAttributes at; cudaFuncGetAttributes(&at, f);
    return nb * 1000000 + at.numRegs * 1000 + at.localSizeBytes;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("moe_forward", &moe_forward); m.def("occ", &occ); m.def("mailbox", &mailbox); }
