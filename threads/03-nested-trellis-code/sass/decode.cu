// Decode micro-kernels: each thread decodes 32 weights (4 groups of 8) and dots them with activations.
#include <cuda_fp16.h>
#include <stdint.h>
__device__ __forceinline__ uint32_t fsh(uint32_t lo, uint32_t hi, int s){ return __funnelshift_r(lo, hi, s); }
// EXL3 mul1 pair decode (exact copy of the op sequence in codebook.cuh)
__device__ __forceinline__ half2 mul1_2(uint32_t x0, uint32_t x1){
  x0 *= 0x83DCD12Du; x1 *= 0x83DCD12Du;
  uint32_t s0 = __dp4a(x0, 0x01010101u, 0x6400u), s1 = __dp4a(x1, 0x01010101u, 0x6400u);
  half2 h = __halves2half2(__ushort_as_half((uint16_t)s0), __ushort_as_half((uint16_t)s1));
  return __hfma2(h, __half2half2(__ushort_as_half(0x1eee)), __half2half2(__ushort_as_half(0xc931)));
}
// QTIP-HYB-like V=2 decode: hash window -> Q-bit LUT index + sign bit, LUT of half2 in smem
template<int Q> __device__ __forceinline__ half2 hyb(uint32_t w, const uint32_t* lut){
  uint32_t h = w * w + w;                                   // IMAD
  uint32_t v = lut[(h >> (15 - Q)) & ((1u << Q) - 1)];      // SHF + LOP3 (+ addr) + LDS
  v ^= h & 0x80008000u;                                     // LOP3: independent sign per half
  uint32_t r = v; return *reinterpret_cast<half2*>(&r);
}
// extract 8 windows of 16 bits, step K, from 64-bit (a:b) (like dq8 in exl3_dq.cuh, align=8/K)
template<int K> __device__ __forceinline__ void ext8(uint32_t a, uint32_t b, uint32_t* w){
  if (K == 4) { uint32_t s = fsh(b, a, 20); w[7]=b&0xffff; w[6]=(b>>4)&0xffff; w[5]=(b>>8)&0xffff; w[4]=(b>>12)&0xffff; w[3]=b>>16; w[2]=s&0xffff; w[1]=(s>>4)&0xffff; w[0]=(s>>8)&0xffff; }
  else if (K == 2) { uint32_t s = b; for (int j=7;j>=0;--j){ w[j]=(s>>(2*(7-j)))&0xffff; } }
  else { for (int j=7;j>=0;--j){ w[j]=(b>>(7-j))&0xffff; } }
}
#define ACC(v) acc = __hfma2(v, x2[i*4 + (j>>1)], acc)
// 1) native EXL3 mul1 at K=4 / K=2
template<int K> __device__ __forceinline__ void native(const uint32_t* sw, const half2* x2, half2* out){
  half2 acc = __float2half2_rn(0.f);
  #pragma unroll
  for (int i=0;i<4;i++){ uint32_t a=sw[threadIdx.x*8+2*i], b=sw[threadIdx.x*8+2*i+1]; uint32_t w[8]; ext8<K>(a,b,w);
    #pragma unroll
    for(int j=0;j<8;j+=2){ half2 v=mul1_2(w[j],w[j+1]); ACC(v);} }
  out[threadIdx.x]=acc;
}
extern "C" __global__ void k_native4(const uint32_t* g, const half2* x2, half2* out){ __shared__ uint32_t sw[2048]; sw[threadIdx.x]=g[threadIdx.x]; __syncthreads(); native<4>(sw,x2,out); }
extern "C" __global__ void k_native2(const uint32_t* g, const half2* x2, half2* out){ __shared__ uint32_t sw[2048]; sw[threadIdx.x]=g[threadIdx.x]; __syncthreads(); native<2>(sw,x2,out); }
// 2) residual mul1 K2 base + mul1 K2 refinement (level 4)
extern "C" __global__ void k_res22(const uint32_t* g, const half2* x2, half2* out){ __shared__ uint32_t sw[2048], sr[2048]; sw[threadIdx.x]=g[threadIdx.x]; sr[threadIdx.x]=g[threadIdx.x+2048]; __syncthreads();
  half2 acc = __float2half2_rn(0.f);
  #pragma unroll
  for (int i=0;i<4;i++){ uint32_t w[8], r[8]; ext8<2>(sw[threadIdx.x*8+2*i],sw[threadIdx.x*8+2*i+1],w); ext8<2>(sr[threadIdx.x*8+2*i],sr[threadIdx.x*8+2*i+1],r);
    #pragma unroll
    for(int j=0;j<8;j+=2){ half2 v=__hadd2(mul1_2(w[j],w[j+1]), __hmul2(__half2half2(__ushort_as_half(0x3400)), mul1_2(r[j],r[j+1]))); ACC(v);} }
  out[threadIdx.x]=acc; }
// 3) residual mul1 K2 + K1 + K1 (level 4 of the 3-level nest)
extern "C" __global__ void k_res211(const uint32_t* g, const half2* x2, half2* out){ __shared__ uint32_t sw[2048], s3[1024], s4[1024]; sw[threadIdx.x]=g[threadIdx.x]; s3[threadIdx.x]=g[threadIdx.x+2048]; s4[threadIdx.x]=g[threadIdx.x+3072]; __syncthreads();
  half2 acc = __float2half2_rn(0.f);
  #pragma unroll
  for (int i=0;i<4;i++){ uint32_t w[8], r[8], t[8]; ext8<2>(sw[threadIdx.x*8+2*i],sw[threadIdx.x*8+2*i+1],w); ext8<1>(s3[threadIdx.x*4+i],s3[threadIdx.x*4+i+1],r); ext8<1>(s4[threadIdx.x*4+i],s4[threadIdx.x*4+i+1],t);
    #pragma unroll
    for(int j=0;j<8;j+=2){ half2 v=__hfma2(__half2half2(__ushort_as_half(0x3000)), mul1_2(t[j],t[j+1]), __hfma2(__half2half2(__ushort_as_half(0x3400)), mul1_2(r[j],r[j+1]), mul1_2(w[j],w[j+1]))); ACC(v);} }
  out[threadIdx.x]=acc; }
// 4) HYB V2 base + HYB V2 refinement (4 weights per 16-bit... V=2: window step = 4 bits per pair)
extern "C" __global__ void k_hyb22(const uint32_t* g, const half2* x2, half2* out){ __shared__ uint32_t sw[2048], sr[2048], lb[512], lr[512]; sw[threadIdx.x]=g[threadIdx.x]; sr[threadIdx.x]=g[threadIdx.x+2048]; lb[threadIdx.x&511]=g[threadIdx.x+4096]; lr[threadIdx.x&511]=g[threadIdx.x+4608]; __syncthreads();
  half2 acc = __float2half2_rn(0.f);
  #pragma unroll
  for (int i=0;i<4;i++){ uint32_t w[8], r[8]; ext8<4>(sw[threadIdx.x*8+2*i],sw[threadIdx.x*8+2*i+1],w); ext8<4>(sr[threadIdx.x*8+2*i],sr[threadIdx.x*8+2*i+1],r);
    // 8 windows at 4 bits/step = 8 pairs = 16 weights per group here; count per weight accordingly
    #pragma unroll
    for(int j=0;j<8;j++){ half2 v=__hadd2(hyb<9>(w[j],lb), hyb<9>(r[j],lr)); acc=__hfma2(v,x2[i*8+j],acc);} }
  out[threadIdx.x]=acc; }
extern "C" __global__ void k_hyb2(const uint32_t* g, const half2* x2, half2* out){ __shared__ uint32_t sw[2048], lb[512]; sw[threadIdx.x]=g[threadIdx.x]; lb[threadIdx.x&511]=g[threadIdx.x+4096]; __syncthreads();
  half2 acc = __float2half2_rn(0.f);
  #pragma unroll
  for (int i=0;i<4;i++){ uint32_t w[8]; ext8<4>(sw[threadIdx.x*8+2*i],sw[threadIdx.x*8+2*i+1],w);
    #pragma unroll
    for(int j=0;j<8;j++){ half2 v=hyb<9>(w[j],lb); acc=__hfma2(v,x2[i*8+j],acc);} }
  out[threadIdx.x]=acc; }
// 5) mul1 K2 base + HYB V2 refinement
extern "C" __global__ void k_mul1_hyb(const uint32_t* g, const half2* x2, half2* out){ __shared__ uint32_t sw[2048], sr[2048], lr[512]; sw[threadIdx.x]=g[threadIdx.x]; sr[threadIdx.x]=g[threadIdx.x+2048]; lr[threadIdx.x&511]=g[threadIdx.x+4608]; __syncthreads();
  half2 acc = __float2half2_rn(0.f);
  #pragma unroll
  for (int i=0;i<4;i++){ uint32_t w[8], r[8]; ext8<2>(sw[threadIdx.x*8+2*i],sw[threadIdx.x*8+2*i+1],w); ext8<4>(sr[threadIdx.x*4+i],sr[threadIdx.x*4+i+1],r);
    #pragma unroll
    for(int j=0;j<8;j+=2){ half2 v=__hadd2(mul1_2(w[j],w[j+1]), hyb<9>(r[j>>1],lr)); ACC(v);} }
  out[threadIdx.x]=acc; }

