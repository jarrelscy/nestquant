"""Grouped GEMM for the NQ prefill path: C[rows of expert j] = A[rows] @ W[j]^T, fp16 in, fp32 accumulate + fp32 out.
One launch per projection per row chunk (instead of one cuBLAS call per expert): tiles = (expert, BM-row block) x
BN-column block, rows of each expert contiguous (pairs sorted by expert). Tile list built on the host from the
expert counts the prefill path already has there."""
import torch,triton,triton.language as tl

@triton.jit
def _gmm(A,W,C,TJ,TM0,TM1,K,sa,swg,sw,sc,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,GRP:tl.constexpr):
    pid=tl.program_id(0);nt=tl.num_programs(0)//GRP
    pm=pid%nt;pn=pid//nt           # consecutive programs share the same weight column block (W tile stays in L2)
    j=tl.load(TJ+pm).to(tl.int64);m0=tl.load(TM0+pm);m1=tl.load(TM1+pm)
    rm=m0+tl.arange(0,BM);rn=pn*BN+tl.arange(0,BN);rk=tl.arange(0,BK)
    mask=rm[:,None]<m1
    a_p=A+rm[:,None].to(tl.int64)*sa+rk[None,:]
    w_p=W+j*swg+rn[None,:].to(tl.int64)*sw+rk[:,None]
    acc=tl.zeros((BM,BN),dtype=tl.float32)
    for k in range(0,K,BK):
        a=tl.load(a_p,mask=mask,other=0.);b=tl.load(w_p)
        acc=tl.dot(a,b,acc)
        a_p+=BK;w_p+=BK
    tl.store(C+rm[:,None].to(tl.int64)*sc+rn[None,:],acc,mask=mask)

def tiles(segs,BM,dev):
    """segs [(j, u, v)] row ranges -> device int32 [3, ntiles] (expert, row0, row1)"""
    tj,t0,t1=[],[],[]
    for j,u,v in segs:
        for m in range(u,v,BM):tj.append(j);t0.append(m);t1.append(min(v,m+BM))
    return torch.tensor([tj,t0,t1],dtype=torch.int32).pin_memory().to(dev,non_blocking=True),len(tj)

def gmm(A,W,C,tl_,nt,N,K,BM=64,BN=128,BK=64,warps=4,stages=3):
    """A [R,K] fp16 (row stride A.stride(0)), W [G,>=N,K] fp16 contiguous rows, C [R,N] fp32; columns [0, N)."""
    if not nt:return
    _gmm[(nt*(N//BN),)](A,W,C,tl_[0],tl_[1],tl_[2],K,A.stride(0),W.stride(0),W.stride(1),C.stride(0),
                        BM=BM,BN=BN,BK=BK,GRP=N//BN,num_warps=warps,num_stages=stages)
