"""(1) ref15_spec == nq15 reference (== kernel, check15 Wmax 0) on random G=4 units; (2) spec Q2 == harness mul1 LUT;
(3) thread-12 level-4 (Q2 + fp16 delta * mul1(sr)) vs RM_P fold with delta -> (Mb, N)."""
import os,sys,numpy as np,torch;os.environ['NQ15_G']='4';torch.cuda.set_per_process_memory_fraction(12/80)
import ref15_spec as R, nq15
sys.path.insert(0,'/home/coder/git/nestquant/threads/05-exl3-harness');import harness as h
lut=h.codebook_lut('mul1','cpu').double().numpy()
st=np.arange(65536);print('(2) Q2 spec == harness mul1 LUT:',np.array_equal(R.q2(st),lut))
for vid in [33,29,30,38,39,37,31,32,28]:
    p=nq15.Proj(vid,256,512);f=nq15.ref_vals(p)            # [nrec,64] lane order
    bb,rb,hasr,rm,v2,bka,bm,rka,rmask=M=nq15.M.info(vid)[:9]
    bad=0
    for u in range(p.nrec//32):
        sl=slice(32*u,32*u+32)
        bs=R.rings_from_lane_words(p.wb[sl],bb);rs=R.rings_from_lane_words(p.wr[sl],rb)
        _,q4=R.decode_unit(bs,rs,int(p.Mb[u]),int(p.Nn[u]),(bka,bm),(rka,rmask),bool(v2))
        bad+=int(not np.array_equal(q4.reshape(8,4,64).reshape(32,64),f[sl]))
    print(f'(1) vid {vid} bits {bb}/{rb} V2={v2}: units mismatching spec vs kernel-ref: {bad}/{p.nrec//32}')
rng=np.random.default_rng(0);n=1<<20
sb=rng.integers(0,65536,n);sr=rng.integers(0,65536,n)
Q2=lut[sb];g=lut[sr]
for lo,hi in [(0.1,0.2),(0.2,0.35),(0.35,0.5),(0.5,1.0)]:
    d=R.f16(rng.uniform(lo,hi,n))
    t12=(Q2.astype(np.float32)+d.astype(np.float32)*g.astype(np.float32)).astype(np.float16).astype(np.float64)
    Mb,N=R.delta_to_MbN(d);qp=R.fold(R.S(sb),R.S(sr),Mb,N)
    ideal=Q2+(N/Mb)*g                                        # same delta after (Mb,N) quantization, no fold rounding
    e_all=qp-t12;e_rnd=qp-ideal;dq=N/Mb-d
    print(f'(3) delta {lo}-{hi}: Mb {Mb.min()}-{Mb.max()}; |dq| max {abs(dq).max():.2e}; '
          f'P-vs-T12 rms {np.sqrt((e_all**2).mean()):.4f} max {abs(e_all).max():.4f}; fold-rounding-only rms {np.sqrt((e_rnd**2).mean()):.4f}; '
          f'rel-MSE vs rms(Q4)^2: {(e_all**2).mean()/(t12**2).mean():.2e}; mean bias {e_all.mean():+.1e}')
