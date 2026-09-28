"""Per-weight decode-loop op counts from nvdisasm -g line info: instructions whose source line lies in the per-chunk decode
functions (wv/dp4a/hfma2u, mma, ext_words..compute_stage) + header-inlined HFMA2/HMUL2/F2FP/HADD2 (fp16 math).
Excludes loads (LDG) and their address math, prologue (x staging, WHT, xsum), epilogue. Per weight = count / (2 stages * CPW * 64)."""
import re,collections,subprocess,json,sys
src=open('nqk15.cu').read().split('\n')
def rng(start_pat,end_pat):
    a=next(i for i,l in enumerate(src) if start_pat in l)+1;b=next(i for i,l in enumerate(src) if end_pat in l and i+1>a)+1;return a,b
R=[rng('template <int O, int WOPT','template <int BKA_'),rng('void mma16816','template <int G> __device__'),
   rng('template <int BITS, int NW>','__device__ __forceinline__ void wht128_warp')]
inreg=lambda n:any(a<=n<b for a,b in R)
txt=open('/tmp/nestquant/15-level4-decode/dis.txt').read()
funcs=re.split(r'\n\s*\.text\.',txt)[1:]
excl={'NOP','EXIT','BRA','BAR','S2R','LDG','STG','STS','ULDC','MOV','CS2R','DEPBAR'}
res={}
for f in funcs:
    mangled=f.split(':')[0].strip()
    dm=subprocess.run(['c++filt',mangled],capture_output=True,text=True).stdout.strip()
    m=re.search(r'nq15_gemv<Dec<(-?\d+), (\d+), (\d+), (\d+), (\d+), (\d+), (\d+)>, 2, (\d), 0>',dm)
    if not m:continue
    key=tuple(int(v) for v in m.groups())
    cur=None;c=collections.Counter()
    for line in f.split('\n'):
        lm=re.search(r'//## File "([^"]*)", line (\d+)',line)
        if lm:cur=(lm.group(1).endswith('nqk15.cu'),int(lm.group(2)));continue
        im=re.search(r'/\*[0-9a-f]{4}\*/\s+(?:@!?U?P\w+\s+)?([A-Z0-9_]+)(\.[A-Z0-9_.]+)?',line)
        if not im or cur is None:continue
        op=im.group(1);full=op+(im.group(2) or '')
        if op in excl:continue
        if cur[0] and inreg(cur[1]):c[op]+=1
        elif not cur[0] and op in('HFMA2','HMUL2','F2FP','HADD2','SHFL','F2F') and 'F32' not in full:c[op]+=1
    res[key]=c
json.dump({str(k):v for k,v in res.items()},open('/tmp/nestquant/15-level4-decode/sassline.json','w'))
import nq15
out={}
for cpw in [1,2]:
  u=res[(0,0,5,0,0,0,0,cpw)];tu=sum(u.values())/(128*cpw)
  for vid in range(53):
    i=nq15.M.info(vid);k=(i[5],i[6],i[3],i[7],i[8],i[9],i[10],cpw)
    c=res[k];t=sum(c.values())/(128*cpw)
    g=lambda ks:sum(c[x] for x in ks)/(128*cpw)
    out[f'{vid}|{cpw}']=dict(abs=t,conv=t-tu+1.75,IMAD=g(['IMAD']),IDP=g(['IDP']),SHF_LOP=g(['SHF','LOP3']),PRMT=g(['PRMT']),HF=g(['HFMA2','HMUL2','HADD2']),FFMA=g(['FFMA','F2FP']),HMMA=g(['HMMA']),LDS=g(['LDS']),other=t-g(['IMAD','IDP','SHF','LOP3','PRMT','HFMA2','HMUL2','HADD2','FFMA','F2FP','HMMA','LDS']))
    if cpw==int(sys.argv[1] if len(sys.argv)>1 else 1):print(vid,' '.join(f'{a}={b:.2f}' for a,b in out[f'{vid}|{cpw}'].items()))
json.dump(out,open('sass15.json','w'),indent=1)
