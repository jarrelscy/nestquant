"""Static SASS op counts per variant (MODE 0, CPW 1, G 2): (ops - ops_UNI4)/128 + 1.75 (thread-04 convention; loop body = 2 chunks x 64 weights/lane)."""
import re,collections,json,subprocess,sys
txt=open('/tmp/nestquant/15-level4-decode/nqk15.sass').read()
funcs=re.split(r'\n\s*Function : ',txt)[1:]
excl={'NOP','EXIT','BRA','BAR','S2R','LDG','STG','STS','ULDC','MOV','CS2R','DEPBAR'}
names={}
for f in funcs:
    mangled=f.split('\n')[0].strip()
    dm=subprocess.run(['c++filt',mangled],capture_output=True,text=True).stdout.strip()
    m=re.search(r'nq15_gemv<Dec<(-?\d+), (\d+), (\d+), (\d+), (\d+), (\d+), (\d+)>, 2, (\d), (\d)>',dm)
    if not m:continue
    ops=re.findall(r'/\*[0-9a-f]{4}\*/\s+(?:@!?U?P\w+\s+)?([A-Z0-9_]+)(?:\.[A-Z0-9_.]+)?',f)
    c=collections.Counter(o for o in ops if o not in excl)
    names[tuple(int(v) for v in m.groups())]=c
import nq15
VAR={}
for vid in range(53):
    i=nq15.M.info(vid);VAR[vid]=(i[5],i[6],i[3],i[7],i[8],i[9],i[10])
out={}
for cpw in [1,2]:
  base=names[(0,0,5,0,0,0,0,cpw,0)];tb=sum(base.values())
  for vid,(bka,bm,rmode,rka,rm,wopt,raw) in VAR.items():
    c=names[(bka,bm,rmode,rka,rm,wopt,raw,cpw,0)];t=sum(c.values())
    pw=(t-tb)/(128*cpw)+1.75
    d={k:(c[k]-base.get(k,0))/(128*cpw) for k in set(c)|set(base)}
    grp=lambda ks:sum(d.get(k,0) for k in ks)
    out[f'{vid}|{cpw}']=dict(ops=pw,IMAD=grp(['IMAD','IMUL']),IDP=grp(['IDP']),SHF_LOP=grp(['SHF','LOP3','SHL','SHR']),PRMT=grp(['PRMT']),HFMA2=grp(['HFMA2','HMUL2','HADD2']),FFMA=grp(['FFMA']),F2FP=grp(['F2FP']),IADD=grp(['IADD3']),SHFL=grp(['SHFL']))
    print(vid,cpw,' '.join(f'{k}={v:.2f}' for k,v in out[f'{vid}|{cpw}'].items()))
json.dump(out,open('sass15.json','w'),indent=1)
