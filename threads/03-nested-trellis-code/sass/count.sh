NV=/home/coder/git/glm52/.venv/lib/python3.12/site-packages/nvidia/cu13; export PATH=$NV/bin:$PATH
cd /home/coder/git/nestquant/threads/03-nested-trellis-code/sass
nvcc -arch=sm_80 -O3 -cubin -o decode.cubin decode.cu 2>&1 | grep -i error
/home/coder/.cache/uv/archive-v0/eoh2YYTFeztcAFn_/triton/backends/nvidia/bin/cuobjdump -sass decode.cubin > decode.sass 2>/dev/null
python3 - <<'PY'
import re,collections,json
txt=open('decode.sass').read()
funcs=re.split(r'\n\s*Function : ',txt)[1:]
wpt={'k_native4':32,'k_native2':32,'k_res22':32,'k_res211':32,'k_hyb22':64,'k_hyb2':64,'k_mul1_hyb':32}
res={}
for f in funcs:
    name=f.split('\n')[0].strip()
    ops=re.findall(r'/\*[0-9a-f]{4}\*/\s+(?:@!?P\d\s+)?([A-Z0-9_]+)(?:\.[A-Z0-9_.]+)?',f)
    c=collections.Counter(ops)
    excl={'NOP','EXIT','BRA','BAR','S2R','LDG','STG','STS','ULDC','MOV','CS2R','DEPBAR'}
    alu={k:v for k,v in c.items() if k not in excl}
    tot=sum(alu.values()); res[name]=dict(per_weight=tot/wpt[name],ops=alu)
    print(f"{name:12s} per weight {tot/wpt[name]:.2f}  LDS/w {alu.get('LDS',0)/wpt[name]:.2f}", dict(sorted(alu.items(),key=lambda x:-x[1])))
json.dump(res,open('sass_counts.json','w'),indent=1)
PY
