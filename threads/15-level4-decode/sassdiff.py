import re,collections,subprocess,sys
txt=open('/tmp/nestquant/15-level4-decode/nqk15.sass').read()
funcs=re.split(r'\n\s*Function : ',txt)[1:]
want=[tuple(int(x) for x in a.split(',')) for a in sys.argv[1:3]];cpw=int(sys.argv[3]) if len(sys.argv)>3 else 1
excl={'NOP','EXIT','BRA','BAR','S2R','LDG','STG','STS','ULDC','MOV','CS2R','DEPBAR'}
got={}
for f in funcs:
    dm=subprocess.run(['c++filt',f.split('\n')[0].strip()],capture_output=True,text=True).stdout.strip()
    m=re.search(r'nq15_gemv<Dec<(-?\d+), (\d+), (\d+), (\d+), (\d+), (\d+), (\d+)>, 2, (\d), 0>',dm)
    if not m or int(m.group(8))!=cpw:continue
    k=tuple(int(v) for v in m.groups()[:7])
    if k in want:
        ops=re.findall(r'/\*[0-9a-f]{4}\*/\s+(?:@!?U?P\w+\s+)?([A-Z0-9_]+(?:\.[A-Z0-9_]+)*)',f)
        got[k]=collections.Counter(o for o in ops if o.split('.')[0] not in excl)
a,b=got[want[0]],got[want[1]]
print(sum(a.values()),sum(b.values()))
for k in sorted(set(a)|set(b),key=lambda k:-(abs(a[k]-b[k]))):
    if a[k]!=b[k]:print(k,a[k],b[k],b[k]-a[k])
