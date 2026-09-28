import re,collections,subprocess,sys
txt=open('/tmp/nestquant/15-level4-decode/nqk15.sass').read()
funcs=re.split(r'\n\s*Function : ',txt)[1:]
want=[tuple(int(x) for x in a.split(',')) for a in sys.argv[1:]]
for f in funcs:
    dm=subprocess.run(['c++filt',f.split('\n')[0].strip()],capture_output=True,text=True).stdout.strip()
    m=re.search(r'nq15_gemv<Dec<(-?\d+), (\d+), (\d+), (\d+), (\d+), (\d+), (\d+)>, 2, 1, 0>',dm)
    if not m:continue
    k=tuple(int(v) for v in m.groups())
    if k not in want:continue
    ops=re.findall(r'/\*[0-9a-f]{4}\*/\s+(?:@!?U?P\w+\s+)?([A-Z0-9_]+(?:\.[A-Z0-9_]+)*)',f)
    c=collections.Counter(o for o in ops if o.split('.')[0] in('IMAD','PRMT','SHF','LOP3','IADD3','LEA','MOV','IDP','HFMA2'))
    print(k,sorted(c.items(),key=lambda x:-x[1])[:14])
