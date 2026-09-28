import sys,json
for l in sys.stdin:
    try: r=json.loads(l)
    except: 
        if 'Warn' not in l: print(l.strip())
        continue
    print(r['K'],r['sigma_reg'],'bpw %.3f'%r['bpw'],'hold %.3f'%r['hold'],{k.replace('pilot/',''):round(v,2) for k,v in r['eval'].items()},flush=True)
