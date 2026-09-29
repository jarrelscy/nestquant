"""verify the repack at OUT covers L0-L1 on every rank: index entry, resident planes file, record region inside rank{r}.bin"""
import json,os,sys
out=sys.argv[1] if len(sys.argv)>1 else '/home/jarrelscy/nq-p4rec/hf';L0,L1=3,77;bad=[]
for r in range(4):
    d=json.load(open(f'{out}/rank{r}.json'));ls=d['layers'];sz=os.path.getsize(f'{out}/rank{r}.bin')
    for L in range(L0,L1+1):
        if str(L) not in ls:bad.append((r,L,'index'));continue
        if not os.path.exists(f'{out}/res/rank{r}/L{L}.pt'):bad.append((r,L,'res'))
    print(f'rank{r}: {len(ls)} layers indexed, bin {sz/2**30:.1f} GiB, rec_bytes {d.get("rec_bytes")}, sample entry L{L1}: {str(ls.get(str(L1)))[:120]}')
print('MISSING',bad if bad else 'none')
