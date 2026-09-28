import json,sys
bits=sys.argv[1];R=json.load(open(f'sweep_final_{bits}.json'));B=json.load(open('baselines.json'));E=['36','92','165']
sel=json.load(open(f'selection_{bits}.json'))['selected'] if len(sys.argv)<3 else sys.argv[2]
names=[k.split('/')[1] for k in R if k.startswith('36/') and '#' not in k]
order=json.load(open(f'selection_{bits}.json'))['table'];rank={r['name']:i for i,r in enumerate(order)}
names.sort(key=lambda n:rank.get(n,999))
M=['ID','OOD','routed_all']
print(f'| setting (sel. score) | '+' | '.join(f'E{e} ID / OOD / routed' for e in E)+' | mean ID | mean OOD |')
print('|---|'+'---|'*(len(E)+2))
def row(label,get):
    vals=[[get(e)[m] for m in M] for e in E]
    print(f'| {label} | '+' | '.join(f'{a:.2f} / {b:.2f} / {c:.2f}' for a,b,c in vals)+f' | {sum(v[0] for v in vals)/3:.2f} | {sum(v[1] for v in vals)/3:.2f} |')
bk='exl3_4' if bits=='4' else 'exl3_2'
row(f'EXL3-{bits} artifact',lambda e:{**B[e][bk],'routed_all':float('nan')})
if bits=='4':row('NVFP4 artifact',lambda e:{**B[e]['nvfp4'],'routed_all':float('nan')})
for n in names:
    sc=next((r['score'] for r in order if r['name']==n),float('nan'))
    row(('**'+n+'** (selected)' if n==sel else n)+f' ({sc:.3f})',lambda e:R[f'{e}/{n}'])
print()
print('| per-OOD-domain forced, mean over 3 experts | fasta | encoded_bytes | smt_bitvectors | scientific_telemetry |')
print('|---|---|---|---|---|')
D=['ood:fasta','ood:encoded_bytes','ood:smt_bitvectors','ood:scientific_telemetry']
def drow(label,get):print(f'| {label} | '+' | '.join(f'{sum(get(e)[d] for e in E)/3:.2f}' for d in D)+' |')
drow(f'EXL3-{bits}',lambda e:B[e][bk])
if bits=='4':drow('NVFP4',lambda e:B[e]['nvfp4'])
drow('baseline refit',lambda e:R[f'{e}/baseline']);drow(sel,lambda e:R[f'{e}/{sel}'])
