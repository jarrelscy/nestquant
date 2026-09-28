"""Byte census from safetensors headers (no tensor reads) + expert sizing at 2/3/4 bpw."""
import json, struct, os, re, collections, sys
def census(root):
    idx = json.load(open(os.path.join(root, 'model.safetensors.index.json')))['weight_map']
    files = sorted(set(idx.values()))
    tot = collections.Counter(); missing = []
    for f in files:
        p = os.path.join(root, f)
        try:
            with open(p, 'rb') as fh:
                n = struct.unpack('<Q', fh.read(8))[0]; h = json.loads(fh.read(n))
        except Exception as e:
            missing.append(f); continue
        for k, v in h.items():
            if k == '__metadata__': continue
            b = v['data_offsets'][1] - v['data_offsets'][0]
            if re.search(r'mlp\.experts\.\d+\.', k): cat = 'routed_expert'
            elif 'shared_expert' in k: cat = 'shared_expert'
            elif re.search(r'(visual|vision|audio)', k): cat = 'vision_audio'
            else: cat = 'other_nonexpert'
            tot[cat] += b
    return tot, missing, len(files)
for name, root in [('mimo', '/tmp/mimo-a100/data/jarrel/mimo-exl3-sequential/source'), ('glm', '/tmp/orbit-duet-glm53-fp8')]:
    tot, missing, nf = census(root)
    print(name, {k: round(v/1e9, 2) for k, v in tot.items()}, 'GB; files', nf, 'missing', len(missing))
