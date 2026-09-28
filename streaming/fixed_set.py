"""Fixed 10% set per MoE layer (updates 02/03): 26 experts by token-weighted REAP salience (boundary weights 50/20/5/2/1).
Source, in order: the artifact's fixed_set.json ({layer: [experts]}), then threads/22-boundary-experts/fixed_set.json
(d["fixed_set"][L], L = absolute layer as a string key). The set is used as-is. If serving usage counts are given, the
route coverage is reported for information (random = 26/256 = 10.2%); expert numbering was checked against tb4 with
streaming/check_numbering.py (median rho 0.36)."""
import json,os
import numpy as np
R=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T22=R+'/threads/22-boundary-experts/fixed_set.json'
K=26
def coverage(sets,usage):
    return float(np.mean([np.asarray(usage[L])[sets[L]].sum()/np.asarray(usage[L]).sum() for L in sets]))
def load(artifact_dir=None,layers=range(3,78),usage=None):
    """-> ({layer: [experts]}, source, coverage or None). usage: optional {layer: counts[256]} from serving routing."""
    for p,src in ((artifact_dir and os.path.join(artifact_dir,'fixed_set.json'),'artifact'),(T22,'thread22')):
        if p and os.path.exists(p):
            d=json.load(open(p));d=d.get('fixed_set',d)
            sets={L:[int(e) for e in d[str(L)]][:K] for L in layers}
            return sets,src,(coverage(sets,usage) if usage is not None else None)
    raise FileNotFoundError('no fixed_set.json (artifact or threads/22-boundary-experts)')
if __name__=='__main__':
    fs,src,cov=load();print(src,len(fs),'layers, e.g. L3',fs[3][:8])
