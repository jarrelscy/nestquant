import sys, time; sys.path.insert(0, '/data/Jarrel/nq-tfpred/src'); import data as D
for t in sys.argv[1].split(','):
    a = time.time(); d = D.ids_blocks(t, sys.argv[2] if len(sys.argv) > 2 else 'ids'); print(t, d['cnt'].shape, d['pf'].shape, '%.0fs' % (time.time() - a), flush=True); del d
