import sys, json, torch
sys.path.insert(0, '/home/coder/git/nestquant/threads/05-exl3-harness'); sys.path.insert(0, '/home/coder/git/orbit-duet')
import harness as H
OD = '/home/coder/git/orbit-duet'; T = '/tmp/nestquant/06-expert-objective'
def ld(model, names): return [torch.load(f'{T}/{model}_full/{n}.pt').cuda() for n in names]
out = {}
for model, L, E, src, st, caps, rec in [
    ('glm', 16, 36, '/tmp/orbit-duet-glm53-fp8', f'{OD}/runs/glm53_pilot_matched_l16/statistics/l16_e36.pt', ['matched'], ('pw1_sr0.3_auto', 'GOdiag_gb0.5')),
    ('mimo', 55, 70, f'{OD}/runs/source_mimo', f'{OD}/runs/full55_statistics/l55_e70.pt',
     [f'{OD}/runs/native_id_control_v1_capture/layer_55.pt', f'{OD}/runs/ood_controlled_v1_capture/layer_55.pt'], ('pw2_sr0.03_auto', 'GOdiag_gb0.5'))]:
    exl = f"{OD}/runs/glm53_pilot_matched_l16/exl3_e36" if model == 'glm' else f"{OD}/runs/full55_exl3_e70"
    for cap in caps:
        data = H.load_expert(L, E, source=src, statistics=st, capture=cap)
        methods = {}
        for b in [2, 4]:
            methods[f'exl3_{b}'] = H.load_exl3_bin(f'{exl}/expert_{b}.bin')
            methods[f'rec_{b}'] = ld(model, [f'g_b{b}_{rec[0]}_{rec[1]}', f'u_b{b}_{rec[0]}_{rec[1]}', f'd_b{b}_{rec[0]}_H'])
        if model == 'glm':
            methods['nvfp4'] = H.load_nvfp4(f'{OD}/runs/glm53_pilot_matched_l16/nvfp4_e36/weights.pt', data)
        tb = H.table(H.evaluate(data, methods)); out[f'{model}:{cap[-40:]}'] = tb
        print(model, cap[-40:], json.dumps(tb, indent=0), flush=True)
        del data, methods; torch.cuda.empty_cache()
json.dump(out, open('xcheck.json', 'w'), indent=1)
