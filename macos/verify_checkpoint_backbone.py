import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import torch
from transformers import AutoTokenizer

from backbone import empty_model, load_backbone
from experts import ExpertStore, NestQuantExperts
from metrics import snapshot
from prefill import layer_major_prefill


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.model / 'nonexpert_manifest.json').read_text())
    files = {entry['file']: entry for entry in manifest['files']}
    for number in range(1, 5):
        name = f'nonexpert-{number:05d}-of-00004.safetensors'
        path = args.model / name
        assert path.stat().st_size == files[name]['bytes'], name
        with path.open('rb') as handle:
            assert hashlib.file_digest(handle, 'sha256').hexdigest() == files[name]['sha256'], name
        print(json.dumps(dict(event='checksum_verified', file=name)), flush=True)
    torch.set_num_threads(8)
    mx.set_cache_limit(1024**3)
    state = SimpleNamespace(phase='prefill', routing={}, wanted=np.ones((42, 288), dtype=bool))
    store = ExpertStore(args.model, cache_gib=0.25)
    checks = []
    try:
        model = empty_model(args.model, lambda layer: NestQuantExperts(layer, store, state))
        head = load_backbone(args.model, model)
        tensors = 0
        for tensor in model.state_dict().values():
            assert tensor.device.type == 'mps'
            assert torch.isfinite(tensor).all()
            tensors += 1
        assert torch.isfinite(head).all()
        loaded_resources = snapshot()
        checks.append(dict(check='backbone_loaded', tensors=tensors, lm_head=True, finite=True))
        del head
        model.layers = torch.nn.ModuleList(model.layers[:4])
        model.config.num_hidden_layers = 4
        for field in ('layer_types', 'indexer_types', 'mlp_layer_types'):
            setattr(model.config, field, getattr(model.config, field)[:4])
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
        tokens = tokenizer.encode('A local language model should produce clear and useful answers.')
        ids = torch.tensor([tokens], device='mps')
        with torch.inference_mode():
            unchunked = model(ids, use_cache=True)
            for chunk_size in (3, 5):
                state.phase = 'prefill'
                layered = layer_major_prefill(model, ids, chunk_size)
                cache = None
                pieces = []
                for start in range(0, len(tokens), chunk_size):
                    standard = model(ids[:, start:start + chunk_size], past_key_values=cache, use_cache=True)
                    cache = standard.past_key_values
                    pieces.append(standard.last_hidden_state)
                expected = torch.cat(pieces, dim=1)
                torch.testing.assert_close(layered.last_hidden_state, expected, atol=0, rtol=0)
                assert torch.isfinite(layered.last_hidden_state).all()
                difference = ((layered.last_hidden_state.float() - unchunked.last_hidden_state.float()).norm()
                              / unchunked.last_hidden_state.float().norm()).item()
                state.phase = 'decode'
                for token in tokens[:2]:
                    next_id = torch.tensor([[token]], device='mps')
                    standard = model(next_id, past_key_values=cache, use_cache=True)
                    layered = model(next_id, past_key_values=layered.past_key_values, use_cache=True)
                    cache = standard.past_key_values
                    torch.testing.assert_close(layered.last_hidden_state, standard.last_hidden_state, atol=0, rtol=0)
                check = dict(check='trained_prefix', layers=4, chunk_size=chunk_size,
                             prompt_tokens=len(tokens), exact_matching_chunks=True,
                             cached_decode_tokens=2, unchunked_prefill_relative_rms=difference)
                checks.append(check)
                print(json.dumps(check), flush=True)
        report = dict(status='passed', checks=checks, resources_after_full_backbone_load=loaded_resources,
                      scope='All trained backbone tensors loaded; four-layer trained prefix with real L3 experts; not full-model generation')
    finally:
        store.close()
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
