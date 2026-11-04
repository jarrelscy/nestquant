import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextModel

from backbone import text_config
from experts import ExpertStore, NestQuantExperts
from prefill import layer_major_prefill


class FixedRouter(torch.nn.Module):
    def forward(self, hidden):
        tokens = hidden.numel() // hidden.shape[-1]
        ids = torch.tensor([0, 7, 98, 177, 218, 241, 277, 287], device=hidden.device).expand(tokens, -1)
        weights = torch.full((tokens, 8), 2.5 / 8, dtype=torch.float32, device=hidden.device)
        return None, weights, ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--reference', choices=['unchunked', 'chunked'], default='unchunked')
    parser.add_argument('--diagnose-chunking', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(1234)
    config = text_config(args.model)
    config.num_hidden_layers = 4
    config.layer_types = config.layer_types[:4]
    config.indexer_types = config.indexer_types[:4]
    config.mlp_layer_types = config.mlp_layer_types[:4]
    config.vocab_size = 64
    config.pad_token_id = 0
    config.eos_token_id = [1]
    state = SimpleNamespace(phase='prefill', routing={}, decode_ready_experts=0, wanted=np.ones((42, 288), dtype=bool))
    store = ExpertStore(args.model, cache_gib=0.25)
    with torch.device('meta'):
        model = Glm5NextTextModel(config)
    model.layers[3].mlp.experts = NestQuantExperts(3, store, state)
    model.layers[3].mlp.gate = FixedRouter()
    model = model.to_empty(device='mps').to(dtype=torch.bfloat16).eval()
    diagnostics = {}
    capture = None
    if args.diagnose_chunking:
        def observer(key):
            def record(_, __, output):
                if capture is not None:
                    value = output[0] if isinstance(output, tuple) else output
                    diagnostics.setdefault(capture, {}).setdefault(key, []).append(value.detach().clone())
            return record

        for index, layer in enumerate(model.layers):
            layer.register_forward_hook(observer(f'{index}.block'))
            layer.self_attn.register_forward_hook(observer(f'{index}.attention'))
    with torch.inference_mode():
        model.apply(model._init_weights)
        for layer in model.layers:
            layer.attn_hc.float()
            layer.ffn_hc.float()
            if hasattr(layer.self_attn, 'conv1d'):
                layer.self_attn.conv1d.float()
                layer.self_attn.forget_gate.A_log.data = layer.self_attn.forget_gate.A_log.data.float()
                layer.self_attn.forget_gate.dt_bias.data = layer.self_attn.forget_gate.dt_bias.data.float()
        tokens = torch.tensor([[5, 12, 25]], device='mps')
        capture = 'unchunked' if args.diagnose_chunking else None
        normal = model(tokens, use_cache=True)
        capture = None
        layered = layer_major_prefill(model, tokens, chunk_size=2)
        capture = 'chunked' if args.diagnose_chunking else None
        standard_first = model(tokens[:, :2], use_cache=True)
        standard_second = model(tokens[:, 2:], past_key_values=standard_first.past_key_values, use_cache=True)
        capture = None
        chunking = {}
        if args.diagnose_chunking:
            for name, full in diagnostics['unchunked'].items():
                expected = full[0].float()
                actual = torch.cat(diagnostics['chunked'][name], dim=1).float()
                error = ((actual - expected).norm() / expected.norm().clamp_min(1e-30)).item()
                chunking[name] = error
            print(json.dumps(dict(diagnostic='unchunked_vs_standard_chunked_layers', relative_rms=chunking)), flush=True)
        standard_hidden = torch.cat((standard_first.last_hidden_state, standard_second.last_hidden_state), dim=1)
        standard_diff = ((layered.last_hidden_state.float()-standard_hidden.float()).norm() / standard_hidden.float().norm()).item()
        print(json.dumps(dict(diagnostic='layer_major_vs_standard_chunked_prefill', relative_rms=standard_diff)), flush=True)
        checks = []
        prefill_reference = normal.last_hidden_state if args.reference == 'unchunked' else standard_hidden
        for phase, actual, expected in [('prefill', layered.last_hidden_state, prefill_reference)]:
            error = ((actual.float()-expected.float()).norm() / expected.float().norm()).item()
            assert torch.isfinite(actual).all()
            if args.reference == 'chunked':
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            else:
                assert error < 0.01, error
            checks.append(dict(phase=phase, relative_rms=error))
        state.phase = 'decode'
        normal_decode = model(torch.tensor([[33]], device='mps'), past_key_values=normal.past_key_values, use_cache=True)
        layered_decode = model(torch.tensor([[33]], device='mps'), past_key_values=layered.past_key_values, use_cache=True)
        standard_decode = model(torch.tensor([[33]], device='mps'), past_key_values=standard_second.past_key_values, use_cache=True)
        standard_diff = ((layered_decode.last_hidden_state.float()-standard_decode.last_hidden_state.float()).norm() / standard_decode.last_hidden_state.float().norm()).item()
        print(json.dumps(dict(diagnostic='layer_major_vs_standard_chunked_decode', relative_rms=standard_diff)), flush=True)
        full_difference = ((layered_decode.last_hidden_state.float()-normal_decode.last_hidden_state.float()).norm() / normal_decode.last_hidden_state.float().norm()).item()
        decode_reference = normal_decode if args.reference == 'unchunked' else standard_decode
        actual, expected = layered_decode.last_hidden_state, decode_reference.last_hidden_state
        error = ((actual.float()-expected.float()).norm() / expected.float().norm()).item()
        assert torch.isfinite(actual).all()
        if args.reference == 'chunked':
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        else:
            assert error < 0.01, error
        checks.append(dict(phase='cached_decode', relative_rms=error))
    store.close()
    report = dict(status='passed', reference=args.reference, unchunked_decode_relative_rms=full_difference, unchunked_1_percent_check_passed=full_difference < 0.01, scope='Four random BF16 backbone layers at real Flash dimensions, real L3 NestQuant experts, fixed routing; no model quality claim', tolerance_relative_rms=0 if args.reference == 'chunked' else 0.01, checks=checks)
    if args.diagnose_chunking:
        report['unchunked_vs_standard_chunked_layers'] = chunking
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
