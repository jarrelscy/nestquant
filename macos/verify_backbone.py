import copy
import json
from pathlib import Path

from prefill import layer_major_prefill

import torch
from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig
from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextModel


def main():
    torch.manual_seed(17)
    torch.set_num_threads(8)
    config = Glm5NextTextConfig(
        vocab_size=128, hidden_size=128, intermediate_size=256, moe_intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
        n_routed_experts=4, num_experts_per_tok=2, kv_lora_rank=32, q_lora_rank=32,
        qk_nope_head_dim=64, v_head_dim=64, index_head_dim=32, index_n_heads=2,
        index_topk=8, index_kpool=4, linear_num_heads=2, linear_head_dim=32,
        mlp_layer_types=['dense', 'sparse'], layer_types=['linear_attention', 'indexed_attention'],
        indexer_types=['full', 'full'], pad_token_id=0,
    )
    config._attn_implementation = 'eager'
    cpu = Glm5NextTextModel(config).eval()
    gpu = copy.deepcopy(cpu).to('mps').eval()
    cache_cpu = cache_gpu = None
    checks = []
    with torch.inference_mode():
        for tokens in ([5, 8, 17, 23], [12], [27]):
            ids = torch.tensor([tokens])
            c = cpu(ids, past_key_values=cache_cpu, use_cache=True)
            g = gpu(ids.to('mps'), past_key_values=cache_gpu, use_cache=True)
            cache_cpu, cache_gpu = c.past_key_values, g.past_key_values
            actual, expected = g.last_hidden_state.cpu(), c.last_hidden_state
            diff = (actual - expected).abs()
            relative_rms = ((actual-expected).norm() / expected.norm()).item()
            check = dict(tokens=len(tokens), max_abs=diff.max().item(), relative_rms=relative_rms)
            checks.append(check)
            print(json.dumps(check), flush=True)
            torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
    with torch.inference_mode():
        ids = torch.tensor([[5, 8, 17, 23, 12, 27, 33]], device='mps')
        expected = gpu(ids, use_cache=True)
        actual = layer_major_prefill(gpu, ids, chunk_size=3)
        torch.testing.assert_close(actual.last_hidden_state, expected.last_hidden_state, atol=1e-4, rtol=1e-4)
        next_id = torch.tensor([[44]], device='mps')
        after_expected = gpu(next_id, past_key_values=expected.past_key_values, use_cache=True)
        after_actual = gpu(next_id, past_key_values=actual.past_key_values, use_cache=True)
        torch.testing.assert_close(after_actual.last_hidden_state, after_expected.last_hidden_state, atol=1e-4, rtol=1e-4)
        print(json.dumps(dict(layer_major_prefill='passed', chunks=[3, 3, 1], cached_decode='passed')), flush=True)
        for chunk_size in (5, 16):
            ids = torch.arange(5, 42, device='mps')[None]
            layered = layer_major_prefill(gpu, ids, chunk_size=chunk_size)
            cache = None
            pieces = []
            for start in range(0, ids.shape[1], chunk_size):
                standard = gpu(ids[:, start:start + chunk_size], past_key_values=cache, use_cache=True)
                cache = standard.past_key_values
                pieces.append(standard.last_hidden_state)
            torch.testing.assert_close(layered.last_hidden_state, torch.cat(pieces, dim=1), atol=0, rtol=0)
            for token in (44, 45, 46):
                next_id = torch.tensor([[token]], device='mps')
                standard = gpu(next_id, past_key_values=cache, use_cache=True)
                layered = gpu(next_id, past_key_values=layered.past_key_values, use_cache=True)
                cache = standard.past_key_values
                torch.testing.assert_close(layered.last_hidden_state, standard.last_hidden_state, atol=0, rtol=0)
            check = dict(layer_major_prefill='passed', prompt_tokens=37, chunk_size=chunk_size,
                         sparse_selection=True, cached_decode_tokens=3, reference='standard matching chunks', exact=True)
            checks.append(check)
            print(json.dumps(check), flush=True)
    report = dict(status='passed', scope='Random tiny GLM5Next CPU versus MPS, KDA and DSA, prefill and cached decode; not checkpoint inference', checks=checks)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
