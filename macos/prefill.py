from types import SimpleNamespace

import torch
from transformers.cache_utils import DynamicCache


@torch.inference_mode()
def layer_major_prefill(model, input_ids, chunk_size: int, on_chunk=None):
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
        raise ValueError('Layer-major prefill requires one nonempty token sequence')
    if chunk_size < 1:
        raise ValueError('Prefill chunk size must be positive')
    if any(kind != 'full' for kind in model.config.indexer_types):
        raise ValueError('Shared attention indexers are not supported by this prefill path')
    cache = DynamicCache(config=model.config)
    embedded = model.embed_tokens(input_ids)
    hidden = embedded.unsqueeze(2).expand(-1, -1, model.config.hc_mult, -1).contiguous()
    length = input_ids.shape[1]
    for layer_index, layer in enumerate(model.layers):
        for start in range(0, length, chunk_size):
            end = min(start + chunk_size, length)
            mask = torch.ones((1, end - start), dtype=torch.bool, device=hidden.device)
            positions = torch.arange(start, end, device=hidden.device)[None]
            updated, _ = layer(
                hidden[:, start:end], attention_mask=mask, position_ids=positions,
                past_key_values=cache, use_cache=True, position_embeddings=None,
                input_ids=input_ids[:, start:end], prev_topk_indices=None,
            )
            hidden[:, start:end] = updated
            if on_chunk is not None:
                on_chunk(layer_index, start, end)
    output = torch.cat([model.norm(model.hc_head(hidden[:, start:start + chunk_size]))
                        for start in range(0, length, chunk_size)], dim=1)
    return SimpleNamespace(last_hidden_state=output, past_key_values=cache)
