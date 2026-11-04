import json
from pathlib import Path

import torch
from safetensors import safe_open
from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig
from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextModel

PREFIX = 'model.language_model.'
RENAME = {
    'hc_attn_fn': 'attn_hc.fn', 'hc_attn_base': 'attn_hc.base', 'hc_attn_scale': 'attn_hc.scale',
    'hc_ffn_fn': 'ffn_hc.fn', 'hc_ffn_base': 'ffn_hc.base', 'hc_ffn_scale': 'ffn_hc.scale',
    'self_attn.f_a_proj.': 'self_attn.forget_gate.f_a_proj.',
    'self_attn.f_b_proj.': 'self_attn.forget_gate.f_b_proj.',
    'self_attn.dt_bias': 'self_attn.forget_gate.dt_bias',
    'self_attn.A_log': 'self_attn.forget_gate.A_log',
}


def text_config(root: Path) -> Glm5NextTextConfig:
    raw = json.loads((root / 'config.json').read_text())['text_config']
    raw['layer_types'] = ['indexed_attention' if t == 'deepseek_sparse_attention' else t for t in raw['layer_types']]
    config = Glm5NextTextConfig(**raw)
    config._attn_implementation = 'eager'
    return config


def parameter_sources(root: Path, config: Glm5NextTextConfig) -> dict[str, list[str]]:
    index = json.loads((root / 'model.safetensors.index.json').read_text())['weight_map']
    result = {}
    convolutions = {}
    for source in index:
        if not source.startswith(PREFIX) or source.endswith('.weight_scale_inv'):
            continue
        target = source.removeprefix(PREFIX)
        if target.startswith('layers.') and int(target.split('.')[1]) >= config.num_hidden_layers:
            continue
        for before, after in RENAME.items():
            target = target.replace(before, after)
        if any(target.endswith(f'self_attn.{p}_conv1d.weight') for p in ('q', 'k', 'v')):
            base = target.rsplit('.', 2)[0]
            convolutions.setdefault(base, []).append(source)
        else:
            if target in result:
                raise ValueError(f'Duplicate mapped parameter: {target}')
            result[target] = [source]
    for base, names in convolutions.items():
        ordered = [next(n for n in names if n.endswith(f'.{p}_conv1d.weight')) for p in ('q', 'k', 'v')]
        result[base + '.conv1d.weight'] = ordered
    return result


def empty_model(root: Path, experts_factory):
    config = text_config(root)
    with torch.device('meta'):
        model = Glm5NextTextModel(config)
    for layer in range(config.first_k_dense_replace, config.num_hidden_layers):
        model.layers[layer].mlp.experts = experts_factory(layer)
    return model.eval()


def load_backbone(root: Path, model, device: str = 'mps', dtype=torch.bfloat16):
    index = json.loads((root / 'model.safetensors.index.json').read_text())['weight_map']
    sources = parameter_sources(root, model.config)
    expected = model.state_dict()
    missing = sorted(set(expected) - set(sources))
    extra = sorted(set(sources) - set(expected))
    if missing or extra:
        raise ValueError(f'Backbone mapping mismatch: missing={missing}, unexpected={extra}')
    handles = {}

    def read(name):
        file = index[name]
        if file not in handles:
            handles[file] = safe_open(root / file, framework='pt', device='cpu')
        handle = handles[file]
        tensor = handle.get_tensor(name)
        scale_name = name.removesuffix('.weight') + '.weight_scale_inv'
        if tensor.dtype == torch.float8_e4m3fn:
            if scale_name not in index:
                raise ValueError(f'Missing FP8 scale: {name}')
            scale = read(scale_name).float()
            if tensor.ndim != 2:
                raise ValueError(f'Unsupported FP8 tensor shape: {name}')
            rows, cols = tensor.shape
            expanded = scale.repeat_interleave(128, 0).repeat_interleave(128, 1)[:rows, :cols]
            tensor = tensor.float() * expanded
        return tensor

    try:
        for position, (target, names) in enumerate(sources.items()):
            value = read(names[0]) if len(names) == 1 else torch.cat([read(n) for n in names], dim=0)
            if value.shape != expected[target].shape:
                raise ValueError(f'Shape mismatch {target}: {tuple(value.shape)} != {tuple(expected[target].shape)}')
            target_dtype = torch.float32 if any(s in target for s in ('.gate.', '_hc.', '.conv1d.', '.forget_gate.A_log', '.forget_gate.dt_bias')) else dtype
            value = value.to(device=device, dtype=target_dtype)
            model.load_state_dict({target: value}, strict=False, assign=True)
            if position % 100 == 0:
                print(json.dumps(dict(event='backbone_load', loaded=position + 1, total=len(sources))), flush=True)
        head = read('lm_head.weight').to(device=device, dtype=dtype)
    finally:
        for handle in handles.values():
            handle.__exit__(None, None, None)
    remaining = [name for name, value in model.state_dict().items() if value.device.type == 'meta']
    if remaining:
        raise ValueError(f'Unloaded backbone tensors: {remaining}')
    return head
