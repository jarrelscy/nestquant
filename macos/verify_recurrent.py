"""Compare trained GLM KDA weights on PyTorch MPS and native MLX."""

import argparse
from contextlib import ExitStack
from functools import wraps
import json
from pathlib import Path
import statistics
import time
from unittest.mock import patch

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
import numpy as np
from safetensors import safe_open
import torch
from transformers.cache_utils import DynamicCache
from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextLinearAttention
from transformers.models.glm5_next import modeling_glm5_next

from backbone import parameter_sources, text_config
from mlx_vlm.models.cache import ArraysCache
from mlx_vlm.models.glm5_next.config import TextConfig
from mlx_vlm.models.glm5_next.language import Glm5NextLinearAttention


def read_checkpoint_tensors(model, names):
    index = json.loads((model / 'model.safetensors.index.json').read_text())['weight_map']
    with ExitStack() as stack:
        handles = {}

        def read(name):
            shard = index[name]
            if shard not in handles:
                handles[shard] = stack.enter_context(safe_open(model / shard, framework='pt', device='cpu', backend='pread'))
            tensor = handles[shard].get_tensor(name)
            if tensor.dtype == torch.float8_e4m3fn:
                scales = read(name.removesuffix('.weight') + '.weight_scale_inv')
                scales = scales.repeat_interleave(128, 0).repeat_interleave(128, 1)
                return tensor.float() * scales[:tensor.shape[0], :tensor.shape[1]]
            return tensor.float()

        return {name: read(name) for name in names}


def weights(model, config):
    prefix = 'layers.0.self_attn.'
    mapping = {name.removeprefix(prefix): sources for name, sources in parameter_sources(model, config).items()
               if name.startswith(prefix)}
    source = read_checkpoint_tensors(model, [name for sources in mapping.values() for name in sources])
    return {name: source[sources[0]] if len(sources) == 1 else torch.cat([source[key] for key in sources], dim=0)
            for name, sources in mapping.items()}


def make_mlx(config, source, bits=None):
    model = Glm5NextLinearAttention(config).eval()

    def array(tensor, dtype=mx.bfloat16):
        return mx.array(tensor.numpy()).astype(dtype)

    mapping = {
        'qkv_proj.weight': torch.cat([source[f'{name}_proj.weight'] for name in ('q', 'k', 'v')]),
        'fbg_a_proj.weight': torch.cat([source[name] for name in
                                      ('forget_gate.f_a_proj.weight', 'b_proj.weight', 'g_a_proj.weight')]),
        'qkv_conv.conv.weight': source['conv1d.weight'].transpose(1, 2),
        **{name: source[name] for name in ('g_b_proj.weight', 'o_norm.weight', 'o_proj.weight')},
        **{name: source[f'forget_gate.{name}'] for name in ('f_b_proj.weight', 'A_log', 'dt_bias')},
    }
    model.load_weights([(name, array(tensor, mx.float32 if name in ('A_log', 'dt_bias') else mx.bfloat16))
                        for name, tensor in mapping.items()])
    if bits is not None:
        nn.quantize(model, group_size=64, bits=bits)
    mx.eval(model.parameters())
    return model


def compare(call, sync, prefix, rows):
    sync(call(prefix))
    times, values = [], []
    for row in rows:
        start = time.perf_counter()
        value = call(row)
        sync(value)
        times.append(time.perf_counter() - start)
        values.append(value)
    return dict(median_ms=1000 * statistics.median(times), samples_ms=[1000 * t for t in times]), values


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--profile-components', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    mx.set_cache_limit(256 * 1024**2)
    config = text_config(args.model)
    raw = json.loads((args.model / 'config.json').read_text())['text_config']
    mlx_config = TextConfig.from_dict(raw)
    source = weights(args.model, config)
    with torch.device('meta'):
        reference = Glm5NextTextLinearAttention(config, 0).eval()
    reference.load_state_dict({
        name: tensor.to('mps', dtype=torch.float32 if name in
                        ('conv1d.weight', 'forget_gate.A_log', 'forget_gate.dt_bias') else torch.bfloat16)
        for name, tensor in source.items()
    }, assign=True)
    inputs = torch.randn(1, 36, config.hidden_size).to(torch.bfloat16)
    cache = DynamicCache(config=config)
    timing, values = compare(lambda x: reference(x, cache_params=cache), lambda _: torch.mps.synchronize(),
                             inputs[:, :4].to('mps'), [inputs[:, i:i + 1].to('mps') for i in range(4, 36)])
    expected = torch.cat(values, dim=1).float().cpu().numpy()
    results = {'torch_mps': timing}
    if args.profile_components:
        component_times = {name: [] for name in ('causal_conv1d_update', 'recurrent_kimi_delta_attention')}

        def timed_component(name):
            original = getattr(modeling_glm5_next, name)

            @wraps(original)
            def run(*args, **kwargs):
                torch.mps.synchronize()
                start = time.perf_counter()
                value = original(*args, **kwargs)
                torch.mps.synchronize()
                component_times[name].append((time.perf_counter() - start) * 1000)
                return value

            return run

        cache = DynamicCache(config=config)
        with torch.inference_mode():
            reference(inputs[:, :4].to('mps'), cache_params=cache)
            torch.mps.synchronize()
            with ExitStack() as stack:
                for name in component_times:
                    stack.enter_context(patch.object(modeling_glm5_next, name, timed_component(name)))
                for position in range(4, 36):
                    reference(inputs[:, position:position + 1].to('mps'), cache_params=cache)
        if any(len(samples) != 32 for samples in component_times.values()):
            raise RuntimeError('Did not observe 32 recurrent fallback calls per component')
        results['torch_mps_synchronized_components'] = {
            name: dict(median_ms=statistics.median(samples[4:]), samples_ms=samples)
            for name, samples in component_times.items()}
    vectors = mx.array(inputs.float().numpy()).astype(mx.bfloat16)
    for bits in (None, 8, 4):
        model = make_mlx(mlx_config, source, bits)
        cache = ArraysCache(2)
        timing, values = compare(lambda x: model(x, cache=cache), mx.eval, vectors[:, :4],
                                 [vectors[:, i:i + 1] for i in range(4, 36)])
        actual = np.asarray(mx.concatenate(values, axis=1).astype(mx.float32))
        timing['relative_rms'] = float(np.linalg.norm(expected - actual) / np.linalg.norm(expected))
        timing['cosine'] = float(np.vdot(expected, actual) / (np.linalg.norm(expected) * np.linalg.norm(actual)))
        timing['parameter_bytes'] = sum(v.nbytes for _, v in tree_flatten(model.parameters()))
        results['mlx_bf16' if bits is None else f'mlx_affine_{bits}'] = timing
    with args.output.open('x') as handle:
        json.dump(dict(scope='Layer zero attention only; four prefix positions and 32 fixed recurrent positions', results=results), handle, indent=2)
        handle.write('\n')


if __name__ == '__main__':
    main()
