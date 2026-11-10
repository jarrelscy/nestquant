"""Prepare a compact backbone and load it without Torch conversion."""

import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_vlm.models.glm5_next import language
from mlx_vlm.models.glm5_next.config import TextConfig

from native_assets import REVISION, fingerprint


PREFIX = 'model.language_model.'


class RoutedPlaceholder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate = language.MoEGate(config)
        self.shared_experts = language.Glm5NextMLP(
            config, intermediate_size=config.moe_intermediate_size * config.n_shared_experts)

    def __call__(self, x):
        raise RuntimeError('The native routed expert bank is not installed')


def _stage(model, root, index, names):
    import torch
    from safetensors import safe_open

    with ExitStack() as stack:
        handles = {}

        def read(name):
            shard = index[name]
            if shard not in handles:
                handles[shard] = stack.enter_context(safe_open(root / shard, framework='pt', backend='pread'))
            tensor = handles[shard].get_tensor(name)
            if tensor.dtype == torch.float8_e4m3fn:
                scales = read(name.removesuffix('.weight') + '.weight_scale_inv')
                scales = scales.repeat_interleave(128, 0).repeat_interleave(128, 1)
                return tensor.float() * scales[:tensor.shape[0], :tensor.shape[1]]
            return tensor.float()

        weights = {}
        for name in names:
            key = name.removeprefix(PREFIX)
            protected = any(part in key for part in (
                '.gate.', 'hc_attn_', 'hc_ffn_', '.A_log', '.dt_bias', 'index_kpool_compress'))
            target = name if name == 'lm_head.weight' else 'language_model.model.' + key
            weights[target] = mx.array(read(name).numpy()).astype(mx.float32 if protected else mx.bfloat16)
    with patch.object(language, 'Glm5NextMoE', RoutedPlaceholder):
        sanitized = model.sanitize(weights)
    return {key.removeprefix('language_model.'): value for key, value in sanitized.items()}


def validate_quantization(bits, embedding_bits):
    if type(bits) is not int or bits not in (4, 8) or type(embedding_bits) is not int or embedding_bits not in (0, 8):
        raise ValueError('Use four- or eight-bit linears and either original or eight-bit embeddings')


def prepare(root, directory, bits=8, embedding_bits=0):
    from checkpoint import validate
    from metrics import MemoryGuard

    validate_quantization(bits, embedding_bits)
    validate(root)
    guard = MemoryGuard(26)
    directory.mkdir()
    source = (root / 'config.json').read_bytes()
    config = TextConfig.from_dict(json.loads(source)['text_config'])
    layers = config.num_hidden_layers
    config.num_hidden_layers = 0
    model = language.LanguageModel(config).eval()
    config.num_hidden_layers = layers
    index = json.loads((root / 'model.safetensors.index.json').read_text())['weight_map']
    files = {}

    def save(name, module):
        params = dict(tree_flatten(module.parameters()))
        mx.eval(params)
        guard.check(force=True)
        file = directory / f'{name}.safetensors'
        mx.save_safetensors(file, params)
        files[name] = dict(state=fingerprint(file), tensor_bytes=sum(x.nbytes for x in params.values()))

    initial = _stage(model, root, index, ['lm_head.weight', PREFIX + 'embed_tokens.weight', PREFIX + 'norm.weight'])
    model.lm_head.load_weights([('weight', initial['lm_head.weight'])])
    model.model.embed_tokens.load_weights([('weight', initial['model.embed_tokens.weight'])])
    model.model.norm.load_weights([('weight', initial['model.norm.weight'])])
    del initial
    if embedding_bits:
        model.model.embed_tokens = model.model.embed_tokens.to_quantized(group_size=64, bits=embedding_bits)
    save('embed', model.model.embed_tokens)
    save('norm', model.model.norm)
    model.lm_head = model.lm_head.to_quantized(group_size=64, bits=bits)
    save('head', model.lm_head)
    for layer_id in range(layers):
        with patch.object(language, 'Glm5NextMoE', RoutedPlaceholder):
            layer = language.DecoderLayer(config, layer_id).eval()
        model.layers.append(layer)
        prefix = f'model.layers.{layer_id}.'
        names = [name for name in index if name.startswith(f'{PREFIX}layers.{layer_id}.')
                 and not name.endswith('.weight_scale_inv')]
        loaded = _stage(model, root, index, names)
        layer.load_weights([(name.removeprefix(prefix), value) for name, value in loaded.items()], strict=True)
        del loaded
        nn.quantize(layer, group_size=64, bits=bits,
                    class_predicate=lambda _, item: isinstance(item, nn.Linear) and item.weight.dtype != mx.float32)
        save(f'L{layer_id}', layer)
    info = dict(revision=REVISION, config_sha256=hashlib.sha256(source).hexdigest(), files=files,
                quantization=dict(bits=bits, embedding_bits=embedding_bits))
    with (directory / 'manifest.json').open('x') as handle:
        json.dump(info, handle, indent=2)
        handle.write('\n')
    print(json.dumps(dict(event='backbone_prepared', tensor_bytes=sum(row['tensor_bytes'] for row in files.values()))),
          flush=True)


def load_prepared(root, directory, event):
    source = (root / 'config.json').read_bytes()
    info = json.loads((directory / 'manifest.json').read_text())
    if info['revision'] != REVISION or info['config_sha256'] != hashlib.sha256(source).hexdigest():
        raise ValueError('Prepared backbone does not match this model')
    quantization = info.get('quantization', dict(bits=8, embedding_bits=0))
    if not isinstance(quantization, dict) or set(quantization) != {'bits', 'embedding_bits'}:
        raise ValueError('Unexpected prepared backbone quantization metadata')
    bits, embedding_bits = quantization['bits'], quantization['embedding_bits']
    validate_quantization(bits, embedding_bits)
    config = TextConfig.from_dict(json.loads(source)['text_config'])
    layers = config.num_hidden_layers
    config.num_hidden_layers = 0
    model = language.LanguageModel(config).eval()
    config.num_hidden_layers = layers

    def load_module(name, module):
        file = directory / f'{name}.safetensors'
        expected = info['files'][name]
        if fingerprint(file) != expected['state']:
            raise ValueError(f'Prepared backbone shard changed: {name}')
        arrays = mx.load(file, format='safetensors')
        if sum(array.nbytes for array in arrays.values()) != expected['tensor_bytes']:
            raise ValueError(f'Prepared backbone size differs: {name}')
        if isinstance(module, nn.Embedding) and ('scales' in arrays) != bool(embedding_bits):
            raise ValueError('Prepared embedding tensors disagree with quantization metadata')
        if isinstance(module, (nn.Linear, nn.Embedding)) and 'scales' in arrays:
            module = module.to_quantized(group_size=64, bits=embedding_bits if isinstance(module, nn.Embedding) else bits)
        else:
            nn.quantize(module, group_size=64, bits=bits,
                        class_predicate=lambda path, item: isinstance(item, nn.Linear) and path + '.scales' in arrays)
        module.load_weights(list(arrays.items()), strict=True)
        mx.eval(module.parameters())
        if fingerprint(file) != expected['state']:
            raise ValueError(f'Prepared backbone shard changed during load: {name}')
        event(name, expected['tensor_bytes'])
        return module

    model.model.embed_tokens = load_module('embed', model.model.embed_tokens)
    model.model.norm = load_module('norm', model.model.norm)
    model.lm_head = load_module('head', model.lm_head)
    for layer in range(layers):
        with patch.object(language, 'Glm5NextMoE', RoutedPlaceholder):
            module = language.DecoderLayer(config, layer).eval()
        model.layers.append(load_module(f'L{layer}', module))
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--bits', type=int, choices=(4, 8), default=8)
    parser.add_argument('--embedding-bits', type=int, choices=(0, 8), default=0)
    args = parser.parse_args()
    mx.set_memory_limit(26 * 1024**3)
    mx.set_cache_limit(256 * 1024**2)
    prepare(args.model, args.output_dir, args.bits, args.embedding_bits)
