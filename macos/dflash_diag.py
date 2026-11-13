"""Offline DFlash2 acceptance diagnostics.

`collect` runs plain greedy decode on the full runtime and stores, for every position, the target's tokens
and the four hyper-connection streams of the outputs of layers i-1 and i for each drafter target layer i.
`evaluate` then loads only the drafter (plus an embedding and output head) and measures how well it
predicts the target's own greedy continuation under several capture conventions, so integration choices
can be compared without the target in memory.
"""

import argparse
import json
from pathlib import Path
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np

PROMPTS = [
    'Explain how a hash map works, in detail.',
    'Write a Python function that reads a CSV file and returns the mean of every numeric column, with tests.',
    'Summarise the causes of the First World War in five paragraphs.',
]


def collect(args):
    import torch
    import native_dflash as nd
    import native_generate as ng
    from metrics import MemoryGuard
    from native_refinements import NativeHotRefinements
    from transformers import AutoTokenizer

    nd._install_capture()
    nd._collapse = lambda h: h   # keep all hyper-connection streams
    mx.set_cache_limit(256 * 1024**2)
    mx.set_memory_limit(96 * 1024**3)
    mx.set_wired_limit(92 * 1024**3)
    torch.set_num_threads(4)
    guard = MemoryGuard(98, 0.8)
    guard.allowed_pressure = (1, 2)
    hot = NativeHotRefinements(args.model, args.hot_fraction, 128, args.budget_gib, 4, 32, 512)
    model, banks = ng.load_runtime(args.model, args.backbone, args.archive, [288] * 42, guard, hot, True, 56)
    targets = json.loads((args.drafter / 'config.json').read_text())['dflash_config']['target_layer_ids']
    wanted = sorted({i for t in targets for i in (t - 1, t)})
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    template = (Path(ng.__file__).resolve().parents[1] / 'spark/flash/thinking_off.jinja').read_text()
    args.data.mkdir(parents=True, exist_ok=True)
    try:
        hot.warm(4, lambda: guard.check(force=True))
        for number, prompt in enumerate(PROMPTS[:args.prompts]):
            ids = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=True,
                                               return_dict=False, add_generation_prompt=True,
                                               chat_template=template)
            cache = model.make_cache()
            hot.reset()
            chunks = [ids[i:i + 8] for i in range(0, len(ids), 8)]
            logits, hidden = nd.layer_major_prefill_capture(model, chunks, cache, guard, hot, wanted)
            rows = [np.array(hidden[0].astype(mx.float32))]          # [N, 4, D * len(wanted)]
            tokens = list(ids) + [ng.greedy(logits)]
            margins = []
            for _ in range(args.tokens):
                token = tokens[-1]
                hot.begin_decode(token)
                sink = []
                model.model._nq_capture_request = (wanted, sink, lambda h: h)
                try:
                    logits = model(mx.array([[token]]), cache=cache).logits
                finally:
                    model.model._nq_capture_request = None
                step = mx.concatenate(sink, axis=-1)[0]
                mx.eval(logits, step, [entry.state for entry in cache])
                hot.commit([token])
                rows.append(np.array(step.astype(mx.float32)))
                top = mx.topk(logits[0, -1].astype(mx.float32), 2)
                margins.append(float(mx.abs(top[0] - top[1]).item()))
                tokens.append(ng.greedy(logits))
                guard.check()
            hidden = np.concatenate(rows, axis=0)
            path = args.data / f'prompt{number}.npz'
            np.savez(path, hidden=hidden.astype(np.float32), tokens=np.array(tokens), prompt_tokens=len(ids),
                     wanted=np.array(wanted), targets=np.array(targets), margins=np.array(margins))
            print(json.dumps(dict(event='collected', prompt=number, prompt_tokens=len(ids), positions=hidden.shape[0],
                                  text=tokenizer.decode(tokens[len(ids):len(ids) + 40]))), flush=True)
    finally:
        hot.close()
        for bank in banks:
            bank.close()


def recapture(args):
    """Re-run each collected sequence through layer-major prefill, where every routed expert call uses its
    4-bit refinement, and store those captures and that target's greedy label at every position."""
    import torch
    import native_dflash as nd
    import native_generate as ng
    from metrics import MemoryGuard
    from native_refinements import NativeHotRefinements
    from mlx_vlm.models.glm5_next.language import create_ssm_mask

    mx.set_cache_limit(256 * 1024**2)
    mx.set_memory_limit(96 * 1024**3)
    mx.set_wired_limit(92 * 1024**3)
    torch.set_num_threads(4)
    guard = MemoryGuard(98, 0.8)
    guard.allowed_pressure = (1, 2)
    hot = NativeHotRefinements(args.model, args.hot_fraction, 128, args.budget_gib, 4, 32, 512)
    model, banks = ng.load_runtime(args.model, args.backbone, args.archive, [288] * 42, guard, hot, True, 56)
    try:
        for path in sorted(args.data.glob('prompt?.npz')):
            data = dict(np.load(path))
            wanted = data['wanted'].tolist()
            ids = data['tokens'][:data['hidden'].shape[0]].tolist()
            cache = model.make_cache()
            hot.reset()
            chunks = [ids[i:i + 8] for i in range(0, len(ids), 8)]
            hidden = [mx.repeat(model.model.embed_tokens(mx.array([chunk]))[:, :, None],
                                model.config.hc_mult, axis=2) for chunk in chunks]
            indices = [None] * len(chunks)
            captured = {}
            for number, (layer, state) in enumerate(zip(model.layers, cache, strict=True)):
                for i, value in enumerate(hidden):
                    hot.chunk_index = i
                    mask = create_ssm_mask(value[:, :, 0], state) if layer.block_type == 'linear_attention' else None
                    hidden[i], indices[i] = layer(value, mask, state, indices[i])
                    mx.eval(hidden[i], indices[i], state.state)
                    guard.check()
                if number in wanted:
                    captured[number] = np.array(mx.concatenate(hidden, axis=1)[0].astype(mx.float32))
            labels = []
            for value in hidden:
                logits = model.lm_head(model.model.norm(value.mean(axis=2)))
                labels.extend(np.array(logits[0].argmax(axis=-1)).tolist())
            for i, chunk in enumerate(chunks):
                hot.commit(chunk, i)
            out = dict(data)
            out['hidden'] = np.concatenate([captured[index] for index in wanted], axis=-1)
            out['labels'] = np.array(labels)
            np.savez(path.with_name(path.stem + '_4bit.npz'), **out)
            start = int(data['prompt_tokens'])
            agree = np.mean(np.array(labels[start - 1:]) == data['tokens'][start:len(labels) + 1])
            print(json.dumps(dict(event='recaptured', path=path.name, label_agreement_with_decode=float(agree))),
                  flush=True)
    finally:
        hot.close()
        for bank in banks:
            bank.close()


def _quantized_embedding(path):
    weights = mx.load(str(path))
    if 'scales' not in weights:
        module = nn.Embedding(*weights['weight'].shape)
        module.weight = weights['weight']
        return module
    rows, packed = weights['weight'].shape
    groups = weights['scales'].shape[1]
    dims = 4096
    bits = packed * 32 // dims
    module = nn.QuantizedEmbedding(rows, dims, group_size=dims // groups, bits=bits)
    module.update(dict(weight=weights['weight'], scales=weights['scales'], biases=weights['biases']))
    return module


def _quantized_head(path):
    weights = mx.load(str(path))
    rows, packed = weights['weight'].shape
    groups = weights['scales'].shape[1]
    dims = 4096
    bits = packed * 32 // dims
    module = nn.QuantizedLinear(dims, rows, bias=False, group_size=dims // groups, bits=bits)
    module.update(dict(weight=weights['weight'], scales=weights['scales'], biases=weights['biases']))
    return module


def _bf16_head(model_root):
    import torch
    from safetensors import safe_open
    index = json.loads((model_root / 'model.safetensors.index.json').read_text())['weight_map']
    with safe_open(str(model_root / index['lm_head.weight']), 'pt') as handle:
        weight = handle.get_tensor('lm_head.weight').to(torch.float32).numpy()
    module = nn.Linear(4096, weight.shape[0], bias=False)
    module.weight = mx.array(weight).astype(mx.bfloat16)
    return module


def _features(data, shift, streams):
    hidden = data['hidden']                    # [T, 4, D * len(wanted)]
    wanted = data['wanted'].tolist()
    dims = hidden.shape[-1] // len(wanted)
    parts = []
    for target in data['targets'].tolist():
        index = wanted.index(target + shift)
        block = hidden[:, :, index * dims:(index + 1) * dims]
        if streams == 'mean':
            parts.append(block.mean(axis=1))
        elif streams == 'sum':
            parts.append(block.sum(axis=1))
        else:
            parts.append(block[:, int(streams)])
    return mx.array(np.concatenate(parts, axis=-1)).astype(mx.bfloat16)


def _argmax(logits):
    return logits.argmax(axis=-1)


def _teacher_forced(drafter, features, tokens, start, depth):
    """Draft at every decode position from the full context; return per-position matches."""
    matches = []
    for t in range(start, len(tokens) - depth):
        cache = drafter.make_cache()
        drafts = drafter.draft_block(int(tokens[t]), features[None, :t], cache, depth + 1, _argmax)
        mx.eval(drafts)
        matches.append(np.array(drafts[0]) == np.asarray(tokens[t + 1:t + 1 + depth]))
    return np.array(matches)


def _first_draft(drafter, features, tokens, t):
    drafts = drafter.draft_block(int(tokens[t]), features[None, :t], drafter.make_cache(), 8, _argmax)
    return int(drafts[0, 0].item())


def _simulated(drafter, features, tokens, start, block, draft_block=None):
    """mlx-vlm's incremental loop against the target's greedy continuation (a fixed block every round)."""
    cache = drafter.make_cache()
    context = features[None, :start]
    t = start
    accepted = []
    while t + block <= len(tokens):
        drafts = np.array(drafter.draft_block(int(tokens[t]), context, cache, draft_block or block, _argmax)[0])
        drafts = drafts[:block - 1]
        truth = np.asarray(tokens[t + 1:t + block])
        count = int(np.cumprod(drafts == truth).sum())
        accepted.append(count)
        context = features[None, t:t + count + 1]
        t += count + 1
    return accepted


def evaluate(args):
    from mlx_vlm.speculative.drafters import load_drafter
    drafter, _ = load_drafter(str(args.drafter))
    mx.eval(drafter.parameters())
    embeds = dict(e8=lambda: _quantized_embedding(args.backbone / 'embed.safetensors'),
                  bf16=lambda: _quantized_embedding(args.bf16_backbone / 'embed.safetensors'))
    heads = dict(h4=lambda: _quantized_head(args.backbone / 'head.safetensors'),
                 h8=lambda: _quantized_head(args.bf16_backbone / 'head.safetensors'),
                 bf16=lambda: _bf16_head(args.model))
    decode = [dict(np.load(path)) for path in sorted(args.data.glob('prompt?.npz'))]
    full = [dict(np.load(path)) for path in sorted(args.data.glob('prompt?_4bit.npz'))]
    loaded = {}
    results = []
    for variant in args.variants.split(','):
        shift, streams, embed, head, source = (variant.split(':') + ['decode'])[:5]
        for key, table in (('e:' + embed, embeds), ('h:' + head, heads)):
            if key not in loaded:
                loaded[key] = table[key[2:]]()
                mx.eval(loaded[key].parameters())
        drafter.embed_tokens = loaded['e:' + embed]
        drafter.embed_scale = 1.0
        drafter.lm_head = loaded['h:' + head]
        started = time.perf_counter()
        forced, simulated, own_first, agree, per_prompt = [], {block: [] for block in (3, 4, 6, 8)}, [], [], []
        for number, data in enumerate(decode):
            features = _features(full[number] if source == '4bit' else data, int(shift), streams)
            if source == 'zero':
                features = mx.zeros_like(features)
            tokens = data['tokens']
            start = int(data['prompt_tokens'])
            matches = _teacher_forced(drafter, features, tokens, start, 7)
            forced.append(matches)
            for block in (3, 4, 6, 8):
                simulated[block].extend(_simulated(drafter, features, tokens, start, block))
            for block in (3, 4, 6):
                simulated.setdefault(f'{block}_draft8', []).extend(
                    _simulated(drafter, features, tokens, start, block, 8))
            per_prompt.append(round(float(np.minimum(np.cumprod(matches, axis=1).sum(axis=1), 3).mean()), 3))
            if full:
                # Positions where the all-4-bit target would itself produce the next 7 decode tokens.
                labels = full[number]['labels']
                window = np.array([np.array_equal(labels[t:t + 7], tokens[t + 1:t + 8])
                                   for t in range(start, len(tokens) - 7)])
                agree.append(window)
                if source == '4bit':
                    # First draft against the all-4-bit target's own greedy label at the same position.
                    own_first.append(np.array(
                        [_first_draft(drafter, features, tokens, t) == labels[t]
                         for t in range(start, len(tokens) - 7)]))
        forced = np.concatenate(forced)
        prefix = np.cumprod(forced, axis=1)
        entry = dict(variant=variant, positions=len(forced),
                     unconditional_match=[round(float(v), 3) for v in forced.mean(axis=0)],
                     prefix_accept=[round(float(v), 3) for v in prefix.mean(axis=0)],
                     tf_mean_accepted={k: round(float(np.minimum(prefix.sum(axis=1), k).mean()), 3)
                                       for k in (3, 5, 7)},
                     sim_mean_accepted={block: round(float(np.mean(v)), 3) for block, v in simulated.items()},
                     per_prompt_tf3=per_prompt)
        if agree:
            agree = np.concatenate(agree)
            entry['agree_positions'] = int(agree.sum())
            entry['agree_tf_mean_accepted'] = {k: round(float(np.minimum(prefix[agree].sum(axis=1), k).mean()), 3)
                                               for k in (3, 7)}
            entry['agree_prefix_accept'] = [round(float(v), 3) for v in prefix[agree].mean(axis=0)]
        if own_first:
            entry['first_draft_vs_4bit_labels'] = round(float(np.concatenate(own_first).mean()), 3)
        entry['seconds'] = round(time.perf_counter() - started, 1)
        results.append(entry)
        print(json.dumps(entry), flush=True)
    if args.output:
        args.output.write_text(json.dumps(results, indent=1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('collect', 'recapture', 'evaluate'))
    parser.add_argument('--model', type=Path, default=Path.home() / 'Models/GLM/GLM-5.3-Flash-NestQuant-1.5-4bit')
    parser.add_argument('--archive', type=Path, default=Path.home() / 'Models/GLM/resident-banks-v1')
    parser.add_argument('--backbone', type=Path, default=Path.home() / 'Models/GLM/backbone-affine4-e8-v1')
    parser.add_argument('--bf16-backbone', type=Path, default=Path.home() / 'Models/GLM/backbone-affine8-v1')
    parser.add_argument('--drafter', type=Path, default=Path.home() / 'Models/GLM/drafters/dflash2')
    parser.add_argument('--data', type=Path, default=Path('/tmp/dfc_data'))
    parser.add_argument('--hot-fraction', type=float, default=0.15)
    parser.add_argument('--budget-gib', type=int, default=16)
    parser.add_argument('--tokens', type=int, default=128)
    parser.add_argument('--prompts', type=int, default=3)
    parser.add_argument('--variants', default='0:mean:e8:h4')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    dict(collect=collect, recapture=recapture, evaluate=evaluate)[args.mode](args)


if __name__ == '__main__':
    main()
