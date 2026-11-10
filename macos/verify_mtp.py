"""Test the checkpoint's actual draft head with bounded native NestQuant banks."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import mlx.core as mx
from transformers import AutoTokenizer

from checkpoint import validate
from metrics import MemoryGuard
from native_generate import greedy, load_runtime, memory_sample, stats, diff
from mlx_vlm.speculative.drafters.glm5_next_mtp.config import Glm5NextMTPConfig
from mlx_vlm.speculative.drafters.glm5_next_mtp.glm5_next_mtp import Glm5NextMTPDraftModel
from mlx_vlm.speculative.drafters.glm5_next_mtp.split import Glm5NextMTPSplitter
from mlx_vlm.speculative.mtp import _mtp_rounds
from mlx_vlm.utils import load_model_weights


MTP_SHA256 = {
    'mtp_experts-00001-of-00002.safetensors': '32defe99506216943b542aa33078832c7741e564cd1cadc8ec0b5a664560f207',
    'mtp_experts-00002-of-00002.safetensors': '28181b15ba7b4760a49c2aed6d8738798f937a92dc39e3560acc0708f45200dc',
    'nonexpert-00003-of-00004.safetensors': 'ec23074b4c30a919ad32072c4b5e640a49133b03b65f41610e39a117609139a1',
}


def verify_draft(root):
    import fcntl

    index = json.loads((root / 'model.safetensors.index.json').read_text())['weight_map']
    files = {name for key, name in index.items() if key.startswith('model.language_model.layers.45.')}
    if files != set(MTP_SHA256):
        raise ValueError('Unexpected MTP shard layout')
    for name, expected in MTP_SHA256.items():
        with (root / name).open('rb') as handle:
            fcntl.fcntl(handle.fileno(), fcntl.F_NOCACHE, 1)
            if hashlib.file_digest(handle, 'sha256').hexdigest() != expected:
                raise ValueError(f'Unexpected MTP checksum: {name}')


def run(args, guard):
    root = args.model
    validate(root)
    if args.mtp:
        verify_draft(root)
    tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True, trust_remote_code=False)
    template = (Path(__file__).resolve().parents[1] / 'spark/flash/thinking_off.jinja').read_text()
    ids = tokenizer.apply_chat_template([dict(role='user', content=args.prompt)], tokenize=True,
                                        return_dict=False, add_generation_prompt=True, chat_template=template)
    if not ids or len(ids) + args.tokens > 4096:
        raise ValueError('Rendered prompt and output exceed 4096 tokens')
    model, banks = load_runtime(root, args.backbone, args.archive,
                                [args.rows] * 42, guard, None)
    try:
        drafter = None
        draft_seconds = None
        if args.mtp:
            started = time.perf_counter()
            config, weights = Glm5NextMTPSplitter().prepare(root, block_size=args.mtp + 1, q_bits=4, q_group_size=64)
            draft = Glm5NextMTPDraftModel(Glm5NextMTPConfig.from_dict(config))
            drafter = load_model_weights(draft, config, weights, root)
            del weights
            mx.eval(drafter.parameters())
            guard.check(force=True)
            draft_seconds = time.perf_counter() - started
            print(json.dumps(dict(event='draft_loaded', seconds=draft_seconds,
                                  quantization=config.get('quantization'))), flush=True)
        cache = model.make_cache()
        hidden = []
        start = time.perf_counter()
        for index in range(0, len(ids), 8):
            out = model(mx.array([ids[index:index+8]]), cache=cache, num_logits_to_keep=1, return_hidden=True)
            mx.eval(out.logits, out.hidden_states[-1], [entry.state for entry in cache])
            hidden.append(out.hidden_states[-1])
            guard.check()
        token = greedy(out.logits)
        ttft = time.perf_counter() - start
        before = stats(model)
        generated = [token]
        end_ids = json.loads((root / 'generation_config.json').read_text())['eos_token_id']
        eos = {end_ids} if isinstance(end_ids, int) else set(end_ids)
        start = time.perf_counter()
        if drafter is None:
            while len(generated) < args.tokens:
                out = model(mx.array([[generated[-1]]]), cache=cache)
                mx.eval(out.logits, [entry.state for entry in cache])
                generated.append(greedy(out.logits))
                guard.check()
                if generated[-1] in eos:
                    break
        else:
            rounds = _mtp_rounds(model, drafter, cache, mx.concatenate(hidden, axis=1), {},
                                 prompt_tokens=mx.array([ids]), first_bonus=token, max_tokens=args.tokens,
                                 sampler=lambda logits: logits.argmax(axis=-1),
                                 draft_block_size=args.mtp + 1, greedy_sampling=True)
            for token, _ in rounds:
                generated.append(int(token))
                guard.check()
                if generated[-1] in eos:
                    rounds.close()
                    break
        mx.synchronize()
        decode = time.perf_counter() - start
        return dict(prompt_tokens=len(ids), generated_ids=generated, ttft_seconds=ttft, decode_seconds=decode,
                    draft_setup_seconds=draft_seconds, decode_tps=(len(generated)-1)/decode,
                    draft_depth=args.mtp, accept_lengths=drafter.accept_lens if drafter is not None else [],
                    proposal_lengths=drafter.draft_lens if drafter is not None else [],
                    memory=memory_sample(), expert_stats=diff(stats(model), before))
    finally:
        for bank in banks:
            bank.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--backbone', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mtp', type=int, choices=(0, 1, 2), required=True)
    p.add_argument('--reference', type=Path)
    p.add_argument('--rows', type=int, choices=(96, 160), default=160)
    p.add_argument('--tokens', type=int, choices=range(8, 65), default=24)
    p.add_argument('--prompt', default='Explain briefly how a database B-tree handles a page split.')
    a = p.parse_args()
    if a.output.exists():
        p.error('Use a fresh output file')
    mx.set_cache_limit(256 * 1024**2)
    mx.set_memory_limit(76 * 1024**3)
    result = run(a, MemoryGuard(82))
    if a.reference is not None:
        reference = json.loads(a.reference.read_text())
        result['serial_tokens_identical'] = result['generated_ids'] == reference['generated_ids']
    with a.output.open('x') as file:
        json.dump(result, file, indent=2)
        file.write('\n')
    print(json.dumps(result), flush=True)
    if result.get('serial_tokens_identical') is False:
        raise RuntimeError('MTP output tokens differ from the serial reference')
