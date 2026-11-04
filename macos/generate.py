import argparse
import json
import os
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
from transformers import AutoTokenizer

from backbone import empty_model, load_backbone
from experts import ExpertStore, NestQuantExperts
from predictor import staged_predictor, configure_slots
from prefill import layer_major_prefill
from metrics import snapshot
from checkpoint import validate


class Runtime:
    def __init__(self, root: Path, cache_gib: float = 4, device: str = 'mps', hot_fraction: float = 2630 / 12096,
                 resident_base: bool = False, resident_hot: bool = False):
        if device == 'mps' and not torch.backends.mps.is_available():
            raise RuntimeError('PyTorch MPS is unavailable')
        if not 2 / 288 <= hot_fraction <= 1:
            raise ValueError('hot_fraction must be between 2/288 and 1')
        if resident_hot and not resident_base:
            raise ValueError('Hot-set residency requires --resident-base')
        root = root.expanduser().resolve()
        validate(root)
        self.phase = 'decode'
        self.device = device
        self.resident_hot = resident_hot
        self.root = root
        self.routing = {}
        self.wanted = np.zeros((42, 288), dtype=bool)
        self.store = ExpertStore(root, cache_gib, resident_base=resident_base)
        predictor_module, self.predictor_staging = staged_predictor()

        os.environ['NQ_FLASH_JT_PARAMS'] = json.dumps(dict(mix=0.75, hm=4.0, G=1, half_life=64.0))
        self.predictor = predictor_module.CommittedPredictor(root, preset='spark_256K_U_2630_flat50',
                                           device='cpu', dtype=torch.float32, fixed_fraction=0.2)
        self.hot_slots = round(hot_fraction * 12096)
        configure_slots(self.predictor, predictor_module, root, hot_fraction)
        self.hot_set_capacity_bytes = self.store.check_hot_capacity(self.predictor.budgets) if resident_hot else None
        self.model = empty_model(root, lambda layer: NestQuantExperts(layer, self.store, self))
        self.head = load_backbone(root, self.model, device=device)
        if resident_base:
            self.store.preload_base()
        self.tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True, trust_remote_code=False)
        self.thinking_off_template = (Path(__file__).resolve().parents[1] / 'spark/flash/thinking_off.jinja').read_text()
        self.reset()

    def reset(self):
        self.cache = None
        self.position = 0
        self.predictor.reset()
        self.routing.clear()
        self.decode_routed_experts = 0
        self.decode_hot_experts = 0

    @torch.inference_mode()
    def step(self, token: int):
        self.phase = 'decode'
        self.wanted = self.predictor.preforward(token, self.position)
        if int(self.wanted.sum()) != self.hot_slots:
            raise RuntimeError('Incorrect hot expert allocation')
        if self.resident_hot:
            self.store.prepare_hot((int(layer) + 3, int(expert)) for layer, expert in np.argwhere(self.wanted))
        self.routing.clear()
        ids = torch.tensor([[token]], device=self.device)
        output = self.model(ids, past_key_values=self.cache, use_cache=True)
        self.cache = output.past_key_values
        if set(self.routing) != set(range(3, 45)):
            raise RuntimeError('Incomplete expert routing capture')
        selected = np.stack([self.routing[layer][0][0] for layer in range(3, 45)])
        weights = np.stack([self.routing[layer][1][0] for layer in range(3, 45)])
        norms = np.array([self.routing[layer][2][0] for layer in range(3, 45)])
        self.decode_routed_experts += selected.size
        self.decode_hot_experts += int(np.take_along_axis(self.wanted, selected, axis=1).sum())
        self.predictor.commit([(self.position, token, selected, weights, norms)])
        self.position += 1
        logits = torch.nn.functional.linear(output.last_hidden_state[:, -1], self.head).float().cpu()[0]
        if not torch.isfinite(logits).all():
            raise FloatingPointError('Nonfinite model logits')
        return logits

    @torch.inference_mode()
    def prefill(self, tokens: list[int], chunk_size: int):
        if self.position != 0:
            raise ValueError('Layer-major prefill must start with an empty request cache')
        self.phase = 'prefill'
        selected = np.empty((len(tokens), 42, 8), dtype=np.int64)
        weights = np.empty((len(tokens), 42, 8), dtype=np.float32)
        norms = np.empty((len(tokens), 42), dtype=np.float32)
        completed = set()

        def capture(layer, start, end):
            if layer >= 3:
                captured = self.routing.pop(layer)
                selected[start:end, layer - 3] = captured[0]
                weights[start:end, layer - 3] = captured[1]
                norms[start:end, layer - 3] = captured[2]
                if end == len(tokens):
                    completed.add(layer)
            if end == len(tokens):
                print(json.dumps(dict(event='prefill_layer', layer=layer, tokens=len(tokens),
                                      expert_precision=4)), file=sys.stderr, flush=True)

        ids = torch.tensor([tokens], device=self.device)
        output = layer_major_prefill(self.model, ids, chunk_size, capture)
        self.cache = output.past_key_values
        if completed != set(range(3, 45)):
            raise RuntimeError('Incomplete layer-major prefill routing capture')
        rows = [(i, token, selected[i], weights[i], norms[i]) for i, token in enumerate(tokens)]
        self.predictor.commit(rows)
        self.position = len(tokens)
        logits = torch.nn.functional.linear(output.last_hidden_state[:, -1], self.head).float().cpu()[0]
        if not torch.isfinite(logits).all():
            raise FloatingPointError('Nonfinite prefill logits')
        return logits

    def generate(self, prompt: str, max_tokens: int, raw: bool = False, thinking: bool = False,
                 prefill_chunk_size: int = 256):
        if max_tokens < 1:
            raise ValueError('max_tokens must be positive')
        self.reset()
        tokens = self.tokenizer.encode(prompt) if raw else self.tokenizer.apply_chat_template(
            [dict(role='user', content=prompt)], tokenize=True, return_dict=False, add_generation_prompt=True,
            chat_template=None if thinking else self.thinking_off_template,
        )
        if not tokens:
            raise ValueError('Empty tokenized prompt')
        if len(tokens) + max_tokens > 4096:
            raise ValueError('The initial Mac reference path is limited to 4096 total tokens')
        resources_before = snapshot()
        expert_metrics_before = self.store.metrics.copy()
        started = time.perf_counter()
        logits = self.prefill(tokens, prefill_chunk_size)
        first_seconds = time.perf_counter() - started
        resources_after_prefill = snapshot()
        expert_metrics_after_prefill = self.store.metrics.copy()
        cache_bytes_after_prefill = self.store.cache_bytes
        result = []
        times = []
        eos = self.model.config.eos_token_id
        eos = set(eos if isinstance(eos, list) else [eos])
        for i in range(max_tokens):
            token = int(logits.argmax())
            result.append(token)
            print(json.dumps(dict(event='token', id=token, text=self.tokenizer.decode([token]))), flush=True)
            if token in eos or i + 1 == max_tokens:
                break
            before = time.perf_counter()
            logits = self.step(token)
            times.append(time.perf_counter() - before)
        resources_after_decode = snapshot()
        return dict(
            text=self.tokenizer.decode(result, skip_special_tokens=True), generated_ids=result,
            prompt_tokens=len(tokens), generated_tokens=len(result), ttft_seconds=first_seconds,
            decode_step_seconds=times, decode_tokens_per_second=len(times) / sum(times) if times else None,
            elapsed_seconds=time.perf_counter() - started,
            prefill_tokens_per_second=len(tokens) / first_seconds,
            expert_metrics={key: value - expert_metrics_before[key] for key, value in self.store.metrics.items()},
            expert_metrics_by_phase=dict(
                prefill={key: value - expert_metrics_before[key] for key, value in expert_metrics_after_prefill.items()},
                decode={key: value - expert_metrics_after_prefill[key] for key, value in self.store.metrics.items()},
            ),
            expert_cache_bytes=dict(limit=self.store.cache_limit, after_prefill=cache_bytes_after_prefill,
                                    after_decode=self.store.cache_bytes, resident_base=self.store.base_bytes),
            disk_reads=dict(prefill=resources_after_prefill['disk_read_bytes'] - resources_before['disk_read_bytes'],
                            decode=resources_after_decode['disk_read_bytes'] - resources_after_prefill['disk_read_bytes']),
            resources=dict(before=resources_before, after_prefill=resources_after_prefill, after_decode=resources_after_decode),
            decode_hot_expert_fraction=self.decode_hot_experts / self.decode_routed_experts if self.decode_routed_experts else None,
            runtime=f'NestQuant Metal experts with Transformers {self.device.upper()} backbone',
            backbone_device=self.device,
            thinking=thinking if not raw else None,
            policy=dict(base_bits=1.5, hot_bits=4, slots=self.hot_slots, fixed=self.predictor.n_fixed, jt_mix=0.75, refresh_tokens=1,
                        prefill_bits=4, prefill_chunk_size=prefill_chunk_size, resident_base=self.store.resident_base,
                        resident_hot=self.resident_hot, hot_set_capacity_bytes=self.hot_set_capacity_bytes),
            limitations=['text only', 'single sequence', '4096 token validation ceiling', 'no speculative decoding',
                         'synchronous expert SSD reads', 'greedy sampling', 'unquantized attention cache'],
        )

    def close(self):
        self.store.close()
        self.predictor_staging.cleanup()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--prompt', default='Reply with exactly: GLM_MAC_OK')
    parser.add_argument('--max-tokens', type=int, default=16)
    parser.add_argument('--cache-gib', type=float, default=4)
    parser.add_argument('--resident-base', action='store_true', help='Preload all base experts, about 55 GiB, outside the bounded refinement cache')
    parser.add_argument('--resident-hot', action='store_true', help='Keep every predicted hot refinement resident; requires resident base and sufficient cache')
    parser.add_argument('--hot-fraction', type=float, default=2630 / 12096)
    parser.add_argument('--prefill-chunk-size', type=int, default=256)
    parser.add_argument('--device', choices=['cpu', 'mps'], default='mps')
    parser.add_argument('--raw', action='store_true')
    parser.add_argument('--thinking', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.max_tokens < 1 or args.prefill_chunk_size < 1:
        parser.error('Token count and prefill chunk size must be positive')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    mx.set_cache_limit(1024**3)
    runtime = Runtime(args.model, args.cache_gib, args.device, args.hot_fraction, args.resident_base, args.resident_hot)
    try:
        report = runtime.generate(args.prompt, args.max_tokens, args.raw, args.thinking, args.prefill_chunk_size)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report), flush=True)
    finally:
        runtime.close()


if __name__ == '__main__':
    main()
