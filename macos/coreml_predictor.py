"""Run the committed jT expert predictor on the Apple Neural Engine through Core ML.

The host gathers the token embedding, scatters the previous row's routing shares into a dense
[rows, 42*288] matrix, and builds the RoPE angles, the attention mask and a one-hot ring-write matrix.
Core ML evaluates the routing embedding bag (as a matmul), the input projections, the 8 transformer blocks and the block head for a fixed number of rows (1, 8 or
128; shorter batches are padded with rows that write nothing). The attention history is Core ML
state: per block, a 256-slot key and value ring that the model reads, attends over together with the
new rows, and updates in place. Nothing larger than one row's inputs crosses the host boundary per
step. The three row counts are functions of one multifunction package that share weights and state.

`cache` reads or writes the ring as PyTorch (k, v) pairs ordered by position, so prefix checkpoints
in native_prefix.py clone and restore it exactly as they do the PyTorch IncrementalJT cache.

Convert once with the separate Core ML tools environment:
  .venv-coreml/bin/python macos/coreml_predictor.py convert
which writes ~/Models/GLM/predictor-coreml-v1/jt_step.mlpackage plus jt_step.json.
NQ_COREML_PREDICTOR names another package.

Core ML on this macOS cannot build a CPU-only plan for these stateful programs (error -14), so the
supported compute units are CPU_AND_NE ('ane', default) and ALL ('all').
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

ROWS = (1, 8, 128)
RING = 256
MASKED = -1e4
NL, NE = 42, 288
DEFAULT_PACKAGE = Path(os.environ.get('NQ_COREML_PREDICTOR',
                                      Path.home() / 'Models/GLM/predictor-coreml-v1/jt_step.mlpackage'))
COMPUTE_UNITS = {'ane': 'CPU_AND_NE', 'all': 'ALL'}


def function_name(rows):
    return f'rows{rows}'


def state_names(layers):
    return [f'{kind}cache{i}' for i in range(layers) for kind in 'kv']


def head_permutation(hd):
    # RoPE rotates interleaved pairs (2i, 2i+1). Reordering each q/k head as [evens, odds] turns it
    # into the half-split rotation, which needs no strided slices; q.k is unchanged.
    return np.concatenate((np.arange(0, hd, 2), np.arange(1, hd, 2)))


def rope_tables(positions, hd=64, base=10000.0):
    f = base ** (-np.arange(0, hd, 2, dtype=np.float32) / hd)
    a = np.asarray(positions, dtype=np.float32)[:, None] * f
    return np.cos(a), np.sin(a)


def step_module(model, rows):
    """jT block head for `rows` new rows with the history in registered fp16 ring buffers (Core ML state)."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class Step(nn.Module):
        def __init__(s):
            super().__init__()
            s.tp, s.rn, s.ln, s.bhead = model.tp, model.rn, model.ln, model.bhead
            # EmbeddingBag(sum, per-sample weights) == dense share matrix @ table.
            s.re = nn.Parameter(model.re.weight.detach().t().contiguous())
            s.blocks = model.blocks
            d = model.ln.normalized_shape[0]
            s.nh = model.blocks[0].nh
            s.hd = d // s.nh
            perm = torch.as_tensor(head_permutation(s.hd))
            index = torch.arange(3 * d).view(3, s.nh, s.hd)
            index[0] = index[0][:, perm]
            index[1] = index[1][:, perm]
            s.qkv = nn.ParameterList([nn.Parameter(b.qkv.weight.detach()[index.reshape(-1)].clone())
                                      for b in model.blocks])
            for name in state_names(len(model.blocks)):
                s.register_buffer(name, torch.zeros(s.nh, RING, s.hd, dtype=torch.float16))

        def forward(s, token, routing, cos, sin, mask, write):
            # token [T,256] embedding, routing [T,42*288] previous-row shares, cos/sin [T,hd/2],
            # mask [T,RING+T] additive, write [T,RING] one-hot ring slot per real row (zero for padding rows).
            T, h = rows, s.hd // 2
            x = s.tp(token) + s.rn(F.linear(routing, s.re))
            c = torch.cat((cos, cos), -1)[None]
            sn = torch.cat((-sin, sin), -1)[None]
            keep = (1 - write.sum(0))[None, :, None]
            scatter = write.t()[None]
            for i, b in enumerate(s.blocks):
                qkv = F.linear(b.l1(x), s.qkv[i]).view(T, 3, s.nh, s.hd).permute(1, 2, 0, 3)
                q, k, v = qkv[0], qkv[1], qkv[2]
                q = q * c + torch.cat((q[..., h:], q[..., :h]), -1) * sn
                k = k * c + torch.cat((k[..., h:], k[..., :h]), -1) * sn
                kb, vb = getattr(s, f'kcache{i}'), getattr(s, f'vcache{i}')
                kc, vc = kb.float(), vb.float()
                kk = torch.cat((kc, k), 1)
                vv = torch.cat((vc, v), 1)
                a = torch.softmax(torch.matmul(q, kk.transpose(1, 2)) * (s.hd ** -0.5) + mask[None], -1)
                a = torch.matmul(a, vv).permute(1, 0, 2).reshape(T, -1)
                x = x + b.o(a)
                x = x + b.f2(F.gelu(b.f1(b.l2(x))))
                kb[:] = (kc * keep + torch.matmul(scatter, k)).half()
                vb[:] = (vc * keep + torch.matmul(scatter, v)).half()
            return torch.softmax(s.bhead(s.ln(x)).view(T, NL, NE), -1)

    return Step().eval()


def convert(args):
    import coremltools as ct
    import torch

    root = args.checkpoint
    sys.path.insert(0, str(root))
    import jt_model
    model = jt_model.load(str(root))
    out = args.out or DEFAULT_PACKAGE.parent
    out.mkdir(parents=True, exist_ok=True)
    d, nl, nh = model.ln.normalized_shape[0], len(model.blocks), model.blocks[0].nh
    hd = d // nh
    packages = {}
    for rows in ROWS:
        step = step_module(model, rows)
        example = (torch.zeros(rows, model.tp.in_features), torch.zeros(rows, NL * NE), torch.zeros(rows, hd // 2),
                   torch.zeros(rows, hd // 2), torch.zeros(rows, RING + rows), torch.zeros(rows, RING))
        with torch.no_grad():
            traced = torch.jit.trace(step, example)
        names = ('token', 'routing', 'cos', 'sin', 'mask', 'write')
        mlmodel = ct.convert(
            traced, inputs=[ct.TensorType(name=n, shape=e.shape, dtype=np.float16) for n, e in zip(names, example)],
            outputs=[ct.TensorType(name='probs', dtype=np.float32)],
            states=[ct.StateType(wrapped_type=ct.TensorType(shape=(nh, RING, hd), dtype=np.float16), name=n)
                    for n in state_names(nl)],
            compute_precision=ct.precision.FLOAT16, minimum_deployment_target=ct.target.macOS15,
            convert_to='mlprogram')
        path = out / f'jt_step_rows{rows}.mlpackage'
        mlmodel.save(str(path))
        packages[rows] = path
        print(json.dumps(dict(saved=str(path), rows=rows)), flush=True)
    descriptor = ct.utils.MultiFunctionDescriptor()
    for rows, path in packages.items():
        descriptor.add_function(str(path), src_function_name='main', target_function_name=function_name(rows))
    descriptor.default_function_name = function_name(1)
    target = out / 'jt_step.mlpackage'
    if target.exists():
        shutil.rmtree(target)
    ct.utils.save_multifunction(descriptor, str(target))
    for path in packages.values():
        shutil.rmtree(path)
    (out / 'jt_step.json').write_text(json.dumps(dict(
        rows=list(ROWS), ring=RING, routing_input='dense_shares', layers=nl, heads=nh, head_dim=hd, stateful=True, permuted_rope=True,
        precision='float16', coremltools=ct.__version__, torch=torch.__version__,
        checkpoint_sha256=json.loads((root / 'jT.json').read_text()).get('sha256')), indent=1))
    print(json.dumps(dict(saved=str(target))), flush=True)


def import_coremltools():
    try:
        import coremltools as ct
    except ImportError:
        # Default: the repository's conversion environment, appended so the runtime's own torch and
        # NumPy stay first on the path. Only coremltools and its small pure-Python deps load from it.
        site = os.environ.get('NQ_COREML_SITE') or next(
            iter(sorted((Path(__file__).resolve().parents[1] / '.venv-coreml/lib').glob('python3.*/site-packages'))),
            None)
        if site is None:
            raise
        sys.path.append(str(site))
        import coremltools as ct
    return ct


class CoreMLIncrementalJT:
    """Drop-in replacement for jt_runtime.IncrementalJT (predict / reset / cache / position)."""

    def __init__(self, checkpoint, package=None, compute_units='ane'):
        import torch
        from safetensors.numpy import load_file
        ct = import_coremltools()
        checkpoint = Path(checkpoint)
        package = Path(package) if package else DEFAULT_PACKAGE
        meta = json.loads((package.parent / 'jt_step.json').read_text())
        expected = json.loads((checkpoint / 'jT.json').read_text())
        if (meta.get('checkpoint_sha256') != expected.get('sha256') or meta['ring'] != RING or not meta.get('stateful')
                or meta.get('routing_input') != 'dense_shares'):
            raise ValueError(f'{package} was not converted from this jT checkpoint; rerun the conversion')
        cfg = expected['model']
        self.win, self.nl, self.nh = cfg['win'], cfg['nl'], cfg['nh']
        self.hd = cfg['d'] // cfg['nh']
        if self.win > RING:
            raise ValueError('Predictor window exceeds the Core ML ring')
        self.rows = tuple(meta['rows'])
        weights = load_file(str(checkpoint / 'jT.safetensors'))
        self.token_table = weights['tok.weight'].astype(np.float16)
        del weights
        if compute_units not in COMPUTE_UNITS:
            raise ValueError(f'Core ML predictor compute units must be one of {sorted(COMPUTE_UNITS)}')
        units = getattr(ct.ComputeUnit, COMPUTE_UNITS[compute_units])
        self.compute_units = compute_units
        self.models = {rows: ct.models.MLModel(str(package), compute_units=units, function_name=function_name(rows))
                       for rows in self.rows}
        self.state = self.models[self.rows[0]].make_state()
        self.names = state_names(self.nl)
        self.zeros = np.zeros((self.nh, RING, self.hd), np.float32)
        self.offsets = np.arange(NL)[:, None] * NE
        self._torch = torch
        self.model = None
        self.reset()

    def reset(self):
        # Fresh Core ML state is not guaranteed to be zero, and NaN garbage would survive the masks.
        for name in self.names:
            self.state.write_state(name, self.zeros)
        self.slot_position = np.full(RING, -(1 << 40), np.int64)
        self.position = 0

    @property
    def cache(self):
        torch = self._torch
        n = min(self.position, RING)
        order = np.arange(self.position - n, self.position) % RING
        pairs = []
        for i in range(self.nl):
            k = self.state.read_state(f'kcache{i}')[:, order].astype(np.float16)
            v = self.state.read_state(f'vcache{i}')[:, order].astype(np.float16)
            pairs.append((torch.from_numpy(k[None]), torch.from_numpy(v[None])))
        return pairs

    @cache.setter
    def cache(self, history):
        """Restore a history from `cache` (set `position` first, as native_prefix.restore_predictor does)."""
        self.slot_position = np.full(RING, -(1 << 40), np.int64)
        if not history or history[0] is None:
            for name in self.names:
                self.state.write_state(name, self.zeros)
            return
        n = history[0][0].shape[-2]
        slots = np.arange(self.position - n, self.position) % RING
        for i, (k, v) in enumerate(history):
            for name, value in ((f'kcache{i}', k), (f'vcache{i}', v)):
                ring = self.zeros.copy()
                ring[:, slots] = value.numpy()[0]
                self.state.write_state(name, ring)
        self.slot_position[slots] = np.arange(self.position - n, self.position)

    def predict(self, tokens, prev_ids, prev_q):
        tokens = np.asarray(tokens, dtype=np.int64)
        ids = np.asarray(prev_ids, dtype=np.int64)
        q = np.asarray(prev_q, dtype=np.float32)
        count = len(tokens)
        if count == 0:
            raise ValueError('Empty predictor batch')
        step = self.rows[-1]
        out = [self._step(tokens[s:s + step], ids[s:s + step], q[s:s + step]) for s in range(0, count, step)]
        return out[0] if len(out) == 1 else np.concatenate(out)

    def _step(self, tokens, ids, q):
        count = len(tokens)
        rows = next(r for r in self.rows if r >= count)
        flat = (ids + self.offsets).reshape(count, -1)
        routing = np.zeros((rows, NL * NE), np.float32)
        np.add.at(routing, (np.arange(count)[:, None], flat), q.reshape(count, -1))
        token = np.zeros((rows, self.token_table.shape[1]), np.float16)
        token[:count] = self.token_table[tokens]
        positions = self.position + np.arange(rows)
        cos, sin = rope_tables(positions, self.hd)
        mask = np.full((rows, RING + rows), MASKED, np.float16)
        delta = positions[:, None] - self.slot_position[None]
        mask[:, :RING][(delta > 0) & (delta < self.win)] = 0
        new = np.arange(rows)
        mask[:, RING:][new[:, None] >= new[None]] = 0
        slots = (self.position + np.arange(count)) % RING
        write = np.zeros((rows, RING), np.float16)
        write[np.arange(count), slots] = 1
        result = self.models[rows].predict(dict(token=token, routing=routing.astype(np.float16), cos=cos.astype(np.float16),
                                                sin=sin.astype(np.float16), mask=mask, write=write), state=self.state)
        self.slot_position[slots] = self.position + np.arange(count)
        self.position += count
        # The fp16 softmax sums to 1 only within ~1e-3; renormalize in fp32 like the PyTorch output.
        probs = np.asarray(result['probs'][:count], dtype=np.float32)
        return probs / probs.sum(-1, keepdims=True)


def install(predictor, checkpoint, package=None, compute_units='ane'):
    """Replace a CommittedPredictor's PyTorch jT network with the Core ML one (drops the torch weights)."""
    if predictor.net.position != 0:
        raise RuntimeError('Switch predictor backends only between requests')
    predictor.net = CoreMLIncrementalJT(checkpoint, package, compute_units)
    return predictor.net


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='command', required=True)
    c = sub.add_parser('convert')
    c.add_argument('--checkpoint', type=Path,
                   default=Path.home() / 'Models/GLM/GLM-5.3-Flash-NestQuant-1.5-4bit/serving/predictor/jt')
    c.add_argument('--out', type=Path)
    args = p.parse_args()
    if args.command == 'convert':
        convert(args)


if __name__ == '__main__':
    main()
