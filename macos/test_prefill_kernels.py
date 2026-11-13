import unittest

import mlx.core as mx
import numpy as np

from decoder import projection_fused
from microbench_window import make_layer
import prefill_kernels as pk


def relative_rms(value, reference):
    return float(mx.sqrt(mx.mean(mx.square(value - reference)) / mx.mean(mx.square(reference))).item())


class PrefillKernelTest(unittest.TestCase):
    """Dense rotated weights and the batched expert path must match the banked matvec path."""

    @classmethod
    def setUpClass(cls):
        mx.random.seed(11)
        cls.layers = {}
        for var_bits in (1, 8):
            layer = make_layer(12, 4, var_bits)
            for name, (base, refinements, meta) in layer.items():
                rank = 2 if name != 'down' else 1
                base['lrV'] = (0.01 * mx.random.normal((12, rank, meta['k']))).astype(mx.float16)
                base['lrU2'] = (0.01 * mx.random.normal((12, rank, meta['n']))).astype(mx.float16)
                refinements['lrU4'] = (0.01 * mx.random.normal((4, rank, meta['n']))).astype(mx.float16)
            mx.eval(layer)
            cls.layers[var_bits] = layer

    def test_dense_weights_reproduce_fused_projection(self):
        experts = mx.array([3, 7, 0], dtype=mx.uint32)
        for var_bits, layer in self.layers.items():
            for refined in (False, True):
                for name, (base, refinements, meta) in layer.items():
                    slots = mx.array([2, 0, 3], dtype=mx.uint32) if refined else None
                    dense = pk.dense_rotated(base, meta, experts, slots, refinements if refined else None)
                    params = pk.group_parameters({name: (base, meta)}, experts, slots,
                                                 {name: refinements} if refined else None)[name]
                    rows = 6
                    local = mx.array([0, 0, 1, 2, 2, 2], dtype=mx.uint32)
                    x = mx.random.normal((rows, meta['k']))
                    rotated = mx.hadamard_transform((x * params['su'][local].astype(mx.float32)).reshape(rows, -1, 128))
                    rotated = rotated.astype(mx.float16).astype(mx.float32).reshape(rows, -1)
                    partial = (rotated[:, None, :] @ dense[local].astype(mx.float32).swapaxes(1, 2))[:, 0]
                    actual = mx.hadamard_transform(partial.reshape(rows, -1, 128)).reshape(rows, -1)
                    actual = actual * params['sv'][local].astype(mx.float32)
                    low = x[:, None, :] @ params['lrv'][local].astype(mx.float32).swapaxes(1, 2)
                    actual = actual + (low @ params['u2'][local].astype(mx.float32))[:, 0]
                    if refined:
                        actual = actual + (low @ params['u4'][local].astype(mx.float32))[:, 0]
                    route = experts[local]
                    options = dict(refinements=refinements, refined_route=slots[local]) if refined else {}
                    expected = projection_fused(base, meta, x, route, **options)
                    mx.eval(actual, expected)
                    with self.subTest(var_bits=var_bits, refined=refined, name=name):
                        self.assertLess(relative_rms(actual, expected), 1e-5)

    def test_group_forward_matches_expert_mlp(self):
        group = [5, 1, 9]
        slots_of = {5: 1, 1: 3, 9: 0}
        tokens_np = np.array([0, 2, 3, 1, 2, 0, 3], dtype=np.uint32)
        local_np = np.array([0, 0, 0, 1, 1, 2, 2], dtype=np.uint32)
        weights = mx.array([0.3, 1.2, 0.7, 0.9, 0.2, 1.5, 0.4])
        X = mx.random.normal((4, 4096)).astype(mx.bfloat16)
        for var_bits, layer in self.layers.items():
            base = {name: (planes, meta) for name, (planes, _, meta) in layer.items()}
            refinements = {name: ref for name, (_, ref, _) in layer.items()}
            for refined in (False, True):
                experts = mx.array(group, dtype=mx.uint32)
                slots = mx.array([slots_of[e] for e in group], dtype=mx.uint32) if refined else None
                source = refinements if refined else None
                dense = {name: pk.dense_rotated(planes, meta, experts, slots, None if source is None else source[name])
                         for name, (planes, meta) in base.items()}
                params = pk.group_parameters(base, experts, slots, source)
                local = mx.array(local_np)
                actual = pk.expert_group_forward(X, mx.array(tokens_np), local, weights, dense, params)
                route = experts[local]
                hot = slots[local] if refined else None
                rows = X.astype(mx.float16).astype(mx.float32)[mx.array(tokens_np)]

                def project(name, value):
                    planes, meta = base[name]
                    options = dict(refinements=refinements[name], refined_route=hot) if refined else {}
                    return projection_fused(planes, meta, value, route, **options)

                gate = mx.minimum(project('gate', rows), 10.0)
                up = mx.clip(project('up', rows), -10.0, 10.0)
                expected = project('down', gate * mx.sigmoid(gate) * up)
                expected = expected * weights.astype(mx.float16).astype(mx.float32)[:, None]
                accumulator = mx.zeros((4, 4096), dtype=mx.float32)
                mx.eval(accumulator)
                marker = pk.expert_group_forward(X, mx.array(tokens_np), local, weights, dense, params,
                                                 accumulator=accumulator)
                mx.eval(actual, expected, marker)
                mx.synchronize()
                summed = mx.zeros((4, 4096), dtype=mx.float32).at[mx.array(tokens_np)].add(actual)
                with self.subTest(var_bits=var_bits, refined=refined, accumulate=True):
                    self.assertLess(relative_rms(accumulator, summed), 1e-6)
                with self.subTest(var_bits=var_bits, refined=refined):
                    # The matmul partials are rounded to fp16 (about 3e-4 relative per projection); random
                    # synthetic planes push the SwiGLU harder than real experts, where the difference is 5e-4.
                    self.assertLess(relative_rms(actual, expected), 4e-3)


if __name__ == '__main__':
    unittest.main()
