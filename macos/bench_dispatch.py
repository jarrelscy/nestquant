"""Per-kernel dispatch cost on this GPU: chains of tiny dependent MLX ops evaluated as one graph.
Run inside the GPU lock."""
import json
import statistics
import time

import mlx.core as mx


def chain(x, n):
    for _ in range(n):
        x = x * 1.0001 + 0.5
    return x


def main():
    x = mx.ones((4096,))
    mx.eval(x)
    out = {}
    for n in (100, 1000, 3000):
        samples = []
        for _ in range(12):
            t0 = time.perf_counter()
            y = chain(x, n)
            t1 = time.perf_counter()
            mx.eval(y)
            t2 = time.perf_counter()
            samples.append((t1 - t0, t2 - t1))
        build = statistics.median(s[0] for s in samples[2:])
        run = statistics.median(s[1] for s in samples[2:])
        # x * a + b is two primitives per iteration (MLX does not fuse without compile)
        out[n] = dict(ops=2 * n, build_us_per_op=round(1e6 * build / (2 * n), 2),
                      eval_us_per_op=round(1e6 * run / (2 * n), 2))
    print(json.dumps(out))


if __name__ == '__main__':
    main()
