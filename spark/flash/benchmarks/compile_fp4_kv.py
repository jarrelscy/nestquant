"""Compile kernels for SM120/SM121 without a CUDA context or GPU execution."""

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
import nq_flash_fp4_kv as kv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arches", type=int, nargs="+", default=[120, 121])
    args = parser.parse_args()
    for arch in args.arches:
        for dtype in ("fp16", "bf16"):
            for width in (2048, 2176):
                source = ASTSource(
                    kv._attention,
                    signature={
                        "Q": "*" + dtype,
                        "C": "*u8",
                        "IDX": "*i32",
                        "OUT": "*fp32",
                        "L": "*fp32",
                    },
                    constexprs={
                        "H": 64,
                        "K": width,
                        "QS": 64 * 512,
                        "QH": 512,
                        "IS": width,
                        "PAGE": 13824,
                        "CS0": 13824 * 320 + 256,
                        "CS1": 320,
                        "NSLOTS": 276480,
                        "SCALE": 0.1,
                        "SPLITS": 8,
                        "BK": 32,
                        "BH": 16,
                    },
                )
                kernel = triton.compile(
                    source, target=GPUTarget("cuda", arch, 32), options={"num_warps": 8}
                )
                print(
                    f"sm{arch} attention {dtype} K={width}: shared={kernel.metadata.shared}"
                )
            source = ASTSource(
                kv._pack,
                signature={"X": "*" + dtype, "C": "*u8", "S": "*i64"},
                constexprs={
                    "XS": 512,
                    "SS": 1,
                    "PAGE": 13824,
                    "CS0": 13824 * 320 + 256,
                    "CS1": 320,
                    "NSLOTS": 276480,
                },
            )
            kernel = triton.compile(
                source, target=GPUTarget("cuda", arch, 32), options={"num_warps": 4}
            )
            print(f"sm{arch} pack {dtype}: shared={kernel.metadata.shared}")
            source = ASTSource(
                kv._merge,
                signature={"P": "*fp32", "L": "*fp32", "OUT": "*" + dtype},
                constexprs={"H": 64, "SPLITS": 8, "OS": 64 * 512, "OH": 512},
            )
            kernel = triton.compile(
                source, target=GPUTarget("cuda", arch, 32), options={"num_warps": 4}
            )
            print(f"sm{arch} merge {dtype}: shared={kernel.metadata.shared}")


if __name__ == "__main__":
    main()
