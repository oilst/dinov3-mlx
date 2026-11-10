"""Measure the two roofs of the roofline of this Mac with MLX: the matmul roof and the memory roof.

Three probes, each the median of 20 calls with one mx.eval after each call:

- matmul, plain: a @ b at square shapes and at the shapes of ViT-H+/16 at 512 px, batch size 8;
- matmul, linear: one compiled nn.Linear with bias on an input (8, 1029, k), the layout of the
  block parts (the weight is stored as (out, in));
- memory: x * 2 (read and write) and mx.sum (read) on 64 to 512 Mi values.

The matmul roof of a dtype is its highest matmul rate, the memory roof the highest GB/s. The roofs
are measured values, not the peaks of a data sheet. One JSON line goes to --results.

    MLX_ENABLE_TF32=0 uv run python -m dinov3_mlx.gpu_lock -- uv run python scripts/probe_roofs.py
"""

import argparse
import json
import time
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import mlx.core as mx
from mlx import nn

from dinov3_mlx.machine import check_machine_state, read_machine_state

MEMORY_LIMIT_BYTES = 10 * 1024**3
CALLS = 20
TOKENS = 8 * 1029  # ViT-H+/16 at 512 px, batch size 8
DTYPES = {"fp32": mx.float32, "fp16": mx.float16, "bf16": mx.bfloat16}
PLAIN_SHAPES = [
    (2048, 2048, 2048),
    (4096, 4096, 4096),
    (8192, 8192, 8192),
    (TOKENS, 1280, 3840),
    (TOKENS, 1280, 1280),
    (TOKENS, 1280, 10240),
    (TOKENS, 5120, 1280),
    (16384, 4096, 4096),
]
# (in, out) of qkv_proj, o_proj, one ff_in matrix, ff_out, both ff_in matrices, a square layer.
LINEAR_SHAPES = [
    (1280, 3840),
    (1280, 1280),
    (1280, 5120),
    (5120, 1280),
    (1280, 10240),
    (4096, 4096),
]
MEMORY_VALUES = [64 * 1024**2, 256 * 1024**2, 512 * 1024**2]


def median_time(function: Callable[[], mx.array], calls: int = CALLS) -> float:
    """Return the median time of one call of function, with mx.eval after each call.

    Args:
        function: The function to time.
        calls: The number of timed calls, after 3 warm-up calls.

    Returns:
        The median time in seconds.
    """
    for _ in range(3):
        mx.eval(function())
    times = []
    for _ in range(calls):
        start = time.perf_counter()
        mx.eval(function())
        times.append(time.perf_counter() - start)
    times.sort()
    return times[len(times) // 2]


def probe_plain(dtype: str) -> list[dict]:
    """Return the rate of a @ b at each shape of PLAIN_SHAPES."""
    rows = []
    for m, k, n in PLAIN_SHAPES:
        a = mx.random.normal((m, k)).astype(DTYPES[dtype])
        b = mx.random.normal((k, n)).astype(DTYPES[dtype])
        mx.eval(a, b)
        seconds = median_time(lambda a=a, b=b: a @ b)
        rows.append(
            {
                "dtype": dtype,
                "shape": [m, k, n],
                "seconds": seconds,
                "tflops": 2 * m * k * n / seconds / 1e12,
            }
        )
        del a, b
    return rows


def probe_linear(dtype: str) -> list[dict]:
    """Return the rate of one compiled nn.Linear at each shape of LINEAR_SHAPES."""
    rows = []
    for k, n in LINEAR_SHAPES:
        x = mx.random.normal((8, TOKENS // 8, k)).astype(DTYPES[dtype])
        layer = nn.Linear(k, n, bias=True)
        layer.set_dtype(DTYPES[dtype])
        mx.eval(x, layer.parameters())
        forward = mx.compile(lambda v, layer=layer: layer(v))
        seconds = median_time(lambda forward=forward, x=x: forward(x))
        rows.append(
            {
                "dtype": dtype,
                "in": k,
                "out": n,
                "seconds": seconds,
                "tflops": 2 * TOKENS * k * n / seconds / 1e12,
            }
        )
        del x, layer
    return rows


def probe_memory(dtype: str) -> list[dict]:
    """Return the GB/s of x * 2 and mx.sum(x) at each size of MEMORY_VALUES."""
    width = DTYPES[dtype].size
    rows = []
    for values in MEMORY_VALUES:
        x = mx.ones((values,), dtype=DTYPES[dtype])
        mx.eval(x)
        for kernel, function, passes in (
            ("scale", lambda x=x: x * 2, 2),
            ("sum", lambda x=x: mx.sum(x), 1),
        ):
            seconds = median_time(function)
            rows.append(
                {
                    "dtype": dtype,
                    "kernel": kernel,
                    "values": values,
                    "seconds": seconds,
                    "gbs": passes * values * width / seconds / 1e9,
                }
            )
        del x
    return rows


def roofs(line: dict) -> dict[str, float]:
    """Return the roofs of one probe line: the matmul roof of each dtype and the memory roof.

    Args:
        line: A line of the probe, with the lists plain, linear and memory.

    Returns:
        "fp32", "fp16" and "bf16" in TFLOP/s (the highest plain or linear rate) and "memory_gbs".
    """
    result = {}
    for dtype in DTYPES:
        rates = [r["tflops"] for r in line["plain"] + line["linear"] if r["dtype"] == dtype]
        if rates:
            result[dtype] = max(rates)
    result["memory_gbs"] = max(r["gbs"] for r in line["memory"])
    return result


def main() -> None:
    """Run the three probes and append one JSON line to --results."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path, default=Path("results/roofs.jsonl"))
    args = parser.parse_args()
    state = read_machine_state()
    check_machine_state(state)
    mx.set_memory_limit(MEMORY_LIMIT_BYTES)
    line = {
        "kind": "roofs",
        "machine": asdict(state),
        "mlx_architecture": mx.device_info()["architecture"],
        "plain": [row for dtype in DTYPES for row in probe_plain(dtype)],
        "linear": [row for dtype in ("fp16", "bf16") for row in probe_linear(dtype)],
        "memory": [row for dtype in ("fp32", "fp16") for row in probe_memory(dtype)],
        "time": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    line["roofs"] = roofs(line)
    with args.results.open("a") as file:
        file.write(json.dumps(line) + "\n")
    print(json.dumps(line["roofs"]))


if __name__ == "__main__":
    main()
