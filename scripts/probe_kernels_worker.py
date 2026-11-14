"""The worker of the C1 probe (Section 3.3 of specs/06e-kernels/plan.html): the rate of the MLX
matmul and attention kernels at the shapes of ViT-H+/16 at 512 px, batch size 8.

The same file runs in two venvs: with the MLX wheel of the repo and with the source build in
~/.cache/dinov3-mlx/mlx-src/ (ADR 0013). The driver scripts/probe_kernels.py starts it as
<python> -I probe_kernels_worker.py '<spec>'. It imports only mlx, numpy and the standard library,
and no module of the repo. It prints one JSON line.

Spec modes:

- "probe" (B3): the six linear shapes in fp16 and bf16 and the SDPA shape, each the median of 20
  calls with mx.eval after each call, after 3 warm-up calls (the protocol of probe_roofs.py).
- "sweep" (B4): one target shape, the control (no override) and each tile configuration in paired
  rounds in this process. The source build reads the tile override from DINOV3_PROBE_GEMM_NAX
  ("bm,bn,bk,wm,wn") or DINOV3_PROBE_SDPA_NAX ("bq,bk,wm,wn") at each dispatch.

Validity: the cosine of each output row to a NumPy float32 computation on the same rounded inputs;
the field cosine holds the smallest row cosine.
"""

import json
import os
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime

import mlx.core as mx
import numpy as np
from mlx import nn

MEMORY_LIMIT_BYTES = 12 * 1024**3
CALLS = 20
WARM_UP = 3
TOKENS = 1029  # 5 + 32 · 32 tokens at 512 px
BATCH = 8
HEADS = 20
HEAD_DIM = 64
DTYPES = {"fp16": mx.float16, "bf16": mx.bfloat16}
LINEAR_SHAPES = [
    (1280, 3840),
    (1280, 1280),
    (1280, 5120),
    (5120, 1280),
    (1280, 10240),
    (4096, 4096),
]
OVERRIDES = ("DINOV3_PROBE_GEMM_NAX", "DINOV3_PROBE_SDPA_NAX")


def median_time(function: Callable[[], mx.array], calls: int = CALLS) -> float:
    """Return the median time of one call, with mx.eval after each call, after WARM_UP calls."""
    for _ in range(WARM_UP):
        mx.eval(function())
    times = []
    for _ in range(calls):
        start = time.perf_counter()
        mx.eval(function())
        times.append(time.perf_counter() - start)
    return float(np.median(times))


def _normal(shape: tuple[int, ...], seed: int, dtype: mx.Dtype) -> mx.array:
    values = np.random.default_rng(seed).standard_normal(shape, dtype=np.float32)
    return mx.array(values).astype(dtype)


def _host(x: mx.array) -> np.ndarray:
    return np.array(x.astype(mx.float32))


def row_cosine(output: np.ndarray, expected: np.ndarray) -> float:
    """Return the smallest cosine between the rows (last axis) of output and expected."""
    a = output.reshape(-1, output.shape[-1]).astype(np.float64)
    b = expected.reshape(-1, expected.shape[-1]).astype(np.float64)
    cosine = (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))
    return float(np.nan_to_num(cosine, nan=-1.0).min())


class Probe:
    """One kernel at one shape: its call, its FLOPs and its NumPy check."""

    def __init__(self, name: str, flops: float, call: Callable[[], mx.array], check: Callable):
        self.name, self.flops, self.call, self.check = name, flops, call, check


def linear_probe(dtype: str, k: int, n: int) -> Probe:
    """One compiled nn.Linear with bias on an input (8, 1029, k), as probe_roofs.py."""
    x = _normal((BATCH, TOKENS, k), 1, DTYPES[dtype])
    layer = nn.Linear(k, n, bias=True)
    layer.set_dtype(DTYPES[dtype])
    mx.eval(x, layer.parameters())
    forward = mx.compile(lambda v: layer(v))

    cache: list[np.ndarray] = []

    def check() -> float:
        if not cache:  # one NumPy matmul for all configurations
            cache.append(_host(x) @ _host(layer.weight).T + _host(layer.bias))
        return row_cosine(_host(forward(x)), cache[0])

    return Probe(f"linear {dtype} {k}->{n}", 2 * BATCH * TOKENS * k * n, lambda: forward(x), check)


def sdpa_probe(dtype: str) -> Probe:
    """mx.fast.scaled_dot_product_attention at B = 8, 20 heads, T = 1,029, head dimension 64."""
    shape = (BATCH, HEADS, TOKENS, HEAD_DIM)
    q, k, v = (_normal(shape, seed, DTYPES[dtype]) for seed in (2, 3, 4))
    mx.eval(q, k, v)
    scale = HEAD_DIM**-0.5

    def call() -> mx.array:
        return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=None)

    cache: list[np.ndarray] = []

    def check() -> float:
        if not cache:  # one NumPy attention for all configurations
            qh, kh, vh = _host(q), _host(k), _host(v)
            expected = np.empty(shape, np.float32)
            for b in range(BATCH):  # one batch element at a time: 85 MB of scores each
                scores = qh[b] @ kh[b].transpose(0, 2, 1) * np.float32(scale)
                scores = np.exp(scores - scores.max(-1, keepdims=True))
                expected[b] = (scores / scores.sum(-1, keepdims=True)) @ vh[b]
            cache.append(expected)
        return row_cosine(_host(call()), cache[0])

    flops = 4 * BATCH * HEADS * TOKENS * TOKENS * HEAD_DIM
    return Probe(f"sdpa {dtype} T{TOKENS} D{HEAD_DIM}", flops, call, check)


def make_probe(name: str) -> Probe:
    """Return the probe of a name of the form "linear fp16 1280->1280" or "sdpa fp16"."""
    parts = name.split()
    if parts[0] == "linear":
        k, n = (int(v) for v in parts[2].split("->"))
        return linear_probe(parts[1], k, n)
    if parts[0] == "sdpa":
        return sdpa_probe(parts[1])
    raise ValueError(f"unknown probe {name!r}")


def probe_names() -> list[str]:
    """Return the probes of B3: the six linear shapes in fp16 and bf16 and the SDPA shape."""
    names = [f"linear {d} {k}->{n}" for d in DTYPES for k, n in LINEAR_SHAPES]
    return names + ["sdpa fp16", "sdpa bf16"]


def run_probe(spec: dict) -> dict:
    """B3: the median time of each probe, and its cosine if spec["check"]."""
    rows = []
    for name in spec.get("probes") or probe_names():
        probe = make_probe(name)
        seconds = median_time(probe.call)
        row = {"name": probe.name, "flops": probe.flops, "seconds": seconds}
        row["tflops"] = probe.flops / seconds / 1e12
        row["cosine"] = probe.check() if spec.get("check") else None
        rows.append(row)
        del probe
    return {"probes": rows}


def _set_override(variable: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(variable, None)
    else:
        os.environ[variable] = value


def run_sweep(spec: dict) -> dict:
    """B4: the control and each configuration of one target in paired rounds; the order rotates
    by one from round to round. A configuration whose first call fails gets its error and no
    times."""
    probe = make_probe(spec["probe"])
    variable = spec["variable"]
    names = ["control", *spec["configs"]]
    configs: dict[str, dict] = {name: {"seconds": [], "cosine": None} for name in names}
    for name in names:
        _set_override(variable, None if name == "control" else name)
        try:
            configs[name]["cosine"] = probe.check()
        except (RuntimeError, ValueError) as error:
            configs[name] = {"seconds": None, "cosine": None, "error": str(error)[-300:]}
    live = [name for name in names if configs[name]["seconds"] is not None]
    for r in range(spec["rounds"]):
        for name in live[r % len(live) :] + live[: r % len(live)]:
            _set_override(variable, None if name == "control" else name)
            configs[name]["seconds"].append(median_time(probe.call))
    _set_override(variable, None)
    return {"probe": probe.name, "flops": probe.flops, "variable": variable, "configs": configs}


def main() -> None:
    """Run the spec of argv[1] and print one JSON line."""
    spec = json.loads(sys.argv[1])
    for variable in OVERRIDES:  # the driver gives a clean environment; the sweep sets them
        os.environ.pop(variable, None)
    mx.set_memory_limit(MEMORY_LIMIT_BYTES)
    start = datetime.now(UTC).isoformat(timespec="seconds")
    try:
        result = run_sweep(spec) if spec["mode"] == "sweep" else run_probe(spec)
        result |= {"status": "done", "reason": ""}
    except (RuntimeError, ValueError, MemoryError) as error:
        result = {"status": "error", "reason": str(error)[-500:]}
    result |= {
        "mlx_version": mx.__version__,
        "mlx_architecture": mx.device_info()["architecture"],
        "start_time": start,
        "time": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
