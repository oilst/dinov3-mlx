"""Tests W2 and W3 of specs/06-speed/plan.html: quantized weights with the gated weights (D18,
F1, ADR 0011).

Report only with a fault bound: each line records the parity against the reference output and
against the unquantized output of the same dtype mode, and whether it passes THRESHOLDS[mode].
The tests assert only the fault bound of Table 21 of the plan. The speedup table counts a
quantized row as valid only if it passes THRESHOLDS[mode].
"""

from collections.abc import Callable

import mlx.core as mx
import numpy as np
import pytest
from parity_checks import CONFIG_IDS, load_peak_lower_bound, load_peak_target, record_skip
from under_test import SLOW

from dinov3_mlx.memory import GB
from dinov3_mlx.mlx_model import MlxBackbone, run_mlx
from dinov3_mlx.parity import THRESHOLDS, Parity, measure_parity

pytestmark = pytest.mark.mlx

BACKEND = "mlx"
MODES = ["fp16-autocast", "bf16-autocast"]
SIZES = ["ViT-S/16", "ViT-H+/16", "ViT-7B/16"] if SLOW else ["ViT-S/16"]
PARAMS = [
    pytest.param(size, mode, bits, id=f"{BACKEND}-{size}-{mode}-{bits}bit")
    for size in SIZES
    for mode in MODES
    for bits in (8, 4)
]
# Table 21 of the plan: the fault bound of each bit width. The patch 1st percentile of 4 bits is
# report only.
FAULT_BOUND = {
    8: {"frame": 0.99, "patch_mean": 0.99, "patch_p01": 0.90},
    4: {"frame": 0.75, "patch_mean": 0.75},
}

Loader = Callable[..., MlxBackbone]
# The unquantized output of each (model size, dtype mode, configuration), so the slot loads each
# unquantized model once.
UNQUANTIZED: dict[tuple[str, str], dict[str, np.ndarray]] = {}


def _unquantized(
    model_size: str,
    dtype_mode: str,
    backbone: Loader,
    mlx_forward: Callable[[MlxBackbone], Callable[[mx.array], mx.array]],
    inputs: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    key = (model_size, dtype_mode)
    if key not in UNQUANTIZED:
        UNQUANTIZED.clear()
        model = backbone(model_size, BACKEND, dtype_mode)
        forward = mlx_forward(model)
        UNQUANTIZED[key] = {
            config: run_mlx(model, inputs[config], dtype_mode, forward)[0] for config in CONFIG_IDS
        }
    return UNQUANTIZED[key]


def _passes(parity: Parity, dtype_mode: str) -> bool:
    threshold = THRESHOLDS[dtype_mode]
    return all(
        getattr(parity, m) >= getattr(threshold, m) for m in ("frame", "patch_mean", "patch_p01")
    )


@pytest.mark.parametrize(("model_size", "dtype_mode", "bits"), PARAMS)
def test_quantized(
    model_size: str,
    dtype_mode: str,
    bits: int,
    backbone: Loader,
    reference: Callable[[str], dict[str, np.ndarray]],
    inputs: dict[str, np.ndarray],
    record: Callable[..., None],
    mlx_forward: Callable[[MlxBackbone], Callable[[mx.array], mx.array]],
) -> None:
    """W2 (D18 A): the 12 configurations with quantized block weights. Asserts only the fault
    bound."""
    key = ("quantized", model_size, dtype_mode, "all")
    with record_skip(record, *key, backend=BACKEND):
        stored = reference(model_size)
        plain = _unquantized(model_size, dtype_mode, backbone, mlx_forward, inputs)
        model = backbone(model_size, BACKEND, dtype_mode, bits)
    forward = mlx_forward(model)
    below = []
    for config in CONFIG_IDS:
        tokens = run_mlx(model, inputs[config], dtype_mode, forward)[0]
        parity = measure_parity(stored[config], tokens)
        against_plain = measure_parity(plain[config], tokens)
        passes = _passes(parity, dtype_mode)
        record(
            "quantized",
            model_size,
            dtype_mode,
            config,
            parity,
            "measured",
            backend=BACKEND,
            values={"unquantized": against_plain.__dict__, "passes_thresholds": passes},
            bits=bits,
        )
        for measure, bound in FAULT_BOUND[bits].items():
            if getattr(parity, measure) < bound:
                below.append(f"{config} {measure} {getattr(parity, measure):.6f} < {bound}")
    assert not below, below


@pytest.mark.parametrize(("model_size", "dtype_mode", "bits"), PARAMS)
def test_quantized_load_peak(
    model_size: str,
    dtype_mode: str,
    bits: int,
    backbone: Loader,
    record: Callable[..., None],
    report: Callable[..., None],
) -> None:
    """W3 (F1, Table 9): the load report of a quantized load: lower bound ≤ load peak ≤ target,
    cache 0 bytes, the need of Table 9."""
    from dinov3_mlx.mlx_weights import required_mlx_load_bytes

    key = ("quantized-load", model_size, dtype_mode, "load")
    with record_skip(record, *key, backend=BACKEND):
        model = backbone(model_size, BACKEND, dtype_mode, bits)
    load_report = model.load_report
    target = load_peak_target(model_size, dtype_mode, bits)
    lower_bound = load_peak_lower_bound(model_size, model)
    need = required_mlx_load_bytes(model_size, dtype_mode, bits=bits)
    report(
        "test_quantized_load_peak",
        model_size=model_size,
        dtype_mode=dtype_mode,
        bits=bits,
        need_bytes=load_report.need_bytes,
        peak_bytes=load_report.peak_bytes,
        cache_bytes=load_report.cache_bytes,
        lower_bound=lower_bound,
        target=target,
        share_of_target=load_report.peak_bytes / target,
    )
    assert lower_bound <= load_report.peak_bytes <= target, (
        f"{load_report.peak_bytes / GB:.4f} GB, target {target / GB:.4f} GB"
    )
    assert load_report.cache_bytes == 0
    assert load_report.need_bytes == need
    assert load_report.bits == bits
