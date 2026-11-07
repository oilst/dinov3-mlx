from collections.abc import Callable

import numpy as np
import pytest
import torch
from parity_checks import BATCH_CONFIGS, CONFIG_IDS, assert_thresholds, record_skip
from under_test import MODEL_SIZES_UNDER_TEST

from dinov3_apple.parity import Parity, measure_parity
from dinov3_apple.reference import DTYPE_MODES, run_torch

PARITY_PARAMS = [
    pytest.param(model_size, dtype_mode, config, id=f"{model_size}-{dtype_mode}-{config}")
    for model_size in MODEL_SIZES_UNDER_TEST
    for dtype_mode in DTYPE_MODES
    for config in CONFIG_IDS
]
BATCH_PARAMS = [
    pytest.param(model_size, dtype_mode, id=f"{model_size}-{dtype_mode}")
    for model_size in MODEL_SIZES_UNDER_TEST
    for dtype_mode in DTYPE_MODES
]
FP16_OVERFLOW = "fp16 overflow: a register token is above 65,504 (explore.html, Table 2)"

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no torch-mps")


def _skip_7b_fp32_weights(model_size: str, dtype_mode: str) -> str | None:
    if model_size == "ViT-7B/16" and DTYPE_MODES[dtype_mode].weights == torch.float32:
        return "ViT-7B/16 runs on torch-mps with fp16 or bf16 weights only (CLAUDE.md run rules)"
    return None


@pytest.mark.parametrize(("model_size", "dtype_mode", "config"), PARITY_PARAMS)
def test_parity(
    model_size: str,
    dtype_mode: str,
    config: str,
    backbone: Callable[[str, str, str], torch.nn.Module],
    reference: Callable[[str], dict[str, np.ndarray]],
    inputs: dict[str, np.ndarray],
    record: Callable[..., None],
) -> None:
    if reason := _skip_7b_fp32_weights(model_size, dtype_mode):
        record("parity", model_size, dtype_mode, config, None, "skipped", reason)
        pytest.skip(reason)
    with record_skip(record, "parity", model_size, dtype_mode, config):
        expected = reference(model_size)[config]
        model = backbone(model_size, "mps", dtype_mode)
    tokens = run_torch(model, inputs[config], dtype_mode)[0]
    if not np.isfinite(tokens).all():
        record("parity", model_size, dtype_mode, config, None, "non-finite")
        if dtype_mode == "fp16-cast":
            pytest.xfail(FP16_OVERFLOW)
        pytest.fail("non-finite output")
    parity = measure_parity(expected, tokens)
    record("parity", model_size, dtype_mode, config, parity, "measured")
    assert_thresholds(parity, dtype_mode)


@pytest.mark.parametrize(("model_size", "dtype_mode"), BATCH_PARAMS)
def test_batch_consistency(
    model_size: str,
    dtype_mode: str,
    backbone: Callable[[str, str, str], torch.nn.Module],
    inputs: dict[str, np.ndarray],
    record: Callable[..., None],
) -> None:
    if reason := _skip_7b_fp32_weights(model_size, dtype_mode):
        record("batch", model_size, dtype_mode, "batch-8-224x224", None, "skipped", reason)
        pytest.skip(reason)
    with record_skip(record, "batch", model_size, dtype_mode, "batch-8-224x224"):
        model = backbone(model_size, "mps", dtype_mode)
    batch = np.concatenate([inputs[config] for config in BATCH_CONFIGS])
    batched = run_torch(model, batch, dtype_mode)
    singles = [run_torch(model, inputs[config], dtype_mode)[0] for config in BATCH_CONFIGS]
    if not (np.isfinite(batched).all() and all(np.isfinite(s).all() for s in singles)):
        record("batch", model_size, dtype_mode, "batch-8-224x224", None, "non-finite")
        if dtype_mode == "fp16-cast":
            pytest.xfail(FP16_OVERFLOW)
        pytest.fail("non-finite output")
    parities = [measure_parity(single, batched[i]) for i, single in enumerate(singles)]
    worst = Parity(
        frame=min(p.frame for p in parities),
        patch_mean=min(p.patch_mean for p in parities),
        patch_p01=min(p.patch_p01 for p in parities),
        patch_min=min(p.patch_min for p in parities),
    )
    record("batch", model_size, dtype_mode, "batch-8-224x224", worst, "measured")
    assert_thresholds(worst, dtype_mode)
