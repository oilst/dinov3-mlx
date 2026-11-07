"""Shared checks of the parity tests of the backends torch-mps and mlx."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import cache

import numpy as np
import pytest
from safetensors import safe_open

from dinov3_apple.bench import WEIGHT_BYTES
from dinov3_apple.hub import parameter_count, weights_dir
from dinov3_apple.memory import LOAD_RATIO, RunRuleError, check_run_rule
from dinov3_apple.mlx_model import MLX_DTYPE_MODES
from dinov3_apple.parity import EXPECTED_FP16_OVERFLOW, FP16_OVERFLOW, THRESHOLDS, Parity
from dinov3_apple.stored import configurations

CONFIG_IDS = [config.id for config in configurations()]
# The batch of 8: the three images at 224 x 224 px, repeated.
BATCH_CONFIGS = [CONFIG_IDS[i % 3 * 4] for i in range(8)]


@contextmanager
def record_skip(
    record: Callable[..., None], *key: str, backend: str = "torch-mps"
) -> Iterator[None]:
    """Record a skip, for example a missing reference file or a failed memory check."""
    try:
        yield
    except pytest.skip.Exception as skip:
        record(*key, None, "skipped", str(skip), backend=backend)
        raise


def assert_thresholds(parity: Parity, dtype_mode: str) -> None:
    """Assert the frame, patch mean and patch 1st percentile thresholds of one dtype mode."""
    threshold = THRESHOLDS[dtype_mode]
    assert parity.frame >= threshold.frame
    assert parity.patch_mean >= threshold.patch_mean
    assert parity.patch_p01 >= threshold.patch_p01


def min_token_cosine(reference: np.ndarray, output: np.ndarray) -> float:
    """Return the smallest cosine similarity of two token sequences, token by token, in fp64.

    Args:
        reference: Tokens with shape (T, D).
        output: Tokens with shape (T, D).

    Returns:
        The minimum over the T tokens. The cosine does not depend on the scale of a token.
    """
    ref, out = reference.astype(np.float64), output.astype(np.float64)
    cos = (ref * out).sum(-1) / (np.linalg.norm(ref, axis=-1) * np.linalg.norm(out, axis=-1))
    return float(np.minimum(cos, 1.0).min())


def check_finite(
    tokens: np.ndarray, model_size: str, dtype_mode: str, note: Callable[[], None]
) -> None:
    """Record and xfail an expected fp16 overflow. Fail any other non-finite output (way M2).

    Args:
        tokens: The output of the backend under test.
        model_size: A key of MODEL_IDS.
        dtype_mode: A key of THRESHOLDS.
        note: Writes the result line with the status "non-finite".
    """
    if np.isfinite(tokens).all():
        return
    note()
    if (model_size, dtype_mode) in EXPECTED_FP16_OVERFLOW:
        pytest.xfail(FP16_OVERFLOW)
    pytest.fail(f"non-finite output: ({model_size}, {dtype_mode}) is not an expected failure")


def run_rule_skip(model_size: str, dtype_mode: str, backend: str) -> str | None:
    """Return the reason to skip a pair that breaks a run rule on one backend, or None.

    mlx: only ViT-7B/16 in fp32 (its autocast modes have half-precision weights). torch-mps:
    ViT-7B/16 in fp32, fp16-autocast and bf16-autocast (fp32 weights on torch).

    Args:
        model_size: A key of MODEL_IDS.
        dtype_mode: A key of THRESHOLDS.
        backend: "mlx" or "torch-mps".
    """
    if backend == "mlx":
        bytes_per_weight = MLX_DTYPE_MODES[dtype_mode].matmul.size
    else:
        bytes_per_weight = WEIGHT_BYTES[dtype_mode]
    try:
        check_run_rule(model_size, bytes_per_weight, f"{model_size} {backend} {dtype_mode}")
    except RunRuleError as error:
        return str(error)
    return None


@cache
def largest_tensor_bytes(model_size: str) -> int:
    """Return the bytes of the largest fp32 tensor of one model size, from the safetensors headers.

    Raises:
        WeightsNotFoundError: If the weights are not on this Mac.
    """
    largest = 0
    for file in sorted(weights_dir(model_size).glob("*.safetensors")):
        with safe_open(file, "np") as tensors:
            for name in tensors.keys():  # noqa: SIM118 (safe_open is not iterable)
                largest = max(largest, 4 * int(np.prod(tensors.get_slice(name).get_shape())))
    return largest


def load_peak_target(model_size: str, dtype_mode: str) -> int:
    """Return the target of the load peak of one mlx load (Table 5 of the plan).

    LOAD_RATIO x the weight bytes in the matmul dtype + the largest fp32 tensor of the headers.

    Raises:
        WeightsNotFoundError: If the weights are not on this Mac.
    """
    weight_bytes = parameter_count(model_size) * MLX_DTYPE_MODES[dtype_mode].matmul.size
    return round(LOAD_RATIO * weight_bytes) + largest_tensor_bytes(model_size)
