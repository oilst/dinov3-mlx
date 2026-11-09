"""Shared checks of the parity tests of the backends torch-mps and mlx, and the range of the
load peak of one mlx load."""

import math
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import cache

import numpy as np
import pytest
from mlx.utils import tree_flatten
from safetensors import safe_open

from dinov3_mlx.bench import bytes_per_weight, quantized_weight_bytes
from dinov3_mlx.hub import parameter_count, weights_dir
from dinov3_mlx.memory import LOAD_RATIO, RunRuleError, check_run_rule
from dinov3_mlx.mlx_model import (
    MLX_DTYPE_MODES,
    MlxBackbone,
    linear_parameter_names,
    quantized_weight_names,
)
from dinov3_mlx.mlx_weights import map_key
from dinov3_mlx.parity import EXPECTED_FP16_OVERFLOW, FP16_OVERFLOW, THRESHOLDS, Parity
from dinov3_mlx.stored import configurations

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
    try:
        check_run_rule(
            model_size,
            bytes_per_weight(backend, dtype_mode),
            f"{model_size} {backend} {dtype_mode}",
        )
    except RunRuleError as error:
        return str(error)
    return None


@cache
def header_shapes(model_size: str) -> dict[str, tuple[int, ...]]:
    """Return the shape of each tensor of one model size, from the safetensors headers.

    The read takes the headers only, no tensor data. All tensors of the files are F32.

    Raises:
        WeightsNotFoundError: If the weights are not on this Mac.
    """
    shapes = {}
    for file in sorted(weights_dir(model_size).glob("*.safetensors")):
        with safe_open(file, "np") as tensors:
            for name in tensors.keys():  # noqa: SIM118 (safe_open is not iterable)
                shapes[name] = tuple(tensors.get_slice(name).get_shape())
    return shapes


def largest_tensor_bytes(model_size: str) -> int:
    """Return the bytes of the largest fp32 tensor of one model size, from the safetensors headers.

    Raises:
        WeightsNotFoundError: If the weights are not on this Mac.
    """
    return max(4 * math.prod(shape) for shape in header_shapes(model_size).values())


def load_peak_target(model_size: str, dtype_mode: str, bits: int | None = None) -> int:
    """Return the target of the load peak of one mlx load (Table 5 of the Step 5 plan, Table 9 of
    specs/06-speed/plan.html).

    LOAD_RATIO x the weight bytes in the matmul dtype + the largest fp32 tensor of the headers.
    With bits: LOAD_RATIO x the quantized weight bytes + the largest fp32 tensor + the same tensor
    in the matmul dtype (one source, its cast and its quantized output live together).

    Raises:
        WeightsNotFoundError: If the weights are not on this Mac.
    """
    matmul = MLX_DTYPE_MODES[dtype_mode].matmul.size
    largest = largest_tensor_bytes(model_size)
    if bits is not None:
        weight_bytes = quantized_weight_bytes(model_size, bits, parameter_count(model_size))
        return round(LOAD_RATIO * weight_bytes) + largest + largest * matmul // 4
    weight_bytes = parameter_count(model_size) * matmul
    return round(LOAD_RATIO * weight_bytes) + largest


def load_peak_lower_bound(model_size: str, model: MlxBackbone) -> int:
    """Return the lower bound of the load peak of one mlx load (F1 of
    specs/05-all-sizes/verify.html).

    The load keeps each mapped tensor, evaluated in its dtype, until load_weights. So the load peak
    is at least the bytes of the mapped tensors of the headers, without the dropped mask token: the
    matmul dtype for the weight and the bias of each nn.Linear, the stream dtype for the other
    parameters. A load that does not evaluate the tensors stays below this bound.

    Args:
        model_size: A key of MODEL_IDS.
        model: The loaded backbone. Its dtype mode and its nn.Linear modules give the dtypes.

    Raises:
        WeightsNotFoundError: If the weights are not on this Mac.
    """
    dtypes = MLX_DTYPE_MODES[model.dtype_mode]
    linear = linear_parameter_names(model)
    quantized = quantized_weight_names(model)
    parameters = dict(tree_flatten(model.parameters()))
    total = 0
    for name, shape in header_shapes(model_size).items():
        if (key := map_key(name)) is None:
            continue
        if key in quantized:  # the packed weight, the scales and the offsets (F1 of Step 6)
            stem = key.removesuffix(".weight")
            total += sum(
                parameters[f"{stem}.{leaf}"].nbytes for leaf in ("weight", "scales", "biases")
            )
        else:
            total += math.prod(shape) * (dtypes.matmul if key in linear else dtypes.stream).size
    return total
