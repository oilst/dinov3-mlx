"""Tests W1 to W6 of specs/04-mlx-port/plan.html: the MLX backbone with the gated weights against
the stored hidden states and the reference files of Step 2.

They need one model load of ViT-S/16 in fp32 on mlx (TF32 off). The backbone fixture checks the
free memory first and skips with the reason when the load does not fit.
"""

from collections.abc import Callable

import mlx.core as mx
import numpy as np
import pytest
from parity_checks import (
    BATCH_CONFIGS,
    CONFIG_IDS,
    assert_thresholds,
    min_token_cosine,
    record_skip,
)
from under_test import MLX_MODEL_SIZES_UNDER_TEST

from dinov3_apple.mlx_model import (
    MLX_DTYPE_MODES,
    MlxBackbone,
    run_mlx,
    run_mlx_hidden_states,
)
from dinov3_apple.outputs import NUM_PREFIX_TOKENS
from dinov3_apple.parity import Parity, measure_parity
from dinov3_apple.stored import HIDDEN_STATES_CONFIG

pytestmark = pytest.mark.mlx

BACKEND = "mlx"
HIDDEN_STATES_KEY = f"{HIDDEN_STATES_CONFIG}/hidden_states"
# An isolated part has no error from the blocks before it: ten times stricter on 1 - cosine than
# the fp32 threshold (Table 8 of the plan).
PART_COSINE = 0.99999
NUM_BLOCKS = 12

MODEL_PARAMS = [
    pytest.param(model_size, dtype_mode, id=f"{BACKEND}-{model_size}-{dtype_mode}")
    for model_size in MLX_MODEL_SIZES_UNDER_TEST
    for dtype_mode in MLX_DTYPE_MODES
]
BLOCK_PARAMS = [
    pytest.param(model_size, dtype_mode, i, id=f"{BACKEND}-{model_size}-{dtype_mode}-block-{i}")
    for model_size in MLX_MODEL_SIZES_UNDER_TEST
    for dtype_mode in MLX_DTYPE_MODES
    for i in range(NUM_BLOCKS)
]
PARITY_PARAMS = [
    pytest.param(model_size, dtype_mode, config, id=f"{BACKEND}-{model_size}-{dtype_mode}-{config}")
    for model_size in MLX_MODEL_SIZES_UNDER_TEST
    for dtype_mode in MLX_DTYPE_MODES
    for config in CONFIG_IDS
]

Loader = Callable[[str, str, str], MlxBackbone]
Reader = Callable[[str], dict[str, np.ndarray]]


def _load(
    backbone: Loader, reference: Reader, record: Callable[..., None], *key: str
) -> tuple[MlxBackbone, dict[str, np.ndarray]]:
    kind, model_size, dtype_mode, config = key
    with record_skip(record, kind, model_size, dtype_mode, config, backend=BACKEND):
        stored = reference(model_size)
        model = backbone(model_size, BACKEND, dtype_mode)
    return model, stored


def _differences(expected: np.ndarray, output: np.ndarray) -> dict[str, float]:
    largest = float(np.abs(output.astype(np.float64) - expected.astype(np.float64)).max())
    return {
        "max_abs": largest,
        "max_rel": largest / float(np.abs(expected).max()),
        "min_cosine": min_token_cosine(expected, output),
    }


@pytest.mark.parametrize(("model_size", "dtype_mode"), MODEL_PARAMS)
def test_mlx_embeddings_match_stored(
    model_size: str,
    dtype_mode: str,
    backbone: Loader,
    reference: Reader,
    inputs: dict[str, np.ndarray],
    record: Callable[..., None],
) -> None:
    key = ("stored-embeddings", model_size, dtype_mode, HIDDEN_STATES_CONFIG)
    model, stored = _load(backbone, reference, record, *key)
    expected = stored[HIDDEN_STATES_KEY][0]
    pixels = mx.array(inputs[HIDDEN_STATES_CONFIG]).astype(MLX_DTYPE_MODES[dtype_mode])
    output = np.array(model.embeddings(pixels)[0].astype(mx.float32))
    values = _differences(expected, output)
    values["prefix_exact"] = bool(
        np.array_equal(output[:NUM_PREFIX_TOKENS], expected[:NUM_PREFIX_TOKENS])
    )
    record(*key, None, "measured", backend=BACKEND, values=values)
    assert values["prefix_exact"]
    assert values["min_cosine"] >= PART_COSINE


@pytest.mark.parametrize(("model_size", "dtype_mode", "block"), BLOCK_PARAMS)
def test_mlx_block_matches_stored(
    model_size: str,
    dtype_mode: str,
    block: int,
    backbone: Loader,
    reference: Reader,
    record: Callable[..., None],
) -> None:
    key = ("stored-block", model_size, dtype_mode, f"block-{block}")
    model, stored = _load(backbone, reference, record, *key)
    hidden = stored[HIDDEN_STATES_KEY]
    grid = int(np.sqrt(hidden.shape[1] - NUM_PREFIX_TOKENS))  # 14 x 14 at 224 x 224 px
    cos, sin = model.rope(grid, grid)
    x = mx.array(hidden[block]).astype(MLX_DTYPE_MODES[dtype_mode])[None]
    output = np.array(model.layer[block](x, cos, sin)[0].astype(mx.float32))
    values = _differences(hidden[block + 1], output)
    record(*key, None, "measured", backend=BACKEND, values=values)
    assert values["min_cosine"] >= PART_COSINE


@pytest.mark.parametrize(("model_size", "dtype_mode"), MODEL_PARAMS)
def test_mlx_final_norm_matches_stored(
    model_size: str,
    dtype_mode: str,
    backbone: Loader,
    reference: Reader,
    record: Callable[..., None],
) -> None:
    key = ("stored-final-norm", model_size, dtype_mode, HIDDEN_STATES_CONFIG)
    model, stored = _load(backbone, reference, record, *key)
    x = mx.array(stored[HIDDEN_STATES_KEY][-1]).astype(MLX_DTYPE_MODES[dtype_mode])
    output = np.array(model.norm(x).astype(mx.float32))
    values = _differences(stored[HIDDEN_STATES_CONFIG], output)
    record(*key, None, "measured", backend=BACKEND, values=values)
    assert values["min_cosine"] >= PART_COSINE


@pytest.mark.parametrize(("model_size", "dtype_mode"), MODEL_PARAMS)
def test_mlx_hidden_states(
    model_size: str,
    dtype_mode: str,
    backbone: Loader,
    reference: Reader,
    inputs: dict[str, np.ndarray],
    record: Callable[..., None],
) -> None:
    key = ("hidden-state", model_size, dtype_mode, HIDDEN_STATES_CONFIG)
    model, stored = _load(backbone, reference, record, *key)
    expected = stored[HIDDEN_STATES_KEY]
    hidden = run_mlx_hidden_states(model, inputs[HIDDEN_STATES_CONFIG])
    assert hidden.shape == expected.shape
    parities = [measure_parity(expected[i], hidden[i]) for i in range(len(expected))]
    for i, parity in enumerate(parities):
        values = _differences(expected[i], hidden[i])
        line = ("hidden-state", model_size, dtype_mode, f"state-{i}", parity, "measured")
        record(*line, backend=BACKEND, values=values)
    for parity in parities:
        assert_thresholds(parity, dtype_mode)


@pytest.mark.parametrize(("model_size", "dtype_mode", "config"), PARITY_PARAMS)
def test_parity(
    model_size: str,
    dtype_mode: str,
    config: str,
    backbone: Loader,
    reference: Reader,
    inputs: dict[str, np.ndarray],
    record: Callable[..., None],
) -> None:
    model, stored = _load(backbone, reference, record, "parity", model_size, dtype_mode, config)
    tokens = run_mlx(model, inputs[config], dtype_mode)[0]
    if not np.isfinite(tokens).all():
        record("parity", model_size, dtype_mode, config, None, "non-finite", backend=BACKEND)
        pytest.fail("non-finite output")
    parity = measure_parity(stored[config], tokens)
    record("parity", model_size, dtype_mode, config, parity, "measured", backend=BACKEND)
    assert_thresholds(parity, dtype_mode)


@pytest.mark.parametrize(("model_size", "dtype_mode"), MODEL_PARAMS)
def test_batch_consistency(
    model_size: str,
    dtype_mode: str,
    backbone: Loader,
    inputs: dict[str, np.ndarray],
    record: Callable[..., None],
) -> None:
    key = ("batch", model_size, dtype_mode, "batch-8-224x224")
    with record_skip(record, *key, backend=BACKEND):
        model = backbone(model_size, BACKEND, dtype_mode)
    batch = np.concatenate([inputs[config] for config in BATCH_CONFIGS])
    batched = run_mlx(model, batch, dtype_mode)
    singles = [run_mlx(model, inputs[config], dtype_mode)[0] for config in BATCH_CONFIGS]
    if not (np.isfinite(batched).all() and all(np.isfinite(s).all() for s in singles)):
        record(*key, None, "non-finite", backend=BACKEND)
        pytest.fail("non-finite output")
    parities = [measure_parity(single, batched[i]) for i, single in enumerate(singles)]
    worst = Parity(
        frame=min(p.frame for p in parities),
        patch_mean=min(p.patch_mean for p in parities),
        patch_p01=min(p.patch_p01 for p in parities),
        patch_min=min(p.patch_min for p in parities),
    )
    largest = max(float(np.abs(batched[i] - s).max()) for i, s in enumerate(singles))
    record(*key, worst, "measured", backend=BACKEND, values={"max_abs": largest})
    assert_thresholds(worst, dtype_mode)
