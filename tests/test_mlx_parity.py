"""Tests W1 to W8 of specs/05-all-sizes/plan.html: the MLX backbone with the gated weights against
the stored hidden states and the reference files of Step 2, for each model size under test and
each of the five dtype modes, and the token range.

The backbone fixture checks the run rule and the free memory first, and skips with the reason when
the load breaks a run rule or does not fit. Each test calls run_rule_skip first, before it reads a
stored file or loads a model. W1 to W4 assert in fp32 and record the values in the four fp16 and
bf16 dtype modes (D8).
"""

from collections.abc import Callable

import mlx.core as mx
import numpy as np
import pytest
from parity_checks import (
    BATCH_CONFIGS,
    CONFIG_IDS,
    assert_thresholds,
    check_finite,
    min_token_cosine,
    record_skip,
    run_rule_skip,
)
from under_test import MLX_MODEL_SIZES_UNDER_TEST

from dinov3_apple.mlx_model import (
    MLX_DTYPE_MODES,
    MlxBackbone,
    run_mlx,
    run_mlx_hidden_states,
    run_mlx_token_range,
)
from dinov3_apple.outputs import NUM_PREFIX_TOKENS
from dinov3_apple.parity import Parity, measure_parity
from dinov3_apple.stored import HIDDEN_STATES_CONFIG

pytestmark = pytest.mark.mlx

BACKEND = "mlx"
HIDDEN_STATES_KEY = f"{HIDDEN_STATES_CONFIG}/hidden_states"
# An isolated part has no error from the blocks before it: ten times stricter on 1 - cosine than
# the fp32 threshold (Table 8 of the Step 4 plan).
PART_COSINE = 0.99999
# W8: the token range in fp32 against the stored hidden states, relative difference.
TOKEN_RANGE_RELATIVE = 1e-4

MODEL_PARAMS = [
    pytest.param(model_size, dtype_mode, id=f"{BACKEND}-{model_size}-{dtype_mode}")
    for model_size in MLX_MODEL_SIZES_UNDER_TEST
    for dtype_mode in MLX_DTYPE_MODES
]
FP32_PARAMS = [
    pytest.param(model_size, "fp32", id=f"{BACKEND}-{model_size}-fp32")
    for model_size in MLX_MODEL_SIZES_UNDER_TEST
]
PARITY_PARAMS = [
    pytest.param(model_size, dtype_mode, config, id=f"{BACKEND}-{model_size}-{dtype_mode}-{config}")
    for model_size in MLX_MODEL_SIZES_UNDER_TEST
    for dtype_mode in MLX_DTYPE_MODES
    for config in CONFIG_IDS
]

Loader = Callable[[str, str, str], MlxBackbone]
Reader = Callable[[str], dict[str, np.ndarray]]


def _skip_run_rule(record: Callable[..., None], *key: str) -> None:
    """Record and skip a pair that breaks a run rule, before any file read."""
    _, model_size, dtype_mode, _ = key
    if reason := run_rule_skip(model_size, dtype_mode, BACKEND):
        record(*key, None, "skipped", reason, backend=BACKEND)
        pytest.skip(reason)


def _load(
    backbone: Loader, reference: Reader, record: Callable[..., None], *key: str
) -> tuple[MlxBackbone, dict[str, np.ndarray]]:
    kind, model_size, dtype_mode, config = key
    _skip_run_rule(record, *key)
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


def _round(values: np.ndarray, dtype: mx.Dtype) -> mx.array:
    """Return stored fp32 values rounded to the stream dtype, as MLX input."""
    return mx.array(values).astype(dtype)


def _host(values: mx.array) -> np.ndarray:
    return np.array(values.astype(mx.float32))


@pytest.mark.parametrize(("model_size", "dtype_mode"), MODEL_PARAMS)
def test_mlx_embeddings_match_stored(
    model_size: str,
    dtype_mode: str,
    backbone: Loader,
    reference: Reader,
    inputs: dict[str, np.ndarray],
    record: Callable[..., None],
) -> None:
    """W1: the embeddings of the stored pixels against stored hidden state 0."""
    key = ("stored-embeddings", model_size, dtype_mode, HIDDEN_STATES_CONFIG)
    model, stored = _load(backbone, reference, record, *key)
    expected = stored[HIDDEN_STATES_KEY][0]
    output = _host(model.embeddings(mx.array(inputs[HIDDEN_STATES_CONFIG]))[0])
    prefix = _host(_round(expected[:NUM_PREFIX_TOKENS], MLX_DTYPE_MODES[dtype_mode].stream))
    values = _differences(expected, output)
    values["prefix_exact"] = bool(np.array_equal(output[:NUM_PREFIX_TOKENS], prefix))
    record(*key, None, "measured", backend=BACKEND, values=values)
    assert values["prefix_exact"]
    if dtype_mode == "fp32":
        assert values["min_cosine"] >= PART_COSINE


@pytest.mark.parametrize(("model_size", "dtype_mode"), MODEL_PARAMS)
def test_mlx_block_matches_stored(
    model_size: str,
    dtype_mode: str,
    backbone: Loader,
    reference: Reader,
    record: Callable[..., None],
) -> None:
    """W2: each block alone on stored hidden state i against stored hidden state i + 1."""
    model, stored = _load(
        backbone, reference, record, "stored-block", model_size, dtype_mode, "blocks"
    )
    hidden = stored[HIDDEN_STATES_KEY]
    assert len(hidden) == len(model.layer) + 1
    grid = int(np.sqrt(hidden.shape[1] - NUM_PREFIX_TOKENS))  # 14 x 14 at 224 x 224 px
    cos, sin = model.rope(grid, grid)
    stream = MLX_DTYPE_MODES[dtype_mode].stream
    below = []
    for block, layer in enumerate(model.layer):
        output = _host(layer(_round(hidden[block], stream)[None], cos, sin)[0])
        values = _differences(hidden[block + 1], output)
        key = ("stored-block", model_size, dtype_mode, f"block-{block}")
        record(*key, None, "measured", backend=BACKEND, values=values)
        if not values["min_cosine"] >= PART_COSINE:
            below.append((block, values["min_cosine"]))
    if dtype_mode == "fp32":
        assert not below, f"first block below {PART_COSINE}: block {below[0][0]} ({below[0][1]})"


@pytest.mark.parametrize(("model_size", "dtype_mode"), MODEL_PARAMS)
def test_mlx_final_norm_matches_stored(
    model_size: str,
    dtype_mode: str,
    backbone: Loader,
    reference: Reader,
    record: Callable[..., None],
) -> None:
    """W3: the final layer norm on the last stored hidden state against the stored tokens."""
    key = ("stored-final-norm", model_size, dtype_mode, HIDDEN_STATES_CONFIG)
    model, stored = _load(backbone, reference, record, *key)
    x = _round(stored[HIDDEN_STATES_KEY][-1], MLX_DTYPE_MODES[dtype_mode].stream)
    values = _differences(stored[HIDDEN_STATES_CONFIG], _host(model.norm(x)))
    record(*key, None, "measured", backend=BACKEND, values=values)
    if dtype_mode == "fp32":
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
    """W4: run_mlx_hidden_states against all L + 1 stored hidden states, in a chain."""
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
    if dtype_mode == "fp32":
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
    """W5: the 12 configurations against the reference file."""
    key = ("parity", model_size, dtype_mode, config)
    model, stored = _load(backbone, reference, record, *key)
    tokens = run_mlx(model, inputs[config], dtype_mode)[0]
    check_finite(
        tokens,
        model_size,
        dtype_mode,
        lambda: record(*key, None, "non-finite", backend=BACKEND),
    )
    parity = measure_parity(stored[config], tokens)
    record(*key, parity, "measured", backend=BACKEND)
    assert_thresholds(parity, dtype_mode)


@pytest.mark.parametrize(("model_size", "dtype_mode"), MODEL_PARAMS)
def test_batch_consistency(
    model_size: str,
    dtype_mode: str,
    backbone: Loader,
    inputs: dict[str, np.ndarray],
    record: Callable[..., None],
) -> None:
    """W6: a batch of 8 against 8 single runs. Needs no reference file."""
    key = ("batch", model_size, dtype_mode, "batch-8-224x224")
    _skip_run_rule(record, *key)
    with record_skip(record, *key, backend=BACKEND):
        model = backbone(model_size, BACKEND, dtype_mode)
    batch = np.concatenate([inputs[config] for config in BATCH_CONFIGS])
    batched = run_mlx(model, batch, dtype_mode)
    singles = [run_mlx(model, inputs[config], dtype_mode)[0] for config in BATCH_CONFIGS]
    check_finite(
        np.concatenate([batched, np.stack(singles)]),
        model_size,
        dtype_mode,
        lambda: record(*key, None, "non-finite", backend=BACKEND),
    )
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


@pytest.mark.parametrize(("model_size", "dtype_mode"), MODEL_PARAMS)
def test_mlx_token_range(
    model_size: str,
    dtype_mode: str,
    backbone: Loader,
    inputs: dict[str, np.ndarray],
    record: Callable[..., None],
) -> None:
    """W7: the token range of the 12 configurations. Report only: a non-finite value is a result.
    Needs no reference file."""
    _skip_run_rule(record, "token-range", model_size, dtype_mode, "all")
    with record_skip(record, "token-range", model_size, dtype_mode, "all", backend=BACKEND):
        model = backbone(model_size, BACKEND, dtype_mode)
    lines = 0
    for config in CONFIG_IDS:
        token_range = run_mlx_token_range(model, inputs[config])
        status = "measured" if token_range.finite else "non-finite"
        record(
            "token-range",
            model_size,
            dtype_mode,
            config,
            None,
            status,
            backend=BACKEND,
            values=token_range.report_values(),
        )
        lines += 1
    assert lines == len(CONFIG_IDS) == 12


@pytest.mark.parametrize(("model_size", "dtype_mode"), FP32_PARAMS)
def test_mlx_token_range_matches_stored(
    model_size: str,
    dtype_mode: str,
    backbone: Loader,
    reference: Reader,
    inputs: dict[str, np.ndarray],
    record: Callable[..., None],
) -> None:
    """W8: the token range in fp32 against the largest absolute value of each stored hidden
    state of iss-kibo-lab/224x224."""
    key = ("token-range-stored", model_size, dtype_mode, HIDDEN_STATES_CONFIG)
    model, stored = _load(backbone, reference, record, *key)
    expected = np.abs(stored[HIDDEN_STATES_KEY].astype(np.float64)).max(axis=(1, 2))
    tokens = run_mlx_token_range(model, inputs[HIDDEN_STATES_CONFIG]).tokens.astype(np.float64)
    relative = np.abs(tokens - expected) / expected
    values = {"tokens": tokens.tolist(), "stored": expected.tolist(), "relative": relative.tolist()}
    record(*key, None, "measured", backend=BACKEND, values=values)
    assert tokens.shape == expected.shape
    assert relative.max() <= TOKEN_RANGE_RELATIVE
