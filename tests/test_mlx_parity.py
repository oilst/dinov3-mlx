"""Tests W1 to W8 of specs/05-all-sizes/plan.html: the MLX backbone with the gated weights against
the stored hidden states and the reference files of Step 2, for each model size under test and
each of the five dtype modes, and the token range.

The backbone fixture checks the run rule and the free memory first, and skips with the reason when
the load breaks a run rule or does not fit. Each test calls run_rule_skip first, before it reads a
stored file or loads a model. W1 to W4 assert in fp32 and record the values in the four fp16 and
bf16 dtype modes (D8).
"""

import math
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

from dinov3_mlx.mlx_model import (
    MLX_DTYPE_MODES,
    MlxBackbone,
    mlx_embeddings,
    run_mlx,
    run_mlx_hidden_states,
    run_mlx_token_range,
)
from dinov3_mlx.outputs import NUM_PREFIX_TOKENS
from dinov3_mlx.parity import Parity, measure_parity
from dinov3_mlx.stored import HIDDEN_STATES_CONFIG

pytestmark = pytest.mark.mlx

BACKEND = "mlx"
HIDDEN_STATES_KEY = f"{HIDDEN_STATES_CONFIG}/hidden_states"
# An isolated part has no error from the blocks before it: ten times stricter on 1 - cosine than
# the fp32 threshold (Table 8 of the Step 4 plan).
PART_COSINE = 0.99999
# W8: the token range in fp32 against the stored hidden states, relative difference.
TOKEN_RANGE_RELATIVE = 1e-4
# The fields of a token-range line: L + 1 values each for the token sequence, L values each for
# the branches.
TOKEN_FIELDS = ["tokens", "tokens_where", "tokens_finite"]
BRANCH_FIELDS = ["attention", "ff_hidden", "ff_out"]

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
Forward = Callable[[mx.array], mx.array]
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
    mlx_forward: Callable[[MlxBackbone], Forward],
) -> None:
    """W5: the 12 configurations against the reference file, on the compiled forward pass (D4 of
    specs/06-speed/plan.html)."""
    key = ("parity", model_size, dtype_mode, config)
    model, stored = _load(backbone, reference, record, *key)
    tokens = run_mlx(model, inputs[config], dtype_mode, mlx_forward(model))[0]
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
    mlx_forward: Callable[[MlxBackbone], Forward],
) -> None:
    """W6: a batch of 8 against 8 single runs, on the compiled forward pass. Needs no reference
    file."""
    key = ("batch", model_size, dtype_mode, "batch-8-224x224")
    _skip_run_rule(record, *key)
    with record_skip(record, *key, backend=BACKEND):
        model = backbone(model_size, BACKEND, dtype_mode)
    forward = mlx_forward(model)
    batch = np.concatenate([inputs[config] for config in BATCH_CONFIGS])
    batched = run_mlx(model, batch, dtype_mode, forward)
    singles = [run_mlx(model, inputs[config], dtype_mode, forward)[0] for config in BATCH_CONFIGS]
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
    Needs no reference file.

    Each line holds L + 1 token values and 3 x L branch values, and its status agrees with its
    values (F6 of specs/05-all-sizes/verify.html).
    """
    _skip_run_rule(record, "token-range", model_size, dtype_mode, "all")
    with record_skip(record, "token-range", model_size, dtype_mode, "all", backend=BACKEND):
        model = backbone(model_size, BACKEND, dtype_mode)
    lines: dict[str, tuple[str, dict[str, list]]] = {}
    for config in CONFIG_IDS:
        token_range = run_mlx_token_range(model, inputs[config])
        status = "measured" if token_range.finite else "non-finite"
        values = token_range.report_values()
        record(
            "token-range",
            model_size,
            dtype_mode,
            config,
            None,
            status,
            backend=BACKEND,
            values=values,
        )
        lines[config] = (status, values)
    assert len(CONFIG_IDS) == 12
    assert list(lines) == CONFIG_IDS
    blocks = model.config.blocks
    for config, (status, values) in lines.items():
        for name in TOKEN_FIELDS:
            assert len(values[name]) == blocks + 1, (config, name)
        for name in BRANCH_FIELDS:
            assert len(values[name]) == blocks, (config, name)
        finite = all(values["tokens_finite"]) and all(
            math.isfinite(value) for name in ("tokens", *BRANCH_FIELDS) for value in values[name]
        )
        assert status == ("measured" if finite else "non-finite"), config


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


HALF_PARAMS = [
    pytest.param(model_size, dtype_mode, id=f"{BACKEND}-{model_size}-{dtype_mode}")
    for model_size in MLX_MODEL_SIZES_UNDER_TEST
    for dtype_mode in MLX_DTYPE_MODES
    if dtype_mode != "fp32"
]
# Table 21 of specs/06-speed/plan.html: the extra parity loss of the fp16 patch embeddings. The
# decisions of 2026-11-07 on W4 (a deviation from AC15): the patch mean keeps 1e-7; the patch 1st
# percentile, a tail value that one rounding moves more, gets a fault bound of 5e-6 (first 1e-6).
# Measured: patch mean at most 5.6e-8; patch 1st percentile up to 1.27e-6 (ViT-H+/16
# bf16-autocast), 3.4e-7 (ViT-S/16 bf16-autocast), 2.5e-7 (ViT-H+/16 fp16-autocast), 0 in the
# cast modes. The gate stays: each finite configuration passes THRESHOLDS[mode].
FP16_EXTRA_LOSS = {"patch_mean": 1e-7, "patch_p01": 5e-6}


@pytest.mark.parametrize(("model_size", "dtype_mode"), MODEL_PARAMS)
def test_compiled_equals_eager(
    model_size: str,
    dtype_mode: str,
    backbone: Loader,
    inputs: dict[str, np.ndarray],
    record: Callable[..., None],
    mlx_forward: Callable[[MlxBackbone], Forward],
) -> None:
    """W1 of specs/06-speed/plan.html (D4): the 12 configurations, run_mlx with the compiled
    forward pass of the slot against run_mlx eager. A finite output: largest difference 0.0. A
    non-finite output: both forms non-finite at the same positions."""
    key = ("compiled-equal", model_size, dtype_mode, "all")
    _skip_run_rule(record, *key)
    with record_skip(record, *key, backend=BACKEND):
        model = backbone(model_size, BACKEND, dtype_mode)
    forward = mlx_forward(model)
    failures = []
    for config in CONFIG_IDS:
        compiled = run_mlx(model, inputs[config], dtype_mode, forward)
        eager = run_mlx(model, inputs[config], dtype_mode)
        finite = bool(np.isfinite(compiled).all() and np.isfinite(eager).all())
        if finite:
            largest = float(np.abs(compiled - eager).max())
        else:
            largest = 0.0 if np.array_equal(compiled, eager, equal_nan=True) else math.inf
        status = "measured" if finite else "non-finite"
        record(
            "compiled-equal",
            model_size,
            dtype_mode,
            config,
            None,
            status,
            backend=BACKEND,
            values={"max_abs": largest},
        )
        if largest != 0.0:
            failures.append(f"{config}: {largest}")
    assert not failures, failures


@pytest.mark.parametrize(("model_size", "dtype_mode"), HALF_PARAMS)
def test_fp16_patches_parity(
    model_size: str,
    dtype_mode: str,
    backbone: Loader,
    reference: Reader,
    inputs: dict[str, np.ndarray],
    record: Callable[..., None],
    mlx_forward: Callable[[MlxBackbone], Forward],
) -> None:
    """W4 of specs/06-speed/plan.html (D13): the fp16 patch embeddings of mlx_embeddings on the
    compiled forward pass against the reference output: THRESHOLDS[mode] (or an expected failure
    through check_finite), an extra loss of at most 1e-7 for the patch mean and at most 5e-6 for
    the patch 1st percentile (FP16_EXTRA_LOSS, decisions of 2026-11-07), and the frame embeddings
    unchanged. The test measures each finite configuration first; a non-finite configuration of
    EXPECTED_FP16_OVERFLOW gives the expected failure at the end."""
    key = ("fp16-output", model_size, dtype_mode, "all")
    model, stored = _load(backbone, reference, record, *key)
    forward = mlx_forward(model)
    largest_extra = dict.fromkeys(FP16_EXTRA_LOSS, 0.0)
    below: list[str] = []
    non_finite: list[str] = []
    for config in CONFIG_IDS:
        tokens = forward(mx.array(inputs[config]))
        frame, patches = mlx_embeddings(tokens, dtype_mode)
        wide = np.array(tokens.astype(mx.float32))[0]
        assert patches.dtype == np.float16
        if not np.isfinite(wide).all():
            record(
                "fp16-output", model_size, dtype_mode, config, None, "non-finite", backend=BACKEND
            )
            non_finite.append(config)
            continue
        narrow = np.concatenate(
            [frame[0][None], wide[1:NUM_PREFIX_TOKENS], patches[0].astype(np.float32)]
        )
        fp32_parity = measure_parity(stored[config], wide)
        parity = measure_parity(stored[config], narrow)
        extra = {
            "patch_mean": fp32_parity.patch_mean - parity.patch_mean,
            "patch_p01": fp32_parity.patch_p01 - parity.patch_p01,
        }
        frame_equal = bool(np.array_equal(frame[0], wide[0]))
        record(
            "fp16-output",
            model_size,
            dtype_mode,
            config,
            parity,
            "measured",
            backend=BACKEND,
            values={"extra_loss": extra, "frame_equal": frame_equal},
        )
        assert frame_equal, config
        try:
            assert_thresholds(parity, dtype_mode)
        except AssertionError:
            below.append(config)
        largest_extra = {m: max(largest_extra[m], extra[m]) for m in FP16_EXTRA_LOSS}
    assert not below, below
    assert all(largest_extra[m] <= FP16_EXTRA_LOSS[m] for m in FP16_EXTRA_LOSS), largest_extra
    if non_finite:  # an expected fp16 overflow xfails, any other fails (way M2)
        check_finite(np.array([np.nan]), model_size, dtype_mode, lambda: None)
