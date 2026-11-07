"""Tests T10 to T20 of specs/04-mlx-port/plan.html and T16 to T26 of specs/05-all-sizes/plan.html:
each part of the MLX backbone against the Hugging Face modules on torch-cpu, with the small models
of tests/small_model.py, and the dtype of each part in each dtype mode.

MLX runs on the GPU with TF32 off (tests/conftest.py). Each test records its measured value next to
its target in tests/fixtures/generated/mlx/report.jsonl (Table 7 of the Step 4 plan, Table 13 of the
Step 5 plan).
"""

import gc
import os
from collections.abc import Callable

import mlx.core as mx
import numpy as np
import pytest
import torch
from mlx import nn
from mlx.utils import tree_flatten
from parity_checks import min_token_cosine
from rope_oracle import rotate_half
from small_model import (
    PIXEL_SHAPES,
    SMALL_CONFIGS,
    small_hf_model,
    small_mlx_model,
    small_pixels,
)
from transformers.activations import ACT2FN
from transformers.models.dinov3_vit.modeling_dinov3_vit import DINOv3ViTModel

from dinov3_apple import mlx_model as mlx_model_module
from dinov3_apple.block_profile import block_parts, run_parts
from dinov3_apple.flops import PART_NAMES
from dinov3_apple.mlx_bench import mlx_block_parts
from dinov3_apple.mlx_model import (
    MLP,
    MLX_DTYPE_MODES,
    Block,
    MlxBackbone,
    MlxConfig,
    SwiGLU,
    TokenRange,
    apply_rope,
    linear_parameter_names,
    patchify,
    rope_layout,
    rope_tables,
    run_mlx,
    run_mlx_hidden_states,
    run_mlx_token_range,
)
from dinov3_apple.outputs import NUM_PREFIX_TOKENS, PATCH_SIZE, split_tokens

pytestmark = pytest.mark.mlx

# Table 7 of the plan: the largest difference of each part, in fp32.
PART_TARGETS = {
    "norm1": 1e-5,
    "qkv_proj": 1e-5,
    "rope": 2e-6,
    "sdpa": 5e-6,
    "o_proj": 1e-5,
    "layer_scale1": 0.0,
    "norm2": 1e-5,
    "ff_in": 1e-5,
    "activation": 2e-6,
    "ff_out": 2e-5,
    "layer_scale2": 0.0,
}
# Table 13 of specs/05-all-sizes/plan.html: the parts of SwiGLU. The other parts keep Table 7.
SWIGLU_PART_TARGETS = PART_TARGETS | {"ff_in": 1e-5, "activation": 1e-5, "ff_out": 2e-5}
# The pixels of the part tests: the batch of 2 at 224 x 224 px.
PART_PIXELS = PIXEL_SHAPES[0]
SHAPE_IDS = ["x".join(map(str, shape)) for shape in PIXEL_SHAPES]
# The grids of the attention and block tests: 14 x 14 and 14 x 19 patches.
GRID_SHAPES = [(2, 3, 224, 224), (1, 3, 224, 304)]
GRID_IDS = ["14x14", "14x19"]
# The small models with SwiGLU. The vits16 cases keep the test ids of Step 4.
SWIGLU_MODELS = ["vits16plus", "vit7b16-shape"]
DTYPE_MODES = list(MLX_DTYPE_MODES)
HALF_DTYPES = {"fp16": (mx.float16, torch.float16), "bf16": (mx.bfloat16, torch.bfloat16)}


def _model_params(names: list[str], cases: list, case_ids: list[str]) -> list:
    """Parameters (model name, case): no prefix for vits16, the model name for the others."""
    return [
        pytest.param(name, case, id=case_id if name == "vits16" else f"{name}-{case_id}")
        for name in names
        for case, case_id in zip(cases, case_ids, strict=True)
    ]


@pytest.fixture(scope="module")
def hf_model() -> DINOv3ViTModel:
    """Return the small Hugging Face backbone vits16, seed 0 (the oracle of the part tests)."""
    return small_hf_model(seed=0)


@pytest.fixture(scope="module")
def mlx_model(hf_model: DINOv3ViTModel) -> MlxBackbone:
    """Return the small MLX backbone with the weights of hf_model."""
    return small_mlx_model(hf_model)


@pytest.fixture(scope="module")
def small_models() -> Callable[[str], tuple[DINOv3ViTModel, MlxBackbone]]:
    """Return a getter of the small models by name, seed 0: the Hugging Face oracle and the MLX
    backbone in fp32. Each model is made once."""
    models: dict[str, tuple[DINOv3ViTModel, MlxBackbone]] = {}

    def get(name: str) -> tuple[DINOv3ViTModel, MlxBackbone]:
        if name not in models:
            hf = small_hf_model(seed=0, name=name)
            models[name] = (hf, small_mlx_model(hf))
        return models[name]

    return get


def _difference(expected: torch.Tensor | np.ndarray, output: mx.array) -> float:
    if isinstance(expected, torch.Tensor):
        expected = expected.numpy()
    assert tuple(output.shape) == expected.shape
    return float(np.abs(np.array(output, dtype=np.float64) - expected.astype(np.float64)).max())


def _rope_tables(hf_model: DINOv3ViTModel, pixels: torch.Tensor) -> tuple[mx.array, mx.array]:
    """The Hugging Face tables of one grid, plus 5 identity rows on top, in the layout of the
    concat-free rotation (rope_layout)."""
    cos, sin = hf_model.rope_embeddings(pixels)
    ones = np.ones((NUM_PREFIX_TOKENS, cos.shape[-1]), np.float32)
    cos2, sin2 = rope_layout(
        np.concatenate([ones, cos.numpy()]), np.concatenate([0 * ones, sin.numpy()])
    )
    return mx.array(cos2), mx.array(sin2)


def _mlx_parts(block: Block) -> dict[str, Callable[..., tuple[mx.array, ...]]]:
    """The 11 parts of one MLX block (moved to mlx_bench.mlx_block_parts in Step 6)."""
    return mlx_block_parts(block)


@pytest.mark.parametrize(
    ("model", "name"), _model_params(["vits16", "vits16plus"], PART_NAMES, PART_NAMES)
)
def test_mlx_block_part(
    model: str,
    name: str,
    small_models: Callable[[str], tuple[DINOv3ViTModel, MlxBackbone]],
    report: Callable[..., None],
) -> None:
    """T16: each part of block 1 against block_parts; for SwiGLU, ff_in gives (gate, up)."""
    hf_model, mlx_model = small_models(model)
    targets = PART_TARGETS if model == "vits16" else SWIGLU_PART_TARGETS
    pixels = torch.from_numpy(small_pixels(PART_PIXELS))
    layer = hf_model.model.layer[0]
    with torch.inference_mode():
        x = hf_model.embeddings(pixels)
        cos, sin = hf_model.rope_embeddings(pixels)
        state = run_parts(block_parts(layer), {"x": x, "cos": cos, "sin": sin})
        mlx_cos, mlx_sin = _rope_tables(hf_model, pixels)
    part = next(part for part in block_parts(layer) if part.name == name)
    tables = {"cos": mlx_cos, "sin": mlx_sin}
    args = [
        tables[i] if i in tables else mx.array(np.ascontiguousarray(state[i].numpy()))
        for i in part.inputs
    ]
    outputs = _mlx_parts(mlx_model.layer[0])[name](*args)
    assert len(outputs) == len(part.outputs)
    difference = max(
        _difference(state[out], output) for out, output in zip(part.outputs, outputs, strict=True)
    )
    report(
        "test_mlx_block_part", model=model, part=name, difference=difference, target=targets[name]
    )
    assert difference <= targets[name]


@pytest.mark.parametrize("shape", PIXEL_SHAPES, ids=SHAPE_IDS)
def test_mlx_part_patch_embedding(
    shape: tuple[int, ...],
    hf_model: DINOv3ViTModel,
    mlx_model: MlxBackbone,
    report: Callable[..., None],
) -> None:
    pixels = small_pixels(shape)
    with torch.inference_mode():
        expected = hf_model.embeddings(torch.from_numpy(pixels)).numpy()
    output = np.array(mlx_model.embeddings(mx.array(pixels)))
    assert output.shape == expected.shape
    prefix_exact = np.array_equal(output[:, :NUM_PREFIX_TOKENS], expected[:, :NUM_PREFIX_TOKENS])
    patch = float(np.abs(output[:, NUM_PREFIX_TOKENS:] - expected[:, NUM_PREFIX_TOKENS:]).max())
    report(
        "test_mlx_part_patch_embedding",
        shape=list(shape),
        patch=patch,
        prefix_exact=prefix_exact,
        target=1e-5,
    )
    assert prefix_exact
    assert patch <= 1e-5


def test_patchify_rejects_bad_size() -> None:
    with pytest.raises(ValueError, match="multiple of 16"):
        patchify(mx.zeros((1, 3, 224, 230)), PATCH_SIZE)


def test_patchify_order() -> None:
    """Token n = Wp i + j; inside one patch the order is channel, row, column (as the weight)."""
    pixels = np.arange(2 * 3 * 32 * 48, dtype=np.float32).reshape(2, 3, 32, 48)
    patches = np.array(patchify(mx.array(pixels), PATCH_SIZE))
    assert patches.shape == (2, 2 * 3, 3 * 16 * 16)
    i, j = 1, 2
    expected = pixels[1, :, 16 * i : 16 * i + 16, 16 * j : 16 * j + 16].reshape(-1)
    assert np.array_equal(patches[1, 3 * i + j], expected)


def test_mlx_forward_casts_the_pixels_to_the_matmul_dtype(
    hf_model: DINOv3ViTModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T25: fp32 pixels get the matmul dtype. bf16-cast gives bf16 embeddings and tokens;
    bf16-autocast gives bf16 pixels into patchify and fp32 embeddings and tokens."""
    pixels = mx.array(small_pixels(PART_PIXELS))
    model = small_mlx_model(hf_model, "bf16-cast")
    embeddings = model.embeddings(pixels)
    assert embeddings.dtype == mx.bfloat16
    assert mx.array_equal(embeddings, model.embeddings(pixels.astype(mx.bfloat16))).item()
    assert model(pixels).dtype == mx.bfloat16
    pixel_dtypes: list[mx.Dtype] = []

    def record_patchify(x: mx.array, patch_size: int) -> mx.array:
        pixel_dtypes.append(x.dtype)
        return patchify(x, patch_size)

    monkeypatch.setattr(mlx_model_module, "patchify", record_patchify)
    model = small_mlx_model(hf_model, "bf16-autocast")
    assert model.embeddings(pixels).dtype == mx.float32
    assert model(pixels).dtype == mx.float32
    assert pixel_dtypes == [mx.bfloat16, mx.bfloat16]


def test_mlx_quantized_patch_embedding_gets_float_pixels(
    hf_model: DINOv3ViTModel,
    mlx_model: MlxBackbone,
    report: Callable[..., None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """nn.quantize packs the patch embedding weight into uint32; the pixels stay floating point.

    F1 of specs/04-mlx-port/verify.html: a cast to the weight dtype turned the pixels into uint32,
    without an error message. The default predicate of nn.quantize quantizes each Linear, also
    the patch embedding (768 inputs, a multiple of the group size 64).

    The bound 0.999 on the min token cosine: with 8-bit weights the small model gives 0.99998 for
    the patch tokens of the embeddings and of run_mlx (measured with mlx 0.32.4). 4-bit weights
    give about 0.995, the uint32 pixels about 0.63. So the bound leaves a margin of about 50 times
    on 1 - cosine for 8 bits and still finds the wrong cast.
    """
    quantized = small_mlx_model(hf_model)
    nn.quantize(quantized, bits=8)
    assert isinstance(quantized.embeddings.patch_embeddings, nn.QuantizedLinear)
    assert quantized.embeddings.patch_embeddings.weight.dtype == mx.uint32
    pixels = small_pixels(PART_PIXELS)
    expected = np.array(mlx_model.embeddings(mx.array(pixels)))
    pixel_dtypes: list[mx.Dtype] = []

    def record_patchify(x: mx.array, patch_size: int) -> mx.array:
        pixel_dtypes.append(x.dtype)
        return patchify(x, patch_size)

    monkeypatch.setattr(mlx_model_module, "patchify", record_patchify)
    output = np.array(quantized.embeddings(mx.array(pixels)))
    assert pixel_dtypes == [mx.float32]
    dim = expected.shape[-1]
    embeddings = min_token_cosine(
        expected[:, NUM_PREFIX_TOKENS:].reshape(-1, dim),
        output[:, NUM_PREFIX_TOKENS:].reshape(-1, dim),
    )
    tokens = run_mlx(quantized, pixels, "fp32")
    reference_tokens = run_mlx(mlx_model, pixels, "fp32")
    assert tokens.dtype == np.float32
    run = min_token_cosine(reference_tokens.reshape(-1, dim), tokens.reshape(-1, dim))
    report(
        "test_mlx_quantized_patch_embedding_gets_float_pixels",
        bits=8,
        embeddings_min_cosine=embeddings,
        tokens_min_cosine=run,
        target=0.999,
    )
    assert embeddings >= 0.999
    assert run >= 0.999


@pytest.mark.parametrize(
    ("model", "shape"), _model_params(["vits16", "vit7b16-shape"], GRID_SHAPES, GRID_IDS)
)
def test_mlx_part_attention(
    model: str,
    shape: tuple[int, ...],
    small_models: Callable[[str], tuple[DINOv3ViTModel, MlxBackbone]],
    report: Callable[..., None],
) -> None:
    """T19 for vit7b16-shape: no q, k, v biases and head dimension 128."""
    hf_model, mlx_model = small_models(model)
    pixels = torch.from_numpy(small_pixels(shape))
    layer = hf_model.model.layer[0]
    with torch.inference_mode():
        h = layer.norm1(hf_model.embeddings(pixels))
        expected, _ = layer.attention(h, position_embeddings=hf_model.rope_embeddings(pixels))
    cos, sin = mlx_model.rope(shape[2] // PATCH_SIZE, shape[3] // PATCH_SIZE)
    output = mlx_model.layer[0].attention(mx.array(h.numpy()), cos, sin)
    difference = _difference(expected, output)
    report(
        "test_mlx_part_attention",
        model=model,
        shape=list(shape),
        difference=difference,
        target=1e-5,
    )
    assert difference <= 1e-5


def test_mlx_part_mlp(
    hf_model: DINOv3ViTModel, mlx_model: MlxBackbone, report: Callable[..., None]
) -> None:
    pixels = torch.from_numpy(small_pixels(PART_PIXELS))
    layer = hf_model.model.layer[0]
    with torch.inference_mode():
        cos, sin = hf_model.rope_embeddings(pixels)
        x = hf_model.embeddings(pixels)
        h2 = run_parts(block_parts(layer), {"x": x, "cos": cos, "sin": sin})["h2"]
        expected = layer.mlp(h2)
    output = mlx_model.layer[0].mlp(mx.array(h2.numpy()))
    difference = _difference(expected, output)
    report("test_mlx_part_mlp", difference=difference, target=2e-5)
    assert difference <= 2e-5


def test_mlx_part_gelu_is_exact_form(report: Callable[..., None]) -> None:
    values = np.linspace(-6, 6, 200_001, dtype=np.float32)
    expected = ACT2FN["gelu"](torch.from_numpy(values)).numpy()
    exact = float(np.abs(np.array(nn.gelu(mx.array(values))) - expected).max())
    approx = float(np.abs(np.array(nn.gelu_approx(mx.array(values))) - expected).max())
    report("test_mlx_part_gelu_is_exact_form", exact=exact, approx=approx, target=2e-6)
    assert exact <= 2e-6
    assert approx >= 1e-4


@pytest.mark.parametrize(
    ("model", "shape"), _model_params(["vits16", *SWIGLU_MODELS], GRID_SHAPES, GRID_IDS)
)
def test_mlx_part_block(
    model: str,
    shape: tuple[int, ...],
    small_models: Callable[[str], tuple[DINOv3ViTModel, MlxBackbone]],
    report: Callable[..., None],
) -> None:
    """T20 for the SwiGLU models: the block against DINOv3ViTLayer."""
    hf_model, mlx_model = small_models(model)
    pixels = torch.from_numpy(small_pixels(shape))
    with torch.inference_mode():
        x = hf_model.embeddings(pixels)
        expected = hf_model.model.layer[0](x, position_embeddings=hf_model.rope_embeddings(pixels))
    cos, sin = mlx_model.rope(shape[2] // PATCH_SIZE, shape[3] // PATCH_SIZE)
    output = mlx_model.layer[0](mx.array(x.numpy()), cos, sin)
    difference = _difference(expected, output)
    report(
        "test_mlx_part_block", model=model, shape=list(shape), difference=difference, target=3e-5
    )
    assert difference <= 3e-5


def test_mlx_part_final_norm(
    hf_model: DINOv3ViTModel, mlx_model: MlxBackbone, report: Callable[..., None]
) -> None:
    pixels = torch.from_numpy(small_pixels(PART_PIXELS))
    with torch.inference_mode():
        last = hf_model(pixel_values=pixels, output_hidden_states=True).hidden_states[-1]
        expected = hf_model.norm(last)
    output = mlx_model.norm(mx.array(last.numpy()))
    difference = _difference(expected, output)
    report("test_mlx_part_final_norm", difference=difference, target=1e-5)
    assert difference <= 1e-5


@pytest.mark.parametrize(
    ("model", "shape"), _model_params(["vits16", *SWIGLU_MODELS], PIXEL_SHAPES, SHAPE_IDS)
)
def test_mlx_small_model_tokens(
    model: str,
    shape: tuple[int, ...],
    small_models: Callable[[str], tuple[DINOv3ViTModel, MlxBackbone]],
    report: Callable[..., None],
) -> None:
    """T21 for the SwiGLU models: run_mlx and run_mlx_hidden_states."""
    hf_model, mlx_model = small_models(model)
    pixels = small_pixels(shape)
    with torch.inference_mode():
        output = hf_model(pixel_values=torch.from_numpy(pixels), output_hidden_states=True)
    tokens = run_mlx(mlx_model, pixels, "fp32")
    tokens_difference = float(np.abs(tokens - output.last_hidden_state.numpy()).max())
    values = {"tokens": tokens_difference}
    if shape[0] == 1:
        hidden = run_mlx_hidden_states(mlx_model, pixels)
        expected = torch.stack([state[0] for state in output.hidden_states]).numpy()
        assert hidden.shape == expected.shape == (3, *tokens.shape[1:])
        values["hidden_states"] = float(np.abs(hidden - expected).max())
    report("test_mlx_small_model_tokens", model=model, shape=list(shape), **values, target=5e-5)
    assert max(values.values()) <= 5e-5


def test_run_mlx_contract(mlx_model: MlxBackbone) -> None:
    height, width = 224, 304
    pixels = small_pixels((2, 3, height, width))
    tokens = run_mlx(mlx_model, pixels, "fp32")
    n = (height // PATCH_SIZE) * (width // PATCH_SIZE)
    assert isinstance(tokens, np.ndarray)
    assert tokens.dtype == np.float32
    assert tokens.shape == (2, NUM_PREFIX_TOKENS + n, 384)
    frame, patches = split_tokens(tokens, height, width)
    assert frame.shape == (2, 384)
    assert patches.shape == (2, height // PATCH_SIZE, width // PATCH_SIZE, 384)
    # The RoPE cache is not a parameter: the tree keeps its parameters after a forward pass.
    assert len(tree_flatten(mlx_model.parameters())) == 2 * 17 + 4 + 2


@pytest.mark.parametrize("shape", [(2, 6, 201, 64), (1, 6, 1029, 64)], ids=["201", "1029"])
def test_mlx_sdpa_float64(shape: tuple[int, ...], report: Callable[..., None]) -> None:
    rng = np.random.default_rng(0)
    q, k, v = (rng.standard_normal(shape).astype(np.float32) for _ in range(3))
    output = mx.fast.scaled_dot_product_attention(
        mx.array(q), mx.array(k), mx.array(v), scale=0.125, mask=None
    )
    scores = q.astype(np.float64) @ k.astype(np.float64).swapaxes(-1, -2) * 0.125
    weights = np.exp(scores - scores.max(-1, keepdims=True))
    weights /= weights.sum(-1, keepdims=True)
    expected = weights @ v.astype(np.float64)
    difference = float(np.abs(np.array(output, dtype=np.float64) - expected).max())
    report("test_mlx_sdpa_float64", shape=list(shape), difference=difference, target=5e-6)
    assert mx.default_device() == mx.gpu
    assert difference <= 5e-6


@pytest.mark.parametrize("model", SWIGLU_MODELS)
def test_mlx_part_swiglu(
    model: str,
    small_models: Callable[[str], tuple[DINOv3ViTModel, MlxBackbone]],
    report: Callable[..., None],
) -> None:
    """T17: SwiGLU against DINOv3ViTGatedMLP on the input of norm2."""
    hf_model, mlx_model = small_models(model)
    pixels = torch.from_numpy(small_pixels(PART_PIXELS))
    layer = hf_model.model.layer[0]
    with torch.inference_mode():
        cos, sin = hf_model.rope_embeddings(pixels)
        x = hf_model.embeddings(pixels)
        h2 = run_parts(block_parts(layer), {"x": x, "cos": cos, "sin": sin})["h2"]
        expected = layer.mlp(h2)
    mlp = mlx_model.layer[0].mlp
    assert isinstance(mlp, SwiGLU)
    difference = _difference(expected, mlp(mx.array(h2.numpy())))
    report("test_mlx_part_swiglu", model=model, difference=difference, target=2e-5)
    assert difference <= 2e-5


def test_mlx_part_silu(report: Callable[..., None]) -> None:
    """T18: nn.silu in fp32 against ACT2FN["silu"]."""
    values = np.linspace(-8, 8, 200_001, dtype=np.float32)
    expected = ACT2FN["silu"](torch.from_numpy(values)).numpy()
    difference = float(np.abs(np.array(nn.silu(mx.array(values))) - expected).max())
    report("test_mlx_part_silu", difference=difference, target=4e-6)
    assert difference <= 4e-6


def _expected_dtypes(dtype_mode: str) -> tuple[mx.Dtype, mx.Dtype]:
    dtypes = MLX_DTYPE_MODES[dtype_mode]
    return dtypes.matmul, dtypes.stream


@pytest.mark.parametrize(
    ("model", "dtype_mode"),
    [
        pytest.param(model, mode, id=f"{model}-{mode}")
        for model in ("vits16", "vits16plus")
        for mode in DTYPE_MODES
    ],
)
def test_mlx_dtype_flow(model: str, dtype_mode: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """T22: the dtype of each part in each dtype mode (Table 3 of the plan).

    The attention branch of each block calls apply_rope twice, with q and k in the matmul dtype
    and the fp32 RoPE tables. So a cast of the tables or a rotation outside apply_rope fails here
    (F7 of specs/05-all-sizes/verify.html).
    """
    matmul, stream = _expected_dtypes(dtype_mode)
    mlx = small_mlx_model(small_hf_model(seed=0, name=model), dtype_mode)
    assert mlx.dtype_mode == dtype_mode
    linear = linear_parameter_names(mlx)
    for name, value in tree_flatten(mlx.parameters()):
        assert value.dtype == (matmul if name in linear else stream), name
    pixels = small_pixels(PART_PIXELS)
    x = mlx.embeddings(mx.array(pixels))
    assert x.dtype == stream
    cos, sin = mlx.rope(14, 14)
    assert cos.dtype == sin.dtype == mx.float32
    assert cos.shape == sin.shape == (201, 2, mlx.config.head_dim // 2)
    rotations: list[tuple[mx.Dtype, mx.Dtype, mx.Dtype]] = []

    def record_rope(values: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        rotations.append((values.dtype, cos.dtype, sin.dtype))
        return apply_rope(values, cos, sin)

    for block in mlx.layer:
        rotations.clear()
        with monkeypatch.context() as patch:
            patch.setattr(mlx_model_module, "apply_rope", record_rope)
            assert block.attention_branch(x, cos, sin).dtype == matmul
        assert rotations == [(matmul, mx.float32, mx.float32)] * 2
        assert block.ff_hidden(x).dtype == matmul
        x = block(x, cos, sin)
        assert x.dtype == stream
    assert mlx(mx.array(pixels)).dtype == stream
    tokens = run_mlx(mlx, pixels, dtype_mode)
    assert isinstance(tokens, np.ndarray)
    assert tokens.dtype == np.float32


def _ordered(values: torch.Tensor) -> np.ndarray:
    """Map half values to integers whose differences count units in the last place."""
    bits = values.view(torch.int16).numpy().astype(np.int32)
    magnitude = bits & 0x7FFF
    return np.where(bits < 0, -magnitude, magnitude)


def _to_torch(output: mx.array, half: torch.dtype) -> torch.Tensor:
    """Return a half MLX array as a torch tensor of the same half dtype (exact)."""
    return torch.from_numpy(np.array(output.astype(mx.float32))).to(half)


def _compare(output: torch.Tensor, oracle: torch.Tensor) -> tuple[float, int]:
    """Return the share of values that differ and the largest difference in units in the last
    place, of a half output against the half oracle."""
    ulps = np.abs(_ordered(output) - _ordered(oracle))
    return float((ulps > 0).mean()), int(ulps.max())


@pytest.mark.parametrize("dtype", list(HALF_DTYPES))
@pytest.mark.parametrize("function", ["gelu", "swiglu"])
def test_mlx_activation_rounds_once(function: str, dtype: str, report: Callable[..., None]) -> None:
    """T23: the activation computes in fp32 and rounds once to the half dtype.

    The oracle is the float64 function, rounded once by torch. GELU below x = -2.5 uses nn.gelu on
    the fp32 values as the oracle: there 1 + erf cancels in fp32, so even a correct fp32 GELU
    differs from the float64 value. The activation in the half dtype misses the target.
    """
    mx_half, half = HALF_DTYPES[dtype]
    x = torch.linspace(-8, 8, 100_001, dtype=torch.float64).to(half)
    x32 = x.float().numpy()
    x_mlx = mx.array(x32).astype(mx_half)
    if function == "gelu":
        module = MLP(MlxConfig.from_hf(SMALL_CONFIGS["vits16"]))
        exact = torch.nn.functional.gelu(x.double()).to(half)
        fp32 = torch.from_numpy(np.array(nn.gelu(mx.array(x32)))).to(half)
        tail = torch.from_numpy(x32 < -2.5)
        oracle = torch.where(tail, fp32, exact)
        output = module.activation(x_mlx)
        in_half = nn.gelu(x_mlx)
        keep = ~tail
    else:
        module = SwiGLU(MlxConfig.from_hf(SMALL_CONFIGS["vits16plus"]))
        up = x.flip(0)
        up_mlx = mx.array(up.float().numpy()).astype(mx_half)
        oracle = (torch.nn.functional.silu(x.double()) * up.double()).to(half)
        output = module.activation(x_mlx, up_mlx)
        in_half = nn.silu(x_mlx) * up_mlx
        keep = torch.ones(len(x32), dtype=torch.bool)
    assert output.dtype == mx_half
    output = _to_torch(output, half)
    share, ulps = _compare(output[keep], oracle[keep])
    tail_share = _compare(output[~keep], oracle[~keep])[0] if (~keep).any() else 0.0
    half_share, half_ulps = _compare(_to_torch(in_half, half), oracle)
    report(
        "test_mlx_activation_rounds_once",
        function=function,
        dtype=dtype,
        share=share,
        ulps=ulps,
        tail_share=tail_share,
        half_share=half_share,
        half_ulps=half_ulps,
        target_share=1e-3,
    )
    assert share <= 1e-3
    assert ulps <= 1
    assert tail_share == 0.0
    assert half_share > 1e-3 or half_ulps > 1


def test_run_mlx_rejects_a_mode_mismatch(hf_model: DINOv3ViTModel) -> None:
    """T24: run_mlx checks its dtype mode against the dtype mode of the model."""
    model = small_mlx_model(hf_model, "bf16-cast")
    with pytest.raises(ValueError, match="bf16-cast"):
        run_mlx(model, small_pixels(PART_PIXELS), "fp32")


@pytest.mark.parametrize(
    ("model", "dtype_mode"),
    [
        pytest.param(model, mode, id=f"{model}-{mode}")
        for model in ("vits16", "vits16plus")
        for mode in DTYPE_MODES
    ],
)
def test_token_range_matches_hidden_states(
    model: str, dtype_mode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T26: the token range and run_mlx_hidden_states agree exactly.

    The branch values equal the largest absolute values of Block.attention_branch, Block.ff_hidden
    and mlp.down_proj, computed apart from run_mlx_token_range: the outputs of the two methods in
    the forward pass of run_mlx_hidden_states, and mlp.down_proj of each recorded ff_hidden output
    (F6 of specs/05-all-sizes/verify.html).
    """
    mlx = small_mlx_model(small_hf_model(seed=0, name=model), dtype_mode)
    pixels = small_pixels((1, 3, 224, 304))
    token_range = run_mlx_token_range(mlx, pixels)
    outputs: dict[str, list[mx.array]] = {"attention": [], "ff_hidden": []}
    attention_branch, ff_hidden = Block.attention_branch, Block.ff_hidden

    def record_attention(block: Block, x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        outputs["attention"].append(output := attention_branch(block, x, cos, sin))
        return output

    def record_ff_hidden(block: Block, x: mx.array) -> mx.array:
        outputs["ff_hidden"].append(output := ff_hidden(block, x))
        return output

    with monkeypatch.context() as patch:
        patch.setattr(Block, "attention_branch", record_attention)
        patch.setattr(Block, "ff_hidden", record_ff_hidden)
        hidden = run_mlx_hidden_states(mlx, pixels)
    ff_out = [
        block.mlp.down_proj(h) for block, h in zip(mlx.layer, outputs["ff_hidden"], strict=True)
    ]
    blocks = len(mlx.layer)
    assert token_range.tokens.shape == (blocks + 1,)
    assert token_range.tokens_where.shape == (blocks + 1,)
    assert token_range.tokens_finite.shape == (blocks + 1,)
    for values in (token_range.attention, token_range.ff_hidden, token_range.ff_out):
        assert values.shape == (blocks,)
        assert values.dtype == np.float32
    assert token_range.last_state.dtype == np.float32
    assert np.array_equal(token_range.last_state, hidden[-1])
    for i, state in enumerate(hidden):
        assert token_range.tokens[i] == np.abs(state).max(), i
        assert token_range.tokens_where[i] == np.abs(state).max(-1).argmax(), i
        assert bool(token_range.tokens_finite[i]) == bool(np.isfinite(state).all()), i

    def largest(values: list[mx.array]) -> list[float]:
        return [mx.abs(v.astype(mx.float32)).max().item() for v in values]

    assert token_range.attention.tolist() == largest(outputs["attention"])
    assert token_range.ff_hidden.tolist() == largest(outputs["ff_hidden"])
    assert token_range.ff_out.tolist() == largest(ff_out)
    assert token_range.finite


def test_token_range_finite_sees_each_inf() -> None:
    """TokenRange.finite is False if one token value or one branch value is not finite. It sets
    the status of each W7 line (F6 of specs/05-all-sizes/verify.html)."""
    blocks = 2
    finite = {
        "tokens": np.ones(blocks + 1, np.float32),
        "tokens_where": np.zeros(blocks + 1, np.int64),
        "tokens_finite": np.ones(blocks + 1, bool),
        "attention": np.ones(blocks, np.float32),
        "ff_hidden": np.ones(blocks, np.float32),
        "ff_out": np.ones(blocks, np.float32),
        "last_state": np.zeros((5, 4), np.float32),
    }
    assert TokenRange(**finite).finite
    assert not TokenRange(**finite | {"tokens_finite": np.array([True, False, True])}).finite
    for name in ("tokens", "attention", "ff_hidden", "ff_out"):
        values = finite[name].copy()
        values[1] = np.inf
        assert not TokenRange(**finite | {name: values}).finite, name

    # An inf in the down_proj bias of block 1 of a small model: the feed-forward branch output of
    # block 1 and the token sequence after it are not finite.
    mlx = small_mlx_model(small_hf_model(seed=0), "fp32")
    down_proj = mlx.layer[1].mlp.down_proj
    down_proj.bias = mx.where(mx.arange(down_proj.bias.shape[0]) == 0, np.inf, down_proj.bias)
    token_range = run_mlx_token_range(mlx, small_pixels((1, 3, 224, 224)))
    assert token_range.tokens_finite.tolist() == [True, True, False]
    assert np.isfinite(token_range.ff_out[0])
    assert np.isinf(token_range.ff_out[1])
    assert not token_range.finite


@pytest.mark.parametrize("dtype", list(HALF_DTYPES))
def test_mlx_rope_rotation_rounds_once(dtype: str, report: Callable[..., None]) -> None:
    """The RoPE rotation of half queries and keys runs in fp32 and rounds once to the half dtype
    (row "RoPE tables and rotation" of Table 3 of the plan; F7 of specs/05-all-sizes/verify.html).

    The oracle is the rotation in NumPy fp32, rounded once by torch. A rotation in the half dtype,
    with the tables cast to half as in the Hugging Face code, misses this target.
    """
    mx_half, half = HALF_DTYPES[dtype]
    head_dim = 64
    cos, sin = rope_tables(14, 19, head_dim, 100.0)
    cos_mlx, sin_mlx = (mx.array(table) for table in rope_layout(cos, sin))
    rng = np.random.default_rng(0)
    shape = (2, 6, NUM_PREFIX_TOKENS + 14 * 19, head_dim)
    for name in ("q", "k"):
        x = mx.array(rng.standard_normal(shape).astype(np.float32)).astype(mx_half)
        output = apply_rope(x, cos_mlx, sin_mlx)
        assert output.dtype == mx_half
        x32 = np.array(x.astype(mx.float32))
        rotated = np.concatenate([-x32[..., head_dim // 2 :], x32[..., : head_dim // 2]], axis=-1)
        oracle = torch.from_numpy(x32 * cos + rotated * sin).to(half)
        in_half = x * mx.array(cos).astype(mx_half) + rotate_half(x) * mx.array(sin).astype(mx_half)
        share = float((_to_torch(output, half) != oracle).float().mean())
        half_share = float((_to_torch(in_half, half) != oracle).float().mean())
        report(
            "test_mlx_rope_rotation_rounds_once",
            dtype=dtype,
            tensor=name,
            share=share,
            half_share=half_share,
            target_share=0.0,
        )
        assert share == 0.0, name
        assert half_share > 0.01, name


# Build step 2 of specs/06-speed/plan.html: the compiled forward pass (D1, D4).
SMALL_NAMES = ["vits16", "vits16plus", "vit7b16-shape"]
DEVICES = {"gpu": mx.gpu, "cpu": mx.cpu}
# R37: on the CPU stream the compiled and the eager forward pass differ by up to 2.0e-6 in fp32.
CPU_COMPILE_TOLERANCE = 1e-5
# The decision of 2026-11-07 on T34 (a deviation from AC1): on the CPU stream the four fp16 and
# bf16 dtype modes are report only, with a fault bound for each mode, at least 5 times its worst
# measured value (fp16-cast 0.0059, bf16-cast 0.047, fp16-autocast 0.0024, bf16-autocast 0.017).
# The compiled CPU kernel rounds once for each fused chain; the eager CPU path rounds after each
# operation. So 1 to 1.5 units in the last place of the half dtype are expected.
CPU_FAULT_BOUND = {
    "fp16-cast": 0.03,
    "bf16-cast": 0.25,
    "fp16-autocast": 0.015,
    "bf16-autocast": 0.1,
}


@pytest.mark.parametrize(
    ("device", "model", "dtype_mode"),
    [
        pytest.param(device, model, mode, id=f"{device}-{model}-{mode}")
        for device in DEVICES
        for model in SMALL_NAMES
        for mode in DTYPE_MODES
    ],
)
def test_compiled_equals_eager_small(
    device: str, model: str, dtype_mode: str, report: Callable[..., None]
) -> None:
    """T34 (D4, R37): compile_backbone against the eager forward pass, 3 pixel shapes. GPU: bit
    for bit. CPU stream: largest difference at most 1e-5 in fp32; in the four fp16 and bf16 dtype
    modes report only with the fault bound CPU_FAULT_BOUND of the mode (decision of 2026-11-07, a
    deviation from AC1)."""
    from dinov3_apple.mlx_model import compile_backbone

    with mx.stream(DEVICES[device]):
        mlx = small_mlx_model(small_hf_model(seed=0, name=model), dtype_mode)
        forward = compile_backbone(mlx)
        largest = 0.0
        for shape in PIXEL_SHAPES:
            pixels = mx.array(small_pixels(shape))
            eager = np.array(mlx(pixels).astype(mx.float32))
            compiled = np.array(forward(pixels).astype(mx.float32))
            assert compiled.shape == eager.shape
            largest = max(largest, float(np.abs(compiled - eager).max()))
    if device == "gpu":
        target = 0.0
    else:
        target = CPU_COMPILE_TOLERANCE if dtype_mode == "fp32" else CPU_FAULT_BOUND[dtype_mode]
    report(
        "test_compiled_equals_eager_small",
        device=device,
        model=model,
        dtype_mode=dtype_mode,
        difference=largest,
        target=target,
    )
    assert largest <= target


def test_compiled_forward_follows_a_weight_change() -> None:
    """T35 (R35): the weights are inputs of the trace, so a weight change reaches the output."""
    from dinov3_apple.mlx_model import compile_backbone

    mlx = small_mlx_model(small_hf_model(seed=0), "fp32")
    forward = compile_backbone(mlx)
    pixels = mx.array(small_pixels(PIXEL_SHAPES[0]))
    before = np.array(forward(pixels))
    q_proj = mlx.layer[0].attention.q_proj
    mlx.update({"layer": [{"attention": {"q_proj": {"weight": q_proj.weight + 0.01}}}]})
    after = np.array(forward(pixels))
    eager = np.array(mlx(pixels))
    assert np.array_equal(after, eager)
    assert not np.array_equal(after, before)


def test_compiled_forward_frees_the_weights() -> None:
    """T36 (R36): after del forward and del model the active memory is back within 1 MB, without
    gc.collect(): the caller holds the function, the model does not."""
    from dinov3_apple.mlx_model import compile_backbone

    gc.collect()
    mx.clear_cache()
    hf = small_hf_model(seed=0, name="vits16plus")
    before = mx.get_active_memory()
    mlx = small_mlx_model(hf, "fp32")
    forward = compile_backbone(mlx)
    mx.eval(forward(mx.array(small_pixels(PIXEL_SHAPES[0]))))
    assert mx.get_active_memory() - before > 1_000_000  # the weights are alive
    gc.disable()
    try:
        del forward, mlx
        mx.clear_cache()
        assert mx.get_active_memory() - before <= 1_000_000
    finally:
        gc.enable()


def test_run_mlx_uses_the_given_forward(mlx_model: MlxBackbone) -> None:
    """T40 (D4): run_mlx runs the given compiled forward pass once; without it, the eager one."""
    from dinov3_apple.mlx_model import compile_backbone

    calls: list[tuple[int, ...]] = []
    compiled = compile_backbone(mlx_model)

    def counting(pixels: mx.array) -> mx.array:
        calls.append(tuple(pixels.shape))
        return compiled(pixels)

    pixels = small_pixels(PIXEL_SHAPES[1])
    tokens = run_mlx(mlx_model, pixels, "fp32", forward=counting)
    assert calls == [PIXEL_SHAPES[1]]
    assert np.array_equal(tokens, run_mlx(mlx_model, pixels, "fp32"))
    assert len(calls) == 1


@pytest.mark.parametrize("dtype_mode", DTYPE_MODES)
def test_mlx_embeddings_fp16_patches(dtype_mode: str) -> None:
    """T55 (D13): frame fp32; patches fp16 in the four fp16 and bf16 dtype modes, fp32 in fp32 and
    with patch_fp32=True; the fp16 values equal the fp32 tokens cast to fp16, bit for bit."""
    from dinov3_apple.mlx_model import mlx_embeddings

    mlx = small_mlx_model(small_hf_model(seed=0), dtype_mode)
    tokens = mlx(mx.array(small_pixels(PIXEL_SHAPES[0])))
    frame, patches = mlx_embeddings(tokens, dtype_mode)
    as_fp32 = np.array(tokens.astype(mx.float32))
    assert frame.dtype == np.float32
    assert np.array_equal(frame, as_fp32[:, 0])
    half = dtype_mode != "fp32"
    assert patches.dtype == (np.float16 if half else np.float32)
    assert np.array_equal(patches, as_fp32[:, NUM_PREFIX_TOKENS:].astype(patches.dtype))
    _, wide = mlx_embeddings(tokens, dtype_mode, patch_fp32=True)
    assert wide.dtype == np.float32
    assert np.array_equal(wide, as_fp32[:, NUM_PREFIX_TOKENS:])


def _sdpa_share(dtype: str, head_dim: int, t: int, via_fp32: bool, seed: int = 0) -> float:
    """The share of SDPA outputs that differ from the float64 result rounded once."""
    rng = np.random.default_rng(seed)
    half = {"fp16": mx.float16, "bf16": mx.bfloat16}[dtype]
    values = [rng.standard_normal((1, 6, t, head_dim)).astype(np.float32) for _ in range(3)]
    q, k, v = (mx.array(a).astype(half) for a in values)
    q64, k64, v64 = (np.array(a.astype(mx.float32)).astype(np.float64) for a in (q, k, v))
    scale = head_dim**-0.5
    scores = q64 @ k64.transpose(0, 1, 3, 2) * scale
    scores = np.exp(scores - scores.max(-1, keepdims=True))
    expected = (scores / scores.sum(-1, keepdims=True)) @ v64
    if via_fp32:
        output = mx.fast.scaled_dot_product_attention(
            q.astype(mx.float32), k.astype(mx.float32), v.astype(mx.float32), scale=scale
        ).astype(half)
    else:
        output = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
    if dtype == "fp16":
        rounded = mx.array(expected.astype(np.float16))
    else:
        rounded = mx.array(expected.astype(np.float32)).astype(half)
    share = mx.not_equal(output.view(mx.uint16), rounded.view(mx.uint16)).astype(mx.float32)
    return float(share.mean().item())


def test_mlx_sdpa_nax_fingerprint(report: Callable[..., None]) -> None:
    """T57 (D19, F3), report only: the share of fp16 and bf16 SDPA outputs that differ from the
    float64 result rounded once, at head dimension 64 and 128 and T = 201, with two controls
    (head dimension 80; the fp32 kernel on fp16 and bf16 inputs). With the Neural Accelerators
    on, the probe gave 54.5 % (fp16) and 12.6 % (bf16) at head dimension 64 with mlx 0.32.4; a new
    MLX version that turns the NAX kernels off or changes the relaxed precision shows here."""
    from dinov3_apple.machine import read_versions

    shares = {}
    for dtype in ("fp16", "bf16"):
        for name, head_dim, via_fp32 in (
            ("sdpa_hd64", 64, False),
            ("sdpa_hd128", 128, False),
            ("control_hd80", 80, False),
            ("control_fp32_kernel_hd64", 64, True),
        ):
            shares[f"{dtype}/{name}"] = _sdpa_share(dtype, head_dim, 201, via_fp32)
    versions = read_versions()
    architecture = mx.device_info()["architecture"]
    report(
        "test_mlx_sdpa_nax_fingerprint",
        shares=shares,
        mlx_architecture=architecture,
        versions=versions,
        nax_variable=os.environ.get("MLX_METAL_GPU_ARCH"),
    )
    assert all(0.0 <= share <= 1.0 for share in shares.values())
    assert versions["mlx-metal"] != "not installed"
    assert isinstance(architecture, str) and architecture


@pytest.mark.parametrize("dtype_mode", list(CPU_FAULT_BOUND))
def test_cpu_fault_bound_catches_the_f1_fault(dtype_mode: str, report: Callable[..., None]) -> None:
    """The red case of the CPU fault bounds of T34: the F1 fault of Step 4 (a wrong input cast:
    the pixels pass through an integer dtype) on the CPU stream gives a difference to the eager
    forward pass of at least 5 times the bound of the mode."""
    from dinov3_apple.mlx_model import compile_backbone

    largest = 0.0
    with mx.stream(mx.cpu):
        for model in SMALL_NAMES:
            mlx = small_mlx_model(small_hf_model(seed=0, name=model), dtype_mode)
            forward = compile_backbone(mlx)
            for shape in PIXEL_SHAPES:
                pixels = mx.array(small_pixels(shape))
                eager = np.array(mlx(pixels).astype(mx.float32))
                faulty = forward(pixels.astype(mx.uint32).astype(mx.float32))
                difference = float(np.abs(np.array(faulty.astype(mx.float32)) - eager).max())
                largest = max(largest, difference)
    margin = largest / CPU_FAULT_BOUND[dtype_mode]
    report(
        "test_cpu_fault_bound_catches_the_f1_fault",
        dtype_mode=dtype_mode,
        difference=largest,
        bound=CPU_FAULT_BOUND[dtype_mode],
        margin=margin,
    )
    assert margin >= 5
