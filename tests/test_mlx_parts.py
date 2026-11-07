"""Tests T10 to T20 of specs/04-mlx-port/plan.html and T16 to T26 of specs/05-all-sizes/plan.html:
each part of the MLX backbone against the Hugging Face modules on torch-cpu, with the small models
of tests/small_model.py, and the dtype of each part in each dtype mode.

MLX runs on the GPU with TF32 off (tests/conftest.py). Each test records its measured value next to
its target in tests/fixtures/generated/mlx/report.jsonl (Table 7 of the Step 4 plan, Table 13 of the
Step 5 plan).
"""

from collections.abc import Callable

import mlx.core as mx
import numpy as np
import pytest
import torch
from mlx import nn
from mlx.utils import tree_flatten
from parity_checks import min_token_cosine
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
from dinov3_apple.mlx_model import (
    MLP,
    MLX_DTYPE_MODES,
    Block,
    MlxBackbone,
    MlxConfig,
    SwiGLU,
    apply_rope,
    linear_parameter_names,
    patchify,
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
    """The Hugging Face tables of one grid, plus 5 identity rows on top."""
    cos, sin = hf_model.rope_embeddings(pixels)
    ones = np.ones((NUM_PREFIX_TOKENS, cos.shape[-1]), np.float32)
    return (
        mx.array(np.concatenate([ones, cos.numpy()])),
        mx.array(np.concatenate([0 * ones, sin.numpy()])),
    )


def _mlx_parts(block: Block) -> dict[str, Callable[..., tuple[mx.array, ...]]]:
    """The 11 parts of one MLX block, with the names, inputs and outputs of block_parts."""
    attention, mlp = block.attention, block.mlp

    def ff_in(h: mx.array) -> tuple[mx.array, ...]:
        """(gate, up) for SwiGLU, (up,) for the MLP, as ff_in of block_parts."""
        if isinstance(mlp, SwiGLU):
            return mlp.gate_proj(h), mlp.up_proj(h)
        return (mlp.up_proj(h),)

    return {
        "norm1": lambda x: (block.norm1(x),),
        "qkv_proj": lambda h: attention.qkv(h),
        "rope": lambda q, k, cos, sin: (apply_rope(q, cos, sin), apply_rope(k, cos, sin)),
        "sdpa": lambda q, k, v: (
            mx.fast.scaled_dot_product_attention(q, k, v, scale=attention.scale, mask=None),
        ),
        "o_proj": lambda a: (attention.output(a),),
        "layer_scale1": lambda o, x: (block.layer_scale1(o) + x,),
        "norm2": lambda x: (block.norm2(x),),
        "ff_in": ff_in,
        "activation": lambda *values: (mlp.activation(*values),),
        "ff_out": lambda g: (mlp.down_proj(g),),
        "layer_scale2": lambda y, x: (block.layer_scale2(y) + x,),
    }


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
def test_mlx_dtype_flow(model: str, dtype_mode: str) -> None:
    """T22: the dtype of each part in each dtype mode (Table 3 of the plan)."""
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
    for block in mlx.layer:
        assert block.attention_branch(x, cos, sin).dtype == matmul
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
def test_token_range_matches_hidden_states(model: str, dtype_mode: str) -> None:
    """T26: the token range and run_mlx_hidden_states agree exactly."""
    mlx = small_mlx_model(small_hf_model(seed=0, name=model), dtype_mode)
    pixels = small_pixels((1, 3, 224, 304))
    token_range = run_mlx_token_range(mlx, pixels)
    hidden = run_mlx_hidden_states(mlx, pixels)
    blocks = len(mlx.layer)
    assert token_range.tokens.shape == (blocks + 1,)
    assert token_range.tokens_where.shape == (blocks + 1,)
    assert token_range.tokens_finite.shape == (blocks + 1,)
    for values in (token_range.attention, token_range.ff_hidden, token_range.ff_out):
        assert values.shape == (blocks,)
    assert token_range.last_state.dtype == np.float32
    assert np.array_equal(token_range.last_state, hidden[-1])
    for i, state in enumerate(hidden):
        assert token_range.tokens[i] == np.abs(state).max(), i
        assert token_range.tokens_where[i] == np.abs(state).max(-1).argmax(), i
        assert bool(token_range.tokens_finite[i]) == bool(np.isfinite(state).all()), i
