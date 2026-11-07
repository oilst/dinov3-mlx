"""Tests T10 to T20 of specs/04-mlx-port/plan.html: each part of the MLX backbone against the
Hugging Face modules on torch-cpu, with the small model of tests/small_model.py.

MLX runs on the GPU with TF32 off (tests/conftest.py). Each test records its measured value next to
its target in tests/fixtures/generated/mlx/report.jsonl (Table 7 of the plan).
"""

from collections.abc import Callable

import mlx.core as mx
import numpy as np
import pytest
import torch
from mlx import nn
from mlx.utils import tree_flatten
from small_model import PIXEL_SHAPES, small_hf_model, small_mlx_model, small_pixels
from transformers.activations import ACT2FN
from transformers.models.dinov3_vit.modeling_dinov3_vit import DINOv3ViTModel

from dinov3_apple.block_profile import block_parts, run_parts
from dinov3_apple.flops import PART_NAMES
from dinov3_apple.mlx_model import (
    Block,
    MlxBackbone,
    apply_rope,
    patchify,
    run_mlx,
    run_mlx_hidden_states,
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
# The pixels of the part tests: the batch of 2 at 224 x 224 px.
PART_PIXELS = PIXEL_SHAPES[0]
SHAPE_IDS = ["x".join(map(str, shape)) for shape in PIXEL_SHAPES]
# The grids of the attention and block tests: 14 x 14 and 14 x 19 patches.
GRID_SHAPES = [(2, 3, 224, 224), (1, 3, 224, 304)]
GRID_IDS = ["14x14", "14x19"]


@pytest.fixture(scope="module")
def hf_model() -> DINOv3ViTModel:
    """Return the small Hugging Face backbone, seed 0 (the oracle of the part tests)."""
    return small_hf_model(seed=0)


@pytest.fixture(scope="module")
def mlx_model(hf_model: DINOv3ViTModel) -> MlxBackbone:
    """Return the small MLX backbone with the weights of hf_model."""
    return small_mlx_model(hf_model)


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
        "ff_in": lambda h: (mlp.up_proj(h),),
        "activation": lambda up: (nn.gelu(up),),
        "ff_out": lambda g: (mlp.down_proj(g),),
        "layer_scale2": lambda y, x: (block.layer_scale2(y) + x,),
    }


@pytest.mark.parametrize("name", PART_NAMES)
def test_mlx_block_part(
    name: str,
    hf_model: DINOv3ViTModel,
    mlx_model: MlxBackbone,
    report: Callable[..., None],
) -> None:
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
    report("test_mlx_block_part", part=name, difference=difference, target=PART_TARGETS[name])
    assert difference <= PART_TARGETS[name]


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


def test_mlx_forward_casts_the_pixels_to_the_weight_dtype(hf_model: DINOv3ViTModel) -> None:
    """fp32 pixels and bf16 weights give bf16 tokens, as in the Hugging Face forward pass."""
    model = small_mlx_model(hf_model)
    model.set_dtype(mx.bfloat16)
    pixels = mx.array(small_pixels(PART_PIXELS))
    embeddings = model.embeddings(pixels)
    assert embeddings.dtype == mx.bfloat16
    assert mx.array_equal(embeddings, model.embeddings(pixels.astype(mx.bfloat16))).item()
    assert model(pixels).dtype == mx.bfloat16


@pytest.mark.parametrize("shape", GRID_SHAPES, ids=GRID_IDS)
def test_mlx_part_attention(
    shape: tuple[int, ...],
    hf_model: DINOv3ViTModel,
    mlx_model: MlxBackbone,
    report: Callable[..., None],
) -> None:
    pixels = torch.from_numpy(small_pixels(shape))
    layer = hf_model.model.layer[0]
    with torch.inference_mode():
        h = layer.norm1(hf_model.embeddings(pixels))
        expected, _ = layer.attention(h, position_embeddings=hf_model.rope_embeddings(pixels))
    cos, sin = mlx_model.rope(shape[2] // PATCH_SIZE, shape[3] // PATCH_SIZE)
    output = mlx_model.layer[0].attention(mx.array(h.numpy()), cos, sin)
    difference = _difference(expected, output)
    report("test_mlx_part_attention", shape=list(shape), difference=difference, target=1e-5)
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


@pytest.mark.parametrize("shape", GRID_SHAPES, ids=GRID_IDS)
def test_mlx_part_block(
    shape: tuple[int, ...],
    hf_model: DINOv3ViTModel,
    mlx_model: MlxBackbone,
    report: Callable[..., None],
) -> None:
    pixels = torch.from_numpy(small_pixels(shape))
    with torch.inference_mode():
        x = hf_model.embeddings(pixels)
        expected = hf_model.model.layer[0](x, position_embeddings=hf_model.rope_embeddings(pixels))
    cos, sin = mlx_model.rope(shape[2] // PATCH_SIZE, shape[3] // PATCH_SIZE)
    output = mlx_model.layer[0](mx.array(x.numpy()), cos, sin)
    difference = _difference(expected, output)
    report("test_mlx_part_block", shape=list(shape), difference=difference, target=3e-5)
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


@pytest.mark.parametrize("shape", PIXEL_SHAPES, ids=SHAPE_IDS)
def test_mlx_small_model_tokens(
    shape: tuple[int, ...],
    hf_model: DINOv3ViTModel,
    mlx_model: MlxBackbone,
    report: Callable[..., None],
) -> None:
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
    report("test_mlx_small_model_tokens", shape=list(shape), **values, target=5e-5)
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
