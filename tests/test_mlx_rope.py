"""Tests T8 and T9 of specs/04-mlx-port/plan.html and T27 of specs/05-all-sizes/plan.html: the
RoPE tables."""

import math
from collections.abc import Callable

import mlx.core as mx
import numpy as np
import pytest
import torch
from small_model import GRIDS, SMALL_CONFIG, SMALL_CONFIGS
from transformers import DINOv3ViTConfig
from transformers.models.dinov3_vit.modeling_dinov3_vit import DINOv3ViTRopePositionEmbedding

from dinov3_mlx.mlx_model import rope_tables
from dinov3_mlx.outputs import NUM_PREFIX_TOKENS, PATCH_SIZE

pytestmark = pytest.mark.mlx

HEAD_DIM = 64
THETA = 100.0

# Table 8 of specs/04-mlx-port/explore.html: the largest angle difference to the Meta formula with
# the bf16 periods of the Meta checkpoints, in rad.
META_ANGLE_DIFFERENCE = {(14, 14): 0.0079, (32, 32): 0.0083}


def _hf_tables(
    grid_h: int, grid_w: int, config: dict = SMALL_CONFIG
) -> tuple[np.ndarray, np.ndarray]:
    rope = DINOv3ViTRopePositionEmbedding(DINOv3ViTConfig(**config)).eval()
    pixels = torch.zeros(1, 3, grid_h * PATCH_SIZE, grid_w * PATCH_SIZE)
    with torch.inference_mode():
        cos, sin = rope(pixels)
    return cos.numpy(), sin.numpy()


def _coordinates(grid_h: int, grid_w: int) -> np.ndarray:
    """The patch-center coordinates (y, x) in [-1, 1], fp32, with shape (N, 2)."""
    coord_h = np.arange(0.5, grid_h, dtype=np.float32) / grid_h
    coord_w = np.arange(0.5, grid_w, dtype=np.float32) / grid_w
    coords = np.stack(np.meshgrid(coord_h, coord_w, indexing="ij"), axis=-1).reshape(-1, 2)
    return 2.0 * coords - 1.0


def _layout(angles: np.ndarray) -> np.ndarray:
    """(N, 2, head_dim / 4) -> (N, head_dim): y for the pairs 0 to 15, x for the pairs 16 to 31."""
    return np.tile(angles.reshape(len(angles), -1), 2)


@pytest.mark.parametrize(("grid_h", "grid_w"), GRIDS, ids=[f"{h}x{w}" for h, w in GRIDS])
def test_rope_tables(grid_h: int, grid_w: int, report: Callable[..., None]) -> None:
    cos, sin = rope_tables(grid_h, grid_w, HEAD_DIM, THETA)
    hf_cos, hf_sin = _hf_tables(grid_h, grid_w)
    n = grid_h * grid_w
    assert cos.shape == sin.shape == (NUM_PREFIX_TOKENS + n, HEAD_DIM)
    assert cos.dtype == sin.dtype == np.float32
    assert np.array_equal(cos[:NUM_PREFIX_TOKENS], np.ones((NUM_PREFIX_TOKENS, HEAD_DIM)))
    assert np.array_equal(sin[:NUM_PREFIX_TOKENS], np.zeros((NUM_PREFIX_TOKENS, HEAD_DIM)))
    cos_difference = float(np.abs(cos[NUM_PREFIX_TOKENS:] - hf_cos).max())
    sin_difference = float(np.abs(sin[NUM_PREFIX_TOKENS:] - hf_sin).max())
    report("test_rope_tables", grid=f"{grid_h}x{grid_w}", cos=cos_difference, sin=sin_difference)
    assert cos_difference <= 1e-6
    assert sin_difference <= 1e-6


@pytest.mark.parametrize(("grid_h", "grid_w"), GRIDS, ids=[f"{h}x{w}" for h, w in GRIDS])
def test_rope_tables_meta_difference(grid_h: int, grid_w: int, report: Callable[..., None]) -> None:
    # The periods 100^(k/16) in bf16, as the Meta checkpoints store them.
    exact = THETA ** (np.arange(HEAD_DIM // 4, dtype=np.float64) / (HEAD_DIM // 4))
    periods = torch.from_numpy(exact).to(torch.bfloat16).to(torch.float32).numpy()
    inv_freq = 1 / THETA ** np.arange(0, 1, 4 / HEAD_DIM, dtype=np.float32)
    coords = _coordinates(grid_h, grid_w)
    meta_angles = _layout(2 * math.pi * coords[:, :, None] / periods[None, None, :])
    hf_angles = _layout(2 * math.pi * coords[:, :, None] * inv_freq[None, None, :])
    cos, sin = rope_tables(grid_h, grid_w, HEAD_DIM, THETA)
    angle = float(np.abs(meta_angles - hf_angles).max())
    cos_difference = float(np.abs(cos[NUM_PREFIX_TOKENS:] - np.cos(meta_angles)).max())
    sin_difference = float(np.abs(sin[NUM_PREFIX_TOKENS:] - np.sin(meta_angles)).max())
    report(
        "test_rope_tables_meta_difference",
        grid=f"{grid_h}x{grid_w}",
        angle=angle,
        cos=cos_difference,
        sin=sin_difference,
    )
    if (grid_h, grid_w) in META_ANGLE_DIFFERENCE:
        assert angle == pytest.approx(META_ANGLE_DIFFERENCE[(grid_h, grid_w)], rel=0.05)


@pytest.mark.parametrize(("grid_h", "grid_w"), GRIDS, ids=[f"{h}x{w}" for h, w in GRIDS])
def test_rope_tables_head_dim_128(grid_h: int, grid_w: int, report: Callable[..., None]) -> None:
    """T27: the tables of head dimension 128 (ViT-7B/16) against Hugging Face."""
    head_dim = 128
    cos, sin = rope_tables(grid_h, grid_w, head_dim, THETA)
    hf_cos, hf_sin = _hf_tables(grid_h, grid_w, SMALL_CONFIGS["vit7b16-shape"])
    n = grid_h * grid_w
    assert hf_cos.shape == (n, head_dim)
    assert cos.shape == sin.shape == (NUM_PREFIX_TOKENS + n, head_dim)
    assert cos.dtype == sin.dtype == np.float32
    assert np.array_equal(cos[:NUM_PREFIX_TOKENS], np.ones((NUM_PREFIX_TOKENS, head_dim)))
    assert np.array_equal(sin[:NUM_PREFIX_TOKENS], np.zeros((NUM_PREFIX_TOKENS, head_dim)))
    cos_difference = float(np.abs(cos[NUM_PREFIX_TOKENS:] - hf_cos).max())
    sin_difference = float(np.abs(sin[NUM_PREFIX_TOKENS:] - hf_sin).max())
    report(
        "test_rope_tables_head_dim_128",
        grid=f"{grid_h}x{grid_w}",
        cos=cos_difference,
        sin=sin_difference,
        target=1e-6,
    )
    assert cos_difference <= 1e-6
    assert sin_difference <= 1e-6


# Build step 2b of specs/06-speed/plan.html: the concat-free rotation and the RoPE cache.
ROPE_DEVICES = {"gpu": mx.gpu, "cpu": mx.cpu}
ROPE_DTYPES = {"fp32": mx.float32, "fp16": mx.float16, "bf16": mx.bfloat16}
ROPE_GRIDS = [(14, 14), (32, 32), (14, 25)]


@pytest.mark.parametrize(
    ("device", "dtype", "head_dim", "grid"),
    [
        pytest.param(device, dtype, hd, grid, id=f"{device}-{dtype}-{hd}-{grid[0]}x{grid[1]}")
        for device in ROPE_DEVICES
        for dtype in ROPE_DTYPES
        for hd in (64, 128)
        for grid in ROPE_GRIDS
    ],
)
def test_concat_free_rope_equals_the_step5_form(
    device: str, dtype: str, head_dim: int, grid: tuple[int, int]
) -> None:
    """T38 (D2, R48): the concat-free rotation equals the Step 5 form bit for bit. The inputs come
    in the transposed layout of Attention.qkv, as in the forward pass."""
    from rope_oracle import apply_rope_step5

    from dinov3_mlx.mlx_model import apply_rope, rope_layout

    cos, sin = rope_tables(*grid, head_dim, THETA)
    cos2, sin2 = rope_layout(cos, sin)
    assert cos2.shape == sin2.shape == (len(cos), 2, head_dim // 2)
    rng = np.random.default_rng(0)
    heads, t = 3, len(cos)
    values = rng.standard_normal((2, t, heads * head_dim)).astype(np.float32)
    with mx.stream(ROPE_DEVICES[device]):
        x = mx.array(values).astype(ROPE_DTYPES[dtype])
        x = x.reshape(2, t, heads, head_dim).transpose(0, 2, 1, 3)
        new = apply_rope(x, mx.array(cos2), mx.array(sin2))
        old = apply_rope_step5(x, mx.array(cos), mx.array(sin))
        assert new.dtype == old.dtype == ROPE_DTYPES[dtype]
        assert np.array_equal(np.array(new.astype(mx.float32)), np.array(old.astype(mx.float32)))


def test_rope_cache_reuses_the_tables() -> None:
    """T39 (A3, R43): the same arrays at a second call, one entry for each grid, no cache in the
    module tree, fp32 tables in the layout (5 + N, 2, head_dim / 2)."""
    from mlx.utils import tree_flatten
    from small_model import small_hf_model, small_mlx_model

    model = small_mlx_model(small_hf_model(seed=0))
    first = model.rope(14, 14)
    assert model.rope(14, 14)[0] is first[0] and model.rope(14, 14)[1] is first[1]
    model.rope(14, 19)
    assert len(model.rope_cache) == 2
    names = [name for name, _ in tree_flatten(model.state)]
    assert not any("rope" in name for name in names)
    assert len(tree_flatten(model.parameters())) == len(names)
    assert first[0].dtype == first[1].dtype == mx.float32
    assert first[0].shape == (NUM_PREFIX_TOKENS + 196, 2, model.config.head_dim // 2)


def test_compiled_forward_traces_once_per_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    """T37 (R43): calls at (1, 224, 224), (2, 224, 224), (1, 448, 448), (1, 224, 224) and
    (2, 224, 224) give 3 traces and 2 cache entries; the new grid does not retrace the old
    shapes."""
    from small_model import small_hf_model, small_mlx_model, small_pixels

    from dinov3_mlx.mlx_model import MlxBackbone, compile_backbone

    model = small_mlx_model(small_hf_model(seed=0))
    traces: list[tuple[int, ...]] = []
    original = MlxBackbone.forward_with_rope

    def counting(self: MlxBackbone, pixels: mx.array, cos2: mx.array, sin2: mx.array) -> mx.array:
        traces.append(tuple(pixels.shape))
        return original(self, pixels, cos2, sin2)

    monkeypatch.setattr(MlxBackbone, "forward_with_rope", counting)
    forward = compile_backbone(model)
    for shape in [(1, 3, 224, 224), (2, 3, 224, 224), (1, 3, 448, 448)] + [
        (1, 3, 224, 224),
        (2, 3, 224, 224),
    ]:
        mx.eval(forward(mx.array(small_pixels(shape))))
    assert len(traces) == 3
    assert len(model.rope_cache) == 2
