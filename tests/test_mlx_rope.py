"""Tests T8 and T9 of specs/04-mlx-port/plan.html: the RoPE tables."""

import math
from collections.abc import Callable

import numpy as np
import pytest
import torch
from small_model import GRIDS, SMALL_CONFIG
from transformers import DINOv3ViTConfig
from transformers.models.dinov3_vit.modeling_dinov3_vit import DINOv3ViTRopePositionEmbedding

from dinov3_apple.mlx_model import rope_tables
from dinov3_apple.outputs import NUM_PREFIX_TOKENS, PATCH_SIZE

pytestmark = pytest.mark.mlx

HEAD_DIM = 64
THETA = 100.0

# Table 8 of specs/04-mlx-port/explore.html: the largest angle difference to the Meta formula with
# the bf16 periods of the Meta checkpoints, in rad.
META_ANGLE_DIFFERENCE = {(14, 14): 0.0079, (32, 32): 0.0083}


def _hf_tables(grid_h: int, grid_w: int) -> tuple[np.ndarray, np.ndarray]:
    rope = DINOv3ViTRopePositionEmbedding(DINOv3ViTConfig(**SMALL_CONFIG)).eval()
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
