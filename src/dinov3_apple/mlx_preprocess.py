"""Preprocessing in MLX: uint8 RGB frames to normalized fp32 pixels on the GPU (D12 of
specs/06-speed/plan.html).

The steps of preprocess.preprocess: divide by 255, resize (bilinear, antialiased, as
F.interpolate with antialias=True), normalize with the ImageNet mean and standard deviation. The
resize is separable: two matmuls with fixed weight matrices, one for the rows and one for the
columns. The resize runs on float values, not on 8-bit pixels (Step 2). No torch import.
"""

from functools import cache

import mlx.core as mx
import numpy as np

from dinov3_apple.mlx_model import TF32ModeError, tf32_enabled
from dinov3_apple.stored import IMAGENET_MEAN_VALUES, IMAGENET_STD_VALUES


@cache
def resize_weights(n_in: int, n_out: int) -> np.ndarray:
    """Return the (n_out, n_in) fp64 weights of the bilinear antialiased resize of torch.

    The rule of upsample_bilinear2d_aa: a triangle filter whose support grows with the scale when
    the image shrinks. Each row sums to 1.

    Args:
        n_in: The number of input pixels along one axis.
        n_out: The number of output pixels along the same axis.
    """
    scale = n_in / n_out
    support = max(scale, 1.0)
    weights = np.zeros((n_out, n_in))
    for i in range(n_out):
        center = scale * (i + 0.5)
        lo = max(int(center - support + 0.5), 0)
        hi = min(int(center + support + 0.5), n_in)
        taps = np.maximum(0.0, 1.0 - np.abs((np.arange(lo, hi) - center + 0.5) / support))
        weights[i, lo:hi] = taps / taps.sum()
    return weights


@cache
def _resize_matrices(h: int, w: int, size: tuple[int, int]) -> tuple[mx.array, mx.array]:
    """The fp32 matrices of the rows (H, h) and of the columns (w, W) of one size pair."""
    rows = mx.array(resize_weights(h, size[0]).astype(np.float32))
    columns = mx.array(resize_weights(w, size[1]).T.astype(np.float32))
    return rows, columns


def preprocess_frames(frames: np.ndarray, size: tuple[int, int]) -> mx.array:
    """Return normalized fp32 pixels (B, 3, H, W) from uint8 RGB frames (B, h, w, 3).

    The uint8 frames go to the GPU: a quarter of the fp32 bytes. The resize matmuls run in fp32.

    Args:
        frames: The RGB frames as uint8 values, all of one size.
        size: The input resolution as (H, W).

    Returns:
        The pixels, divided by 255, resized and normalized, on the GPU.

    Raises:
        TF32ModeError: If TF32 is on: the resize matmuls run in fp32 (ADR 0005).
    """
    if tf32_enabled():
        raise TF32ModeError("set MLX_ENABLE_TF32=0: the resize matmuls run in fp32")
    _, h, w, _ = frames.shape
    rows, columns = _resize_matrices(h, w, tuple(size))
    x = mx.array(frames).astype(mx.float32).transpose(0, 3, 1, 2) / 255.0
    x = rows @ x @ columns
    mean = mx.array(IMAGENET_MEAN_VALUES).reshape(1, 3, 1, 1)
    std = mx.array(IMAGENET_STD_VALUES).reshape(1, 3, 1, 1)
    return (x - mean) / std
