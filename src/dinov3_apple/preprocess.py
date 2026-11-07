"""Image file to normalized fp32 pixel tensor.

The resize works on float pixels with antialiasing, as the Hugging Face image processor of DINOv3
does (transformers, image_processing_dinov3_vit.py). A resize of 8-bit pixels moves each pixel by
up to 0.004 and drops the patch minimum of ViT-H+/16 to 0.93 (specs/02-reference/explore.html).

The resize and the normalization run in fp64, and the result is cast to fp32. An fp32 antialiased
resize in torch is off by up to 2.3e-5 per pixel; in fp64 the error is below 1e-7, so other resize
code (for example a Pillow float resize) can match the stored pixels.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from dinov3_apple.stored import IMAGENET_MEAN_VALUES, IMAGENET_STD_VALUES, own_aspect_size

# own_aspect_size moved to stored.py, which imports no torch (Step 4). This module re-exports it.
__all__ = [
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "PATCH_SIZE",
    "load_image",
    "own_aspect_size",
    "preprocess",
]

IMAGENET_MEAN = torch.from_numpy(IMAGENET_MEAN_VALUES.copy())
IMAGENET_STD = torch.from_numpy(IMAGENET_STD_VALUES.copy())
PATCH_SIZE = 16


def load_image(path: Path) -> np.ndarray:
    """Return an image file as an (h, w, 3) uint8 RGB array."""
    with Image.open(path) as image:
        return np.array(image.convert("RGB"), dtype=np.uint8)


def preprocess(image: np.ndarray, size: tuple[int, int]) -> torch.Tensor:
    """Return a normalized (1, 3, H, W) fp32 tensor from an (h, w, 3) uint8 RGB image.

    Args:
        image: The RGB image as uint8 values.
        size: The input resolution as (H, W).

    Returns:
        The pixels, divided by 255, resized (bilinear, antialiased) and normalized with the
        ImageNet mean and standard deviation. All steps run in fp64.
    """
    pixels = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(torch.float64) / 255.0
    resized = F.interpolate(pixels, size=size, mode="bilinear", antialias=True)
    mean = IMAGENET_MEAN.to(torch.float64).view(1, 3, 1, 1)
    std = IMAGENET_STD.to(torch.float64).view(1, 3, 1, 1)
    return ((resized - mean) / std).to(torch.float32)
