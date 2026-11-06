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

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
PATCH_SIZE = 16


def own_aspect_size(height: int, width: int, short_side: int = 224) -> tuple[int, int]:
    """Return the input resolution that keeps the aspect ratio of an image.

    The short side becomes short_side. The long side is rounded to the nearest multiple of 16.

    Args:
        height: The image height in pixels.
        width: The image width in pixels.
        short_side: The short side of the input resolution, a multiple of 16.

    Returns:
        The input resolution as (height, width).
    """
    long_side = round(max(height, width) / min(height, width) * short_side / PATCH_SIZE)
    long_side *= PATCH_SIZE
    return (short_side, long_side) if height <= width else (long_side, short_side)


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
