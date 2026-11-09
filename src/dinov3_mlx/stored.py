"""The 12 configurations of Step 2 and the readers of the stored pixels and reference files.

No torch import, so an mlx process can read the stored files without the memory of torch in its
footprint (D6 of specs/04-mlx-port/explore.html). reference.py and preprocess.py re-export these
names, so the imports of Steps 2 and 3 keep working.
"""

import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
from PIL import Image
from safetensors import safe_open

from dinov3_mlx.hub import MODEL_IDS
from dinov3_mlx.outputs import PATCH_SIZE
from dinov3_mlx.parity import TEST_IMAGES

REPO_ROOT = Path(__file__).parents[2]
IMAGES_DIR = REPO_ROOT / "tests" / "fixtures" / "images"
REFERENCE_DIR = REPO_ROOT / "tests" / "fixtures" / "generated" / "reference"
INPUTS_FILE = REFERENCE_DIR / "inputs.safetensors"

SQUARE_SIZES = [224, 448, 512]
# The ImageNet mean and standard deviation of the normalization, fp32. preprocess.py (torch) and
# mlx_preprocess.py (MLX) take them from here (D12 of specs/06-speed/plan.html).
IMAGENET_MEAN_VALUES = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD_VALUES = np.array([0.229, 0.224, 0.225], dtype=np.float32)
HIDDEN_STATES_CONFIG = "iss-kibo-lab/224x224"


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


@dataclass(frozen=True)
class Configuration:
    """One test image at one input resolution."""

    image: str
    height: int
    width: int

    @property
    def id(self) -> str:
        """The key in the stored files, for example "forest/224x304"."""
        return f"{self.image}/{self.height}x{self.width}"

    @property
    def image_path(self) -> Path:
        """The test image file."""
        return IMAGES_DIR / f"{self.image}.jpg"

    @cached_property
    def num_patches(self) -> int:
        """The number of patch tokens."""
        return (self.height // PATCH_SIZE) * (self.width // PATCH_SIZE)


def configurations() -> list[Configuration]:
    """Return the 12 configurations of Table 2: 3 images x (3 square sizes + own aspect ratio)."""
    configs = []
    for image in TEST_IMAGES:
        with Image.open(IMAGES_DIR / f"{image}.jpg") as file:
            width, height = file.size
        sizes = [(size, size) for size in SQUARE_SIZES] + [own_aspect_size(height, width)]
        configs += [Configuration(image, h, w) for h, w in sizes]
    return configs


def model_slug(model_size: str) -> str:
    """Return the part of the Hugging Face id between "dinov3-" and "-pretrain", e.g. vits16."""
    return MODEL_IDS[model_size].split("dinov3-", 1)[1].split("-pretrain", 1)[0]


def reference_path(model_size: str) -> Path:
    """Return the reference file of one model size."""
    return REFERENCE_DIR / f"reference-{model_slug(model_size)}.safetensors"


def read_inputs(path: Path = INPUTS_FILE) -> dict[str, np.ndarray]:
    """Return the stored pixels with shape (1, 3, H, W), keyed by configuration id."""
    with safe_open(path, "np") as tensors:
        keys = tensors.keys()
        return {key.removesuffix("/pixels"): tensors.get_tensor(key) for key in keys}


def read_reference(model_size: str) -> tuple[dict[str, np.ndarray], dict]:
    """Return the reference tokens and the metadata of one model size.

    Returns:
        The tensors keyed by configuration id (tokens with shape (1 + 4 + N, D)), plus the key
        "{HIDDEN_STATES_CONFIG}/hidden_states", and the metadata as a dict.
    """
    with safe_open(reference_path(model_size), "np") as tensors:
        metadata = json.loads(tensors.metadata()["reference"])
        keys = tensors.keys()
        data = {key.removesuffix("/tokens"): tensors.get_tensor(key) for key in keys}
    return data, metadata
