from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from dinov3_mlx.preprocess import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    load_image,
    own_aspect_size,
    preprocess,
)
from dinov3_mlx.reference import INPUTS_FILE, Configuration, configurations, read_inputs

IMAGES_DIR = Path(__file__).parent / "fixtures" / "images"


@pytest.mark.parametrize(
    ("size", "expected"),
    [
        ((576, 1024), (224, 400)),
        ((768, 1024), (224, 304)),
        ((1024, 768), (304, 224)),
        ((500, 500), (224, 224)),
    ],
)
def test_own_aspect_size(size: tuple[int, int], expected: tuple[int, int]) -> None:
    assert own_aspect_size(*size) == expected


def test_preprocess_shape_and_dtype() -> None:
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    pixels = preprocess(image, (32, 48))
    assert pixels.shape == (1, 3, 32, 48)
    assert pixels.dtype.is_floating_point and pixels.dtype.itemsize == 4


def _pillow_float_resize(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize each color channel as a float Pillow image (mode F) with BILINEAR."""
    height, width = size
    channels = [
        np.asarray(
            Image.fromarray(image[..., c].astype(np.float32) / 255.0, mode="F").resize(
                (width, height), Image.Resampling.BILINEAR
            )
        )
        for c in range(3)
    ]
    return np.stack(channels)[None]


@pytest.mark.parametrize("config", configurations(), ids=lambda config: config.id)
def test_preprocess_matches_pillow_float_resize(config: Configuration) -> None:
    image = load_image(config.image_path)
    size = (config.height, config.width)
    pixels = preprocess(image, size).numpy()
    unnormalized = pixels * IMAGENET_STD.numpy().reshape(
        1, 3, 1, 1
    ) + IMAGENET_MEAN.numpy().reshape(1, 3, 1, 1)
    assert np.abs(unnormalized - _pillow_float_resize(image, size)).max() <= 1e-5


def test_preprocess_matches_stored_pixels() -> None:
    if not INPUTS_FILE.exists():
        pytest.skip(f"{INPUTS_FILE} is missing. Run scripts/make_reference.py.")
    stored = read_inputs()
    configs = configurations()
    assert len(configs) == 12
    for config in configs:
        pixels = preprocess(load_image(config.image_path), (config.height, config.width)).numpy()
        assert np.abs(pixels - stored[config.id]).max() == 0.0, config.id
