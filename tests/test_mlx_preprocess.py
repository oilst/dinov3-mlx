"""Tests T51 to T53 and W5 of specs/06-speed/plan.html: preprocessing in MLX (D12)."""

from collections.abc import Callable

import mlx.core as mx
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from dinov3_mlx import mlx_preprocess, stored
from dinov3_mlx.mlx_model import TF32ModeError
from dinov3_mlx.preprocess import load_image

pytestmark = pytest.mark.mlx

SIZE_PAIRS = [(1080, 224), (1920, 224), (768, 512), (100, 448)]
CONFIGS = stored.configurations()
CONFIG_IDS = [config.id for config in CONFIGS]
PIXEL_TARGET = 2e-6  # Table 21 of the plan: about 8 units in the last place at 2.64


@pytest.mark.parametrize(("n_in", "n_out"), SIZE_PAIRS, ids=[f"{a}-{b}" for a, b in SIZE_PAIRS])
def test_resize_weights_match_torch(n_in: int, n_out: int) -> None:
    """T51: the weights of the antialiased bilinear resize of torch, from an identity basis."""
    weights = mlx_preprocess.resize_weights(n_in, n_out)
    basis = torch.eye(n_in, dtype=torch.float64).reshape(n_in, 1, 1, n_in)
    expected = F.interpolate(basis, size=(1, n_out), mode="bilinear", antialias=True)
    expected = expected.reshape(n_in, n_out).T.numpy()
    assert weights.shape == (n_out, n_in)
    assert np.abs(weights - expected).max() <= 1e-12
    assert np.abs(weights.sum(axis=1) - 1).max() <= 1e-12


@pytest.mark.parametrize("config", CONFIGS, ids=CONFIG_IDS)
def test_preprocess_frames_matches_stored_pixels(
    config: stored.Configuration, report: Callable[..., None]
) -> None:
    """T52 (D12): the 12 configurations from the image files against the stored pixels."""
    if not stored.INPUTS_FILE.exists():
        pytest.skip(f"{stored.INPUTS_FILE.name} is missing. Run scripts/make_reference.py.")
    frame = load_image(config.image_path)[None]
    pixels = mlx_preprocess.preprocess_frames(frame, (config.height, config.width))
    assert pixels.dtype == mx.float32
    assert tuple(pixels.shape) == (1, 3, config.height, config.width)
    expected = stored.read_inputs()[config.id]
    difference = float(np.abs(np.array(pixels) - expected).max())
    report(
        "test_preprocess_frames_matches_stored_pixels",
        config=config.id,
        difference=difference,
        target=PIXEL_TARGET,
    )
    assert difference <= PIXEL_TARGET


def test_preprocess_frames_refuses_tf32(monkeypatch: pytest.MonkeyPatch) -> None:
    """T53: the resize matmuls run in fp32, so TF32 must be off (ADR 0005)."""
    monkeypatch.setattr(mlx_preprocess, "tf32_enabled", lambda: True)
    with pytest.raises(TF32ModeError):
        mlx_preprocess.preprocess_frames(np.zeros((1, 32, 32, 3), np.uint8), (16, 16))


def test_imagenet_constants_have_one_source() -> None:
    """D12: preprocess.py builds its torch tensors from the NumPy values of stored.py."""
    from dinov3_mlx import preprocess

    assert stored.IMAGENET_MEAN_VALUES.dtype == stored.IMAGENET_STD_VALUES.dtype == np.float32
    assert np.array_equal(preprocess.IMAGENET_MEAN.numpy(), stored.IMAGENET_MEAN_VALUES)
    assert np.array_equal(preprocess.IMAGENET_STD.numpy(), stored.IMAGENET_STD_VALUES)


@pytest.mark.parametrize(
    ("model_size", "dtype_mode"), [pytest.param("ViT-S/16", "fp32", id="mlx-ViT-S/16-fp32")]
)
def test_preprocess_frames_parity(
    model_size: str,
    dtype_mode: str,
    backbone: Callable[..., object],
    reference: Callable[[str], dict[str, np.ndarray]],
    inputs: dict[str, np.ndarray],
    record: Callable[..., None],
    mlx_forward: Callable[[object], Callable[[mx.array], mx.array]],
) -> None:
    """W5 (D12): ViT-S/16 in fp32 on the pixels of preprocess_frames passes THRESHOLDS["fp32"] in
    12 of 12 configurations; the values go next to the values on the stored pixels."""
    from parity_checks import assert_thresholds, record_skip

    from dinov3_mlx.mlx_model import run_mlx
    from dinov3_mlx.parity import measure_parity

    key = ("preprocess-mlx", model_size, dtype_mode, "all")
    with record_skip(record, *key, backend="mlx"):
        stored_tokens = reference(model_size)
        model = backbone(model_size, "mlx", dtype_mode)
    forward = mlx_forward(model)
    for config in CONFIGS:
        pixels = mlx_preprocess.preprocess_frames(
            load_image(config.image_path)[None], (config.height, config.width)
        )
        tokens = run_mlx(model, np.array(pixels), dtype_mode, forward)[0]
        on_stored = run_mlx(model, inputs[config.id], dtype_mode, forward)[0]
        parity = measure_parity(stored_tokens[config.id], tokens)
        stored_parity = measure_parity(stored_tokens[config.id], on_stored)
        record(
            "preprocess-mlx",
            model_size,
            dtype_mode,
            config.id,
            parity,
            "measured",
            backend="mlx",
            values={"stored_pixels": stored_parity.__dict__},
        )
        assert_thresholds(parity, dtype_mode)
