import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from fakes import fail_if_called
from small_model import save_small_model, small_hf_model, small_pixels
from transformers import DINOv3ViTModel
from under_test import MODEL_SIZES_UNDER_TEST

import dinov3_mlx.reference as torch_reference
from dinov3_mlx.memory import NO_FP32_WEIGHTS
from dinov3_mlx.outputs import NUM_PREFIX_TOKENS
from dinov3_mlx.reference import (
    HIDDEN_STATES_CONFIG,
    REQUIRED_METADATA,
    configurations,
    read_reference,
    reference_path,
    run_torch,
)


def reproduced_tokens(
    model_size: str,
    backbone: Callable[[str, str, str], torch.nn.Module],
    free_backbone: Callable[[], None],
    pixels: np.ndarray,
) -> np.ndarray:
    """Return the tokens of the hidden-states configuration again, on torch-cpu in fp32.

    A model size of NO_FP32_WEIGHTS (ViT-7B/16) never loads the whole model in fp32: it runs
    through run_torch_streamed on this configuration only, with the fp32 weights of one block at a
    time (path D4-Y, Table 7 of specs/05-all-sizes/plan.html). The backbone slot frees its model
    first, so only one model is in memory.

    Args:
        model_size: A key of MODEL_IDS.
        backbone: The loader of the backbone slot.
        free_backbone: Frees the model of the backbone slot.
        pixels: The stored pixels of HIDDEN_STATES_CONFIG, shape (1, 3, H, W).

    Returns:
        The tokens after the final layer norm, shape (1 + 4 + N, D).
    """
    if model_size in NO_FP32_WEIGHTS:
        free_backbone()
        tokens, _ = torch_reference.run_torch_streamed(
            model_size, {HIDDEN_STATES_CONFIG: pixels}, None
        )
        return tokens[HIDDEN_STATES_CONFIG]
    return run_torch(backbone(model_size, "cpu", "fp32"), pixels, "fp32")[0]


@pytest.mark.parametrize("model_size", MODEL_SIZES_UNDER_TEST)
def test_reference_files_are_complete(model_size: str) -> None:
    if not reference_path(model_size).exists():
        pytest.skip(f"{reference_path(model_size).name} is missing. Run make_reference.py.")
    tensors, metadata = read_reference(model_size)
    dim = tensors[HIDDEN_STATES_CONFIG].shape[-1]
    configs = configurations()
    assert len(configs) == 12
    for config in configs:
        tokens = tensors[config.id]
        assert tokens.shape == (NUM_PREFIX_TOKENS + config.num_patches, dim), config.id
        assert tokens.dtype == np.float32
        assert np.isfinite(tokens).all(), config.id
    hidden = tensors[f"{HIDDEN_STATES_CONFIG}/hidden_states"]
    assert hidden.shape == (metadata["num_blocks"] + 1, NUM_PREFIX_TOKENS + 196, dim)
    assert hidden.dtype == np.float32
    assert np.isfinite(hidden).all()
    assert set(REQUIRED_METADATA) <= set(metadata)
    assert metadata["model_size"] == model_size


@pytest.mark.parametrize("model_size", MODEL_SIZES_UNDER_TEST)
def test_reference_is_reproducible(
    model_size: str,
    backbone: Callable[[str, str, str], torch.nn.Module],
    backbone_slot: Any,
    inputs: dict[str, np.ndarray],
    reference: Callable[[str], dict[str, np.ndarray]],
) -> None:
    """The tokens of the hidden-states configuration again, within 1e-5 of the reference file.
    ViT-7B/16 uses the streamed path (reproduced_tokens)."""
    stored = reference(model_size)[HIDDEN_STATES_CONFIG]
    pixels = inputs[HIDDEN_STATES_CONFIG]
    tokens = reproduced_tokens(model_size, backbone, backbone_slot.free, pixels)
    assert np.abs(tokens - stored).max() <= 1e-5


def test_reproduced_tokens_of_7b_never_load_the_whole_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ViT-7B/16 runs through run_torch_streamed on the hidden-states configuration only. The
    backbone slot, load_torch_backbone and from_pretrained are never called. The weights are a
    small model with the shape of ViT-7B/16 in a temporary folder."""
    hf_model = small_hf_model(seed=4, name="vit7b16-shape")
    save_small_model(hf_model, tmp_path / "weights", files=2)
    pixels = small_pixels((1, 3, 224, 224), seed=5)
    expected = run_torch(hf_model, pixels, "fp32")[0]
    monkeypatch.setattr("dinov3_mlx.reference.weights_dir", lambda model_size: tmp_path / "weights")
    monkeypatch.setattr(logging.getLogger("dinov3_mlx.memory"), "handlers", [])
    monkeypatch.setattr("dinov3_mlx.memory.available_memory_bytes", lambda: 10**12)
    monkeypatch.setattr(torch_reference, "load_torch_backbone", fail_if_called("load_torch"))
    monkeypatch.setattr(DINOv3ViTModel, "from_pretrained", fail_if_called("from_pretrained"))
    calls: list[tuple[str, list[str], str | None]] = []
    streamed = torch_reference.run_torch_streamed

    def record(model_size: str, pixels: dict, hidden_config: str | None) -> tuple:
        calls.append((model_size, list(pixels), hidden_config))
        return streamed(model_size, pixels, hidden_config)

    monkeypatch.setattr(torch_reference, "run_torch_streamed", record)
    freed: list[bool] = []
    tokens = reproduced_tokens(
        "ViT-7B/16", fail_if_called("backbone"), lambda: freed.append(True), pixels
    )
    assert calls == [("ViT-7B/16", [HIDDEN_STATES_CONFIG], None)]
    assert freed == [True]
    assert np.abs(tokens - expected).max() == 0.0
