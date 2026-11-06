from collections.abc import Callable

import numpy as np
import pytest
import torch
from under_test import MODEL_SIZES_UNDER_TEST

from dinov3_apple.outputs import NUM_PREFIX_TOKENS
from dinov3_apple.reference import (
    HIDDEN_STATES_CONFIG,
    REQUIRED_METADATA,
    configurations,
    read_reference,
    reference_path,
    run_torch,
)


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
    inputs: dict[str, np.ndarray],
    reference: Callable[[str], dict[str, np.ndarray]],
) -> None:
    stored = reference(model_size)[HIDDEN_STATES_CONFIG]
    tokens = run_torch(backbone(model_size, "cpu", "fp32"), inputs[HIDDEN_STATES_CONFIG], "fp32")
    assert np.abs(tokens[0] - stored).max() <= 1e-5
