from collections.abc import Callable

import numpy as np
import pytest
import torch
from under_test import SLOW

from dinov3_apple.block_profile import block_parts, run_parts, time_part
from dinov3_apple.flops import PART_NAMES

# ViT-S/16 has an MLP and ViT-S+/16 has SwiGLU. ViT-S+/16 runs with DINOV3_SLOW=1.
SPLIT_MODEL_SIZES = ["ViT-S/16", "ViT-S+/16"] if SLOW else ["ViT-S/16"]


def test_time_part_returns_seconds_per_call() -> None:
    x = torch.ones(64, 64)
    seconds = time_part(lambda: x @ x, "cpu", min_calls=20, min_seconds=0.01, loops=3)
    assert 0 < seconds < 0.01


@pytest.mark.parametrize("model_size", SPLIT_MODEL_SIZES)
def test_run_parts_equals_block_output(
    model_size: str,
    backbone: Callable[[str, str, str], torch.nn.Module],
    inputs: dict[str, np.ndarray],
) -> None:
    model = backbone(model_size, "cpu", "fp32")
    pixels = torch.from_numpy(inputs["iss-kibo-lab/224x224"])
    with torch.inference_mode():
        x = model.embeddings(pixels)
        position = model.rope_embeddings(pixels)
        layer = model.model.layer[0]
        expected = layer(x, position_embeddings=position)
        parts = block_parts(layer)
        assert [part.name for part in parts] == PART_NAMES
        state = run_parts(parts, {"x": x, "cos": position[0], "sin": position[1]})
    difference = (state["out"] - expected).abs().max().item()
    assert difference <= 1e-5
