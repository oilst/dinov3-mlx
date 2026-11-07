"""The torch runs of Step 2 in each dtype mode, and the load of the Hugging Face backbone.

The reference output is the output of the Hugging Face backbone on torch-cpu in fp32. The parity
tests run torch-mps and mlx on the same stored pixels and compare with it. The configurations and
the readers of the stored files live in stored.py, which imports no torch.
"""

from dataclasses import dataclass

import numpy as np
import torch
from transformers import DINOv3ViTModel

from dinov3_apple.hub import parameter_count, weights_dir
from dinov3_apple.memory import LOAD_RATIO, check_free_memory, check_run_rule, load_bytes
from dinov3_apple.parity import TEST_IMAGES
from dinov3_apple.stored import (
    HIDDEN_STATES_CONFIG,
    IMAGES_DIR,
    INPUTS_FILE,
    REFERENCE_DIR,
    REPO_ROOT,
    SQUARE_SIZES,
    Configuration,
    configurations,
    model_slug,
    read_inputs,
    read_reference,
    reference_path,
)

# The readers of the stored files moved to stored.py, which imports no torch (Step 4). This module
# re-exports them, so the imports of Steps 2 and 3 keep working.
__all__ = [
    "DTYPE_MODES",
    "HIDDEN_STATES_CONFIG",
    "IMAGES_DIR",
    "INPUTS_FILE",
    "LOAD_RATIO",
    "REFERENCE_DIR",
    "REPO_ROOT",
    "REQUIRED_METADATA",
    "SQUARE_SIZES",
    "TEST_IMAGES",
    "Configuration",
    "DtypeMode",
    "configurations",
    "load_torch_backbone",
    "model_slug",
    "parameter_count",
    "read_inputs",
    "read_reference",
    "reference_path",
    "required_load_bytes",
    "run_torch",
    "run_torch_hidden_states",
]

# The fields of the header metadata of a reference file (Section 3.4 of the plan).
REQUIRED_METADATA = [
    "versions",
    "image_sha256",
    "preprocessing",
    "model_size",
    "num_blocks",
    "weight_source",
    "weight_sha256",
    "attn_implementation",
    "run_time_s",
    "peak_memory_bytes",
    "date",
]


@dataclass(frozen=True)
class DtypeMode:
    """How a backbone runs in one dtype mode.

    Attributes:
        weights: The dtype of the weights and the inputs.
        autocast: The dtype of torch.autocast, or None for no autocast.
    """

    weights: torch.dtype
    autocast: torch.dtype | None


# Table 5 of specs/02-reference/plan.html.
DTYPE_MODES: dict[str, DtypeMode] = {
    "fp32": DtypeMode(torch.float32, None),
    "fp16-cast": DtypeMode(torch.float16, None),
    "bf16-cast": DtypeMode(torch.bfloat16, None),
    "fp16-autocast": DtypeMode(torch.float32, torch.float16),
    "bf16-autocast": DtypeMode(torch.float32, torch.bfloat16),
}


def required_load_bytes(model_size: str, dtype: torch.dtype, ratio: float = LOAD_RATIO) -> int:
    """Return the memory that the memory check asks for before a load of one model size.

    Args:
        model_size: A key of MODEL_IDS.
        dtype: The dtype of the loaded weights.
        ratio: The ratio of the load peak to the weight bytes.

    Returns:
        ratio x the weight bytes in dtype + LOAD_RESERVE_BYTES.
    """
    return load_bytes(parameter_count(model_size) * dtype.itemsize, ratio)


def load_torch_backbone(
    model_size: str, device: str, dtype_mode: str, ratio: float = LOAD_RATIO
) -> DINOv3ViTModel:
    """Load the Hugging Face backbone of one model size after the run rule and a memory check.

    Args:
        model_size: A key of MODEL_IDS.
        device: "cpu" or "mps".
        dtype_mode: A key of DTYPE_MODES. It sets the dtype of the weights.
        ratio: The ratio of the load peak to the weight bytes for the memory check.

    Returns:
        The backbone in eval mode on the device.

    Raises:
        RunRuleError: For ViT-7B/16 with fp32 weights (fp32 and the two autocast modes), before
            the memory check and before any file read.
        InsufficientMemoryError: If the memory check fails.
        WeightsNotFoundError: If the weights are not on this Mac.
    """
    dtype = DTYPE_MODES[dtype_mode].weights
    label = f"{model_size} torch-{device} {dtype_mode}"
    check_run_rule(model_size, dtype.itemsize, label)
    check_free_memory(required_load_bytes(model_size, dtype, ratio), label)
    model = DINOv3ViTModel.from_pretrained(weights_dir(model_size), dtype=dtype)
    return model.to(device).eval()


def _device_type(model: torch.nn.Module) -> str:
    return next(model.parameters()).device.type


@torch.inference_mode()
def run_torch(model: DINOv3ViTModel, pixels: np.ndarray, dtype_mode: str) -> np.ndarray:
    """Run the backbone on stored pixels and return all tokens after the final layer norm.

    Args:
        model: A backbone from load_torch_backbone.
        pixels: Normalized fp32 pixels with shape (B, 3, H, W).
        dtype_mode: A key of DTYPE_MODES. It must match the dtype of the model weights.

    Returns:
        The tokens with shape (B, 1 + 4 + N, D), cast to fp32.
    """
    mode = DTYPE_MODES[dtype_mode]
    device = _device_type(model)
    tensor = torch.from_numpy(pixels).to(device=device, dtype=mode.weights)
    if mode.autocast is None:
        output = model(pixel_values=tensor)
    else:
        with torch.autocast(device_type=device, dtype=mode.autocast):
            output = model(pixel_values=tensor)
    return output.last_hidden_state.float().cpu().numpy()


@torch.inference_mode()
def run_torch_hidden_states(model: DINOv3ViTModel, pixels: np.ndarray) -> np.ndarray:
    """Return the token sequence before block 1 and after each block, before the final layer norm.

    Args:
        model: A backbone with fp32 weights.
        pixels: Normalized fp32 pixels with shape (1, 3, H, W).

    Returns:
        The hidden states with shape (L + 1, 1 + 4 + N, D) in fp32.
    """
    tensor = torch.from_numpy(pixels).to(_device_type(model))
    output = model(pixel_values=tensor, output_hidden_states=True)
    hidden = torch.stack([state[0] for state in output.hidden_states])
    if hidden.shape[0] != model.config.num_hidden_layers + 1:
        raise ValueError(f"{hidden.shape[0]} hidden states for {model.config.num_hidden_layers}")
    return hidden.float().cpu().numpy()
