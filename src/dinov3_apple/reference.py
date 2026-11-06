"""The configurations of Step 2, the torch runs in each dtype mode and the stored files.

The reference output is the output of the Hugging Face backbone on torch-cpu in fp32. The parity
tests run torch-mps (and later mlx) on the same stored pixels and compare with it.
"""

import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors import safe_open
from transformers import DINOv3ViTModel

from dinov3_apple.hub import MODEL_IDS, weights_dir
from dinov3_apple.memory import check_free_memory
from dinov3_apple.outputs import PATCH_SIZE
from dinov3_apple.preprocess import own_aspect_size

REPO_ROOT = Path(__file__).parents[2]
IMAGES_DIR = REPO_ROOT / "tests" / "fixtures" / "images"
REFERENCE_DIR = REPO_ROOT / "tests" / "fixtures" / "generated" / "reference"
INPUTS_FILE = REFERENCE_DIR / "inputs.safetensors"

TEST_IMAGES = ["iss-kibo-lab", "forest", "iss-conference"]
SQUARE_SIZES = [224, 448, 512]
HIDDEN_STATES_CONFIG = "iss-kibo-lab/224x224"

# The memory check of a model load: LOAD_RATIO x the weight bytes + LOAD_RESERVE_BYTES.
LOAD_RATIO = 1.2
LOAD_RESERVE_BYTES = 4 * 10**9

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


def parameter_count(model_size: str) -> int:
    """Return the number of weights of one model size, from the safetensors headers."""
    count = 0
    for file in sorted(weights_dir(model_size).glob("*.safetensors")):
        with safe_open(file, "np") as tensors:
            for name in tensors.keys():  # noqa: SIM118 (safe_open is not iterable)
                count += int(np.prod(tensors.get_slice(name).get_shape()))
    return count


def required_load_bytes(model_size: str, dtype: torch.dtype, ratio: float = LOAD_RATIO) -> int:
    """Return the memory that the memory check asks for before a load of one model size.

    Args:
        model_size: A key of MODEL_IDS.
        dtype: The dtype of the loaded weights.
        ratio: The ratio of the load peak to the weight bytes.

    Returns:
        ratio x the weight bytes in dtype + LOAD_RESERVE_BYTES.
    """
    weight_bytes = parameter_count(model_size) * dtype.itemsize
    return round(ratio * weight_bytes) + LOAD_RESERVE_BYTES


def load_torch_backbone(
    model_size: str, device: str, dtype_mode: str, ratio: float = LOAD_RATIO
) -> DINOv3ViTModel:
    """Load the Hugging Face backbone of one model size after a memory check.

    Args:
        model_size: A key of MODEL_IDS.
        device: "cpu" or "mps".
        dtype_mode: A key of DTYPE_MODES. It sets the dtype of the weights.
        ratio: The ratio of the load peak to the weight bytes for the memory check.

    Returns:
        The backbone in eval mode on the device.

    Raises:
        InsufficientMemoryError: If the memory check fails.
        WeightsNotFoundError: If the weights are not on this Mac.
    """
    dtype = DTYPE_MODES[dtype_mode].weights
    label = f"{model_size} torch-{device} {dtype_mode}"
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
