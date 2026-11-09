"""The torch runs of Step 2 in each dtype mode, and the load of the Hugging Face backbone.

The reference output is the output of the Hugging Face backbone on torch-cpu in fp32. The parity
tests run torch-mps and mlx on the same stored pixels and compare with it. The configurations and
the readers of the stored files live in stored.py, which imports no torch.

ViT-7B/16 never loads with fp32 weights. Its reference output is a streamed reference
(run_torch_streamed): the fp32 weights of one block at a time (path D4-Y, Section 3.8 of
specs/05-all-sizes/plan.html).
"""

import json
import logging
import math
import re
import time
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from torch import nn
from transformers import DINOv3ViTConfig, DINOv3ViTModel
from transformers.models.dinov3_vit.modeling_dinov3_vit import (
    DINOv3ViTEmbeddings,
    DINOv3ViTLayer,
    DINOv3ViTRopePositionEmbedding,
)

from dinov3_mlx.hub import parameter_count, weights_dir
from dinov3_mlx.memory import LOAD_RATIO, check_free_memory, check_run_rule, load_bytes
from dinov3_mlx.parity import TEST_IMAGES
from dinov3_mlx.stored import (
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
    "STREAMED_ATTN_IMPLEMENTATION",
    "TEST_IMAGES",
    "Configuration",
    "DtypeMode",
    "OuterParts",
    "ShardIndex",
    "configurations",
    "load_torch_backbone",
    "model_slug",
    "parameter_count",
    "read_inputs",
    "read_reference",
    "reference_path",
    "required_load_bytes",
    "required_streamed_bytes",
    "run_torch",
    "run_torch_hidden_states",
    "run_torch_streamed",
    "streamed_label",
]

logger = logging.getLogger(__name__)

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

# The attention of the reference files of Step 2 (their header field attn_implementation). The
# streamed reference uses the same, so its arithmetic is the arithmetic of a full-model run.
STREAMED_ATTN_IMPLEMENTATION = "sdpa"
# A block key of the safetensors files: layer.{i}.… in the Hub files, model.layer.{i}.… in a
# file that a transformers 5 model saves from its state dict.
BLOCK_KEY = re.compile(r"(?:model\.)?layer\.(?P<block>\d+)\.(?P<name>.+)")
# The bytes of one value of each safetensors dtype. All weights of the DINOv3 files are F32.
SAFETENSORS_DTYPE_BYTES = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2}


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


class ShardIndex:
    """The safetensors file of each tensor of one model folder, and the bytes of each tensor.

    A model with several files has model.safetensors.index.json, which maps each key to its file.
    ViT-7B/16 has six files, and five of its blocks span two files. A model with one file has no
    index. The constructor reads the headers only, no tensor data.

    Attributes:
        folder: The model folder with config.json.
        files: The file of each key, for example "layer.7.mlp.up_proj.weight".
        nbytes: The bytes of each tensor, from the headers.
        read_seconds: The time of all read calls so far. A read maps the file; the pages of a
            tensor enter memory at its first use, so this time does not include the disk reads.
    """

    def __init__(self, folder: Path) -> None:
        """Read the index, or the header of the one file, and the headers of all files.

        Args:
            folder: A folder from weights_dir.
        """
        self.folder = folder
        index = folder / "model.safetensors.index.json"
        if index.exists():
            weight_map = json.loads(index.read_text())["weight_map"]
            self.files = {key: folder / name for key, name in weight_map.items()}
        else:
            single = folder / "model.safetensors"
            with safe_open(single, "pt") as tensors:
                self.files = dict.fromkeys(tensors.keys(), single)
        self.nbytes: dict[str, int] = {}
        for file in sorted(set(self.files.values())):
            with safe_open(file, "pt") as tensors:
                for key in tensors.keys():  # noqa: SIM118 (safe_open is not iterable)
                    part = tensors.get_slice(key)
                    size = SAFETENSORS_DTYPE_BYTES[part.get_dtype()]
                    self.nbytes[key] = size * math.prod(part.get_shape())
        self._blocks: dict[int, dict[str, str]] = defaultdict(dict)
        for key in self.files:
            if match := BLOCK_KEY.fullmatch(key):
                self._blocks[int(match["block"])][match["name"]] = key
        self.read_seconds = 0.0

    @property
    def num_blocks(self) -> int:
        """The number of blocks in the files."""
        return len(self._blocks)

    def block_keys(self, block: int) -> dict[str, str]:
        """Return the key in the files of each parameter of one block.

        Args:
            block: The 0-based block index.

        Returns:
            {parameter name in DINOv3ViTLayer: key in the files}, for example
            {"norm1.weight": "layer.0.norm1.weight"}.

        Raises:
            KeyError: If the files have no block with this index.
        """
        if block not in self._blocks:
            raise KeyError(f"{self.folder}: no block {block}")
        return dict(self._blocks[block])

    def outer_keys(self, prefix: str) -> dict[str, str]:
        """Return the key in the files of each parameter of one part outside the blocks.

        Args:
            prefix: The module name, "embeddings" or "norm".

        Returns:
            {parameter name in the module: key in the files}.
        """
        return {
            key.removeprefix(f"{prefix}."): key
            for key in self.files
            if key.startswith(f"{prefix}.")
        }

    def block_bytes(self, block: int) -> int:
        """Return the bytes of the tensors of one block, from the headers."""
        return sum(self.nbytes[key] for key in self.block_keys(block).values())

    def outer_bytes(self) -> int:
        """Return the bytes of the tensors outside the blocks, with the mask token."""
        return sum(size for key, size in self.nbytes.items() if not BLOCK_KEY.fullmatch(key))

    def streamed_weight_bytes(self) -> int:
        """Return the largest weight bytes that the streamed path holds at one time: the largest
        block and the parts outside the blocks."""
        largest = max(self.block_bytes(block) for block in self._blocks)
        return largest + self.outer_bytes()

    def read(self, keys: Mapping[str, str]) -> dict[str, torch.Tensor]:
        """Read tensors from the files. Each file opens once for each call.

        safetensors maps the file privately: the tensors of one call share one map of each file,
        and the pages of a tensor enter memory at its first use. The map ends when the last tensor
        of the call is freed. So the pages of a freed block leave the process.

        Args:
            keys: {name in the result: key in the files}.

        Returns:
            The tensors in the dtype of the files, keyed by their names in the result.
        """
        start = time.perf_counter()
        by_file: dict[Path, list[tuple[str, str]]] = defaultdict(list)
        for name, key in keys.items():
            by_file[self.files[key]].append((name, key))
        tensors = {}
        for file, pairs in by_file.items():
            with safe_open(file, "pt") as handle:
                for name, key in pairs:
                    tensors[name] = handle.get_tensor(key)
        self.read_seconds += time.perf_counter() - start
        return tensors

    def block(self, block: int) -> dict[str, torch.Tensor]:
        """Read the tensors of one block, keyed by the parameter names of DINOv3ViTLayer."""
        return self.read(self.block_keys(block))


def _check_materialized(module: nn.Module, name: str) -> None:
    """Stop the run if a load with assign=True left a tensor on the meta device."""
    tensors = [*module.parameters(), *module.buffers()]
    if any(tensor.is_meta for tensor in tensors):
        raise ValueError(f"{name}: a tensor stayed on the meta device after the load")


@dataclass(frozen=True)
class OuterParts:
    """The parts of the backbone outside the blocks, on torch-cpu in fp32: the embeddings, the
    RoPE tables and the final layer norm, with the config of the model size.

    Attributes:
        config: The config of config.json with attn_implementation STREAMED_ATTN_IMPLEMENTATION.
        embeddings: The patch embedding, the CLS token and the register tokens.
        rope_embeddings: The RoPE tables in fp32, in eval mode (no RoPE-box jittering).
        norm: The final layer norm.
    """

    config: DINOv3ViTConfig
    embeddings: DINOv3ViTEmbeddings
    rope_embeddings: DINOv3ViTRopePositionEmbedding
    norm: nn.LayerNorm

    @classmethod
    def from_shards(cls, shards: ShardIndex) -> "OuterParts":
        """Build the parts on the meta device and load their tensors with assign=True.

        Args:
            shards: The index of the model folder.

        Returns:
            The parts in eval mode. Only the tensors outside the blocks are read.
        """
        config = DINOv3ViTConfig.from_pretrained(
            shards.folder, attn_implementation=STREAMED_ATTN_IMPLEMENTATION
        )
        with torch.device("meta"):
            embeddings = DINOv3ViTEmbeddings(config)
            norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        embeddings.load_state_dict(shards.read(shards.outer_keys("embeddings")), assign=True)
        norm.load_state_dict(shards.read(shards.outer_keys("norm")), assign=True)
        _check_materialized(embeddings, "embeddings")
        _check_materialized(norm, "norm")
        # inv_freq comes from the config, as in from_pretrained; it is not in the files.
        rope_embeddings = DINOv3ViTRopePositionEmbedding(config)
        return cls(config, embeddings.eval(), rope_embeddings.eval(), norm.eval())


def streamed_label(model_size: str) -> str:
    """Return the name of the streamed run of one model size in the memory-check lines.

    The name differs from the name of a full fp32 load ("{model_size} torch-cpu fp32: …"), so
    the logs show 0 full fp32 loads of ViT-7B/16.
    """
    return f"{model_size} torch-cpu fp32, streamed: one block at a time"


def required_streamed_bytes(model_size: str) -> int:
    """Return the need of the memory check of the streamed reference of one model size.

    LOAD_RATIO x (the fp32 bytes of the largest block + the parts outside the blocks) + 4 GB,
    from the safetensors headers: about 4.82 GB for ViT-7B/16. No tensor data is read.

    Args:
        model_size: A key of MODEL_IDS.

    Raises:
        WeightsNotFoundError: If the weights are not on this Mac.
    """
    return load_bytes(ShardIndex(weights_dir(model_size)).streamed_weight_bytes())


@torch.inference_mode()
def run_torch_streamed(
    model_size: str,
    pixels: Mapping[str, np.ndarray],
    hidden_config: str | None,
    shards: ShardIndex | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray | None]:
    """Return the reference tokens of each configuration with the fp32 weights of one block in
    memory at a time.

    The run uses the classes of the Hugging Face backbone, attn_implementation "sdpa", fp32 RoPE
    tables and batch size 1, as a full-model run does, so the arithmetic is the same (D4 of
    specs/05-all-sizes/explore.html). It runs the embeddings and the RoPE tables of all
    configurations first. Then, for each block: it reads the tensors of the block, builds one
    DINOv3ViTLayer on the meta device, loads the tensors with assign=True, pushes every
    configuration through the block and frees the block. At the end it applies the final layer
    norm. It never calls load_torch_backbone.

    Args:
        model_size: A key of MODEL_IDS.
        pixels: Normalized fp32 pixels with shape (1, 3, H, W), keyed by configuration id.
        hidden_config: The configuration whose hidden states the run keeps, or None.
        shards: The index of the weights of model_size. None reads it from weights_dir.

    Returns:
        The tokens of each configuration after the final layer norm, (1 + 4 + N, D) in fp32, and
        the L + 1 hidden states of hidden_config (the state before block 1 and after each block),
        stacked as (L + 1, 1 + 4 + N, D). None if hidden_config is None.

    Raises:
        ValueError: If hidden_config is not a key of pixels.
        InsufficientMemoryError: If the memory check fails. It runs before any tensor read.
        WeightsNotFoundError: If the weights are not on this Mac.
    """
    if hidden_config is not None and hidden_config not in pixels:
        raise ValueError(f"hidden_config {hidden_config} is not a key of pixels")
    # Run rule of CLAUDE.md: ViT-7B/16 never loads with fp32 weights. This path reads the rule as
    # a limit on the fp32 weights in memory: one block and the parts outside the blocks, never
    # the whole model. The user said yes to this reading on 2026-11-07 (Table 20 of
    # specs/05-all-sizes/plan.html). The guards of load_torch_backbone and load_mlx_backbone stay.
    shards = shards or ShardIndex(weights_dir(model_size))
    check_free_memory(load_bytes(shards.streamed_weight_bytes()), streamed_label(model_size))
    outer = OuterParts.from_shards(shards)
    states, rope = {}, {}
    for config, values in pixels.items():
        tensor = torch.from_numpy(values)
        states[config] = outer.embeddings(tensor)
        rope[config] = outer.rope_embeddings(tensor)
    hidden = [states[hidden_config][0].numpy().copy()] if hidden_config is not None else []
    for block in range(outer.config.num_hidden_layers):
        start = time.perf_counter()
        with torch.device("meta"):
            layer = DINOv3ViTLayer(outer.config)
        layer.load_state_dict(shards.block(block), assign=True)
        _check_materialized(layer, f"block {block}")
        layer.eval()
        for config, state in states.items():
            states[config] = layer(state, position_embeddings=rope[config])
        if hidden_config is not None:
            hidden.append(states[hidden_config][0].numpy().copy())
        del layer  # the fp32 weights of one block at a time
        logger.info(
            f"{model_size} streamed: block {block + 1} of {outer.config.num_hidden_layers} in "
            f"{time.perf_counter() - start:.1f} s"
        )
    tokens = {config: outer.norm(state)[0].numpy() for config, state in states.items()}
    return tokens, (np.stack(hidden) if hidden_config is not None else None)
