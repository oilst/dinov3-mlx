"""The MLX weights: the key map from the Hugging Face safetensors and the load of the MLX backbone.

The key map runs at load time; no converted file (D2 of specs/04-mlx-port/explore.html). Each
load checks the free memory first and raises the errors of the torch loader, so the test fixture
skips in the same way. No torch import.
"""

import json
import re
from collections.abc import Mapping
from functools import cache

import mlx.core as mx
from mlx.utils import tree_flatten

from dinov3_apple.hub import parameter_count, weights_dir
from dinov3_apple.memory import LOAD_RATIO, check_free_memory, load_bytes
from dinov3_apple.mlx_model import (
    MLX_DTYPE_MODES,
    Block,
    MlxBackbone,
    MlxConfig,
    TF32ModeError,
    tf32_enabled,
)

DROPPED_KEYS = frozenset({"embeddings.mask_token"})  # training only (bool_masked_pos)
PATCH_WEIGHT = "embeddings.patch_embeddings.weight"
OUTER_KEYS = frozenset(
    {
        "embeddings.cls_token",
        "embeddings.register_tokens",
        PATCH_WEIGHT,
        "embeddings.patch_embeddings.bias",
        "norm.weight",
        "norm.bias",
    }
)
# layer.{i}.{parameter} in the safetensors file, model.layer.{i}.{parameter} in a transformers 5
# model in memory. {parameter} must be a parameter name of the MLX block (block_parameter_names).
BLOCK_KEY = re.compile(r"(?:model\.)?(layer\.\d+\.(?P<parameter>.+))")


class UnknownWeightError(KeyError):
    """A tensor name of the weights has no place in the MLX backbone."""


@cache
def block_parameter_names() -> frozenset[str]:
    """Return the parameter names of one MLX block with all biases, for example norm1.weight.

    The names come from the Block module itself, so the key map and the module cannot differ.
    The block has a small width. Its arrays stay lazy and are never computed; its random init
    advances the global key of mx.random once, as the init of each MlxBackbone does.
    """
    config = MlxConfig(
        dim=8,
        heads=1,
        blocks=1,
        ff=8,
        patch_size=1,
        num_channels=1,
        num_register_tokens=0,
        eps=1e-6,
        rope_theta=100.0,
        query_bias=True,
        key_bias=True,
        value_bias=True,
        proj_bias=True,
        mlp_bias=True,
        gated=False,
        hidden_act="gelu",
    )
    return frozenset(name for name, _ in tree_flatten(Block(config).parameters()))


def map_key(name: str) -> str | None:
    """Return the MLX parameter name of one Hugging Face tensor name, or None for a dropped tensor.

    map_key checks the name and the parameter name of the block in a block key, not the block
    index: it does not know the number of blocks. load_weights(strict=True) stops a block index
    beyond the config, for example layer.12 of ViT-S/16, and a bias that the config does not have.

    Args:
        name: A tensor name of the safetensors file (layer.{i}.…) or of the state dict of a
            transformers 5 model (model.layer.{i}.…).

    Returns:
        The name in the MLX backbone, or None for embeddings.mask_token.

    Raises:
        UnknownWeightError: If the name is not in OUTER_KEYS or DROPPED_KEYS and is not
            layer.{i}. followed by a parameter name of the MLX block.
    """
    if name in DROPPED_KEYS:
        return None
    if name in OUTER_KEYS:
        return name
    if (match := BLOCK_KEY.fullmatch(name)) and match.group("parameter") in block_parameter_names():
        return match.group(1)
    raise UnknownWeightError(name)


def mlx_weights(tensors: Mapping[str, mx.array], dtype: mx.Dtype) -> list[tuple[str, mx.array]]:
    """Return the MLX weights as (name, array) pairs for load_weights(strict=True).

    The patch embedding weight (D, C, p, p) becomes (D, C · p · p): channel, row, column, as in
    patchify. Each array is cast to dtype. Lazy arrays stay lazy.

    Args:
        tensors: The tensors by Hugging Face name, for example from mx.load.
        dtype: The dtype of the dtype mode.

    Returns:
        The pairs, without the dropped tensors.

    Raises:
        UnknownWeightError: If map_key rejects a name. load_weights(strict=True) stops a name
            that map_key accepts and the backbone does not have (a block index beyond the config).
    """
    pairs = []
    for name, array in tensors.items():
        if (key := map_key(name)) is None:
            continue
        if key == PATCH_WEIGHT:
            array = array.reshape(array.shape[0], -1)
        pairs.append((key, array.astype(dtype)))
    return pairs


def required_mlx_load_bytes(
    model_size: str, dtype_mode: str = "fp32", ratio: float = LOAD_RATIO
) -> int:
    """Return the memory that the memory check asks for before an mlx load of one model size.

    Args:
        model_size: A key of MODEL_IDS.
        dtype_mode: A key of MLX_DTYPE_MODES.
        ratio: The ratio of the load peak to the weight bytes.

    Returns:
        ratio x the weight bytes in the dtype of the mode + LOAD_RESERVE_BYTES. ViT-S/16 in fp32:
        1.2 x 86.4 MB + 4 GB = 4.10 GB.

    Raises:
        WeightsNotFoundError: If the weights are not on this Mac.
    """
    return load_bytes(parameter_count(model_size) * MLX_DTYPE_MODES[dtype_mode].size, ratio)


def load_mlx_backbone(
    model_size: str, dtype_mode: str = "fp32", tf32: bool = False, ratio: float = LOAD_RATIO
) -> MlxBackbone:
    """Load the MLX backbone of one model size after a memory check (Section 3.3 of the plan).

    Args:
        model_size: A key of MODEL_IDS.
        dtype_mode: A key of MLX_DTYPE_MODES.
        tf32: The TF32 state that the fp32 dtype mode asks for. It must equal the state of
            MLX_ENABLE_TF32 in the environment.
        ratio: The ratio of the load peak to the weight bytes for the memory check.

    Returns:
        The backbone with evaluated parameters, so the load cost stays out of the first call.

    Raises:
        TF32ModeError: If the fp32 dtype mode asks for another TF32 state than MLX_ENABLE_TF32.
        InsufficientMemoryError: If the memory check fails.
        WeightsNotFoundError: If the weights are not on this Mac.
        NotImplementedError: For a model size with SwiGLU (Step 5).
    """
    dtype = MLX_DTYPE_MODES[dtype_mode]
    if dtype == mx.float32 and tf32_enabled() != tf32:
        raise TF32ModeError(f"set MLX_ENABLE_TF32={int(tf32)} before the first use of mlx")
    need = required_mlx_load_bytes(model_size, dtype_mode, ratio)
    check_free_memory(need, f"{model_size} mlx {dtype_mode}")
    mx.set_memory_limit(need)
    folder = weights_dir(model_size)
    model = MlxBackbone(MlxConfig.from_hf(json.loads((folder / "config.json").read_text())))
    tensors: dict[str, mx.array] = {}
    for file in sorted(folder.glob("*.safetensors")):
        tensors |= mx.load(str(file))  # lazy; a str, not a Path (R18)
    model.load_weights(mlx_weights(tensors, dtype), strict=True)
    mx.eval(model.parameters())  # the load cost stays out of the first forward pass (R14)
    return model
