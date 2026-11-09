"""The MLX small model from saved weights, without torch (T4 and T5 of specs/07-release/plan.html).

save_small_model of small_model.py writes the weights with torch. This module reads them with mlx
only, so a process without torch and transformers can build the same MLX small model.
"""

import json
from pathlib import Path

import mlx.core as mx

from dinov3_mlx.mlx_model import MLX_DTYPE_MODES, MlxBackbone, MlxConfig, linear_parameter_names
from dinov3_mlx.mlx_weights import mlx_weights


def load_small_mlx(folder: Path, dtype_mode: str = "fp32") -> MlxBackbone:
    """Return the MLX backbone of a small model that save_small_model wrote.

    Args:
        folder: The folder with config.json and the safetensors files.
        dtype_mode: A key of MLX_DTYPE_MODES.

    Returns:
        The backbone in the dtype mode, with evaluated parameters.
    """
    model = MlxBackbone(
        MlxConfig.from_hf(json.loads((folder / "config.json").read_text())), dtype_mode
    )
    tensors: dict[str, mx.array] = {}
    for file in sorted(folder.glob("*.safetensors")):
        tensors |= mx.load(str(file))
    linear = linear_parameter_names(model)
    model.load_weights(list(mlx_weights(tensors, MLX_DTYPE_MODES[dtype_mode], linear)), strict=True)
    mx.eval(model.parameters())
    return model
