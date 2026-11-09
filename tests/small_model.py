"""The small DINOv3 backbones with random weights for the part tests (Section 6.1 of
specs/04-mlx-port/plan.html and of specs/05-all-sizes/plan.html).

The Hugging Face model runs on torch-cpu in fp32. It is the test oracle. The MLX backbone gets the
same weights through the key map, from the state dict with the spelling model.layer.{i}.… and
with the mask token. So each part test also tests the key map.
"""

import json
import math
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
from safetensors.torch import save_file
from transformers import DINOv3ViTConfig, DINOv3ViTModel

from dinov3_mlx.mlx_model import MLX_DTYPE_MODES, MlxBackbone, MlxConfig, linear_parameter_names
from dinov3_mlx.mlx_weights import mlx_weights

# The values of config.json of ViT-S/16, with 2 blocks.
SMALL_CONFIG = {
    "hidden_size": 384,
    "num_attention_heads": 6,
    "intermediate_size": 1536,
    "num_hidden_layers": 2,
    "num_register_tokens": 4,
    "patch_size": 16,
    "num_channels": 3,
    "image_size": 224,
    "rope_theta": 100.0,
    "layer_norm_eps": 1e-5,
    "query_bias": True,
    "key_bias": False,
    "value_bias": True,
    "proj_bias": True,
    "mlp_bias": True,
    "hidden_act": "gelu",
    "use_gated_mlp": False,
    "layerscale_value": 1.0,
    "drop_path_rate": 0.0,
    "attention_dropout": 0.0,
    "pos_embed_rescale": 2.0,
}

# The three small models of Section 6.1 of specs/05-all-sizes/plan.html.
SMALL_CONFIGS = {
    # The values of ViT-S/16: MLP with GELU.
    "vits16": SMALL_CONFIG,
    # The values of ViT-S+/16: SwiGLU with SiLU, FF 1,536, q and v biases, no k bias.
    "vits16plus": SMALL_CONFIG | {"hidden_act": "silu", "use_gated_mlp": True},
    # The shape of ViT-7B/16 at D = 256: head dimension 128, FF = 2 D, SwiGLU, no q, k, v biases.
    "vit7b16-shape": SMALL_CONFIG
    | {
        "hidden_size": 256,
        "num_attention_heads": 2,
        "intermediate_size": 512,
        "hidden_act": "silu",
        "use_gated_mlp": True,
        "query_bias": False,
        "key_bias": False,
        "value_bias": False,
    },
}

# Normal pixels, seed 1. The two shapes that are not square find a swap of height and width.
PIXEL_SHAPES = [(2, 3, 224, 224), (1, 3, 224, 304), (1, 3, 304, 224)]

# The patch grids of the 12 configurations.
GRIDS = [(14, 14), (28, 28), (32, 32), (14, 25), (14, 19), (19, 14)]


def _std_and_mean(name: str, param: torch.Tensor) -> tuple[float, float]:
    if "norm" in name:
        return 0.1, 1.0 if name.endswith("weight") else 0.0
    if name.endswith("lambda1"):
        return 0.3, 0.0
    if name.endswith("bias"):
        return 0.5, 0.0
    if name.endswith("weight"):
        return 1.0 / math.sqrt(param[0].numel()), 0.0
    return 0.5, 0.0  # the CLS token, the register tokens and the mask token


def small_hf_model(seed: int = 0, name: str = "vits16") -> DINOv3ViTModel:
    """Return a small Hugging Face backbone on torch-cpu in fp32, in eval mode.

    Every parameter gets new random values. The Hugging Face init sets the biases to 0 and lambda1
    to 1; with these values a wrong bias or a wrong LayerScale axis does not show.

    Args:
        seed: The seed of the random values.
        name: A key of SMALL_CONFIGS.

    Returns:
        The model. Its rope_embeddings run in eval mode, so RoPE-box jittering is off.
    """
    model = DINOv3ViTModel(DINOv3ViTConfig(**SMALL_CONFIGS[name])).eval()
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for param_name, param in model.named_parameters():
            std, mean = _std_and_mean(param_name, param)
            param.copy_(mean + std * torch.randn(param.shape, generator=generator))
    return model


def small_mlx_model(hf_model: DINOv3ViTModel, dtype_mode: str = "fp32") -> MlxBackbone:
    """Return the MLX backbone with the weights of a small Hugging Face backbone.

    Args:
        hf_model: A model from small_hf_model.
        dtype_mode: A key of MLX_DTYPE_MODES.

    Returns:
        The backbone in the dtype mode, with evaluated parameters.
    """
    model = MlxBackbone(MlxConfig.from_hf(hf_model.config.to_dict()), dtype_mode)
    tensors = {name: mx.array(t.detach().numpy()) for name, t in hf_model.state_dict().items()}
    pairs = list(mlx_weights(tensors, MLX_DTYPE_MODES[dtype_mode], linear_parameter_names(model)))
    model.load_weights(pairs, strict=True)
    mx.eval(model.parameters())
    return model


def small_pixels(shape: tuple[int, ...], seed: int = 1) -> np.ndarray:
    """Return normal fp32 pixels of one shape."""
    return np.random.default_rng(seed).standard_normal(shape).astype(np.float32)


def save_small_model(hf_model: DINOv3ViTModel, folder: Path, files: int) -> None:
    """Save the weights of a small model as one file, or as two files and an index, with the key
    spelling of the Hub files (layer.{i}.…) and its config.json.

    Args:
        hf_model: A model from small_hf_model.
        folder: The new folder. It must not exist.
        files: 1 for model.safetensors, 2 for two files and model.safetensors.index.json. The
            split runs through block 0, as the files of ViT-7B/16 split 5 blocks.
    """
    folder.mkdir()
    state = {
        name.removeprefix("model."): tensor.detach().contiguous()
        for name, tensor in hf_model.state_dict().items()
    }
    hf_model.config.to_json_file(folder / "config.json")
    if files == 1:
        save_file(state, folder / "model.safetensors", metadata={"format": "pt"})
        return
    names = list(state)
    split = next(i for i, name in enumerate(names) if name.startswith("layer.0.mlp"))
    weight_map = {}
    for part, chunk in enumerate((names[:split], names[split:]), start=1):
        file = f"model-0000{part}-of-00002.safetensors"
        save_file({n: state[n] for n in chunk}, folder / file, metadata={"format": "pt"})
        weight_map |= dict.fromkeys(chunk, file)
    index = {"metadata": {"total_size": 0}, "weight_map": weight_map}
    (folder / "model.safetensors.index.json").write_text(json.dumps(index))
