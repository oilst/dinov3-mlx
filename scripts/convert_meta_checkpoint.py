# Adapted from transformers/models/dinov3_vit/convert_dinov3_vit_to_hf.py
# Copyright 2025 The HuggingFace Team. Licensed under the Apache License, Version 2.0.
"""Convert a local Meta DINOv3 checkpoint (.pth) into a Hugging Face model folder.

Use this script only while the gated Hugging Face repos are not available. The output goes to
~/.cache/dinov3-mlx/converted/<Hugging Face id>, not into the Hugging Face cache.
"""

import argparse
import hashlib
import re
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import DINOv3ViTConfig, DINOv3ViTModel

from dinov3_mlx.hub import CONVERTED_DIR, MODEL_IDS

TEST_IMAGE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"
TEST_IMAGE = (
    Path(__file__).parents[1] / "tests" / "fixtures" / "generated" / "coco-000000039769.jpg"
)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

CONFIGS: dict[str, dict] = {
    "ViT-S/16": {
        "hidden_size": 384,
        "intermediate_size": 1536,
        "num_hidden_layers": 12,
        "num_attention_heads": 6,
        "use_gated_mlp": False,
        "hidden_act": "gelu",
    },
    "ViT-S+/16": {
        "hidden_size": 384,
        "intermediate_size": 1536,
        "num_hidden_layers": 12,
        "num_attention_heads": 6,
        "use_gated_mlp": True,
        "hidden_act": "silu",
    },
    "ViT-B/16": {
        "hidden_size": 768,
        "intermediate_size": 3072,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "use_gated_mlp": False,
        "hidden_act": "gelu",
    },
    "ViT-L/16": {
        "hidden_size": 1024,
        "intermediate_size": 4096,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "use_gated_mlp": False,
        "hidden_act": "gelu",
    },
    "ViT-H+/16": {
        "hidden_size": 1280,
        "intermediate_size": 5120,
        "num_hidden_layers": 32,
        "num_attention_heads": 20,
        "use_gated_mlp": True,
        "hidden_act": "silu",
    },
    "ViT-7B/16": {
        "hidden_size": 4096,
        "intermediate_size": 8192,
        "num_hidden_layers": 40,
        "num_attention_heads": 32,
        "query_bias": False,
        "value_bias": False,
        "use_gated_mlp": True,
        "hidden_act": "silu",
    },
}

# The first 5 values of the frame embedding and of the first patch embedding of TEST_IMAGE at
# 224 x 224 px, from the Hugging Face converter.
EXPECTED: dict[str, tuple[list[float], list[float]]] = {
    "ViT-S/16": (
        [0.463561, -0.415609, 0.408236, -0.126613, -0.286636],
        [-0.038754, -0.250895, -0.016392, -0.455473, 0.571582],
    ),
    "ViT-S+/16": (
        [-0.471349, -1.365778, -0.317983, 0.377219, -0.769085],
        [0.144551, -0.388117, -0.393433, -0.157695, -0.600380],
    ),
    "ViT-B/16": (
        [1.034643, -0.180609, -0.341018, -0.066376, -0.011383],
        [-0.082523, -0.456272, -0.728029, -0.430680, -0.152880],
    ),
    "ViT-L/16": (
        [0.484527, -0.582214, 0.480636, 0.592040, 0.945166],
        [-0.211367, -0.490863, -0.257131, 0.101763, 0.154511],
    ),
    "ViT-H+/16": (
        [-0.064575, -0.148866, -0.621524, 0.634878, 0.152695],
        [-0.093817, 0.287407, -0.050036, 0.428043, 0.094561],
    ),
    "ViT-7B/16": (
        [0.275439, -0.261353, 0.067772, 0.049936, -0.158747],
        [0.044442, -0.052542, 0.070777, -0.065111, -0.026546],
    ),
}

KEY_RENAMES: list[tuple[str, str]] = [
    (r"^cls_token$", "embeddings.cls_token"),
    (r"^mask_token$", "embeddings.mask_token"),
    (r"^storage_tokens$", "embeddings.register_tokens"),
    (r"^patch_embed\.proj\.", "embeddings.patch_embeddings."),
    (r"^blocks\.(\d+)\.attn\.proj\.", r"model.layer.\1.attention.o_proj."),
    (r"^blocks\.(\d+)\.attn\.", r"model.layer.\1.attention."),
    (r"^blocks\.(\d+)\.ls(\d+)\.gamma$", r"model.layer.\1.layer_scale\2.lambda1"),
    (r"^blocks\.(\d+)\.mlp\.fc1\.", r"model.layer.\1.mlp.up_proj."),
    (r"^blocks\.(\d+)\.mlp\.fc2\.", r"model.layer.\1.mlp.down_proj."),
    (r"^blocks\.(\d+)\.mlp\.w1\.", r"model.layer.\1.mlp.gate_proj."),
    (r"^blocks\.(\d+)\.mlp\.w2\.", r"model.layer.\1.mlp.up_proj."),
    (r"^blocks\.(\d+)\.mlp\.w3\.", r"model.layer.\1.mlp.down_proj."),
    (r"^blocks\.(\d+)\.norm(\d)\.", r"model.layer.\1.norm\2."),
]
SKIPPED_KEYS = re.compile(r"bias_mask|k_proj\.bias|local_cls_norm|^projectors\.|^rope_embed\.")


class ChecksumError(ValueError):
    """The SHA-256 of a checkpoint does not start with the hash in its file name."""


def check_checksum(checkpoint: Path) -> None:
    """Compare the SHA-256 of the checkpoint with the 8 hex digits at the end of its name."""
    expected = checkpoint.stem.rsplit("-", 1)[-1]
    sha256 = hashlib.sha256()
    with checkpoint.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 24), b""):
            sha256.update(chunk)
    if not sha256.hexdigest().startswith(expected):
        raise ChecksumError(f"{checkpoint.name}: SHA-256 {sha256.hexdigest()[:8]} != {expected}")


def convert_state_dict(meta_state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Rename the Meta keys to the Hugging Face keys and split each qkv weight into q, k and v."""
    converted = {}
    for key, tensor in meta_state_dict.items():
        if ".qkv." in key:
            for name, part in zip(("q_proj", "k_proj", "v_proj"), tensor.chunk(3, dim=0)):
                converted[key.replace("qkv", name)] = part
        else:
            converted[key] = tensor
    renamed = {}
    for key, tensor in converted.items():
        if SKIPPED_KEYS.search(key):
            continue
        for pattern, replacement in KEY_RENAMES:
            key = re.sub(pattern, replacement, key)
        renamed[key] = tensor.unsqueeze(1) if key == "embeddings.mask_token" else tensor
    return renamed


def load_test_image() -> torch.Tensor:
    """Return TEST_IMAGE as a normalized (1, 3, 224, 224) tensor.

    The resize works on float pixels with antialiasing, as torchvision's Resize in the Hugging Face
    converter does. A resize of 8-bit pixels moves the outputs by more than the 1e-3 tolerance.
    """
    if not TEST_IMAGE.exists():
        TEST_IMAGE.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(TEST_IMAGE_URL, TEST_IMAGE)
    with Image.open(TEST_IMAGE) as image:
        pixels = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0)
    resized = F.interpolate(tensor, size=(224, 224), mode="bilinear", antialias=True)
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (resized - mean) / std


@torch.inference_mode()
def check_outputs(model: DINOv3ViTModel, model_size: str) -> None:
    """Compare the outputs for TEST_IMAGE with the values of the Hugging Face converter."""
    output = model(pixel_values=load_test_image())
    frame = output.pooler_output[0, :5]
    patch = output.last_hidden_state[0, model.config.num_register_tokens + 1, :5]
    expected_frame, expected_patch = EXPECTED[model_size]
    print(f"frame embedding {[round(x, 6) for x in frame.tolist()]}, expected {expected_frame}")
    torch.testing.assert_close(frame, torch.tensor(expected_frame), atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(patch, torch.tensor(expected_patch), atol=1e-3, rtol=1e-3)


def main() -> None:
    """Convert one checkpoint, check it on the test image and save it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_size", choices=list(MODEL_IDS))
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()

    check_checksum(args.checkpoint)
    config = DINOv3ViTConfig(patch_size=16, num_register_tokens=4, **CONFIGS[args.model_size])
    model = DINOv3ViTModel(config).eval()
    meta_state_dict = torch.load(args.checkpoint, map_location="cpu", mmap=True, weights_only=True)
    model.load_state_dict(convert_state_dict(meta_state_dict), strict=True)
    check_outputs(model, args.model_size)

    out_dir = CONVERTED_DIR / MODEL_IDS[args.model_size]
    model.save_pretrained(out_dir)
    print(f"{args.model_size}: saved to {out_dir}")


if __name__ == "__main__":
    main()
