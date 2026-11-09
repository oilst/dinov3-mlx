"""Write our ViT-H+/16 weights with the Meta names for the port C2 (mlx_dinov3), for job J5 only
(Table 1 of specs/06b-prior-art/plan.html).

The converter of C2 reads a pickle checkpoint, so J5 does not use it (ADR 0013). This script reads
the Hugging Face safetensors files (NumPy, no torch, no pickle) and writes one safetensors file
with the keys and layouts that the converter of C2 gives: the Meta names of
convert_meta_checkpoint.py backwards, one fused qkv matmul with a zero k bias, and the patch
embedding weight in the OHWI layout of the MLX Conv2d. The RoPE periods and the k-bias mask are
not in the file; the port worker keeps the values of the port for them.

Output: ~/.cache/dinov3-mlx/prior-art/weights/ours-meta-names/<slug>.safetensors.

Usage:
    uv run python -m dinov3_mlx.gpu_lock -- uv run python scripts/prior_art_meta_names.py ViT-H+/16
"""

import argparse
import re
from pathlib import Path

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

from dinov3_mlx.hub import weights_dir
from dinov3_mlx.prior_art import PRIOR_ART_DIR
from dinov3_mlx.stored import model_slug

OUT_DIR = PRIOR_ART_DIR / "weights" / "ours-meta-names"
# The Hugging Face names (without "model.") to the Meta names: KEY_RENAMES of
# convert_meta_checkpoint.py, backwards. q, k and v join in meta_tensors.
RENAMES: list[tuple[str, str]] = [
    (r"^embeddings\.cls_token$", "cls_token"),
    (r"^embeddings\.mask_token$", "mask_token"),
    (r"^embeddings\.register_tokens$", "storage_tokens"),
    (r"^embeddings\.patch_embeddings\.", "patch_embed.proj."),
    (r"^layer\.(\d+)\.attention\.o_proj\.", r"blocks.\1.attn.proj."),
    (r"^layer\.(\d+)\.layer_scale(\d)\.lambda1$", r"blocks.\1.ls\2.gamma"),
    (r"^layer\.(\d+)\.mlp\.gate_proj\.", r"blocks.\1.mlp.w1."),
    (r"^layer\.(\d+)\.norm(\d)\.", r"blocks.\1.norm\2."),
    (r"^norm\.", "norm."),
]
QKV = re.compile(r"^layer\.(\d+)\.attention\.([qkv])_proj\.(weight|bias)$")


def meta_name(name: str, gated: bool) -> str:
    """Return the Meta name of one Hugging Face tensor name (not q, k, v).

    up_proj is w2 in SwiGLU and fc1 in the MLP; down_proj is w3 or fc2.
    """
    ff = re.match(r"^layer\.(\d+)\.mlp\.(up|down)_proj\.(.+)$", name)
    if ff:
        block, kind, leaf = ff.groups()
        meta = {"up": "w2" if gated else "fc1", "down": "w3" if gated else "fc2"}[kind]
        return f"blocks.{block}.mlp.{meta}.{leaf}"
    for pattern, replacement in RENAMES:
        if re.match(pattern, name):
            return re.sub(pattern, replacement, name)
    raise KeyError(f"no Meta name for {name}")


def meta_tensors(tensors: dict[str, np.ndarray], gated: bool) -> dict[str, np.ndarray]:
    """Return the tensors with the Meta names and layouts of the converter of C2."""
    out: dict[str, np.ndarray] = {}
    qkv: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for name, tensor in tensors.items():
        name = name.removeprefix("model.")
        if match := QKV.match(name):
            block, part, leaf = match.groups()
            qkv.setdefault((block, leaf), {})[part] = tensor
            continue
        key = meta_name(name, gated)
        if key == "patch_embed.proj.weight":
            tensor = tensor.transpose(0, 2, 3, 1)  # OIHW to the OHWI of the MLX Conv2d
        if key == "mask_token":
            tensor = tensor.reshape(1, -1)
        out[key] = np.ascontiguousarray(tensor)
    for (block, leaf), parts in qkv.items():
        if leaf == "bias":
            dim = parts["q"].shape[0]
            parts.setdefault("k", np.zeros(dim, parts["q"].dtype))  # no k bias: the mask zeroes it
        out[f"blocks.{block}.attn.qkv.{leaf}"] = np.concatenate([parts[p] for p in "qkv"])
    return out


def main() -> None:
    """Read the Hugging Face files of one model size and write the Meta-name file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_size", choices=["ViT-H+/16"])
    args = parser.parse_args()
    folder = weights_dir(args.model_size)
    tensors = {}
    for file in sorted(folder.glob("*.safetensors")):
        with safe_open(file, "np") as source:
            names = source.keys()  # a safe_open object is not a dict
            tensors |= {key: source.get_tensor(key) for key in names}
    out = meta_tensors(tensors, gated=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = Path(OUT_DIR / f"{model_slug(args.model_size)}.safetensors")
    save_file(out, str(path))
    print(f"{args.model_size}: {len(out)} tensors to {path}")


if __name__ == "__main__":
    main()
