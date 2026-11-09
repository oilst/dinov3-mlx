"""Hugging Face ids of the DINOv3 model sizes, the download of their weights and their folder."""

import json
import shutil
from pathlib import Path

import numpy as np
from huggingface_hub import snapshot_download, try_to_load_from_cache
from safetensors import safe_open

MODEL_IDS: dict[str, str] = {
    "ViT-S/16": "facebook/dinov3-vits16-pretrain-lvd1689m",
    "ViT-S+/16": "facebook/dinov3-vits16plus-pretrain-lvd1689m",
    "ViT-B/16": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "ViT-L/16": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "ViT-H+/16": "facebook/dinov3-vith16plus-pretrain-lvd1689m",
    "ViT-7B/16": "facebook/dinov3-vit7b16-pretrain-lvd1689m",
}

# The converted weights of scripts/convert_meta_checkpoint.py. Only a fallback.
CONVERTED_DIR = Path.home() / ".cache" / "dinov3-mlx" / "converted"

# The files that a backbone needs. ViT-7B/16 has six shards and an index.
WEIGHT_PATTERNS: list[str] = ["config.json", "*.safetensors", "model.safetensors.index.json"]


class WeightsNotFoundError(FileNotFoundError):
    """Neither the Hugging Face cache nor the converted folder has all weight files."""


class InsufficientDiskError(RuntimeError):
    """A download would leave less free disk than the limit."""


def check_free_disk(required_bytes: int, keep_free_bytes: int, path: Path) -> None:
    """Make sure that a download leaves enough free disk.

    Args:
        required_bytes: The size of the download.
        keep_free_bytes: The free disk that must stay after the download.
        path: A path on the disk that receives the download.

    Raises:
        InsufficientDiskError: If the free disk after the download is less than keep_free_bytes.
    """
    free_bytes = shutil.disk_usage(path).free
    if free_bytes - required_bytes < keep_free_bytes:
        raise InsufficientDiskError(
            f"{path}: {free_bytes / 1e9:.1f} GB free, the download needs "
            f"{required_bytes / 1e9:.1f} GB and {keep_free_bytes / 1e9:.1f} GB must stay free."
        )


def download(model_id: str) -> Path:
    """Download the weights of one model size into the Hugging Face cache.

    The DINOv3 repos are gated. The Hugging Face token must have access.

    Args:
        model_id: A Hugging Face id from MODEL_IDS.

    Returns:
        The path of the snapshot in the Hugging Face cache.
    """
    return Path(snapshot_download(model_id, allow_patterns=WEIGHT_PATTERNS))


def _cached(model_id: str, filename: str, cache_dir: Path | None) -> Path | None:
    path = try_to_load_from_cache(model_id, filename, cache_dir=cache_dir)
    return Path(path) if isinstance(path, str) else None


def _has_all_weight_files(model_id: str, cache_dir: Path | None) -> bool:
    index = _cached(model_id, "model.safetensors.index.json", cache_dir)
    if index is None:
        names = {"model.safetensors"}
    else:
        names = set(json.loads(index.read_text())["weight_map"].values())
    return all(_cached(model_id, name, cache_dir) is not None for name in names)


def weights_dir(
    model_size: str, cache_dir: Path | None = None, converted_dir: Path = CONVERTED_DIR
) -> Path:
    """Return the folder with config.json and the safetensors files of one model size.

    The Hugging Face cache comes first, the converted folder second.

    Args:
        model_size: A key of MODEL_IDS, for example "ViT-S/16".
        cache_dir: The Hugging Face cache. None means the default cache.
        converted_dir: The folder of the converted weights.

    Returns:
        The snapshot folder in the Hugging Face cache, or the converted folder.

    Raises:
        WeightsNotFoundError: If neither place has all weight files.
    """
    model_id = MODEL_IDS[model_size]
    config = _cached(model_id, "config.json", cache_dir)
    if config is not None and _has_all_weight_files(model_id, cache_dir):
        return config.parent
    converted = converted_dir / model_id
    if (converted / "config.json").exists() and any(converted.glob("*.safetensors")):
        return converted
    raise WeightsNotFoundError(f"{model_size}: not in the Hugging Face cache or in {converted}")


def parameter_count(model_size: str) -> int:
    """Return the number of weights of one model size, from the safetensors headers."""
    count = 0
    for file in sorted(weights_dir(model_size).glob("*.safetensors")):
        with safe_open(file, "np") as tensors:
            for name in tensors.keys():  # noqa: SIM118 (safe_open is not iterable)
                count += int(np.prod(tensors.get_slice(name).get_shape()))
    return count
