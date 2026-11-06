"""Hugging Face ids of the DINOv3 model sizes and the download of their weights."""

import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

MODEL_IDS: dict[str, str] = {
    "ViT-S/16": "facebook/dinov3-vits16-pretrain-lvd1689m",
    "ViT-S+/16": "facebook/dinov3-vits16plus-pretrain-lvd1689m",
    "ViT-B/16": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "ViT-L/16": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "ViT-H+/16": "facebook/dinov3-vith16plus-pretrain-lvd1689m",
    "ViT-7B/16": "facebook/dinov3-vit7b16-pretrain-lvd1689m",
}

# The files that a backbone needs. ViT-7B/16 has six shards and an index.
WEIGHT_PATTERNS: list[str] = ["config.json", "*.safetensors", "model.safetensors.index.json"]


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
