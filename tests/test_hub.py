import json
import re
import shutil
from pathlib import Path

import pytest
from huggingface_hub import try_to_load_from_cache

from dinov3_apple.hub import MODEL_IDS, InsufficientDiskError, check_free_disk

TECH_STACK = Path(__file__).parents[1] / "tech-stack.md"


def test_model_ids_cover_six_model_sizes() -> None:
    documented_ids = re.findall(r"`(facebook/dinov3-[^`]+)`", TECH_STACK.read_text())
    assert sorted(MODEL_IDS.values()) == sorted(documented_ids)
    assert len(MODEL_IDS) == 6


def test_check_free_disk_raises_when_space_is_too_small(tmp_path: Path) -> None:
    free_bytes = shutil.disk_usage(tmp_path).free
    with pytest.raises(InsufficientDiskError):
        check_free_disk(required_bytes=free_bytes + 10**12, keep_free_bytes=1, path=tmp_path)


def test_check_free_disk_passes_when_space_is_enough(tmp_path: Path) -> None:
    check_free_disk(required_bytes=0, keep_free_bytes=0, path=tmp_path)


def _cached_path(model_id: str, filename: str) -> Path | None:
    path = try_to_load_from_cache(model_id, filename)
    return Path(path) if isinstance(path, str) else None


@pytest.mark.parametrize("model_size", MODEL_IDS)
def test_weights_are_in_cache(model_size: str) -> None:
    model_id = MODEL_IDS[model_size]
    if _cached_path(model_id, "config.json") is None:
        pytest.skip(
            f"{model_size} is not in the Hugging Face cache. Run scripts/download_weights.py."
        )

    index = _cached_path(model_id, "model.safetensors.index.json")
    if index is None:
        weight_files = {"model.safetensors"}
    else:
        weight_files = set(json.loads(index.read_text())["weight_map"].values())
    missing = sorted(name for name in weight_files if _cached_path(model_id, name) is None)
    assert not missing, f"{model_size}: missing {missing}"
