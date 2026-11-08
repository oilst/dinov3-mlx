import hashlib
import json
from pathlib import Path

import pytest
import torch
from convert_meta_checkpoint import (
    CONVERTED_DIR,
    ChecksumError,
    check_checksum,
    convert_state_dict,
)
from huggingface_hub import try_to_load_from_cache
from safetensors import safe_open
from under_test import SLOW

from dinov3_apple.hub import MODEL_IDS


def test_convert_state_dict_splits_qkv_and_renames_keys() -> None:
    qkv = torch.arange(6.0).reshape(6, 1)
    converted = convert_state_dict(
        {
            "storage_tokens": torch.zeros(1, 4, 2),
            "mask_token": torch.zeros(1, 2),
            "blocks.0.attn.qkv.weight": qkv,
            "blocks.0.attn.qkv.bias": torch.arange(6.0),
            "blocks.0.attn.qkv.bias_mask": torch.ones(6),
            "blocks.0.ls1.gamma": torch.ones(2),
            "blocks.0.mlp.w1.weight": torch.ones(2),
            "rope_embed.periods": torch.ones(2),
        }
    )
    assert set(converted) == {
        "embeddings.register_tokens",
        "embeddings.mask_token",
        "model.layer.0.attention.q_proj.weight",
        "model.layer.0.attention.k_proj.weight",
        "model.layer.0.attention.v_proj.weight",
        "model.layer.0.attention.q_proj.bias",
        "model.layer.0.attention.v_proj.bias",
        "model.layer.0.layer_scale1.lambda1",
        "model.layer.0.mlp.gate_proj.weight",
    }
    assert converted["model.layer.0.attention.k_proj.weight"].flatten().tolist() == [2.0, 3.0]
    assert converted["embeddings.mask_token"].shape == (1, 1, 2)


def test_check_checksum_accepts_matching_name(tmp_path: Path) -> None:
    content = b"weights"
    prefix = hashlib.sha256(content).hexdigest()[:8]
    checkpoint = tmp_path / f"model-{prefix}.pth"
    checkpoint.write_bytes(content)
    check_checksum(checkpoint)


def test_check_checksum_rejects_other_content(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model-00000000.pth"
    checkpoint.write_bytes(b"weights")
    with pytest.raises(ChecksumError):
        check_checksum(checkpoint)


def _hub_weight_files(model_id: str) -> list[Path] | None:
    index = try_to_load_from_cache(model_id, "model.safetensors.index.json")
    if isinstance(index, str):
        names = sorted(set(json.loads(Path(index).read_text())["weight_map"].values()))
    else:
        names = ["model.safetensors"]
    paths = [try_to_load_from_cache(model_id, name) for name in names]
    return [Path(path) for path in paths] if all(isinstance(p, str) for p in paths) else None


def _tensor_locations(files: list[Path]) -> dict[str, tuple[Path, str]]:
    """Map each tensor name without the "model." prefix to its file and its stored name.

    Older Hub files name the blocks "layer.N", newer transformers versions "model.layer.N".
    """
    locations = {}
    for file in files:
        with safe_open(file, "pt") as tensors:
            for key in list(tensors.keys()):
                locations[key.removeprefix("model.")] = (file, key)
    return locations


def _load(location: tuple[Path, str]) -> torch.Tensor:
    file, key = location
    with safe_open(file, "pt") as tensors:
        return tensors.get_tensor(key)


@pytest.mark.parametrize("model_size", MODEL_IDS)
def test_converted_weights_match_hub(model_size: str) -> None:
    """The default session checks ViT-S/16 only: the ViT-H+/16 file takes 14 s (finding V9 of
    specs/06b-prior-art/verify.html). The slow session checks every converted file."""
    if model_size != "ViT-S/16" and not SLOW:
        pytest.skip("slow session: DINOV3_SLOW=1")
    model_id = MODEL_IDS[model_size]
    converted = sorted((CONVERTED_DIR / model_id).glob("*.safetensors"))
    hub = _hub_weight_files(model_id)
    if not converted or hub is None:
        pytest.skip(f"{model_size} needs both a converted file and a Hub file.")

    ours, theirs = _tensor_locations(converted), _tensor_locations(hub)
    assert set(ours) == set(theirs)
    for name in ours:
        assert torch.equal(_load(ours[name]), _load(theirs[name])), name
