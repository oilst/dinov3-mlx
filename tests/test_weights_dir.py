from pathlib import Path

import pytest

from dinov3_apple.hub import MODEL_IDS, WeightsNotFoundError, weights_dir

MODEL_SIZE = "ViT-S/16"


def _fake_hub_cache(cache_dir: Path, model_id: str) -> Path:
    repo = cache_dir / f"models--{model_id.replace('/', '--')}"
    snapshot = repo / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text("abc123")
    (snapshot / "config.json").write_text("{}")
    (snapshot / "model.safetensors").write_bytes(b"x")
    return snapshot


def _fake_converted(converted_dir: Path, model_id: str) -> Path:
    folder = converted_dir / model_id
    folder.mkdir(parents=True)
    (folder / "config.json").write_text("{}")
    (folder / "model.safetensors").write_bytes(b"x")
    return folder


def test_weights_dir_prefers_hugging_face_cache(tmp_path: Path) -> None:
    model_id = MODEL_IDS[MODEL_SIZE]
    snapshot = _fake_hub_cache(tmp_path / "hub", model_id)
    _fake_converted(tmp_path / "converted", model_id)
    found = weights_dir(
        MODEL_SIZE, cache_dir=tmp_path / "hub", converted_dir=tmp_path / "converted"
    )
    assert found == snapshot


def test_weights_dir_falls_back_to_converted_folder(tmp_path: Path) -> None:
    model_id = MODEL_IDS[MODEL_SIZE]
    (tmp_path / "hub").mkdir()
    folder = _fake_converted(tmp_path / "converted", model_id)
    found = weights_dir(
        MODEL_SIZE, cache_dir=tmp_path / "hub", converted_dir=tmp_path / "converted"
    )
    assert found == folder


def test_weights_dir_falls_back_when_cache_misses_a_shard(tmp_path: Path) -> None:
    model_id = MODEL_IDS[MODEL_SIZE]
    snapshot = _fake_hub_cache(tmp_path / "hub", model_id)
    (snapshot / "model.safetensors").unlink()
    (snapshot / "model.safetensors.index.json").write_text(
        '{"weight_map": {"a": "model-00001-of-00002.safetensors"}}'
    )
    folder = _fake_converted(tmp_path / "converted", model_id)
    found = weights_dir(
        MODEL_SIZE, cache_dir=tmp_path / "hub", converted_dir=tmp_path / "converted"
    )
    assert found == folder


def test_weights_dir_raises_when_no_weights(tmp_path: Path) -> None:
    (tmp_path / "hub").mkdir()
    with pytest.raises(WeightsNotFoundError):
        weights_dir(MODEL_SIZE, cache_dir=tmp_path / "hub", converted_dir=tmp_path / "converted")
