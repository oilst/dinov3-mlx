"""Tests of the streamed reference of path D4-Y (Section 3.8 and Table 7 of
specs/05-all-sizes/plan.html): run_torch_streamed, ShardIndex and OuterParts.

The tests save small models as safetensors in a temporary folder, as two files and an index that
split block 0, as the files of ViT-7B/16 split 5 blocks. They need no gated weights, except the
test of the need of ViT-7B/16, which reads only the headers of its files.
"""

import logging
import weakref
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from fakes import fail_if_called
from safetensors.torch import save_file
from small_model import save_small_model, small_hf_model, small_pixels
from transformers import DINOv3ViTModel

import dinov3_mlx.reference as torch_reference
from dinov3_mlx.hub import WeightsNotFoundError
from dinov3_mlx.memory import InsufficientMemoryError, load_bytes
from dinov3_mlx.reference import (
    STREAMED_ATTN_IMPLEMENTATION,
    OuterParts,
    ShardIndex,
    required_streamed_bytes,
    run_torch,
    run_torch_hidden_states,
    run_torch_streamed,
)

# Three pixel shapes at batch size 1: a square grid and the two grids that swap height and width.
PIXELS = {
    "square/224x224": small_pixels((1, 3, 224, 224), seed=1),
    "wide/224x304": small_pixels((1, 3, 224, 304), seed=2),
    "tall/304x224": small_pixels((1, 3, 304, 224), seed=3),
}
# Not the first key of PIXELS: a run that keeps the states of the first configuration fails.
HIDDEN_CONFIG = "wide/224x304"
# The need of the memory check of the streamed reference of ViT-7B/16: 1.2 x (one block,
# 671,285,248 bytes, + the outer parts, 12,730,368 bytes) + 4 GB (Table 7 of the plan: 4.82 GB).
VIT7B16_STREAMED_NEED = 4_820_818_739


@pytest.fixture
def quiet_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    """Let the memory check pass, and keep its lines out of memory-checks.log of the session."""
    monkeypatch.setattr(logging.getLogger("dinov3_mlx.memory"), "handlers", [])
    monkeypatch.setattr("dinov3_mlx.memory.available_memory_bytes", lambda: 10**12)


def _folder(tmp_path: Path, name: str, files: int = 2, seed: int = 0) -> tuple[Path, dict]:
    """Save a small model and return its folder and its state dict with the file spelling."""
    hf_model = small_hf_model(seed=seed, name=name)
    folder = tmp_path / f"{name}-{files}"
    save_small_model(hf_model, folder, files=files)
    state = {key.removeprefix("model."): value for key, value in hf_model.state_dict().items()}
    return folder, state


def _use(monkeypatch: pytest.MonkeyPatch, folder: Path) -> None:
    monkeypatch.setattr("dinov3_mlx.reference.weights_dir", lambda model_size: folder)


@pytest.mark.parametrize("name", ["vits16", "vits16plus", "vit7b16-shape"])
def test_streamed_equals_the_full_model(
    name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, quiet_memory: None
) -> None:
    """The streamed tokens and hidden states equal run_torch and run_torch_hidden_states of the
    full model bit for bit, for an MLP, a SwiGLU and the shape of ViT-7B/16. The streamed path
    never calls load_torch_backbone and never loads the full model."""
    assert next(iter(PIXELS)) != HIDDEN_CONFIG
    folder, _ = _folder(tmp_path, name)
    full = DINOv3ViTModel.from_pretrained(folder, dtype=torch.float32).eval()
    assert full.config._attn_implementation == STREAMED_ATTN_IMPLEMENTATION
    expected = {config: run_torch(full, pixels, "fp32")[0] for config, pixels in PIXELS.items()}
    expected_hidden = run_torch_hidden_states(full, PIXELS[HIDDEN_CONFIG])
    del full

    _use(monkeypatch, folder)
    monkeypatch.setattr(torch_reference, "load_torch_backbone", fail_if_called("load_torch"))
    monkeypatch.setattr(DINOv3ViTModel, "from_pretrained", fail_if_called("from_pretrained"))
    tokens, hidden = run_torch_streamed("ViT-7B/16", PIXELS, HIDDEN_CONFIG)

    assert list(tokens) == list(PIXELS)
    for config, values in expected.items():
        assert tokens[config].shape == values.shape, config
        assert tokens[config].dtype == np.float32
        assert np.abs(tokens[config] - values).max() == 0.0, config
    assert hidden.shape == expected_hidden.shape == (3, 5 + 14 * 19, expected_hidden.shape[-1])
    assert np.abs(hidden - expected_hidden).max() == 0.0


def test_streamed_without_hidden_states(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, quiet_memory: None
) -> None:
    """hidden_config None returns no hidden states; an unknown hidden_config stops the run."""
    folder, _ = _folder(tmp_path, "vits16plus", files=1)
    _use(monkeypatch, folder)
    pixels = {HIDDEN_CONFIG: PIXELS[HIDDEN_CONFIG]}
    tokens, hidden = run_torch_streamed("ViT-S+/16", pixels, None)
    assert list(tokens) == [HIDDEN_CONFIG]
    assert hidden is None
    with pytest.raises(ValueError, match="tall/304x224"):
        run_torch_streamed("ViT-S+/16", pixels, "tall/304x224")


def test_shard_index_finds_a_block_across_two_files(tmp_path: Path) -> None:
    """ShardIndex maps each key to its file. Block 0 spans the two files; a folder with one file
    and no index gives the same tensors. The byte counts come from the headers."""
    two, state = _folder(tmp_path, "vit7b16-shape", files=2)
    one, _ = _folder(tmp_path, "vit7b16-shape", files=1)
    shards = ShardIndex(two)
    single = ShardIndex(one)
    assert set(shards.files) == set(single.files) == set(state)
    assert {shards.files[key].name for key in shards.block_keys(0).values()} == {
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    }
    assert {shards.files[key].name for key in shards.block_keys(1).values()} == {
        "model-00002-of-00002.safetensors"
    }
    assert set(single.files.values()) == {one / "model.safetensors"}
    assert shards.num_blocks == single.num_blocks == 2
    for index in (shards, single):
        block = index.block(0)
        assert sorted(block) == sorted(
            key.removeprefix("layer.0.") for key in state if key.startswith("layer.0.")
        )
        for key, tensor in block.items():
            assert torch.equal(tensor, state[f"layer.0.{key}"]), key
        embeddings = index.read(index.outer_keys("embeddings"))
        assert torch.equal(embeddings["cls_token"], state["embeddings.cls_token"])
        assert sorted(index.outer_keys("norm")) == ["bias", "weight"]
    block_bytes = 4 * sum(t.numel() for key, t in state.items() if key.startswith("layer.0."))
    outer_bytes = 4 * sum(t.numel() for key, t in state.items() if not key.startswith("layer."))
    assert shards.block_bytes(0) == shards.block_bytes(1) == block_bytes
    assert shards.outer_bytes() == outer_bytes
    assert shards.streamed_weight_bytes() == single.streamed_weight_bytes()
    assert shards.streamed_weight_bytes() == block_bytes + outer_bytes


def test_shard_index_reads_the_spelling_of_a_transformers_model(tmp_path: Path) -> None:
    """A file with the spelling model.layer.{i}.… gives the same blocks as layer.{i}.…."""
    hf_model = small_hf_model(seed=0, name="vits16")
    state = {key: value.detach().contiguous() for key, value in hf_model.state_dict().items()}
    assert "model.layer.1.norm1.weight" in state
    folder = tmp_path / "model-spelling"
    folder.mkdir()
    save_file(state, folder / "model.safetensors", metadata={"format": "pt"})
    hf_model.config.to_json_file(folder / "config.json")
    shards = ShardIndex(folder)
    assert shards.num_blocks == 2
    for key, tensor in shards.block(1).items():
        assert torch.equal(tensor, state[f"model.layer.1.{key}"]), key


def test_outer_parts_match_the_full_model(tmp_path: Path) -> None:
    """OuterParts has the config with attn_implementation "sdpa", the embeddings, the RoPE
    tables in eval mode and the final layer norm of the full model."""
    folder, state = _folder(tmp_path, "vits16")
    full = small_hf_model(seed=0, name="vits16")
    with torch.inference_mode():
        outer = OuterParts.from_shards(ShardIndex(folder))
        pixels = torch.from_numpy(PIXELS["wide/224x304"])
        assert torch.equal(outer.embeddings(pixels), full.embeddings(pixels))
        for mine, theirs in zip(outer.rope_embeddings(pixels), full.rope_embeddings(pixels)):
            assert mine.dtype == torch.float32
            assert torch.equal(mine, theirs)
    assert outer.config._attn_implementation == STREAMED_ATTN_IMPLEMENTATION
    assert outer.config.num_hidden_layers == 2
    assert not outer.rope_embeddings.training
    assert torch.equal(outer.norm.weight, state["norm.weight"])


@pytest.fixture
def tracked_blocks(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Track each DINOv3ViTLayer of the streamed path, each tensor that ShardIndex.block reads and
    each parameter that load_state_dict puts into a layer. load_state_dict(assign=True) wraps each
    read tensor in a new Parameter, so the read tensors can die while the parameters stay alive.
    At each new layer and at each block read, record how many layers and how many tensors and
    parameters of earlier blocks are alive. The key "alive" counts them at any time."""
    live_layers: weakref.WeakSet = weakref.WeakSet()
    read: list[list[weakref.ref]] = []
    events: dict[str, Any] = {"new": [], "read": []}

    def alive_tensors() -> int:
        return sum(ref() is not None for refs in read for ref in refs)

    class TrackedLayer(torch_reference.DINOv3ViTLayer):
        def __init__(self, config: object) -> None:
            events["new"].append((len(live_layers), alive_tensors()))
            super().__init__(config)
            live_layers.add(self)

        def load_state_dict(self, *args: Any, **kwargs: Any) -> Any:
            result = super().load_state_dict(*args, **kwargs)
            read.append([weakref.ref(parameter) for parameter in self.parameters()])
            return result

    original_block = ShardIndex.block

    def block(self: ShardIndex, index: int) -> dict[str, torch.Tensor]:
        events["read"].append((index, len(live_layers), alive_tensors()))
        tensors = original_block(self, index)
        read.append([weakref.ref(tensor) for tensor in tensors.values()])
        return tensors

    monkeypatch.setattr(torch_reference, "DINOv3ViTLayer", TrackedLayer)
    monkeypatch.setattr(ShardIndex, "block", block)
    events["alive"] = lambda: (len(live_layers), alive_tensors())
    return events


def test_streamed_frees_each_block(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    quiet_memory: None,
    tracked_blocks: dict[str, Any],
) -> None:
    """The streamed path holds the fp32 weights of one block at a time: each new layer finds no
    layer, no block tensor and no block parameter alive, each block read finds only the new empty
    layer, and no layer, no block tensor and no block parameter stays alive after the run."""
    folder, _ = _folder(tmp_path, "vit7b16-shape")
    _use(monkeypatch, folder)
    run_torch_streamed("ViT-7B/16", PIXELS, HIDDEN_CONFIG)
    assert tracked_blocks["new"] == [(0, 0), (0, 0)]
    assert tracked_blocks["read"] == [(0, 1, 0), (1, 1, 0)]
    assert tracked_blocks["alive"]() == (0, 0)


def test_streamed_checks_memory_before_any_tensor_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The memory check comes first, from the headers, before any tensor read. A low free memory
    stops the run with InsufficientMemoryError and no tensor read."""
    folder, _ = _folder(tmp_path, "vits16plus")
    _use(monkeypatch, folder)
    monkeypatch.setattr(logging.getLogger("dinov3_mlx.memory"), "handlers", [])
    shards = ShardIndex(folder)
    need = load_bytes(shards.streamed_weight_bytes())
    events: list[str] = []
    original_read = ShardIndex.read
    original_check = torch_reference.check_free_memory

    def read(self: ShardIndex, keys: dict[str, str]) -> dict[str, torch.Tensor]:
        events.append("read")
        return original_read(self, keys)

    def check(required_bytes: int, label: str) -> int:
        events.append(f"check {required_bytes} {label}")
        return original_check(required_bytes, label)

    monkeypatch.setattr(ShardIndex, "read", read)
    monkeypatch.setattr(torch_reference, "check_free_memory", check)
    monkeypatch.setattr("dinov3_mlx.memory.available_memory_bytes", lambda: 0)
    with pytest.raises(InsufficientMemoryError, match="streamed"):
        run_torch_streamed("ViT-S+/16", PIXELS, HIDDEN_CONFIG)
    assert events == [f"check {need} ViT-S+/16 torch-cpu fp32, streamed: one block at a time"]

    events.clear()
    monkeypatch.setattr("dinov3_mlx.memory.available_memory_bytes", lambda: 10**12)
    run_torch_streamed("ViT-S+/16", PIXELS, HIDDEN_CONFIG)
    assert events[0].startswith(f"check {need} ")
    assert events.count("read") == 2 + 2  # embeddings and norm, then two blocks
    assert all(event == "read" for event in events[1:])


def test_required_streamed_bytes_of_7b() -> None:
    """The need of the streamed reference of ViT-7B/16 from the headers of its six files: about
    4.82 GB. The read takes the headers only."""
    try:
        need = required_streamed_bytes("ViT-7B/16")
    except WeightsNotFoundError as error:
        pytest.skip(str(error))
    assert need == VIT7B16_STREAMED_NEED


def test_streamed_stops_at_a_missing_block_tensor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    quiet_memory: None,
    tracked_blocks: dict[str, Any],
) -> None:
    """T32 (V1 of specs/05-all-sizes/verify.html): a weights file without one tensor of block 1
    stops the run before block 1 runs. The defences are the strict load_state_dict and
    _check_materialized."""
    hf_model = small_hf_model(seed=0, name="vit7b16-shape")
    folder = tmp_path / "missing"
    folder.mkdir()
    state = {
        name.removeprefix("model."): tensor.detach().contiguous()
        for name, tensor in hf_model.state_dict().items()
    }
    missing = "layer.1.mlp.down_proj.weight"
    del state[missing]
    hf_model.config.to_json_file(folder / "config.json")
    save_file(state, folder / "model.safetensors", metadata={"format": "pt"})
    _use(monkeypatch, folder)
    ran: list[int] = []
    original = torch_reference.DINOv3ViTLayer.forward

    def forward(self: Any, *args: Any, **kwargs: Any) -> Any:
        ran.append(id(self))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(torch_reference.DINOv3ViTLayer, "forward", forward)
    # The first defence: the strict load_state_dict names the missing tensor.
    with pytest.raises(RuntimeError, match=r'Missing key.*"mlp\.down_proj\.weight"'):
        run_torch_streamed("ViT-7B/16", PIXELS, HIDDEN_CONFIG)
    assert len(tracked_blocks["read"]) == 2  # block 0 and block 1 were read
    assert len(set(ran)) <= 1  # only block 0 ran
    # The second defence: a tensor left on the meta device stops the run.
    with torch.device("meta"):
        layer = torch.nn.Linear(2, 2)
    with pytest.raises(ValueError, match="meta device"):
        torch_reference._check_materialized(layer, "block 1")
