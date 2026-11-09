"""Tests of the key map, the MLX config and the load: T1 to T7b of specs/04-mlx-port/plan.html, and
T1 to T10, T34, H1 and H2 of specs/05-all-sizes/plan.html. Also the range check of the load
report in the backbone slot (F1 and F5 of specs/05-all-sizes/verify.html)."""

import dataclasses
import json
import logging
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import mlx.core as mx
import numpy as np
import parity_checks
import pytest
import torch
from fakes import VIT7B16_PARAMETER_COUNT, fail_if_called
from mlx.utils import tree_flatten
from parity_checks import load_peak_lower_bound, load_peak_target
from safetensors import safe_open
from small_model import SMALL_CONFIGS, save_small_model, small_hf_model

from dinov3_mlx.bench import DTYPE_OF_MODE, bytes_per_weight
from dinov3_mlx.hub import MODEL_IDS, WeightsNotFoundError, weights_dir
from dinov3_mlx.memory import InsufficientMemoryError, RunRuleError
from dinov3_mlx.mlx_model import (
    MLP,
    MLX_DTYPE_MODES,
    Block,
    MlxBackbone,
    MlxConfig,
    SwiGLU,
    TF32ModeError,
    linear_parameter_names,
)
from dinov3_mlx.mlx_weights import (
    PATCH_WEIGHT,
    LoadReport,
    UnknownWeightError,
    block_parameter_names,
    load_mlx_backbone,
    map_key,
    mlx_weights,
    required_mlx_load_bytes,
)
from dinov3_mlx.parity import THRESHOLDS

pytestmark = pytest.mark.mlx

VITS16_PARAMETER_COUNT = 21_596_544
HALF_MODES = ["fp16-cast", "bf16-cast", "fp16-autocast", "bf16-autocast"]
TORCH_DTYPES = {mx.float32: torch.float32, mx.float16: torch.float16, mx.bfloat16: torch.bfloat16}

# The values of config.json of ViT-S/16 (snapshot 114c137 of the Hub repo).
VITS16_CONFIG = {
    "architectures": ["DINOv3ViTModel"],
    "attention_dropout": 0.0,
    "drop_path_rate": 0.0,
    "hidden_act": "gelu",
    "hidden_size": 384,
    "image_size": 224,
    "initializer_range": 0.02,
    "intermediate_size": 1536,
    "key_bias": False,
    "layer_norm_eps": 1e-05,
    "layerscale_value": 1.0,
    "mlp_bias": True,
    "model_type": "dinov3_vit",
    "num_attention_heads": 6,
    "num_channels": 3,
    "num_hidden_layers": 12,
    "num_register_tokens": 4,
    "patch_size": 16,
    "pos_embed_jitter": None,
    "pos_embed_rescale": 2.0,
    "pos_embed_shift": None,
    "proj_bias": True,
    "query_bias": True,
    "rope_theta": 100.0,
    "torch_dtype": "float32",
    "use_gated_mlp": False,
    "value_bias": True,
}
# The values of config.json of ViT-7B/16 that differ from ViT-S/16.
VIT7B16_CONFIG = VITS16_CONFIG | {
    "hidden_act": "silu",
    "hidden_size": 4096,
    "intermediate_size": 8192,
    "num_attention_heads": 32,
    "num_hidden_layers": 40,
    "query_bias": False,
    "use_gated_mlp": True,
    "value_bias": False,
}


@dataclasses.dataclass(frozen=True)
class KeyMap:
    """The expected values of the key map of one model size (Table 4 of the plan)."""

    files: int
    tensors: int
    parameters: int
    linear: int
    stream: int
    gated: bool
    qv_bias: bool


KEY_MAPS = {
    "ViT-S/16": KeyMap(1, 211, 210, 134, 76, False, True),
    "ViT-S+/16": KeyMap(1, 235, 234, 158, 76, True, True),
    "ViT-B/16": KeyMap(1, 211, 210, 134, 76, False, True),
    "ViT-L/16": KeyMap(1, 415, 414, 266, 148, False, True),
    "ViT-H+/16": KeyMap(1, 615, 614, 418, 196, True, True),
    "ViT-7B/16": KeyMap(6, 687, 686, 442, 244, True, False),
}


def _torch_tensors(hf_model: torch.nn.Module) -> dict[str, mx.array]:
    return {name: mx.array(t.detach().numpy()) for name, t in hf_model.state_dict().items()}


def test_map_key_both_spellings() -> None:
    assert map_key("layer.3.attention.q_proj.weight") == "layer.3.attention.q_proj.weight"
    assert map_key("model.layer.3.attention.q_proj.weight") == "layer.3.attention.q_proj.weight"
    assert map_key("model.layer.11.layer_scale2.lambda1") == "layer.11.layer_scale2.lambda1"
    assert map_key("embeddings.cls_token") == "embeddings.cls_token"
    assert map_key(PATCH_WEIGHT) == PATCH_WEIGHT
    assert map_key("norm.bias") == "norm.bias"
    assert map_key("embeddings.mask_token") is None
    for name in ("rope_embeddings.inv_freq", "pooler.dense.weight"):
        with pytest.raises(UnknownWeightError):
            map_key(name)


def test_map_key_rejects_a_block_name_that_is_not_a_parameter() -> None:
    """A block key must name a parameter of the MLX block (F4 of specs/04-mlx-port/verify.html)."""
    for name in (
        "layer.0.foo.weight",
        "model.layer.0.attention.qkv.weight",
        "layer.3.norm3.bias",
        "layer.3.attention.q_proj.base_layer.weight",  # the name of a PEFT LoRA wrapper
        "layer.3.layer_scale1.gamma",
    ):
        with pytest.raises(UnknownWeightError):
            map_key(name)
    # map_key accepts each parameter name of a block with all biases, also k_proj.bias.
    config = dataclasses.replace(MlxConfig.from_hf(VITS16_CONFIG), key_bias=True)
    for name, _ in tree_flatten(Block(config).parameters()):
        assert map_key(f"model.layer.7.{name}") == f"layer.7.{name}"


def test_map_key_leaves_the_block_index_to_the_strict_load() -> None:
    """map_key does not know the number of blocks. The strict load stops a block index beyond it."""
    assert map_key("layer.12.norm1.weight") == "layer.12.norm1.weight"
    hf_model = small_hf_model()
    tensors = _torch_tensors(hf_model)
    tensors["layer.2.norm1.weight"] = tensors["model.layer.1.norm1.weight"]  # blocks 0 and 1 only
    model = MlxBackbone(MlxConfig.from_hf(hf_model.config.to_dict()))
    pairs = list(mlx_weights(tensors, MLX_DTYPE_MODES["fp32"], linear_parameter_names(model)))
    with pytest.raises(ValueError, match=r"layer\.2\.norm1\.weight"):
        model.load_weights(pairs, strict=True)


def test_mlx_weights_load_small_model() -> None:
    hf_model = small_hf_model()
    state = hf_model.state_dict()
    tensors = _torch_tensors(hf_model)
    model = MlxBackbone(MlxConfig.from_hf(hf_model.config.to_dict()))
    pairs = list(mlx_weights(tensors, MLX_DTYPE_MODES["fp32"], linear_parameter_names(model)))
    model.load_weights(pairs, strict=True)
    params = dict(tree_flatten(model.parameters()))
    assert len(params) == len(state) - 1
    for name, tensor in state.items():
        key = map_key(name)
        if key is None:
            continue
        expected = tensor.numpy().reshape(params[key].shape)
        assert params[key].dtype == mx.float32, name
        assert np.array_equal(np.array(params[key]), expected), name


def test_block_parameter_names_union() -> None:
    """T1: the names of an MLP block and of a SwiGLU block, both with all biases."""
    names = block_parameter_names()
    assert len(names) == 20
    assert {"mlp.gate_proj.weight", "mlp.gate_proj.bias", "attention.k_proj.bias"} <= names
    assert {"mlp.up_proj.weight", "mlp.down_proj.bias", "layer_scale2.lambda1"} <= names


def test_map_key_accepts_gate_proj() -> None:
    """T2: the gate_proj keys of both spellings map; Meta and wrong names do not."""
    assert map_key("layer.5.mlp.gate_proj.weight") == "layer.5.mlp.gate_proj.weight"
    assert map_key("model.layer.5.mlp.gate_proj.bias") == "layer.5.mlp.gate_proj.bias"
    for name in ("layer.5.mlp.w3.weight", "layer.5.mlp.gate.weight"):
        with pytest.raises(UnknownWeightError):
            map_key(name)


def test_gate_proj_of_an_mlp_model_fails_strict_load() -> None:
    """T3: map_key accepts a gate_proj key for an MLP model size; the strict load stops it."""
    hf_model = small_hf_model(name="vits16")
    tensors = _torch_tensors(hf_model)
    tensors["layer.0.mlp.gate_proj.weight"] = tensors["model.layer.0.mlp.up_proj.weight"]
    model = MlxBackbone(MlxConfig.from_hf(hf_model.config.to_dict()))
    pairs = list(mlx_weights(tensors, MLX_DTYPE_MODES["fp32"], linear_parameter_names(model)))
    with pytest.raises(ValueError, match=r"layer\.0\.mlp\.gate_proj\.weight"):
        model.load_weights(pairs, strict=True)


@pytest.mark.parametrize("dtype_mode", list(MLX_DTYPE_MODES))
def test_mlx_weights_pops_and_casts(dtype_mode: str) -> None:
    """T4: mlx_weights empties the dict and casts each tensor to its dtype of the mode."""
    hf_model = small_hf_model(name="vits16plus")
    state = hf_model.state_dict()
    dtypes = MLX_DTYPE_MODES[dtype_mode]
    model = MlxBackbone(MlxConfig.from_hf(hf_model.config.to_dict()), dtype_mode)
    linear = linear_parameter_names(model)
    assert "layer.1.mlp.gate_proj.weight" in linear
    assert "embeddings.patch_embeddings.bias" in linear
    assert "layer.1.norm2.weight" not in linear
    tensors = _torch_tensors(hf_model)
    pairs = list(mlx_weights(tensors, dtypes, linear))
    assert tensors == {}
    mapped = dict(pairs)
    for name, tensor in state.items():
        key = map_key(name)
        if key is None:
            continue
        dtype = dtypes.matmul if key in linear else dtypes.stream
        assert mapped[key].dtype == dtype, key
        rounded = tensor.to(TORCH_DTYPES[dtype]).float().numpy().reshape(mapped[key].shape)
        assert np.array_equal(np.array(mapped[key].astype(mx.float32)), rounded), key
    model.load_weights(list(mlx_weights(_torch_tensors(hf_model), dtypes, linear)), strict=True)


def test_mlx_dtype_modes_are_known() -> None:
    """T5: the five modes in the order of THRESHOLDS, each a pair of dtypes."""
    assert list(MLX_DTYPE_MODES) == list(THRESHOLDS)
    names = {mx.float32: "fp32", mx.float16: "fp16", mx.bfloat16: "bf16"}
    for dtype_mode, dtypes in MLX_DTYPE_MODES.items():
        assert names[dtypes.matmul] == DTYPE_OF_MODE[dtype_mode]
        # The bytes of each weight of the run rule and of the needs (F20 of verify.html).
        assert bytes_per_weight("mlx", dtype_mode) == dtypes.matmul.size
        if dtype_mode.endswith("-autocast"):
            assert dtypes.stream == mx.float32
        else:
            assert dtypes.stream == dtypes.matmul


def test_mlx_config_from_hf() -> None:
    """T6: the MLX config of ViT-S/16, of a gated config and of ViT-7B/16."""
    config = MlxConfig.from_hf(VITS16_CONFIG)
    assert config.dim == 384
    assert config.heads == 6
    assert config.head_dim == 64
    assert config.blocks == 12
    assert config.ff == 1536
    assert config.eps == 1e-5
    assert config.rope_theta == 100.0
    assert config.key_bias is False
    assert config.query_bias and config.value_bias and config.proj_bias and config.mlp_bias
    assert config.patch_size == 16
    assert config.num_register_tokens == 4
    assert config.gated is False
    assert isinstance(Block(config).mlp, MLP)
    gated = MlxConfig.from_hf(VITS16_CONFIG | {"use_gated_mlp": True, "hidden_act": "silu"})
    assert gated.gated is True
    assert isinstance(Block(gated).mlp, SwiGLU)
    with pytest.raises(NotImplementedError):
        Block(dataclasses.replace(gated, hidden_act="gelu"))
    with pytest.raises(NotImplementedError):
        Block(dataclasses.replace(config, hidden_act="silu"))
    giant = MlxConfig.from_hf(VIT7B16_CONFIG)
    assert giant.head_dim == 128
    assert not (giant.query_bias or giant.key_bias or giant.value_bias)
    attention = dict(tree_flatten(Block(giant).attention.parameters()))
    assert sorted(attention) == [
        "k_proj.weight",
        "o_proj.bias",
        "o_proj.weight",
        "q_proj.weight",
        "v_proj.weight",
    ]


def test_load_mlx_backbone_refuses_tf32_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLX_ENABLE_TF32", "1")
    monkeypatch.setattr("dinov3_mlx.mlx_weights.parameter_count", fail_if_called("parameter_count"))
    monkeypatch.setattr("dinov3_mlx.mlx_weights.weights_dir", fail_if_called("weights_dir"))
    monkeypatch.setattr("dinov3_mlx.memory.available_memory_bytes", fail_if_called("memory check"))
    with pytest.raises(TF32ModeError):
        load_mlx_backbone("ViT-S/16", "fp32", tf32=False)


def test_load_mlx_backbone_refuses_7b_fp32(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """T7: the run rule stops ViT-7B/16 in fp32 before any file read and before the memory check.
    The four fp16 and bf16 dtype modes pass the run rule and stop at the memory check."""
    # The lines of a monkeypatched memory check must not reach memory-checks.log of the session.
    monkeypatch.setattr(logging.getLogger("dinov3_mlx.memory"), "handlers", [])
    monkeypatch.setattr("dinov3_mlx.mlx_weights.parameter_count", fail_if_called("parameter_count"))
    monkeypatch.setattr("dinov3_mlx.mlx_weights.weights_dir", fail_if_called("weights_dir"))
    monkeypatch.setattr("dinov3_mlx.memory.available_memory_bytes", fail_if_called("memory check"))
    monkeypatch.setattr("dinov3_mlx.mlx_weights.tf32_enabled", fail_if_called("tf32_enabled"))
    with pytest.raises(RunRuleError, match="ViT-7B/16"):
        load_mlx_backbone("ViT-7B/16", "fp32")
    monkeypatch.setattr(
        "dinov3_mlx.mlx_weights.parameter_count", lambda model_size: VIT7B16_PARAMETER_COUNT
    )
    monkeypatch.setattr("dinov3_mlx.memory.available_memory_bytes", lambda: 0)
    caplog.set_level(logging.INFO, logger="dinov3_mlx.memory")
    for dtype_mode in HALF_MODES:
        caplog.clear()
        with pytest.raises(InsufficientMemoryError):
            load_mlx_backbone("ViT-7B/16", dtype_mode)
        messages = [r.getMessage() for r in caplog.records if r.name == "dinov3_mlx.memory"]
        assert len(messages) == 1
        assert f"ViT-7B/16 mlx {dtype_mode}" in messages[0]
        assert "needs 20.1 GB" in messages[0]


def test_required_mlx_load_bytes_by_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """T8: the need counts the bytes of the matmul dtype."""
    counts = {"ViT-S/16": VITS16_PARAMETER_COUNT, "ViT-7B/16": VIT7B16_PARAMETER_COUNT}
    monkeypatch.setattr("dinov3_mlx.mlx_weights.parameter_count", counts.__getitem__)
    assert required_mlx_load_bytes("ViT-S/16", "fp32") == 4_103_663_411
    for dtype_mode in HALF_MODES:
        assert required_mlx_load_bytes("ViT-S/16", dtype_mode) == 4_051_831_706
        assert required_mlx_load_bytes("ViT-7B/16", dtype_mode) == 20_118_484_173


def test_load_mlx_backbone_checks_memory_first(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """T9: one memory-check line for fp32 and for bf16-autocast."""
    # The lines of a monkeypatched memory check must not reach memory-checks.log of the session.
    monkeypatch.setattr(logging.getLogger("dinov3_mlx.memory"), "handlers", [])
    monkeypatch.setattr(
        "dinov3_mlx.mlx_weights.parameter_count", lambda model_size: VITS16_PARAMETER_COUNT
    )
    monkeypatch.setattr("dinov3_mlx.mlx_weights.weights_dir", fail_if_called("weights_dir"))
    monkeypatch.setattr("dinov3_mlx.memory.available_memory_bytes", lambda: 0)
    caplog.set_level(logging.INFO, logger="dinov3_mlx.memory")
    for dtype_mode in ("fp32", "bf16-autocast"):
        caplog.clear()
        with pytest.raises(InsufficientMemoryError):
            load_mlx_backbone("ViT-S/16", dtype_mode)
        messages = [r.getMessage() for r in caplog.records if r.name == "dinov3_mlx.memory"]
        assert len(messages) == 1
        assert f"ViT-S/16 mlx {dtype_mode}" in messages[0]
        assert "needs 4.1 GB" in messages[0]


def test_load_mlx_backbone_sets_memory_limit_to_need(monkeypatch: pytest.MonkeyPatch) -> None:
    """T10: the MLX memory limit is the need of the load, in fp32 and in bf16-cast."""
    # The lines of a monkeypatched memory check must not reach memory-checks.log of the session.
    monkeypatch.setattr(logging.getLogger("dinov3_mlx.memory"), "handlers", [])

    class Sentinel(Exception):
        pass

    limits: list[int] = []

    def recorder(limit: int) -> int:
        limits.append(limit)
        raise Sentinel

    monkeypatch.setattr(
        "dinov3_mlx.mlx_weights.parameter_count", lambda model_size: VITS16_PARAMETER_COUNT
    )
    monkeypatch.setattr("dinov3_mlx.mlx_weights.weights_dir", fail_if_called("weights_dir"))
    monkeypatch.setattr("dinov3_mlx.memory.available_memory_bytes", lambda: 10**12)
    monkeypatch.setattr(mx, "set_memory_limit", recorder)
    for dtype_mode in ("fp32", "bf16-cast"):
        with pytest.raises(Sentinel):
            load_mlx_backbone("ViT-S/16", dtype_mode)
    assert limits == [
        required_mlx_load_bytes("ViT-S/16", "fp32"),
        required_mlx_load_bytes("ViT-S/16", "bf16-cast"),
    ]
    assert limits == [4_103_663_411, 4_051_831_706]


def test_load_mlx_backbone_reads_two_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, report: Callable[..., None]
) -> None:
    """T34: the load of two files and an index gives the parameters of the load of one file, with
    a load peak within the formula of Table 5 and an empty cache.

    The load peak is also at least the bytes of the mapped tensors in bf16, without the mask
    token: a load that does not evaluate the tensors stays below this bound (F1 of
    specs/05-all-sizes/verify.html).
    """
    hf_model = small_hf_model(seed=3, name="vit7b16-shape")
    save_small_model(hf_model, tmp_path / "one", files=1)
    save_small_model(hf_model, tmp_path / "two", files=2)
    assert len(list((tmp_path / "two").glob("*.safetensors"))) == 2
    folders = {"one": tmp_path / "one", "two": tmp_path / "two"}
    # The lines of this small model must not reach memory-checks.log of the test session.
    monkeypatch.setattr(logging.getLogger("dinov3_mlx.memory"), "handlers", [])
    session_limit = mx.get_memory_limit()
    models: dict[str, Any] = {}
    try:
        for name, folder in folders.items():
            monkeypatch.setattr("dinov3_mlx.hub.weights_dir", lambda model_size, f=folder: f)
            monkeypatch.setattr(
                "dinov3_mlx.mlx_weights.weights_dir", lambda model_size, f=folder: f
            )
            try:
                models[name] = load_mlx_backbone("ViT-7B/16", "bf16-cast")
            except InsufficientMemoryError as error:
                pytest.skip(str(error))
    finally:
        mx.set_memory_limit(session_limit)
    one = dict(tree_flatten(models["one"].parameters()))
    two = dict(tree_flatten(models["two"].parameters()))
    assert sorted(one) == sorted(two)
    assert len(two) == sum(1 for name in hf_model.state_dict() if map_key(name) is not None)
    for key, value in one.items():
        assert value.dtype == two[key].dtype, key
        assert mx.array_equal(value, two[key]).item(), key
    count = sum(t.numel() for t in hf_model.state_dict().values())
    largest = max(4 * t.numel() for t in hf_model.state_dict().values())
    target = round(1.2 * count * 2) + largest
    state = hf_model.state_dict()
    mapped = sum(state[name].numel() for name in state if map_key(name) is not None)
    lower_bound = 2 * mapped  # bf16-cast: each mapped tensor in bf16
    load_report = models["two"].load_report
    report(
        "test_load_mlx_backbone_reads_two_files",
        peak_bytes=load_report.peak_bytes,
        cache_bytes=load_report.cache_bytes,
        lower_bound=lower_bound,
        target=target,
    )
    assert lower_bound <= load_report.peak_bytes <= target
    assert load_report.cache_bytes == 0


def _header_shapes(folder: Path) -> tuple[int, dict[str, tuple[int, ...]]]:
    files = sorted(folder.glob("*.safetensors"))
    shapes = {}
    for file in files:
        with safe_open(file, "np") as header:
            for name in header.keys():  # noqa: SIM118
                shapes[name] = tuple(header.get_slice(name).get_shape())
    return len(files), shapes


@pytest.mark.parametrize("model_size", list(MODEL_IDS))
def test_key_map_covers_file(model_size: str, report: Callable[..., None]) -> None:
    """H1: the headers of all files of one model size against the lazy parameter tree of the
    backbone in bf16-autocast. mx.load is lazy and the test evaluates no array."""
    try:
        folder = weights_dir(model_size)
    except WeightsNotFoundError as error:
        pytest.skip(str(error))
    expected = KEY_MAPS[model_size]
    active_before = mx.get_active_memory()
    files, shapes = _header_shapes(folder)
    dropped, unknown = [], []
    for name in shapes:
        try:
            if map_key(name) is None:
                dropped.append(name)
        except UnknownWeightError:
            unknown.append(name)
    config = MlxConfig.from_hf(json.loads((folder / "config.json").read_text()))
    model = MlxBackbone(config, "bf16-autocast")
    linear = linear_parameter_names(model)
    tensors: dict[str, mx.array] = {}
    for file in sorted(folder.glob("*.safetensors")):
        tensors |= mx.load(str(file))
    mapped = dict(mlx_weights(tensors, MLX_DTYPE_MODES["bf16-autocast"], linear))
    params = dict(tree_flatten(model.parameters()))
    missing = sorted(params.keys() - mapped.keys())
    extra = sorted(mapped.keys() - params.keys())
    differences = [
        key for key in params.keys() & mapped.keys() if params[key].shape != mapped[key].shape
    ]
    wrong_dtype = [
        key
        for key, array in mapped.items()
        if array.dtype != (mx.bfloat16 if key in linear else mx.float32)
    ]
    growth = mx.get_active_memory() - active_before
    report(
        "test_key_map_covers_file",
        model_size=model_size,
        tensors=len(shapes),
        parameters=len(params),
        linear=len(linear),
        active_memory_growth=growth,
    )
    assert files == expected.files
    assert len(shapes) == expected.tensors
    assert len(params) == expected.parameters
    assert len(linear) == expected.linear
    assert len(params) - len(linear) == expected.stream
    assert config.gated is expected.gated
    assert config.query_bias is config.value_bias is expected.qv_bias
    assert config.key_bias is False
    assert dropped == ["embeddings.mask_token"]
    assert unknown == []
    assert missing == []
    assert extra == []
    assert differences == []
    assert wrong_dtype == []
    assert mapped[PATCH_WEIGHT].shape == (config.dim, 3 * 16 * 16)
    assert growth < 16 * 10**6


@pytest.mark.parametrize(
    ("model_size", "dtype_mode"),
    [pytest.param("ViT-S/16", "bf16-cast", id="mlx-ViT-S/16-bf16-cast")],
)
def test_mlx_load_peak(
    model_size: str,
    dtype_mode: str,
    backbone: Callable[[str, str, str], MlxBackbone],
    report: Callable[..., None],
) -> None:
    """H2: the load of one tensor at a time keeps the load peak of ViT-S/16 in bf16-cast within
    1.2 x 43.2 MB + 2.36 MB, and the cache is empty after the load. The load peak is at least the
    bytes of the mapped tensors in bf16 (F1 of specs/05-all-sizes/verify.html)."""
    model = backbone(model_size, "mlx", dtype_mode)
    load_report = model.load_report
    target = load_peak_target(model_size, dtype_mode)
    lower_bound = load_peak_lower_bound(model_size, model)
    report(
        "test_mlx_load_peak",
        model_size=model_size,
        dtype_mode=dtype_mode,
        need_bytes=load_report.need_bytes,
        peak_bytes=load_report.peak_bytes,
        cache_bytes=load_report.cache_bytes,
        footprint_bytes=load_report.footprint_bytes,
        lower_bound=lower_bound,
        target=target,
    )
    assert round(target / 1e5) == 542  # 54.2 MB (Table 5)
    assert lower_bound == 2 * (VITS16_PARAMETER_COUNT - 384)  # without the mask token
    assert lower_bound <= load_report.peak_bytes <= target
    assert load_report.cache_bytes == 0
    assert load_report.need_bytes == 4_051_831_706


@pytest.fixture
def lone_slot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Callable[[str, str, str], Any]]:
    """Return the loader of a new backbone slot with its own memory log in tmp_path. The slot of
    the test session keeps its model and its memory-checks.log."""
    import conftest

    monkeypatch.setattr(conftest, "MEMORY_LOG", tmp_path / "memory-checks.log")
    slot = conftest._BackboneSlot()
    yield slot.get
    slot.free()
    logging.getLogger("dinov3_mlx.memory").removeHandler(slot.handler)
    slot.handler.close()


@pytest.mark.parametrize(
    ("peak_bytes", "cache_bytes", "fails"),
    [
        pytest.param(1_000, 0, False, id="at-the-target"),
        pytest.param(100, 0, False, id="at-the-lower-bound"),
        pytest.param(1_001, 0, True, id="above-the-target"),
        pytest.param(99, 0, True, id="below-the-lower-bound"),
        pytest.param(500, 1, True, id="cache-above-0"),
    ],
)
def test_backbone_slot_checks_the_load_report(
    peak_bytes: int,
    cache_bytes: int,
    fails: bool,
    lone_slot: Callable[[str, str, str], Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After each mlx load the backbone slot fails the test if the load peak is above its target
    or below the bytes of the mapped tensors, or if the cache is above 0 bytes (F1 and F5 of
    specs/05-all-sizes/verify.html). A fake load gives the load report, so the test reads no
    weights."""
    load_report = LoadReport(
        need_bytes=1, peak_bytes=peak_bytes, cache_bytes=cache_bytes, footprint_bytes=1
    )
    model = SimpleNamespace(load_report=load_report)
    monkeypatch.setattr(
        "dinov3_mlx.mlx_weights.load_mlx_backbone", lambda model_size, dtype_mode: model
    )
    monkeypatch.setattr(
        parity_checks, "load_peak_target", lambda model_size, dtype_mode, bits=None: 1_000
    )
    monkeypatch.setattr(parity_checks, "load_peak_lower_bound", lambda model_size, model: 100)
    if not fails:
        assert lone_slot("ViT-S/16", "mlx", "bf16-cast") is model
        return
    with pytest.raises(pytest.fail.Exception, match="load of ViT-S/16 mlx bf16-cast: load peak"):
        lone_slot("ViT-S/16", "mlx", "bf16-cast")


def test_load_peak_target_of_an_autocast_pair(monkeypatch: pytest.MonkeyPatch) -> None:
    """The target counts the bytes of the matmul dtype, also in an autocast mode with the stream
    dtype fp32: 1.2 x 2 bytes x 6,716,035,072 + 134,217,728 bytes for ViT-7B/16 (Table 5 of the
    plan; F5 of specs/05-all-sizes/verify.html). The test gives the parameter count and the
    largest tensor, so it reads no header."""
    monkeypatch.setattr(
        parity_checks, "parameter_count", lambda model_size: VIT7B16_PARAMETER_COUNT
    )
    monkeypatch.setattr(parity_checks, "largest_tensor_bytes", lambda model_size: 134_217_728)
    for dtype_mode in ("fp16-autocast", "bf16-autocast"):
        assert load_peak_target("ViT-7B/16", dtype_mode) == 16_252_701_901


def test_backbone_slot_holds_the_compiled_forward(monkeypatch: pytest.MonkeyPatch) -> None:
    """T41 (R36, D4): the slot gives the same compiled forward pass twice for one load and a new
    one after a new load; after free() the function is gone; a model outside the slot raises
    ValueError."""
    import gc
    import weakref

    import conftest
    from small_model import small_mlx_model

    hf = small_hf_model(seed=0)
    monkeypatch.setattr(
        "dinov3_mlx.mlx_weights.load_mlx_backbone",
        lambda model_size, dtype_mode, **kwargs: small_mlx_model(hf, dtype_mode),
    )
    monkeypatch.setattr(conftest._BackboneSlot, "check_load_report", lambda *args: None)
    slot = conftest._BackboneSlot()
    try:
        model = slot.get("ViT-S/16", "mlx", "fp32")
        first = slot.forward_of(model)
        assert slot.forward_of(model) is first
        other = slot.get("ViT-S/16", "mlx", "bf16-cast")
        second = slot.forward_of(other)
        assert second is not first
        with pytest.raises(ValueError, match="slot"):
            slot.forward_of(model)
        reference = weakref.ref(second)
        del first, second
        slot.free()
        gc.collect()
        assert reference() is None
    finally:
        slot.free()
        logging.getLogger("dinov3_mlx.memory").removeHandler(slot.handler)
        slot.handler.close()


# Build step 3 of specs/06-speed/plan.html: the quantized loader (D14 to D18, F1, R50).
VITH16PLUS_PARAMETER_COUNT = 840_592_640
AUTOCAST_MODES = ["fp16-autocast", "bf16-autocast"]


def _quantized_small(name: str, dtype_mode: str, bits: int) -> MlxBackbone:
    """A small backbone with quantized block modules, before any load."""
    from mlx import nn

    from dinov3_mlx.mlx_weights import block_linear_predicate

    model = MlxBackbone(MlxConfig.from_hf(SMALL_CONFIGS[name]), dtype_mode)
    nn.quantize(model, class_predicate=block_linear_predicate(bits))
    return model


def test_required_mlx_load_bytes_quantized(monkeypatch: pytest.MonkeyPatch) -> None:
    """T42 (Table 9 of the plan)."""
    counts = {"ViT-H+/16": VITH16PLUS_PARAMETER_COUNT, "ViT-7B/16": VIT7B16_PARAMETER_COUNT}
    monkeypatch.setattr("dinov3_mlx.mlx_weights.parameter_count", counts.__getitem__)
    for mode in AUTOCAST_MODES:
        assert required_mlx_load_bytes("ViT-H+/16", mode, bits=8) == 5_073_703_936
        assert required_mlx_load_bytes("ViT-H+/16", mode, bits=4) == 4_570_387_456
        assert required_mlx_load_bytes("ViT-7B/16", mode, bits=8) == 12_568_736_973
        assert required_mlx_load_bytes("ViT-7B/16", mode, bits=4) == 8_542_205_133


def test_linear_parameter_names_sees_quantized_linear() -> None:
    """T43 (F1): QuantizedLinear is not a subclass of nn.Linear; its weight, scales, biases and
    bias count as parameters of the matmul dtype."""
    from mlx import nn

    from dinov3_mlx.mlx_model import quantized_weight_names

    model = _quantized_small("vits16plus", "bf16-autocast", 8)
    linear = linear_parameter_names(model)
    quantized = quantized_weight_names(model)
    assert len(quantized) == 14  # 2 blocks x 7 modules
    for name, module in model.named_modules():
        if isinstance(module, nn.QuantizedLinear):
            leaves = {leaf for leaf, _ in tree_flatten(module.parameters())}
            assert {f"{name}.{leaf}" for leaf in leaves} <= linear
            expected = {"weight", "scales", "biases"} | ({"bias"} if "bias" in module else set())
            assert leaves == expected, name
            assert f"{name}.weight" in quantized
    assert "layer.0.attention.k_proj.bias" not in linear  # the small configs have no k_proj bias
    assert "embeddings.patch_embeddings.weight" in linear
    assert "embeddings.patch_embeddings.weight" not in quantized


def test_block_linear_predicate() -> None:
    """T44 (D16)."""
    from mlx import nn

    from dinov3_mlx.mlx_weights import GROUP_SIZE, QUANTIZED_BITS, block_linear_predicate

    assert GROUP_SIZE == 64 and QUANTIZED_BITS == (8, 4)
    predicate = block_linear_predicate(8)
    linear = nn.Linear(64, 64)
    assert predicate("layer.0.attention.q_proj", linear) == {"group_size": 64, "bits": 8}
    assert predicate("embeddings.patch_embeddings", linear) is False
    assert predicate("layer.0.norm1", nn.LayerNorm(64)) is False
    for name, per_block in (("vits16plus", 7), ("vits16", 6)):
        model = _quantized_small(name, "fp16-autocast", 4)
        count = sum(isinstance(m, nn.QuantizedLinear) for _, m in model.named_modules())
        assert count == per_block * 2


def test_placeholder_skeleton_stays_small() -> None:
    """T45 (R50): an evaluation of the placeholder skeleton allocates one value for each
    parameter. The control without placeholders allocates the random fp32 weights."""
    import gc

    from mlx import nn

    from dinov3_mlx.mlx_weights import block_linear_predicate, placeholder_skeleton

    config = MlxConfig.from_hf(
        SMALL_CONFIGS["vit7b16-shape"]
        | {
            "hidden_size": 1024,
            "num_attention_heads": 8,
            "intermediate_size": 2048,
            "num_hidden_layers": 8,
        }
    )
    gc.collect()
    mx.clear_cache()
    before = mx.get_active_memory()
    model = MlxBackbone(config, "bf16-autocast")
    nn.quantize(model, class_predicate=block_linear_predicate(8))
    placeholder_skeleton(model)
    mx.eval(model.parameters())
    growth = mx.get_active_memory() - before
    del model
    gc.collect()
    mx.clear_cache()
    before = mx.get_active_memory()
    control = MlxBackbone(config, "fp32")
    fp32_bytes = 4 * sum(v.size for _, v in tree_flatten(control.parameters()))
    mx.eval(control.parameters())
    control_growth = mx.get_active_memory() - before
    del control
    assert growth <= 1_000_000
    assert control_growth >= 0.9 * fp32_bytes
    assert fp32_bytes > 300_000_000


@pytest.fixture
def small_folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The weights of vits16plus in a temporary folder, as the folder of ViT-S+/16."""
    folder = tmp_path / "vits16plus"
    save_small_model(small_hf_model(seed=2, name="vits16plus"), folder, files=2)
    monkeypatch.setattr("dinov3_mlx.hub.weights_dir", lambda model_size: folder)
    monkeypatch.setattr("dinov3_mlx.mlx_weights.weights_dir", lambda model_size: folder)
    monkeypatch.setattr(
        "dinov3_mlx.mlx_weights.parameter_count", lambda model_size: VITS16_PARAMETER_COUNT
    )
    monkeypatch.setattr(logging.getLogger("dinov3_mlx.memory"), "handlers", [])
    return folder


@pytest.mark.parametrize(
    ("dtype_mode", "bits"),
    [pytest.param(m, b, id=f"{m}-{b}") for m in AUTOCAST_MODES for b in (8, 4)],
)
def test_quantized_loader_equals_load_then_quantize(
    dtype_mode: str, bits: int, small_folder: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T46 (D17, R57): L2 (quantize in the loader, one tensor at a time) equals L1 (load, then
    nn.quantize) bit for bit in every parameter; the strict load passes; the memory check runs."""
    from mlx import nn

    from dinov3_mlx.mlx_weights import block_linear_predicate

    checks: list[int] = []
    original = check_free_memory_of_module()
    monkeypatch.setattr(
        "dinov3_mlx.mlx_weights.check_free_memory",
        lambda need, label: checks.append(need) or original(need, label),
    )
    session_limit = mx.get_memory_limit()
    try:
        l2 = load_mlx_backbone("ViT-S+/16", dtype_mode, bits=bits)
        l1 = load_mlx_backbone("ViT-S+/16", dtype_mode)
    finally:
        mx.set_memory_limit(session_limit)
    nn.quantize(l1, class_predicate=block_linear_predicate(bits))
    one, two = dict(tree_flatten(l1.parameters())), dict(tree_flatten(l2.parameters()))
    assert sorted(one) == sorted(two)
    for key, value in one.items():
        assert value.dtype == two[key].dtype, key
        assert mx.array_equal(value, two[key]).item(), key
    assert len(checks) == 2
    assert l2.load_report.bits == bits and l1.load_report.bits is None


def check_free_memory_of_module() -> Callable[[int, str], int]:
    from dinov3_mlx import mlx_weights

    return mlx_weights.check_free_memory


@pytest.mark.parametrize(
    ("dtype_mode", "bits"),
    [pytest.param(m, b, id=f"{m}-{b}") for m in AUTOCAST_MODES for b in (8, 4)],
)
def test_quantized_parameters_have_the_matmul_dtype(
    dtype_mode: str, bits: int, small_folder: Path
) -> None:
    """T47 (R46): the strict load checks no dtype, so the test does."""
    from mlx import nn

    session_limit = mx.get_memory_limit()
    try:
        model = load_mlx_backbone("ViT-S+/16", dtype_mode, bits=bits)
    finally:
        mx.set_memory_limit(session_limit)
    matmul = MLX_DTYPE_MODES[dtype_mode].matmul
    for name, module in model.named_modules():
        if isinstance(module, nn.QuantizedLinear):
            assert module.weight.dtype == mx.uint32, name
            for leaf in ("scales", "biases", "bias"):
                if leaf in module:
                    assert module[leaf].dtype == matmul, f"{name}.{leaf}"
    assert model.embeddings.patch_embeddings.weight.dtype == matmul


def test_load_peak_target_of_a_quantized_load(
    small_folder: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T48 (F1, Table 9): the target and the lower bound know the quantized bytes."""
    counts = {"ViT-H+/16": VITH16PLUS_PARAMETER_COUNT, "ViT-7B/16": VIT7B16_PARAMETER_COUNT}
    largest = {"ViT-H+/16": 26_214_400, "ViT-7B/16": 134_217_728}
    monkeypatch.setattr(parity_checks, "parameter_count", counts.__getitem__)
    monkeypatch.setattr(parity_checks, "largest_tensor_bytes", largest.__getitem__)
    assert load_peak_target("ViT-H+/16", "bf16-autocast", 8) == 1_113_025_536
    assert load_peak_target("ViT-H+/16", "bf16-autocast", 4) == 609_709_056
    assert load_peak_target("ViT-7B/16", "fp16-autocast", 8) == 8_770_063_565
    assert load_peak_target("ViT-7B/16", "fp16-autocast", 4) == 4_743_531_725

    session_limit = mx.get_memory_limit()
    try:
        model = load_mlx_backbone("ViT-S+/16", "bf16-autocast", bits=4)
    finally:
        mx.set_memory_limit(session_limit)
    _, shapes = _header_shapes(small_folder)
    monkeypatch.setattr(parity_checks, "header_shapes", lambda model_size: shapes)
    # Each mapped tensor in its loaded form: the packed weight, the scales and the offsets of a
    # quantized module; the matmul dtype or the stream dtype for the others.
    expected = sum(v.nbytes for _, v in tree_flatten(model.parameters()))
    assert load_peak_lower_bound("ViT-S+/16", model) == expected
    assert load_peak_lower_bound("ViT-S+/16", model) <= model.load_report.peak_bytes


def test_backbone_slot_checks_a_quantized_load(
    lone_slot: Callable[..., Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """T49 (F1): the slot key has bits; the slot checks the load report of a quantized load with
    the quantized target; the sort key of the session puts the unquantized tests first."""
    import conftest

    loads: list[Any] = []

    def load(model_size: str, dtype_mode: str, **kwargs: Any) -> Any:
        loads.append(kwargs.get("bits"))
        report = LoadReport(
            need_bytes=1, peak_bytes=500, cache_bytes=0, footprint_bytes=1, bits=kwargs.get("bits")
        )
        return SimpleNamespace(load_report=report)

    targets: list[Any] = []
    monkeypatch.setattr("dinov3_mlx.mlx_weights.load_mlx_backbone", load)
    monkeypatch.setattr(
        parity_checks,
        "load_peak_target",
        lambda model_size, dtype_mode, bits=None: targets.append(bits) or 1_000,
    )
    monkeypatch.setattr(parity_checks, "load_peak_lower_bound", lambda model_size, model: 100)
    lone_slot("ViT-S/16", "mlx", "bf16-autocast")
    lone_slot("ViT-S/16", "mlx", "bf16-autocast", 8)
    lone_slot("ViT-S/16", "mlx", "bf16-autocast", 8)
    assert loads == [None, 8] and targets == [None, 8]
    monkeypatch.setattr(parity_checks, "load_peak_lower_bound", lambda model_size, model: 600)
    with pytest.raises(pytest.fail.Exception, match="bf16-autocast 4-bit"):
        lone_slot("ViT-S/16", "mlx", "bf16-autocast", 4)

    def item(name: str, **params: Any) -> SimpleNamespace:
        return SimpleNamespace(
            name=name,
            callspec=SimpleNamespace(params=params),
            get_closest_marker=lambda marker: object(),
        )

    items = [
        item("q8", model_size="ViT-S/16", dtype_mode="bf16-autocast", bits=8),
        item("plain", model_size="ViT-S/16", dtype_mode="bf16-autocast"),
        item("q4", model_size="ViT-S/16", dtype_mode="bf16-autocast", bits=4),
    ]
    conftest.pytest_collection_modifyitems(items)
    assert [i.name for i in items] == ["plain", "q4", "q8"]


@pytest.mark.parametrize(
    ("dtype_mode", "bits"),
    [("fp32", 8), ("fp16-cast", 8), ("bf16-cast", 4), ("bf16-autocast", 6)],
)
def test_load_mlx_backbone_refuses_bits_outside_the_autocast_modes(
    dtype_mode: str, bits: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T50 (D15): bits in fp32 and in the cast modes, and bits=6, raise ValueError before any
    file read."""
    monkeypatch.setattr("dinov3_mlx.mlx_weights.weights_dir", fail_if_called("weights_dir"))
    monkeypatch.setattr("dinov3_mlx.mlx_weights.parameter_count", fail_if_called("parameter_count"))
    with pytest.raises(ValueError, match="bits"):
        load_mlx_backbone("ViT-S/16", dtype_mode, bits=bits)
