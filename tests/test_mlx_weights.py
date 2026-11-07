"""Tests T1 to T7b of specs/04-mlx-port/plan.html: the key map, the MLX config and the load."""

import dataclasses
import logging
from collections.abc import Callable

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten
from safetensors import safe_open
from small_model import small_hf_model

from dinov3_apple.bench import DTYPE_OF_MODE
from dinov3_apple.hub import WeightsNotFoundError, weights_dir
from dinov3_apple.memory import InsufficientMemoryError
from dinov3_apple.mlx_model import (
    MLX_DTYPE_MODES,
    Block,
    MlxBackbone,
    MlxConfig,
    TF32ModeError,
)
from dinov3_apple.mlx_weights import (
    PATCH_WEIGHT,
    UnknownWeightError,
    load_mlx_backbone,
    map_key,
    mlx_weights,
    required_mlx_load_bytes,
)
from dinov3_apple.parity import THRESHOLDS

pytestmark = pytest.mark.mlx

VITS16_PARAMETER_COUNT = 21_596_544

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
    tensors = {name: mx.array(t.numpy()) for name, t in hf_model.state_dict().items()}
    tensors["layer.2.norm1.weight"] = tensors["model.layer.1.norm1.weight"]  # blocks 0 and 1 only
    model = MlxBackbone(MlxConfig.from_hf(hf_model.config.to_dict()))
    with pytest.raises(ValueError, match=r"layer\.2\.norm1\.weight"):
        model.load_weights(mlx_weights(tensors, mx.float32), strict=True)


def test_mlx_weights_load_small_model() -> None:
    hf_model = small_hf_model()
    state = hf_model.state_dict()
    tensors = {name: mx.array(t.numpy()) for name, t in state.items()}
    model = MlxBackbone(MlxConfig.from_hf(hf_model.config.to_dict()))
    model.load_weights(mlx_weights(tensors, mx.float32), strict=True)
    params = dict(tree_flatten(model.parameters()))
    assert len(params) == len(state) - 1
    for name, tensor in state.items():
        key = map_key(name)
        if key is None:
            continue
        expected = tensor.numpy().reshape(params[key].shape)
        assert params[key].dtype == mx.float32, name
        assert np.array_equal(np.array(params[key]), expected), name


def test_key_map_covers_vits16_file() -> None:
    try:
        folder = weights_dir("ViT-S/16")
    except WeightsNotFoundError as error:
        pytest.skip(str(error))
    file = folder / "model.safetensors"
    with safe_open(file, "np") as header:
        shapes = {name: tuple(header.get_slice(name).get_shape()) for name in header.keys()}  # noqa: SIM118
    dropped, unknown = [], []
    for name in shapes:
        try:
            if map_key(name) is None:
                dropped.append(name)
        except UnknownWeightError:
            unknown.append(name)
    # mx.load is lazy, and the backbone tree is lazy: the test computes no array.
    mapped = dict(mlx_weights(mx.load(str(file)), mx.float32))
    config = MlxConfig.from_hf(VITS16_CONFIG)
    params = dict(tree_flatten(MlxBackbone(config).parameters()))
    missing = sorted(params.keys() - mapped.keys())
    extra = sorted(mapped.keys() - params.keys())
    differences = [
        key for key in params.keys() & mapped.keys() if params[key].shape != mapped[key].shape
    ]
    assert len(shapes) == 211
    assert len(params) == 210
    assert dropped == ["embeddings.mask_token"]
    assert unknown == []
    assert missing == []
    assert extra == []
    assert differences == []
    assert mapped[PATCH_WEIGHT].shape == (384, 768)


def test_mlx_dtype_modes_are_known() -> None:
    assert list(MLX_DTYPE_MODES) == ["fp32"]
    for dtype_mode in MLX_DTYPE_MODES:
        assert dtype_mode in THRESHOLDS
        assert dtype_mode in DTYPE_OF_MODE


def test_mlx_config_from_hf() -> None:
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
    gated = MlxConfig.from_hf(VITS16_CONFIG | {"use_gated_mlp": True, "hidden_act": "silu"})
    assert gated.gated is True
    with pytest.raises(NotImplementedError):
        Block(gated)


def _fail(name: str) -> Callable[..., None]:
    def fail(*args: object, **kwargs: object) -> None:
        raise AssertionError(f"{name} was called")

    return fail


def test_load_mlx_backbone_refuses_tf32_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLX_ENABLE_TF32", "1")
    monkeypatch.setattr("dinov3_apple.mlx_weights.parameter_count", _fail("parameter_count"))
    monkeypatch.setattr("dinov3_apple.mlx_weights.weights_dir", _fail("weights_dir"))
    monkeypatch.setattr("dinov3_apple.memory.available_memory_bytes", _fail("memory check"))
    with pytest.raises(TF32ModeError):
        load_mlx_backbone("ViT-S/16", "fp32", tf32=False)


def test_load_mlx_backbone_checks_memory_first(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(
        "dinov3_apple.mlx_weights.parameter_count", lambda model_size: VITS16_PARAMETER_COUNT
    )
    monkeypatch.setattr("dinov3_apple.mlx_weights.weights_dir", _fail("weights_dir"))
    monkeypatch.setattr("dinov3_apple.memory.available_memory_bytes", lambda: 0)
    caplog.set_level(logging.INFO, logger="dinov3_apple.memory")
    with pytest.raises(InsufficientMemoryError):
        load_mlx_backbone("ViT-S/16", "fp32")
    messages = [r.getMessage() for r in caplog.records if r.name == "dinov3_apple.memory"]
    assert len(messages) == 1
    assert "ViT-S/16 mlx fp32" in messages[0]
    assert "needs 4.1 GB" in messages[0]


def test_load_mlx_backbone_sets_memory_limit_to_need(monkeypatch: pytest.MonkeyPatch) -> None:
    class Sentinel(Exception):
        pass

    limits: list[int] = []

    def recorder(limit: int) -> int:
        limits.append(limit)
        raise Sentinel

    monkeypatch.setattr(
        "dinov3_apple.mlx_weights.parameter_count", lambda model_size: VITS16_PARAMETER_COUNT
    )
    monkeypatch.setattr("dinov3_apple.mlx_weights.weights_dir", _fail("weights_dir"))
    monkeypatch.setattr("dinov3_apple.memory.available_memory_bytes", lambda: 10**12)
    monkeypatch.setattr(mx, "set_memory_limit", recorder)
    with pytest.raises(Sentinel):
        load_mlx_backbone("ViT-S/16", "fp32")
    assert limits == [required_mlx_load_bytes("ViT-S/16", "fp32")]
    assert limits == [4_103_663_411]
