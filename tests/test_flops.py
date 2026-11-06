import json

import pytest

from dinov3_apple.flops import (
    MODEL_SHAPES,
    PART_NAMES,
    image_flops,
    part_bytes,
    part_flops,
    tokens,
)
from dinov3_apple.hub import MODEL_IDS, WeightsNotFoundError, weights_dir

# Table 2 of specs/03-baseline/explore.html (corrected in the build): GFLOP and attention share (%)
# at 224, 448 and 512 px.
EXPLORE_TABLE_2: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] = {
    "ViT-S/16": ((9.4, 45.4, 63.8), (7.93, 25.25, 30.58)),
    "ViT-S+/16": ((12.2, 56.6, 78.4), (6.08, 20.27, 24.90)),
    "ViT-B/16": ((35.9, 157.9, 215.0), (4.15, 14.53, 18.15)),
    "ViT-L/16": ((125.7, 539.0, 727.2), (3.16, 11.35, 14.31)),
    "ViT-H+/16": ((344.2, 1427.3, 1901.9), (1.92, 7.15, 9.12)),
    "ViT-7B/16": ((2725.5, 11002.7, 14511.4), (0.97, 3.71, 4.78)),
}
SIDES = (224, 448, 512)


def test_tokens_per_side() -> None:
    assert [tokens(side) for side in SIDES] == [201, 789, 1029]


@pytest.mark.parametrize("model_size", MODEL_IDS)
def test_model_shapes_match_config(model_size: str) -> None:
    try:
        config = json.loads((weights_dir(model_size) / "config.json").read_text())
    except WeightsNotFoundError as error:
        pytest.skip(str(error))
    shape = MODEL_SHAPES[model_size]
    assert shape.dim == config["hidden_size"]
    assert shape.blocks == config["num_hidden_layers"]
    assert shape.heads == config["num_attention_heads"]
    assert shape.ff == config["intermediate_size"]
    assert shape.gated == config["use_gated_mlp"]


def test_part_flops_formulas() -> None:
    shape = MODEL_SHAPES["ViT-S/16"]
    flops = part_flops(shape, batch_size=2, num_tokens=201)
    assert list(flops) == PART_NAMES
    b, t, d, f = 2, 201, 384, 1536
    assert flops["qkv_proj"] == 3 * 2 * b * t * d * d
    assert flops["sdpa"] == 4 * b * t * t * d
    assert flops["o_proj"] == 2 * b * t * d * d
    assert flops["ff_in"] == 2 * b * t * d * f
    assert flops["ff_out"] == 2 * b * t * f * d
    assert flops["norm1"] == flops["rope"] == flops["activation"] == 0
    gated = part_flops(MODEL_SHAPES["ViT-S+/16"], batch_size=2, num_tokens=201)
    assert gated["ff_in"] == 2 * flops["ff_in"]


def test_part_bytes_scale_with_bytes_per_value() -> None:
    shape = MODEL_SHAPES["ViT-B/16"]
    fp32 = part_bytes(shape, batch_size=1, num_tokens=201, bytes_per_value=4)
    fp16 = part_bytes(shape, batch_size=1, num_tokens=201, bytes_per_value=2)
    assert list(fp32) == PART_NAMES
    assert all(fp32[name] == 2 * fp16[name] for name in PART_NAMES)
    assert fp32["norm1"] == 2 * 201 * 768 * 4


@pytest.mark.parametrize("model_size", MODEL_IDS)
def test_part_flops_match_explore_table_2(model_size: str) -> None:
    gflops, shares = EXPLORE_TABLE_2[model_size]
    for side, expected_gflop, expected_share in zip(SIDES, gflops, shares, strict=True):
        total, attention_share = image_flops(MODEL_SHAPES[model_size], side)
        assert total / 1e9 == pytest.approx(expected_gflop, rel=0.005), side
        assert 100 * attention_share == pytest.approx(expected_share, rel=0.005), side
