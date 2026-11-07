"""Test T32 of specs/05-all-sizes/plan.html: the table of scripts/token_range_table.py."""

import math

import numpy as np
from token_range_table import stored_range, write_table


def _line(model_size: str, dtype_mode: str, config: str, tokens: list[float]) -> dict:
    blocks = len(tokens) - 1
    finite = [math.isfinite(value) for value in tokens]
    return {
        "kind": "token-range",
        "backend": "mlx",
        "model_size": model_size,
        "dtype_mode": dtype_mode,
        "config": config,
        "status": "measured" if all(finite) else "non-finite",
        "reason": "",
        "parity": None,
        "values": {
            "tokens": tokens,
            "tokens_where": [0, 2, 3, 7][: blocks + 1],
            "tokens_finite": finite,
            "attention": [10.0 * (i + 1) for i in range(blocks)],
            "ff_hidden": [20.0 * (i + 1) for i in range(blocks)],
            "ff_out": [30.0 * (i + 1) for i in range(blocks)],
        },
        "mlx": "9.9.9",
    }


def test_token_range_table() -> None:
    """T32: the largest value, its block and token, the share of 65,504 and the first non-finite
    block of each row."""
    lines = [
        _line("ViT-L/16", "fp32", "forest/224x224", [12.5, 900.0, 98_256.0, 4_000.0]),
        _line("ViT-L/16", "fp16-cast", "forest/224x224", [12.5, 900.0, float("inf"), float("nan")]),
    ]
    text = write_table(lines)
    assert text.startswith("# Token range of mlx")
    assert "mlx 9.9.9" in text
    # After block 2, register token R3 (token 3 of the sequence), 150 % of 65,504; all finite.
    assert (
        "| ViT-L/16 | fp32 | forest/224x224 | 98,256.0 | 2 | R3 | 150.0 % | – | 30.0 | 60.0 | 90.0 |"
    ) in text
    # The largest finite value is 900 after block 1 (token R2); block 2 is the first non-finite.
    assert (
        "| ViT-L/16 | fp16-cast | forest/224x224 | 900.0 | 1 | R2 | 1.4 % | 2 | 30.0 | 60.0 | 90.0 |"
    ) in text
    # The summary row of each model size and dtype mode.
    assert "| ViT-L/16 | fp32 | 98,256.0 | forest/224x224 | 2 | R3 | 150.0 % | 0 of 1 | – |" in text
    assert "| ViT-L/16 | fp16-cast | 900.0 | forest/224x224 | 1 | R2 | 1.4 % | 1 of 1 | 2 |" in text


def test_token_range_table_from_the_stored_hidden_states() -> None:
    """The fp32 token range of the hidden-states configuration from the reference files: the
    largest absolute value, its block and token, and the value after the last block. ViT-7B/16
    comes from its streamed reference file (Table 7 of specs/05-all-sizes/plan.html)."""
    hidden = np.zeros((3, 6, 4), np.float32)
    hidden[1, 2, 3] = -70_000.0  # after block 1, register token R2
    hidden[2, 0, 0] = 500.0  # after block 2, the CLS token
    assert stored_range(hidden) == {
        "largest": 70_000.0,
        "block": 1,
        "token": "R2",
        "last": 500.0,
        "blocks": 2,
    }
    small = np.ones((3, 6, 4), np.float32)
    text = write_table(
        [_line("ViT-L/16", "fp32", "forest/224x224", [1.0, 2.0])],
        stored={"ViT-7B/16": (hidden, True), "ViT-S/16": (small, False)},
    )
    rows = [row for row in text.splitlines() if row.startswith("| ViT-7B/16 | fp32 |")]
    assert rows == ["| ViT-7B/16 | fp32 | streamed | 2 | 70,000.0 | 1 | R2 | 106.9 % | 500.0 |"]
    assert "| ViT-S/16 | fp32 | full model | 2 | 1.0 | 0 | CLS | 0.0 % | 1.0 |" in text
    assert text.index("| ViT-S/16 | fp32 | full model") < text.index("| ViT-7B/16 | fp32 |")
    assert "## 3 fp32 from the stored hidden states" in text
    assert "## 3" not in write_table([_line("ViT-L/16", "fp32", "forest/224x224", [1.0, 2.0])])
