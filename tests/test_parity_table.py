"""The parity table of scripts/parity_table.py for the backends torch-mps and mlx (T31 of
specs/05-all-sizes/plan.html)."""

import json
from collections.abc import Callable
from pathlib import Path

import pytest
from parity_table import results_path, table_order, write_table


def _line(backend: str, package: str, kind: str = "parity", **extra: object) -> dict:
    parity = {"frame": 0.99999, "patch_mean": 0.99998, "patch_p01": 0.99995, "patch_min": 0.9999}
    return {
        "kind": kind,
        "backend": backend,
        "model_size": "ViT-S/16",
        "dtype_mode": "fp32",
        "config": "forest/224x304",
        "status": "measured",
        "reason": "",
        "parity": parity,
        package: "9.9.9",
    } | extra


def test_write_table_names_the_backend_and_its_version() -> None:
    text = write_table([_line("mlx", "mlx")], "mlx")
    assert text.startswith("# Parity of mlx")
    assert "mlx 9.9.9" in text
    assert "Table 1. Parity of mlx against the reference output." in text
    assert "| ViT-S/16 | fp32 | 1 | 0 | 0.999990 |" in text
    assert "on mlx in the same dtype mode" in text
    assert "torch 9.9.9" not in text


def test_write_table_keeps_the_torch_mps_text() -> None:
    text = write_table([_line("torch-mps", "torch")], "torch-mps")
    assert text.startswith("# Parity of torch-mps")
    assert "(torch 9.9.9)" in text
    assert "on torch-mps in the same dtype mode" in text


def test_write_table_lists_the_stored_hidden_state_checks() -> None:
    values = {"max_abs": 1e-4, "max_rel": 1e-7, "min_cosine": 0.9999999}
    line = _line("mlx", "mlx", kind="stored-block", config="block-3", parity=None, values=values)
    text = write_table([_line("mlx", "mlx"), line], "mlx")
    assert "| ViT-S/16 | fp32 | stored-block | block-3 | 1.00e-07 | 0.0001 | 1e-07 |" in text


def test_results_path_of_each_backend() -> None:
    assert results_path("mlx").name == "parity-mlx.md"
    assert results_path("torch-mps").name == "parity-torch-mps.md"


def test_parity_table_uses_the_expected_failure_list() -> None:
    """T31: the expected failures come from EXPECTED_FP16_OVERFLOW; the stored checks of two model
    sizes are sorted; the non-finite configurations of both backends are compared, the parity
    lines and the batch lines (F9 of specs/05-all-sizes/verify.html)."""
    non_finite = {"status": "non-finite", "parity": None, "dtype_mode": "fp16-cast"}
    lines = [
        _line("mlx", "mlx", model_size="ViT-L/16", **non_finite),
        _line("mlx", "mlx", model_size="ViT-S/16", **non_finite),
    ]
    text = write_table(lines, "mlx")
    assert "| ViT-S/16 | fp16-cast | 0 | 1 | – | – | – | – | fail |" in text
    assert (
        "| ViT-L/16 | fp16-cast | 0 | 1 | – | – | – | – | expected failure (fp16 overflow) |"
        in text
    )

    values = {"max_abs": 1e-4, "max_rel": 1e-7, "min_cosine": 0.9999999}
    stored = [
        _line(
            "mlx", "mlx", kind="stored-final-norm", model_size=size, dtype_mode=mode, values=values
        )
        for size, mode in [
            ("ViT-B/16", "bf16-cast"),
            ("ViT-S+/16", "fp32"),
            ("ViT-B/16", "fp32"),
            ("ViT-S+/16", "fp16-cast"),
        ]
    ]
    rows = [row for row in write_table(stored, "mlx").splitlines() if "stored-final-norm" in row]
    assert [row.split(" | ")[:2] for row in rows] == [
        ["| ViT-S+/16", "fp32"],
        ["| ViT-S+/16", "fp16-cast"],
        ["| ViT-B/16", "fp32"],
        ["| ViT-B/16", "bf16-cast"],
    ]
    assert "the L + 1 hidden states" in write_table(stored, "mlx")

    title = "Non-finite configurations: mlx against torch-mps"
    pair = [("ViT-L/16", "forest/224x224"), ("ViT-H+/16", "iss-kibo-lab/224x400")]
    mlx = [_line("mlx", "mlx", model_size=s, config=c, **non_finite) for s, c in pair]
    torch_mps = [_line("torch-mps", "torch", model_size=s, config=c, **non_finite) for s, c in pair]
    text = write_table(mlx, "mlx", torch_mps)
    assert title in text
    assert (
        "| parity | ViT-H+/16 | fp16-cast | iss-kibo-lab/224x400 | non-finite | non-finite |"
        in text
    )
    assert text.rstrip().endswith("equal")
    extra = _line("mlx", "mlx", model_size="ViT-B/16", config="forest/512x512", **non_finite)
    text = write_table([*mlx, extra], "mlx", torch_mps)
    last = text.rstrip().splitlines()[-1]
    assert last.startswith("differences:")
    assert "ViT-B/16 fp16-cast forest/512x512" in last
    assert title not in write_table(mlx, "mlx")

    batch = {"kind": "batch", "model_size": "ViT-L/16", "config": "batch-8-224x224"}
    mlx_batch = _line("mlx", "mlx", **non_finite | batch)
    torch_batch = _line("torch-mps", "torch", **non_finite | batch)
    text = write_table([*mlx, mlx_batch], "mlx", [*torch_mps, torch_batch])
    assert "| batch | ViT-L/16 | fp16-cast | batch-8-224x224 | non-finite | non-finite |" in text
    assert text.rstrip().endswith("equal")
    text = write_table([*mlx, mlx_batch], "mlx", torch_mps)
    assert "| batch | ViT-L/16 | fp16-cast | batch-8-224x224 | non-finite | – |" in text
    assert text.rstrip().splitlines()[-1] == (
        "differences: ViT-L/16 fp16-cast batch-8-224x224 (batch): mlx only"
    )
    text = write_table(mlx, "mlx", [*torch_mps, torch_batch])
    assert text.rstrip().splitlines()[-1] == (
        "differences: ViT-L/16 fp16-cast batch-8-224x224 (batch): torch-mps only"
    )


def test_table_order_follows_model_ids_and_thresholds() -> None:
    """One sort key of model size and dtype mode for the tables of scripts/parity_table.py and
    scripts/token_range_table.py (F22 of specs/05-all-sizes/verify.html)."""
    assert table_order("ViT-S/16", "fp32") == (0, 0)
    assert table_order("ViT-7B/16", "bf16-autocast") == (5, 4)
    pairs = [("ViT-B/16", "fp32"), ("ViT-S+/16", "bf16-cast"), ("ViT-S+/16", "fp32")]
    assert sorted(pairs, key=lambda pair: table_order(*pair)) == [
        ("ViT-S+/16", "fp32"),
        ("ViT-S+/16", "bf16-cast"),
        ("ViT-B/16", "fp32"),
    ]


def test_parity_table_keeps_both_bits(
    tmp_path: Path, record: Callable[..., None], monkeypatch: pytest.MonkeyPatch
) -> None:
    """T14: the key has bits; record() writes bits; the sections of the kinds compiled-equal,
    quantized and fp16-output; --log and --out write the table from a given log."""
    import conftest
    import parity_table

    unquantized = {"frame": 0.9999, "patch_mean": 0.9999, "patch_p01": 0.999, "patch_min": 0.99}
    quantized = [
        _line(
            "mlx",
            "mlx",
            kind="quantized",
            model_size="ViT-S/16",
            dtype_mode="bf16-autocast",
            bits=bits,
            values={"unquantized": unquantized, "passes_thresholds": bits == 8},
        )
        for bits in (8, 4)
    ]
    log = tmp_path / "mlx.jsonl"
    log.write_text("".join(json.dumps(line) + "\n" for line in quantized))
    assert sorted(line["bits"] for line in parity_table.latest_lines(log)) == [4, 8]
    text = write_table(parity_table.latest_lines(log), "mlx")
    assert "Quantized weights" in text
    assert "| ViT-S/16 | bf16-autocast | 8 | 1 | 1 |" in text
    assert "| ViT-S/16 | bf16-autocast | 4 | 1 | 0 |" in text

    equal = _line("mlx", "mlx", kind="compiled-equal", parity=None, values={"max_abs": 0.0})
    fp16 = _line(
        "mlx",
        "mlx",
        kind="fp16-output",
        dtype_mode="bf16-cast",
        values={"extra_loss": {"patch_mean": 2e-8, "patch_p01": 3e-8}, "frame_equal": True},
    )
    text = write_table([equal, fp16], "mlx")
    assert "Compiled forward pass against the eager forward pass" in text
    assert "| ViT-S/16 | fp32 | 1 | 1 | 0 |" in text
    assert "fp16 patch embeddings" in text
    assert "| ViT-S/16 | bf16-cast | 1 | 0.999980 | 0.999950 | 3e-08 | yes |" in text

    monkeypatch.setattr(conftest, "PARITY_DIR", tmp_path / "parity")
    (tmp_path / "parity").mkdir()
    record(
        "quantized",
        "ViT-S/16",
        "fp16-autocast",
        "forest/224x224",
        None,
        "skipped",
        "x",
        backend="mlx",
        bits=4,
    )
    written = json.loads((tmp_path / "parity" / "mlx.jsonl").read_text())
    assert written["bits"] == 4

    out = tmp_path / "out"
    parity_table.main(["--backend", "mlx", "--log", str(log), "--out", str(out)])
    assert "Quantized weights" in (out / "parity-mlx.md").read_text()
