"""The parity table of scripts/parity_table.py for the backends torch-mps and mlx (T31 of
specs/05-all-sizes/plan.html)."""

from parity_table import results_path, write_table


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
    sizes are sorted; the non-finite configurations of both backends are compared."""
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
