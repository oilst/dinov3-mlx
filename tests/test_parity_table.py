"""The parity table of scripts/parity_table.py for the backends torch-mps and mlx."""

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
    assert "| stored-block | block-3 | 1.00e-07 | 0.0001 | 1e-07 |" in text


def test_results_path_of_each_backend() -> None:
    assert results_path("mlx").name == "parity-mlx.md"
    assert results_path("torch-mps").name == "parity-torch-mps.md"
