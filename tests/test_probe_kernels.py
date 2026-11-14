"""Tests T9 to T11 of specs/06e-kernels/plan.html: the C1 probe of the MLX tile sizes in a source
build (ADR 0013)."""

import ast
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[1]
WORKER = REPO_ROOT / "scripts" / "probe_kernels_worker.py"
FORBIDDEN_MODULES = {"pickle", "torch", "subprocess", "importlib", "marshal", "shelve"}
FORBIDDEN_CALLS = {"eval", "exec", "compile", "__import__"}


def _imports(tree: ast.Module) -> set[str]:
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            names.add((node.module or "").split(".")[0])
    return names


def test_probe_kernels_worker_imports() -> None:
    """T9: the worker imports only mlx, numpy and the standard library, and calls no subprocess,
    pickle or torch."""
    tree = ast.parse(WORKER.read_text())
    imports = _imports(tree)
    third_party = imports - set(sys.stdlib_module_names) - {"__future__"}
    assert third_party <= {"mlx", "numpy"}, third_party
    assert "dinov3_mlx" not in imports
    assert not FORBIDDEN_MODULES & imports
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            function = node.func
            if isinstance(function, ast.Name):  # a builtin; mx.compile is an attribute
                assert function.id not in FORBIDDEN_CALLS, f"call of {function.id}"
            for keyword in node.keywords:
                assert not (keyword.arg == "shell" and getattr(keyword.value, "value", False))
    for project_file in ("pyproject.toml", "uv.lock"):
        assert "mlx-src" not in (REPO_ROOT / project_file).read_text()


def test_probe_kernels_driver_refuses(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """T10: no source build without --allow-source-build; no worker file inside the source
    folder; the worker command uses -I."""
    import probe_kernels

    calls: list[list[str]] = []

    def fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("the driver ran a worker or took the lock")

    monkeypatch.setattr(probe_kernels, "run_worker", fail)
    monkeypatch.setattr(probe_kernels, "heavy_job", fail)
    results = tmp_path / "kernels.jsonl"
    for argv in (["control"], ["sweep", "--target", "T1"]):
        with pytest.raises(SystemExit) as stop:
            probe_kernels.main([*argv, "--results", str(results)])
        assert stop.value.code == 2
    assert not results.exists() and calls == []
    with pytest.raises(probe_kernels.SourceBuildRuleError):
        probe_kernels.check_worker_path(probe_kernels.SOURCE_DIR / "mlx" / "worker.py")
    probe_kernels.check_worker_path(WORKER)
    for side in ("wheel", "source"):
        command = probe_kernels.worker_command(side, {"mode": "probe"})
        assert command[1] == "-I" and command[2] == str(WORKER)
        assert json.loads(command[3]) == {"mode": "probe"}
    assert probe_kernels.worker_command("source", {})[0] == str(probe_kernels.SOURCE_PYTHON)
    assert probe_kernels.worker_command("wheel", {})[0] == sys.executable
    env = probe_kernels.worker_env({"PATH": "/usr/bin", "DINOV3_PROBE_GEMM_NAX": "64,64,256,2,2"})
    assert "DINOV3_PROBE_GEMM_NAX" not in env and env["MLX_ENABLE_TF32"] == "0"
    assert len(probe_kernels.SWEEP_CONFIGS["T1"]) <= 12
    assert all(len(configs) <= 12 for configs in probe_kernels.SWEEP_CONFIGS.values())


def _control(side: str, round_: int, seconds: float) -> dict:
    return {
        "kind": "control",
        "side": side,
        "round": round_,
        "status": "done",
        "probes": [{"name": "linear fp16 1280->1280", "seconds": seconds, "flops": 1e12}],
    }


def test_kernels_table() -> None:
    """T11: the ratio of each configuration to the control build and to the wheel; a
    configuration with a cosine below 0.99999 is not valid (fixture lines)."""
    import probe_kernels

    lines = []
    for r in range(1, 6):
        lines.append(_control("wheel", r, 1.0))
        lines.append(_control("source", r, 1.02 if r != 3 else 1.5))  # round 3 is an outlier
    control = probe_kernels.control_rows(lines)
    assert len(control) == 1
    row = control[0]
    assert row["ratio"] == pytest.approx(1 / 1.02)  # wheel time ÷ source time, median of 5
    assert row["within"]  # within ± 3 %
    sweep = {
        "kind": "sweep",
        "status": "done",
        "target": "T1",
        "probe": "linear fp16 1280->1280",
        "flops": 1e12,
        "configs": {
            "control": {"seconds": [1.0, 1.0, 1.0, 1.0, 1.0], "cosine": 0.9999995},
            "64,64,256,2,2": {"seconds": [0.9, 0.8, 0.8, 0.8, 0.8], "cosine": 0.9999990},
            "128,128,64,4,4": {"seconds": [0.5, 0.5, 0.5, 0.5, 0.5], "cosine": 0.99990},
            "128,64,256,4,2": {"seconds": None, "cosine": None, "error": "no kernel"},
        },
    }
    rows = probe_kernels.sweep_rows([sweep], control)
    by_config = {r["config"]: r for r in rows}
    assert by_config["64,64,256,2,2"]["ratio"] == pytest.approx(1.25)
    assert by_config["64,64,256,2,2"]["ratio_to_wheel"] == pytest.approx(1.25 / 1.02)
    assert by_config["64,64,256,2,2"]["valid"]
    assert by_config["64,64,256,2,2"]["tflops"] == pytest.approx(1.25)
    assert by_config["128,128,64,4,4"]["ratio"] == pytest.approx(2.0)
    assert not by_config["128,128,64,4,4"]["valid"]
    assert by_config["128,64,256,4,2"]["ratio"] is None and not by_config["128,64,256,4,2"]["valid"]
    best = probe_kernels.best_rows(rows)
    assert [(b["target"], b["config"]) for b in best] == [("T1", "64,64,256,2,2")]
    text = probe_kernels.write_table(lines + [sweep])
    assert "64,64,256,2,2" in text and "not valid" in text
