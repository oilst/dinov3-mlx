"""Tests T1 to T16 of specs/06b-prior-art/plan.html: the prior-art comparison.

The default session needs no gated weights, no coremltools and no third-party code. T15 and T16
run in the slow session (DINOV3_SLOW=1): T15 starts the Core ML worker in venv-coreml, T16 loads
ViT-S/16.
"""

import ast
import json
import math
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from small_model import small_hf_model, small_mlx_model, small_pixels
from under_test import SLOW

from dinov3_apple import prior_art
from dinov3_apple.mlx_model import apply_rope, compile_backbone, rope_layout, rope_tables
from dinov3_apple.outputs import NUM_PREFIX_TOKENS
from dinov3_apple.parity import TEST_IMAGES, THRESHOLDS
from dinov3_apple.speedup import NOISE
from dinov3_apple.stored import IMAGENET_MEAN_VALUES, IMAGENET_STD_VALUES

REPO_ROOT = Path(__file__).parents[1]
SCRIPTS = REPO_ROOT / "scripts"
COREML_WORKER = SCRIPTS / "prior_art_coreml_worker.py"
PORT_WORKER = SCRIPTS / "prior_art_port_worker.py"
HEAD_DIM = 64
THETA = 100.0
slow = pytest.mark.skipif(not SLOW, reason="slow session: DINOV3_SLOW=1")


# T1 to T5: the candidates, the exports, the parity of an export, the pair plan and the rows.


def test_candidates_cover_the_explore_table() -> None:
    """T1."""
    assert list(prior_art.CANDIDATES) == [f"C{i}" for i in range(1, 10)]
    for number, candidate in prior_art.CANDIDATES.items():
        assert candidate.license, number
        assert candidate.commit, number
        assert candidate.verdict in prior_art.VERDICTS, number
        if candidate.verdict in ("skip", "gated"):
            assert candidate.reason, number
    assert [n for n, c in prior_art.CANDIDATES.items() if c.verdict == "gated"] == ["C1", "C2"]
    assert [n for n, c in prior_art.CANDIDATES.items() if c.verdict == "measure"] == [
        "C7",
        "C8",
        "C9",
    ]


def test_exports_have_package_paths_inside_prior_art_dir() -> None:
    """T2."""
    from dinov3_apple.flops import tokens

    assert list(prior_art.EXPORTS) == [f"E{i}" for i in range(1, 8)]
    for name, export in prior_art.EXPORTS.items():
        assert export.path.is_relative_to(prior_art.PRIOR_ART_DIR), name
        assert export.path.suffix == ".mlpackage", name
        assert export.candidate in ("C7", "C8", "C9"), name
        assert export.dtype_mode in THRESHOLDS, name
    e7 = prior_art.EXPORTS["E7"]
    assert e7.family == "dinov2" and e7.pixels == "random"
    assert e7.tokens == tokens(e7.ours_side) == 1374
    assert all(e.tokens == tokens(e.side) for n, e in prior_art.EXPORTS.items() if n != "E7")


def test_export_parity_patch_only_and_normalized() -> None:
    """T3."""
    rng = np.random.default_rng(0)
    d, h, w = 8, 3, 4
    reference = rng.standard_normal((NUM_PREFIX_TOKENS + h * w, d)).astype(np.float32)
    factors = rng.uniform(0.1, 5.0, (h * w, 1)).astype(np.float32)
    patches = reference[NUM_PREFIX_TOKENS:] * factors
    parity = prior_art.export_parity(reference, None, patches)
    assert parity["frame"] is None
    assert parity["patch_mean"] == pytest.approx(1.0, abs=1e-12)
    assert parity["patch_p01"] == pytest.approx(1.0, abs=1e-12)
    with_frame = prior_art.export_parity(reference, 2.0 * reference[0], patches)
    assert with_frame["frame"] == pytest.approx(1.0, abs=1e-12)
    # A grid (1, D, h, w) maps to (h · w, D) in row-major order: token n = w · i + j.
    grid = np.arange(d * h * w, dtype=np.float32).reshape(1, d, h, w)
    tokens_ = prior_art.grid_tokens(grid)
    assert tokens_.shape == (h * w, d)
    assert np.array_equal(tokens_[w * 1 + 2], grid[0, :, 1, 2])
    frame, mapped = prior_art.export_outputs({"features": grid})
    assert frame is None and np.array_equal(mapped, tokens_)
    frame, mapped = prior_art.export_outputs(
        {"embedding": np.ones((1, d)), "patch_embeddings": np.ones((1, h * w, d))}
    )
    assert frame.shape == (d,) and mapped.shape == (h * w, d)


def test_coreml_pair_plan() -> None:
    """T4."""
    specs = prior_art.coreml_plan(rounds=5)
    assert len(specs) == 5 * 24
    assert [s["position"] for s in specs] == list(range(1, 121))
    group_names = [group.name for group in prior_art.GROUPS]
    for r in range(1, 6):
        members = [s for s in specs if s["round"] == r]
        assert len(members) == 24
        assert sum(s["kind"] == "ours" for s in members) == 10
        assert sum(s["kind"] == "export" for s in members) == 14
        first_groups = list(dict.fromkeys(s["group"] for s in members))
        assert first_groups == group_names[r - 1 :] + group_names[: r - 1]  # the rotation
        for group in group_names:
            for b in (1, 8):
                cell = [s for s in members if s["group"] == group and s["batch_size"] == b]
                kinds = [s["kind"] for s in cell]
                assert kinds.count("ours") == 1
                assert kinds[0 if r % 2 else -1] == "ours"
    for r in range(1, 6):
        for shared in (("E1", "E2"), ("E4", "E5")):
            groups = {s["group"] for s in specs if s["round"] == r and s.get("export") in shared}
            assert len(groups) == 1
    for spec in specs:
        assert spec["job"] == "J1"
        assert spec.get("units", "CPU_AND_GPU") == "CPU_AND_GPU"
        if spec["kind"] == "ours":
            assert spec["config"]["compile"] is True
            assert spec["pixels"] == ("random" if spec["config"]["side"] == 592 else "stored")


def _line(kind: str, r: int, ips: float, status: str = "done", **fields: object) -> dict:
    return {
        "job": "J1",
        "side_kind": kind,
        "round": r,
        "status": status,
        "batch_size": 8,
        "group": "G1",
        "patch": {"throughput_ips": ips} if status == "done" else None,
        **fields,
    }


def test_cell_row_ratio_and_verdict() -> None:
    """T5."""
    ours = [_line("ours", r, ips) for r, ips in enumerate([100, 110, 90, 100, 120], start=1)]
    export = [
        _line("export", r, ips, export="E1") for r, ips in enumerate([50, 55, 100, 100, 60], 1)
    ]
    row = prior_art.cell_row(ours, export)
    assert row.ratios == pytest.approx([2.0, 2.0, 0.9, 1.0, 2.0])
    assert row.median == pytest.approx(2.0)
    assert row.spread == pytest.approx((2.0 - 0.9) / 0.9)
    assert row.verdict == "above 1"
    assert row.n == 5
    # The lower median of an even count, and the noise mark.
    export[4] = _line("export", 5, 0, status="error", export="E1")
    row = prior_art.cell_row(ours, export)
    assert row.n == 4
    assert row.ratios == pytest.approx([2.0, 2.0, 0.9, 1.0])
    assert row.median == pytest.approx(1.0)
    assert row.verdict == "not above 1, inside the noise"
    # A failed round of ours drops only its own pair.
    ours[0] = _line("ours", 1, 0, status="skipped")
    row = prior_art.cell_row(ours, export)
    assert row.ratios == pytest.approx([2.0, 0.9, 1.0])
    assert prior_art.single_verdict(1 + NOISE + 0.01) == "above 1"


def test_slow_cell_facts() -> None:
    """T5b (build addition for AC4): a cell below 0.97 gets the transfer share of ours and the
    ratio of our device throughput to the export throughput."""
    lines = []
    for r in (1, 2, 3):
        ours = _line("ours", r, 100.0, group="G2", config={"side": 512})
        ours["device"] = {"throughput_ips": 110.0}
        lines += [ours, _line("export", r, 120.0, export="E3", group="G2")]
    facts = prior_art.slow_cell_facts(lines)
    assert [(f["export"], f["batch_size"]) for f in facts] == [("E3", 8)]
    fact = facts[0]
    assert fact["ratio"] == pytest.approx(100 / 120)
    assert fact["transfer_share"] == pytest.approx(1 - 100 / 110)
    assert fact["device_ratio"] == pytest.approx(110 / 120)
    assert fact["tokens"] == 1029


def test_between_runs() -> None:
    """T5c (finding V1 of verify.html): two runs of J1 give the run medians, the pooled lower
    median of all ratios, the spread between the run medians and pass_verdict."""
    run1, run2 = [], []
    for r, (a, b) in enumerate([(100, 95), (100, 96), (100, 97)], start=1):
        run1 += [_line("ours", r, a, group="G2"), _line("export", r, b, export="E3", group="G2")]
    for r, (a, b) in enumerate([(100, 101), (100, 102), (100, 99)], start=1):
        run2 += [_line("ours", r, a, group="G2"), _line("export", r, b, export="E3", group="G2")]
    rows = prior_art.between_runs(run1, run2)
    assert [(row["export"], row["batch_size"]) for row in rows] == [("E3", 8)]
    row = rows[0]
    assert row["run_medians"] == pytest.approx([100 / 96, 100 / 101])
    assert row["pooled"] == pytest.approx(100 / 99)
    assert row["spread"] == pytest.approx((100 / 96 - 100 / 101) / (100 / 101))
    assert row["verdict"] == "not above 1, inside the noise"
    assert row["n"] == 6


# T6 to T11: the port-change variants.


def _bf16(values: np.ndarray) -> np.ndarray:
    """Round fp32 values to bf16 (round to nearest even), returned as fp32."""
    bits = np.asarray(values, np.float32).view(np.uint32).astype(np.uint64)
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return rounded.astype(np.uint32).view(np.float32)


def _port_tables_oracle(grid_h: int, grid_w: int) -> tuple[np.ndarray, np.ndarray]:
    """The steps of the RoPE of C1 and C2, each rounded to bf16, in NumPy."""
    k = _bf16(np.arange(HEAD_DIM // 4, dtype=np.float32))
    exponent = _bf16(_bf16(2 * k) / np.float32(HEAD_DIM // 2))
    periods = _bf16(np.float32(THETA) ** exponent)
    coords_h = _bf16(np.arange(0.5, grid_h, dtype=np.float32) / np.float32(grid_h))
    coords_w = _bf16(np.arange(0.5, grid_w, dtype=np.float32) / np.float32(grid_w))
    mesh = np.stack(np.meshgrid(coords_h, coords_w, indexing="ij"), axis=-1).reshape(-1, 2)
    mesh = _bf16(_bf16(2.0 * mesh) - 1.0)
    angles = _bf16(_bf16(_bf16(np.float32(2 * math.pi)) * mesh[:, :, None]) / periods)
    angles = np.tile(angles.reshape(len(angles), -1), 2)
    return _bf16(np.cos(angles)), _bf16(np.sin(angles))


@pytest.mark.parametrize("grid", [(14, 14), (32, 32), (14, 19)], ids=lambda g: f"{g[0]}x{g[1]}")
def test_port_rope_tables_match_a_numpy_oracle(grid: tuple[int, int]) -> None:
    """T6. The GPU and NumPy compute cos and sin in fp32 with other code, so a value next to a
    rounding boundary of bf16 may round to the next bf16 value: at most 1 step of 2^-8.

    The bound on the difference to the fp32 tables is 0.04, not the 0.02 of the plan (measured
    0.0314 at 14 × 14): the angles reach 2π, where one bf16 step is 2^-5 = 0.031 (deviation of the
    build).
    """
    from dinov3_apple.port_variants import port_rope_tables

    cos, sin = port_rope_tables(*grid, HEAD_DIM, THETA)
    assert cos.dtype == sin.dtype == mx.bfloat16
    n = grid[0] * grid[1]
    assert cos.shape == sin.shape == (n, HEAD_DIM)
    cos, sin = np.array(cos.astype(mx.float32)), np.array(sin.astype(mx.float32))
    oracle_cos, oracle_sin = _port_tables_oracle(*grid)
    for got, want in ((cos, oracle_cos), (sin, oracle_sin)):
        difference = np.abs(got - want)
        assert difference.max() <= 2**-8
        assert np.mean(difference > 0) <= 0.01
    fp32_cos, fp32_sin = rope_tables(*grid, HEAD_DIM, THETA)
    for got, fp32 in ((cos, fp32_cos), (sin, fp32_sin)):
        difference = float(np.abs(got - fp32[NUM_PREFIX_TOKENS:]).max())
        assert 0 < difference <= 0.04


@pytest.mark.parametrize(("grid", "angle"), [((14, 14), 0.00791), ((32, 32), 0.00825)])
def test_meta_periods_angle_difference(grid: tuple[int, int], angle: float) -> None:
    """T7."""
    from dinov3_apple.port_variants import meta_periods, meta_rope_tables

    periods = meta_periods(HEAD_DIM, THETA)
    exact = THETA ** (np.arange(HEAD_DIM // 4, dtype=np.float64) / (HEAD_DIM // 4))
    assert np.array_equal(periods, _bf16(exact.astype(np.float32)))
    assert periods[:2].tolist() == [1.0, 1.3359375]
    cos, sin = meta_rope_tables(*grid, HEAD_DIM, THETA)
    hf_cos, hf_sin = rope_tables(*grid, HEAD_DIM, THETA)
    assert cos.shape == hf_cos.shape and cos.dtype == np.float32
    assert np.array_equal(cos[:NUM_PREFIX_TOKENS], hf_cos[:NUM_PREFIX_TOKENS])
    meta_angles = np.arctan2(sin[NUM_PREFIX_TOKENS:], cos[NUM_PREFIX_TOKENS:]).astype(np.float64)
    hf_angles = np.arctan2(hf_sin[NUM_PREFIX_TOKENS:], hf_cos[NUM_PREFIX_TOKENS:])
    difference = np.abs(np.angle(np.exp(1j * (meta_angles - hf_angles))))
    assert float(difference.max()) == pytest.approx(angle, abs=1e-5)


@pytest.fixture(scope="module")
def small_hf() -> object:
    return small_hf_model(seed=3)


def _tokens(forward: object, pixels: np.ndarray) -> np.ndarray:
    tokens_ = forward(mx.array(pixels)).astype(mx.float32)
    mx.eval(tokens_)
    return np.array(tokens_)


@pytest.mark.mlx
def test_variant_v0_is_bit_for_bit(small_hf: object) -> None:
    """T8."""
    from dinov3_apple.port_variants import variant_forward

    pixels = small_pixels((2, 3, 224, 224))
    want = _tokens(compile_backbone(small_mlx_model(small_hf)), pixels)
    got = _tokens(variant_forward(small_mlx_model(small_hf), "V0"), pixels)
    assert float(np.abs(got - want).max()) == 0.0
    non_square = small_pixels((1, 3, 224, 304))
    want = _tokens(compile_backbone(small_mlx_model(small_hf)), non_square)
    got = _tokens(variant_forward(small_mlx_model(small_hf), "V0"), non_square)
    assert float(np.abs(got - want).max()) == 0.0


@pytest.mark.mlx
@pytest.mark.parametrize(("dtype_mode", "bound"), [("fp32", 1e-5), ("fp16-cast", 0.03)])
def test_unfused_attention_matches_sdpa(small_hf: object, dtype_mode: str, bound: float) -> None:
    """T9. The bound is relative to the largest absolute value of the V0 tokens."""
    from dinov3_apple.port_variants import UnfusedAttention, variant_forward

    pixels = small_pixels((2, 3, 224, 224))
    want = _tokens(variant_forward(small_mlx_model(small_hf, dtype_mode), "V0"), pixels)
    model = small_mlx_model(small_hf, dtype_mode)
    got = _tokens(variant_forward(model, "V4"), pixels)
    assert all(isinstance(block.attention, UnfusedAttention) for block in model.layer)
    assert np.isfinite(got).all()
    assert float(np.abs(got - want).max()) <= bound * float(np.abs(want).max())


@pytest.mark.mlx
def test_bf16_rotation_rounds_prefix_tokens(small_hf: object) -> None:
    """T10. Bf16RotationAttention calls port_apply_rope, so a check of port_apply_rope on fp32 q
    covers the cast of the prefix tokens (finding V5 of verify.html)."""
    from dinov3_apple.port_variants import (
        Bf16RotationAttention,
        port_apply_rope,
        port_rope_tables,
        variant_forward,
    )

    pixels = small_pixels((2, 3, 224, 224))
    want = _tokens(variant_forward(small_mlx_model(small_hf), "V0"), pixels)
    got = _tokens(variant_forward(small_mlx_model(small_hf), "V1"), pixels)
    assert float(np.abs(got - want).max()) > 0
    assert np.isfinite(got).all()
    # The rotation: the q of a prefix token is the bf16 value of the fp32 q, without a rotation.
    q = mx.array(small_pixels((2, 6, NUM_PREFIX_TOKENS + 196, HEAD_DIM), seed=4))
    cos, sin = port_rope_tables(14, 14, HEAD_DIM, THETA)
    rotated = port_apply_rope(q, cos, sin)
    assert rotated.dtype == mx.float32
    prefix = np.array(rotated[:, :, :NUM_PREFIX_TOKENS])
    assert np.array_equal(prefix, _bf16(np.array(q[:, :, :NUM_PREFIX_TOKENS])))
    cos32, sin32 = rope_layout(*rope_tables(14, 14, HEAD_DIM, THETA))
    exact = np.array(apply_rope(q, mx.array(cos32), mx.array(sin32)))
    patch_difference = np.abs(np.array(rotated) - exact)[:, :, NUM_PREFIX_TOKENS:]
    assert 0 < float(patch_difference.max()) <= 0.1 * float(np.abs(exact).max())
    import inspect

    assert "port_apply_rope" in inspect.getsource(Bf16RotationAttention.__call__)


def test_eight_bit_pixels() -> None:
    """T11."""
    mean = IMAGENET_MEAN_VALUES.reshape(1, 3, 1, 1)
    std = IMAGENET_STD_VALUES.reshape(1, 3, 1, 1)
    pixels = ((np.random.default_rng(5).uniform(0, 1, (2, 3, 16, 16)) - mean) / std).astype(
        np.float32
    )
    out = prior_art.eight_bit_pixels(pixels)
    assert out.dtype == np.float32 and out.shape == pixels.shape
    bound = 0.5 / 255 / std + 1e-6
    assert (np.abs(out - pixels) <= bound).all()
    assert (out != pixels).any()
    levels = np.random.default_rng(6).integers(0, 256, (2, 3, 16, 16)).astype(np.float32)
    on_grid = ((levels / 255 - mean) / std).astype(np.float32)
    assert np.array_equal(prior_art.eight_bit_pixels(on_grid), on_grid)


# T12 to T14: the worker files and the driver.

COREML_IMPORTS = {"numpy", "safetensors", "coremltools", "PIL"}
PORT_IMPORTS = {"numpy", "safetensors", "mlx", "mlxim", "dinov3"}  # dinov3: mlx_dinov3, the clone
FORBIDDEN_CALLS = {"eval", "exec", "compile", "__import__"}


def _imports(tree: ast.Module) -> set[str]:
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            names.add((node.module or "").split(".")[0])
    return names


@pytest.mark.parametrize(
    ("path", "allowed"), [(COREML_WORKER, COREML_IMPORTS), (PORT_WORKER, PORT_IMPORTS)]
)
def test_workers_are_standalone_and_safe(path: Path, allowed: set[str]) -> None:
    """T12."""
    tree = ast.parse(path.read_text())
    third_party = _imports(tree) - set(sys.stdlib_module_names) - {"__future__"}
    assert third_party <= allowed, third_party - allowed
    assert "dinov3_apple" not in _imports(tree)
    assert not {"pickle", "torch", "subprocess", "importlib", "marshal", "shelve"} & _imports(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            function = node.func
            if isinstance(function, ast.Name):  # a builtin; model.eval() of mlx is a method
                assert function.id not in FORBIDDEN_CALLS, f"{path.name}: call of {function.id}"
            name = getattr(function, "attr", "")
            assert name not in ("load_state_dict", "torch_load"), f"{path.name}: {name}"
            for keyword in node.keywords:
                assert not (keyword.arg == "shell" and getattr(keyword.value, "value", False))
    for project_file in ("pyproject.toml", "uv.lock"):
        text = (REPO_ROOT / project_file).read_text().lower()
        for package in ("mlxim", "mlx-image", "mlx_dinov3", "mlx-dinov3", "mlxdinov3"):
            assert package not in text, f"{project_file} names {package}"
    # Step 6c added coremltools as its own dependency group "coreml" (commit 092e6e3): Apple's tool
    # for our own conversion, not a candidate. It stays out of the main dependencies.
    import tomllib

    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    assert not any("coremltools" in d for d in project["project"]["dependencies"])


def test_ports_subcommand_is_gated(tmp_path: Path) -> None:
    """T13."""
    import bench_prior_art

    calls = []
    results = tmp_path / "prior-art.jsonl"
    counts = bench_prior_art.run_ports(
        rounds=5,
        allowed=False,
        results=results,
        bench_dir=tmp_path / "ports",
        runner=lambda spec, bench_dir: calls.append(spec) or {"status": "done"},
    )
    assert calls == []
    lines = [json.loads(text) for text in results.read_text().splitlines()]
    assert len(lines) == 8 == counts["blocked"]
    assert len(prior_art.PORT_CELLS) == 8
    for line in lines:
        assert line["status"] == "blocked"
        assert line["side_kind"] == "port" and line["job"] == "J5"
        assert "no permission to run third-party Python code" in line["reason"]
        assert "--allow-third-party-python" in line["command"]
        assert "dinov3_apple.gpu_lock" in line["command"]


def test_driver_env_and_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """T14."""
    import bench_prior_art

    ours = prior_art.coreml_plan(rounds=1)[0]
    assert ours["kind"] == "ours"
    env = bench_prior_art.worker_env(ours, {"PATH": "/usr/bin", "MLX_ENABLE_TF32": "1"})
    assert env["MLX_ENABLE_TF32"] == "0"
    export = next(s for s in prior_art.coreml_plan(rounds=1) if s["kind"] == "export")
    command = bench_prior_art.worker_command(export, tmp_path)
    venv = prior_art.PRIOR_ART_DIR / "venv-coreml"
    assert command[0] == str(venv / "bin" / "python")
    assert command[1] == "-I"
    assert command[2] == str(COREML_WORKER)
    assert json.loads(command[3])["package"] == str(prior_art.EXPORTS[export["export"]].path)
    # A worker path inside the prior-art folder is refused.
    with pytest.raises(bench_prior_art.ThirdPartyRuleError):
        bench_prior_art.check_worker_path(prior_art.PRIOR_ART_DIR / "mlx-image" / "x.py")
    # A missing venv stops the driver before the lock.
    monkeypatch.setattr(bench_prior_art, "PRIOR_ART_DIR", tmp_path)
    monkeypatch.setattr(prior_art, "PRIOR_ART_DIR", tmp_path)

    def fail_lock(label: str) -> None:
        raise AssertionError("the driver took the lock")

    monkeypatch.setattr(bench_prior_art, "heavy_job", fail_lock)
    with pytest.raises(SystemExit) as stop:
        bench_prior_art.main(["coreml", "--rounds", "1", "--results", str(tmp_path / "r.jsonl")])
    assert stop.value.code == 2
    assert not (tmp_path / "r.jsonl").exists()


def test_variant_parity_line_names_twelve_configurations() -> None:
    """Finding V7 of verify.html: a J3 line has no input resolution; its label names the 12
    configurations."""
    import bench_prior_art

    spec = prior_art.variant_parity_plan(["ViT-S/16"])[0]
    line = bench_prior_art.spec_line(spec, {"status": "done", "configurations": []}, 1, 2, None)
    assert line["side"] is None and line["batch_size"] == 1
    assert "12 configurations" in bench_prior_art.spec_label(spec)


def test_lines_have_no_local_folders() -> None:
    """A line of results/prior-art.jsonl has a repo-relative outputs_file and ~ for the home
    folder in its reason (a public repo)."""
    import bench_prior_art

    from dinov3_apple.stored import REPO_ROOT

    spec = prior_art.export_spec("E1", batch_size=1, units="CPU_AND_GPU")
    outputs = REPO_ROOT / "tests" / "fixtures" / "generated" / "bench" / "o.safetensors"
    worker = {
        "status": "error",
        "reason": f"exit code 1: {Path.home()}/.cache/x.py and {REPO_ROOT}/.venv/y.py",
        "outputs_file": str(outputs),
    }
    line = bench_prior_art.spec_line(spec, worker, 1, 2, None)
    assert line["outputs_file"] == "tests/fixtures/generated/bench/o.safetensors"
    assert line["reason"] == "exit code 1: ~/.cache/x.py and .venv/y.py"
    port = {"kind": "port", "job": "J5", "candidate": "C1", "model": "ViT-S/16", "dtype": "fp32"}
    line = bench_prior_art.spec_line(port, worker | {"status": "done"}, 1, 2, None)
    assert line["outputs_file"] == "tests/fixtures/generated/bench/o.safetensors"
    assert str(Path.home()) not in json.dumps(line)


def test_parity_error_note_has_no_local_folders() -> None:
    """A parity read that fails names the outputs file in the validity note: the note names it
    repo-relative (a public repo)."""
    import bench_prior_art

    from dinov3_apple.stored import REPO_ROOT

    spec = prior_art.export_spec("E1", batch_size=1, units="CPU_AND_GPU")
    missing = REPO_ROOT / "tests" / "fixtures" / "generated" / "bench" / "missing.safetensors"
    worker = {
        "status": "done",
        "finite": True,
        "outputs_file": str(missing),
        "output_names": ["patch"],
    }
    line = bench_prior_art.spec_line(spec, worker, 1, 2, None)
    assert line["valid"] is False
    assert line["validity_note"].startswith("parity error: FileNotFoundError")
    assert "tests/fixtures/generated/bench/missing.safetensors" in line["validity_note"]
    assert str(REPO_ROOT) not in json.dumps(line)
    assert str(Path.home()) not in json.dumps(line)


def test_port_weights_fill_only_known_keys() -> None:
    """Finding V8 of verify.html: the port worker fills only the RoPE periods and the k-bias
    masks with the values of the port; any other missing key stops it."""
    import prior_art_port_worker as worker

    params = {
        "blocks.0.attn.qkv.weight": 1,
        "blocks.0.attn.qkv.bias_mask": 2,
        "rope_embed.periods": 3,
    }
    pairs, filled = worker.port_weights(params, {"blocks.0.attn.qkv.weight": 5})
    assert dict(pairs) == {"blocks.0.attn.qkv.weight": 5, **{k: params[k] for k in filled}}
    assert sorted(filled) == ["blocks.0.attn.qkv.bias_mask", "rope_embed.periods"]
    with pytest.raises(KeyError):
        worker.port_weights(params | {"norm.weight": 4}, {"blocks.0.attn.qkv.weight": 5})


# T15 and T16: the slow session.


@slow
def test_coreml_worker_end_to_end(tmp_path: Path) -> None:
    """T15: E1 at batch size 1 with a 1 s budget, in venv-coreml with -I."""
    import bench_prior_art

    export = prior_art.EXPORTS["E1"]
    if not (prior_art.PRIOR_ART_DIR / "venv-coreml").exists() or not export.path.exists():
        pytest.skip("venv-coreml or the E1 package is missing")
    spec = prior_art.export_spec("E1", batch_size=1, units="CPU_AND_GPU")
    spec["budget"] = "short"
    result = bench_prior_art.run_worker(spec, tmp_path)
    assert result["status"] == "done", result.get("reason")
    line = bench_prior_art.export_line_fields(spec, result)
    assert len(line["parity"]["images"]) == len(TEST_IMAGES)
    for image in line["parity"]["images"]:
        assert image["patch_p01"] >= THRESHOLDS["fp32"].patch_p01
    assert line["valid"] is True
    assert line["patch"]["throughput_ips"] > 0


@slow
@pytest.mark.mlx
def test_variant_parity_vits16() -> None:
    """T16: V0 equals the compiled forward pass bit for bit on ViT-S/16 fp32 at 224 px; V1 gives a
    lower patch 1st percentile."""
    from dinov3_apple.port_variants import run_variant_parity

    v0 = run_variant_parity("ViT-S/16", "fp32", "V0", configurations=["iss-kibo-lab/224x224"])
    v1 = run_variant_parity("ViT-S/16", "fp32", "V1", configurations=["iss-kibo-lab/224x224"])
    assert v0["compiled_difference"] == 0.0
    row0, row1 = v0["configurations"][0], v1["configurations"][0]
    assert row0["passes"] is True
    assert row1["parity"]["patch_p01"] < row0["parity"]["patch_p01"]
