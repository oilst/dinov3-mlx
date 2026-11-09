"""Tests T1 to T13 of specs/06c-neural-engine/plan.html and T14 to T17 of
specs/06d-neural-engine-fault/plan.html: Core ML on the Neural Engine.

T1 to T5, T9, T13 and T14 use synthetic tensors or the small models of small_model.py in torch only
and run in the default session. Every test that converts or loads a Core ML package (T6 to T8, T10
to T12, T15 to T17) runs in the slow session (DINOV3_SLOW=1).
"""

import copy
import hashlib
import types
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

ct = pytest.importorskip("coremltools")

import torch
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.frontend.torch import ops as torch_ops
from small_model import small_hf_model, small_pixels
from transformers.models.dinov3_vit.modeling_dinov3_vit import eager_attention_forward
from under_test import SLOW

from dinov3_apple import coreml
from dinov3_apple.machine import MachineState
from dinov3_apple.memory import RunRuleError
from dinov3_apple.parity import TEST_IMAGES, THRESHOLDS, measure_parity
from dinov3_apple.stored import read_inputs, read_reference

slow = pytest.mark.skipif(not SLOW, reason="slow session: DINOV3_SLOW=1")
ORIGINAL_CAST = torch_ops._cast
SMALL_SIDE = 32


def _qkv(dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Synthetic q, k, v (B 2, 3 heads, 41 tokens, head dimension 64), scores up to 1e7."""
    generator = torch.Generator().manual_seed(0)
    shape = (2, 3, 41, 64)
    q = torch.randn(shape, generator=generator, dtype=torch.float64) * 1000.0
    k = torch.randn(shape, generator=generator, dtype=torch.float64) * 1000.0
    v = torch.randn(shape, generator=generator, dtype=torch.float64)
    largest = float((torch.matmul(q, k.transpose(2, 3)) * 64**-0.5).abs().max())
    q = q * (1e7 / largest)  # the largest score is 1e7: 1e7 / 256 = 39,063 fits fp16
    return q.to(dtype), k.to(dtype), v.to(dtype)


def _relative(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max() / b.abs().max())


# T1 and T2: the attention of the Neural Engine rewrite.


def test_ne_attention_equals_eager_in_fp32() -> None:
    """T1: exact in real numbers, also with scores far above the fp16 range."""
    q, k, v = _qkv(torch.float64)
    module = torch.nn.Module().eval()
    scaling = 64**-0.5
    scores = torch.matmul(q, k.transpose(2, 3)) * scaling
    assert float(scores.abs().max()) > 1e6
    expected, _ = eager_attention_forward(module, q, k, v, None, scaling=scaling)
    output, weights = coreml.ne_attention(module, q, k, v, None, scaling=scaling)
    assert weights is None
    assert output.shape == expected.shape
    assert _relative(output, expected) <= 1e-6


def test_ne_attention_fp16_stays_finite() -> None:
    """T2 (finding F1): eager attention in fp16 breaks, ne_attention stays finite and close."""
    module = torch.nn.Module().eval()
    scaling = 64**-0.5
    q64, k64, v64 = _qkv(torch.float64)
    expected, _ = eager_attention_forward(module, q64, k64, v64, None, scaling=scaling)
    q, k, v = _qkv(torch.float16)
    broken, _ = eager_attention_forward(module, q, k, v, None, scaling=scaling)
    assert not bool(torch.isfinite(broken).all())
    output, _ = coreml.ne_attention(module, q, k, v, None, scaling=scaling)
    assert bool(torch.isfinite(output).all())
    cosine = torch.nn.functional.cosine_similarity(
        output.double().flatten(), expected.flatten(), dim=0
    )
    assert float(cosine) >= 0.999


# T3 and T4: the rewrite of the backbone.


@pytest.mark.parametrize("name", ["vits16", "vits16plus"])
@pytest.mark.parametrize("scale", [1, 8])
def test_rewrite_is_exact_in_fp32(name: str, scale: int) -> None:
    """T3: the rewritten backbone gives the tokens of the Hugging Face backbone. The layer norm
    sees the residual stream divided by r, and its ε becomes ε / r² (plan amendment)."""
    model = small_hf_model(seed=3, name=name)
    reference = copy.deepcopy(model)
    pixels = torch.from_numpy(small_pixels((2, 3, 64, 64)))
    with torch.no_grad():
        expected = reference(pixel_values=pixels).last_hidden_state
        output = coreml.rewrite_for_neural_engine(model, scale)(pixels)
    assert output.shape == expected.shape
    assert _relative(output, expected) <= 1e-6


def test_rewrite_keeps_layer_scale() -> None:
    """T4: no LayerScale fold; o_proj and down_proj divided by r bit for bit; ε / r²."""
    model = small_hf_model(seed=4, name="vits16plus")
    original = copy.deepcopy(model)
    backbone = coreml.rewrite_for_neural_engine(model, 8)
    assert backbone.residual_scale == 8
    for layer, before in zip(model.model.layer, original.model.layer, strict=True):
        for name in ("layer_scale1", "layer_scale2"):
            assert torch.equal(getattr(layer, name).lambda1, getattr(before, name).lambda1)
        for proj, old in [
            (layer.attention.o_proj, before.attention.o_proj),
            (layer.mlp.down_proj, before.mlp.down_proj),
        ]:
            assert torch.equal(proj.weight, old.weight / 8)
            assert torch.equal(proj.bias, old.bias / 8)
        assert torch.equal(layer.attention.q_proj.weight, before.attention.q_proj.weight)
        assert layer.norm1.eps == before.norm1.eps / 64
        assert layer.norm2.eps == before.norm2.eps / 64
    assert model.norm.eps == original.norm.eps / 64
    assert model.config._attn_implementation == coreml.NE_ATTENTION


# T14: the embeddings with one constant prefix (Step 6d).


@pytest.mark.parametrize("name", ["vits16", "vits16plus"])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("side", [32, 64])
def test_ne_embeddings_equal_hugging_face(name: str, batch_size: int, side: int) -> None:
    """T14: ne_embeddings gives the tokens of model.embeddings bit for bit."""
    model = small_hf_model(seed=14, name=name)
    pixels = torch.from_numpy(small_pixels((batch_size, 3, side, side)))
    with torch.no_grad():
        expected = model.embeddings(pixels)
        output = coreml.ne_embeddings(model.embeddings, pixels)
    assert output.shape == expected.shape
    assert torch.equal(output, expected)


# T5: the residual-stream scale.


def test_residual_scale_from_token_range(tmp_path: Path) -> None:
    """T5."""
    sizes = ["ViT-S/16", "ViT-S+/16", "ViT-B/16", "ViT-L/16", "ViT-H+/16"]
    assert [coreml.residual_scale(size) for size in sizes] == [1, 1, 1, 8, 8]
    with pytest.raises(RunRuleError):
        coreml.residual_scale("ViT-7B/16")
    table = tmp_path / "token-range.md"
    table.write_text(
        "| Model size | Dtype mode | Largest token value | Configuration |\n"
        "| --- | --- | --- | --- |\n"
        "| ViT-S/16 | fp32 | 70,000.0 | forest/224x224 |\n"
        "| ViT-S/16 | fp16-cast | 99,000.0 | forest/224x224 |\n"
    )
    assert coreml.residual_scale("ViT-S/16", table) == 4


# T9: the package name, the line fields, the validity of a broken output, the stream schedule.


def test_package_path_and_line_fields() -> None:
    """T9."""
    import bench_coreml

    path = coreml.package_path("ViT-L/16", 512, 8)
    assert path.name == f"vitl16-512-b8-ne3-s256-r8-ct{coreml.COREMLTOOLS_VERSION}.mlpackage"
    assert path.parent == coreml.COREML_DIR
    plain = coreml.package_path("ViT-S/16", 224, 1, rewrite=False)
    assert plain.name == f"vits16-224-b1-plain-ct{coreml.COREMLTOOLS_VERSION}.mlpackage"
    assert coreml.compiled_path(path).name == path.name.replace(".mlpackage", ".mlmodelc")

    spec = bench_coreml.coreml_spec("matrix", "coreml-ne", "ViT-S/16", 224, 8, group=0)
    worker = {
        "status": "done",
        "compute_unit": "CPU_AND_NE",
        "package": path.name,
        "rewrite": {"score_scale": 256, "residual_scale": 8},
        "first_load_s": 2.0,
        "cached_load_s": 0.1,
    }
    machine = MachineState("AC Power", "off", "", 0, "Apple M5 Max", 0)
    line = bench_coreml.spec_line(spec, worker, 1, 2, machine)
    for field in ["backend", "compute_unit", "package", "rewrite", "first_load_s"]:
        assert field in line, field
    assert line["cached_load_s"] == 0.1
    assert line["phase"] == "matrix"
    assert line["backend"] == "coreml-ne"
    assert line["ne_batch_form"] == "batch-api"  # the default form at batch size 8

    rng = np.random.default_rng(0)
    reference = rng.standard_normal((5 + 196, 384)).astype(np.float32)
    broken = reference.copy()
    broken[0] = rng.standard_normal(384)  # frame cosine near 0 (risk R53: finite but wrong)
    validity = coreml.output_validity(broken, reference)
    assert validity.finite
    assert not validity.valid
    assert coreml.output_validity(reference, reference).valid

    schedule = bench_coreml.stream_schedule(3)
    assert len(schedule) == 3
    for cases in schedule:
        assert sorted(cases) == sorted(bench_coreml.STREAM_CASES)
    assert len({cases[0] for cases in schedule}) == 3


def test_stream_table_shows_each_round_ratio() -> None:
    """Finding V6 of specs/06d-neural-engine-fault/verify.html: Table 6 gives the stream ratio of
    each round next to the median, so the spread inside a run shows."""
    import bench_coreml

    lines = []
    rounds = [(1000.0, 1300.0), (1100.0, 1210.0), (900.0, 1260.0)]
    for round_, (mlx_alone, total) in enumerate(rounds, start=1):
        common = {
            "phase": "streams",
            "status": "done",
            "group": "P1-b8",
            "run_id": 1,
            "ne_batch_size": 8,
            "ne_batch_form": "batch-api",
            "round": round_,
            "model_size": "ViT-S/16",
            "side": 224,
        }
        both = {"ne": {"throughput_ips": 500.0}, "mlx": {"throughput_ips": total - 500.0}}
        lines += [
            common | {"case": "ne", "sides": {"ne": {"throughput_ips": 600.0}}},
            common | {"case": "mlx", "sides": {"mlx": {"throughput_ips": mlx_alone}}},
            common | {"case": "both", "sides": both, "total_ips": total},
        ]
    section = bench_coreml.table_text(lines).split("## 6 ")[1]
    assert "| 1.300 | 1.300, 1.100, 1.400 | 3 |" in section


def _ne_line(model_size: str, side: int, batch_size: int, form: str, ips: float, valid: bool):
    return {
        "phase": "matrix",
        "backend": "coreml-ne",
        "model_size": model_size,
        "side": side,
        "batch_size": batch_size,
        "status": "done",
        "valid": valid,
        "ne_batch_form": form,
        "package": f"x-{coreml.REWRITE_VERSION}-y",
        "patch": {"throughput_ips": ips},
    }


def test_step6d_plans() -> None:
    """The jobs of Table 3 of specs/06d-neural-engine-fault/plan.html: J1 25 packages, J2 70
    workers, J3 70 workers, J4 60 workers, J5 27 cases. The Neural Engine side of J4 and J5 takes
    the faster valid form of the matrix."""
    import bench_coreml

    assert bench_coreml.RUN == "S6d"
    assert bench_coreml.RESULTS.name == "coreml-6d.jsonl"
    assert bench_coreml.TABLE.name == "coreml-6d-table.md"
    assert bench_coreml.POINTS["P3"] == ("ViT-L/16", 224, 8, "patch")
    assert bench_coreml.mlx_mode("ViT-L/16", 224, 8) == "fp16-autocast"

    convert = bench_coreml.convert_plan()
    assert len(convert) == 25
    assert all(spec["backend"] == "convert" and spec["rewrite"] for spec in convert)
    assert len(bench_coreml.parity_plan()) == 70
    matrix = bench_coreml.matrix_plan()
    assert len(matrix) == 70
    forms = Counter(spec.get("ne_batch_form") for spec in matrix if spec["backend"] == "coreml-ne")
    assert forms == {None: 10, "batch-api": 10, "package": 10}

    lines = [
        _ne_line("ViT-S/16", 224, 8, "package", 500.0, True),
        _ne_line("ViT-S/16", 224, 8, "batch-api", 680.0, True),
        _ne_line("ViT-S/16", 224, 1, "package", 300.0, True),
        _ne_line("ViT-L/16", 224, 8, "package", 90.0, True),
        _ne_line("ViT-L/16", 224, 8, "batch-api", 120.0, False),  # invalid: never chosen
        _ne_line("ViT-L/16", 224, 1, "package", 60.0, True),
    ]
    assert bench_coreml.faster_ne_form("ViT-S/16", 224, lines) == "batch-api"
    assert bench_coreml.faster_ne_form("ViT-L/16", 224, lines) == "package"
    assert bench_coreml.stream_ne_choice("ViT-S/16", 224, lines) == (8, "batch-api")
    assert bench_coreml.stream_ne_choice("ViT-L/16", 224, lines) == (8, "package")
    assert bench_coreml.stream_ne_choice("ViT-H+/16", 512, []) == (1, "package")

    points = bench_coreml.parse_points(["P1-b8", "P1-b1", "P3-b8", "P2-b8"])
    pairs = bench_coreml.pairs_plan(points, 5, "1", lines)
    assert len(pairs) == 60
    assert {spec["backend"] for spec in pairs} == {"coreml-ne", "coreml-gpu", "mlx"}
    ne = {(s["group"], s.get("ne_batch_form")) for s in pairs if s["backend"] == "coreml-ne"}
    assert ne == {
        ("P1-b8", "batch-api"),
        ("P1-b1", None),
        ("P3-b8", "package"),
        ("P2-b8", "batch-api"),  # no matrix line: the default form
    }
    streams = bench_coreml.streams_plan(points[:1] + points[2:], 3, "1", lines)
    assert len(streams) == 27
    p3 = {(s["ne_batch_size"], s["ne_batch_form"]) for s in streams if s["priority"] == "P3"}
    assert p3 == {(8, "package")}


def test_disk_check_skips_a_worker(tmp_path: Path) -> None:
    """Below DISK_WORKER_BYTES free disk a worker does not start; its line says "disk"."""
    import bench_coreml

    from dinov3_apple.memory import GB

    def runner(spec: dict, bench_dir: Path) -> dict:
        raise AssertionError("the worker must not start")

    spec = bench_coreml.coreml_spec("convert", "convert", "ViT-S/16", 224, 1)
    results = tmp_path / "lines.jsonl"
    counts = bench_coreml.run_specs([spec], results, tmp_path, runner=runner, disk=lambda: 5 * GB)
    assert counts == {"skipped": 1}
    line = bench_coreml.read_lines(results)[0]
    assert line["reason"].startswith("disk:")
    assert line["disk_free_bytes"] == 5 * GB
    assert bench_coreml.DISK_JOB_BYTES == 10 * GB


def test_line_reason_has_no_local_folders() -> None:
    """The reason of a line of results/coreml.jsonl names the repo folders repo-relative and the
    home folder as ~ (a public repo)."""
    import bench_coreml

    from dinov3_apple.stored import REPO_ROOT

    spec = bench_coreml.coreml_spec("convert", "convert", "ViT-S/16", 224, 1, rewrite=False)
    reason = (
        f"exit code 1: {REPO_ROOT}/.venv/lib/a.py:82: SyntaxWarning\n"
        f"Error: Destination path '{Path.home()}/.cache/dinov3-apple/coreml/b.mlmodelc' exists"
    )
    machine = MachineState("AC Power", "off", "", 0, "Apple M5 Max", 0)
    line = bench_coreml.spec_line(spec, {"status": "error", "reason": reason}, 1, 2, machine)
    assert line["reason"] == (
        "exit code 1: .venv/lib/a.py:82: SyntaxWarning\n"
        "Error: Destination path '~/.cache/dinov3-apple/coreml/b.mlmodelc' exists"
    )


# T13: the _cast patch of coremltools 9.0.


class _Context:
    def __init__(self) -> None:
        self.added: dict = {}

    def add(self, res: object, name: str) -> None:
        self.added[name] = res


def _cast_constant(cast: object, monkeypatch: pytest.MonkeyPatch) -> dict:
    """Run one _cast function on the constant np.array([3]) inside a MIL program."""
    found: dict = {}

    @mb.program(input_specs=[mb.TensorSpec(shape=(1,))])
    def program(x):
        constant = mb.const(val=np.array([3], dtype=np.int32), name="c")
        monkeypatch.setattr(torch_ops, "_get_inputs", lambda *args, **kwargs: [constant])
        context = _Context()
        try:
            cast(context, types.SimpleNamespace(name="n"), int, "int32")
            found["value"] = context.added["n"].val
        except TypeError as error:
            found["error"] = error
        return x

    return found


def test_patch_coremltools_cast(monkeypatch: pytest.MonkeyPatch) -> None:
    """T13."""
    monkeypatch.setattr(torch_ops, "_cast", ORIGINAL_CAST)  # restored after the test
    ops_file = Path(torch_ops.__file__)
    digest = hashlib.sha256(ops_file.read_bytes()).hexdigest()

    assert "error" in _cast_constant(ORIGINAL_CAST, monkeypatch)  # red: numpy 2.5 refuses int()
    assert _cast_constant(coreml._cast_with_item, monkeypatch)["value"] == 3

    monkeypatch.setattr(ct, "__version__", "9.1")
    with pytest.raises(coreml.CoremltoolsVersionError):
        coreml.patch_coremltools_cast()
    assert torch_ops._cast is ORIGINAL_CAST
    monkeypatch.setattr(ct, "__version__", coreml.COREMLTOOLS_VERSION)

    coreml.patch_coremltools_cast()
    assert torch_ops._cast is coreml._cast_with_item
    coreml.patch_coremltools_cast()
    assert torch_ops._cast is coreml._cast_with_item
    assert hashlib.sha256(ops_file.read_bytes()).hexdigest() == digest


# T6 to T8: a small package (slow session).


def _small_backbone() -> torch.nn.Module:
    return coreml.rewrite_for_neural_engine(small_hf_model(seed=6, name="vits16"), 1)


@pytest.fixture(scope="module")
def small_package(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The small model vits16 at 32 px, batch size 1, converted with the patch."""
    folder = tmp_path_factory.mktemp("coreml")
    path = folder / "small.mlpackage"
    coreml.convert_module(_small_backbone(), (1, 3, SMALL_SIDE, SMALL_SIDE), path)
    return path


@slow
def test_cast_patch_converts_with_numpy_2_5(
    monkeypatch: pytest.MonkeyPatch, small_package: Path
) -> None:
    """T6."""
    monkeypatch.setattr(torch_ops, "_cast", ORIGINAL_CAST)
    traced = coreml.trace_module(_small_backbone(), (1, 3, SMALL_SIDE, SMALL_SIDE))
    with pytest.raises(TypeError, match="0-dimensional"):
        coreml.convert_traced(traced, (1, 3, SMALL_SIDE, SMALL_SIDE))
    spec = ct.utils.load_spec(str(small_package))
    assert [i.name for i in spec.description.input] == ["pixels"]
    assert [o.name for o in spec.description.output] == ["tokens"]
    assert coreml.compiled_path(small_package).exists()


@slow
@pytest.mark.parametrize("unit", ["CPU_AND_NE", "CPU_AND_GPU", "ALL"])
def test_small_package_parity_on_each_unit(small_package: Path, unit: str) -> None:
    """T7."""
    pixels = small_pixels((1, 3, SMALL_SIDE, SMALL_SIDE))
    with torch.no_grad():
        expected = small_hf_model(seed=6, name="vits16")(pixel_values=torch.from_numpy(pixels))
    expected = expected.last_hidden_state[0].numpy()
    model, first_load_s, cached_load_s = coreml.load_package(small_package, unit)
    tokens = model.predict({"pixels": pixels})["tokens"][0]
    assert np.isfinite(tokens).all()
    assert measure_parity(expected, tokens).frame >= 0.999
    assert first_load_s > 0 and cached_load_s > 0


@slow
def test_compute_plan_report(small_package: Path) -> None:
    """T8."""
    report = coreml.compute_plan_report(coreml.compiled_path(small_package), "CPU_AND_NE")
    assert sum(report["ops_per_device"].values()) == report["ops"]
    assert abs(sum(report["cost_share"].values()) - 1.0) <= 1e-6
    for row in report["off_neural_engine"]:
        assert {"position", "name", "op_type", "device", "cost"} <= set(row)
    assert 0.0 <= report["neural_engine_share"] <= 1.0


# T10 to T12: ViT-S/16 and ViT-7B/16 (slow session, gated weights).


def _package(model_size: str, rewrite: bool, side: int = 224) -> Path:
    path = coreml.package_path(model_size, side, 1, rewrite=rewrite)
    if not coreml.compiled_path(path).exists():
        coreml.convert_backbone(model_size, side, 1, rewrite=rewrite)
    return path


def _frame(path: Path, unit: str, image: str) -> float:
    model, _, _ = coreml.load_package(path, unit)
    pixels = read_inputs()[f"{image}/224x224"].astype(np.float32)
    tokens = model.predict({"pixels": pixels})["tokens"][0]
    reference, _ = read_reference("ViT-S/16")
    if not np.isfinite(tokens).all():
        return float("nan")
    return measure_parity(reference[f"{image}/224x224"], tokens).frame


@slow
def test_plain_graph_fails_on_neural_engine() -> None:
    """T10 (F1 evidence)."""
    path = _package("ViT-S/16", rewrite=False)
    ne = _frame(path, "CPU_AND_NE", "iss-kibo-lab")
    assert not ne >= 0.5  # a NaN also counts as a failure
    assert _frame(path, "CPU_AND_GPU", "iss-kibo-lab") >= 0.999


@slow
def test_vits16_package_passes_on_neural_engine() -> None:
    """T11: the fp16 thresholds of ADR 0003 for the 3 test images."""
    path = _package("ViT-S/16", rewrite=True)
    model, _, _ = coreml.load_package(path, "CPU_AND_NE")
    reference, _ = read_reference("ViT-S/16")
    inputs = read_inputs()
    threshold = THRESHOLDS["fp16-cast"]
    for image in TEST_IMAGES:
        tokens = model.predict({"pixels": inputs[f"{image}/224x224"].astype(np.float32)})["tokens"]
        parity = measure_parity(reference[f"{image}/224x224"], tokens[0])
        assert parity.frame >= threshold.frame, image
        assert parity.patch_mean >= threshold.patch_mean, image
        assert parity.patch_p01 >= threshold.patch_p01, image


@slow
def test_convert_refuses_vit7b16_before_load(monkeypatch: pytest.MonkeyPatch) -> None:
    """T12."""

    def no_load(*args: object, **kwargs: object) -> None:
        raise AssertionError("the weights must not load")

    monkeypatch.setattr(coreml, "load_torch_backbone", no_load)
    with pytest.raises(RunRuleError):
        coreml.convert_backbone("ViT-7B/16", 224, 1)


# T15 to T17: the concat fault and the ne3 packages (Step 6d, slow session).


class _Embeddings(torch.nn.Module):
    """pixels -> the tokens of the embeddings: the Hugging Face form or the ne3 form."""

    def __init__(self, embeddings: torch.nn.Module, ne3: bool) -> None:
        super().__init__()
        self.embeddings = embeddings
        self.ne3 = ne3

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        if self.ne3:
            return coreml.ne_embeddings(self.embeddings, pixels)
        return self.embeddings(pixels)


def _value(op_type: str | None, val: object = None) -> types.SimpleNamespace:
    op = None if op_type is None else types.SimpleNamespace(op_type=op_type)
    return types.SimpleNamespace(op=op, val=val)


def test_constant_concats_counts_folded_values() -> None:
    """Finding V7 of specs/06d-neural-engine-fault/verify.html: a value with val counts as a
    constant also when a non-const op makes it (for example a tile of the prefix at batch size 8
    that value inference folded); a value without val does not."""
    folded = _value("tile", np.zeros((8, 1, 4)))
    patches = _value("transpose")
    concats = [
        types.SimpleNamespace(name="hf", op_type="concat", values=[folded, folded, patches]),
        types.SimpleNamespace(name="ne3", op_type="concat", values=[folded, patches]),
        types.SimpleNamespace(name="mixed", op_type="concat", values=[_value("const", 1), folded]),
        types.SimpleNamespace(name="input", op_type="concat", values=[_value(None, 2), patches]),
        types.SimpleNamespace(name="add", op_type="add", values=[folded, folded]),
    ]
    main = types.SimpleNamespace(operations=concats)
    mlmodel = types.SimpleNamespace(_mil_program=types.SimpleNamespace(functions={"main": main}))
    assert coreml.constant_concats(mlmodel) == ["hf", "mixed"]


@slow
@pytest.mark.parametrize("batch_size", [1, 8])
def test_constant_concats_finds_the_fault(tmp_path: Path, batch_size: int) -> None:
    """T15: the Hugging Face form has one concat with two constant inputs, the ne3 form none, at
    batch size 1 and 8 (finding V7). convert_module refuses the Hugging Face form and writes no
    package."""
    coreml.patch_coremltools_cast()
    embeddings = small_hf_model(seed=15, name="vits16").embeddings
    shape = (batch_size, 3, 64, 64)
    found = {}
    for ne3 in (False, True):
        traced = coreml.trace_module(_Embeddings(embeddings, ne3), shape)
        found[ne3] = coreml.constant_concats(coreml.convert_traced(traced, shape))
    assert len(found[False]) == 1
    assert found[True] == []
    path = tmp_path / "hf-embeddings.mlpackage"
    with pytest.raises(coreml.NeuralEngineGraphError):
        coreml.convert_module(_Embeddings(embeddings, ne3=False), shape, path)
    assert not path.exists()
    assert not coreml.compiled_path(path).exists()
    # The whole ne3 backbone (RoPE, blocks, final norm) has no concat with two constant inputs.
    backbone = coreml.rewrite_for_neural_engine(small_hf_model(seed=15, name="vits16"), 1)
    traced = coreml.trace_module(backbone, shape)
    assert coreml.constant_concats(coreml.convert_traced(traced, shape)) == []


def _token_cosines(expected: np.ndarray, tokens: np.ndarray) -> np.ndarray:
    expected = expected.astype(np.float64)
    tokens = tokens.astype(np.float64)
    dots = (expected * tokens).sum(-1)
    return dots / (np.linalg.norm(expected, axis=-1) * np.linalg.norm(tokens, axis=-1))


@slow
def test_embeddings_part_on_neural_engine(tmp_path: Path) -> None:
    """T16: the ne3 embeddings of ViT-S/16 at 448 px are right on CPU_AND_NE, with the concat on
    the Neural Engine. A new macOS or Core ML release that changes the concat fault shows here."""
    from dinov3_apple.reference import load_torch_backbone

    embeddings = load_torch_backbone("ViT-S/16", "cpu", "fp32").embeddings
    shape = (1, 3, 448, 448)
    pixels = read_inputs()[f"{TEST_IMAGES[0]}/448x448"].astype(np.float32)
    with torch.no_grad():
        expected = embeddings(torch.from_numpy(pixels))[0].numpy()
    path = tmp_path / "vits16-448-embeddings-ne3.mlpackage"
    fields = coreml.convert_module(_Embeddings(embeddings, ne3=True), shape, path)
    assert fields["constant_concats"] == 0
    assert fields["ops"].get("concat") == 1
    model, _, _ = coreml.load_package(path, "CPU_AND_NE")
    tokens = model.predict({"pixels": pixels})["tokens"][0]
    cosines = _token_cosines(expected, tokens)
    assert len(cosines) == 789
    assert int((~(cosines >= 0.999)).sum()) == 0, cosines.min()  # a NaN counts as wrong
    report = coreml.compute_plan_report(coreml.compiled_path(path), "CPU_AND_NE")
    assert report["op_devices"]["concat"] == {"NeuralEngine": 1}

    # The Hugging Face form, for the test log only (probe E1 of explore.html: 788 wrong tokens).
    traced = coreml.trace_module(_Embeddings(embeddings, ne3=False), shape)
    hf_path = tmp_path / "vits16-448-embeddings-hf.mlpackage"
    coreml.convert_traced(traced, shape).save(str(hf_path))
    ct.models.utils.compile_model(str(hf_path), str(coreml.compiled_path(hf_path)))
    hf_model, _, _ = coreml.load_package(hf_path, "CPU_AND_NE")
    hf_tokens = hf_model.predict({"pixels": pixels})["tokens"][0]
    wrong = int((~(_token_cosines(expected, hf_tokens) >= 0.999)).sum())
    print(f"Hugging Face form on CPU_AND_NE: {wrong} of 789 tokens below cosine 0.999")


@slow
def test_vits16_448_package_passes_on_neural_engine() -> None:
    """T17: the J1 package vits16-448-b1-ne3 on CPU_AND_NE passes the fp16 thresholds of ADR 0003
    for the 3 test images (with ne2: frame -0.0147)."""
    path = _package("ViT-S/16", rewrite=True, side=448)
    model, _, _ = coreml.load_package(path, "CPU_AND_NE")
    reference, _ = read_reference("ViT-S/16")
    inputs = read_inputs()
    threshold = THRESHOLDS["fp16-cast"]
    for image in TEST_IMAGES:
        key = f"{image}/448x448"
        tokens = model.predict({"pixels": inputs[key].astype(np.float32)})["tokens"]
        parity = measure_parity(reference[key], tokens[0])
        assert parity.frame >= threshold.frame, image
        assert parity.patch_mean >= threshold.patch_mean, image
        assert parity.patch_p01 >= threshold.patch_p01, image
