"""Tests T1 to T13 of specs/06c-neural-engine/plan.html: Core ML on the Neural Engine.

T1 to T5, T9 and T13 use synthetic tensors or the small models of small_model.py in torch only and
run in the default session. Every test that converts or loads a Core ML package (T6 to T8, T10 to
T12) runs in the slow session (DINOV3_SLOW=1).
"""

import copy
import hashlib
import types
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
    assert path.name == f"vitl16-512-b8-ne2-s256-r8-ct{coreml.COREMLTOOLS_VERSION}.mlpackage"
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
    assert line["ne_batch_form"] == "batch-api"  # F4: batch size 8 on the batch-size-1 package
    pairs = bench_coreml.pairs_plan(["P1"], [8], 1, "1")
    assert [p.get("ne_batch_form") for p in pairs if p["group"] == "P1-b8-f4"] == ["package"]

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


def _package(model_size: str, rewrite: bool) -> Path:
    path = coreml.package_path(model_size, 224, 1, rewrite=rewrite)
    if not coreml.compiled_path(path).exists():
        coreml.convert_backbone(model_size, 224, 1, rewrite=rewrite)
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
