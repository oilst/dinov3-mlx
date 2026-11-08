"""Core ML on the Apple Neural Engine (specs/06c-neural-engine/plan.html).

The Neural Engine stores the activations in fp16. The attention scores of DINOv3 reach 2.5 million,
36 times the largest fp16 value, so the plain fp16 export of the Hugging Face backbone breaks on
the Neural Engine (finding F1). The Neural Engine rewrite (Section 3.2) keeps every value inside
fp16 and is exact in real numbers:

- score scaling: ne_attention divides the scores by SCORE_SCALE before the max subtraction and
  multiplies them back after it;
- residual-stream scale: the residual stream is divided by a power of two r. The embeddings get
  1 / r, and the weights and biases of o_proj and down_proj get 1 / r. Each layer norm ε becomes
  ε / r². LayerScale stays a separate multiplication (no fold: the fold made most LayerScale
  products subnormal in fp16).

This module imports coremltools only inside the functions that need it, so the torch-only parts
(the rewrite, the package names) work without it.
"""

import shutil
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AttentionInterface, DINOv3ViTModel

from dinov3_apple.bench import (
    GIANT_MODEL_SIZE,
    TIMED,
    WARM_UP,
    TimeBudget,
    Validity,
    check_validity,
    keep_going,
    make_batch,
    summarize,
)
from dinov3_apple.footprint import read_footprint
from dinov3_apple.hub import parameter_count
from dinov3_apple.machine import read_versions
from dinov3_apple.memory import RunRuleError, check_free_memory, check_run_rule
from dinov3_apple.outputs import NUM_PREFIX_TOKENS
from dinov3_apple.parity import TEST_IMAGES
from dinov3_apple.reference import load_torch_backbone, required_load_bytes
from dinov3_apple.stored import REPO_ROOT, model_slug, read_inputs, read_reference

SCORE_SCALE = 256.0  # a power of two: the division adds no rounding error
SCORE_FLOOR = -60000.0  # the clip of the shifted scores, before the multiplication by SCORE_SCALE
NE_ATTENTION = "coreml-ne"  # the name of ne_attention in transformers.AttentionInterface
MAX_STREAM_VALUE = 32768.0  # the largest residual-stream value after the scale: half of fp16
# The version of the Neural Engine rewrite in the package name. "ne": score scale and residual-stream
# scale (first J1 and J2 of the build); "ne2": plus ε / r² (plan amendment of 2026-11-08).
REWRITE_VERSION = "ne2"
COREMLTOOLS_VERSION = "9.0"  # the version of the coreml dependency group and of the _cast patch
COREML_DIR = Path.home() / ".cache" / "dinov3-apple" / "coreml"
TOKEN_RANGE_TABLE = REPO_ROOT / "results" / "token-range-mlx.md"
# The compute-unit setting of each Core ML backend.
COMPUTE_UNITS = {"coreml-ne": "CPU_AND_NE", "coreml-gpu": "CPU_AND_GPU", "coreml-all": "ALL"}
# The thresholds of a Core ML output: fp16 compute precision with fp16 weights (ADR 0003).
THRESHOLD_MODE = "fp16-cast"


class CoremltoolsVersionError(RuntimeError):
    """The _cast patch is for coremltools 9.0 only. A new release needs a fresh check."""


# The _cast patch.


def _cast_with_item(context: Any, node: Any, dtype: type, dtype_name: str) -> None:
    """The torch op aten::Int (and bool) of coremltools 9.0, with .item() for a constant.

    coremltools 9.0 calls int() on a constant array with one value. numpy 2.5.4 (the version of
    uv.lock) refuses this: "only 0-dimensional arrays can be converted to Python scalars". Remove
    this function when a coremltools release after 9.0 fixes _cast.
    """
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.frontend.torch import ops as torch_ops

    x = torch_ops._get_inputs(context, node, expected=1)[0]
    if not (len(x.shape) == 0 or np.all([d == 1 for d in x.shape])):
        raise ValueError("input to cast must be either a scalar or a length 1 tensor")
    if x.can_be_folded_to_const():
        value = x.val.item() if isinstance(x.val, np.ndarray) else x.val
        res = x if isinstance(x.val, dtype) else mb.const(val=dtype(value), name=node.name)
    elif len(x.shape) > 0:
        res = mb.cast(x=mb.squeeze(x=x, name=node.name + "_item"), dtype=dtype_name, name=node.name)
    else:
        res = mb.cast(x=x, dtype=dtype_name, name=node.name)
    context.add(res, node.name)


def patch_coremltools_cast() -> None:
    """Point the torch frontend of coremltools at _cast_with_item, at run time.

    The installed coremltools files do not change. A second call changes nothing.

    Raises:
        CoremltoolsVersionError: For a coremltools version other than 9.0. _cast stays as it is.
    """
    import coremltools
    from coremltools.converters.mil.frontend.torch import ops as torch_ops

    if coremltools.__version__ != COREMLTOOLS_VERSION:
        raise CoremltoolsVersionError(
            f"the _cast patch is for coremltools {COREMLTOOLS_VERSION}, "
            f"not {coremltools.__version__}"
        )
    torch_ops._cast = _cast_with_item  # _int and _bool look _cast up in the module at each call


# The Neural Engine rewrite.


def ne_attention(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float | None = None,
    dropout: float = 0.0,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    """Attention whose scores fit fp16 on the Neural Engine. Exact in real numbers:
    softmax(z) = softmax((z / s - max(z / s)) * s).

    The signature is the one of eager_attention_forward of transformers. DINOv3 has no attention
    mask and no attention dropout at inference.

    Args:
        module: The attention module (not used).
        query: (B, heads, T, head dimension).
        key: (B, heads, T, head dimension).
        value: (B, heads, T, head dimension).
        attention_mask: Must be None.
        scaling: The factor on the scores. None means 1 / sqrt(head dimension).
        dropout: Must be 0.

    Returns:
        The attention output (B, T, heads, head dimension) and None for the weights.
    """
    if attention_mask is not None or dropout:
        raise ValueError("ne_attention has no attention mask and no dropout")
    if scaling is None:
        scaling = query.size(-1) ** -0.5
    s = SCORE_SCALE
    z = torch.matmul(query * (scaling / s), key.transpose(2, 3))  # at most 9,877 for ViT-S/16
    z = z - z.amax(-1, keepdim=True)  # every value at most 0
    probs = torch.softmax(torch.clamp(z, min=SCORE_FLOOR / s) * s, dim=-1)
    return torch.matmul(probs, value).transpose(1, 2).contiguous(), None


AttentionInterface.register(NE_ATTENTION, ne_attention)


class NeuralEngineBackbone(torch.nn.Module):
    """A module pixels -> tokens after the final layer norm, with the residual stream divided by r.

    It runs the parts of DINOv3ViTModel.forward one by one (embeddings, RoPE, each block, the final
    layer norm), without the output plumbing of the Hugging Face forward.

    Attributes:
        model: The Hugging Face backbone.
        residual_scale: r, a power of two.
    """

    def __init__(self, model: DINOv3ViTModel, residual_scale: int) -> None:
        super().__init__()
        self.model = model
        self.residual_scale = residual_scale

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        """Return the tokens (B, 5 + N, D) of pixels (B, 3, H, W)."""
        model = self.model
        hidden = model.embeddings(pixels)
        if self.residual_scale != 1:
            hidden = hidden * (1.0 / self.residual_scale)
        rope = model.rope_embeddings(pixels)
        for layer in model.model.layer:
            hidden = layer(hidden, position_embeddings=rope)
        return model.norm(hidden)


def rewrite_for_neural_engine(model: DINOv3ViTModel, residual_scale: int) -> NeuralEngineBackbone:
    """Rewrite a backbone in place so that its values fit fp16 (Section 3.2). No LayerScale fold.

    Args:
        model: A Hugging Face backbone in fp32.
        residual_scale: r, a power of two (residual_scale()).

    Returns:
        The module pixels -> tokens. In fp32 it gives the tokens of the Hugging Face backbone: each
        layer norm sees the tokens divided by r, and its ε becomes ε / r² (plan amendment of
        2026-11-08), so it gives its old output.

    Raises:
        ValueError: If residual_scale is not a power of two.
    """
    if residual_scale < 1 or residual_scale & (residual_scale - 1):
        raise ValueError(f"the residual scale must be a power of two, not {residual_scale}")
    for module in model.modules():  # the layers share model.config; set each config to be safe
        config = getattr(module, "config", None)
        if config is not None:
            config._attn_implementation = NE_ATTENTION
        if isinstance(module, torch.nn.LayerNorm):
            module.eps = module.eps / residual_scale**2
    with torch.no_grad():
        for layer in model.model.layer:
            for proj in (layer.attention.o_proj, layer.mlp.down_proj):
                proj.weight.mul_(1.0 / residual_scale)
                if proj.bias is not None:
                    proj.bias.mul_(1.0 / residual_scale)
    return NeuralEngineBackbone(model, residual_scale)


def largest_token_value(model_size: str, table: Path = TOKEN_RANGE_TABLE) -> float:
    """Return the largest token value of the fp32 row of one model size in Table 1 of the
    token-range table."""
    for row in table.read_text().splitlines():
        cells = [cell.strip() for cell in row.strip().strip("|").split("|")]
        if len(cells) < 3 or cells[0] != model_size or cells[1] != "fp32":
            continue
        try:
            return float(cells[2].replace(",", ""))
        except ValueError:
            continue  # a row of Section 2: the third cell is the configuration
    raise KeyError(f"no fp32 row of {model_size} in {table}")


def residual_scale(model_size: str, table: Path = TOKEN_RANGE_TABLE) -> int:
    """Return the smallest power of two r with (largest token value) / r <= 32,768.

    Args:
        model_size: A key of MODEL_IDS.
        table: The token-range table (results/token-range-mlx.md).

    Returns:
        r: 1 for ViT-S/16 to ViT-B/16, 8 for ViT-L/16 and ViT-H+/16 with the table of Step 5.

    Raises:
        RunRuleError: For ViT-7B/16 (deferred, D6 of explore.html): the conversion traces the fp32
            forward pass of the whole model.
    """
    if model_size == GIANT_MODEL_SIZE:
        raise RunRuleError(f"{model_size}: the Core ML conversion traces fp32 weights (D6)")
    value = largest_token_value(model_size, table)
    scale = 1
    while value / scale > MAX_STREAM_VALUE:
        scale *= 2
    return scale


# Packages.


def package_path(model_size: str, side: int, batch_size: int, rewrite: bool = True) -> Path:
    """Return the path of one package. The name holds every setting, so a stale package never
    serves a new configuration.

    Args:
        model_size: A key of MODEL_IDS.
        side: The square input resolution.
        batch_size: The fixed batch size of the package.
        rewrite: True for the Neural Engine rewrite, False for the plain graph (F1 evidence).

    Returns:
        For example ~/.cache/dinov3-apple/coreml/vits16-224-b8-ne2-s256-r1-ct9.0.mlpackage. After
        convert_backbone only its compiled model (compiled_path) stays on disk.
    """
    stem = f"{model_slug(model_size)}-{side}-b{batch_size}"
    if rewrite:
        stem += f"-{REWRITE_VERSION}-s{SCORE_SCALE:g}-r{residual_scale(model_size)}"
    else:
        stem += "-plain"
    return COREML_DIR / f"{stem}-ct{COREMLTOOLS_VERSION}.mlpackage"


def compiled_path(package: Path) -> Path:
    """Return the path of the compiled model (.mlmodelc) next to a package."""
    return package.with_suffix(".mlmodelc")


def trace_module(module: torch.nn.Module, shape: tuple[int, ...]) -> torch.jit.ScriptModule:
    """Trace a module pixels -> tokens with a zero batch of one shape on torch-cpu fp32."""
    example = torch.zeros(shape)
    # run_torch runs in inference mode; inference tensors can break a trace (explore trap 5).
    with torch.inference_mode(False), torch.no_grad():
        return torch.jit.trace(module.eval(), example)


def convert_traced(traced: torch.jit.ScriptModule, shape: tuple[int, ...]) -> Any:
    """Convert a traced module to an ML Program with fp16 compute precision.

    Input "pixels" fp32 (B, 3, H, W), output "tokens" fp32 (B, 5 + N, D), macOS 15.
    """
    import coremltools as ct

    return ct.convert(
        traced,
        inputs=[ct.TensorType(name="pixels", shape=shape, dtype=np.float32)],
        outputs=[ct.TensorType(name="tokens", dtype=np.float32)],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS15,
    )


def _op_counts(mlmodel: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for op in mlmodel._mil_program.functions["main"].operations:
        counts[op.op_type] = counts.get(op.op_type, 0) + 1
    return counts


def _size_bytes(path: Path) -> int:
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def convert_module(module: torch.nn.Module, shape: tuple[int, ...], path: Path) -> dict:
    """Trace, convert, save and compile one module with the _cast patch.

    Args:
        module: A module pixels -> tokens on torch-cpu fp32.
        shape: The fixed input shape (B, 3, H, W).
        path: The .mlpackage path. The .mlmodelc goes next to it.

    Returns:
        The times of the trace, the conversion, the save and the compile, the package size and
        the op counts.
    """
    import coremltools as ct

    patch_coremltools_cast()
    start = time.perf_counter()
    traced = trace_module(module, shape)
    trace_s = time.perf_counter() - start
    start = time.perf_counter()
    mlmodel = convert_traced(traced, shape)
    convert_s = time.perf_counter() - start
    path.parent.mkdir(parents=True, exist_ok=True)
    for old in (path, compiled_path(path)):  # compile_model does not overwrite a .mlmodelc
        if old.exists():
            shutil.rmtree(old)
    start = time.perf_counter()
    mlmodel.save(str(path))
    save_s = time.perf_counter() - start
    start = time.perf_counter()
    ct.models.utils.compile_model(str(path), str(compiled_path(path)))
    compile_s = time.perf_counter() - start
    return {
        "trace_s": trace_s,
        "convert_s": convert_s,
        "save_s": save_s,
        "compile_s": compile_s,
        "package_bytes": _size_bytes(path),
        "ops": _op_counts(mlmodel),
    }


def convert_need_bytes(model_size: str) -> int:
    """Return the memory need of one conversion: the fp32 torch load plus 2 times the fp32
    weights for the trace and the MIL program."""
    return required_load_bytes(model_size, torch.float32) + 2 * 4 * parameter_count(model_size)


def convert_backbone(model_size: str, side: int, batch_size: int, rewrite: bool = True) -> dict:
    """Convert the backbone of one model size into one package (Section 3.3).

    The run rule and a memory check come first, then the fp32 load on torch-cpu, the rewrite, the
    trace and the conversion.

    Args:
        model_size: A key of MODEL_IDS.
        side: The square input resolution.
        batch_size: The fixed batch size.
        rewrite: True for the Neural Engine rewrite, False for the plain graph.

    Returns:
        The fields of convert_module plus the package name, the rewrite and the lifetime peak of
        the process footprint. Only the compiled model stays on disk: the .mlpackage holds the
        same weights again (ViT-H+/16: 1.68 GB), and the workers load the .mlmodelc.

    Raises:
        RunRuleError: For ViT-7B/16, before any weight read.
        InsufficientMemoryError: If the memory check fails.
    """
    label = f"{model_size} Core ML conversion {side}px b{batch_size}"
    check_run_rule(model_size, 4, label)
    scale = residual_scale(model_size) if rewrite else 1
    path = package_path(model_size, side, batch_size, rewrite)
    check_free_memory(convert_need_bytes(model_size), label)
    start = time.perf_counter()
    model = load_torch_backbone(model_size, "cpu", "fp32")
    load_s = time.perf_counter() - start
    module = rewrite_for_neural_engine(model, scale) if rewrite else NeuralEngineBackbone(model, 1)
    fields = convert_module(module, (batch_size, 3, side, side), path)
    shutil.rmtree(path)  # the workers load the .mlmodelc; the package would hold the weights twice
    return {
        "package": path.name,
        "rewrite": rewrite_fields(model_size) if rewrite else None,
        "load_s": load_s,
        **fields,
        "footprint_peak_bytes": read_footprint().lifetime_max_bytes,
    }


def rewrite_fields(model_size: str) -> dict[str, float | int]:
    """Return the field "rewrite" of a result line: the score scale and the residual scale."""
    return {"score_scale": SCORE_SCALE, "residual_scale": residual_scale(model_size)}


# Running a package.


def load_package(path: Path, compute_unit: str) -> tuple[Any, float, float]:
    """Load the compiled model of a package twice and return the second one with both load times.

    The first load compiles for the Neural Engine (Core ML caches it); the second load is the
    cached load.

    Args:
        path: The .mlpackage path. Its .mlmodelc must exist (convert_module).
        compute_unit: "CPU_AND_NE", "CPU_AND_GPU", "ALL" or "CPU_ONLY".

    Returns:
        The CompiledMLModel, the first load time and the cached load time in seconds.
    """
    import coremltools as ct

    compiled = compiled_path(path)
    if not compiled.exists():
        raise FileNotFoundError(f"no compiled model {compiled}: run the conversion first")
    unit = getattr(ct.ComputeUnit, compute_unit)
    start = time.perf_counter()
    model = ct.models.CompiledMLModel(str(compiled), compute_units=unit)
    first_load_s = time.perf_counter() - start
    del model
    start = time.perf_counter()
    model = ct.models.CompiledMLModel(str(compiled), compute_units=unit)
    return model, first_load_s, time.perf_counter() - start


def output_validity(tokens: np.ndarray, reference: np.ndarray | None) -> Validity:
    """Check the tokens (5 + N, D) of one image with the fp16 thresholds (ADR 0003).

    A wrong Neural Engine output is finite (risk R53), so the parity decides.
    """
    tokens = np.asarray(tokens, dtype=np.float32)
    return check_validity(tokens[0], tokens[NUM_PREFIX_TOKENS:], reference, THRESHOLD_MODE)


def _worst(parities: list[dict]) -> dict:
    return {name: min(p[name] for p in parities) for name in parities[0]}


def run_coreml(
    model_size: str,
    side: int,
    batch_size: int,
    backend: str,
    rewrite: bool = True,
    parity_only: bool = False,
    batch_api: bool = False,
    warm_up: TimeBudget = WARM_UP,
    timed: TimeBudget = TIMED,
) -> dict:
    """Run one package on one compute unit: load twice, check the first output, time the batches.

    With parity_only (batch size 1) the worker checks the 3 test images and times no batch. Else
    it fills a batch with the stored pixels of the 3 test images, checks image 0 of the first
    output, warms up and times the batches (Section 3.4). model.predict returns NumPy arrays, so
    the output is on the host: the frame time and the patch time are the same call.

    With batch_api the worker runs the batch-size-1 package with one call on a list of batch_size
    inputs (the batch API of Core ML). At batch size 8 the Neural Engine computes the patch
    embedding of the batch-size-8 package wrong (finding F4 of explore.html).

    Args:
        model_size: A key of MODEL_IDS.
        side: The square input resolution.
        batch_size: The batch size of the package.
        backend: A key of COMPUTE_UNITS.
        rewrite: True for the Neural Engine rewrite, False for the plain graph.
        parity_only: True for the parity of the 3 test images only.
        batch_api: True to run the batch-size-1 package with the batch API.
        warm_up: The time budget of the warm-up batches.
        timed: The time budget of the timed batches.

    Returns:
        The worker fields of a result line with status "done", plus the fields of Table 3.
    """
    import coremltools as ct

    start_time = datetime.now(UTC).isoformat(timespec="seconds")
    base = read_footprint()
    compute_unit = COMPUTE_UNITS[backend]
    path = package_path(model_size, side, 1 if batch_api else batch_size, rewrite)
    model, first_load_s, cached_load_s = load_package(path, compute_unit)
    load_peak = read_footprint().lifetime_max_bytes
    reference, _ = read_reference(model_size)
    inputs = read_inputs()
    line: dict = {
        "status": "done",
        "reason": "",
        "compute_unit": compute_unit,
        "package": path.name,
        "rewrite": rewrite_fields(model_size) if rewrite else None,
        "first_load_s": first_load_s,
        "cached_load_s": cached_load_s,
        "load_time_s": first_load_s,
        "ne_batch_form": "batch-api" if batch_api else "package",
    }
    if parity_only:
        start = time.perf_counter()
        images = {}
        for image in TEST_IMAGES:
            key = f"{image}/{side}x{side}"
            pixels = inputs[key].astype(np.float32)
            tokens = model.predict({"pixels": pixels})["tokens"][0]
            validity = output_validity(tokens, reference[key])
            images[image] = {
                "valid": validity.valid,
                "finite": validity.finite,
                "parity": None if validity.parity is None else validity.parity.__dict__,
                "note": validity.note,
                "max_abs": float(np.nanmax(np.abs(tokens))),
            }
        line["first_call_s"] = time.perf_counter() - start
        parities = [v["parity"] for v in images.values() if v["parity"] is not None]
        line |= {
            "valid": all(v["valid"] for v in images.values()),
            "finite": all(v["finite"] for v in images.values()),
            "parity": _worst(parities) if len(parities) == len(images) else None,
            "validity_note": "; ".join(f"{name}: {v['note']}" for name, v in images.items()),
            "images": images,
        }
    else:
        batch = make_batch(inputs, side, batch_size)
        if batch_api:  # one call with a list of batch-size-1 inputs (F4)
            feed: Any = [{"pixels": batch[i : i + 1]} for i in range(batch_size)]
        else:
            feed = {"pixels": batch}
        start = time.perf_counter()
        result = model.predict(feed)
        tokens = result[0]["tokens"] if batch_api else result["tokens"]
        line["first_call_s"] = time.perf_counter() - start
        key = f"{TEST_IMAGES[0]}/{side}x{side}"
        validity = output_validity(tokens[0], reference.get(key))
        line |= {
            "valid": validity.valid,
            "finite": validity.finite,
            "parity": None if validity.parity is None else validity.parity.__dict__,
            "validity_note": validity.note,
        }
        warm = _time_calls(lambda: model.predict(feed), warm_up)
        times = _time_calls(lambda: model.predict(feed), timed)
        line |= {
            "warm_up_batches": len(warm),
            "warm_up_s": sum(warm),
            "timed_batches": len(times),
            "timed_s": sum(times),
            "frame": summarize(times, batch_size),
            "patch": summarize(times, batch_size),
        }
    end = read_footprint()
    line |= {
        "footprint_base_bytes": base.current_bytes,
        "load_footprint_peak_bytes": load_peak,
        "peak_bytes": end.lifetime_max_bytes,
        "versions": read_versions() | {"coremltools": ct.__version__},
        "attn_implementation": NE_ATTENTION if rewrite else "sdpa",
        "start_time": start_time,
        "time": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    return line


def _time_calls(call: Any, budget: TimeBudget) -> list[float]:
    """Run calls until the time budget ends and return the time of each call."""
    times: list[float] = []
    while keep_going(budget, len(times), sum(times)):
        start = time.perf_counter()
        call()
        times.append(time.perf_counter() - start)
    return times


# The compute plan.


def _device_name(device: Any) -> str:
    """ "MLNeuralEngineComputeDevice" -> "NeuralEngine", "MLGPUComputeDevice" -> "GPU"."""
    return type(device).__name__.removeprefix("ML").removesuffix("ComputeDevice")


def compute_plan_report(compiled: Path, compute_unit: str = "CPU_AND_NE") -> dict:
    """Return the compute plan of one compiled model: the device of each op and its estimated cost.

    Args:
        compiled: The .mlmodelc path. MLComputePlan needs the compiled model.
        compute_unit: The compute-unit setting.

    Returns:
        ops (the ops with a device: the constants have none), constants, ops_per_device,
        cost_share (the estimated cost share of each device, sum 1), neural_engine_share and
        off_neural_engine: position, name, op type, device, supported devices and cost of each op
        that does not run on the Neural Engine.
    """
    import coremltools as ct
    from coremltools.models.compute_plan import MLComputePlan

    plan = MLComputePlan.load_from_path(
        str(compiled), compute_units=getattr(ct.ComputeUnit, compute_unit)
    )
    operations = plan.model_structure.program.functions["main"].block.operations
    per_device: dict[str, int] = {}
    cost: dict[str, float] = {}
    off: list[dict] = []
    constants = 0
    for position, op in enumerate(operations):
        usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
        if usage is None:
            constants += 1
            continue
        device = _device_name(usage.preferred_compute_device)
        per_device[device] = per_device.get(device, 0) + 1
        estimate = plan.get_estimated_cost_for_mlprogram_operation(op)
        weight = 0.0 if estimate is None else float(estimate.weight)
        cost[device] = cost.get(device, 0.0) + weight
        if device != "NeuralEngine":
            off.append(
                {
                    "position": position,
                    "name": op.outputs[0].name if op.outputs else "",
                    "op_type": op.operator_name,
                    "device": device,
                    "supported": sorted(_device_name(d) for d in usage.supported_compute_devices),
                    "cost": weight,
                }
            )
    total = sum(cost.values())
    share = {device: value / total for device, value in cost.items()} if total else {}
    total_off = sum(row["cost"] for row in off)
    for row in off:
        row["cost"] = row["cost"] / total if total else 0.0
    return {
        "compute_unit": compute_unit,
        "ops": sum(per_device.values()),
        "constants": constants,
        "ops_per_device": per_device,
        "cost_share": share,
        "neural_engine_share": share.get("NeuralEngine", 0.0),
        "off_neural_engine_cost": total_off / total if total else 0.0,
        "off_neural_engine": off,
    }
