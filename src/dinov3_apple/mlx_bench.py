"""The mlx workers of the benchmarks: one configuration in one process (Section 3.1 of
specs/06-speed/plan.html).

scripts/bench_mlx_port.py (Steps 4 and 5) and scripts/bench_speed.py (Step 6) share these
workers. The functions import mlx.core, mlx_model and mlx_weights only inside their bodies, so a
driver that imports this module imports no mlx (T23). No torch import.

The worker runs the compiled forward pass for a configuration with compile=True (D1), raises the
MLX limit from the load need to the benchmark need after the first call (D8), and writes the mlx
fields of Table 8 of the plan.
"""

import json
import os
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from dinov3_apple.bench import (
    SIDES,
    TIMED,
    WARM_UP,
    BenchConfig,
    TimeBudget,
    check_validity,
    keep_going,
    make_batch,
    reference_tokens,
    required_bench_bytes,
    summarize,
)
from dinov3_apple.footprint import read_footprint
from dinov3_apple.machine import read_versions
from dinov3_apple.outputs import NUM_PREFIX_TOKENS
from dinov3_apple.stored import model_slug, read_inputs

if TYPE_CHECKING:  # the module imports mlx only inside its functions
    import mlx.core as mx

    from dinov3_apple.mlx_model import Block

Forward = Callable[["mx.array"], "mx.array"]


def output_transfer(frame: np.ndarray, patches: np.ndarray) -> str:
    """Return how an output moved to the host.

    Returns:
        "view-uint16" for bf16 bits (the lines of Step 5), "fp16-patches" for fp32 frame
        embeddings with fp16 patch embeddings (D13 of specs/06-speed/plan.html), else "native".
    """
    if frame.dtype == np.uint16:
        return "view-uint16"
    if frame.dtype == np.float32 and patches.dtype == np.float16:
        return "fp16-patches"
    return "native"


def host_to_fp32(host: np.ndarray) -> np.ndarray:
    """Return the fp32 values of a host output. bf16 bits (uint16) become fp32; others stay."""
    if host.dtype != np.uint16:
        return host
    return (host.astype(np.uint32) << 16).view(np.float32)


def time_batch_mlx(
    forward: Forward, batch: np.ndarray, dtype_mode: str, patch_fp32: bool = False
) -> tuple[tuple[float, float, float], np.ndarray, np.ndarray]:
    """Run one batch host to host on mlx and time it (Table 4 of the Step 3 plan).

    The frame embeddings move to the host in fp32. The patch embeddings move in the dtype of
    patch_dtype: fp16 in the four fp16 and bf16 dtype modes (D13), fp32 in fp32. The casts run on
    the GPU, after the device time.

    Args:
        forward: The compiled forward pass of compile_backbone, or the backbone (eager).
        batch: fp32 pixels with shape (B, 3, H, W) in NumPy. The model casts them.
        dtype_mode: The dtype mode of the backbone.
        patch_fp32: True for fp32 patch embeddings in every dtype mode.

    Returns:
        The device time, the frame time and the patch time in seconds, then the frame embeddings
        (B, D) and the patch embeddings (B, N, D) in NumPy.
    """
    import mlx.core as mx

    from dinov3_apple.mlx_model import patch_dtype

    t0 = time.perf_counter()
    tokens = forward(mx.array(batch))
    mx.eval(tokens)  # mlx synchronizes with mx.eval
    t1 = time.perf_counter()
    frame = np.array(tokens[:, 0].astype(mx.float32))
    t2 = time.perf_counter()
    patches = np.array(tokens[:, NUM_PREFIX_TOKENS:].astype(patch_dtype(dtype_mode, patch_fp32)))
    t3 = time.perf_counter()
    return (t1 - t0, t2 - t0, (t1 - t0) + (t3 - t2)), frame, patches


def phase(
    run: Callable[[], tuple[float, float, float]], budget: TimeBudget
) -> list[tuple[float, float, float]]:
    """Run batches until the time budget ends. The elapsed time is the sum of the patch times."""
    times: list[tuple[float, float, float]] = []
    elapsed = 0.0
    while keep_going(budget, len(times), elapsed):
        batch_times = run()
        times.append(batch_times)
        elapsed += batch_times[2]
    return times


def load_report_fields(load_report: object | None) -> dict[str, int | None]:
    """Return the fields of the load report of an mlx worker, or null fields without a report."""
    return {
        f"load_report_{name}": getattr(load_report, name, None)
        for name in ("need_bytes", "peak_bytes", "cache_bytes", "footprint_bytes")
    }


def batch_file_name(config: BenchConfig, run_id: str | None = None) -> str:
    """Return the name of the per-batch file of one worker: the model slug, the dtype mode, the
    input resolution, the batch size, the compile flag, bits, the repeat and the run id."""
    name = f"{model_slug(config.model_size)}_{config.dtype_mode}_{config.side}_b{config.batch_size}"
    name += "_compile" if config.compile else ""
    name += f"_{config.bits}bit" if config.bits else ""
    name += f"_r{config.repeat}"
    name += f"_run{run_id}" if run_id else ""
    return f"{name}.json"


def run_mlx_worker(
    config: BenchConfig,
    warm_up: TimeBudget = WARM_UP,
    timed: TimeBudget = TIMED,
    bench_dir: Path = Path("."),
    run_id: str | None = None,
    batch: np.ndarray | None = None,
) -> dict:
    """Load the MLX backbone, check its first output, warm up, time the batches, return the result.

    The same phases and fields as torch_bench.run_worker of Step 3, plus the mlx fields of Table 8
    of specs/06-speed/plan.html. TF32 is off (D7 of Step 4). With config.compile the worker runs
    the compiled forward pass (D1); the worker holds it, not the model (R36).

    Args:
        config: The configuration. bits loads quantized block weights (D22).
        warm_up: The time budget of the warm-up batches.
        timed: The time budget of the timed batches.
        bench_dir: The folder for the per-batch times.
        run_id: The run of a paired run, for the name of the per-batch file.
        batch: fp32 pixels (B, 3, H, W). None fills a batch with the stored pixels. Above 512 px
            the check is finite only: no reference output exists.

    Returns:
        The worker fields of LINE_FIELDS with status "done", plus the mlx fields of MLX_FIELDS and
        batch_file.

    Raises:
        TF32ModeError: If MLX_ENABLE_TF32 is not 0 in the environment.
        InsufficientMemoryError: If the memory check of the load fails.
        WeightsNotFoundError: If the weights are not on this Mac.
    """
    import mlx.core as mx

    from dinov3_apple import mlx_weights
    from dinov3_apple.mlx_model import compile_backbone, patch_dtype

    start_time = datetime.now(UTC).isoformat(timespec="seconds")
    base = read_footprint()
    start = time.perf_counter()
    kwargs: dict[str, Any] = {"bits": config.bits} if config.bits else {}
    model = mlx_weights.load_mlx_backbone(
        config.model_size, config.dtype_mode, tf32=False, **kwargs
    )
    load_time_s = time.perf_counter() - start
    load_footprint_peak = read_footprint().lifetime_max_bytes
    forward: Forward = compile_backbone(model) if config.compile else model

    if batch is None:
        batch = make_batch(read_inputs(), config.side, config.batch_size)
    start = time.perf_counter()
    tokens = forward(mx.array(batch))  # the trace: the compiled call before mx.eval (R38)
    compile_trace_s = time.perf_counter() - start
    mx.eval(tokens)
    frame = np.array(tokens[:, 0].astype(mx.float32))
    patches = np.array(tokens[:, NUM_PREFIX_TOKENS:].astype(patch_dtype(config.dtype_mode)))
    first_call_s = time.perf_counter() - start
    del tokens
    transfer = output_transfer(frame, patches)
    output_dtype = f"{frame.dtype}/{patches.dtype}"
    reference = reference_tokens(config.model_size, config.side) if config.side in SIDES else None
    validity = check_validity(frame[0], patches[0], reference, config.dtype_mode)
    del frame, patches

    need = required_bench_bytes(config, backend="mlx")
    load_limit = mx.set_memory_limit(need)  # returns the previous limit: the load need (D8)
    warm = phase(lambda: time_batch_mlx(forward, batch, config.dtype_mode)[0], warm_up)
    mlx_peak = mx.get_peak_memory()  # the Step 5 meaning: no reset after the load
    mx.reset_peak_memory()
    timed_times = phase(lambda: time_batch_mlx(forward, batch, config.dtype_mode)[0], timed)
    timed_peak = mx.get_peak_memory()
    end = read_footprint()

    device, frame_t, patch_t = (list(column) for column in zip(*timed_times, strict=True))
    bench_dir.mkdir(parents=True, exist_ok=True)
    name = batch_file_name(config, run_id)
    (bench_dir / name).write_text(
        json.dumps({"config": config.key, "device": device, "frame": frame_t, "patch": patch_t})
    )
    above_load = end.lifetime_max_bytes > load_footprint_peak
    load_report = getattr(model, "load_report", None)
    return {
        "status": "done",
        "reason": "",
        "valid": validity.valid,
        "finite": validity.finite,
        "parity": None if validity.parity is None else validity.parity.__dict__,
        "validity_note": validity.note,
        "load_time_s": load_time_s,
        "first_call_s": first_call_s,
        "compile_trace_s": compile_trace_s,
        "warm_up_batches": len(warm),
        "warm_up_s": sum(t[2] for t in warm),
        "timed_batches": len(timed_times),
        "timed_s": sum(patch_t),
        "frame": summarize(frame_t, config.batch_size),
        "patch": summarize(patch_t, config.batch_size),
        "device": summarize(device, config.batch_size),
        "output_dtype": output_dtype,
        "footprint_base_bytes": base.current_bytes,
        "load_footprint_peak_bytes": load_footprint_peak,
        "peak_bytes": end.lifetime_max_bytes,
        "run_peak_bytes": end.lifetime_max_bytes if above_load else None,
        "run_peak_note": "" if above_load else f"below the load peak; end {end.current_bytes}",
        "mps_driver_max_bytes": None,
        "versions": read_versions(),
        "attn_implementation": "mx.fast.scaled_dot_product_attention",
        "start_time": start_time,
        "time": datetime.now(UTC).isoformat(timespec="seconds"),
        "mlx_architecture": mx.device_info()["architecture"],
        "mlx_disable_compile": os.environ.get("MLX_DISABLE_COMPILE"),
        "mlx_load_limit_bytes": load_limit,
        "mlx_memory_limit_bytes": need,
        "mlx_peak_bytes": mlx_peak,
        "mlx_timed_peak_bytes": timed_peak,
        "mlx_cache_bytes": mx.get_cache_memory(),
        "tf32": False,
        "output_transfer": transfer,
        "batch_file": name,
        **load_report_fields(load_report),
    }


def mlx_block_parts(block: "Block") -> dict[str, Callable[..., tuple["mx.array", ...]]]:
    """Return the 11 parts of one MLX block, with the names, inputs and outputs of block_parts.

    The P1 profile (Table 12 of specs/06-speed/plan.html) and the part tests use it. The rope part
    takes the tables of the block: (T, head_dim) or the layout of the concat-free rotation.
    """
    import mlx.core as mx

    from dinov3_apple import mlx_model

    attention, mlp = block.attention, block.mlp

    def ff_in(h: mx.array) -> tuple[mx.array, ...]:
        """(gate, up) for SwiGLU, (up,) for the MLP, as ff_in of block_parts."""
        if isinstance(mlp, mlx_model.SwiGLU):
            return mlp.gate_proj(h), mlp.up_proj(h)
        return (mlp.up_proj(h),)

    def rope(q: mx.array, k: mx.array, cos: mx.array, sin: mx.array) -> tuple[mx.array, ...]:
        return mlx_model.apply_rope(q, cos, sin), mlx_model.apply_rope(k, cos, sin)

    return {
        "norm1": lambda x: (block.norm1(x),),
        "qkv_proj": lambda h: attention.qkv(h),
        "rope": rope,
        "sdpa": lambda q, k, v: (
            mx.fast.scaled_dot_product_attention(q, k, v, scale=attention.scale, mask=None),
        ),
        "o_proj": lambda a: (attention.output(a),),
        "layer_scale1": lambda o, x: (block.layer_scale1(o) + x,),
        "norm2": lambda x: (block.norm2(x),),
        "ff_in": ff_in,
        "activation": lambda *values: (mlp.activation(*values),),
        "ff_out": lambda g: (mlp.down_proj(g),),
        "layer_scale2": lambda y, x: (block.layer_scale2(y) + x,),
    }


# The change ratios of Table 23 of specs/06-speed/plan.html: both forms of one configuration in one
# process, in alternation and in a rotated order.


def _apply_rope_step5(x: "mx.array", cos: "mx.array", sin: "mx.array") -> "mx.array":
    """The RoPE rotation of Step 5 (rotate_half and a join), with tables (T, head_dim): form A of
    C2 and the RoPE of the forms E and A of C1 (R56)."""
    import mlx.core as mx

    half = x.shape[-1] // 2
    xf = x.astype(mx.float32)
    rotated = mx.concatenate([-xf[..., half:], xf[..., :half]], axis=-1)
    return (xf * cos + rotated * sin).astype(x.dtype)


class _Step5Rope:
    """A context that runs the attention of mlx_model with the Step 5 rotation."""

    def __enter__(self) -> None:
        from dinov3_apple import mlx_model

        self.saved = mlx_model.apply_rope
        mlx_model.apply_rope = _apply_rope_step5

    def __exit__(self, *args: object) -> None:
        from dinov3_apple import mlx_model

        mlx_model.apply_rope = self.saved


def change_forms(model: Any, forms: list[str], side: int) -> dict[str, tuple[Forward, bool]]:
    """Return the forward function of each form of a change and its patch_fp32 flag.

    E: the eager forward pass with the Step 5 RoPE form. A: the compiled forward pass with the
    Step 5 RoPE form (its tables (T, head_dim) as arguments). R: the compiled forward pass of
    compile_backbone (the concat-free RoPE). R-fp32 and R-fp16: R with fp32 or fp16 patch
    embeddings (C4).
    """
    import mlx.core as mx

    from dinov3_apple.mlx_model import compile_backbone, rope_tables

    grid = side // model.config.patch_size
    cos, sin = rope_tables(grid, grid, model.config.head_dim, model.config.rope_theta)
    cos5, sin5 = mx.array(cos), mx.array(sin)
    traced5 = mx.compile(lambda p, c, s: model.forward_with_rope(p, c, s), inputs=model.state)

    def eager(pixels: mx.array) -> mx.array:
        with _Step5Rope():
            return model.forward_with_rope(pixels, cos5, sin5)

    def compiled5(pixels: mx.array) -> mx.array:
        with _Step5Rope():  # a trace of a new shape sees the Step 5 rotation
            return traced5(pixels, cos5, sin5)

    compiled = compile_backbone(model)
    known = {
        "E": (eager, False),
        "A": (compiled5, False),
        "R": (compiled, False),
        "R-fp32": (compiled, True),
        "R-fp16": (compiled, False),
    }
    return {form: known[form] for form in forms}


def run_change_worker(
    config: BenchConfig,
    change: str,
    forms: list[str],
    rounds: int,
    warm_up: TimeBudget,
    timed: TimeBudget,
) -> dict:
    """Run the forms of one change ratio in one process (C1, C2, C4 of Table 23).

    Before the timed batches the worker checks that the forms give the same tokens bit for bit
    (R56); a difference stops the process. C4 compares the frame embeddings bit for bit and the
    patch embeddings after the fp16 cast. Each round runs each form with its own warm-up and its
    timed batches; the order of the forms rotates from round to round.

    Returns:
        The throughput of each form in each round, the ratio of the second form to the first form
        in each round, its median, minimum and maximum, for the frame and the patch output.

    Raises:
        AssertionError: If the forms differ where Table 23 asks for bit equality.
    """
    import mlx.core as mx

    from dinov3_apple import mlx_weights

    model = mlx_weights.load_mlx_backbone(config.model_size, config.dtype_mode, tf32=False)
    batch = make_batch(read_inputs(), config.side, config.batch_size)
    functions = change_forms(model, forms, config.side)
    first = {}
    for form, (forward, patch_fp32) in functions.items():
        frame, patches = time_batch_mlx(forward, batch, config.dtype_mode, patch_fp32)[1:]
        first[form] = (frame, patches)
    base_frame, base_patches = first[forms[0]]
    other_frame, other_patches = first[forms[1]]
    frame_equal = bool(np.array_equal(base_frame, other_frame))
    patches_equal = bool(
        np.array_equal(base_patches.astype(other_patches.dtype), other_patches, equal_nan=True)
    )
    if change != "C4" and not (frame_equal and patches_equal):
        raise AssertionError(f"{change}: the forms {forms} differ on the first batch (R56)")
    if change == "C4" and not frame_equal:
        raise AssertionError("C4: the frame embeddings differ")
    throughput: dict[str, dict[str, list[float]]] = {
        form: {"frame": [], "patch": [], "device": []} for form in forms
    }
    for r in range(rounds):
        order = forms[r % len(forms) :] + forms[: r % len(forms)]
        for form in order:
            forward, patch_fp32 = functions[form]

            def run(forward: Forward = forward, patch_fp32: bool = patch_fp32) -> tuple:
                return time_batch_mlx(forward, batch, config.dtype_mode, patch_fp32)[0]

            phase(run, warm_up)
            times = phase(run, timed)
            device, frame_t, patch_t = (list(column) for column in zip(*times, strict=True))
            for name, values in (("device", device), ("frame", frame_t), ("patch", patch_t)):
                throughput[form][name].append(
                    summarize(values, config.batch_size)["throughput_ips"]
                )
    ratios = {
        output: [
            new / old
            for old, new in zip(
                throughput[forms[0]][output], throughput[forms[1]][output], strict=True
            )
        ]
        for output in ("frame", "patch", "device")
    }
    summary = {
        output: {
            "median": float(np.median(values)),
            "min": float(min(values)),
            "max": float(max(values)),
        }
        for output, values in ratios.items()
    }
    return {
        "status": "done",
        "reason": "",
        "forms": forms,
        "throughput": throughput,
        "ratios": ratios,
        "ratio": summary,
        "frame_equal": frame_equal,
        "patches_equal": patches_equal,
        "rounds": rounds,
        "warm_up_batches": warm_up.min_batches,
        "timed_batches": timed.min_batches,
        "versions": read_versions(),
        "mlx_architecture": mx.device_info()["architecture"],
        "mlx_disable_compile": os.environ.get("MLX_DISABLE_COMPILE"),
        "time": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def _median_time(function: Callable[[], Any], warm_up: int = 3, calls: int = 20) -> float:
    """The median time of calls of function with mx.eval after each, after the warm-up calls."""
    import mlx.core as mx

    for _ in range(warm_up):
        mx.eval(function())
    times = []
    for _ in range(calls):
        start = time.perf_counter()
        mx.eval(function())
        times.append(time.perf_counter() - start)
    return float(np.median(times))


def _queued_time(function: Callable[[], Any], warm_up: int = 3, calls: int = 20) -> float:
    """The time of one call when calls calls run back to back with one mx.eval at the end: the
    GPU time without the wait of each mx.eval."""
    import mlx.core as mx

    for _ in range(warm_up):
        mx.eval(function())
    start = time.perf_counter()
    mx.eval([function() for _ in range(calls)])
    return (time.perf_counter() - start) / calls


def _count_kernels(outputs: "mx.array", folder: Path) -> int:
    """The number of operation nodes of the graph of outputs, from a DOT export without an
    evaluation."""
    import re

    import mlx.core as mx

    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "graph.dot"
    mx.export_to_dot(str(path), outputs)
    text = path.read_text()
    return len(re.findall(r"shape\s*=\s*rectangle", text)) or len(re.findall(r"label=", text))


def run_profile_worker(config: BenchConfig, bench_dir: Path = Path(".")) -> dict:
    """Profile the compiled forward pass at P1 (Table 12 of specs/06-speed/plan.html).

    The host share of 50 calls (the compiled call before mx.eval, the time to the end of mx.eval,
    the gap between two calls), the 11 parts of block 1 each alone and compiled (3 warm-up and 20
    calls, mx.eval after each; and, report only, 20 calls with one mx.eval, which leaves out the
    wait of each mx.eval), the kernel count of one forward pass (compiled and eager) and the
    compiled throughput at batch sizes 8 and 32.
    """
    import mlx.core as mx

    from dinov3_apple import mlx_weights
    from dinov3_apple.flops import PART_NAMES
    from dinov3_apple.mlx_model import compile_backbone

    model = mlx_weights.load_mlx_backbone(config.model_size, config.dtype_mode, tf32=False)
    forward = compile_backbone(model)
    inputs = read_inputs()
    batch = mx.array(make_batch(inputs, config.side, config.batch_size))
    for _ in range(5):
        mx.eval(forward(batch))
    call_s, eval_s, gap_s = [], [], []
    last_end = None
    for _ in range(50):
        start = time.perf_counter()
        if last_end is not None:
            gap_s.append(start - last_end)
        tokens = forward(batch)
        called = time.perf_counter()
        mx.eval(tokens)
        last_end = time.perf_counter()
        call_s.append(called - start)
        eval_s.append(last_end - start)
    host = {
        "call_median_s": float(np.median(call_s)),
        "eval_median_s": float(np.median(eval_s)),
        "gap_median_s": float(np.median(gap_s)),
        "host_share": float(np.median(call_s) / np.median(eval_s)),
    }

    block = model.layer[0]
    dtypes = model.dtypes
    grid = config.side // model.config.patch_size
    cos, sin = model.rope(grid, grid)
    x = model.embeddings(batch)
    parts = mlx_block_parts(block)
    state: dict[str, Any] = {"x": x}
    state["h"] = parts["norm1"](x)[0].astype(dtypes.matmul)
    state["q"], state["k"], state["v"] = parts["qkv_proj"](state["h"])
    state["qr"], state["kr"] = parts["rope"](state["q"], state["k"], cos, sin)
    state["a"] = parts["sdpa"](state["qr"], state["kr"], state["v"])[0]
    state["o"] = parts["o_proj"](state["a"])[0]
    state["x1"] = parts["layer_scale1"](state["o"], x)[0]
    state["h2"] = parts["norm2"](state["x1"])[0].astype(dtypes.matmul)
    state["ff"] = parts["ff_in"](state["h2"])
    state["g"] = parts["activation"](*state["ff"])[0]
    state["y"] = parts["ff_out"](state["g"])[0]
    arguments = {
        "norm1": (x,),
        "qkv_proj": (state["h"],),
        "rope": (state["q"], state["k"], cos, sin),
        "sdpa": (state["qr"], state["kr"], state["v"]),
        "o_proj": (state["a"],),
        "layer_scale1": (state["o"], x),
        "norm2": (state["x1"],),
        "ff_in": (state["h2"],),
        "activation": tuple(state["ff"]),
        "ff_out": (state["g"],),
        "layer_scale2": (state["y"], state["x1"]),
    }
    mx.eval(list(arguments.values()))
    parts_s, parts_queued_s = {}, {}
    for name in PART_NAMES:
        compiled = mx.compile(parts[name])
        parts_s[name] = _median_time(lambda c=compiled, a=arguments[name]: c(*a))
        parts_queued_s[name] = _queued_time(lambda c=compiled, a=arguments[name]: c(*a))
    block_compiled = mx.compile(lambda y: block(y, cos, sin))
    block_s = _median_time(lambda: block_compiled(x))
    block_queued_s = _queued_time(lambda: block_compiled(x))

    kernels = {
        "compiled": _count_kernels(forward(batch), bench_dir / "dot-compiled"),
        "eager": _count_kernels(model(batch), bench_dir / "dot-eager"),
    }
    batch_sizes = {}
    for batch_size in (8, 32):
        pixels = mx.array(make_batch(inputs, config.side, batch_size))
        seconds = _median_time(lambda p=pixels: forward(p))
        batch_sizes[str(batch_size)] = {"median_s": seconds, "throughput_ips": batch_size / seconds}
    return {
        "status": "done",
        "reason": "",
        "host": host,
        "parts_s": parts_s,
        "parts_queued_s": parts_queued_s,
        "block_s": block_s,
        "block_queued_s": block_queued_s,
        "blocks": model.config.blocks,
        "kernels": kernels,
        "batch_sizes": batch_sizes,
        "versions": read_versions(),
        "mlx_architecture": mx.device_info()["architecture"],
        "time": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def run_token_range_worker(config: BenchConfig, pixels: np.ndarray) -> dict:
    """Return the token range of image 0 (eager run_mlx_token_range) above 512 px (D11): the
    report for the prediction that fp16-autocast stays valid at high input resolution."""
    from dinov3_apple import mlx_weights
    from dinov3_apple.mlx_model import run_mlx_token_range

    model = mlx_weights.load_mlx_backbone(config.model_size, config.dtype_mode, tf32=False)
    token_range = run_mlx_token_range(model, pixels[:1])
    largest = max(
        float(np.max(token_range.tokens)),
        float(np.max(token_range.attention)),
        float(np.max(token_range.ff_hidden)),
        float(np.max(token_range.ff_out)),
    )
    return {
        "status": "done",
        "reason": "",
        "finite": token_range.finite,
        "largest_value": largest,
        "below_fp16_max": largest < 65_504,
        "token_range": token_range.report_values(),
        "versions": read_versions(),
        "time": datetime.now(UTC).isoformat(timespec="seconds"),
    }
