"""The benchmark matrix, the time budget, the memory need, the input batch, the validity check,
the summary statistics and the baseline rows (specs/03-baseline/plan.html). Also the bytes of
each weight of a backend and the runner of one worker subprocess, which the scripts share.

No torch import, so the mlx harness of Step 6 can reuse this module without the memory of torch in
its footprint.
"""

import json
import subprocess
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from functools import cache

import numpy as np
from safetensors import safe_open

from dinov3_apple.flops import MODEL_SHAPES, tokens
from dinov3_apple.hub import MODEL_IDS, parameter_count
from dinov3_apple.machine import MachineState
from dinov3_apple.memory import load_bytes, replay_checks
from dinov3_apple.parity import TEST_IMAGES, THRESHOLDS, Parity, measure_parity
from dinov3_apple.stored import reference_path

MAIN_MODEL_SIZES = [size for size in MODEL_IDS if size != "ViT-7B/16"]
GIANT_MODEL_SIZE = "ViT-7B/16"
DTYPE_MODE_NAMES = ["fp32", "fp16-cast", "bf16-cast", "fp16-autocast", "bf16-autocast"]
CAST_MODES = ["fp16-cast", "bf16-cast"]
SIDES = [224, 448, 512]
BATCH_SIZES = [1, 8, 32]
OUTPUTS = ["frame", "patch"]

DTYPE_OF_MODE = {
    "fp32": "fp32",
    "fp16-cast": "fp16",
    "fp16-autocast": "fp16",
    "bf16-cast": "bf16",
    "bf16-autocast": "bf16",
}
# Bytes per weight value. Autocast keeps the weights in fp32.
WEIGHT_BYTES = {"fp32": 4, "fp16-cast": 2, "bf16-cast": 2, "fp16-autocast": 4, "bf16-autocast": 4}
# The bytes of one value of each dtype of DTYPE_OF_MODE.
DTYPE_BYTES = {"fp32": 4, "fp16": 2, "bf16": 2}

# The factor on the activation terms of the memory need (Section 3.6). The build may only raise it.
ACTIVATION_FACTOR = 1.0

# Table 3 of the plan: name -> (model size, input resolution, batch size, output).
PRIORITY = {
    "P1": ("ViT-S/16", 224, 8, "frame"),
    "P2": ("ViT-H+/16", 512, 8, "patch"),
}

DEFERRED_REASON = "ViT-7B/16 runs in its own session (D8)"

# Quantized weights (D14 to D16 of specs/06-speed/plan.html): the model sizes of the quantized rows
# and the dtype modes that load quantized weights.
QUANTIZED_MODEL_SIZES = ["ViT-H+/16", "ViT-7B/16"]
QUANTIZED_MODES = ["fp16-autocast", "bf16-autocast"]


@dataclass(frozen=True)
class TimeBudget:
    """When a phase of batches ends (Table 5).

    Attributes:
        min_batches: The phase runs at least this many batches.
        min_seconds: The phase runs at least this long.
        max_seconds: The phase stops at this time, if floor_batches are done. None for no limit.
        floor_batches: The batches that must be done before max_seconds can stop the phase.
    """

    min_batches: int
    min_seconds: float
    max_seconds: float | None = None
    floor_batches: int = 0


WARM_UP = TimeBudget(min_batches=3, min_seconds=2.0)
TIMED = TimeBudget(min_batches=30, min_seconds=10.0, max_seconds=60.0, floor_batches=10)


def keep_going(budget: TimeBudget, batches: int, elapsed_s: float) -> bool:
    """Return True while the phase needs another batch.

    Args:
        budget: The time budget of the phase.
        batches: The batches done in the phase.
        elapsed_s: The sum of the batch times of the phase.
    """
    stop_time = budget.max_seconds is not None and elapsed_s >= budget.max_seconds
    if stop_time and batches >= budget.floor_batches:
        return False
    return batches < budget.min_batches or elapsed_s < budget.min_seconds


@dataclass(frozen=True)
class BenchConfig:
    """One configuration of the benchmark matrix. The output is not part of it (D3).

    Attributes:
        run: "B1" to "B4" (Table 2 of the plan).
        model_size: A key of MODEL_IDS.
        dtype_mode: A name of DTYPE_MODE_NAMES.
        side: The square input resolution in pixels.
        batch_size: The batch size.
        compile: True to wrap the backbone with torch.compile (torch-mps) or to run the compiled
            forward pass (mlx, D7 of specs/06-speed/plan.html).
        bits: None for unquantized weights; 8 or 4 for quantized block weights (mlx, D22).
        repeat: 0 for the main run, 1 and 2 for the repeats of B2. It stays the last field of the
            key (add_rounds).
    """

    run: str
    model_size: str
    dtype_mode: str
    side: int
    batch_size: int
    compile: bool = False
    bits: int | None = None
    repeat: int = 0

    @property
    def key(self) -> tuple:
        """The identity of the configuration in the results, in the order of the fields. The run
        is not part of it."""
        return (
            self.model_size,
            self.dtype_mode,
            self.side,
            self.batch_size,
            self.compile,
            self.bits,
            self.repeat,
        )

    @property
    def label(self) -> str:
        """A short name for logs, for example "ViT-S/16 fp32 224px b8"."""
        extra = (" compile" if self.compile else "") + (f" {self.bits}-bit" if self.bits else "")
        extra += f" r{self.repeat}" if self.repeat else ""
        return f"{self.model_size} {self.dtype_mode} {self.side}px b{self.batch_size}{extra}"


def line_key(line: dict) -> tuple:
    """Return the BenchConfig.key of one JSON line. A line without the field bits (Steps 3 to 5)
    has unquantized weights."""
    return (
        line["model_size"],
        line["dtype_mode"],
        line["side"],
        line["batch_size"],
        line["compile"],
        line.get("bits"),
        line["repeat"],
    )


def latest_lines(lines: list[dict]) -> list[dict]:
    """Return the last line of each configuration, in the order of first appearance."""
    latest: dict[tuple, dict] = {}
    for line in lines:
        latest[line_key(line)] = line
    return list(latest.values())


def _modes(model_size: str, backend: str = "torch-mps") -> list[str]:
    """Return the dtype modes of one model size on one backend.

    ViT-7B/16 never loads fp32 weights (CLAUDE.md run rules, ADR 0008). On torch-mps its autocast
    modes keep fp32 weights, so it runs the two cast modes. On mlx its autocast modes keep fp16 or
    bf16 weights (ADR 0007), so it runs the four fp16 and bf16 dtype modes (D23).
    """
    if model_size != GIANT_MODEL_SIZE:
        return DTYPE_MODE_NAMES
    if backend == "mlx":
        return [mode for mode in DTYPE_MODE_NAMES if mode != "fp32"]
    return CAST_MODES


def bench_configurations(
    model_sizes: list[str] | None = None,
    priority: bool = False,
    repeats: int = 0,
    compile: bool = False,
) -> list[BenchConfig]:
    """Return the configurations of one run of Table 2 of the plan.

    Args:
        model_sizes: The model sizes of B1 or B4. None means the five sizes of B1.
        priority: True for the priority configurations P1 and P2 in all five dtype modes.
        repeats: With priority, the number of repeats (B2). Repeat numbers start at 1.
        compile: The compile rows: B3 with priority, else compile rows of B1 or B4 (D9 of
            specs/06-speed/plan.html).

    Returns:
        The configurations from the smallest to the largest model size.
    """
    if priority:
        run = "B3" if compile else ("B2" if repeats else "B1")
        return [
            BenchConfig(run, model_size, mode, side, batch_size, compile, repeat=repeat)
            for model_size, side, batch_size, _ in PRIORITY.values()
            for repeat in (range(1, repeats + 1) if repeats else [0])
            for mode in DTYPE_MODE_NAMES
        ]
    configs = []
    for model_size in model_sizes or MAIN_MODEL_SIZES:
        run = "B4" if model_size == GIANT_MODEL_SIZE else "B1"
        configs += [
            BenchConfig(run, model_size, mode, side, batch_size, compile)
            for mode in _modes(model_size)
            for side in SIDES
            for batch_size in BATCH_SIZES
        ]
    return configs


def add_rounds(configs: list[BenchConfig], lines: list[dict], rounds: int) -> list[BenchConfig]:
    """Run the configurations in rounds: the whole list once per round (finding F1 of verify.html).

    One run of a configuration can be off by up to 2x on this Mac, so the baseline rows take the
    median of the runs. Each round gives each configuration the next free repeat number. A
    configuration without a line starts with repeat 0, the main run.

    Args:
        configs: The configurations of one round.
        lines: The JSON lines that exist already.
        rounds: The number of rounds. 0 returns the configurations unchanged.

    Returns:
        The configurations of all rounds, round after round.
    """
    if not rounds:
        return configs
    taken: dict[tuple, int] = {}
    for line in lines:
        identity = line_key(line)[:-1]
        taken[identity] = max(taken.get(identity, -1), line["repeat"])
    return [
        replace(config, repeat=taken.get(config.key[:-1], -1) + 1 + index)
        for index in range(rounds)
        for config in configs
    ]


@cache
def _parameter_count(model_size: str) -> int:
    return parameter_count(model_size)


def bytes_per_weight(backend: str, dtype_mode: str) -> int:
    """Return the bytes of each loaded weight of one backend in one dtype mode.

    The run rule and the memory need of a load use this value (F20 of
    specs/05-all-sizes/verify.html). mlx casts the weights to the matmul dtype at the load, so its
    autocast modes have half weights. torch keeps fp32 weights in the autocast modes.

    Args:
        backend: "mlx", "torch-mps" or "torch-cpu".
        dtype_mode: A key of DTYPE_OF_MODE.

    Returns:
        4 or 2.

    Raises:
        ValueError: For another backend.
    """
    if backend == "mlx":
        return DTYPE_BYTES[DTYPE_OF_MODE[dtype_mode]]
    if backend in ("torch-mps", "torch-cpu"):
        return WEIGHT_BYTES[dtype_mode]
    raise ValueError(f"unknown backend {backend!r}")


def quantized_weight_bytes(model_size: str, bits: int, parameters: int | None = None) -> int:
    """Return the bytes of the weights of one model size with quantized block weights (Table 9 of
    specs/06-speed/plan.html).

    Each block weight takes bits / 8 bytes plus one fp16 or bf16 scale and one offset for each
    group of 64 weights. The other parameters take 2 bytes. The loader keeps the layer norms,
    LayerScale and the prefix tokens in fp32, so the value leaves out at most 2.0 MB (ViT-7B/16).

    Args:
        model_size: A key of MODEL_SHAPES.
        bits: 8 or 4.
        parameters: The parameter count. None reads it from the safetensors headers.

    Returns:
        block weights x (bits / 8 + 4 / 64) + the other parameters x 2.
    """
    shape = MODEL_SHAPES[model_size]
    count = _parameter_count(model_size) if parameters is None else parameters
    ff_matrices = 3 if shape.gated else 2
    block = shape.blocks * (4 * shape.dim**2 + ff_matrices * shape.dim * shape.ff)
    return block * (8 * bits + 4) // 64 + (count - block) * 2


def required_bench_bytes(
    config: BenchConfig, activation_factor: float = ACTIVATION_FACTOR, backend: str = "torch-mps"
) -> int:
    """Return the memory need of one configuration on one backend (Section 3.6 of the Step 3 plan,
    D8 of specs/06-speed/plan.html).

    torch-mps: need = load_bytes(weights) + factor x (B T (8 D + 3 F) a + 2 B heads T^2 a), with
    a = 2 bytes in cast mode and 4 bytes otherwise. The attention term assumes a full score matrix.
    mlx: the weights take the bytes of mlx (or the quantized bytes), and the need has no score
    term: the MLX SDPA kernel needs at most 0.012 MB extra (probe sdpa).

    Args:
        config: The configuration.
        activation_factor: The factor on the activation terms.
        backend: "torch-mps" or "mlx".

    Returns:
        The memory need in bytes.

    Raises:
        ValueError: For bits on a backend other than mlx.
        WeightsNotFoundError: If the weights of the model size are not on this Mac.
    """
    if config.bits is not None and backend != "mlx":
        raise ValueError(f"quantized weights run on mlx only, not on {backend}")
    shape = MODEL_SHAPES[config.model_size]
    if config.bits is not None:
        weights = quantized_weight_bytes(config.model_size, config.bits)
    else:
        count = _parameter_count(config.model_size)
        weights = count * bytes_per_weight(backend, config.dtype_mode)
    a = 2 if config.dtype_mode in CAST_MODES else 4
    b, t = config.batch_size, tokens(config.side)
    activations = b * t * (8 * shape.dim + 3 * shape.ff) * a
    if backend != "mlx":
        activations += 2 * b * shape.heads * t * t * a
    return load_bytes(weights) + round(activation_factor * activations)


def make_batch(inputs: dict[str, np.ndarray], side: int, batch_size: int) -> np.ndarray:
    """Fill a batch with the stored pixels of the test images, in the order of TEST_IMAGES.

    Args:
        inputs: The stored pixels from read_inputs, each with shape (1, 3, H, W).
        side: The square input resolution.
        batch_size: The batch size.

    Returns:
        fp32 pixels with shape (batch_size, 3, side, side).
    """
    images = [inputs[f"{image}/{side}x{side}"][0] for image in TEST_IMAGES]
    batch = np.stack([images[i % len(images)] for i in range(batch_size)])
    return batch.astype(np.float32, copy=False)


def reference_tokens(model_size: str, side: int) -> np.ndarray | None:
    """Return the reference tokens of image 0 at side x side px. None if the file is missing."""
    path = reference_path(model_size)
    if not path.exists():
        return None
    with safe_open(path, "np") as tensors:
        return tensors.get_tensor(f"{TEST_IMAGES[0]}/{side}x{side}/tokens")


@dataclass(frozen=True)
class Validity:
    """The validity of the first output of a configuration (Section 3.5).

    Attributes:
        valid: True when the output is finite and passes the parity thresholds.
        finite: True when every value of the frame embedding and the patch embeddings is finite.
        parity: The parity against the reference output, or None without a reference.
        note: The reason in words.
    """

    valid: bool
    finite: bool
    parity: Parity | None
    note: str


def check_validity(
    frame: np.ndarray, patches: np.ndarray, reference: np.ndarray | None, dtype_mode: str
) -> Validity:
    """Check the frame embedding and the patch embeddings of image 0 of the first batch.

    Args:
        frame: The frame embedding with shape (D,).
        patches: The patch embeddings with shape (N, D).
        reference: The reference tokens with shape (1 + 4 + N, D), or None (finite check only).
        dtype_mode: A key of THRESHOLDS.

    Returns:
        The validity.
    """
    frame, patches = frame.astype(np.float32), patches.astype(np.float32)
    finite = bool(np.isfinite(frame).all() and np.isfinite(patches).all())
    if not finite:
        return Validity(False, False, None, "non-finite output")
    if reference is None:
        return Validity(True, True, None, "finite only: no reference file")
    # measure_parity takes a full token sequence. The register rows come from the reference, so
    # they do not change the parity of the frame embedding and the patch embeddings.
    tokens_ = np.concatenate([frame[None], reference[1 : 1 + 4], patches])
    parity = measure_parity(reference, tokens_)
    threshold = THRESHOLDS[dtype_mode]
    failed = [
        name
        for name in ("frame", "patch_mean", "patch_p01")
        if getattr(parity, name) < getattr(threshold, name)
    ]
    if failed:
        return Validity(False, True, parity, f"parity below the threshold: {', '.join(failed)}")
    return Validity(True, True, parity, "parity passes")


def summarize(times: list[float], batch_size: int) -> dict[str, float]:
    """Return the throughput, the median and the p90 of the batch times of one output.

    Args:
        times: The time of each timed batch in seconds.
        batch_size: The batch size.

    Returns:
        throughput_ips (images per second), median_s and p90_s.
    """
    values = np.asarray(times, dtype=np.float64)
    return {
        "throughput_ips": float(batch_size * len(values) / values.sum()),
        "median_s": float(np.median(values)),
        "p90_s": float(np.percentile(values, 90)),
    }


# Table 7 of the plan. The worker gives the fields from "valid" to "time"; the driver adds the rest.
# Step 6 adds bits, compile_trace_s and the name load_footprint_peak_bytes (Table 8 of
# specs/06-speed/plan.html). The lines of Steps 3 to 5 keep the name load_peak_bytes.
LINE_FIELDS = [
    "run",
    "model_size",
    "backend",
    "dtype_mode",
    "side",
    "batch_size",
    "compile",
    "bits",
    "repeat",
    "status",
    "reason",
    "valid",
    "finite",
    "parity",
    "validity_note",
    "load_time_s",
    "first_call_s",
    "compile_trace_s",
    "warm_up_batches",
    "warm_up_s",
    "timed_batches",
    "timed_s",
    "frame",
    "patch",
    "device",
    "output_dtype",
    "need_bytes",
    "available_bytes",
    "footprint_base_bytes",
    "load_footprint_peak_bytes",
    "peak_bytes",
    "run_peak_bytes",
    "run_peak_note",
    "mps_driver_max_bytes",
    "machine",
    "versions",
    "attn_implementation",
    "start_time",
    "time",
]
# The pairing fields of a Step 6 line (Table 8 of specs/06-speed/plan.html), from the driver.
PAIR_FIELDS = [
    "run_id",
    "group",
    "round",
    "position",
    "backend",
    "priority",
    "other_heavy_jobs_seen",
]
# The fields of an mlx worker of Step 6 (Table 8), on top of LINE_FIELDS.
MLX_FIELDS = [
    "compile_trace_s",
    "mlx_architecture",
    "mlx_disable_compile",
    "mlx_load_limit_bytes",
    "mlx_memory_limit_bytes",
    "mlx_timed_peak_bytes",
    "mlx_cache_bytes",
    "mlx_peak_bytes",
    "output_transfer",
    "tf32",
    "load_report_need_bytes",
    "load_report_peak_bytes",
    "load_report_cache_bytes",
    "load_report_footprint_bytes",
]


def run_worker_subprocess(
    command: list[str], env: Mapping[str, str], timeout_s: float, stderr_lines: int = 20
) -> dict:
    """Run one worker in a new subprocess and return its result.

    The worker prints its result as a JSON line on stdout; the last JSON line counts. The
    memory-check lines of the worker (send_checks_to) go to the log of the driver, with the time
    of the worker, also after an error or a timeout. scripts/bench_mlx_port.py,
    scripts/cross_check.py and scripts/benchmark.py share this runner (F20 of
    specs/05-all-sizes/verify.html).

    Args:
        command: The command line of the worker.
        env: The environment of the worker.
        timeout_s: The time limit of the worker in seconds.
        stderr_lines: The number of the last stderr lines in the reason of an error.

    Returns:
        The result of the worker. Else the status "error" with the reason: the timeout, or the
        exit code and the last lines of the stderr of the worker.
    """
    try:
        done = subprocess.run(
            command,
            check=False,
            env=dict(env),
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as error:
        stderr = error.stderr or ""
        replay_checks(stderr.decode() if isinstance(stderr, bytes) else stderr)
        return {"status": "error", "reason": f"timeout after {error.timeout} s"}
    replay_checks(done.stderr)
    if done.returncode == 0:
        for text in reversed(done.stdout.splitlines()):
            if text.startswith("{"):
                return json.loads(text)
    tail = "\n".join(done.stderr.splitlines()[-stderr_lines:])
    return {"status": "error", "reason": f"exit code {done.returncode}: {tail}"}


def result_line(
    config: BenchConfig,
    worker: dict,
    need_bytes: int,
    available_bytes: int,
    machine: MachineState,
    backend: str = "torch-mps",
) -> dict:
    """Return one JSON line of a results file of the benchmarks.

    Args:
        config: The configuration.
        worker: The result of the worker, or {"status": ..., "reason": ...} for a line without a
            worker. Missing fields become None.
        need_bytes: The memory need from required_bench_bytes.
        available_bytes: The available memory before the worker.
        machine: The machine state before the worker.
        backend: "torch-mps" or "mlx".

    Returns:
        A dict with every field of LINE_FIELDS.
    """
    line = dict.fromkeys(LINE_FIELDS)
    line |= asdict(config) | {"backend": backend}
    line |= worker
    line |= {"need_bytes": need_bytes, "available_bytes": available_bytes}
    line["machine"] = asdict(machine)
    line["versions"] = line["versions"] or machine.versions
    # A line without a worker (skipped, deferred) gets the time of the driver.
    line["time"] = line["time"] or datetime.now(UTC).isoformat(timespec="seconds")
    return line


@dataclass(frozen=True)
class BaselineRow:
    """The baseline row of one model size, input resolution, batch size, output and dtype.

    Attributes:
        line: The chosen JSON line, or None for an empty row. With several runs of the chosen
            configuration, the run with the median throughput (the lower one of the two middle
            runs for an even count).
        reason: Why the row is empty, or "" when it has a line.
        runs: The valid runs of the chosen configuration.
    """

    line: dict | None
    reason: str
    runs: int = 0


def baseline_rows(lines: list[dict], output: str) -> dict[tuple, BaselineRow]:
    """Pick the fastest valid configuration for each model size, input resolution, batch size
    and dtype (D2, Section 3.7).

    Each configuration counts with the median of its valid runs: the main run, the repeats and
    the rounds (finding F1 of verify.html). Compile rows count: the user decided on 2026-11-06
    that they join the baseline, because they were more than 10 % faster.

    Args:
        lines: JSON lines of results/baseline-torch-mps.jsonl.
        output: "frame" or "patch".

    Returns:
        A row for each (model size, input resolution, batch size, dtype) that has a line.
    """
    candidates: dict[tuple, list[dict]] = {}
    for line in latest_lines(lines):
        key = (
            line["model_size"],
            line["side"],
            line["batch_size"],
            DTYPE_OF_MODE[line["dtype_mode"]],
        )
        candidates.setdefault(key, []).append(line)
    rows = {}
    for key, group in candidates.items():
        runs: dict[tuple, list[dict]] = {}
        for line in group:
            if line["status"] == "done" and line["valid"]:
                runs.setdefault((line["dtype_mode"], line["compile"]), []).append(line)
        if runs:
            medians = [_median_run(found, output) for found in runs.values()]
            best = max(medians, key=lambda run: run[0][output]["throughput_ips"])
            rows[key] = BaselineRow(best[0], "", best[1])
            continue
        reasons = []
        for line in group:
            if line["repeat"]:
                continue
            if line["status"] != "done":
                reasons.append(f"{line['dtype_mode']}: {line['status']}, {line['reason']}")
            else:
                reasons.append(f"{line['dtype_mode']}: not valid, {line.get('validity_note', '')}")
        rows[key] = BaselineRow(None, "; ".join(reasons))
    return rows


def _median_run(runs: list[dict], output: str) -> tuple[dict, int]:
    ordered = sorted(runs, key=lambda line: line[output]["throughput_ips"])
    return ordered[(len(ordered) - 1) // 2], len(ordered)
