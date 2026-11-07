"""The small paired benchmarks of mlx against torch-mps at a priority configuration.

P1 (Step 4): ViT-S/16 in fp32 at 224 px, batch sizes 8 and 1 (Section 7 of
specs/04-mlx-port/plan.html). P2 (Step 5): ViT-H+/16 at 512 px, batch size 8, in the five dtype
modes (Section 7 of specs/05-all-sizes/plan.html).

Report only, no target. The throughput of one
configuration on this Mac changes by up to 2x from one run to the next (finding F1 of
specs/03-baseline/verify.html). So the mlx workers and the torch-mps workers run in alternation in
one session, round after round, and verify.html takes the median of the paired ratios.

Each worker runs in its own subprocess, one after the other. Before each worker the driver reads
the machine state, stops on battery power or in Low Power Mode, and checks the free memory. A
worker that does not fit gives a "skipped" line. The mlx worker imports no torch and runs with
MLX_ENABLE_TF32=0 (D7). The torch-mps worker is the eager worker of Step 3 with the MPS
watermarks 0.8 and 0.6.

Each worker sends the memory check and the load peak of its model load to the driver, and the
driver writes them to driver.log. A bf16 output of mlx moves to the host as a view of uint16,
because NumPy has no bf16; the validity check converts the bits to fp32 outside the timed batches.

Each priority has its own default results file and folder. P1: results/bench-mlx-port.jsonl and
tests/fixtures/generated/bench/mlx-port/. P2: results/bench-all-sizes.jsonl and
tests/fixtures/generated/bench/all-sizes/. With another results file (--results), the per-batch
files and driver.log go to the folder <stem>-batches/ next to that file.

Usage:
    uv run python scripts/bench_mlx_port.py --dry-run     # the plan and the needs, no model load
    uv run python scripts/bench_mlx_port.py               # P1: 3 rounds, 12 workers, about 4 min
    uv run python scripts/bench_mlx_port.py --priority P2 --dry-run
    uv run python scripts/bench_mlx_port.py --priority P2   # 3 rounds, 30 workers
    uv run python scripts/bench_mlx_port.py --priority P2 --dtype-modes fp32 bf16-cast
    uv run python scripts/bench_mlx_port.py --results <file>.jsonl   # a run that keeps the files
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from dinov3_apple.bench import (
    DTYPE_MODE_NAMES,
    PRIORITY,
    TIMED,
    WARM_UP,
    BenchConfig,
    TimeBudget,
    check_validity,
    keep_going,
    make_batch,
    reference_tokens,
    required_bench_bytes,
    result_line,
    summarize,
)
from dinov3_apple.footprint import read_footprint
from dinov3_apple.machine import (
    MachineStateError,
    check_machine_state,
    read_machine_state,
    read_versions,
)
from dinov3_apple.memory import (
    GB,
    InsufficientMemoryError,
    available_memory_bytes,
    replay_checks,
    send_checks_to,
)
from dinov3_apple.outputs import NUM_PREFIX_TOKENS
from dinov3_apple.stored import model_slug, read_inputs

if TYPE_CHECKING:  # the module imports mlx only in the mlx worker
    from dinov3_apple.mlx_model import MlxBackbone

REPO_ROOT = Path(__file__).parents[1]
RESULTS = REPO_ROOT / "results" / "bench-mlx-port.jsonl"
BENCH_DIR = REPO_ROOT / "tests" / "fixtures" / "generated" / "bench" / "mlx-port"
RESULTS_P2 = REPO_ROOT / "results" / "bench-all-sizes.jsonl"
BENCH_DIR_P2 = REPO_ROOT / "tests" / "fixtures" / "generated" / "bench" / "all-sizes"
# The default results file of each priority, and the folder of each default results file.
DEFAULT_RESULTS = {"P1": RESULTS, "P2": RESULTS_P2}
DEFAULT_BENCH_DIRS = {RESULTS: BENCH_DIR, RESULTS_P2: BENCH_DIR_P2}
LOG_FORMAT = "%(asctime)s %(name)s %(message)s"
WORKER_TIMEOUT_S = 600
STDERR_LINES = 20

# P1 of Step 4: one dtype mode, two batch sizes.
MODEL_SIZE = "ViT-S/16"
DTYPE_MODE = "fp32"
SIDE = 224
BATCH_SIZES = (8, 1)  # 8: the priority configuration P1
RUN = "step4"
# P2 of Step 5: one batch size, the five dtype modes.
RUN_P2 = "step5"

WATERMARKS = {"PYTORCH_MPS_HIGH_WATERMARK_RATIO": "0.8", "PYTORCH_MPS_LOW_WATERMARK_RATIO": "0.6"}
# (backend, tf32). TF32 off on mlx (D7); None: not an mlx worker.
WORKERS: list[tuple[str, bool | None]] = [("torch-mps", None), ("mlx", False)]

logger = logging.getLogger("bench_mlx_port")


def bench_dir_of(results: Path) -> Path:
    """Return the folder of the per-batch files and of driver.log for one results file.

    Args:
        results: The results file of the run.

    Returns:
        BENCH_DIR for the default results file of P1, BENCH_DIR_P2 for the default results file of
        P2. For another results file, the folder <stem>-batches next to it, so a second run (for
        example of the verifier) keeps the files of the first run.
    """
    for default, folder in DEFAULT_BENCH_DIRS.items():
        if results.resolve() == default.resolve():
            return folder
    return results.with_name(f"{results.stem}-batches")


def worker_plan(
    rounds: int = 3, priority: str = "P1", dtype_modes: list[str] | None = None
) -> list[tuple[int, int | str, str, bool | None]]:
    """Return (round, variant, backend, tf32) for each worker, in the order of the run.

    The variant is the batch size of P1 or the dtype mode of P2. The order of the two workers
    alternates from round to round (finding F1): round 1 torch-mps first, round 2 mlx first,
    round 3 torch-mps first. Every mlx worker runs with TF32 off.

    Args:
        rounds: The number of rounds.
        priority: "P1" or "P2".
        dtype_modes: The dtype modes of P2, in this order. None: all five.

    Returns:
        rounds x 2 batch sizes x 2 workers entries for P1, rounds x the dtype modes x 2 workers
        entries for P2.
    """
    variants = BATCH_SIZES if priority == "P1" else (dtype_modes or DTYPE_MODE_NAMES)
    plan = []
    for r in range(rounds):
        order = WORKERS[r % 2 :] + WORKERS[: r % 2]
        for variant in variants:
            plan += [(r + 1, variant, backend, tf32) for backend, tf32 in order]
    return plan


def worker_env(backend: str, tf32: bool | None, base: Mapping[str, str]) -> dict[str, str]:
    """Return the environment of one worker subprocess.

    Args:
        backend: "torch-mps" or "mlx".
        tf32: The TF32 state of an mlx worker. None for torch-mps.
        base: The environment of the driver.

    Returns:
        torch-mps: base with the MPS watermarks. mlx: base without the MPS watermarks, with
        MLX_ENABLE_TF32 set to the TF32 state (MLX reads it once, at its first use).
    """
    env = {k: v for k, v in base.items() if k not in WATERMARKS and k != "MLX_ENABLE_TF32"}
    if backend == "mlx":
        return env | {"MLX_ENABLE_TF32": str(int(bool(tf32)))}
    return env | WATERMARKS


def bench_config(round_: int, variant: int | str, priority: str = "P1") -> BenchConfig:
    """Return the configuration of one worker. The repeat number is the round minus 1.

    Args:
        round_: The round, from 1.
        variant: The batch size of P1 or the dtype mode of P2.
        priority: "P1" or "P2".
    """
    if priority == "P1":
        return BenchConfig(RUN, MODEL_SIZE, DTYPE_MODE, SIDE, variant, repeat=round_ - 1)
    model_size, side, batch_size, _ = PRIORITY[priority]
    return BenchConfig(RUN_P2, model_size, variant, side, batch_size, repeat=round_ - 1)


def output_transfer(host: np.ndarray) -> str:
    """Return how an output moved to the host: "view-uint16" for bf16 bits, else "native"."""
    return "view-uint16" if host.dtype == np.uint16 else "native"


def host_to_fp32(host: np.ndarray) -> np.ndarray:
    """Return the fp32 values of a host output. bf16 bits (uint16) become fp32; others stay."""
    if host.dtype != np.uint16:
        return host
    return (host.astype(np.uint32) << 16).view(np.float32)


def time_batch_mlx(
    model: "MlxBackbone", batch: np.ndarray
) -> tuple[tuple[float, float, float], np.ndarray, np.ndarray]:
    """Run one batch host to host on mlx and time it (Table 4 of the Step 3 plan).

    The output stays in its dtype, as in the torch-mps worker. NumPy has no bf16, so a bf16
    output moves to the host as a view of uint16 (the bits); host_to_fp32 converts them.

    Args:
        model: An MlxBackbone.
        batch: fp32 pixels with shape (B, 3, H, W) in NumPy. The model casts them.

    Returns:
        The device time, the frame time and the patch time in seconds, then the frame embeddings
        (B, D) and the patch embeddings (B, N, D) in NumPy: fp32, fp16 or uint16 bits of bf16.
    """
    import mlx.core as mx

    t0 = time.perf_counter()
    tokens = model(mx.array(batch))
    mx.eval(tokens)  # mlx synchronizes with mx.eval
    t1 = time.perf_counter()
    if tokens.dtype == mx.bfloat16:
        tokens = tokens.view(mx.uint16)
    frame = np.array(tokens[:, 0])
    t2 = time.perf_counter()
    patches = np.array(tokens[:, NUM_PREFIX_TOKENS:])
    t3 = time.perf_counter()
    return (t1 - t0, t2 - t0, (t1 - t0) + (t3 - t2)), frame, patches


def _phase(
    run: Callable[[], tuple[float, float, float]], budget: TimeBudget
) -> list[tuple[float, float, float]]:
    times: list[tuple[float, float, float]] = []
    elapsed = 0.0
    while keep_going(budget, len(times), elapsed):
        batch_times = run()
        times.append(batch_times)
        elapsed += batch_times[2]
    return times


def run_mlx_worker(
    config: BenchConfig,
    warm_up: TimeBudget = WARM_UP,
    timed: TimeBudget = TIMED,
    bench_dir: Path = BENCH_DIR / "mlx",
) -> dict:
    """Load the MLX backbone, check its first output, warm up, time the batches, return the result.

    The same phases and fields as torch_bench.run_worker of Step 3. TF32 is off (D7).

    Args:
        config: The configuration.
        warm_up: The time budget of the warm-up batches.
        timed: The time budget of the timed batches.
        bench_dir: The folder for the per-batch times.

    Returns:
        The worker fields of LINE_FIELDS with status "done", plus mlx_peak_bytes,
        mlx_memory_limit_bytes (the value that the load gave to mx.set_memory_limit), tf32,
        output_transfer and the fields of the load report (null if the model has none).

    Raises:
        TF32ModeError: If MLX_ENABLE_TF32 is not 0 in the environment.
        InsufficientMemoryError: If the memory check of the load fails.
        WeightsNotFoundError: If the weights are not on this Mac.
    """
    import mlx.core as mx

    from dinov3_apple.mlx_weights import load_mlx_backbone

    start_time = datetime.now(UTC).isoformat(timespec="seconds")
    base = read_footprint()
    start = time.perf_counter()
    model = load_mlx_backbone(config.model_size, config.dtype_mode, tf32=False)
    load_time_s = time.perf_counter() - start
    load_peak = read_footprint().lifetime_max_bytes

    batch = make_batch(read_inputs(), config.side, config.batch_size)
    (_, _, first_call_s), frame, patches = time_batch_mlx(model, batch)
    transfer = output_transfer(frame)
    validity = check_validity(
        host_to_fp32(frame[0]),
        host_to_fp32(patches[0]),
        reference_tokens(config.model_size, config.side),
        config.dtype_mode,
    )
    output_dtype = "bfloat16" if transfer == "view-uint16" else str(frame.dtype)
    del frame, patches

    warm = _phase(lambda: time_batch_mlx(model, batch)[0], warm_up)
    timed_times = _phase(lambda: time_batch_mlx(model, batch)[0], timed)
    end = read_footprint()

    device, frame_t, patch_t = (list(column) for column in zip(*timed_times, strict=True))
    bench_dir.mkdir(parents=True, exist_ok=True)
    slug = model_slug(config.model_size)
    name = f"{slug}_{config.dtype_mode}_{config.side}_b{config.batch_size}_r{config.repeat}.json"
    (bench_dir / name).write_text(
        json.dumps({"config": config.key, "device": device, "frame": frame_t, "patch": patch_t})
    )
    above_load = end.lifetime_max_bytes > load_peak
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
        "warm_up_batches": len(warm),
        "warm_up_s": sum(t[2] for t in warm),
        "timed_batches": len(timed_times),
        "timed_s": sum(patch_t),
        "frame": summarize(frame_t, config.batch_size),
        "patch": summarize(patch_t, config.batch_size),
        "device": summarize(device, config.batch_size),
        "output_dtype": output_dtype,
        "footprint_base_bytes": base.current_bytes,
        "load_peak_bytes": load_peak,
        "peak_bytes": end.lifetime_max_bytes,
        "run_peak_bytes": end.lifetime_max_bytes if above_load else None,
        "run_peak_note": "" if above_load else f"below the load peak; end {end.current_bytes}",
        "mps_driver_max_bytes": None,
        "versions": read_versions(),
        "attn_implementation": "mx.fast.scaled_dot_product_attention",
        "start_time": start_time,
        "time": datetime.now(UTC).isoformat(timespec="seconds"),
        "mlx_peak_bytes": mx.get_peak_memory(),
        "mlx_memory_limit_bytes": mx.get_memory_limit(),
        "tf32": False,
        "output_transfer": transfer,
        **load_report_fields(load_report),
    }


def load_report_fields(load_report: object | None) -> dict[str, int | None]:
    """Return the fields of the load report of an mlx worker, or null fields without a report."""
    return {
        f"load_report_{name}": getattr(load_report, name, None)
        for name in ("need_bytes", "peak_bytes", "cache_bytes", "footprint_bytes")
    }


def worker_main(spec_json: str) -> None:
    """Run one worker in this process and print the result as one JSON line.

    The memory check of the model load goes to stderr; the driver writes it to driver.log.

    Args:
        spec_json: The fields of BenchConfig, plus backend and bench_dir (the folder of the
            per-batch files), as JSON.
    """
    send_checks_to(sys.stderr)
    spec = json.loads(spec_json)
    backend = spec.pop("backend")
    bench_dir = Path(spec.pop("bench_dir"))
    config = BenchConfig(**spec)
    try:
        if backend == "mlx":
            result = run_mlx_worker(config, bench_dir=bench_dir / "mlx")
        else:
            from dinov3_apple.torch_bench import run_worker

            result = run_worker(config, bench_dir=bench_dir / "torch-mps")
            result |= {"mlx_peak_bytes": None, "mlx_memory_limit_bytes": None, "tf32": None}
            result |= {"output_transfer": "native", **load_report_fields(None)}
    except InsufficientMemoryError as error:
        result = {"status": "skipped", "reason": f"load check in the worker: {error}"}
    watermarks = {k: os.environ[k] for k in WATERMARKS if k in os.environ}
    result["mps_watermarks"] = watermarks or None
    result["torch_imported"] = "torch" in sys.modules
    print(json.dumps(result))


def run_worker_subprocess(
    config: BenchConfig, backend: str, tf32: bool | None, bench_dir: Path
) -> dict:
    """Run one worker in a new subprocess and return its result.

    The memory checks of the worker go to the log of the driver, with the time of the worker.
    A worker that exits with an error gives status "error" and the last lines of its stderr.

    Args:
        config: The configuration.
        backend: "torch-mps" or "mlx".
        tf32: The TF32 state of an mlx worker. None for torch-mps.
        bench_dir: The folder of the per-batch files.

    Returns:
        The result of the worker.
    """
    spec = json.dumps(asdict(config) | {"backend": backend, "bench_dir": str(bench_dir)})
    command = [sys.executable, __file__, "--worker", spec]
    try:
        done = subprocess.run(
            command,
            check=False,
            env=worker_env(backend, tf32, os.environ),
            capture_output=True,
            text=True,
            timeout=WORKER_TIMEOUT_S,
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
    tail = "\n".join(done.stderr.splitlines()[-STDERR_LINES:])
    return {"status": "error", "reason": f"exit code {done.returncode}: {tail}"}


def dry_run(plan: list[tuple[int, int | str, str, bool | None]], priority: str = "P1") -> None:
    """Print each worker of the plan with its memory need. Load no model."""
    available = available_memory_bytes()
    print(f"available memory: {available / GB:.1f} GB")
    for round_, variant, backend, tf32 in plan:
        config = bench_config(round_, variant, priority)
        need = required_bench_bytes(config)
        fits = "fits" if need <= available else "does not fit"
        print(
            f"round {round_} {backend:9s} tf32 {tf32!s:5} {config.label:30s} "
            f"need {need / GB:4.2f} GB  {fits}"
        )


def run_plan(
    plan: list[tuple[int, int | str, str, bool | None]],
    results: Path,
    bench_dir: Path,
    priority: str = "P1",
) -> Counter:
    """Run the workers one after the other and append one line for each.

    Args:
        plan: The workers, from worker_plan.
        results: The results file.
        bench_dir: The folder of the per-batch files.
        priority: "P1" or "P2".

    Returns:
        The number of lines for each status.

    Raises:
        MachineStateError: On battery power or in Low Power Mode. No line is written.
    """
    counts: Counter = Counter()
    for index, (round_, variant, backend, tf32) in enumerate(plan, start=1):
        config = bench_config(round_, variant, priority)
        machine = read_machine_state()
        check_machine_state(machine)
        need = required_bench_bytes(config)
        available = available_memory_bytes()
        fits = available >= need
        label = f"{backend} {config.label}"
        logger.info(
            f"memory check: {label}: needs {need / GB:.1f} GB, "
            f"{available / GB:.1f} GB available, {'ok' if fits else 'does not fit'}"
        )
        if fits:
            worker = run_worker_subprocess(config, backend, tf32, bench_dir)
        else:
            reason = f"needs {need / GB:.1f} GB, {available / GB:.1f} GB available"
            worker = {"status": "skipped", "reason": reason}
        line = result_line(config, worker, need, available, machine)
        position = 1 + [entry[2] for entry in plan if entry[:2] == (round_, variant)].index(backend)
        line |= {
            "priority": priority,
            "backend": backend,
            "round": round_,
            "position": position,
            "tf32": tf32,
            "mlx_peak_bytes": worker.get("mlx_peak_bytes"),
            "torch_imported": worker.get("torch_imported"),
            "mlx_memory_limit_bytes": worker.get("mlx_memory_limit_bytes"),
            "mps_watermarks": worker.get("mps_watermarks"),
            "output_transfer": worker.get("output_transfer"),
            **{key: worker.get(key) for key in load_report_fields(None)},
        }
        results.parent.mkdir(parents=True, exist_ok=True)
        with results.open("a") as file:
            file.write(json.dumps(line) + "\n")
        counts[line["status"]] += 1
        summary = line["reason"].splitlines()[-1] if line["reason"] else ""
        if line["status"] == "done":
            summary = (
                f"valid {line['valid']}, frame {line['frame']['throughput_ips']:.1f} img/s, "
                f"patch {line['patch']['throughput_ips']:.1f} img/s, "
                f"peak {line['peak_bytes'] / GB:.2f} GB"
            )
        logger.info(f"[{index}/{len(plan)}] {label}: {line['status']}: {summary}")
    return counts


def main() -> None:
    """Parse the arguments and run the benchmark."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--rounds", type=int, default=3, help="the number of rounds")
    parser.add_argument("--priority", choices=["P1", "P2"], default="P1", help="P1 or P2")
    parser.add_argument(
        "--dtype-modes", nargs="+", choices=DTYPE_MODE_NAMES, help="P2 only: fewer dtype modes"
    )
    parser.add_argument("--dry-run", action="store_true", help="print the plan, load no model")
    parser.add_argument("--results", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        worker_main(args.worker)
        return
    if args.dtype_modes and args.priority != "P2":
        parser.error("--dtype-modes needs --priority P2")
    results = args.results or DEFAULT_RESULTS[args.priority]
    plan = worker_plan(args.rounds, args.priority, args.dtype_modes)
    if args.dry_run:
        dry_run(plan, args.priority)
        return
    bench_dir = bench_dir_of(results)
    bench_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
        handlers=[logging.StreamHandler(sys.stderr), logging.FileHandler(bench_dir / "driver.log")],
    )
    logger.info(f"run of {len(plan)} workers")
    try:
        counts = run_plan(plan, results, bench_dir, args.priority)
    except MachineStateError as error:
        logger.error(f"stop: {error}")
        sys.exit(1)
    logger.info(f"end of run: {dict(counts)}")


if __name__ == "__main__":
    main()
