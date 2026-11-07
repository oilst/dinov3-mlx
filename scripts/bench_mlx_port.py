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
driver writes them to driver.log. Since Step 6 the mlx worker comes from dinov3_apple.mlx_bench:
the frame embeddings move to the host in fp32, the patch embeddings in fp16 in the four fp16 and
bf16 dtype modes (D13 of specs/06-speed/plan.html). The driver takes the GPU lock.

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
import sys
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path

from dinov3_apple.bench import (
    DTYPE_MODE_NAMES,
    PRIORITY,
    BenchConfig,
    required_bench_bytes,
    result_line,
    run_worker_subprocess,
)
from dinov3_apple.gpu_lock import heavy_job
from dinov3_apple.machine import (
    MachineStateError,
    check_machine_state,
    read_machine_state,
)
from dinov3_apple.memory import (
    GB,
    InsufficientMemoryError,
    available_memory_bytes,
    send_checks_to,
)
from dinov3_apple.mlx_bench import (  # noqa: F401 (the tests of Steps 4 and 5 import them here)
    host_to_fp32,
    load_report_fields,
    output_transfer,
    run_mlx_worker,
    time_batch_mlx,
)

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
    """Return (round, batch size or dtype mode, backend, tf32) for each worker, in the order of the
    run.

    The second field is the batch size of P1 or the dtype mode of P2. The order of the two workers
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
    batch_sizes_or_dtype_modes = (
        BATCH_SIZES if priority == "P1" else (dtype_modes or DTYPE_MODE_NAMES)
    )
    plan = []
    for r in range(rounds):
        order = WORKERS[r % 2 :] + WORKERS[: r % 2]
        for batch_size_or_dtype_mode in batch_sizes_or_dtype_modes:
            plan += [(r + 1, batch_size_or_dtype_mode, backend, tf32) for backend, tf32 in order]
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


def bench_config(
    round_: int, batch_size_or_dtype_mode: int | str, priority: str = "P1"
) -> BenchConfig:
    """Return the configuration of one worker. The repeat number is the round minus 1.

    Args:
        round_: The round, from 1.
        batch_size_or_dtype_mode: The batch size of P1 or the dtype mode of P2.
        priority: "P1" or "P2".
    """
    repeat = round_ - 1
    if priority == "P1":
        batch_size = batch_size_or_dtype_mode
        return BenchConfig(RUN, MODEL_SIZE, DTYPE_MODE, SIDE, batch_size, repeat=repeat)
    model_size, side, batch_size, _ = PRIORITY[priority]
    dtype_mode = batch_size_or_dtype_mode
    return BenchConfig(RUN_P2, model_size, dtype_mode, side, batch_size, repeat=repeat)


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
            from dinov3_apple.torch_bench import run_worker as run_torch_worker

            result = run_torch_worker(config, bench_dir=bench_dir / "torch-mps")
            result |= {"mlx_peak_bytes": None, "mlx_memory_limit_bytes": None, "tf32": None}
            result |= {"output_transfer": "native", **load_report_fields(None)}
    except InsufficientMemoryError as error:
        result = {"status": "skipped", "reason": f"load check in the worker: {error}"}
    watermarks = {k: os.environ[k] for k in WATERMARKS if k in os.environ}
    result["mps_watermarks"] = watermarks or None
    result["torch_imported"] = "torch" in sys.modules
    print(json.dumps(result))


def run_worker(config: BenchConfig, backend: str, tf32: bool | None, bench_dir: Path) -> dict:
    """Run one worker in a new subprocess and return its result (run_worker_subprocess).

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
    env = worker_env(backend, tf32, os.environ)
    return run_worker_subprocess(command, env, WORKER_TIMEOUT_S, STDERR_LINES)


def dry_run(plan: list[tuple[int, int | str, str, bool | None]], priority: str = "P1") -> None:
    """Print each worker of the plan with its memory need. Load no model."""
    available = available_memory_bytes()
    print(f"available memory: {available / GB:.1f} GB")
    for round_, batch_size_or_dtype_mode, backend, tf32 in plan:
        config = bench_config(round_, batch_size_or_dtype_mode, priority)
        need = required_bench_bytes(config, backend=backend)
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
    for index, (round_, batch_size_or_dtype_mode, backend, tf32) in enumerate(plan, start=1):
        config = bench_config(round_, batch_size_or_dtype_mode, priority)
        machine = read_machine_state()
        check_machine_state(machine)
        need = required_bench_bytes(config, backend=backend)
        available = available_memory_bytes()
        fits = available >= need
        label = f"{backend} {config.label}"
        logger.info(
            f"memory check: {label}: needs {need / GB:.1f} GB, "
            f"{available / GB:.1f} GB available, {'ok' if fits else 'does not fit'}"
        )
        if fits:
            worker = run_worker(config, backend, tf32, bench_dir)
        else:
            reason = f"needs {need / GB:.1f} GB, {available / GB:.1f} GB available"
            worker = {"status": "skipped", "reason": reason}
        line = result_line(config, worker, need, available, machine, backend)
        pair = (round_, batch_size_or_dtype_mode)
        position = 1 + [entry[2] for entry in plan if entry[:2] == pair].index(backend)
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


def main(argv: list[str] | None = None) -> None:
    """Parse the arguments and run the benchmark. A run that is not a dry run takes the GPU lock.

    Args:
        argv: The arguments. None reads sys.argv.
    """
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
    args = parser.parse_args(argv)

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
    with heavy_job("scripts/bench_mlx_port.py"):
        logger.info(f"run of {len(plan)} workers")
        try:
            counts = run_plan(plan, results, bench_dir, args.priority)
        except MachineStateError as error:
            logger.error(f"stop: {error}")
            sys.exit(1)
        logger.info(f"end of run: {dict(counts)}")


if __name__ == "__main__":
    main()
