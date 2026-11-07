"""Measure the baseline: the throughput and the peak memory of the Hugging Face backbone on torch-mps.

The driver runs one worker subprocess for each configuration, one after the other. Before each
worker it reads the machine state, stops on battery power or in Low Power Mode, and checks the
free memory against the need of the configuration. A configuration that does not fit gives a
"skipped" line. Each result or skip is one JSON line in results/baseline-torch-mps.jsonl.
Sections 3.3 to 3.13 of specs/03-baseline/plan.html give the rules.

Usage:
    uv run python scripts/benchmark.py --dry-run                  # needs only, no model load
    uv run python scripts/benchmark.py                            # B1: five model sizes
    uv run python scripts/benchmark.py --model-size ViT-S/16      # B1 for one model size
    uv run python scripts/benchmark.py --only-skipped             # B1 lines that were skipped
    uv run python scripts/benchmark.py --priority --repeats 2     # B2
    uv run python scripts/benchmark.py --priority --compile       # B3
    uv run python scripts/benchmark.py --model-size ViT-7B/16     # B4, cast modes only
    uv run python scripts/benchmark.py --model-size ViT-S/16 --rounds 2   # 2 more runs of each
"""

import os

# The MPS watermarks must be set before torch starts (CLAUDE.md run rules).
WATERMARKS = {"PYTORCH_MPS_HIGH_WATERMARK_RATIO": "0.8", "PYTORCH_MPS_LOW_WATERMARK_RATIO": "0.6"}
os.environ.update(WATERMARKS)

import argparse
import json
import logging
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from dinov3_apple.bench import (
    ACTIVATION_FACTOR,
    DEFERRED_REASON,
    GIANT_MODEL_SIZE,
    MAIN_MODEL_SIZES,
    BenchConfig,
    add_rounds,
    bench_configurations,
    latest_lines,
    line_key,
    required_bench_bytes,
    result_line,
)
from dinov3_apple.bench import run_worker_subprocess as run_worker_command
from dinov3_apple.hub import MODEL_IDS, WeightsNotFoundError
from dinov3_apple.machine import (
    MachineStateError,
    check_machine_state,
    read_machine_state,
)
from dinov3_apple.memory import GB, InsufficientMemoryError, available_memory_bytes

REPO_ROOT = Path(__file__).parents[1]
RESULTS = REPO_ROOT / "results" / "baseline-torch-mps.jsonl"
BENCH_DIR = REPO_ROOT / "tests" / "fixtures" / "generated" / "bench"
DRIVER_LOG = BENCH_DIR / "driver.log"
WORKER_TIMEOUT_S = 3600
STDERR_LINES = 20

logger = logging.getLogger("benchmark")


def read_lines(path: Path) -> list[dict]:
    """Return the JSON lines of a results file, or [] if it does not exist."""
    if not path.exists():
        return []
    return [json.loads(text) for text in path.read_text().splitlines() if text.strip()]


def append_line(path: Path, line: dict) -> None:
    """Append one JSON line to a results file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as file:
        file.write(json.dumps(line) + "\n")


def worker_main(config_json: str) -> None:
    """Run one configuration in this process and print the result as one JSON line."""
    from dinov3_apple.torch_bench import run_worker

    config = BenchConfig(**json.loads(config_json))
    try:
        result = run_worker(config)
    except InsufficientMemoryError as error:
        result = {"status": "skipped", "reason": f"load check in the worker: {error}"}
    print(json.dumps(result))


def run_worker_subprocess(config: BenchConfig) -> dict:
    """Run one configuration in a new subprocess and return its result.

    A worker that exits with an error gives status "error" and the last lines of its stderr.
    """
    command = [sys.executable, __file__, "--worker", json.dumps(asdict(config))]
    return run_worker_command(command, os.environ | WATERMARKS, WORKER_TIMEOUT_S, STDERR_LINES)


def select_configs(args: argparse.Namespace, lines: list[dict]) -> list[BenchConfig]:
    """Return the configurations of this run from the command-line options."""
    model_sizes = sorted(args.model_size or MAIN_MODEL_SIZES, key=list(MODEL_IDS).index)
    configs = bench_configurations(
        model_sizes=model_sizes,
        priority=args.priority,
        repeats=args.repeats,
        compile=args.compile,
    )
    latest = {line_key(line): line for line in latest_lines(lines)}
    if args.only_skipped:
        configs = [c for c in configs if latest.get(c.key, {}).get("status") == "skipped"]
    if args.skip_done:
        configs = [c for c in configs if latest.get(c.key, {}).get("status") != "done"]
    return add_rounds(configs, lines, args.rounds)


def dry_run(configs: list[BenchConfig], factor: float) -> None:
    """Print the need of each configuration and the available memory. Load no model."""
    available = available_memory_bytes()
    print(f"available memory: {available / GB:.1f} GB, activation factor {factor}")
    fits = 0
    for config in configs:
        need = required_bench_bytes(config, factor)
        ok = need <= available
        fits += ok
        print(f"{config.label:42s} need {need / GB:5.1f} GB  {'fits' if ok else 'does not fit'}")
    largest = max((required_bench_bytes(c, factor) for c in configs), default=0)
    print(f"{fits} of {len(configs)} configurations fit. Largest need: {largest / GB:.1f} GB.")


def run_configs(configs: list[BenchConfig], factor: float, results: Path) -> Counter:
    """Run the configurations one after the other and append one line for each.

    Raises:
        MachineStateError: On battery power or in Low Power Mode. No line is written.
    """
    counts: Counter = Counter()
    for index, config in enumerate(configs, start=1):
        machine = read_machine_state()
        check_machine_state(machine)
        need = required_bench_bytes(config, factor)
        available = available_memory_bytes()
        fits = available >= need
        logger.info(
            f"memory check: {config.label}: needs {need / GB:.1f} GB, "
            f"{available / GB:.1f} GB available, {'ok' if fits else 'does not fit'}"
        )
        if fits:
            worker = run_worker_subprocess(config)
        else:
            reason = f"needs {need / GB:.1f} GB, {available / GB:.1f} GB available"
            worker = {"status": "skipped", "reason": reason}
        line = result_line(config, worker, need, available, machine)
        append_line(results, line)
        counts[line["status"]] += 1
        summary = line["reason"].splitlines()[-1] if line["reason"] else ""
        if line["status"] == "done":
            summary = (
                f"valid {line['valid']}, frame {line['frame']['throughput_ips']:.1f} img/s, "
                f"patch {line['patch']['throughput_ips']:.1f} img/s, "
                f"peak {line['peak_bytes'] / GB:.2f} GB"
            )
        logger.info(f"[{index}/{len(configs)}] {config.label}: {line['status']}: {summary}")
    return counts


def write_deferred(results: Path, lines: list[dict], factor: float) -> int:
    """Write one "deferred" line for each ViT-7B/16 configuration that has no line yet (D8)."""
    known = {line_key(line) for line in lines}
    machine = read_machine_state()
    written = 0
    for config in bench_configurations(model_sizes=[GIANT_MODEL_SIZE]):
        if config.key in known:
            continue
        deferred = BenchConfig("B1", *config.key)
        need = required_bench_bytes(deferred, factor)
        worker = {"status": "deferred", "reason": DEFERRED_REASON}
        append_line(results, result_line(deferred, worker, need, machine.available_bytes, machine))
        written += 1
    return written


def main() -> None:
    """Parse the arguments and run the benchmark."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--model-size", nargs="+", choices=list(MODEL_IDS), help="B1 or B4")
    parser.add_argument("--priority", action="store_true", help="P1 and P2 in five dtype modes")
    parser.add_argument("--repeats", type=int, default=0, help="with --priority: B2")
    parser.add_argument("--compile", action="store_true", help="with --priority: B3")
    parser.add_argument("--dry-run", action="store_true", help="print the needs, load no model")
    parser.add_argument(
        "--only-skipped", action="store_true", help="only configurations whose latest line skipped"
    )
    parser.add_argument(
        "--skip-done", action="store_true", help="leave out configurations with a done line"
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=0,
        help="run the configurations this many more times, round after round (repeat numbers)",
    )
    parser.add_argument(
        "--activation-factor",
        type=float,
        default=ACTIVATION_FACTOR,
        help=f"factor on the activation terms of the need, at least {ACTIVATION_FACTOR}",
    )
    parser.add_argument("--results", type=Path, default=RESULTS, help=argparse.SUPPRESS)
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        worker_main(args.worker)
        return
    if args.activation_factor < ACTIVATION_FACTOR:
        parser.error(f"the activation factor can only be raised above {ACTIVATION_FACTOR}")
    if (args.repeats or args.compile) and not args.priority:
        parser.error("--repeats and --compile need --priority")

    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(message)s",
        handlers=[logging.StreamHandler(sys.stderr), logging.FileHandler(DRIVER_LOG)],
    )
    lines = read_lines(args.results)
    configs = select_configs(args, lines)
    if args.dry_run:
        dry_run(configs, args.activation_factor)
        return
    logger.info(f"run of {len(configs)} configurations, activation factor {args.activation_factor}")
    try:
        counts = run_configs(configs, args.activation_factor, args.results)
    except MachineStateError as error:
        logger.error(f"stop: {error}")
        sys.exit(1)
    except WeightsNotFoundError as error:
        logger.error(f"stop: {error}")
        sys.exit(1)
    main_run = not args.priority and GIANT_MODEL_SIZE not in (args.model_size or [])
    if main_run:
        counts["deferred"] += write_deferred(
            args.results, read_lines(args.results), args.activation_factor
        )
    logger.info(f"end of run: {dict(counts)}")


if __name__ == "__main__":
    main()
