"""The driver of all Step 6c benchmarks: the conversion, the parity and the compute plan, the
matrix, the paired runs, the two-stream run and the result table (Section 7 of
specs/06c-neural-engine/plan.html).

Each worker runs in its own subprocess, one after the other. Before each worker the driver reads
the machine state, stops on battery power or in Low Power Mode, and checks the free memory against
the need of the worker. A worker that does not fit gives a "skipped" line. Before and after each
worker the driver writes the heavy jobs outside the GPU lock into other_heavy_jobs_seen. The driver
takes the GPU lock in main; a dry run takes no lock and loads no model.

Results: results/coreml.jsonl (every phase; the field "phase" tells them apart) and
results/coreml-table.md. The per-batch files of the mlx workers and driver.log:
tests/fixtures/generated/bench/coreml/<subcommand>/.

Usage (always through the GPU lock and with the coreml dependency group):
    uv run --group coreml python scripts/bench_coreml.py convert
    uv run --group coreml python scripts/bench_coreml.py parity
    uv run --group coreml python scripts/bench_coreml.py matrix
    uv run --group coreml python scripts/bench_coreml.py pairs --rounds 5 --run 1
    uv run --group coreml python scripts/bench_coreml.py streams --rounds 3
    uv run --group coreml python scripts/bench_coreml.py table
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from dinov3_apple.bench import (
    MAIN_MODEL_SIZES,
    PRIORITY,
    BenchConfig,
    required_bench_bytes,
    result_line,
    run_worker_subprocess,
)
from dinov3_apple.gpu_lock import HEAVY_JOB_PATTERN, LOCK_REGISTRY, heavy_job, other_heavy_jobs
from dinov3_apple.machine import (
    MachineState,
    MachineStateError,
    check_machine_state,
    read_machine_state,
)
from dinov3_apple.memory import GB, available_memory_bytes, send_checks_to

REPO_ROOT = Path(__file__).parents[1]
RESULTS = REPO_ROOT / "results" / "coreml.jsonl"
TABLE = REPO_ROOT / "results" / "coreml-table.md"
BENCH_ROOT = REPO_ROOT / "tests" / "fixtures" / "generated" / "bench" / "coreml"
LOG_FORMAT = "%(asctime)s %(name)s %(message)s"
RUN = "S6c"
CORE_ML_MODE = "fp16-cast"  # Core ML lines: fp16 weights, fp16 compute precision (ADR 0003)
WORKER_TIMEOUT_S = 1200
STDERR_LINES = 20
WATERMARKS = {"PYTORCH_MPS_HIGH_WATERMARK_RATIO": "0.8", "PYTORCH_MPS_LOW_WATERMARK_RATIO": "0.6"}
CORE_ML_BACKENDS = ["coreml-ne", "coreml-gpu", "coreml-all"]
PARITY_SIDES = [224, 448, 512]
MATRIX_SIDES = [224, 512]
MATRIX_BATCH_SIZES = [1, 8]
PLAN_NEED_BYTES = 2 * 10**9
STREAM_CASES = ["ne", "mlx", "both"]
STREAM_SECONDS = 10.0
STREAM_LEAD_S = 2.0
STREAM_READY_TIMEOUT_S = 900.0

logger = logging.getLogger("bench_coreml")

Spec = dict
Runner = Callable[[Spec, Path], dict]


def mlx_mode(model_size: str, side: int, batch_size: int) -> str:
    """Return the dtype mode of the mlx side: the fastest valid fp16 dtype mode of the Step 6
    matrix (Section 3.5): fp16-cast for ViT-S/16 to ViT-B/16 and at P2, else fp16-autocast."""
    if (model_size, side, batch_size) == PRIORITY["P2"][:3]:
        return "fp16-cast"
    return "fp16-autocast" if model_size in ("ViT-L/16", "ViT-H+/16") else "fp16-cast"


def _rotate(items: list, offset: int) -> list:
    if not items:
        return items
    offset %= len(items)
    return items[offset:] + items[:offset]


def coreml_spec(
    phase: str,
    backend: str,
    model_size: str,
    side: int,
    batch_size: int,
    rewrite: bool = True,
    **fields: object,
) -> Spec:
    """Return one worker spec. backend: a Core ML backend, "mlx", "convert" or "plan".

    A coreml-ne worker at batch size 8 runs the batch-size-1 package with the batch API (F4 of
    explore.html, plan amendment of 2026-11-08), unless the spec sets ne_batch_form "package".
    """
    if backend == "coreml-ne" and batch_size > 1 and phase != "convert":
        fields.setdefault("ne_batch_form", "batch-api")
    return {
        "phase": phase,
        "backend": backend,
        "model_size": model_size,
        "side": side,
        "batch_size": batch_size,
        "rewrite": rewrite,
        **fields,
    }


def spec_config(spec: Spec) -> BenchConfig:
    """Return the configuration of one spec: Core ML lines run fp16-cast; mlx compiled."""
    if spec["backend"] == "mlx":
        mode = mlx_mode(spec["model_size"], spec["side"], spec["batch_size"])
        return BenchConfig(RUN, spec["model_size"], mode, spec["side"], spec["batch_size"], True)
    return BenchConfig(RUN, spec["model_size"], CORE_ML_MODE, spec["side"], spec["batch_size"])


def stream_schedule(rounds: int) -> list[list[str]]:
    """Return the cases of each round of the two-stream run: the order rotates (Section 3.6)."""
    return [_rotate(STREAM_CASES, index) for index in range(rounds)]


# The plans of the subcommands.


def convert_plan(model_sizes: list[str] | None = None) -> list[Spec]:
    """J1: the 26 packages (Table 2) and the deferred line of ViT-7B/16."""
    specs = []
    for model_size in model_sizes or MAIN_MODEL_SIZES:
        for side in PARITY_SIDES:
            specs.append(coreml_spec("convert", "convert", model_size, side, 1))
        for side in MATRIX_SIDES:
            specs.append(coreml_spec("convert", "convert", model_size, side, 8))
    specs.append(coreml_spec("convert", "convert", "ViT-S/16", 224, 1, rewrite=False))
    specs.append(coreml_spec("convert", "deferred", "ViT-7B/16", 224, 1))
    return specs


def parity_plan(model_sizes: list[str] | None = None) -> list[Spec]:
    """J2: the 15 parity packages on the 3 compute units, the plain ViT-S/16 package (F1) and the
    compute plan of each of the 26 packages on CPU_AND_NE."""
    specs = []
    for model_size in model_sizes or MAIN_MODEL_SIZES:
        for side in PARITY_SIDES:
            for backend in CORE_ML_BACKENDS:
                specs.append(coreml_spec("parity", backend, model_size, side, 1))
    for backend in CORE_ML_BACKENDS:
        specs.append(coreml_spec("parity", backend, "ViT-S/16", 224, 1, rewrite=False))
    for spec in convert_plan(model_sizes):
        if spec["backend"] == "convert":
            specs.append(spec | {"phase": "plan", "backend": "plan"})
    return specs


def matrix_plan(model_sizes: list[str] | None = None) -> list[Spec]:
    """J3: one paired group (coreml-ne, coreml-gpu, mlx) for each model size, input resolution
    and batch size, in a rotated order (ADR 0004)."""
    specs = []
    group = 0
    for model_size in model_sizes or MAIN_MODEL_SIZES:
        for side in MATRIX_SIDES:
            for batch_size in MATRIX_BATCH_SIZES:
                backends = ["coreml-ne", "coreml-gpu", "mlx"]
                if batch_size > 1:  # the batch-size-8 package on the Neural Engine: is F4 gone?
                    backends.append("coreml-ne-package")
                for position, backend in enumerate(_rotate(backends, group)):
                    fields: dict = {"group": group, "position": position}
                    if backend == "coreml-ne-package":
                        backend, fields["ne_batch_form"] = "coreml-ne", "package"
                    specs.append(
                        coreml_spec("matrix", backend, model_size, side, batch_size, **fields)
                    )
                group += 1
    return specs


def pairs_plan(
    priorities: list[str], batch_sizes: list[int], rounds: int, run_id: str
) -> list[Spec]:
    """J4: paired groups at P1 and P2: coreml-ne, coreml-gpu and mlx, plus coreml-all at batch
    size 8, in an order that rotates with the round."""
    specs = []
    for round_ in range(rounds):
        for priority in priorities:
            model_size, side, _, output = PRIORITY[priority]
            for batch_size in batch_sizes:
                backends = ["coreml-ne", "coreml-gpu", "mlx"]
                if batch_size == 8:
                    backends.append("coreml-all")
                for position, backend in enumerate(_rotate(backends, round_)):
                    specs.append(
                        coreml_spec(
                            "pairs",
                            backend,
                            model_size,
                            side,
                            batch_size,
                            run_id=run_id,
                            group=f"{priority}-b{batch_size}",
                            round=round_,
                            position=position,
                            priority=priority,
                            output=output,
                        )
                    )
    if "P1" in priorities and 8 in batch_sizes:
        model_size, side, _, output = PRIORITY["P1"]
        specs.append(
            coreml_spec(
                "pairs",
                "coreml-ne",
                model_size,
                side,
                8,
                run_id=run_id,
                group="P1-b8-f4",
                round=0,
                position=0,
                priority="P1",
                output=output,
                ne_batch_form="package",
            )
        )
    return specs


def ne_batch_size(model_size: str, side: int, lines: list[dict]) -> int:
    """Return the faster batch size (1 or 8) of coreml-ne in the matrix lines; 1 without lines."""
    best: dict[int, float] = {}
    for line in lines:
        if (
            line.get("phase") == "matrix"
            and line["backend"] == "coreml-ne"
            and line["model_size"] == model_size
            and line["side"] == side
            and line["status"] == "done"
            and line.get("valid")
        ):
            best[line["batch_size"]] = line["patch"]["throughput_ips"]
    return max(best, key=best.__getitem__) if best else 1


def streams_plan(
    priorities: list[str], batch_sizes: list[int], rounds: int, run_id: str, lines: list[dict]
) -> list[Spec]:
    """J5: the two-stream run. The mlx side runs the batch size of the group; the Neural Engine
    side runs the faster batch size of the matrix (Section 3.6)."""
    specs = []
    for priority in priorities:
        model_size, side, _, output = PRIORITY[priority]
        ne_batch = ne_batch_size(model_size, side, lines)
        for batch_size in batch_sizes:
            for round_, cases in enumerate(stream_schedule(rounds)):
                for position, case in enumerate(cases):
                    specs.append(
                        coreml_spec(
                            "streams",
                            "streams",
                            model_size,
                            side,
                            batch_size,
                            run_id=run_id,
                            group=f"{priority}-b{batch_size}",
                            round=round_,
                            position=position,
                            priority=priority,
                            output=output,
                            case=case,
                            ne_batch_size=ne_batch,
                        )
                    )
    return specs


# The memory need and the environment of a worker.


def spec_need(spec: Spec) -> int:
    """Return the memory need of one worker."""
    from dinov3_apple.coreml import convert_need_bytes

    backend = spec["backend"]
    if backend == "deferred":
        return 0
    if backend == "convert":
        return convert_need_bytes(spec["model_size"])
    if backend == "plan":
        return PLAN_NEED_BYTES
    config = spec_config(spec)
    if backend == "mlx":
        return required_bench_bytes(config, backend="mlx")
    if backend == "streams":
        ne = BenchConfig(RUN, spec["model_size"], CORE_ML_MODE, spec["side"], spec["ne_batch_size"])
        mlx = spec_config(spec | {"backend": "mlx"})
        return required_bench_bytes(ne) + required_bench_bytes(mlx, backend="mlx")
    return required_bench_bytes(config)  # the torch-mps need: weights, activations and scores


def worker_env(base: dict[str, str]) -> dict[str, str]:
    """Return the environment of one worker: the MPS watermarks (every worker imports torch) and
    MLX_ENABLE_TF32=0. The child inherits DINOV3_GPU_LOCK, so it does not wait for the driver."""
    env = {k: v for k, v in base.items() if k != "MLX_ENABLE_TF32"}
    return env | WATERMARKS | {"MLX_ENABLE_TF32": "0"}


# The run.


def run_worker(spec: Spec, bench_dir: Path) -> dict:
    """Run one worker in a new subprocess and return its result."""
    if spec["backend"] == "deferred":
        return {"status": "deferred", "reason": "ViT-7B/16 is deferred (D6 of explore.html)"}
    if spec["backend"] == "streams":
        return run_streams_case(spec, bench_dir)
    command = [
        sys.executable,
        __file__,
        "--worker",
        json.dumps(spec | {"bench_dir": str(bench_dir)}),
    ]
    return run_worker_subprocess(
        command, worker_env(dict(os.environ)), WORKER_TIMEOUT_S, STDERR_LINES
    )


SPEC_KEYS = [
    "phase",
    "run_id",
    "group",
    "round",
    "position",
    "priority",
    "output",
    "case",
    "ne_batch_size",
    "ne_batch_form",
]


def spec_line(spec: Spec, worker: dict, need: int, available: int, machine: MachineState) -> dict:
    """Return the JSON line of one worker: the fields of LINE_FIELDS, the fields of Table 3 and
    the pairing fields of the spec."""
    line = result_line(spec_config(spec), worker, need, available, machine, spec["backend"])
    for key in SPEC_KEYS:
        if key in spec:
            line[key] = spec[key]
    for key in ["compute_unit", "package", "rewrite", "first_load_s", "cached_load_s"]:
        line.setdefault(key, None)
    line["plain"] = spec.get("rewrite") is False
    return line


def _summary(line: dict) -> str:
    if line["status"] != "done":
        return (line.get("reason") or "").splitlines()[-1] if line.get("reason") else ""
    if line["phase"] == "convert":
        return f"trace {line['trace_s']:.1f} s, convert {line['convert_s']:.1f} s"
    if line["phase"] == "plan":
        return f"Neural Engine share {line['neural_engine_share']:.3f}"
    if line["phase"] == "streams":
        return f"{line['case']}: total {line['total_ips']:.1f} img/s"
    parity = line.get("parity") or {}
    text = f"valid {line.get('valid')}, frame {parity.get('frame')}"
    if isinstance(line.get("patch"), dict):
        text += f", {line['patch']['throughput_ips']:.1f} img/s"
    return text


def run_specs(
    specs: list[Spec],
    results: Path,
    bench_dir: Path,
    runner: Runner = run_worker,
    pattern: str = HEAVY_JOB_PATTERN,
    registry: Path = LOCK_REGISTRY,
) -> Counter:
    """Run the workers one after the other and append one line for each.

    Raises:
        MachineStateError: On battery power or in Low Power Mode. No line is written.
    """
    counts: Counter = Counter()
    for index, spec in enumerate(specs, start=1):
        machine = read_machine_state()
        check_machine_state(machine)
        need = spec_need(spec)
        available = available_memory_bytes()
        fits = available >= need
        label = (
            f"{spec['phase']} {spec['backend']} {spec['model_size']} {spec['side']}px "
            f"b{spec['batch_size']}{'' if spec.get('rewrite', True) else ' plain'}"
            f"{' ' + spec['case'] if 'case' in spec else ''}"
        )
        logger.info(
            f"memory check: {label}: needs {need / GB:.1f} GB, "
            f"{available / GB:.1f} GB available, {'ok' if fits else 'does not fit'}"
        )
        seen = other_heavy_jobs(pattern, registry)
        if fits:
            worker = runner(spec, bench_dir)
        else:
            worker = {
                "status": "skipped",
                "reason": f"needs {need / GB:.1f} GB, {available / GB:.1f} GB available",
            }
        seen += other_heavy_jobs(pattern, registry)
        line = spec_line(spec, worker, need, available, machine)
        line["other_heavy_jobs_seen"] = sorted(set(seen))
        results.parent.mkdir(parents=True, exist_ok=True)
        with results.open("a") as file:
            file.write(json.dumps(line) + "\n")
        counts[line["status"]] += 1
        logger.info(f"[{index}/{len(specs)}] {label}: {line['status']}: {_summary(line)}")
    return counts


def read_lines(path: Path) -> list[dict]:
    """Return the JSON lines of a results file, or [] if it does not exist."""
    if not path.exists():
        return []
    return [json.loads(text) for text in path.read_text().splitlines() if text.strip()]


def _identity(item: dict) -> tuple:
    return (
        item["phase"],
        item["backend"],
        item["model_size"],
        item["side"],
        item["batch_size"],
        item.get("rewrite") is not False and item.get("plain") is not True,
    )


def skip_done(specs: list[Spec], lines: list[dict]) -> list[Spec]:
    """Leave out each spec whose latest line has the status "done" or "deferred"."""
    latest = {_identity(line): line["status"] for line in lines}
    return [spec for spec in specs if latest.get(_identity(spec)) not in ("done", "deferred")]


# The two-stream run (Section 3.6).


def run_streams_case(spec: Spec, bench_dir: Path) -> dict:
    """Run one case of the two-stream run: the Neural Engine side, the mlx side or both.

    Each side loads and warms up in its own process, then writes a ready file and waits. When all
    ready files exist, the driver writes a start time STREAM_LEAD_S ahead. Each side counts its
    batches from the start time for STREAM_SECONDS.
    """
    folder = (
        bench_dir / "streams" / f"{spec['run_id']}-{spec['group']}-r{spec['round']}-{spec['case']}"
    )
    folder.mkdir(parents=True, exist_ok=True)
    start_file = folder / "start"
    start_file.unlink(missing_ok=True)
    sides = ["ne", "mlx"] if spec["case"] == "both" else [spec["case"]]
    processes = {}
    for side in sides:
        ready = folder / f"ready-{side}"
        ready.unlink(missing_ok=True)
        worker = spec | {
            "stream_side": side,
            "ready_file": str(ready),
            "start_file": str(start_file),
            "bench_dir": str(bench_dir),
        }
        command = [sys.executable, __file__, "--worker", json.dumps(worker)]
        processes[side] = subprocess.Popen(
            command,
            env=worker_env(dict(os.environ)),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    deadline = time.time() + STREAM_READY_TIMEOUT_S
    while not all((folder / f"ready-{side}").exists() for side in sides):
        if time.time() > deadline or any(p.poll() is not None for p in processes.values()):
            for process in processes.values():
                process.kill()
            errors = {s: p.communicate()[1][-2000:] for s, p in processes.items()}
            return {"status": "error", "reason": f"a stream side did not get ready: {errors}"}
        time.sleep(0.05)
    start_at = time.time() + STREAM_LEAD_S
    start_file.write_text(repr(start_at))
    results = {}
    for side, process in processes.items():
        out, err = process.communicate(timeout=STREAM_SECONDS + 600)
        found = [json.loads(t) for t in out.splitlines() if t.startswith("{")]
        if process.returncode != 0 or not found:
            return {
                "status": "error",
                "reason": f"{side}: exit {process.returncode}: {err[-2000:]}",
            }
        results[side] = found[-1]
    total = sum(r["throughput_ips"] for r in results.values())
    return {
        "status": "done",
        "reason": "",
        "sides": results,
        "total_ips": total,
        "start_at": start_at,
        "time": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def _stream_loop(step: Callable[[], object], spec: Spec) -> dict:
    """Warm up, write the ready file, wait for the start time, count batches for STREAM_SECONDS."""
    from dinov3_apple.bench import WARM_UP
    from dinov3_apple.coreml import _time_calls

    _time_calls(step, WARM_UP)
    Path(spec["ready_file"]).write_text("ready")
    start_file = Path(spec["start_file"])
    while not start_file.exists():
        time.sleep(0.01)
    start_at = float(start_file.read_text())
    while time.time() < start_at:
        time.sleep(0.001)
    end_at = start_at + STREAM_SECONDS
    times = []
    while time.time() < end_at:
        begin = time.perf_counter()
        step()
        times.append(time.perf_counter() - begin)
    elapsed = time.time() - start_at
    return {"batches": len(times), "elapsed_s": elapsed, "median_s": float(np.median(times))}


def stream_worker(spec: Spec) -> dict:
    """One side of the two-stream run, in its own process."""
    from dinov3_apple.bench import make_batch
    from dinov3_apple.stored import read_inputs

    side = spec["stream_side"]
    if side == "ne":
        from dinov3_apple.coreml import load_package, package_path

        batch_size = spec["ne_batch_size"]
        path = package_path(spec["model_size"], spec["side"], 1)
        model, _, _ = load_package(path, "CPU_AND_NE")
        batch = make_batch(read_inputs(), spec["side"], batch_size)
        if batch_size == 1:
            feed: object = {"pixels": batch}
        else:  # the batch API on the batch-size-1 package (F4)
            feed = [{"pixels": batch[i : i + 1]} for i in range(batch_size)]
        result = _stream_loop(lambda: model.predict(feed), spec)
        result |= {"backend": "coreml-ne", "package": path.name, "ne_batch_form": "batch-api"}
    else:
        import mlx.core as mx

        from dinov3_apple.mlx_bench import time_batch_mlx
        from dinov3_apple.mlx_model import compile_backbone
        from dinov3_apple.mlx_weights import load_mlx_backbone

        batch_size = spec["batch_size"]
        mode = mlx_mode(spec["model_size"], spec["side"], batch_size)
        model = load_mlx_backbone(spec["model_size"], mode, tf32=False)
        forward = compile_backbone(model)
        batch = make_batch(read_inputs(), spec["side"], batch_size)
        mx.set_memory_limit(
            required_bench_bytes(spec_config(spec | {"backend": "mlx"}), backend="mlx")
        )
        result = _stream_loop(lambda: time_batch_mlx(forward, batch, mode), spec)
        result |= {"backend": "mlx", "dtype_mode": mode}
    result |= {
        "batch_size": batch_size,
        "throughput_ips": result["batches"] * batch_size / result["elapsed_s"],
    }
    return result


# The worker side.


def worker_main(spec_json: str) -> None:
    """Run one worker in this process and print the result as one JSON line."""
    send_checks_to(sys.stderr)
    spec = json.loads(spec_json)
    backend = spec["backend"]
    try:
        if backend == "streams":
            result = stream_worker(spec)
        elif backend == "convert":
            from dinov3_apple.coreml import convert_backbone

            result = {"status": "done", "reason": ""} | convert_backbone(
                spec["model_size"], spec["side"], spec["batch_size"], spec["rewrite"]
            )
        elif backend == "plan":
            from dinov3_apple.coreml import (
                compiled_path,
                compute_plan_report,
                package_path,
            )

            path = package_path(
                spec["model_size"], spec["side"], spec["batch_size"], spec["rewrite"]
            )
            result = {"status": "done", "reason": "", "package": path.name}
            result |= compute_plan_report(compiled_path(path), "CPU_AND_NE")
        elif backend == "mlx":
            from dinov3_apple.mlx_bench import run_mlx_worker

            result = run_mlx_worker(
                spec_config(spec),
                bench_dir=Path(spec["bench_dir"]) / "mlx",
                run_id=spec.get("run_id"),
            )
        else:
            from dinov3_apple.coreml import run_coreml

            result = run_coreml(
                spec["model_size"],
                spec["side"],
                spec["batch_size"],
                backend,
                rewrite=spec["rewrite"],
                parity_only=spec["phase"] == "parity",
                batch_api=spec.get("ne_batch_form") == "batch-api",
            )
    except MemoryError as error:
        result = {"status": "skipped", "reason": f"load check in the worker: {error}"}
    except (RuntimeError, ValueError, TypeError, KeyError, OSError) as error:
        result = {"status": "error", "reason": f"{type(error).__name__}: {error}"}
    print(json.dumps(result))


# The table.


def _ratio_rows(lines: list[dict]) -> list[dict]:
    """Return the ratio of each Core ML line to the mlx line of the same group and round."""
    mlx = {}
    for line in lines:
        if line["backend"] == "mlx" and line["status"] == "done":
            mlx[(line.get("run_id"), line["group"], line.get("round"))] = line
    rows = []
    for line in lines:
        if line["backend"] not in CORE_ML_BACKENDS:
            continue
        partner = mlx.get((line.get("run_id"), line["group"], line.get("round")))
        ratio = None
        if line["status"] == "done" and partner is not None:
            output = line.get("output") or "patch"
            ratio = line[output]["throughput_ips"] / partner[output]["throughput_ips"]
        rows.append({"line": line, "mlx": partner, "ratio": ratio})
    return rows


def _fmt(value: float | None, digits: int = 1) -> str:
    return "–" if value is None else f"{value:,.{digits}f}"


def _parity_text(parity: dict | None) -> str:
    if not parity:
        return "–"
    return f"{parity['frame']:.4f} / {parity['patch_mean']:.4f} / {parity['patch_p01']:.4f}"


def _latest(lines: list[dict], phase: str) -> list[dict]:
    found: dict[tuple, dict] = {}
    for line in lines:
        if line.get("phase") == phase:
            extra = (
                line.get("run_id"),
                line.get("round"),
                line.get("case"),
                line.get("ne_batch_form"),
            )
            found[_identity(line) + extra] = line
    return list(found.values())


def _size_name(line: dict) -> str:
    return line["model_size"] + (" (plain graph)" if line.get("plain") else "")


def table_text(lines: list[dict]) -> str:
    """Return the Markdown text of results/coreml-table.md."""
    from dinov3_apple.speedup import pass_verdict, spread_between_runs

    out = [
        "# Core ML on the Neural Engine",
        "",
        (
            "Made by `scripts/bench_coreml.py table` from `results/coreml.jsonl` "
            f"on {datetime.now(UTC).date()}. Step 6c (specs/06c-neural-engine/plan.html). Core ML "
            "packages run with fp16 compute precision; the mlx side runs the compiled forward pass in "
            "the fastest valid fp16 dtype mode of Step 6. Parity: frame / patch mean / patch 1st "
            "percentile, the worst of the 3 test images (parity) or image 0 (speed)."
        ),
        "",
        "## 1 Conversion",
        "",
        "| Package | Status | Trace s | Convert s | Save s | Compile s | Size MB | Ops | Peak footprint GB |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for line in _latest(lines, "convert"):
        ops = sum((line.get("ops") or {}).values()) if line.get("ops") else None
        out.append(
            f"| {line.get('package') or line['model_size']} | {line['status']}"
            f"{': ' + line['reason'] if line['status'] != 'done' and line.get('reason') else ''} | "
            f"{_fmt(line.get('trace_s'))} | {_fmt(line.get('convert_s'))} | "
            f"{_fmt(line.get('save_s'))} | {_fmt(line.get('compile_s'))} | "
            f"{_fmt((line.get('package_bytes') or 0) / 1e6 if line.get('package_bytes') else None)} | "
            f"{ops if ops is not None else '–'} | "
            f"{_fmt(line.get('footprint_peak_bytes') / 1e9 if line.get('footprint_peak_bytes') else None, 2)} |"
        )
    out += [
        "",
        "Table 1. The conversion of each package (job J1).",
        "",
        "## 2 Parity",
        "",
        "| Model size | Input resolution | Backend | Status | Valid | Parity | Largest token value | First load s | Cached load s | Peak footprint GB |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for line in _latest(lines, "parity"):
        images = line.get("images") or {}
        largest = max((v["max_abs"] for v in images.values()), default=None)
        out.append(
            f"| {_size_name(line)} | {line['side']} | {line['backend']} | {line['status']} | "
            f"{line.get('valid')} | {_parity_text(line.get('parity'))} | {_fmt(largest)} | "
            f"{_fmt(line.get('first_load_s'), 2)} | {_fmt(line.get('cached_load_s'), 3)} | "
            f"{_fmt(line['peak_bytes'] / 1e9 if line.get('peak_bytes') else None, 2)} |"
        )
    out += [
        "",
        (
            "Table 2. The parity of each package on each compute unit at batch size 1 (job J2). "
            "Thresholds of ADR 0003 for fp16: 0.999 / 0.999 / 0.99."
        ),
        "",
        "## 3 Compute plan",
        "",
        "| Package | Ops | Ops per device | Neural Engine cost share | Ops off the Neural Engine |",
        "| --- | --- | --- | --- | --- |",
    ]
    for line in _latest(lines, "plan"):
        if line["status"] != "done":
            out.append(f"| {line.get('package')} | {line['status']} | – | – | – |")
            continue
        off = Counter(row["op_type"] for row in line["off_neural_engine"])
        out.append(
            f"| {line['package']} | {line['ops']} | "
            f"{', '.join(f'{k} {v}' for k, v in sorted(line['ops_per_device'].items()))} | "
            f"{line['neural_engine_share']:.3f} | "
            f"{', '.join(f'{k} {v}' for k, v in off.most_common()) or '–'} |"
        )
    out += [
        "",
        "Table 3. The compute plan of each package on CPU_AND_NE (job J2).",
        "",
        "## 4 Matrix",
        "",
        "| Model size | Input resolution | Batch size | Backend | Status | Valid | Parity (image 0) | img/s | Ratio to mlx | mlx dtype mode | mlx img/s | Reason |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in _ratio_rows(_latest(lines, "matrix")):
        line, partner = row["line"], row["mlx"]
        ips = line["patch"]["throughput_ips"] if line["status"] == "done" else None
        reason = (
            ""
            if line["status"] == "done" and line.get("valid")
            else (line.get("reason") or line.get("validity_note") or "")
        )
        out.append(
            f"| {line['model_size']} | {line['side']} | {line['batch_size']} | {line['backend']}"
            f"{' (b8 package)' if line.get('ne_batch_form') == 'package' else ''} | "
            f"{line['status']} | {line.get('valid')} | {_parity_text(line.get('parity'))} | "
            f"{_fmt(ips)} | {_fmt(row['ratio'], 3)} | "
            f"{partner['dtype_mode'] if partner else '–'} | "
            f"{_fmt(partner['patch']['throughput_ips']) if partner else '–'} | "
            f"{reason.splitlines()[-1] if reason else ''} |"
        )
    out += [
        "",
        (
            "Table 4. The matrix: patch embeddings, 1 round (job J3). The ratio is Core ML / mlx "
            "of the same paired group."
        ),
        "",
        "## 5 Paired runs at P1 and P2",
        "",
        "| Group | Backend | Run medians | Pooled median | Spread between runs | Verdict D21 | Core ML img/s (median) | mlx img/s (median) | Rounds |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    pair_rows = _ratio_rows([line for line in lines if line.get("phase") == "pairs"])
    groups: dict[tuple, list[dict]] = {}
    for row in pair_rows:
        if row["ratio"] is not None:
            key = (row["line"]["group"], row["line"]["backend"])
            groups.setdefault(key, []).append(row)
    for (group, backend), rows in sorted(groups.items()):
        runs: dict[str, list[float]] = {}
        for row in rows:
            runs.setdefault(row["line"]["run_id"], []).append(row["ratio"])
        medians = [float(np.median(v)) for _, v in sorted(runs.items())]
        pooled = float(np.median([row["ratio"] for row in rows]))
        output = rows[0]["line"].get("output") or "patch"
        out.append(
            f"| {group} | {backend} | {', '.join(f'{m:.3f}' for m in medians)} | {pooled:.3f} | "
            f"{spread_between_runs(medians):.3f} | {pass_verdict(medians, pooled)} | "
            f"{np.median([r['line'][output]['throughput_ips'] for r in rows]):,.1f} | "
            f"{np.median([r['mlx'][output]['throughput_ips'] for r in rows]):,.1f} | {len(rows)} |"
        )
    out += [
        "",
        (
            "Table 5. The paired ratio Core ML / mlx at P1 (frame) and P2 (patch), median over the "
            "rounds of each run (run 1: build, run 2: verify). Report only: Step 6c has no speed target."
        ),
    ]
    for line in lines:
        if line.get("group") == "P1-b8-f4":
            out.append("")
            out.append(
                f"F4 evidence (run {line.get('run_id')}): the ViT-S/16 batch-size-8 package on "
                f"CPU_AND_NE gives valid {line.get('valid')}, parity "
                f"{_parity_text(line.get('parity'))}. The coreml-ne rows at batch size 8 run the "
                "batch-size-1 package with the batch API."
            )
    out += [
        "",
        "## 6 Two-stream run",
        "",
        "| Group | Run | Neural Engine batch size | ne alone img/s | mlx alone img/s | both: ne img/s | both: mlx img/s | both: total img/s | Stream ratio | Rounds |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    streams: dict[tuple, dict[int, dict[str, dict]]] = {}
    for line in lines:
        if line.get("phase") == "streams" and line["status"] == "done":
            key = (line["group"], line["run_id"], line["ne_batch_size"])
            streams.setdefault(key, {}).setdefault(line["round"], {})[line["case"]] = line
    for (group, run_id, ne_batch), rounds in sorted(streams.items()):
        full = [r for r in rounds.values() if set(r) == set(STREAM_CASES)]
        if not full:
            continue
        ne_alone = [r["ne"]["sides"]["ne"]["throughput_ips"] for r in full]
        mlx_alone = [r["mlx"]["sides"]["mlx"]["throughput_ips"] for r in full]
        both_ne = [r["both"]["sides"]["ne"]["throughput_ips"] for r in full]
        both_mlx = [r["both"]["sides"]["mlx"]["throughput_ips"] for r in full]
        ratios = [r["both"]["total_ips"] / r["mlx"]["sides"]["mlx"]["throughput_ips"] for r in full]
        out.append(
            f"| {group} | {run_id} | {ne_batch} | {np.median(ne_alone):,.1f} | "
            f"{np.median(mlx_alone):,.1f} | {np.median(both_ne):,.1f} | {np.median(both_mlx):,.1f} | "
            f"{np.median([a + b for a, b in zip(both_ne, both_mlx, strict=True)]):,.1f} | "
            f"{np.median(ratios):.3f} | {len(full)} |"
        )
    out += [
        "",
        (
            "Table 6. The two-stream run: Core ML on CPU_AND_NE and compiled mlx on the GPU, alone and "
            "at the same time, 10 s per case. Stream ratio = (ne + mlx, both) / (mlx alone), median "
            "over the rounds."
        ),
        "",
    ]
    return "\n".join(out)


# The command line.


def plan_of(args: argparse.Namespace) -> list[Spec]:
    """Return the workers of the subcommand."""
    sizes = args.model_sizes
    if args.subcommand == "convert":
        specs = convert_plan(sizes)
    elif args.subcommand == "parity":
        specs = parity_plan(sizes)
        if args.plain:
            specs = [spec for spec in specs if spec["rewrite"] is False]
    elif args.subcommand == "matrix":
        specs = matrix_plan(sizes)
    elif args.subcommand == "pairs":
        specs = pairs_plan(args.priority, args.batch_sizes, args.rounds, str(args.run))
    else:
        lines = read_lines(RESULTS)
        specs = streams_plan(args.priority, args.batch_sizes, args.rounds, str(args.run), lines)
    if args.skip_done:
        specs = skip_done(specs, read_lines(args.results))
    return specs


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    sub = parser.add_subparsers(dest="subcommand")
    for name in ["convert", "parity", "matrix", "pairs", "streams", "table"]:
        p = sub.add_parser(name)
        p.add_argument("--results", type=Path, default=RESULTS)
        p.add_argument("--model-sizes", nargs="+", default=None)
        p.add_argument("--priority", nargs="+", default=["P1", "P2"])
        p.add_argument("--batch-sizes", nargs="+", type=int, default=[8, 1])
        p.add_argument("--rounds", type=int, default=5 if name == "pairs" else 3)
        p.add_argument("--run", default="1")
        p.add_argument("--skip-done", action="store_true")
        p.add_argument("--plain", action="store_true", help="parity: only the plain graph")
        p.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.worker is None and args.subcommand is None:
        parser.error("give a subcommand")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    """Parse the arguments and run the workers under the GPU lock."""
    args = parse_args(argv)
    if args.worker:
        worker_main(args.worker)
        return
    if args.subcommand == "table":
        TABLE.write_text(table_text(read_lines(args.results)))
        print(f"wrote {TABLE}")
        return
    specs = plan_of(args)
    if args.dry_run:
        for spec in specs:
            print(json.dumps(spec))
        print(f"{len(specs)} workers")
        return
    bench_dir = BENCH_ROOT / args.subcommand
    bench_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
        handlers=[logging.StreamHandler(sys.stderr), logging.FileHandler(bench_dir / "driver.log")],
    )
    with heavy_job(f"scripts/bench_coreml.py {args.subcommand}"):
        logger.info(f"run of {len(specs)} workers: {args.subcommand}")
        try:
            counts = run_specs(specs, args.results, bench_dir)
        except MachineStateError as error:
            logger.error(f"stop: {error}")
            sys.exit(1)
        logger.info(f"end of run: {dict(counts)}")


if __name__ == "__main__":
    main()
