"""The driver of Step 6b: the Core ML pairs, the port-change variants, the gated Python ports and
the result table (Section 3 of specs/06b-prior-art/plan.html).

Each worker runs in its own subprocess, one after the other, as in scripts/bench_speed.py. Before
each worker the driver reads the machine state, stops on battery power or in Low Power Mode, and
checks the free memory against the need of the worker; a worker that does not fit gives a
"skipped" line. Before and after each worker the driver writes the heavy jobs outside the GPU lock
into the field other_heavy_jobs_seen. The driver takes the GPU lock in main; a dry run takes no
lock and imports no backend.

Third-party rules (ADR 0013): a Core ML worker starts as <venv-coreml>/bin/python -I
scripts/prior_art_coreml_worker.py '<spec>'; a port worker as <venv-mlx>/bin/python -I
scripts/prior_art_port_worker.py '<spec>', and only with --allow-third-party-python (Q1). The
driver refuses a worker file inside the prior-art folder, and it stops before the lock when the
venv of a job is missing.

Results: results/prior-art.jsonl and results/prior-art-table.md. The per-worker files and
driver.log: tests/fixtures/generated/bench/prior-art/<subcommand>/. With another --results file,
they go to the folder <stem>-batches/<subcommand>/ next to that file.

Usage:
    uv run python scripts/bench_prior_art.py coreml --rounds 5 --dry-run
    uv run python -m dinov3_apple.gpu_lock -- uv run python scripts/bench_prior_art.py coreml --rounds 5
    ... bench_prior_art.py coreml --units ALL --rounds 1
    ... bench_prior_art.py coreml --image-input
    ... bench_prior_art.py variants --parity [--model-sizes ViT-S/16]
    ... bench_prior_art.py variants --speed --rounds 3 [--priority P1]
    ... bench_prior_art.py ports --rounds 5 [--allow-third-party-python]
    uv run python scripts/bench_prior_art.py table
"""

import argparse
import json
import logging
import os
import sys
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from dinov3_apple import prior_art
from dinov3_apple.bench import BenchConfig, TimeBudget, required_bench_bytes, run_worker_subprocess
from dinov3_apple.gpu_lock import HEAVY_JOB_PATTERN, LOCK_REGISTRY, heavy_job, other_heavy_jobs
from dinov3_apple.hub import WeightsNotFoundError
from dinov3_apple.machine import MachineStateError, check_machine_state, read_machine_state
from dinov3_apple.memory import GB, InsufficientMemoryError, available_memory_bytes, send_checks_to
from dinov3_apple.parity import TEST_IMAGES
from dinov3_apple.prior_art import (
    BLOCKED_REASON,
    BUDGETS,
    EXPORTS,
    J5_COMMAND,
    PORT_CELLS,
    PRIOR_ART_DIR,
    RUN,
    export_outputs,
    export_parity,
    parity_passes,
    worst_parity,
)
from dinov3_apple.stored import INPUTS_FILE, reference_path

REPO_ROOT = Path(__file__).parents[1]
SCRIPTS = Path(__file__).parent
RESULTS = REPO_ROOT / "results" / "prior-art.jsonl"
TABLE = REPO_ROOT / "results" / "prior-art-table.md"
BENCH_ROOT = REPO_ROOT / "tests" / "fixtures" / "generated" / "bench" / "prior-art"
COREML_WORKER = SCRIPTS / "prior_art_coreml_worker.py"
PORT_WORKER = SCRIPTS / "prior_art_port_worker.py"
VENVS = {"export": "venv-coreml", "port": "venv-mlx"}
WORKER_TIMEOUT_S = 900
STDERR_LINES = 20
EXPORT_RESERVE_BYTES = 2 * 10**9
LOG_FORMAT = "%(asctime)s %(name)s %(message)s"
SIDE_KIND = {
    "ours": "ours",
    "export": "export",
    "variant-parity": "variant",
    "variant-speed": "variant",
    "port": "port",
}
# The fields of a worker result that go into its line, for each side kind.
OURS_FIELDS = [
    "valid",
    "finite",
    "parity",
    "validity_note",
    "load_time_s",
    "first_call_s",
    "compile_trace_s",
    "warm_up_batches",
    "timed_batches",
    "timed_s",
    "frame",
    "patch",
    "device",
    "output_dtype",
    "mlx_memory_limit_bytes",
    "mlx_peak_bytes",
    "mlx_timed_peak_bytes",
    "tf32",
    "versions",
    "start_time",
    "batch_file",
]

logger = logging.getLogger("bench_prior_art")

Spec = dict
Runner = Callable[[Spec, Path], dict]


class ThirdPartyRuleError(RuntimeError):
    """A third-party worker would break a rule of ADR 0013."""


def venv_python(kind: str) -> Path:
    """Return the interpreter of the venv of a third-party worker kind ("export" or "port")."""
    return PRIOR_ART_DIR / VENVS[kind] / "bin" / "python"


def check_worker_path(path: Path) -> None:
    """Refuse a worker file inside the prior-art folder: a worker is our own file in the repo.

    Raises:
        ThirdPartyRuleError: For a path inside PRIOR_ART_DIR.
    """
    if path.resolve().is_relative_to(PRIOR_ART_DIR.resolve()):
        raise ThirdPartyRuleError(f"the worker {path} is inside {PRIOR_ART_DIR}")


def missing_venvs(kinds: list[str]) -> list[Path]:
    """Return the interpreters of the venvs that the kinds need and that do not exist."""
    return [venv_python(kind) for kind in kinds if not venv_python(kind).exists()]


def spec_config(spec: Spec) -> BenchConfig | None:
    """Return the configuration of a worker of ours or of a variant, else None."""
    return BenchConfig(**spec["config"]) if "config" in spec else None


def _budget_dict(budget: TimeBudget) -> dict:
    return asdict(budget)


def worker_spec(spec: Spec, bench_dir: Path) -> dict:
    """Return the spec that a third-party worker gets: the files and the time budgets."""
    warm_up, timed = BUDGETS[spec.get("budget", "full")]
    name = f"{spec['job']}-r{spec.get('round', 1)}-p{spec.get('position', 0)}"
    fields = {
        "inputs_file": str(INPUTS_FILE),
        "outputs_file": str(bench_dir / "outputs" / f"{name}.safetensors"),
        "warm_up": _budget_dict(warm_up),
        "timed": _budget_dict(timed),
    }
    if spec["kind"] == "export":
        keys = ("package", "input", "units", "batch_size", "side", "pixels", "parity_only")
        return {key: spec[key] for key in keys} | fields
    clone, weights = port_files(spec)
    keys = ("candidate", "model", "dtype", "side", "batch_size", "pixels")
    limit = required_bench_bytes(port_config(spec), backend="mlx")
    return (
        {key: spec[key] for key in keys}
        | fields
        | {
            "clone": str(clone),
            "weights": str(weights),
            "limit_bytes": limit,
        }
    )


def port_files(spec: Spec) -> tuple[Path, Path]:
    """Return the source folder of the clone and the weights file of a port worker."""
    if spec["candidate"] == "C1":
        name = {
            "ViT-S/16": "vit_small_patch16_224.dinov3",
            "dinov2": "vit_small_patch14_518.dinov2",
        }
        repo = f"{name[spec['model']]}-mlxim"
        return (
            PRIOR_ART_DIR / "mlx-image" / "src" / "src",
            PRIOR_ART_DIR / "weights" / "mlx-vision" / repo / "model.safetensors",
        )
    return (
        PRIOR_ART_DIR / "mlx_dinov3" / "src",
        PRIOR_ART_DIR / "weights" / "ours-meta-names" / "vith16plus.safetensors",
    )


def port_config(spec: Spec) -> BenchConfig:
    """Return the configuration of ours of a port cell: the memory need of the port worker."""
    cell = next(c for c in PORT_CELLS if c[0] == spec["candidate"] and c[1] == spec["model"])
    mode = {"fp32": "fp32", "bf16": "bf16-cast", "fp16": "fp16-cast"}[spec["dtype"]]
    return BenchConfig(RUN, cell[2], mode, cell[4], spec["batch_size"], True)


def worker_command(spec: Spec, bench_dir: Path) -> list[str]:
    """Return the command line of one worker.

    Ours and the variants: this file in the repo environment with --worker. A third-party worker:
    the interpreter of its venv with -I, our worker file and its spec.

    Raises:
        ThirdPartyRuleError: For a worker file inside the prior-art folder.
    """
    if spec["kind"] in ("export", "port"):
        worker = COREML_WORKER if spec["kind"] == "export" else PORT_WORKER
        check_worker_path(worker)
        payload = json.dumps(worker_spec(spec, bench_dir))
        return [str(venv_python(spec["kind"])), "-I", str(worker), payload]
    payload = json.dumps(spec | {"bench_dir": str(bench_dir)})
    return [sys.executable, str(Path(__file__).resolve()), "--worker", payload]


def worker_env(spec: Spec, base: dict[str, str]) -> dict[str, str]:
    """Return the environment of one worker: MLX_ENABLE_TF32=0 for every mlx worker (ours, the
    variants and the ports, ADR 0005). The child inherits DINOV3_GPU_LOCK, so it does not wait for
    the driver. A third-party worker runs with -I, so the PYTHON variables do not reach it."""
    env = {k: v for k, v in base.items() if k != "MLX_ENABLE_TF32"}
    env.pop("MLX_METAL_GPU_ARCH", None)
    env["MLX_ENABLE_TF32"] = "0"
    return env


def run_worker(spec: Spec, bench_dir: Path) -> dict:
    """Run one worker in a new subprocess and return its result (run_worker_subprocess)."""
    command = worker_command(spec, bench_dir)
    logger.info(f"command: {command[0]} {' '.join(command[1:3])}")
    env = worker_env(spec, dict(os.environ))
    return run_worker_subprocess(command, env, WORKER_TIMEOUT_S, STDERR_LINES)


def spec_need(spec: Spec) -> int:
    """Return the memory need of one worker: the benchmark need of mlx for ours, the variants and
    the ports; 3 × the weight file + 2 GB for an export (Section 3.8)."""
    if spec["kind"] == "export":
        package = Path(spec["package"])
        weights = package / "Data" / "com.apple.CoreML" / "weights" / "weight.bin"
        size = weights.stat().st_size if weights.exists() else 0
        return 3 * size + EXPORT_RESERVE_BYTES
    if spec["kind"] == "port":
        return required_bench_bytes(port_config(spec), backend="mlx")
    return required_bench_bytes(spec_config(spec), backend="mlx")


def spec_label(spec: Spec) -> str:
    """A short name of one worker for the log."""
    if spec["kind"] == "export":
        return f"export {spec['export']} {spec['input']} {spec['units']} b{spec['batch_size']}"
    if spec["kind"] == "port":
        return f"port {spec['candidate']} {spec['model']} {spec['dtype']} b{spec['batch_size']}"
    label = spec_config(spec).label
    return f"{spec['kind']} {spec.get('variant', '')} {label}".replace("  ", " ")


# The lines.


def _reference_tokens(model_size: str, side: int, image: str) -> "object":
    from safetensors import safe_open

    with safe_open(reference_path(model_size), "np") as tensors:
        return tensors.get_tensor(f"{image}/{side}x{side}/tokens")


def export_line_fields(spec: Spec, worker: dict) -> dict:
    """Return the fields of an export line: the throughput, the parity of each test image
    against the reference output (one code path for all sides) and the verdict.

    The frame throughput equals the patch throughput: one predict call gives every output. An
    export without a frame embedding gets frame: null.
    """
    from safetensors import safe_open

    export = EXPORTS[spec["export"]]
    fields = {
        "status": worker["status"],
        "reason": worker.get("reason", ""),
        "load_s": worker.get("load_s"),
        "first_predict_s": worker.get("first_predict_s"),
        "warm_up_batches": worker.get("warm_up_batches"),
        "timed_batches": worker.get("timed_batches"),
        "timed_s": worker.get("timed_s"),
        "patch": worker.get("patch"),
        "frame": worker.get("patch") if "frame" in export.outputs else None,
        "finite": worker.get("finite"),
        "outputs_file": worker.get("outputs_file"),
        "coremltools": worker.get("coremltools"),
        "parity": None,
        "valid": None,
    }
    if worker["status"] != "done":
        return fields
    if not export.reference_model_size or spec["pixels"] != "stored":
        fields["valid"] = worker["finite"]
        fields["validity_note"] = "finite only: no reference output"
        return fields
    images = []
    try:
        with safe_open(worker["outputs_file"], "np") as tensors:
            for image in TEST_IMAGES:
                names = worker["output_names"]
                outputs = {name: tensors.get_tensor(f"{image}/{name}") for name in names}
                frame, patches = export_outputs(outputs)
                reference = _reference_tokens(export.reference_model_size, export.side, image)
                images.append({"image": image} | export_parity(reference, frame, patches))
    except (KeyError, ValueError, OSError) as error:  # the line keeps the reason; the run goes on
        fields["valid"] = False
        fields["validity_note"] = f"parity error: {type(error).__name__}: {error}"
        return fields
    worst = worst_parity([{k: v for k, v in p.items() if k != "image"} for p in images])
    fields["parity"] = {"images": images, "worst": worst}
    fields["valid"] = bool(worker["finite"]) and parity_passes(worst, export.threshold_mode)
    fields["validity_note"] = f"thresholds {export.threshold_mode}" + (
        ", report only" if export.report_only else ""
    )
    return fields


def spec_line(spec: Spec, worker: dict, need: int, available: int, machine: object) -> dict:
    """Return the line of one worker (Section 3.5 of the plan)."""
    config = spec_config(spec)
    export = EXPORTS.get(spec.get("export", ""))
    line = {
        "run": RUN,
        "job": spec["job"],
        "round": spec.get("round"),
        "position": spec.get("position"),
        "group": spec.get("group"),
        "side_kind": SIDE_KIND[spec["kind"]],
        "candidate": spec.get("candidate"),
        "export": spec.get("export"),
        "variant": spec.get("variant"),
        "priority": spec.get("priority"),
        "model_size": export.model_size if export else (config.model_size if config else None),
        "side": spec.get("side", config.side if config else None),
        "batch_size": spec.get("batch_size", config.batch_size if config else None),
        "dtype_mode": config.dtype_mode if config else None,
        "precision": export.precision if export else spec.get("dtype"),
        "units": spec.get("units"),
        "input": spec.get("input", spec.get("pixels")),
        "status": worker.get("status"),
        "reason": worker.get("reason", ""),
    }
    if spec["kind"] == "port":
        line["model_size"] = spec["model"]
    if spec["kind"] == "export":
        line |= export_line_fields(spec, worker)
    elif spec["kind"] in ("ours", "variant-speed"):
        line |= {key: worker.get(key) for key in OURS_FIELDS}
        line["load_s"] = worker.get("load_time_s")
        line["pixels"] = spec.get("pixels", "stored")
    elif spec["kind"] == "variant-parity":
        line |= {
            key: worker.get(key)
            for key in ("configurations", "compiled_difference", "thresholds", "versions")
        }
        line["load_s"] = worker.get("load_time_s")
    else:
        line |= {k: v for k, v in worker.items() if k not in ("status", "reason")}
    line |= {
        "need_bytes": need,
        "available_bytes": available,
        "memory_check": "ok" if available >= need else "does not fit",
        "mlx_enable_tf32": worker_env(spec, {}).get("MLX_ENABLE_TF32"),
        "machine": asdict(machine) if machine is not None else None,
        "time": worker.get("time") or datetime.now(UTC).isoformat(timespec="seconds"),
    }
    return line


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
        label = spec_label(spec)
        logger.info(
            f"memory check: {label}: needs {need / GB:.1f} GB, "
            f"{available / GB:.1f} GB available, {'ok' if fits else 'does not fit'}"
        )
        seen = other_heavy_jobs(pattern, registry)
        if fits:
            worker = runner(spec, bench_dir)
        else:
            reason = f"needs {need / GB:.1f} GB, {available / GB:.1f} GB available"
            worker = {"status": "skipped", "reason": reason}
        seen += other_heavy_jobs(pattern, registry)
        line = spec_line(spec, worker, need, available, machine)
        line["other_heavy_jobs_seen"] = sorted(set(seen))
        results.parent.mkdir(parents=True, exist_ok=True)
        with results.open("a") as file:
            file.write(json.dumps(line) + "\n")
        counts[line["status"]] += 1
        summary = (line["reason"] or "").splitlines()[-1] if line["reason"] else ""
        if line["status"] == "done" and isinstance(line.get("patch"), dict):
            summary = (
                f"valid {line.get('valid')}, patch {line['patch']['throughput_ips']:.1f} img/s"
            )
        logger.info(f"[{index}/{len(specs)}] {label}: {line['status']}: {summary}")
    return counts


def run_ports(
    rounds: int,
    allowed: bool,
    results: Path,
    bench_dir: Path,
    runner: Runner = run_worker,
) -> Counter:
    """Run J5, or write its 8 "blocked" lines without the user's yes to Q1 (D2).

    Without allowed, no subprocess starts: each port cell gets one line with the status
    "blocked", the reason and the command of J5.
    """
    if allowed:
        return run_specs(prior_art.port_plan(rounds), results, bench_dir, runner)
    counts: Counter = Counter()
    results.parent.mkdir(parents=True, exist_ok=True)
    with results.open("a") as file:
        for number, cell in enumerate(PORT_CELLS, start=1):
            candidate, model, ours_size, side, ours_side, dtype, mode, batch_size = cell
            line = {
                "run": RUN,
                "job": "J5",
                "side_kind": "port",
                "group": f"port-{number}",
                "candidate": candidate,
                "model_size": model,
                "side": side,
                "ours_model_size": ours_size,
                "ours_side": ours_side,
                "precision": dtype,
                "dtype_mode": mode,
                "batch_size": batch_size,
                "status": "blocked",
                "reason": BLOCKED_REASON,
                "command": J5_COMMAND,
                "time": datetime.now(UTC).isoformat(timespec="seconds"),
            }
            file.write(json.dumps(line) + "\n")
            counts["blocked"] += 1
    return counts


# The table.


def read_lines(path: Path) -> list[dict]:
    """Return the JSON lines of a results file, or [] if it does not exist."""
    if not path.exists():
        return []
    return [json.loads(text) for text in path.read_text().splitlines() if text.strip()]


def _f(value: float | None, digits: int = 6) -> str:
    return "–" if value is None else f"{value:.{digits}f}"


def _parity_cells(parity: dict | None) -> str:
    if not parity:
        return "– | – | –"
    return f"{_f(parity['frame'])} | {_f(parity['patch_mean'])} | {_f(parity['patch_p01'])}"


def write_table(lines: list[dict]) -> str:
    """Return results/prior-art-table.md: 1 the candidates, 2 the Core ML cells, 3 the image-input
    and ALL rows, 4 the variant parity, 5 the variant speed, 6 the port cells."""
    out = ["# Step 6b: prior-art comparison", ""]
    out += [
        (
            "Generated by `scripts/bench_prior_art.py table` from `results/prior-art.jsonl`. Ratios: "
            "ours ÷ other (above 1: ours is faster), patch output, lower median of the paired rounds "
            "(ADR 0004). Verdict: single_verdict of speedup.py, inside the noise within 1 ± 0.03."
        ),
        "",
        "## 1 Candidates",
        "",
        "| No. | Candidate | Kind | License | Commit | Verdict | Reason |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for number, c in prior_art.CANDIDATES.items():
        out.append(
            f"| {number} | {c.name} | {c.kind} | {c.license} | {c.commit} | {c.verdict} | "
            f"{c.reason or '–'} |"
        )
    out += [
        "",
        "## 2 Core ML cells (J1, compute units CPU_AND_GPU)",
        "",
        (
            "| Export | Candidate | Model size, px | Precision | Batch size | Ours (dtype mode, px) | "
            "Ours img/s | Export img/s | Ratios | Median | Spread | Verdict | Worst frame | "
            "Worst patch mean | Worst patch p01 | Parity |"
        ),
        (
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | "
            "--- | --- |"
        ),
    ]
    j1_rows = prior_art.prior_art_rows(lines)
    for row in j1_rows:
        fields = row.fields if isinstance(row, prior_art.PriorArtRow) else row
        head = (
            f"| {fields['export']} | {fields['candidate']} | {fields['model_size']}, "
            f"{fields['side']} | {fields['precision']} | {fields['batch_size']} | "
            f"{fields['dtype_mode']}, {fields['ours_side']} |"
        )
        if fields["parity"] is None:
            verdict = "finite only" if fields.get("valid") else "–"
        else:
            verdict = "pass" if fields["valid"] else "fail"
        if fields["report_only"] and fields["parity"] is not None:
            verdict += " (report only)"
        if isinstance(row, prior_art.PriorArtRow):
            ratios = ", ".join(f"{r:.3f}" for r in row.ratios)
            out.append(
                f"{head} {row.ours_ips:.1f} | {row.other_ips:.1f} | {ratios} | {row.median:.3f} | "
                f"{row.spread:.3f} | {row.verdict} | {_parity_cells(fields['parity'])} | "
                f"{verdict} |"
            )
        else:
            out.append(f"{head} – | – | – | – | – | {row['reason']} | – | – | – | – |")
    out += [
        "",
        "## 3 Report-only rows (J2)",
        "",
        (
            "8-bit image input (as shipped), batch size 1, CPU_AND_GPU, against the spec-array rows of "
            "J1:"
        ),
        "",
        (
            "| Export | Input | Worst frame | Worst patch mean | Worst patch p01 | Spec-array patch "
            "p01 (J1) | Difference |"
        ),
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    spec_p01 = {}
    for row in j1_rows:
        fields = row.fields if isinstance(row, prior_art.PriorArtRow) else row
        if fields["parity"]:
            spec_p01[fields["export"]] = min(
                spec_p01.get(fields["export"], 1.0), fields["parity"]["patch_p01"]
            )
    for line in lines:
        if line.get("job") == "J2" and line.get("input") == "image":
            worst = (line.get("parity") or {}).get("worst")
            base = spec_p01.get(line["export"])
            difference = None if worst is None or base is None else worst["patch_p01"] - base
            out.append(
                f"| {line['export']} | 8-bit image | {_parity_cells(worst)} | {_f(base)} | "
                f"{_f(difference)} |"
            )
    out += [
        "",
        "Compute units ALL (one round), against the median export throughput of J1:",
        "",
        "| Export | Batch size | ALL img/s | CPU_AND_GPU img/s (J1 median) | ALL ÷ CPU_AND_GPU |",
        "| --- | --- | --- | --- | --- |",
    ]
    j1_ips = {
        (row.fields["export"], row.fields["batch_size"]): row.other_ips
        for row in j1_rows
        if isinstance(row, prior_art.PriorArtRow)
    }
    for line in lines:
        if line.get("job") == "J2" and line.get("units") == "ALL":
            ips = (line.get("patch") or {}).get("throughput_ips")
            base = j1_ips.get((line["export"], line["batch_size"]))
            ratio = None if ips is None or base is None else ips / base
            out.append(
                f"| {line['export']} | {line['batch_size']} | {_f(ips, 1)} | {_f(base, 1)} | "
                f"{_f(ratio, 3)} |"
            )
    out += [
        "",
        "## 4 Variant parity (J3): 12 configurations at batch size 1",
        "",
        (
            "| Model size | Dtype mode | Variant | Change | Pass | Finite | Worst frame | "
            "Worst patch mean | Worst patch p01 |"
        ),
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in prior_art.variant_parity_rows(lines):
        if "worst" not in row:
            out.append(
                f"| {row['model_size']} | {row['dtype_mode']} | {row['variant']} | "
                f"{prior_art.VARIANTS[row['variant']]} | {row['status']}: {row['reason']} | – | "
                "– | – | – |"
            )
            continue
        out.append(
            f"| {row['model_size']} | {row['dtype_mode']} | {row['variant']} | "
            f"{prior_art.VARIANTS[row['variant']]} | {row['passes']} of {row['n']} | "
            f"{row['finite']} | {_parity_cells(row['worst'])} |"
        )
    out += [
        "",
        "## 5 Variant speed (J4): change ratio variant ÷ V0 (above 1: the change is faster)",
        "",
        (
            "| Point | Dtype mode | Variant | V0 img/s | Variant img/s | Ratios | Median | Spread | "
            "Verdict |"
        ),
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in prior_art.variant_speed_rows(lines):
        ratios = ", ".join(f"{r:.3f}" for r in row.ratios)
        out.append(
            f"| {row.fields['priority']} | {row.fields['dtype_mode']} | {row.fields['variant']} | "
            f"{row.other_ips:.1f} | {row.ours_ips:.1f} | {ratios} | {row.median:.3f} | "
            f"{row.spread:.3f} | {row.verdict} |"
        )
    out += [
        "",
        "## 6 Python ports (J5, gated by Q1)",
        "",
        "| Cell | Candidate | Model | px (ours) | Dtype | Batch size | Status | Reason or result |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for line in lines:
        if line.get("job") == "J5" and line.get("status") == "blocked":
            out.append(
                f"| {line['group']} | {line['candidate']} | {line['model_size']} | {line['side']} "
                f"({line['ours_side']}) | {line['precision']} | {line['batch_size']} | blocked | "
                f"{line['reason']}. Command: `{line['command']}` |"
            )
    out.append("")
    return "\n".join(out)


# The command line.


def plan_of(args: argparse.Namespace) -> list[Spec]:
    """Return the workers of the subcommand from the command-line options."""
    if args.subcommand == "coreml":
        if args.image_input:
            return prior_art.image_input_plan()
        if args.units == "ALL":
            return prior_art.all_units_plan()
        return prior_art.coreml_plan(args.rounds)
    if args.subcommand == "variants":
        if args.parity:
            return prior_art.variant_parity_plan(args.model_sizes)
        return prior_art.variant_speed_plan(args.priority, args.rounds)
    if args.subcommand == "ports":
        return prior_art.port_plan(args.rounds) if args.allow_third_party_python else []
    return []


def venv_kinds(args: argparse.Namespace) -> list[str]:
    """Return the third-party venvs that the subcommand needs."""
    if args.subcommand == "coreml":
        return ["export"]
    if args.subcommand == "ports" and args.allow_third_party_python:
        return ["port"]
    return []


def dry_run(specs: list[Spec]) -> None:
    """Print each worker with its memory need. Load no model, import no backend."""
    available = available_memory_bytes()
    print(f"available memory: {available / GB:.1f} GB")
    for spec in specs:
        need = spec_need(spec)
        fits = "fits" if need <= available else "does not fit"
        print(
            f"{spec.get('position', 0):4d} {spec['job']} r{spec.get('round', 1)} "
            f"{spec.get('group', ''):24s} {spec_label(spec):58s} need {need / GB:5.2f} GB  {fits}"
        )
    print(f"{len(specs)} workers")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Return the command-line options."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    sub = parser.add_subparsers(dest="subcommand")
    commands = {name: sub.add_parser(name) for name in ("coreml", "variants", "ports", "table")}
    for name in ("coreml", "variants", "ports"):
        commands[name].add_argument("--dry-run", action="store_true", help="print the plan only")
        commands[name].add_argument("--results", type=Path, default=RESULTS)
    for name, rounds in (("coreml", 5), ("variants", 3), ("ports", 5)):
        commands[name].add_argument("--rounds", type=int, default=rounds)
    coreml = commands["coreml"]
    coreml.add_argument("--units", choices=["CPU_AND_GPU", "ALL"], default="CPU_AND_GPU")
    coreml.add_argument("--image-input", action="store_true", help="the 8-bit image rows (J2)")
    variants = commands["variants"]
    kind = variants.add_mutually_exclusive_group(required=True)
    kind.add_argument("--parity", action="store_true", help="J3")
    kind.add_argument("--speed", action="store_true", help="J4")
    variants.add_argument("--model-sizes", nargs="+", choices=prior_art.VARIANT_MODEL_SIZES)
    variants.add_argument("--priority", nargs="+", choices=list(prior_art.SPEED_POINTS))
    commands["ports"].add_argument(
        "--allow-third-party-python",
        action="store_true",
        help="run the Python ports C1 and C2: only after the user's yes to Q1",
    )
    commands["table"].add_argument("--results", type=Path, default=RESULTS)
    commands["table"].add_argument("--table", type=Path, default=TABLE)
    args = parser.parse_args(argv)
    if args.worker is None and args.subcommand is None:
        parser.error("give a subcommand")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    """Parse the arguments and run the workers. A run that is not a dry run takes the GPU lock.

    Exit code 2: a venv that the subcommand needs is missing (before the lock).
    """
    args = parse_args(argv)
    if args.worker:
        worker_main(args.worker)
        return
    if args.subcommand == "table":
        args.table.write_text(write_table(read_lines(args.results)))
        print(f"wrote {args.table}")
        return
    missing = missing_venvs(venv_kinds(args))
    if missing:
        print(f"bench_prior_art.py: missing venv: {', '.join(map(str, missing))}", file=sys.stderr)
        sys.exit(2)
    if os.environ.get("MLX_DISABLE_COMPILE"):
        print("bench_prior_art.py: MLX_DISABLE_COMPILE is set (R42). Unset it.", file=sys.stderr)
        sys.exit(1)
    specs = plan_of(args)
    if args.dry_run:
        dry_run(specs)
        return
    bench_dir = BENCH_ROOT / args.subcommand
    if args.results.resolve() != RESULTS.resolve():  # a Verify run keeps its files apart
        bench_dir = args.results.with_name(f"{args.results.stem}-batches") / args.subcommand
    bench_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
        handlers=[logging.StreamHandler(sys.stderr), logging.FileHandler(bench_dir / "driver.log")],
    )
    if args.subcommand == "ports" and not args.allow_third_party_python:
        counts = run_ports(args.rounds, False, args.results, bench_dir)
        logger.info(f"ports: {dict(counts)} (Q1 open: no third-party Python ran)")
        return
    with heavy_job(f"scripts/bench_prior_art.py {args.subcommand}"):
        logger.info(f"run of {len(specs)} workers: {args.subcommand}")
        try:
            if args.subcommand == "ports":
                counts = run_ports(args.rounds, True, args.results, bench_dir)
            else:
                counts = run_specs(specs, args.results, bench_dir)
        except (MachineStateError, WeightsNotFoundError) as error:
            logger.error(f"stop: {error}")
            sys.exit(1)
        logger.info(f"end of run: {dict(counts)}")


# The worker side (repo environment).


def _random_batch(side: int, batch_size: int) -> "object":
    """Random normal pixels (B, 3, side, side), seed 0: the input of the 592 px cell (D7)."""
    import numpy as np

    rng = np.random.default_rng(0)
    return rng.standard_normal((batch_size, 3, side, side)).astype(np.float32)


def worker_main(spec_json: str) -> None:
    """Run one worker of ours or of a variant in this process and print one JSON line."""
    send_checks_to(sys.stderr)
    spec = json.loads(spec_json)
    bench_dir = Path(spec["bench_dir"])
    config = spec_config(spec)
    warm_up, timed = BUDGETS[spec.get("budget", "full")]
    try:
        if spec["kind"] == "ours":
            from dinov3_apple.mlx_bench import run_mlx_worker

            batch = None
            if spec.get("pixels") == "random":
                batch = _random_batch(config.side, config.batch_size)
            run_id = f"{spec['job']}-r{spec.get('round', 1)}"
            result = run_mlx_worker(config, warm_up, timed, bench_dir / "mlx", run_id, batch)
        elif spec["kind"] == "variant-parity":
            from dinov3_apple.port_variants import run_variant_parity

            result = run_variant_parity(config.model_size, config.dtype_mode, spec["variant"])
        elif spec["kind"] == "variant-speed":
            from dinov3_apple.port_variants import run_variant_speed

            result = run_variant_speed(config, spec["variant"], warm_up, timed)
        else:
            raise ValueError(f"unknown worker kind {spec['kind']!r}")
    except InsufficientMemoryError as error:
        result = {"status": "skipped", "reason": f"load check in the worker: {error}"}
    print(json.dumps(result))


if __name__ == "__main__":
    main()
