"""The driver of the C1 probe: the MLX tile sizes in a source build (B3 and B4 of
specs/06e-kernels/plan.html).

The source build lives outside the repo, in ~/.cache/dinov3-mlx/mlx-src/: the clone of MLX at tag
v0.32.4 (mlx/), its venv (venv-build/) and the probe patch (probe.patch). The patch reads a tile
override from DINOV3_PROBE_GEMM_NAX or DINOV3_PROBE_SDPA_NAX at each dispatch and instantiates the
extra kernels of SWEEP_CONFIGS. The repo gets numbers only (ADR 0013).

Rules (ADR 0013): the driver runs the source build only with --allow-source-build (the yes of Q1
of the plan). The worker is our own file scripts/probe_kernels_worker.py; the driver refuses a
worker file inside the source folder. It starts the worker as <python> -I <worker> '<spec>': the
wheel side with the Python of the repo venv, the source side with the Python of venv-build. Before
each worker the driver checks the machine state (AC power, no Low Power Mode) and the free memory.
The driver takes the GPU lock; one worker runs at a time.

Usage:
    uv run python -m dinov3_mlx.gpu_lock -- uv run python scripts/probe_kernels.py control \\
        --rounds 5 --allow-source-build
    uv run python -m dinov3_mlx.gpu_lock -- uv run python scripts/probe_kernels.py sweep \\
        --target T1 T2 T3 --rounds 5 --allow-source-build
    uv run python scripts/probe_kernels.py table
"""

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from dinov3_mlx.bench import run_worker_subprocess
from dinov3_mlx.gpu_lock import heavy_job
from dinov3_mlx.machine import MachineStateError, check_machine_state, read_machine_state
from dinov3_mlx.memory import GB, available_memory_bytes

REPO_ROOT = Path(__file__).parents[1]
WORKER = Path(__file__).with_name("probe_kernels_worker.py")
SOURCE_DIR = Path.home() / ".cache" / "dinov3-mlx" / "mlx-src"
CLONE = SOURCE_DIR / "mlx"
PATCH = SOURCE_DIR / "probe.patch"
SOURCE_PYTHON = SOURCE_DIR / "venv-build" / "bin" / "python"
RESULTS = REPO_ROOT / "results" / "kernels-c1.jsonl"
TABLE = REPO_ROOT / "results" / "kernels-c1.md"
NEED_BYTES = 4 * 10**9
WORKER_TIMEOUT_S = 1800
OVERRIDES = ("DINOV3_PROBE_GEMM_NAX", "DINOV3_PROBE_SDPA_NAX")
CONTROL_LIMIT = 0.03  # AC9: each shape within ± 3 %
COSINE_LIMIT = 0.99999  # AC10
# The choice of MLX 0.32.4 on the M5 Max (applegpu_g17s): steel_matmul_regular_axpby_nax takes
# bm 64, bn 128, bk 256, wm 2, wn 4; sdpa_full_self_attention_nax takes bq 64, bk 32, wm 4, wn 1
# for head dimension 64.
GEMM_CONFIGS = [
    "64,128,256,2,4",  # the choice of MLX
    "64,128,512,2,4",
    "64,128,64,2,4",
    "64,128,128,2,4",
    "128,64,256,4,2",
    "128,64,512,4,2",
    "128,128,64,4,4",
    "128,128,128,4,4",
    "128,128,256,4,4",
    "128,128,512,4,4",
    "64,64,256,2,2",
    "128,128,256,2,2",
]
SDPA_CONFIGS = [
    "64,32,4,1",  # the choice of MLX
    "64,64,4,1",
    "64,16,4,1",
    "64,128,4,1",
    "32,16,2,1",
    "32,32,2,1",
    "32,64,2,1",
    "32,128,2,1",
    "128,16,8,1",
    "128,32,8,1",
    "128,64,8,1",
    "128,128,8,1",
]
# The three targets of Section 3.3: (probe of the worker, override variable, configurations).
TARGETS = {
    "T1": ("linear fp16 1280->1280", "DINOV3_PROBE_GEMM_NAX", GEMM_CONFIGS),
    "T2": ("linear bf16 1280->10240", "DINOV3_PROBE_GEMM_NAX", GEMM_CONFIGS),
    "T3": ("sdpa fp16", "DINOV3_PROBE_SDPA_NAX", SDPA_CONFIGS),
}
SWEEP_CONFIGS = {target: configs for target, (_, _, configs) in TARGETS.items()}

logger = logging.getLogger("probe_kernels")


class SourceBuildRuleError(RuntimeError):
    """A step breaks a rule of ADR 0013 for the source build."""


def check_worker_path(path: Path) -> None:
    """Refuse a worker file inside the source folder: the worker is our own file in the repo.

    Raises:
        SourceBuildRuleError: For a path inside SOURCE_DIR.
    """
    if path.resolve().is_relative_to(SOURCE_DIR.resolve()):
        raise SourceBuildRuleError(f"the worker {path} is inside {SOURCE_DIR}")


def worker_command(side: str, spec: dict, worker: Path = WORKER) -> list[str]:
    """Return the command line of one worker: the Python of the side with -I, the worker file and
    its spec.

    Args:
        side: "wheel" (the Python of the repo venv) or "source" (the Python of venv-build).
        spec: The spec of the worker.
        worker: The worker file.

    Raises:
        SourceBuildRuleError: For a worker file inside the source folder.
    """
    check_worker_path(worker)
    python = sys.executable if side == "wheel" else str(SOURCE_PYTHON)
    return [python, "-I", str(worker), json.dumps(spec)]


def worker_env(base: dict[str, str]) -> dict[str, str]:
    """Return the environment of a worker: without the tile overrides and the GPU architecture
    override, with MLX_ENABLE_TF32=0. The child inherits DINOV3_GPU_LOCK."""
    env = {k: v for k, v in base.items() if k not in OVERRIDES and k != "MLX_METAL_GPU_ARCH"}
    env["MLX_ENABLE_TF32"] = "0"
    return env


def run_worker(side: str, spec: dict) -> dict:
    """Run one worker in a new subprocess and return its result."""
    command = worker_command(side, spec)
    return run_worker_subprocess(command, worker_env(dict(os.environ)), WORKER_TIMEOUT_S)


def build_info() -> dict:
    """Return the commit of the clone and the SHA-256 of the probe patch: no local path.

    Raises:
        SourceBuildRuleError: If venv-build imports an mlx from outside the source folder.
    """
    commit = subprocess.run(
        ["git", "-C", str(CLONE), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    where = subprocess.run(
        [str(SOURCE_PYTHON), "-I", "-c", "import mlx.core as mx; print(mx.__file__)"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not Path(where).resolve().is_relative_to(SOURCE_DIR.resolve()):
        raise SourceBuildRuleError("venv-build does not import the source build")
    patch = hashlib.sha256(PATCH.read_bytes()).hexdigest() if PATCH.exists() else None
    return {"mlx_commit": commit, "patch_sha256": patch, "tag": "v0.32.4"}


def control_plan(rounds: int) -> list[tuple[str, dict]]:
    """B3: in each round one wheel worker and one source worker; the order alternates. Round 1
    also checks the cosine of each probe."""
    plan = []
    for r in range(1, rounds + 1):
        for side in ("wheel", "source") if r % 2 else ("source", "wheel"):
            plan.append((side, {"mode": "probe", "check": r == 1, "round": r}))
    return plan


def sweep_plan(targets: list[str], rounds: int) -> list[tuple[str, dict]]:
    """B4: one source worker for each target, with the control and all configurations of the
    target in paired rounds."""
    plan = []
    for target in targets:
        probe, variable, configs = TARGETS[target]
        spec = {"mode": "sweep", "probe": probe, "variable": variable, "configs": configs}
        plan.append(("source", spec | {"target": target, "rounds": rounds}))
    return plan


def run_plan(plan: list[tuple[str, dict]], kind: str, results: Path, build: dict) -> int:
    """Run the workers one after the other and append one line for each. Return the number of
    lines with the status "done"."""
    run_id = datetime.now(UTC).isoformat(timespec="seconds")
    done = 0
    for index, (side, spec) in enumerate(plan, start=1):
        machine = read_machine_state()
        check_machine_state(machine)
        available = available_memory_bytes()
        line = {"kind": kind, "side": side, "run_id": run_id, "build": build}
        line |= {k: spec[k] for k in ("round", "target") if k in spec}
        if available < NEED_BYTES:
            reason = f"needs {NEED_BYTES / GB:.1f} GB, {available / GB:.1f} GB available"
            line |= {"status": "skipped", "reason": reason}
        else:
            line |= run_worker(side, spec)
        line |= {"machine": asdict(machine), "available_bytes": available}
        results.parent.mkdir(parents=True, exist_ok=True)
        with results.open("a") as file:
            file.write(json.dumps(line) + "\n")
        done += line["status"] == "done"
        logger.info(
            f"[{index}/{len(plan)}] {kind} {side} {spec.get('target', '')}: {line['status']}"
        )
    return done


# The tables.


def _median(values: list[float]) -> float:
    return float(np.median(values))


def control_rows(lines: list[dict]) -> list[dict]:
    """Return one row for each probe of B3: the ratio of the source build to the wheel (wheel
    time ÷ source time, so above 1 is faster) in each round, its median and the verdict of AC9.
    Only the lines of the newest run count."""
    controls = [
        line for line in lines if line.get("kind") == "control" and line["status"] == "done"
    ]
    if not controls:
        return []
    newest = max(line.get("run_id", "") for line in controls)
    controls = [line for line in controls if line.get("run_id", "") == newest]
    seconds: dict[tuple[str, str], dict[int, float]] = {}
    cosine: dict[str, dict[str, float | None]] = {}
    flops: dict[str, float] = {}
    for line in controls:
        for probe in line["probes"]:
            seconds.setdefault((probe["name"], line["side"]), {})[line["round"]] = probe["seconds"]
            flops[probe["name"]] = probe["flops"]
            if probe.get("cosine") is not None:
                cosine.setdefault(probe["name"], {})[line["side"]] = probe["cosine"]
    rows = []
    for name, name_flops in flops.items():
        wheel, source = seconds.get((name, "wheel"), {}), seconds.get((name, "source"), {})
        rounds = sorted(set(wheel) & set(source))
        ratios = [wheel[r] / source[r] for r in rounds]
        ratio = _median(ratios) if ratios else None
        rows.append(
            {
                "probe": name,
                "rounds": len(rounds),
                "ratios": ratios,
                "ratio": ratio,
                "wheel_tflops": name_flops / _median(list(wheel.values())) / 1e12,
                "source_tflops": name_flops / _median(list(source.values())) / 1e12,
                "cosine": cosine.get(name, {}),
                "within": ratio is not None and abs(ratio - 1) <= CONTROL_LIMIT,
            }
        )
    return rows


def sweep_rows(sweeps: list[dict], control: list[dict]) -> list[dict]:
    """Return one row for each target and configuration of B4: the ratio to the control build
    (control time ÷ configuration time, the median of the rounds), the ratio to the wheel (times
    the ratio of the control row of the probe) and the validity of AC10."""
    to_wheel = {row["probe"]: row["ratio"] for row in control}
    rows = []
    for line in sweeps:
        if line.get("kind") != "sweep" or line["status"] != "done":
            continue
        base = line["configs"]["control"]["seconds"]
        for name, values in line["configs"].items():
            if name == "control":
                continue
            seconds, cosine = values["seconds"], values["cosine"]
            ratio = None
            if seconds:
                ratio = _median([b / s for b, s in zip(base, seconds, strict=True)])
            wheel = to_wheel.get(line["probe"])
            rows.append(
                {
                    "target": line["target"],
                    "probe": line["probe"],
                    "config": name,
                    "ratio": ratio,
                    "ratio_to_wheel": None if ratio is None or wheel is None else ratio * wheel,
                    "tflops": None if not seconds else line["flops"] / _median(seconds) / 1e12,
                    "control_tflops": line["flops"] / _median(base) / 1e12,
                    "cosine": cosine,
                    "valid": bool(seconds) and cosine is not None and cosine >= COSINE_LIMIT,
                    "error": values.get("error", ""),
                }
            )
    return rows


def best_rows(rows: list[dict]) -> list[dict]:
    """Return the valid configuration with the highest ratio for each target (AC10)."""
    best: dict[str, dict] = {}
    for row in rows:
        if row["valid"] and (
            row["target"] not in best or row["ratio"] > best[row["target"]]["ratio"]
        ):
            best[row["target"]] = row
    return [best[target] for target in sorted(best)]


def _number(value: float | None, digits: int = 3) -> str:
    return "–" if value is None else f"{value:.{digits}f}"


def write_table(lines: list[dict]) -> str:
    """Return the Markdown text of results/kernels-c1.md from the lines of the results file."""
    control = control_rows(lines)
    sweeps = [line for line in lines if line.get("kind") == "sweep"]
    newest: dict[str, dict] = {}
    for line in sweeps:
        newest[line["target"]] = line  # the newest line of each target
    rows = sweep_rows(list(newest.values()), control)
    out = [
        "# C1 probe: MLX tile sizes in a source build",
        "",
        "Step 6e, B3 and B4 of `specs/06e-kernels/plan.html`. Source: `results/kernels-c1.jsonl`.",
        "ViT-H+/16 shapes at 512 px, batch size 8 (8 × 1,029 tokens). Each time is the median of",
        "20 calls with `mx.eval` after each call. A ratio above 1 is faster.",
        "",
        "## Table 1. B3: the source build against the wheel (AC9: each shape within ± 3 %)",
        "",
        "| Probe | Rounds | Wheel TFLOP/s | Source TFLOP/s | Ratio (median) | Min | Max | Verdict |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in control:
        verdict = "pass" if row["within"] else "fail"
        out.append(
            f"| {row['probe']} | {row['rounds']} | {row['wheel_tflops']:.1f} | "
            f"{row['source_tflops']:.1f} | {_number(row['ratio'])} | "
            f"{_number(min(row['ratios'], default=None))} | "
            f"{_number(max(row['ratios'], default=None))} | {verdict} |"
        )
    out += [
        "",
        "## Table 2. B4: each tile configuration against the control build",
        "",
        "GEMM configurations: bm, bn, bk, wm, wn. SDPA configurations: bq, bk, wm, wn. Valid: the",
        f"smallest row cosine to NumPy float32 is ≥ {COSINE_LIMIT}.",
        "",
        (
            "| Target | Probe | Configuration | TFLOP/s | Control TFLOP/s | Ratio to control | "
            "Ratio to wheel | Smallest row cosine | Valid |"
        ),
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        valid = "valid" if row["valid"] else "not valid"
        if row["error"]:
            valid += f" ({row['error'].splitlines()[-1][:80]})"
        cosine = "–" if row["cosine"] is None else f"{row['cosine']:.7f}"
        out.append(
            f"| {row['target']} | {row['probe']} | {row['config']} | {_number(row['tflops'], 1)} | "
            f"{row['control_tflops']:.1f} | {_number(row['ratio'])} | "
            f"{_number(row['ratio_to_wheel'])} | {cosine} | {valid} |"
        )
    out += [
        "",
        "## Table 3. AC10: the best valid configuration of each target",
        "",
        "| Target | Probe | Configuration | TFLOP/s | Ratio to control | Ratio to wheel |",
        "|---|---|---|---|---|---|",
    ]
    for row in best_rows(rows):
        out.append(
            f"| {row['target']} | {row['probe']} | {row['config']} | {_number(row['tflops'], 1)} | "
            f"{_number(row['ratio'])} | {_number(row['ratio_to_wheel'])} |"
        )
    return "\n".join(out) + "\n"


def read_lines(path: Path) -> list[dict]:
    """Return the JSON lines of a results file, or [] if it does not exist."""
    if not path.exists():
        return []
    return [json.loads(text) for text in path.read_text().splitlines() if text.strip()]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Return the command-line options."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    sub = parser.add_subparsers(dest="subcommand", required=True)
    commands = {name: sub.add_parser(name) for name in ("control", "sweep", "table")}
    for name in ("control", "sweep"):
        commands[name].add_argument("--rounds", type=int, default=5)
        commands[name].add_argument(
            "--allow-source-build", action="store_true", help="the yes of Q1 (ADR 0013)"
        )
    for command in commands.values():
        command.add_argument("--results", type=Path, default=RESULTS)
    commands["sweep"].add_argument(
        "--target", nargs="+", choices=list(TARGETS), default=list(TARGETS)
    )
    commands["table"].add_argument("--table", type=Path, default=TABLE)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Parse the arguments and run the probe. A run without --allow-source-build or without
    venv-build stops with exit code 2 before the lock."""
    args = parse_args(argv)
    if args.subcommand == "table":
        args.table.write_text(write_table(read_lines(args.results)))
        return
    if not args.allow_source_build:
        print(
            "probe_kernels.py: the source build of MLX runs only with --allow-source-build "
            "(ADR 0013, Q1 of specs/06e-kernels/plan.html).",
            file=sys.stderr,
        )
        sys.exit(2)
    if not SOURCE_PYTHON.exists():
        print(f"probe_kernels.py: no venv-build in {SOURCE_DIR}", file=sys.stderr)
        sys.exit(2)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    if args.subcommand == "control":
        plan = control_plan(args.rounds)
    else:
        plan = sweep_plan(args.target, args.rounds)
    with heavy_job(f"scripts/probe_kernels.py {args.subcommand}"):
        try:
            done = run_plan(plan, args.subcommand, args.results, build_info())
        except MachineStateError as error:
            logger.error(f"stop: {error}")
            sys.exit(1)
    logger.info(f"end of run: {done} of {len(plan)} workers done")


if __name__ == "__main__":
    main()
