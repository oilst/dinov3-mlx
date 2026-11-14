"""The driver of all Step 6 benchmarks: change ratios, the paired matrix, the paired runs at P1 and
P2, the batch-size sweep, the preprocessing row, the Learn data and the P1 profile (Section 7 of
specs/06-speed/plan.html). Also the change C7 of Step 6e: the joined projections (B1 and B2 of
specs/06e-kernels/plan.html).

Each worker runs in its own subprocess, one after the other. Before each worker the driver reads
the machine state, stops on battery power or in Low Power Mode, and checks the free memory
against the need of the backend of the worker (D8). A worker that does not fit gives a "skipped"
line. Before and after each worker the driver writes the heavy jobs outside the GPU lock into the
field other_heavy_jobs_seen (Section 3.3.1). The driver takes the GPU lock in main; a dry run
takes no lock and imports no backend. The driver stops when MLX_DISABLE_COMPILE is set: the
compiled rows would run eager without an error (R42).

Results: results/speed-{changes,matrix,pairs,sweep,learn,profile-p1}.jsonl; C7 alone goes to
results/speed-joined.jsonl. The per-batch files
and driver.log of each subcommand: tests/fixtures/generated/bench/speed/<subcommand>/. With
--results, they go to the folder <stem>-batches/ next to that file.

Usage:
    uv run python scripts/bench_speed.py change --change compile --priority P1 P2 --dry-run
    uv run python scripts/bench_speed.py change --change rope autocast --priority P1 P2
    uv run python scripts/bench_speed.py change --change quantized --priority P2
    uv run python scripts/bench_speed.py change --change fp16-output --priority P1 P2
    uv run python scripts/bench_speed.py change --change joined --run 1
    uv run python scripts/bench_speed.py preprocess
    uv run python scripts/bench_speed.py profile --priority P1
    uv run python scripts/bench_speed.py pairs --priority P1 P2 --rounds 5 --run 1
    uv run python scripts/bench_speed.py matrix --model-sizes ViT-S/16 ViT-H+/16 --quantized
    uv run python scripts/bench_speed.py matrix --gate-7b
    uv run python scripts/bench_speed.py matrix --only-skipped
    uv run python scripts/bench_speed.py matrix --compile-check
    uv run python scripts/bench_speed.py sweep
    uv run python scripts/bench_speed.py learn
"""

import argparse
import json
import logging
import os
import sys
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict, replace
from pathlib import Path

from dinov3_mlx.bench import (
    BATCH_SIZES,
    DTYPE_MODE_NAMES,
    DTYPE_OF_MODE,
    GIANT_MODEL_SIZE,
    PRIORITY,
    QUANTIZED_MODEL_SIZES,
    QUANTIZED_MODES,
    SIDES,
    TIMED,
    WARM_UP,
    BenchConfig,
    TimeBudget,
    _modes,
    line_key,
    required_bench_bytes,
    result_line,
    run_worker_subprocess,
)
from dinov3_mlx.gpu_lock import HEAVY_JOB_PATTERN, LOCK_REGISTRY, heavy_job, other_heavy_jobs
from dinov3_mlx.hub import MODEL_IDS, WeightsNotFoundError
from dinov3_mlx.machine import MachineStateError, check_machine_state, read_machine_state
from dinov3_mlx.memory import GB, InsufficientMemoryError, available_memory_bytes, send_checks_to

REPO_ROOT = Path(__file__).parents[1]
RESULTS_DIR = REPO_ROOT / "results"
BENCH_ROOT = REPO_ROOT / "tests" / "fixtures" / "generated" / "bench" / "speed"
RESULTS = {
    "change": RESULTS_DIR / "speed-changes.jsonl",
    "preprocess": RESULTS_DIR / "speed-changes.jsonl",
    "matrix": RESULTS_DIR / "speed-matrix.jsonl",
    "pairs": RESULTS_DIR / "speed-pairs.jsonl",
    "sweep": RESULTS_DIR / "speed-sweep.jsonl",
    "learn": RESULTS_DIR / "speed-learn.jsonl",
    "profile": RESULTS_DIR / "speed-profile-p1.jsonl",
}
JOINED_RESULTS = RESULTS_DIR / "speed-joined.jsonl"
LOG_FORMAT = "%(asctime)s %(name)s %(message)s"
RUN = "S6"  # the run of the Step 6 lines; the torch-mps rows of ViT-7B/16 keep "B4" (B4)
WORKER_TIMEOUT_S = {"mlx": 900, "torch-mps": 3600, "cpu": 900}
STDERR_LINES = 20
WATERMARKS = {"PYTORCH_MPS_HIGH_WATERMARK_RATIO": "0.8", "PYTORCH_MPS_LOW_WATERMARK_RATIO": "0.6"}
# The change ratios of Table 23 of the plan. In-process changes hold both forms in one process;
# the others run one subprocess for each configuration, in alternation.
IN_PROCESS_CHANGES = {
    "compile": ("C1", ["E", "A"]),
    "rope": ("C2", ["A", "R"]),
    "fp16-output": ("C4", ["R-fp32", "R-fp16"]),
    # Step 6e (A2): separate, both joins, only qkv, only gate and up.
    "joined": ("C7", ["R", "R-J", "R-Jqkv", "R-Jff"]),
}
# B1 of specs/06e-kernels/plan.html: the dtype modes of C7 at each priority, and ViT-H+/16 at
# 512 px with batch size 1. Batch size 1 takes the P1 budget of 20 timed batches: 4 batches of
# about 50 ms would give a timed phase of 0.2 s.
JOINED_MODES = {"P1": ["fp32", "fp16-cast", "bf16-cast"], "P2": DTYPE_MODE_NAMES}
JOINED_EXTRA = [("P2", 1, ["fp16-cast", "bf16-cast"], "change-P1")]
CHANGES = [*IN_PROCESS_CHANGES, "autocast", "quantized"]
HALF_MODES = [mode for mode in DTYPE_MODE_NAMES if mode != "fp32"]
# Table 23: 3 warm-up batches and 20 timed batches at P1, 3 and 4 at P2.
CHANGE_BUDGETS = {
    "P1": (TimeBudget(3, 0.0), TimeBudget(20, 0.0)),
    "P2": (TimeBudget(3, 0.0), TimeBudget(4, 0.0)),
}
BUDGETS = {"full": (WARM_UP, TIMED), **{f"change-{p}": b for p, b in CHANGE_BUDGETS.items()}}
# Section 3.12: the batch-size sweep in bf16-cast.
SWEEP = [("ViT-S/16", 224, [1, 8, 32, 64, 128, 256]), ("ViT-H+/16", 512, [1, 2, 4, 8, 16])]
# Section 3.13: the Learn data above 512 px.
LEARN_SIDES = [768, 1024]
LEARN_MODES = [("ViT-S/16", "fp32"), ("ViT-S/16", "bf16-cast"), ("ViT-H+/16", "bf16-cast")]
NAX_OFF = {"MLX_METAL_GPU_ARCH": "applegpu_g16s"}  # turns the NAX kernels off (D19)
PREPROCESS_NEED_BYTES = 2 * 10**9

logger = logging.getLogger("bench_speed")

Spec = dict
Runner = Callable[[Spec, Path], dict]


def model_order(model_size: str) -> int:
    return list(MODEL_IDS).index(model_size)


def spec_config(spec: Spec) -> BenchConfig:
    """Return the configuration of one worker spec."""
    return BenchConfig(**spec["config"])


def make_spec(
    kind: str, backend: str, config: BenchConfig, budget: str = "full", **fields: object
) -> Spec:
    """Return one worker spec: the kind, the backend, the configuration and the pairing fields."""
    return {"kind": kind, "backend": backend, "config": asdict(config), "budget": budget, **fields}


def _rotate(items: list, offset: int) -> list:
    if not items:
        return items
    offset %= len(items)
    return items[offset:] + items[:offset]


# The plans.


def all_groups() -> list[tuple[str, int, int]]:
    """Return every (model size, input resolution, batch size) of the matrix, in the order of the
    run: from the smallest to the largest model size. The index is the group number."""
    return [(size, side, b) for size in MODEL_IDS for side in SIDES for b in BATCH_SIZES]


def group_workers(model_size: str, quantized: bool) -> list[tuple[str, str, int | None]]:
    """Return (backend, dtype mode, bits) of each worker of one group, before the rotation.

    torch-mps: torch.compile in every valid dtype mode (ADR 0009). mlx: the compiled forward pass
    in the dtype modes of _modes(model_size, "mlx"). Quantized: 8 and 4 bits in fp16-autocast and
    bf16-autocast for ViT-H+/16 and ViT-7B/16 (Table 11 of the plan).
    """
    workers: list[tuple[str, str, int | None]] = []
    torch_modes, mlx_modes = _modes(model_size, "torch-mps"), _modes(model_size, "mlx")
    for mode in DTYPE_MODE_NAMES:
        if mode in torch_modes:
            workers.append(("torch-mps", mode, None))
        if mode in mlx_modes:
            workers.append(("mlx", mode, None))
    if quantized and model_size in QUANTIZED_MODEL_SIZES:
        workers += [("mlx", mode, bits) for bits in (8, 4) for mode in QUANTIZED_MODES]
    return workers


def group_specs(
    group: int,
    workers: list[tuple[str, str, int | None, bool]],
    phase: str = "matrix",
    rotate: bool = True,
) -> list[Spec]:
    """Return the specs of one paired group, rotated by the group number."""
    model_size, side, batch_size = all_groups()[group]
    specs = []
    ordered = _rotate(workers, group) if rotate else workers
    for position, (backend, mode, bits, compile_) in enumerate(ordered, start=1):
        giant_torch = backend == "torch-mps" and model_size == GIANT_MODEL_SIZE
        config = BenchConfig(
            "B4" if giant_torch else RUN, model_size, mode, side, batch_size, compile_, bits
        )
        specs.append(
            make_spec(
                "bench",
                backend,
                config,
                run_id="matrix",
                group=group,
                round=1,
                position=position,
                phase=phase,
            )
        )
    return specs


def matrix_plan(model_sizes: list[str] | None = None, quantized: bool = False) -> list[Spec]:
    """Return the workers of the matrix run: one paired group for each model size, input
    resolution and batch size, from the smallest to the largest model size (Section 3.8)."""
    sizes = set(model_sizes or MODEL_IDS)
    specs = []
    for group, (model_size, _, _) in enumerate(all_groups()):
        if model_size in sizes:
            workers = [(*w, True) for w in group_workers(model_size, quantized)]
            specs += group_specs(group, workers)
    return specs


def gate_plan() -> list[Spec]:
    """Return the workers of the gate J13 (R53): ViT-7B/16 at 224 px batch size 1 and at 512 px
    batch size 8 in bf16-cast: torch-mps eager, torch-mps compiled and the mlx worker."""
    specs = []
    groups = all_groups()
    for side, batch_size in ((224, 1), (512, 8)):
        group = groups.index((GIANT_MODEL_SIZE, side, batch_size))
        workers = [
            ("torch-mps", "bf16-cast", None, False),
            ("torch-mps", "bf16-cast", None, True),
            ("mlx", "bf16-cast", None, True),
        ]
        specs += group_specs(group, workers, "gate", rotate=False)
    return specs


def follow_up_plan(listed: list[tuple[str, int, int, str]], matrix_lines: list[dict]) -> list[Spec]:
    """Return the follow-up groups of the compile check (job J16): torch-mps eager, torch-mps
    compiled and the fastest valid mlx mode of the dtype in that group."""
    from dinov3_mlx.speedup import _fastest

    groups = all_groups()
    specs = []
    for model_size, side, batch_size, mode in listed:
        group = groups.index((model_size, side, batch_size))
        mlx = [
            line
            for line in matrix_lines
            if line["backend"] == "mlx"
            and not line.get("bits")
            and (line["model_size"], line["side"], line["batch_size"])
            == (model_size, side, batch_size)
            and DTYPE_OF_MODE[line["dtype_mode"]] == DTYPE_OF_MODE[mode]
        ]
        best = _fastest(mlx, "frame")
        workers = [("torch-mps", mode, None, False), ("torch-mps", mode, None, True)]
        if best is not None:
            workers.append(("mlx", best["dtype_mode"], None, True))
        specs += group_specs(group, workers, "compile-check")
    return specs


def pairs_plan(priorities: list[str], rounds: int, run_id: str) -> list[Spec]:
    """Return the workers of one paired run at P1 and P2 (Section 3.11): in each round and dtype
    mode one compiled mlx worker and one torch.compile worker; the order of the two alternates
    from round to round, and the order of the dtype modes rotates by one."""
    specs = []
    for priority in priorities:
        model_size, side, batch_size, _ = PRIORITY[priority]
        position = 0
        for r in range(rounds):
            backends = ["torch-mps", "mlx"] if r % 2 == 0 else ["mlx", "torch-mps"]
            for mode in _rotate(DTYPE_MODE_NAMES, r):
                for backend in backends:
                    position += 1
                    config = BenchConfig(RUN, model_size, mode, side, batch_size, True)
                    specs.append(
                        make_spec(
                            "bench",
                            backend,
                            config,
                            run_id=run_id,
                            group=priority,
                            round=r + 1,
                            position=position,
                            priority=priority,
                            phase="pairs",
                        )
                    )
    return specs


def change_plan(
    changes: list[str], priorities: list[str], rounds: int, run_id: str | None = None
) -> list[Spec]:
    """Return the processes of the change ratios (Table 23, and C7 of Step 6e).

    C1, C2, C4 and C7: one process for each priority configuration and dtype mode, with all forms
    in it. C7 also runs ViT-H+/16 at 512 px with batch size 1 (JOINED_EXTRA). C3 and C5 compare
    configurations: one subprocess for each configuration and round, in alternation, rotated over
    the rounds.

    Args:
        changes: Keys of IN_PROCESS_CHANGES, "autocast" or "quantized".
        priorities: Keys of PRIORITY.
        rounds: The number of rounds.
        run_id: The run of a change that runs twice (C7: "1" or "2"), or None.
    """
    specs = []
    for change in changes:
        for priority in priorities:

            def config(
                mode: str, bits: int | None = None, priority: str = priority, batch: int = 0
            ) -> BenchConfig:
                size, side, batch_size, _ = PRIORITY[priority]
                return BenchConfig(RUN, size, mode, side, batch or batch_size, True, bits)

            budget = f"change-{priority}"
            if change in IN_PROCESS_CHANGES:
                number, forms = IN_PROCESS_CHANGES[change]
                cells = [(mode, 0, budget) for mode in DTYPE_MODE_NAMES]
                if change == "fp16-output":
                    cells = [(mode, 0, budget) for mode in HALF_MODES]
                if change == "joined":
                    cells = [(mode, 0, budget) for mode in JOINED_MODES[priority]]
                    for extra, batch, modes, extra_budget in JOINED_EXTRA:
                        if extra == priority:
                            cells += [(mode, batch, extra_budget) for mode in modes]
                for mode, batch, cell_budget in cells:
                    fields = {"run_id": run_id} if run_id else {}
                    specs.append(
                        make_spec(
                            "change",
                            "mlx",
                            config(mode, batch=batch),
                            cell_budget,
                            change=number,
                            forms=forms,
                            rounds=rounds,
                            priority=priority,
                            phase="change",
                            **fields,
                        )
                    )
            elif change == "autocast":
                for dtype in ("fp16", "bf16"):
                    forms = [f"{dtype}-cast", f"{dtype}-autocast"]
                    for r in range(rounds):
                        for mode in _rotate(forms, r):
                            specs.append(
                                make_spec(
                                    "bench",
                                    "mlx",
                                    config(mode),
                                    budget,
                                    change="C3",
                                    form=mode,
                                    round=r + 1,
                                    priority=priority,
                                    phase="change",
                                )
                            )
            elif change == "quantized":
                if priority != "P2":
                    continue  # Table 23: C5 runs at P2 only
                for mode in QUANTIZED_MODES:
                    forms: list[int | None] = [None, 8, 4]
                    for r in range(rounds):
                        for bits in _rotate(forms, r):
                            specs.append(
                                make_spec(
                                    "bench",
                                    "mlx",
                                    config(mode, bits),
                                    budget,
                                    change="C5",
                                    form=f"{bits}-bit" if bits else "unquantized",
                                    round=r + 1,
                                    priority=priority,
                                    phase="change",
                                )
                            )
            else:
                raise ValueError(f"unknown change {change!r}")
    return specs


def sweep_plan(rounds: int = 2) -> list[Spec]:
    """Return the workers of the batch-size sweep (Section 3.12): compiled mlx in bf16-cast."""
    points = [(size, side, b) for size, side, sizes in SWEEP for b in sizes]
    specs = []
    for r in range(rounds):
        for position, (size, side, b) in enumerate(_rotate(points, r * len(points) // 2), 1):
            config = BenchConfig(RUN, size, "bf16-cast", side, b, True)
            specs.append(
                make_spec(
                    "bench",
                    "mlx",
                    config,
                    run_id="sweep",
                    round=r + 1,
                    position=position,
                    phase="sweep",
                )
            )
    return specs


def learn_plan() -> list[Spec]:
    """Return the workers of the Learn data (Section 3.13): 768 and 1024 px on the pixels of
    preprocess_frames (finite only), the token range of ViT-H+/16 in bf16-autocast, and pairs
    with the Neural Accelerators off and on at P1 and P2."""
    specs = []
    for side in LEARN_SIDES:
        for size, mode in LEARN_MODES:
            for b in (1, 8):
                config = BenchConfig(RUN, size, mode, side, b, True)
                specs.append(make_spec("bench", "mlx", config, pixels="preprocess", phase="learn"))
    for side in LEARN_SIDES:
        config = BenchConfig(RUN, "ViT-H+/16", "bf16-autocast", side, 1)
        specs.append(make_spec("token-range", "mlx", config, phase="learn"))
    for priority in ("P1", "P2"):
        model_size, side, batch_size, _ = PRIORITY[priority]
        config = BenchConfig(RUN, model_size, "bf16-cast", side, batch_size, True)
        for r, nax in enumerate(["off", "on", "on", "off"]):
            specs.append(
                make_spec(
                    "bench",
                    "mlx",
                    config,
                    env=NAX_OFF if nax == "off" else {},
                    nax=nax,
                    round=r // 2 + 1,
                    priority=priority,
                    phase="nax",
                )
            )
    return specs


def profile_plan(priorities: list[str]) -> list[Spec]:
    """Return the P1 profile (Table 12): one subprocess for each of fp16-cast and bf16-cast."""
    specs = []
    for priority in priorities:
        model_size, side, batch_size, _ = PRIORITY[priority]
        for mode in ("fp16-cast", "bf16-cast"):
            config = BenchConfig(RUN, model_size, mode, side, batch_size, True)
            specs.append(make_spec("profile", "mlx", config, priority=priority, phase="profile"))
    return specs


def preprocess_plan(rounds: int = 5) -> list[Spec]:
    """Return the process of C6: the CPU path against preprocess_frames at 224 and 512 px."""
    config = BenchConfig(RUN, "ViT-S/16", "fp32", 224, 8)
    return [make_spec("preprocess", "cpu", config, change="C6", rounds=rounds, phase="change")]


# The memory need and the environment of a worker.


def spec_need(spec: Spec) -> int:
    """Return the memory need of one worker: the benchmark need of its backend (D8)."""
    config = spec_config(spec)
    if spec["kind"] == "preprocess":
        return PREPROCESS_NEED_BYTES
    if spec["kind"] == "profile":
        config = replace(config, batch_size=32)  # the profile also runs batch size 32
    backend = "torch-mps" if spec["backend"] == "torch-mps" else "mlx"
    return required_bench_bytes(config, backend=backend)


def worker_env(spec: Spec, base: dict[str, str]) -> dict[str, str]:
    """Return the environment of one worker subprocess: the MPS watermarks for torch-mps (and for
    the CPU path of C6, which imports torch), MLX_ENABLE_TF32=0 for mlx, plus the env of the spec.
    The child inherits DINOV3_GPU_LOCK, so it does not wait for the driver."""
    env = {k: v for k, v in base.items() if k not in WATERMARKS and k != "MLX_ENABLE_TF32"}
    env.pop("MLX_METAL_GPU_ARCH", None)
    env["MLX_ENABLE_TF32"] = "0"
    if spec["backend"] in ("torch-mps", "cpu"):
        env |= WATERMARKS
    return env | spec.get("env", {})


# The run.


def run_worker(spec: Spec, bench_dir: Path) -> dict:
    """Run one worker in a new subprocess and return its result (run_worker_subprocess)."""
    command = [
        sys.executable,
        __file__,
        "--worker",
        json.dumps(spec | {"bench_dir": str(bench_dir)}),
    ]
    timeout = WORKER_TIMEOUT_S[spec["backend"]]
    return run_worker_subprocess(command, worker_env(spec, dict(os.environ)), timeout, STDERR_LINES)


PAIR_KEYS = ["run_id", "group", "round", "position", "priority", "phase", "change", "form", "forms"]


def spec_line(spec: Spec, worker: dict, need: int, available: int, machine: object) -> dict:
    """Return the JSON line of one worker: the fields of LINE_FIELDS, the pairing fields and the
    fields of the spec."""
    backend = spec["backend"]
    line = result_line(spec_config(spec), worker, need, available, machine, backend)
    for key in PAIR_KEYS + ["nax", "mark", "kind"]:
        if key in spec:
            line[key] = spec[key]
    if spec.get("env"):
        line["env"] = spec["env"]
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

    A compiled torch-mps worker of the matrix that ends with an error gets an eager worker at the
    same position; both lines stay, and the eager line carries the mark of D9 option b.

    Args:
        specs: The workers.
        results: The results file.
        bench_dir: The folder of the per-batch files.
        runner: Runs one worker. The tests give a fake runner.
        pattern: The pattern of the heavy jobs for other_heavy_jobs_seen.
        registry: The registry of the lock-aware processes.

    Returns:
        The number of lines for each status.

    Raises:
        MachineStateError: On battery power or in Low Power Mode. No line is written.
    """
    counts: Counter = Counter()
    queue = list(specs)
    index = 0
    while queue:
        spec = queue.pop(0)
        index += 1
        config = spec_config(spec)
        machine = read_machine_state()
        check_machine_state(machine)
        need = spec_need(spec)
        available = available_memory_bytes()
        fits = available >= need
        label = f"{spec['backend']} {spec['kind']} {config.label}"
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
        summary = line["reason"].splitlines()[-1] if line["reason"] else ""
        if line["status"] == "done" and isinstance(line.get("frame"), dict):
            summary = (
                f"valid {line['valid']}, frame {line['frame']['throughput_ips']:.1f} img/s, "
                f"patch {line['patch']['throughput_ips']:.1f} img/s"
            )
        logger.info(f"[{index}/{index + len(queue)}] {label}: {line['status']}: {summary}")
        fallback = (
            spec["backend"] == "torch-mps" and config.compile and spec.get("phase") == "matrix"
        )
        if fallback and line["status"] == "error":
            eager = spec | {
                "config": asdict(replace(config, compile=False)),
                "mark": "eager torch: torch.compile failed",
            }
            queue.insert(0, eager)
    return counts


def read_lines(path: Path) -> list[dict]:
    """Return the JSON lines of a results file, or [] if it does not exist."""
    if not path.exists():
        return []
    return [json.loads(text) for text in path.read_text().splitlines() if text.strip()]


def _line_identity(line: dict) -> tuple:
    return (line.get("phase", "matrix"), line["backend"], line_key(line))


def _spec_identity(spec: Spec) -> tuple:
    return (spec.get("phase", "matrix"), spec["backend"], spec_config(spec).key)


def skip_done(specs: list[Spec], lines: list[dict]) -> list[Spec]:
    """Leave out each group whose workers all have a line (--skip-done)."""
    known = {_line_identity(line) for line in lines}
    groups: dict[object, list[Spec]] = {}
    for spec in specs:
        groups.setdefault(spec.get("group"), []).append(spec)
    return [
        spec
        for group in groups.values()
        if not all(_spec_identity(s) in known for s in group)
        for spec in group
    ]


def only_skipped(specs: list[Spec], lines: list[dict]) -> list[Spec]:
    """Keep each group with a worker whose latest line is "skipped" (job J15): the whole group
    runs again."""
    latest = {_line_identity(line): line for line in lines}
    skipped_groups = {
        spec.get("group")
        for spec in specs
        if latest.get(_spec_identity(spec), {}).get("status") == "skipped"
    }
    return [spec for spec in specs if spec.get("group") in skipped_groups]


def dry_run(specs: list[Spec]) -> None:
    """Print each worker with its memory need. Load no model, import no backend."""
    available = available_memory_bytes()
    print(f"available memory: {available / GB:.1f} GB")
    for number, spec in enumerate(specs, start=1):
        config = spec_config(spec)
        need = spec_need(spec)
        extra = " ".join(
            f"{key} {spec[key]}"
            for key in ("group", "round", "run_id", "change", "form", "forms", "nax")
            if key in spec
        )
        fits = "fits" if need <= available else "does not fit"
        print(
            f"{number:4d} {spec['backend']:9s} {spec['kind']:11s} {config.label:44s} {extra:30s} "
            f"need {need / GB:5.2f} GB  {fits}"
        )
    print(f"{len(specs)} workers")


def bench_dir_of(results: Path, subcommand: str) -> Path:
    """Return the folder of the per-batch files and of driver.log for one results file."""
    if results.resolve() == RESULTS[subcommand].resolve():
        return BENCH_ROOT / subcommand
    if results.resolve() == JOINED_RESULTS.resolve():
        return BENCH_ROOT / "joined"
    return results.with_name(f"{results.stem}-batches")


def plan_of(args: argparse.Namespace) -> list[Spec]:
    """Return the workers of the subcommand from the command-line options."""
    if args.subcommand == "change":
        return change_plan(args.change, args.priority, args.rounds, args.run)
    if args.subcommand == "pairs":
        return pairs_plan(args.priority, args.rounds, str(args.run))
    if args.subcommand == "sweep":
        return sweep_plan(args.rounds)
    if args.subcommand == "learn":
        return learn_plan()
    if args.subcommand == "profile":
        return profile_plan(args.priority)
    if args.subcommand == "preprocess":
        return preprocess_plan(args.rounds)
    lines = read_lines(args.results)
    if args.gate_7b:
        return gate_plan()
    if args.compile_check:
        from dinov3_mlx.speedup import compile_check

        step3 = read_lines(RESULTS_DIR / "baseline-torch-mps.jsonl")
        return follow_up_plan(compile_check(lines, step3), lines)
    specs = matrix_plan(args.model_sizes, args.quantized)
    if args.only_skipped:
        specs = only_skipped(specs, lines)
    if args.skip_done:
        specs = skip_done(specs, lines)
    return specs


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Return the command-line options.

    Args:
        argv: The arguments. None reads sys.argv.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    sub = parser.add_subparsers(dest="subcommand")
    commands = {
        name: sub.add_parser(name)
        for name in ("change", "matrix", "pairs", "sweep", "preprocess", "learn", "profile")
    }
    for name, command in commands.items():
        command.add_argument("--dry-run", action="store_true", help="print the plan, load no model")
        command.add_argument("--results", type=Path, default=RESULTS[name])
    for name in ("change", "pairs", "profile"):
        default = ["P1"] if name == "profile" else ["P1", "P2"]
        commands[name].add_argument(
            "--priority", nargs="+", choices=list(PRIORITY), default=default
        )
    for name, rounds in (("change", 5), ("pairs", 5), ("sweep", 2), ("preprocess", 5)):
        commands[name].add_argument("--rounds", type=int, default=rounds)
    commands["change"].add_argument("--change", nargs="+", choices=CHANGES, required=True)
    commands["change"].add_argument("--run", help="the run of C7: 1 or 2")
    commands["pairs"].add_argument("--run", type=int, required=True, help="the run: 1 or 2")
    matrix = commands["matrix"]
    matrix.add_argument("--model-sizes", nargs="+", choices=list(MODEL_IDS))
    matrix.add_argument("--quantized", action="store_true", help="the quantized rows (Table 11)")
    matrix.add_argument("--skip-done", action="store_true", help="leave out complete groups")
    matrix.add_argument("--only-skipped", action="store_true", help="groups with a skip (J15)")
    matrix.add_argument("--compile-check", action="store_true", help="follow-up groups (J16)")
    matrix.add_argument("--gate-7b", action="store_true", help="the gate of ViT-7B/16 (J13)")
    argv_list = sys.argv[1:] if argv is None else list(argv)
    args = parser.parse_args(argv)
    if args.worker is None and args.subcommand is None:
        parser.error("give a subcommand")
    if args.subcommand == "change" and args.change == ["joined"] and "--results" not in argv_list:
        args.results = JOINED_RESULTS
    return args


def main(argv: Sequence[str] | None = None) -> None:
    """Parse the arguments and run the workers. A run that is not a dry run takes the GPU lock.

    Args:
        argv: The arguments. None reads sys.argv.
    """
    args = parse_args(argv)
    if args.worker:
        worker_main(args.worker)
        return
    if os.environ.get("MLX_DISABLE_COMPILE"):
        print(
            "bench_speed.py: MLX_DISABLE_COMPILE is set, so the compiled rows would run eager "
            "(R42). Unset it.",
            file=sys.stderr,
        )
        sys.exit(1)
    specs = plan_of(args)
    if args.dry_run:
        dry_run(specs)
        return
    bench_dir = bench_dir_of(args.results, args.subcommand)
    bench_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
        handlers=[logging.StreamHandler(sys.stderr), logging.FileHandler(bench_dir / "driver.log")],
    )
    with heavy_job(f"scripts/bench_speed.py {args.subcommand}"):
        logger.info(f"run of {len(specs)} workers: {args.subcommand}")
        try:
            counts = run_specs(specs, args.results, bench_dir)
        except (MachineStateError, WeightsNotFoundError) as error:
            logger.error(f"stop: {error}")
            sys.exit(1)
        logger.info(f"end of run: {dict(counts)}")


# The worker side.


def _learn_batch(side: int, batch_size: int) -> "object":
    """The pixels of the Learn data: preprocess_frames of the three test images, cycled."""
    import numpy as np

    from dinov3_mlx.mlx_preprocess import preprocess_frames
    from dinov3_mlx.parity import TEST_IMAGES
    from dinov3_mlx.preprocess import load_image
    from dinov3_mlx.stored import IMAGES_DIR

    images = []
    for name in TEST_IMAGES:
        frame = load_image(IMAGES_DIR / f"{name}.jpg")[None]
        images.append(np.array(preprocess_frames(frame, (side, side)))[0])
    return np.stack([images[i % len(images)] for i in range(batch_size)]).astype(np.float32)


def worker_main(spec_json: str) -> None:
    """Run one worker in this process and print the result as one JSON line.

    The memory check of the model load goes to stderr; the driver writes it to driver.log.

    Args:
        spec_json: The worker spec with bench_dir, as JSON.
    """
    send_checks_to(sys.stderr)
    spec = json.loads(spec_json)
    bench_dir = Path(spec["bench_dir"])
    config = spec_config(spec)
    warm_up, timed = BUDGETS[spec.get("budget", "full")]
    try:
        if spec["kind"] == "bench" and spec["backend"] == "mlx":
            from dinov3_mlx.mlx_bench import run_mlx_worker

            batch = None
            if spec.get("pixels") == "preprocess":
                batch = _learn_batch(config.side, config.batch_size)
            result = run_mlx_worker(
                config, warm_up, timed, bench_dir / "mlx", run_id=spec.get("run_id"), batch=batch
            )
        elif spec["kind"] == "bench":
            from dinov3_mlx.torch_bench import run_worker as run_torch_worker

            result = run_torch_worker(config, warm_up, timed, bench_dir / "torch-mps")
        elif spec["kind"] == "change":
            from dinov3_mlx.mlx_bench import run_change_worker

            result = run_change_worker(
                config, spec["change"], spec["forms"], spec["rounds"], warm_up, timed
            )
        elif spec["kind"] == "profile":
            from dinov3_mlx.mlx_bench import run_profile_worker

            result = run_profile_worker(config, bench_dir / "profile")
        elif spec["kind"] == "token-range":
            from dinov3_mlx.mlx_bench import run_token_range_worker

            result = run_token_range_worker(config, _learn_batch(config.side, 1))
        elif spec["kind"] == "preprocess":
            result = run_preprocess_worker(spec["rounds"])
        else:
            raise ValueError(f"unknown worker kind {spec['kind']!r}")
    except InsufficientMemoryError as error:
        result = {"status": "skipped", "reason": f"load check in the worker: {error}"}
    watermarks = {k: os.environ[k] for k in WATERMARKS if k in os.environ}
    result["mps_watermarks"] = watermarks or None
    result["torch_imported"] = "torch" in sys.modules
    print(json.dumps(result))


def run_preprocess_worker(rounds: int) -> dict:
    """C6 (D12): the CPU path (preprocess, torch fp64, frame by frame) against preprocess_frames
    on a batch of 8 uint8 frames of 1080 × 1920 px, host to device included, at 224 and 512 px.
    The process imports torch for the CPU path; it is not an mlx worker."""
    import time

    import mlx.core as mx
    import numpy as np
    from PIL import Image

    from dinov3_mlx.mlx_preprocess import preprocess_frames
    from dinov3_mlx.parity import TEST_IMAGES
    from dinov3_mlx.preprocess import load_image, preprocess
    from dinov3_mlx.stored import IMAGES_DIR

    frames = []
    for i in range(8):
        image = Image.fromarray(load_image(IMAGES_DIR / f"{TEST_IMAGES[i % 3]}.jpg"))
        frames.append(np.asarray(image.resize((1920, 1080), Image.Resampling.BILINEAR)))
    batch = np.stack(frames)
    per_side = {}
    for side in (224, 512):
        cpu_ms, mlx_ms = [], []
        mx.eval(preprocess_frames(batch, (side, side)))  # the first call builds the matrices
        for r in range(rounds):
            for form in ["cpu", "mlx"] if r % 2 == 0 else ["mlx", "cpu"]:
                start = time.perf_counter()
                if form == "cpu":
                    for frame in frames:
                        preprocess(frame, (side, side))
                    cpu_ms.append((time.perf_counter() - start) * 1000 / len(frames))
                else:
                    mx.eval(preprocess_frames(batch, (side, side)))
                    mlx_ms.append((time.perf_counter() - start) * 1000 / len(frames))
        per_side[str(side)] = {
            "cpu_ms_per_frame": cpu_ms,
            "mlx_ms_per_frame": mlx_ms,
            "cpu_median_ms": float(np.median(cpu_ms)),
            "mlx_median_ms": float(np.median(mlx_ms)),
            "ratios": [c / m for c, m in zip(cpu_ms, mlx_ms, strict=True)],
        }
    from datetime import UTC, datetime

    from dinov3_mlx.machine import read_versions

    return {
        "status": "done",
        "reason": "",
        "preprocess": per_side,
        "frame_size": [1080, 1920],
        "versions": read_versions(),
        "mlx_architecture": mx.device_info()["architecture"],
        "time": datetime.now(UTC).isoformat(timespec="seconds"),
    }


if __name__ == "__main__":
    main()
