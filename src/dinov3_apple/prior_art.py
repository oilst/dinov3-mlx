"""The prior-art comparison of Step 6b: the candidates, the Core ML exports, the parity of an
export, the pair plans and the rows of the result table (specs/06b-prior-art/plan.html).

The third-party code and files live outside the repo, in PRIOR_ART_DIR, each candidate in its own
folder; each environment there has its own venv (ADR 0013). This module reads no file of a
candidate and imports no backend: the driver scripts/bench_prior_art.py imports it for a dry run.
The port-change variants, which need mlx, live in port_variants.py.
"""

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from dinov3_apple.bench import TIMED, WARM_UP, BenchConfig, TimeBudget
from dinov3_apple.outputs import NUM_PREFIX_TOKENS
from dinov3_apple.parity import THRESHOLDS, measure_parity
from dinov3_apple.speedup import NOISE, _lower_median, pass_verdict, single_verdict
from dinov3_apple.stored import IMAGENET_MEAN_VALUES, IMAGENET_STD_VALUES

PRIOR_ART_DIR = Path(
    os.environ.get("DINOV3_PRIOR_ART_DIR", Path.home() / ".cache" / "dinov3-apple" / "prior-art")
)
RUN = "P6b"  # the run of every line of results/prior-art.jsonl
VERDICTS = ("measure", "gated", "skip")
UNITS = "CPU_AND_GPU"  # the compute units of the paired Core ML rows (D4)
BATCH_SIZES = (1, 8)  # each export runs at batch sizes 1 and 8 (D5)
# The time budgets of a worker: the full budget of bench.py (D6) and a short one for T15.
BUDGETS = {
    "full": (WARM_UP, TIMED),
    "short": (TimeBudget(1, 0.0), TimeBudget(3, 1.0)),
    "none": (TimeBudget(0, 0.0), TimeBudget(0, 0.0)),
}
BLOCKED_REASON = (
    "blocked: no permission to run third-party Python code (Q1 of specs/06b-prior-art/plan.html)"
)
J5_COMMAND = (
    "uv run python -m dinov3_apple.gpu_lock -- uv run python scripts/bench_prior_art.py ports "
    "--rounds 5 --allow-third-party-python"
)


@dataclass(frozen=True)
class Candidate:
    """One related public repo or file of Table 1 of specs/06b-prior-art/explore.html.

    Attributes:
        name: The name of the repo or the file set.
        kind: "MLX port", "Core ML export", "pipeline" or "DINOv2 port".
        license: The license of the code or the weights.
        commit: The commit or the revisions that the Explore read.
        verdict: "measure", "gated" (measure after the user's yes to Q1) or "skip".
        reason: Why the candidate is skipped or gated; "" for "measure".
    """

    name: str
    kind: str
    license: str
    commit: str
    verdict: str
    reason: str = ""


CANDIDATES: dict[str, Candidate] = {
    "C1": Candidate(
        "mlx-image (riccardomusmeci), DINOv3 part",
        "MLX port, Python",
        "MIT",
        "da92532",
        "gated",
        "third-party Python code: gated by Q1; the user said yes on 2026-11-08, so J5 ran "
        "(Section 6); the port changes run as variants V1 to V5",
    ),
    "C2": Candidate(
        "mlx_dinov3 (etornam45)",
        "MLX port, Python",
        "DINOv3 License",
        "43d139b",
        "gated",
        "third-party Python code: gated by Q1; the user said yes on 2026-11-08, so J5 ran "
        "(Section 6); its converter reads a pickle, so J5 uses our own converted weights",
    ),
    "C3": Candidate(
        "MLXDINOv3 (vincentamato)",
        "MLX port, Swift",
        "MIT",
        "3122d79",
        "skip",
        "needs a third-party Swift build and a new Swift harness; its design is our eager "
        "forward pass in Swift (Table 2 of explore.html)",
    ),
    "C4": Candidate(
        "mlx-dinov3 (sarveshmayil)",
        "MLX port, Python",
        "MIT",
        "0bdeaa0",
        "skip",
        "no ViT model in the repo",
    ),
    "C5": Candidate(
        "pipelines: sam-3d-body-dinov3-bf16, mlx-trellis2-swift, pixal3d-mlx, mlx-swift-INSID3",
        "pipeline",
        "mixed",
        "2026-06 to 2026-11",
        "skip",
        "not a backbone: large dependencies, and the backbone weights can differ from the "
        "pretrained weights",
    ),
    "C6": Candidate(
        "bahaehmimdi/dinov2-mlx",
        "DINOv2 port",
        "none",
        "3d1fcf9",
        "skip",
        "no license; the install gets more code from two repos (a torch replacement and a "
        "torchvision fork)",
    ),
    "C7": Candidate(
        "SharpAI (Aegis AI) Core ML exports",
        "Core ML export",
        "DINOv3 License (weights)",
        "c76cab6, fdec13c, 42e4018, 0baa6bd",
        "measure",
    ),
    "C8": Candidate(
        "rinpatch/dinov3-coreml",
        "Core ML export",
        "MIT (code), DINOv3 License (weights)",
        "5ea0dea",
        "measure",
    ),
    "C9": Candidate(
        "VimalMollyn/dinov2-coreml",
        "Core ML export",
        "no license file: numbers only, no file in the repo",
        "6306a84",
        "measure",
    ),
}


@dataclass(frozen=True)
class Export:
    """One Core ML export of Table 2 of the plan and the cell of ours that it pairs with.

    Attributes:
        candidate: The candidate, "C7" to "C9".
        model_size: The model size of the export, in words.
        side: The input resolution of the export.
        patch_size: The patch size of the export.
        precision: "FP32", "FP16" or "INT8".
        package: The path of the .mlpackage below PRIOR_ART_DIR.
        input_kind: "spec-array" (an image input, changed in the loaded spec) or "array".
        outputs: The outputs that the export gives: "patch" or "frame, patch".
        family: "dinov3" or "dinov2".
        reference_model_size: The model size of the reference file, or "" (no parity).
        ours_model_size: The model size of the cell of ours.
        ours_side: The input resolution of the cell of ours.
        dtype_mode: The dtype mode of the cell of ours (D5).
        pixels: "stored" for the stored pixels, "random" for random pixels (E7).
    """

    candidate: str
    model_size: str
    side: int
    patch_size: int
    precision: str
    package: str
    input_kind: str
    outputs: str
    family: str
    reference_model_size: str
    ours_model_size: str
    ours_side: int
    dtype_mode: str
    pixels: str = "stored"

    @property
    def path(self) -> Path:
        """The .mlpackage folder."""
        return PRIOR_ART_DIR / self.package

    @property
    def tokens(self) -> int:
        """The token count: the CLS token, 4 register tokens and the patch tokens."""
        return NUM_PREFIX_TOKENS + (self.side // self.patch_size) ** 2

    @property
    def threshold_mode(self) -> str:
        """The key of THRESHOLDS of the verdict: fp32 for FP32 and INT8, fp16-cast for FP16."""
        return "fp16-cast" if self.precision == "FP16" else "fp32"

    @property
    def report_only(self) -> bool:
        """True for an export without a pass rule: INT8 (a quantized row) and DINOv2."""
        return self.precision == "INT8" or not self.reference_model_size


_SHARP = "coreml/SharpAI"
EXPORTS: dict[str, Export] = {
    "E1": Export(
        "C7",
        "ViT-S/16",
        224,
        16,
        "FP32",
        f"{_SHARP}/dinov3-vits16-pretrain-lvd1689m-coreml-FP32-224x224/"
        "dinov3_vits16_224x224_fp32.mlpackage",
        "spec-array",
        "patch",
        "dinov3",
        "ViT-S/16",
        "ViT-S/16",
        224,
        "fp32",
    ),
    "E2": Export(
        "C7",
        "ViT-S/16",
        224,
        16,
        "INT8",
        f"{_SHARP}/dinov3-vits16-pretrain-lvd1689m-coreml-INT8-224x224/"
        "dinov3_vits16_224x224_int8.mlpackage",
        "spec-array",
        "patch",
        "dinov3",
        "ViT-S/16",
        "ViT-S/16",
        224,
        "fp32",
    ),
    "E3": Export(
        "C9",
        "ViT-S/16",
        512,
        16,
        "FP16",
        "coreml/VimalMollyn/dinov2-coreml/dinov3_vits16_518_fp16.mlpackage",
        "array",
        "patch",
        "dinov3",
        "ViT-S/16",
        "ViT-S/16",
        512,
        "fp16-cast",
    ),
    "E4": Export(
        "C7",
        "ViT-B/16",
        448,
        16,
        "FP32",
        f"{_SHARP}/dinov3-vitb16-pretrain-lvd1689m-coreml-FP32-448x448/"
        "dinov3_vitb16_448x448_fp32.mlpackage",
        "spec-array",
        "patch",
        "dinov3",
        "ViT-B/16",
        "ViT-B/16",
        448,
        "fp32",
    ),
    "E5": Export(
        "C8",
        "ViT-B/16",
        448,
        16,
        "FP32",
        "coreml/rinpatch/dinov3-coreml/models/DINOv3ViTB16-FP32-448.mlpackage",
        "spec-array",
        "frame, patch",
        "dinov3",
        "ViT-B/16",
        "ViT-B/16",
        448,
        "fp32",
    ),
    "E6": Export(
        "C7",
        "ViT-B/16",
        448,
        16,
        "FP16",
        f"{_SHARP}/dinov3-vitb16-pretrain-lvd1689m-coreml-FP16-448x448/"
        "dinov3_vitb16_448x448_fp16.mlpackage",
        "spec-array",
        "patch",
        "dinov3",
        "ViT-B/16",
        "ViT-B/16",
        448,
        "fp16-cast",
    ),
    "E7": Export(
        "C9",
        "DINOv2 ViT-S/14 + 4 registers",
        518,
        14,
        "FP16",
        "coreml/VimalMollyn/dinov2-coreml/dinov2_small_14_registers_518_fp16.mlpackage",
        "array",
        "patch",
        "dinov2",
        "",
        "ViT-S/16",
        592,
        "fp16-cast",
        "random",
    ),
}


@dataclass(frozen=True)
class Group:
    """The exports that pair with one cell of ours (Table 2 of the plan).

    Attributes:
        name: "G1" to "G5".
        model_size: The model size of ours.
        side: The input resolution of ours.
        dtype_mode: The dtype mode of ours.
        pixels: "stored" or "random".
        exports: The exports of the group. Two exports share the worker of ours in each round.
    """

    name: str
    model_size: str
    side: int
    dtype_mode: str
    pixels: str
    exports: tuple[str, ...]


def _groups() -> list[Group]:
    cells: dict[tuple, list[str]] = {}
    for name, export in EXPORTS.items():
        key = (export.ours_model_size, export.ours_side, export.dtype_mode, export.pixels)
        cells.setdefault(key, []).append(name)
    return [
        Group(f"G{i}", *key, tuple(names)) for i, (key, names) in enumerate(cells.items(), start=1)
    ]


GROUPS = _groups()
GROUP_OF_EXPORT = {name: group.name for group in GROUPS for name in group.exports}

# The port cells of J5 (D2, Table 4 of the plan): (candidate, model, our model size, port side, our
# side, dtype of the port, dtype mode of ours, batch size). "dinov2" is the DINOv2 model of C1.
PORT_CELLS: list[tuple[str, str, str, int, int, str, str, int]] = [
    ("C1", "ViT-S/16", "ViT-S/16", 224, 224, "fp32", "fp32", 8),
    ("C1", "ViT-S/16", "ViT-S/16", 224, 224, "bf16", "bf16-cast", 8),
    ("C1", "ViT-S/16", "ViT-S/16", 224, 224, "fp16", "fp16-cast", 8),
    ("C1", "ViT-S/16", "ViT-S/16", 224, 224, "fp32", "fp32", 1),
    ("C2", "ViT-H+/16", "ViT-H+/16", 512, 512, "fp32", "fp32", 8),
    ("C2", "ViT-H+/16", "ViT-H+/16", 512, 512, "bf16", "bf16-cast", 8),
    ("C1", "dinov2", "ViT-S/16", 518, 592, "fp32", "fp32", 8),
    ("C1", "dinov2", "ViT-S/16", 518, 592, "bf16", "bf16-cast", 8),
]

# The port-change variants (D8, Table 3 of the plan).
VARIANTS: dict[str, str] = {
    "V0": "none: our compiled forward pass",
    "V1": "RoPE of C1 and C2: bf16 tables, q and k rotated in bf16",
    "V2": "the bf16 tables of V1 in fp32, our fp32 rotation",
    "V3": "Meta periods: the fp32 periods rounded to bf16; fp32 tables and rotation",
    "V4": "unfused attention: softmax(q kᵀ · scale) v in the matmul dtype",
    "V5": "8-bit input: the stored pixels rounded to 8-bit RGB",
}
VARIANT_MODEL_SIZES = ["ViT-S/16", "ViT-H+/16"]
VARIANT_MODES = ["fp32", "bf16-cast"]
SPEED_VARIANTS = ["V0", "V1", "V4"]
# P1 and P2 of Step 6 (PRIORITY of bench.py): model size, input resolution, batch size.
SPEED_POINTS = {"P1": ("ViT-S/16", 224, 8), "P2": ("ViT-H+/16", 512, 8)}


# The parity of an export (Section 3.6).


def grid_tokens(grid: np.ndarray) -> np.ndarray:
    """Return the tokens (h · w, D) of a grid output (1, D, h, w) or (D, h, w), row-major."""
    grid = grid.reshape(grid.shape[-3:])
    return grid.reshape(grid.shape[0], -1).T


def export_outputs(outputs: dict[str, np.ndarray]) -> tuple[np.ndarray | None, np.ndarray]:
    """Return the frame embedding (D,) or None and the patch embeddings (N, D) of one image.

    The output names of the exports: "features" (1, D, h, w) of C7, "embedding" (1, D) and
    "patch_embeddings" (1, N, D) of C8, "x_norm_patchtokens" (1, N, D) of C9.

    Raises:
        KeyError: For outputs without a known patch output.
    """
    frame = outputs.get("embedding")
    frame = None if frame is None else np.asarray(frame, np.float32).reshape(-1)
    if "features" in outputs:
        return frame, grid_tokens(np.asarray(outputs["features"], np.float32))
    for name in ("patch_embeddings", "x_norm_patchtokens"):
        if name in outputs:
            patches = np.asarray(outputs[name], np.float32)
            return frame, patches.reshape(-1, patches.shape[-1])
    raise KeyError(f"no patch output in {sorted(outputs)}")


def export_parity(
    reference: np.ndarray, frame: np.ndarray | None, patches: np.ndarray
) -> dict[str, float | None]:
    """Return the parity of one export output against the reference tokens (as check_validity).

    The token sequence gets the frame embedding (the reference CLS row without one), the 4
    reference register rows and the patch embeddings. The cosine does not depend on the length of
    a vector, so L2-normalized outputs need no change.

    Args:
        reference: The reference tokens (1 + 4 + N, D).
        frame: The frame embedding (D,), or None.
        patches: The patch embeddings (N, D).

    Returns:
        The fields of Parity; frame is None for an export without a frame embedding.
    """
    first = reference[:1] if frame is None else np.asarray(frame, np.float32)[None]
    tokens_ = np.concatenate([first, reference[1:NUM_PREFIX_TOKENS], patches]).astype(np.float32)
    parity = asdict(measure_parity(reference, tokens_))
    if frame is None:
        parity["frame"] = None
    return parity


def parity_passes(parity: dict[str, float | None], threshold_mode: str) -> bool:
    """Return True if each value of the parity reaches the threshold of THRESHOLDS[threshold_mode].
    A frame of None does not count."""
    threshold = THRESHOLDS[threshold_mode]
    names = ("frame", "patch_mean", "patch_p01")
    return all(
        parity[name] >= getattr(threshold, name) for name in names if parity[name] is not None
    )


def worst_parity(parities: list[dict[str, float | None]]) -> dict[str, float | None]:
    """Return the smallest value of each field over the test images."""
    return {
        name: (None if any(p[name] is None for p in parities) else min(p[name] for p in parities))
        for name in parities[0]
    }


def eight_bit_pixels(pixels: np.ndarray) -> np.ndarray:
    """Round normalized pixels to the 8-bit RGB grid and back (V5, the input of C7 and C8).

    Args:
        pixels: Normalized fp32 pixels (B, 3, H, W).

    Returns:
        fp32 pixels on the 8-bit grid: each pixel moves by at most 0.5 / 255 / std of its channel.
    """
    mean = IMAGENET_MEAN_VALUES.reshape(1, 3, 1, 1)
    std = IMAGENET_STD_VALUES.reshape(1, 3, 1, 1)
    levels = np.clip(np.rint((pixels * std + mean) * 255), 0, 255)
    return ((levels / 255 - mean) / std).astype(np.float32)


# The pair plans (Section 3.4). A spec is a dict: kind, job, the cell, round, position, group.


def _rotate(items: list, offset: int) -> list:
    offset %= len(items)
    return items[offset:] + items[:offset]


def ours_spec(group: Group, batch_size: int, job: str = "J1") -> dict:
    """Return the spec of one worker of ours: the compiled forward pass of the cell of a group."""
    config = BenchConfig(RUN, group.model_size, group.dtype_mode, group.side, batch_size, True)
    return {
        "kind": "ours",
        "job": job,
        "config": asdict(config),
        "pixels": group.pixels,
        "group": group.name,
        "batch_size": batch_size,
        "budget": "full",
    }


def export_spec(
    name: str,
    batch_size: int,
    units: str = UNITS,
    input_kind: str | None = None,
    job: str = "J1",
    parity_only: bool = False,
) -> dict:
    """Return the spec of one Core ML worker.

    Args:
        name: The export, "E1" to "E7".
        batch_size: 1, or 8: one predict call with a list of 8 inputs.
        units: The compute units.
        input_kind: None for the input kind of the export; "image" for the 8-bit image input.
        job: The job of Table 4 of the plan.
        parity_only: True for a worker without timed batches.
    """
    export = EXPORTS[name]
    return {
        "kind": "export",
        "job": job,
        "export": name,
        "candidate": export.candidate,
        "package": str(export.path),
        "input": input_kind or export.input_kind,
        "units": units,
        "batch_size": batch_size,
        "side": export.side,
        "pixels": export.pixels,
        "group": GROUP_OF_EXPORT[name],
        "parity_only": parity_only,
        "budget": "none" if parity_only else "full",
    }


def _number(specs: list[dict]) -> list[dict]:
    for position, spec in enumerate(specs, start=1):
        spec["position"] = position
    return specs


def coreml_plan(rounds: int = 5) -> list[dict]:
    """Return the workers of J1: in each round the groups in a rotated order; in each group and
    batch size, ours first in odd rounds and last in even rounds (D6)."""
    specs = []
    for r in range(rounds):
        for group in _rotate(GROUPS, r):
            for batch_size in BATCH_SIZES:
                ours = ours_spec(group, batch_size)
                exports = [export_spec(name, batch_size) for name in group.exports]
                cell = [ours, *exports] if r % 2 == 0 else [*exports, ours]
                specs += [spec | {"round": r + 1} for spec in cell]
    return _number(specs)


def all_units_plan() -> list[dict]:
    """Return the 14 export workers of J2 with the compute units ALL, one round, report only."""
    specs = [
        export_spec(name, b, units="ALL", job="J2") | {"round": 1}
        for name in EXPORTS
        for b in BATCH_SIZES
    ]
    return _number(specs)


def image_input_plan() -> list[dict]:
    """Return the 3 parity-only workers of J2 with the 8-bit image input (D3): E1, E4, E5."""
    specs = [
        export_spec(name, 1, input_kind="image", job="J2", parity_only=True) | {"round": 1}
        for name in ("E1", "E4", "E5")
    ]
    return _number(specs)


def variant_parity_plan(model_sizes: list[str] | None = None) -> list[dict]:
    """Return the workers of J3: V0 to V5 × model sizes × fp32, bf16-cast; one load each."""
    specs = []
    for model_size in model_sizes or VARIANT_MODEL_SIZES:
        for mode in VARIANT_MODES:
            for variant in VARIANTS:
                config = BenchConfig(RUN, model_size, mode, 512, 1, True)
                specs.append(
                    {
                        "kind": "variant-parity",
                        "job": "J3",
                        "config": asdict(config),
                        "variant": variant,
                        "group": f"{model_size} {mode}",
                        "round": 1,
                    }
                )
    return _number(specs)


def variant_speed_plan(priorities: list[str] | None = None, rounds: int = 3) -> list[dict]:
    """Return the workers of J4: V0, V1, V4 at P1 and P2 in fp32 and bf16-cast; the order of the
    variants rotates each round."""
    specs = []
    for r in range(rounds):
        for priority in priorities or list(SPEED_POINTS):
            model_size, side, batch_size = SPEED_POINTS[priority]
            for mode in VARIANT_MODES:
                for variant in _rotate(SPEED_VARIANTS, r):
                    config = BenchConfig(RUN, model_size, mode, side, batch_size, True)
                    specs.append(
                        {
                            "kind": "variant-speed",
                            "job": "J4",
                            "config": asdict(config),
                            "variant": variant,
                            "priority": priority,
                            "group": f"{priority} {mode}",
                            "round": r + 1,
                            "budget": "full",
                        }
                    )
    return _number(specs)


def port_plan(rounds: int = 5) -> list[dict]:
    """Return the workers of J5 (gated, D2): each port cell paired with ours, 5 rounds; ours first
    in odd rounds and last in even rounds."""
    specs = []
    for r in range(rounds):
        for index, cell in enumerate(_rotate(PORT_CELLS, r)):
            candidate, model, ours_size, side, ours_side, dtype, mode, batch_size = cell
            group = f"port-{PORT_CELLS.index(cell) + 1}"
            config = BenchConfig(RUN, ours_size, mode, ours_side, batch_size, True)
            ours = {
                "kind": "ours",
                "job": "J5",
                "config": asdict(config),
                "pixels": "random" if ours_side == 592 else "stored",
                "group": group,
                "batch_size": batch_size,
                "budget": "full",
            }
            port = {
                "kind": "port",
                "job": "J5",
                "candidate": candidate,
                "model": model,
                "side": side,
                "dtype": dtype,
                "batch_size": batch_size,
                "pixels": "random" if model == "dinov2" else "stored",
                "group": group,
                "budget": "full",
            }
            pair = [ours, port] if r % 2 == 0 else [port, ours]
            specs += [spec | {"round": r + 1} for spec in pair]
    return _number(specs)


# The rows (Section 3.7).


@dataclass(frozen=True)
class PriorArtRow:
    """One cell of the Core ML pairs (D6), or one change ratio of a port-change variant.

    Attributes:
        ratios: The paired ratios, one for each round with both lines done.
        median: The lower median of the ratios (ADR 0004).
        spread: (largest − smallest) ÷ smallest.
        verdict: single_verdict of the median.
        n: The number of ratios.
        ours_ips: The lower median of the throughput of ours (patch output).
        other_ips: The lower median of the throughput of the other side.
        fields: The fields of the cell: export, model size, input resolution, and so on.
    """

    ratios: list[float]
    median: float
    spread: float
    verdict: str
    n: int
    ours_ips: float
    other_ips: float
    fields: dict = field(default_factory=dict)


def _done(line: dict) -> bool:
    return line.get("status") == "done" and isinstance(line.get("patch"), dict)


def cell_row(ours: list[dict], other: list[dict], fields: dict | None = None) -> PriorArtRow | None:
    """Return the row of one cell: the ratio ours ÷ other of each round with both lines done.

    Args:
        ours: The lines of ours of the cell, one for each round.
        other: The lines of the export (or of the variant) of the cell.
        fields: The fields of the cell for the table.

    Returns:
        The row, or None if no round has both lines done.
    """
    by_round = {line["round"]: line for line in ours if _done(line)}
    pairs = [
        (by_round[line["round"]], line)
        for line in other
        if _done(line) and line["round"] in by_round
    ]
    if not pairs:
        return None
    ratios = [a["patch"]["throughput_ips"] / b["patch"]["throughput_ips"] for a, b in pairs]
    median = _lower_median(ratios)
    return PriorArtRow(
        ratios=ratios,
        median=median,
        spread=(max(ratios) - min(ratios)) / min(ratios),
        verdict=single_verdict(median),
        n=len(ratios),
        ours_ips=_lower_median([a["patch"]["throughput_ips"] for a, _ in pairs]),
        other_ips=_lower_median([b["patch"]["throughput_ips"] for _, b in pairs]),
        fields=fields or {},
    )


def prior_art_rows(lines: list[dict]) -> list[PriorArtRow | dict]:
    """Return one row for each export and batch size of J1, in the order of EXPORTS.

    A cell without a done pair gives a dict with the export, the batch size and the reasons.
    """
    j1 = [line for line in lines if line.get("job") == "J1"]
    rows: list[PriorArtRow | dict] = []
    for name, export in EXPORTS.items():
        group = GROUP_OF_EXPORT[name]
        for b in BATCH_SIZES:
            ours = [
                line
                for line in j1
                if line["side_kind"] == "ours"
                and line["group"] == group
                and line["batch_size"] == b
            ]
            other = [
                line
                for line in j1
                if line["side_kind"] == "export"
                and line["export"] == name
                and line["batch_size"] == b
            ]
            parities = [line["parity"]["worst"] for line in other if line.get("parity")]
            fields = {
                "export": name,
                "candidate": export.candidate,
                "model_size": export.model_size,
                "side": export.side,
                "precision": export.precision,
                "batch_size": b,
                "dtype_mode": export.dtype_mode,
                "ours_side": export.ours_side,
                "parity": worst_parity(parities) if parities else None,
                "valid": all(line.get("valid") for line in other if _done(line)),
                "report_only": export.report_only,
            }
            row = cell_row(ours, other, fields)
            if row is None:
                reasons = sorted({line.get("reason") or line["status"] for line in other + ours})
                rows.append(fields | {"reason": "; ".join(reasons) or "no line"})
            else:
                rows.append(row)
    return rows


def variant_parity_rows(lines: list[dict]) -> list[dict]:
    """Return the parity rows of J3: one for each model size, dtype mode and variant, with the
    worst values over the 12 configurations and the number that pass."""
    rows = []
    for line in lines:
        if line.get("job") != "J3":
            continue
        row = {key: line.get(key) for key in ("model_size", "dtype_mode", "variant", "status")}
        configurations = line.get("configurations") or []
        if configurations:
            row["worst"] = worst_parity([c["parity"] for c in configurations])
            row["passes"] = sum(c["passes"] for c in configurations)
            row["finite"] = all(c["finite"] for c in configurations)
            row["n"] = len(configurations)
        row["reason"] = line.get("reason", "")
        rows.append(row)
    return rows


def variant_speed_rows(lines: list[dict]) -> list[PriorArtRow]:
    """Return the change ratios of J4: V1 ÷ V0 and V4 ÷ V0 at P1 and P2 in fp32 and bf16-cast, the
    lower median of the rounds (the ratio of a change: above 1 means the change is faster)."""
    j4 = [line for line in lines if line.get("job") == "J4"]
    rows = []
    for priority in SPEED_POINTS:
        for mode in VARIANT_MODES:
            cell = [
                line
                for line in j4
                if line.get("priority") == priority and line.get("dtype_mode") == mode
            ]
            base = [line for line in cell if line["variant"] == "V0"]
            for variant in SPEED_VARIANTS[1:]:
                changed = [line for line in cell if line["variant"] == variant]
                fields = {"priority": priority, "dtype_mode": mode, "variant": variant}
                row = cell_row(changed, base, fields)
                if row is not None:
                    rows.append(row)
    return rows


def port_rows(lines: list[dict]) -> list[PriorArtRow | dict]:
    """Return one row for each port cell of J5, in the order of PORT_CELLS.

    The last line of each round and side counts, so a later run of J5 replaces an earlier run in
    the same file. A cell without run lines gives a dict with the status "blocked" and the reason
    and the command of its "blocked" line; a cell without a done pair gives a dict with the
    status "error" and the reasons.
    """
    j5 = [line for line in lines if line.get("job") == "J5"]
    rows: list[PriorArtRow | dict] = []
    for number, cell in enumerate(PORT_CELLS, start=1):
        candidate, model, _, side, ours_side, dtype, mode, batch_size = cell
        group = f"port-{number}"
        fields = {
            "group": group,
            "candidate": candidate,
            "model_size": model,
            "side": side,
            "ours_side": ours_side,
            "precision": dtype,
            "dtype_mode": mode,
            "batch_size": batch_size,
        }
        latest: dict[tuple[int, str], dict] = {}
        blocked = None
        for line in j5:
            if line.get("group") != group:
                continue
            if line["status"] == "blocked":
                blocked = line
            else:
                latest[(line["round"], line["side_kind"])] = line
        if not latest:
            reason = blocked["reason"] if blocked else "no line"
            command = blocked.get("command") if blocked else None
            rows.append(fields | {"status": "blocked", "reason": reason, "command": command})
            continue
        ours = [line for (_, kind), line in latest.items() if kind == "ours"]
        port = [line for (_, kind), line in latest.items() if kind == "port"]
        done = [line for line in port if _done(line)]
        parities = [line["parity"]["worst"] for line in done if line.get("parity")]
        fields |= {
            "parity": worst_parity(parities) if parities else None,
            "valid_lines": sum(bool(line.get("valid")) for line in done),
            "done_lines": len(done),
            "own_values": max((len(line.get("own_values") or []) for line in done), default=0),
        }
        row = cell_row(ours, port, fields)
        if row is None:
            failed = [line for line in latest.values() if line["status"] != "done"]
            reasons = sorted({line.get("reason") or line["status"] for line in failed})
            rows.append(fields | {"status": "error", "reason": "; ".join(reasons) or "no pair"})
        else:
            rows.append(row)
    return rows


def slow_cell_facts(lines: list[dict], limit: float = 0.97) -> list[dict]:
    """Return the measured facts of each J1 cell with a prior-art ratio below limit (AC4).

    The facts: the transfer share of ours (1 − patch throughput ÷ device throughput: the time of the
    move of the outputs to the host) and the ratio of our device throughput to the export
    throughput (the lower median over the rounds), with the token count of the export.
    """
    j1 = [line for line in lines if line.get("job") == "J1"]
    facts = []
    for row in prior_art_rows(lines):
        if not isinstance(row, PriorArtRow) or row.median >= limit:
            continue
        name, batch_size = row.fields["export"], row.fields["batch_size"]
        ours = [
            line
            for line in j1
            if line["side_kind"] == "ours"
            and line["group"] == GROUP_OF_EXPORT[name]
            and line["batch_size"] == batch_size
            and _done(line)
            and isinstance(line.get("device"), dict)
        ]
        exports = {
            line["round"]: line
            for line in j1
            if line["side_kind"] == "export"
            and line["export"] == name
            and line["batch_size"] == batch_size
            and _done(line)
        }
        device = _lower_median([line["device"]["throughput_ips"] for line in ours])
        patch = _lower_median([line["patch"]["throughput_ips"] for line in ours])
        device_ratios = [
            line["device"]["throughput_ips"] / exports[line["round"]]["patch"]["throughput_ips"]
            for line in ours
            if line["round"] in exports
        ]
        facts.append(
            {
                "export": name,
                "batch_size": batch_size,
                "ratio": row.median,
                "transfer_share": 1 - patch / device,
                "device_ratio": _lower_median(device_ratios),
                "tokens": EXPORTS[name].tokens,
            }
        )
    return facts


def between_runs(run1: list[dict], run2: list[dict]) -> list[dict]:
    """Return the cells of two runs of J1 (finding V1 of specs/06b-prior-art/verify.html).

    For each export and batch size with a row in both runs: the lower median of each run, the
    pooled lower median of all paired ratios, the spread between the run medians
    ((largest − smallest) ÷ smallest) and pass_verdict, as the P1 and P2 rows of Step 6, with
    the mark "inside the noise" within 1 ± NOISE.
    """
    second = {
        (row.fields["export"], row.fields["batch_size"]): row
        for row in prior_art_rows(run2)
        if isinstance(row, PriorArtRow)
    }
    cells = []
    for row in prior_art_rows(run1):
        if not isinstance(row, PriorArtRow):
            continue
        key = (row.fields["export"], row.fields["batch_size"])
        if key not in second:
            continue
        medians = [row.median, second[key].median]
        pooled = _lower_median(row.ratios + second[key].ratios)
        verdict = pass_verdict(medians, pooled)
        if abs(pooled - 1) <= NOISE:
            verdict += ", inside the noise"
        cells.append(
            {
                "export": key[0],
                "batch_size": key[1],
                "run_medians": medians,
                "pooled": pooled,
                "spread": (max(medians) - min(medians)) / min(medians),
                "verdict": verdict,
                "n": len(row.ratios) + len(second[key].ratios),
            }
        )
    return cells
