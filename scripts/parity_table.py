"""Turn the parity results of the parity tests of one backend into results/parity-{backend}.md.

The tests append one JSON line for each configuration to
tests/fixtures/generated/parity/{backend}.jsonl. For each configuration the last line counts. The
mlx table also lists the cross-check lines of scripts/cross_check.py and compares the non-finite
configurations of mlx with those of torch-mps.

Usage:
    uv run python scripts/parity_table.py                   # torch-mps
    uv run python scripts/parity_table.py --backend mlx     # mlx
    uv run python scripts/parity_table.py --backend mlx --log <file>.jsonl --out <folder>
"""

import argparse
import json
from collections import defaultdict
from datetime import date
from pathlib import Path

from dinov3_mlx.hub import MODEL_IDS
from dinov3_mlx.parity import EXPECTED_FP16_OVERFLOW, THRESHOLDS

REPO_ROOT = Path(__file__).parents[1]
PARITY_DIR = REPO_ROOT / "tests" / "fixtures" / "generated" / "parity"
MEASURES = ["frame", "patch_mean", "patch_p01", "patch_min"]
# The package whose version the lines of each backend carry.
BACKEND_PACKAGES = {"torch-mps": "torch", "mlx": "mlx"}
# The checks of the mlx tests against the stored hidden states (W1 to W4 of the Step 5 plan).
STORED_KINDS = ["stored-embeddings", "stored-block", "stored-final-norm", "hidden-state"]
# The kinds of test whose non-finite configurations the two backends must share (AC8).
NON_FINITE_KINDS = ["parity", "batch"]


def parity_log(backend: str) -> Path:
    """Return the JSON lines file of the parity tests of one backend."""
    return PARITY_DIR / f"{backend}.jsonl"


def results_path(backend: str) -> Path:
    """Return the Markdown results file of one backend."""
    return REPO_ROOT / "results" / f"parity-{backend}.md"


def latest_lines(path: Path) -> list[dict]:
    """Return the last line of each (kind, model size, dtype mode, configuration, bits).

    The lines of Steps 4 and 5 have no field bits: they read as None. So the 8-bit line and the
    4-bit line of one configuration both stay (T14 of specs/06-speed/plan.html).
    """
    latest = {}
    for text in path.read_text().splitlines():
        line = json.loads(text)
        key = (line["kind"], line["model_size"], line["dtype_mode"], line["config"])
        latest[(*key, line.get("bits"))] = line
    return list(latest.values())


def _passes(minimum: dict[str, float], dtype_mode: str) -> bool:
    threshold = THRESHOLDS[dtype_mode]
    return all(minimum[m] >= getattr(threshold, m) for m in ("frame", "patch_mean", "patch_p01"))


def summary_rows(lines: list[dict], kind: str) -> list[str]:
    """Return one Markdown table row for each model size and dtype mode of one kind of test."""
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for line in lines:
        if line["kind"] == kind:
            groups[(line["model_size"], line["dtype_mode"])].append(line)
    rows = []
    for model_size in MODEL_IDS:
        for dtype_mode in THRESHOLDS:
            group = groups.get((model_size, dtype_mode))
            if not group:
                continue
            expected = (model_size, dtype_mode) in EXPECTED_FP16_OVERFLOW
            measured = [line["parity"] for line in group if line["status"] == "measured"]
            non_finite = sum(line["status"] == "non-finite" for line in group)
            skipped = [line for line in group if line["status"] == "skipped"]
            if skipped and not measured and not non_finite:
                values = ["–"] * 4
                result = f"skipped: {skipped[0]['reason']}"
            elif not measured:
                values = ["–"] * 4
                result = "expected failure (fp16 overflow)" if expected else "fail"
            else:
                minimum = {m: min(p[m] for p in measured) for m in MEASURES}
                values = [f"{minimum[m]:.6f}" for m in MEASURES]
                result = "pass" if _passes(minimum, dtype_mode) else "fail"
                if non_finite:
                    if expected:
                        result += f", {non_finite} expected failures (fp16 overflow)"
                    else:
                        result += f", {non_finite} non-finite: fail"
            rows.append(
                f"| {model_size} | {dtype_mode} | {len(measured)} | {non_finite} | "
                + " | ".join(values)
                + f" | {result} |"
            )
    return rows


def _groups(lines: list[dict], kind: str, *fields: str) -> dict[tuple, list[dict]]:
    """Group the lines of one kind by model size, dtype mode and the given fields, in the table
    order."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for line in lines:
        if line["kind"] == kind:
            key = (line["model_size"], line["dtype_mode"], *(line.get(f) for f in fields))
            groups[key].append(line)
    return dict(sorted(groups.items(), key=lambda item: (*table_order(*item[0][:2]), item[0][2:])))


def compiled_equal_rows(lines: list[dict]) -> list[str]:
    """Return one row for each model size and dtype mode of the kind "compiled-equal" (W1 of
    specs/06-speed/plan.html): the configurations, the bit-equal ones and the largest difference.
    A non-finite configuration counts as equal when both forms are non-finite at the same
    positions."""
    rows = []
    for (model_size, dtype_mode), group in _groups(lines, "compiled-equal").items():
        measured = [line for line in group if line["status"] in ("measured", "non-finite")]
        equal = sum(line["values"]["max_abs"] == 0.0 for line in measured)
        largest = max((line["values"]["max_abs"] for line in measured), default=None)
        rows.append(
            f"| {model_size} | {dtype_mode} | {len(measured)} | {equal} | "
            f"{'–' if largest is None else f'{largest:.3g}'} |"
        )
    return rows


def quantized_rows(lines: list[dict]) -> list[str]:
    """Return one row for each model size, dtype mode and bits of the kind "quantized" (W2 of
    specs/06-speed/plan.html): the configurations, the ones that pass THRESHOLDS[mode], and the
    minimum parity against the reference output and against the unquantized output."""
    rows = []
    for (model_size, dtype_mode, bits), group in _groups(lines, "quantized", "bits").items():
        measured = [line for line in group if line["status"] == "measured"]
        passes = sum(bool(line["values"]["passes_thresholds"]) for line in measured)
        if measured:
            ref = [min(line["parity"][m] for line in measured) for m in MEASURES[:3]]
            unq = [min(line["values"]["unquantized"][m] for line in measured) for m in MEASURES[:3]]
            values = " | ".join(f"{v:.6f}" for v in ref + unq)
        else:
            skipped = [line["reason"] for line in group if line["status"] == "skipped"]
            values = " | ".join(["–"] * 6) + (f" | skipped: {skipped[0]}" if skipped else "")
        rows.append(
            f"| {model_size} | {dtype_mode} | {bits} | {len(measured)} | {passes} | {values} |"
        )
    return rows


def fp16_output_rows(lines: list[dict]) -> list[str]:
    """Return one row for each model size and dtype mode of the kind "fp16-output" (W4 of
    specs/06-speed/plan.html): the minimum parity with the fp16 patch embeddings and the largest
    extra loss against the fp32 patch embeddings."""
    rows = []
    for (model_size, dtype_mode), group in _groups(lines, "fp16-output").items():
        measured = [line for line in group if line["status"] == "measured"]
        if not measured:
            rows.append(f"| {model_size} | {dtype_mode} | 0 | – | – | – | – |")
            continue
        patch_mean = min(line["parity"]["patch_mean"] for line in measured)
        patch_p01 = min(line["parity"]["patch_p01"] for line in measured)
        extra = max(max(line["values"]["extra_loss"].values()) for line in measured)
        frame = all(line["values"]["frame_equal"] for line in measured)
        rows.append(
            f"| {model_size} | {dtype_mode} | {len(measured)} | {patch_mean:.6f} | "
            f"{patch_p01:.6f} | {extra:.2g} | {'yes' if frame else 'no'} |"
        )
    return rows


def table_order(model_size: str, dtype_mode: str) -> tuple[int, int]:
    """Return the sort key of the rows of the tables: the position of the model size in MODEL_IDS
    and of the dtype mode in THRESHOLDS. scripts/token_range_table.py uses it too."""
    return list(MODEL_IDS).index(model_size), list(THRESHOLDS).index(dtype_mode)


def stored_rows(lines: list[dict]) -> list[str]:
    """Return one Markdown table row for each check against the stored hidden states.

    The rows come in the order of MODEL_IDS, then THRESHOLDS, then STORED_KINDS.
    """
    rows = []
    selected = [
        line
        for line in lines
        if line["kind"] in STORED_KINDS and line["status"] == "measured" and line.get("values")
    ]
    selected.sort(
        key=lambda line: (
            *table_order(line["model_size"], line["dtype_mode"]),
            STORED_KINDS.index(line["kind"]),
        )
    )
    for line in selected:
        values = line["values"]
        rows.append(
            f"| {line['model_size']} | {line['dtype_mode']} | {line['kind']} | {line['config']} | "
            f"{1 - values['min_cosine']:.2e} | {values['max_abs']:.4g} | {values['max_rel']:.4g} |"
        )
    return rows


def cross_check_rows(lines: list[dict]) -> list[str]:
    """Return one Markdown table row for each cross-check line (scripts/cross_check.py)."""
    selected = sorted(
        (line for line in lines if line["kind"] == "cross-check"),
        key=lambda line: (*table_order(line["model_size"], line["dtype_mode"]), line["config"]),
    )
    rows = []
    for line in selected:
        torch_device = (line.get("values") or {}).get("torch_device", "–")
        if line["status"] == "measured":
            values = [f"{line['parity'][m]:.6f}" for m in MEASURES]
        else:
            values = ["–"] * 4
        rows.append(
            f"| {line['model_size']} | {line['dtype_mode']} | {line['config']} | "
            f"torch-{torch_device} | {line['status']} | " + " | ".join(values) + " |"
        )
    return rows


def non_finite_keys(lines: list[dict]) -> set[tuple[str, str, str, str]]:
    """Return (kind, model size, dtype mode, configuration) of each non-finite parity or batch
    line."""
    return {
        (line["kind"], line["model_size"], line["dtype_mode"], line["config"])
        for line in lines
        if line["kind"] in NON_FINITE_KINDS and line["status"] == "non-finite"
    }


def non_finite_section(mlx_lines: list[dict], torch_lines: list[dict]) -> tuple[list[str], str]:
    """Return the rows of the non-finite configurations of both backends and one last line:
    "equal", or the list of differences."""
    mlx, torch_mps = non_finite_keys(mlx_lines), non_finite_keys(torch_lines)
    keys = sorted(
        mlx | torch_mps,
        key=lambda key: (NON_FINITE_KINDS.index(key[0]), *table_order(key[1], key[2]), key[3]),
    )
    rows, differences = [], []
    for key in keys:
        kind, model_size, dtype_mode, config = key
        on_mlx = "non-finite" if key in mlx else "–"
        on_torch = "non-finite" if key in torch_mps else "–"
        rows.append(f"| {kind} | {model_size} | {dtype_mode} | {config} | {on_mlx} | {on_torch} |")
        if (key in mlx) != (key in torch_mps):
            only = "mlx" if key in mlx else "torch-mps"
            differences.append(f"{model_size} {dtype_mode} {config} ({kind}): {only} only")
    last = "equal" if not differences else "differences: " + "; ".join(differences)
    return rows, last


def write_table(
    lines: list[dict], backend: str = "torch-mps", torch_lines: list[dict] | None = None
) -> str:
    """Return the Markdown text of the results file of one backend.

    Args:
        lines: The latest JSON lines of the parity tests of the backend.
        backend: "torch-mps" or "mlx".
        torch_lines: The latest JSON lines of torch-mps, for the comparison of the non-finite
            configurations of mlx with those of torch-mps. None: no comparison.

    Returns:
        The text with the parity, the batch consistency and the thresholds. Further sections
        follow for the stored hidden-state checks, the cross-check lines and the comparison of
        the non-finite configurations, if there are lines for them.
    """
    package = BACKEND_PACKAGES[backend]
    versions = sorted({line[package] for line in lines if package in line})
    header = (
        "| Model size | Dtype mode | Measured | Non-finite | Frame | Patch mean | "
        "Patch 1st percentile | Patch minimum | Result |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    )
    thresholds = "\n".join(
        f"| {mode} | {t.frame} | {t.patch_mean} | {t.patch_p01} |" for mode, t in THRESHOLDS.items()
    )
    text = [
        f"# Parity of {backend}",
        "",
        (
            f"Made by `scripts/parity_table.py` on {date.today().isoformat()} from the parity "
            f"tests ({package} {', '.join(versions)}). The reference output is torch-cpu in "
            "fp32. Each value is the minimum over the configurations of the row. The patch "
            "minimum is report only. Section 3 gives the thresholds."
        ),
        "",
        "## 1 Parity against the reference output",
        "",
        "Each row covers 12 configurations: 3 test images × 4 input resolutions.",
        "",
        header,
        *summary_rows(lines, "parity"),
        "",
        f"Table 1. Parity of {backend} against the reference output.",
        "",
        "## 2 Batch consistency",
        "",
        (
            "A batch of 8 at 224 × 224 px against 8 runs with batch size 1, on "
            f"{backend} in the same dtype mode."
        ),
        "",
        header,
        *summary_rows(lines, "batch"),
        "",
        "Table 2. Batch consistency.",
        "",
        "## 3 Thresholds",
        "",
        "| Dtype mode | Frame | Patch mean | Patch 1st percentile |",
        "| --- | --- | --- | --- |",
        thresholds,
        "",
        (
            "Table 3. Thresholds of `src/dinov3_mlx/parity.py`. On 2026-11-06 the user gave "
            "bf16 its own row: bf16 activations give lower parity on torch-mps and on "
            "torch-cpu alike."
        ),
        "",
    ]
    section = 4
    if stored := stored_rows(lines):
        text += [
            f"## {section} Stored hidden states",
            "",
            (
                "Each check runs one part on the stored input of iss-kibo-lab/224x224 and compares "
                "with the stored output: the embeddings, each block alone, the final layer norm "
                "and the L + 1 hidden states in a chain (L is the number of blocks). In the cast "
                "modes the stored inputs are first rounded to the stream dtype. The fp32 rows "
                "assert; the four fp16 and bf16 dtype modes are report only. The minimum token "
                "cosine runs in fp64."
            ),
            "",
            (
                "| Model size | Dtype mode | Check | Part | 1 − minimum token cosine | "
                "Largest difference | Largest relative difference |"
            ),
            "| --- | --- | --- | --- | --- | --- | --- |",
            *stored,
            "",
            f"Table {section}. Checks of {backend} against the stored hidden states.",
            "",
        ]
        section += 1
    if cross := cross_check_rows(lines):
        text += [
            f"## {section} Cross-check against torch",
            "",
            (
                "scripts/cross_check.py compares the mlx output with the torch output in the same "
                "dtype mode, without the reference output. Report only: no threshold applies."
            ),
            "",
            (
                "| Model size | Dtype mode | Configuration | Torch backend | Status | Frame | "
                "Patch mean | Patch 1st percentile | Patch minimum |"
            ),
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            *cross,
            "",
            f"Table {section}. Cross-check of {backend} against torch.",
            "",
        ]
        section += 1
    if equal := compiled_equal_rows(lines):
        text += [
            f"## {section} Compiled forward pass against the eager forward pass",
            "",
            (
                "Each configuration runs the compiled forward pass and the eager forward pass of "
                "one backbone (W1 of specs/06-speed/plan.html). A non-finite configuration counts "
                "as equal when both forms are non-finite at the same positions."
            ),
            "",
            "| Model size | Dtype mode | Configurations | Bit for bit equal | Largest difference |",
            "| --- | --- | --- | --- | --- |",
            *equal,
            "",
            f"Table {section}. The compiled forward pass against the eager forward pass.",
            "",
        ]
        section += 1
    if quantized := quantized_rows(lines):
        text += [
            f"## {section} Quantized weights",
            "",
            (
                "Report only, with a fault bound in the tests (ADR 0011). Each row has the "
                "minimum over its configurations, against the reference output and against the "
                "unquantized output of the same dtype mode. The column Pass counts the "
                "configurations that pass the thresholds of the dtype mode (Section 3)."
            ),
            "",
            (
                "| Model size | Dtype mode | Bits | Measured | Pass | Frame | Patch mean | "
                "Patch 1st percentile | Frame (unquantized) | Patch mean (unquantized) | "
                "Patch 1st percentile (unquantized) |"
            ),
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            *quantized,
            "",
            f"Table {section}. Parity of the quantized weights.",
            "",
        ]
        section += 1
    if fp16 := fp16_output_rows(lines):
        text += [
            f"## {section} fp16 patch embeddings",
            "",
            (
                "The patch embeddings of mlx_embeddings in fp16 (D13 of specs/06-speed/plan.html) "
                "against the reference output. The extra loss is the fp32-output parity minus the "
                "fp16-output parity of the patch mean and the patch 1st percentile."
            ),
            "",
            (
                "| Model size | Dtype mode | Measured | Patch mean | Patch 1st percentile | "
                "Largest extra loss | Frame unchanged |"
            ),
            "| --- | --- | --- | --- | --- | --- | --- |",
            *fp16,
            "",
            f"Table {section}. Parity with the fp16 patch embeddings.",
            "",
        ]
        section += 1
    if torch_lines is not None:
        rows, last = non_finite_section(lines, torch_lines)
        text += [
            f"## {section} Non-finite configurations: mlx against torch-mps",
            "",
            (
                "Each row is a parity or batch configuration with a non-finite output on at least "
                "one backend. The last line says if the two sets are equal."
            ),
            "",
            "| Test | Model size | Dtype mode | Configuration | mlx | torch-mps |",
            "| --- | --- | --- | --- | --- | --- |",
            *rows,
            "",
            f"Table {section}. Non-finite configurations of mlx and torch-mps.",
            "",
            last,
            "",
        ]
    return "\n".join(text)


def main(argv: list[str] | None = None) -> None:
    """Write results/parity-{backend}.md, or parity-{backend}.md in the folder of --out.

    Args:
        argv: The arguments. None reads sys.argv.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=list(BACKEND_PACKAGES), default="torch-mps")
    parser.add_argument("--log", type=Path, help="the JSON lines file of the backend")
    parser.add_argument("--out", type=Path, help="the folder of the Markdown file")
    args = parser.parse_args(argv)
    results = results_path(args.backend)
    if args.out is not None:
        results = args.out / results.name
    results.parent.mkdir(parents=True, exist_ok=True)
    torch_lines = None
    if args.backend == "mlx" and parity_log("torch-mps").exists():
        torch_lines = latest_lines(parity_log("torch-mps"))
    lines = latest_lines(args.log or parity_log(args.backend))
    results.write_text(write_table(lines, args.backend, torch_lines))
    print(f"wrote {results}")


if __name__ == "__main__":
    main()
