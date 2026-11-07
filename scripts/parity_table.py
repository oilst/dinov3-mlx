"""Turn the parity results of the parity tests of one backend into results/parity-{backend}.md.

The tests append one JSON line for each configuration to
tests/fixtures/generated/parity/{backend}.jsonl. For each configuration the last line counts. The
mlx table also lists the cross-check lines of scripts/cross_check.py and compares the non-finite
configurations of mlx with those of torch-mps.

Usage:
    uv run python scripts/parity_table.py                   # torch-mps
    uv run python scripts/parity_table.py --backend mlx     # mlx
"""

import argparse
import json
from collections import defaultdict
from datetime import date
from pathlib import Path

from dinov3_apple.hub import MODEL_IDS
from dinov3_apple.parity import EXPECTED_FP16_OVERFLOW, THRESHOLDS

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
    """Return the last line of each (kind, model size, dtype mode, configuration)."""
    latest = {}
    for text in path.read_text().splitlines():
        line = json.loads(text)
        latest[(line["kind"], line["model_size"], line["dtype_mode"], line["config"])] = line
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


def _order(line: dict) -> tuple[int, int]:
    """The position of the model size and the dtype mode of one line in MODEL_IDS and THRESHOLDS."""
    return list(MODEL_IDS).index(line["model_size"]), list(THRESHOLDS).index(line["dtype_mode"])


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
    selected.sort(key=lambda line: (*_order(line), STORED_KINDS.index(line["kind"])))
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
        key=lambda line: (*_order(line), line["config"]),
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
        key=lambda key: (
            NON_FINITE_KINDS.index(key[0]),
            list(MODEL_IDS).index(key[1]),
            list(THRESHOLDS).index(key[2]),
            key[3],
        ),
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
            "Table 3. Thresholds of `src/dinov3_apple/parity.py`. On 2026-11-06 the user gave "
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


def main() -> None:
    """Write results/parity-{backend}.md."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=list(BACKEND_PACKAGES), default="torch-mps")
    args = parser.parse_args()
    results = results_path(args.backend)
    results.parent.mkdir(parents=True, exist_ok=True)
    torch_lines = None
    if args.backend == "mlx" and parity_log("torch-mps").exists():
        torch_lines = latest_lines(parity_log("torch-mps"))
    lines = latest_lines(parity_log(args.backend))
    results.write_text(write_table(lines, args.backend, torch_lines))
    print(f"wrote {results}")


if __name__ == "__main__":
    main()
