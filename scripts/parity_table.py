"""Turn the parity results of the parity tests of one backend into results/parity-{backend}.md.

The tests append one JSON line for each configuration to
tests/fixtures/generated/parity/{backend}.jsonl. For each configuration the last line counts.

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
from dinov3_apple.parity import THRESHOLDS

REPO_ROOT = Path(__file__).parents[1]
PARITY_DIR = REPO_ROOT / "tests" / "fixtures" / "generated" / "parity"
MEASURES = ["frame", "patch_mean", "patch_p01", "patch_min"]
# The package whose version the lines of each backend carry.
BACKEND_PACKAGES = {"torch-mps": "torch", "mlx": "mlx"}
# The checks of the mlx tests against the stored hidden states (W1 to W4 of the Step 4 plan).
STORED_KINDS = ["stored-embeddings", "stored-block", "stored-final-norm", "hidden-state"]


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
            measured = [line["parity"] for line in group if line["status"] == "measured"]
            non_finite = sum(line["status"] == "non-finite" for line in group)
            skipped = [line for line in group if line["status"] == "skipped"]
            if skipped and not measured and not non_finite:
                values = ["–"] * 4
                result = f"skipped: {skipped[0]['reason']}"
            elif not measured:
                values = ["–"] * 4
                result = "expected failure (fp16 overflow)" if dtype_mode == "fp16-cast" else "fail"
            else:
                minimum = {m: min(p[m] for p in measured) for m in MEASURES}
                values = [f"{minimum[m]:.6f}" for m in MEASURES]
                result = "pass" if _passes(minimum, dtype_mode) else "fail"
                if non_finite:
                    result += f", {non_finite} expected failures (fp16 overflow)"
            rows.append(
                f"| {model_size} | {dtype_mode} | {len(measured)} | {non_finite} | "
                + " | ".join(values)
                + f" | {result} |"
            )
    return rows


def stored_rows(lines: list[dict]) -> list[str]:
    """Return one Markdown table row for each check against the stored hidden states."""
    rows = []
    for kind in STORED_KINDS:
        for line in lines:
            if line["kind"] != kind or line["status"] != "measured" or not line.get("values"):
                continue
            values = line["values"]
            rows.append(
                f"| {kind} | {line['config']} | {1 - values['min_cosine']:.2e} | "
                f"{values['max_abs']:.4g} | {values['max_rel']:.4g} |"
            )
    return rows


def write_table(lines: list[dict], backend: str = "torch-mps") -> str:
    """Return the Markdown text of the results file of one backend.

    Args:
        lines: The latest JSON lines of the parity tests of the backend.
        backend: "torch-mps" or "mlx".

    Returns:
        The text with the parity, the batch consistency, the thresholds and, for lines of the
        stored hidden-state checks, a fourth section.
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
    if stored := stored_rows(lines):
        text += [
            "## 4 Stored hidden states",
            "",
            (
                "Each check runs one part on the stored input of iss-kibo-lab/224x224 and compares "
                "with the stored output: the embeddings, each block alone, the final layer norm "
                "and the 13 hidden states in a chain. The minimum token cosine runs in fp64."
            ),
            "",
            (
                "| Check | Part | 1 − minimum token cosine | Largest difference | "
                "Largest relative difference |"
            ),
            "| --- | --- | --- | --- | --- |",
            *stored,
            "",
            f"Table 4. Checks of {backend} against the stored hidden states.",
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
    results.write_text(write_table(latest_lines(parity_log(args.backend)), args.backend))
    print(f"wrote {results}")


if __name__ == "__main__":
    main()
