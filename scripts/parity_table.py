"""Turn the parity results of the torch-mps tests into results/parity-torch-mps.md.

The tests append one JSON line for each configuration to
tests/fixtures/generated/parity/torch-mps.jsonl. For each configuration the last line counts.
"""

import json
from collections import defaultdict
from datetime import date
from pathlib import Path

from dinov3_apple.hub import MODEL_IDS
from dinov3_apple.parity import THRESHOLDS
from dinov3_apple.reference import DTYPE_MODES

REPO_ROOT = Path(__file__).parents[1]
PARITY_LOG = REPO_ROOT / "tests" / "fixtures" / "generated" / "parity" / "torch-mps.jsonl"
RESULTS = REPO_ROOT / "results" / "parity-torch-mps.md"
MEASURES = ["frame", "patch_mean", "patch_p01", "patch_min"]


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
        for dtype_mode in DTYPE_MODES:
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


def write_table(lines: list[dict]) -> str:
    """Return the Markdown text of the results file."""
    torch_versions = sorted({line["torch"] for line in lines})
    header = (
        "| Model size | Dtype mode | Measured | Non-finite | Frame | Patch mean | "
        "Patch 1st percentile | Patch minimum | Result |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    )
    thresholds = "\n".join(
        f"| {mode} | {t.frame} | {t.patch_mean} | {t.patch_p01} |" for mode, t in THRESHOLDS.items()
    )
    return "\n".join(
        [
            "# Parity of torch-mps",
            "",
            (
                f"Made by `scripts/parity_table.py` on {date.today().isoformat()} from the parity "
                f"tests (torch {', '.join(torch_versions)}). The reference output is torch-cpu in "
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
            "Table 1. Parity of torch-mps against the reference output.",
            "",
            "## 2 Batch consistency",
            "",
            (
                "A batch of 8 at 224 × 224 px against 8 runs with batch size 1, on torch-mps in "
                "the same dtype mode."
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
    )


def main() -> None:
    """Write results/parity-torch-mps.md."""
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(write_table(latest_lines(PARITY_LOG)))
    print(f"wrote {RESULTS}")


if __name__ == "__main__":
    main()
