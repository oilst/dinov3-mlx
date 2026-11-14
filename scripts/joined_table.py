"""Write results/joined-table.md from results/speed-joined.jsonl: the paired ratio of each joined
form to R for each configuration of B1, the first-batch check, the MLX peak of B2 and the default
rule D6 (Section 7 of specs/06e-kernels/plan.html).

The ratio of a configuration is the ratio of the output of its priority: the frame embedding at
P1, the patch embedding at P2 (PRIORITY of bench.py). The pooled median is the median of the
paired ratios of both runs (ADR 0004).

    uv run python scripts/joined_table.py
"""

import argparse
import json
from pathlib import Path

import numpy as np

from dinov3_mlx.bench import PRIORITY

REPO_ROOT = Path(__file__).parents[1]
RESULTS = REPO_ROOT / "results" / "speed-joined.jsonl"
TABLE = REPO_ROOT / "results" / "joined-table.md"
JOINED = ["R-J", "R-Jqkv", "R-Jff"]
GAIN_LIMIT = 1.03  # D6: the pooled median at P2
PEAK_LIMIT_BYTES = 0.10e9  # AC7: R-J at most R + 0.10 GB


def default_rule(run_medians: list[float], pooled: float) -> bool:
    """D6: a join is on if the pooled median of two runs is at least 1.03 and the median of each
    run is above 1."""
    return len(run_medians) == 2 and pooled >= GAIN_LIMIT and all(m > 1 for m in run_medians)


def _cell(line: dict) -> tuple:
    return (
        line["priority"],
        line["model_size"],
        line["side"],
        line["batch_size"],
        line["dtype_mode"],
    )


def joined_rows(lines: list[dict]) -> list[dict]:
    """Return one row for each configuration and joined form: the median of each run, the pooled
    median, the extremes, the first-batch check, the peak difference and the D6 decision. The
    rule applies at P2 with batch size 8 only."""
    cells: dict[tuple, dict[str, dict]] = {}
    for line in lines:
        if line.get("change") == "C7" and line["status"] == "done":
            cells.setdefault(_cell(line), {})[str(line["run_id"])] = line
    rows = []
    for cell, runs in cells.items():
        priority, _, _, batch_size, _ = cell
        output = PRIORITY[priority][3]
        for form in JOINED:
            per_run = {run: line["form_ratios"][form][output] for run, line in sorted(runs.items())}
            pooled_values = [v for values in per_run.values() for v in values]
            pooled = float(np.median(pooled_values))
            medians = [float(np.median(values)) for values in per_run.values()]
            checks = [line["first_batch"][form] for line in runs.values()]
            equal = all(c["frame_equal"] and c["patches_equal"] for c in checks)
            difference = max(max(c["frame_difference"], c["patch_difference"]) for c in checks)
            valid = all(line["validity"][form]["valid"] for line in runs.values())
            peaks = [
                line["mlx_timed_peak_bytes"][form] - line["mlx_timed_peak_bytes"]["R"]
                for line in runs.values()
            ]
            rule_applies = priority == "P2" and batch_size == 8
            rows.append(
                {
                    "cell": cell,
                    "form": form,
                    "output": output,
                    "runs": sorted(runs),
                    "run_medians": medians,
                    "pooled": pooled,
                    "min": min(pooled_values),
                    "max": max(pooled_values),
                    "bit_equal": equal,
                    "largest_difference": difference,
                    "valid": valid,
                    "peak_above_r_bytes": max(peaks),
                    "rule_applies": rule_applies,
                    "on": rule_applies and default_rule(medians, pooled),
                }
            )
    return rows


def write_table(lines: list[dict]) -> str:
    """Return the Markdown text of results/joined-table.md."""
    rows = joined_rows(lines)
    out = [
        "# C7: joined projections against separate projections",
        "",
        "Step 6e, B1 and B2 of `specs/06e-kernels/plan.html`. Source: `results/speed-joined.jsonl`.",
        "Each run has 5 paired rounds; the forms R, R-J, R-Jqkv and R-Jff rotate in one process.",
        "Ratio: throughput of the form ÷ throughput of R in the same round (frame embedding at P1,",
        "patch embedding at P2). Pooled: the median of the 10 ratios of both runs (ADR 0004).",
        f"D6 rule (P2, batch size 8 only): on if pooled ≥ {GAIN_LIMIT} and each run median > 1.",
        "Peak: MLX peak of the timed batches of the form − that of R (largest of the runs).",
        "",
        (
            "| Priority | Model size | px | Batch | Dtype mode | Form | Run medians | Pooled | Min "
            "| Max | First batch | Valid | Peak − R (MB) | D6 |"
        ),
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        priority, size, side, batch, mode = row["cell"]
        medians = " / ".join(f"{m:.3f}" for m in row["run_medians"])
        first = "bit equal" if row["bit_equal"] else f"differs, {row['largest_difference']:.2e}"
        d6 = ("on" if row["on"] else "off") if row["rule_applies"] else "–"
        out.append(
            f"| {priority} | {size} | {side} | {batch} | {mode} | {row['form']} | {medians} | "
            f"{row['pooled']:.3f} | {row['min']:.3f} | {row['max']:.3f} | {first} | "
            f"{'yes' if row['valid'] else 'no'} | {row['peak_above_r_bytes'] / 1e6:.1f} | {d6} |"
        )
    return "\n".join(out) + "\n"


def main() -> None:
    """Write the table."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--table", type=Path, default=TABLE)
    args = parser.parse_args()
    lines = [json.loads(t) for t in args.results.read_text().splitlines() if t.strip()]
    args.table.write_text(write_table(lines))


if __name__ == "__main__":
    main()
