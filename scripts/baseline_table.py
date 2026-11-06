"""Turn the benchmark JSON lines into results/baseline-torch-mps.md and
results/block-profile-torch-mps.md.

For each configuration the latest line counts. A later "done" line replaces an earlier "skipped"
line; the JSON file keeps both. Section 7 of specs/03-baseline/plan.html lists the sections.

Usage:
    uv run python scripts/baseline_table.py
"""

import json
import statistics
from datetime import date
from pathlib import Path

from dinov3_apple.bench import (
    BATCH_SIZES,
    DTYPE_MODE_NAMES,
    GIANT_MODEL_SIZE,
    MAIN_MODEL_SIZES,
    OUTPUTS,
    PRIORITY,
    SIDES,
    TIMED,
    WARM_UP,
    baseline_rows,
    latest_lines,
)
from dinov3_apple.flops import PART_NAMES

REPO_ROOT = Path(__file__).parents[1]
RESULTS_DIR = REPO_ROOT / "results"
BASELINE_LINES = RESULTS_DIR / "baseline-torch-mps.jsonl"
BASELINE_TABLE = RESULTS_DIR / "baseline-torch-mps.md"
PROFILE_LINES = RESULTS_DIR / "block-profile-torch-mps.jsonl"
PROFILE_TABLE = RESULTS_DIR / "block-profile-torch-mps.md"
GB = 1e9
DTYPES = ["fp32", "fp16", "bf16"]
OUTPUT_NAMES = {"frame": "frame embedding", "patch": "patch embeddings"}
SPREAD_LIMIT = 0.05
COMPILE_GAIN_LIMIT = 0.10
PART_SUM_LIMIT = 0.15
# Section 3.5 of the plan: fp16-cast overflows for ViT-L/16 and for ViT-H+/16 at 224 px.
EXPECTED_INVALID = {("ViT-L/16", "fp16-cast", side) for side in SIDES} | {
    ("ViT-H+/16", "fp16-cast", 224)
}


def read_lines(path: Path) -> list[dict]:
    """Return the JSON lines of a file, or [] if it does not exist."""
    if not path.exists():
        return []
    return [json.loads(text) for text in path.read_text().splitlines() if text.strip()]


def _gb(value: int | None) -> str:
    return "–" if value is None else f"{value / GB:.2f}"


def _ms(value: float | None) -> str:
    return "–" if value is None else f"{1000 * value:.2f}"


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + " --- |" * len(header)]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def _model_sizes(lines: list[dict]) -> list[str]:
    sizes = list(MAIN_MODEL_SIZES)
    if any(line["model_size"] == GIANT_MODEL_SIZE for line in lines):
        sizes.append(GIANT_MODEL_SIZE)
    return sizes


def _baseline_section(lines: list[dict], output: str) -> str:
    rows_by_key = baseline_rows(lines, output)
    rows = []
    for model_size in _model_sizes(lines):
        dtypes = ["fp16", "bf16"] if model_size == GIANT_MODEL_SIZE else DTYPES
        for side in SIDES:
            for batch_size in BATCH_SIZES:
                for dtype in dtypes:
                    row = rows_by_key.get((model_size, side, batch_size, dtype))
                    key = [model_size, side, batch_size, dtype]
                    if row is None or row.line is None:
                        reason = row.reason if row else "no line"
                        rows.append(key + ["–", "–", "–", "–", "–", "–", f"empty: {reason}"])
                        continue
                    line, result = row.line, row.line[output]
                    rows.append(
                        key
                        + [
                            line["dtype_mode"] + (" + compile" if line["compile"] else ""),
                            str(row.runs),
                            f"{result['throughput_ips']:.1f}",
                            _ms(result["median_s"]),
                            _ms(result["p90_s"]),
                            _gb(line["peak_bytes"]),
                            line.get("validity_note", ""),
                        ]
                    )
    header = [
        "Model size",
        "Input resolution",
        "Batch size",
        "Dtype",
        "Dtype mode",
        "Runs",
        "Throughput (img/s)",
        "Median (ms)",
        "p90 (ms)",
        "Peak memory (GB)",
        "Note",
    ]
    return _table(header, rows)


def _all_modes_section(lines: list[dict]) -> str:
    rows = []
    for line in latest_lines(lines):
        if line["status"] != "done" or line["compile"] or line["repeat"]:
            continue
        rows.append(
            [
                line["model_size"],
                line["side"],
                line["batch_size"],
                line["dtype_mode"],
                "yes" if line["valid"] else f"no: {line['validity_note']}",
                f"{line['frame']['throughput_ips']:.1f}",
                f"{line['patch']['throughput_ips']:.1f}",
                f"{line['device']['throughput_ips']:.1f}",
                _ms(line["frame"]["median_s"]),
                _ms(line["frame"]["p90_s"]),
                _gb(line["peak_bytes"]),
                _gb(line.get("load_peak_bytes")),
                _gb(line.get("mps_driver_max_bytes")),
                _gb(line["need_bytes"]),
                line.get("timed_batches"),
            ]
        )
    header = [
        "Model size",
        "Input resolution",
        "Batch size",
        "Dtype mode",
        "Valid",
        "Frame (img/s)",
        "Patch (img/s)",
        "Device (img/s)",
        "Frame median (ms)",
        "Frame p90 (ms)",
        "Peak (GB)",
        "Load peak (GB)",
        "MPS driver (GB)",
        "Need (GB)",
        "Timed batches",
    ]
    return _table(header, rows)


def _priority_lines(lines: list[dict], name: str, compile: bool) -> dict[str, list[dict]]:
    model_size, side, batch_size, _ = PRIORITY[name]
    found: dict[str, list[dict]] = {mode: [] for mode in DTYPE_MODE_NAMES}
    for line in latest_lines(lines):
        same = (line["model_size"], line["side"], line["batch_size"]) == (
            model_size,
            side,
            batch_size,
        )
        if same and line["compile"] == compile:
            found[line["dtype_mode"]].append(line)
    return found


def spreads(lines: list[dict]) -> list[tuple[str, str, list[float], float | None]]:
    """Return (priority name, dtype mode, throughputs, spread) for P1 and P2 (Section 3.4)."""
    result = []
    for name, (_, _, _, output) in PRIORITY.items():
        for mode, group in _priority_lines(lines, name, compile=False).items():
            done = sorted(
                (line for line in group if line["status"] == "done" and line["valid"]),
                key=lambda line: line["repeat"],
            )
            values = [line[output]["throughput_ips"] for line in done]
            spread = None
            if len(values) >= 2:
                spread = (max(values) - min(values)) / statistics.median(values)
            result.append((name, mode, values, spread))
    return result


def repeatability_note(lines: list[dict]) -> str:
    """Return the warning about the spread of one run for the header (finding F1 of verify.html)."""
    found = [(values, spread) for *_, values, spread in spreads(lines) if spread is not None]
    if not found:
        return "Repeatability: no priority configuration has more than one run yet."
    counts = sorted(len(values) for values, _ in found)
    low, high = min(spread for _, spread in found), max(spread for _, spread in found)
    ratio = max(max(values) / min(values) for values, _ in found)
    return (
        f"Repeatability: over {counts[0]} to {counts[-1]} runs, the throughput of P1 and P2 "
        f"spreads by {100 * low:.1f} to {100 * high:.1f} % (Section 4, limit 5 %). The fastest "
        f"and the slowest run of one configuration differ by up to {ratio:.1f}x. So one run can "
        "be far off on this Mac (finding F1 of `specs/03-baseline/verify.html`). A baseline row "
        "with 1 run is one sample."
    )


def _repeats_section(lines: list[dict]) -> str:
    rows = []
    for name, mode, values, spread in spreads(lines):
        output = PRIORITY[name][3]
        if spread is None:
            verdict, text = f"{len(values)} valid runs", "–"
        else:
            verdict = "pass" if spread <= SPREAD_LIMIT else "above 5 %"
            text = f"{100 * spread:.2f} %"
        rows.append(
            [
                name,
                OUTPUT_NAMES[output],
                mode,
                ", ".join(f"{v:.1f}" for v in values) or "–",
                text,
                verdict,
            ]
        )
    header = ["Priority", "Output", "Dtype mode", "Throughputs (img/s)", "Spread", "Result"]
    return _table(header, rows)


def compile_gains(
    lines: list[dict],
) -> list[tuple[str, str, dict | None, list[float], float | None]]:
    """Return (priority name, dtype mode, compile line, eager throughputs, gain) for P1 and P2.

    The gain compares the compile row with the median of the valid eager runs: the main run and
    the repeats (D7, finding F2 of verify.html).
    """
    result = []
    for name, (_, _, _, output) in PRIORITY.items():
        compiled = _priority_lines(lines, name, compile=True)
        eager = _priority_lines(lines, name, compile=False)
        for mode in DTYPE_MODE_NAMES:
            c_line = next(iter(compiled[mode]), None)
            values = [
                line[output]["throughput_ips"]
                for line in eager[mode]
                if line["status"] == "done" and line["valid"]
            ]
            gain = None
            if c_line and c_line["status"] == "done" and values:
                gain = c_line[output]["throughput_ips"] / statistics.median(values) - 1
            result.append((name, mode, c_line, values, gain))
    return result


def compile_summary(lines: list[dict]) -> str:
    """Return one sentence with the number of compile rows above the 10 % limit."""
    gains = [gain for *_, gain in compile_gains(lines) if gain is not None]
    above = sum(gain > COMPILE_GAIN_LIMIT for gain in gains)
    return (
        f"{above} of {len(gains)} compile rows are more than 10 % faster than the median of "
        "their eager runs."
    )


def _compile_section(lines: list[dict]) -> str:
    rows = []
    for name, mode, c_line, values, gain in compile_gains(lines):
        output = PRIORITY[name][3]
        if c_line is None:
            rows.append([name, mode, "no line", "–", "–", "–", "–"])
            continue
        if c_line["status"] != "done":
            reason = c_line["reason"].splitlines()[-1] if c_line["reason"] else ""
            rows.append([name, mode, c_line["status"], "–", "–", "–", reason])
            continue
        eager = f"{statistics.median(values):.1f} ({len(values)} runs)" if values else "–"
        rows.append(
            [
                name,
                mode,
                "done" + ("" if c_line["valid"] else ", not valid"),
                f"{c_line[output]['throughput_ips']:.1f}",
                eager,
                "–" if gain is None else f"{100 * gain:+.1f} %",
                f"first call {c_line['first_call_s']:.1f} s",
            ]
        )
    header = [
        "Priority",
        "Dtype mode",
        "Status",
        "Compiled (img/s)",
        "Eager median (img/s)",
        "Gain",
        "Note",
    ]
    return _table(header, rows)


def _skip_section(lines: list[dict]) -> str:
    rows = []
    for line in latest_lines(lines):
        if line["status"] == "done":
            continue
        reason = " ".join(line["reason"].split())[:300]
        rows.append(
            [
                line["model_size"],
                line["dtype_mode"],
                line["side"],
                line["batch_size"],
                "yes" if line["compile"] else "no",
                line["repeat"],
                line["status"],
                _gb(line["need_bytes"]),
                _gb(line["available_bytes"]),
                reason,
            ]
        )
    header = [
        "Model size",
        "Dtype mode",
        "Input resolution",
        "Batch size",
        "Compile",
        "Repeat",
        "Status",
        "Need (GB)",
        "Available (GB)",
        "Reason",
    ]
    return _table(header, rows) if rows else "None."


def _budget_ok(line: dict) -> bool:
    timed = line["timed_batches"] >= TIMED.min_batches and line["timed_s"] >= TIMED.min_seconds
    capped = line["timed_s"] >= TIMED.max_seconds and line["timed_batches"] >= TIMED.floor_batches
    warm = (
        line["warm_up_batches"] >= WARM_UP.min_batches and line["warm_up_s"] >= WARM_UP.min_seconds
    )
    return (timed or capped) and warm


def checks(lines: list[dict]) -> list[str]:
    """Return the check lines of AC1, AC3, AC4 and AC7 for the latest lines."""
    latest = latest_lines(lines)
    main = [line for line in latest if not line["compile"] and not line["repeat"]]
    expected = 225 + 2 * len(SIDES) * len(BATCH_SIZES)
    statuses = {}
    for line in main:
        statuses[line["status"]] = statuses.get(line["status"], 0) + 1
    done = [line for line in latest if line["status"] == "done"]
    invalid = {
        (line["model_size"], line["dtype_mode"], line["side"])
        for line in done
        if not line["valid"] and not line["compile"]
    }
    invalid_count = sum(1 for line in main if line["status"] == "done" and not line["valid"])
    over_budget = [line for line in done if not _budget_ok(line)]
    above_need = [line for line in done if line["peak_bytes"] > line["need_bytes"]]
    worst = max((line["peak_bytes"] / line["need_bytes"] for line in done), default=0.0)
    unexpected = sorted(invalid - EXPECTED_INVALID)
    missing_invalid = sorted(EXPECTED_INVALID - invalid)
    return [
        f"- AC1: {len(main)} of {expected} configurations of B1 and B4 have a line: {statuses}.",
        (
            f"- AC3: {invalid_count} done lines of B1 and B4 are not valid. Not expected: "
            f"{unexpected or 'none'}. Expected but valid or not run: {missing_invalid or 'none'}."
        ),
        f"- AC4: {len(over_budget)} done lines miss the time budget of Table 5.",
        (
            f"- AC7: {len(above_need)} done lines have a peak memory above the need. "
            f"Largest peak / need: {worst:.2f}."
        ),
    ]


def _machine_section(lines: list[dict]) -> str:
    seen: dict[str, set[str]] = {}
    for line in lines:
        machine = line.get("machine") or {}
        for key in (
            "power_source",
            "adapter_watts",
            "battery_percent",
            "low_power_mode",
            "thermal",
            "chip",
        ):
            seen.setdefault(key, set()).add(str(machine.get(key)))
        if machine.get("memory_bytes"):
            seen.setdefault("memory", set()).add(f"{machine['memory_bytes'] / 2**30:.0f} GiB")
        versions = line.get("versions") or machine.get("versions") or {}
        for package, value in versions.items():
            seen.setdefault(package, set()).add(value)
    rows = [[key, ", ".join(sorted(values))] for key, values in seen.items()]
    return _table(["Item", "Values"], rows)


def write_baseline_table(lines: list[dict]) -> str:
    """Return the Markdown text of results/baseline-torch-mps.md."""
    times = sorted(line["time"] for line in lines if line.get("time"))
    period = f"{times[0]} to {times[-1]}" if times else "no lines"
    parts = [
        "# Baseline: Hugging Face backbone on torch-mps",
        "",
        (
            f"Written by `scripts/baseline_table.py` on {date.today().isoformat()} from "
            f"`results/baseline-torch-mps.jsonl` ({len(lines)} lines, {period}). "
            "Spec: `specs/03-baseline/plan.html`. The latest line of each configuration counts."
        ),
        "",
        (
            "Throughput is images per second, host to host (D3). Peak memory is the lifetime maximum "
            "of the macOS process footprint of the worker (D5). Latency is the median and the p90 of "
            "the batch times."
        ),
        "",
        repeatability_note(lines),
        "",
        "## Checks",
        "",
        *checks(lines),
    ]
    for number, output in enumerate(OUTPUTS, start=1):
        parts += [
            "",
            f"## {number}. Baseline rows: {OUTPUT_NAMES[output]}",
            "",
            (
                "The fastest valid dtype mode of each dtype (D2, Section 3.7 of the plan). Each "
                "dtype mode counts with the median of its valid runs; Runs gives their number "
                "(finding F1 of `specs/03-baseline/verify.html`). Compile rows count since the "
                "decision of the user on 2026-11-06; only P1 and P2 have them."
            ),
            "",
            _baseline_section(lines, output),
        ]
    parts += [
        "",
        "## 3. All dtype modes",
        "",
        _all_modes_section(lines),
        "",
        "## 4. Repeats and spread",
        "",
        (
            "Spread = (largest − smallest) / median of the throughputs of the main run and the "
            "repeats. Limit: 5 %."
        ),
        "",
        _repeats_section(lines),
        "",
        "## 5. Compile rows",
        "",
        compile_summary(lines)
        + " On 2026-11-06 the user decided that the compile rows join the baseline (D7).",
        "",
        _compile_section(lines),
        "",
        "## Skipped, deferred and error lines",
        "",
        _skip_section(lines),
        "",
        "## Machine state and versions",
        "",
        _machine_section(lines),
        "",
    ]
    return "\n".join(parts)


def forward_time(line: dict) -> float:
    """Return the forward time of the checks: the mean of the times before and after the parts.

    Lines from before finding F4 of verify.html have only the time after the parts.
    """
    before = line.get("forward_before_s")
    return line["forward_s"] if before is None else (before + line["forward_s"]) / 2


def profile_checks(line: dict) -> tuple[float, float]:
    """Return the sum of the parts / the block, and (L x block + outer parts) / the forward pass."""
    part_sum = sum(line["parts_s"].values()) / line["block_s"]
    whole = (line["blocks"] * line["block_s"] + sum(line["outer_s"].values())) / forward_time(line)
    return part_sum, whole


def write_profile_table(lines: list[dict]) -> str:
    """Return the Markdown text of results/block-profile-torch-mps.md."""
    skipped = [line for line in lines if line.get("status", "done") != "done"]
    lines = [line for line in lines if line.get("status", "done") == "done"]
    parts = [
        "# Block profile: Hugging Face backbone on torch-mps",
        "",
        (
            f"Written by `scripts/baseline_table.py` on {date.today().isoformat()} from "
            "`results/block-profile-torch-mps.jsonl`. Spec: Section 3.11 of "
            "`specs/03-baseline/plan.html` (D6)."
        ),
        "",
        (
            "Each part of block 1 runs alone in a loop with one synchronize at the end. The time is "
            "the median of five loops. The FLOP share comes from `dinov3_apple.flops`. Limits of the "
            "checks: 15 %."
        ),
        "",
        (
            "The forward pass runs before and after the parts; the check uses the mean of the two. "
            "A large change between the two shows a change of the speed of the Mac during the "
            "profile. At batch size 1 a block alone takes longer than a block inside the forward "
            "pass, so Forward / L gives the time of one block inside the forward pass (finding F4 "
            "of `specs/03-baseline/verify.html`). Lines from before F4 have only the time after "
            "the parts."
        ),
        "",
        "## Summary",
        "",
    ]
    summary_rows = []
    for line in lines:
        part_sum, whole = profile_checks(line)
        before = line.get("forward_before_s")
        forward = (
            _ms(line["forward_s"])
            if before is None
            else (f"{_ms(before)} → {_ms(line['forward_s'])}")
        )
        summary_rows.append(
            [
                line["model_size"],
                line["dtype_mode"],
                line["side"],
                line["batch_size"],
                _ms(line["block_s"]),
                _ms(forward_time(line) / line["blocks"]),
                f"{part_sum:.3f}",
                "pass" if abs(part_sum - 1) <= PART_SUM_LIMIT else "outside 15 %",
                forward,
                f"{whole:.3f}",
                "pass" if abs(whole - 1) <= PART_SUM_LIMIT else "outside 15 %",
            ]
        )
    parts.append(
        _table(
            [
                "Model size",
                "Dtype mode",
                "Input resolution",
                "Batch size",
                "Block (ms)",
                "Forward / L (ms)",
                "Sum of parts / block",
                "Check",
                "Forward pass, before → after (ms)",
                "(L × block + outer) / forward",
                "Check",
            ],
            summary_rows,
        )
    )
    for line in lines:
        total_time = sum(line["parts_s"].values())
        total_flops = sum(line["part_flops"].values())
        rows = []
        for name in PART_NAMES:
            seconds, flops = line["parts_s"][name], line["part_flops"][name]
            rows.append(
                [
                    name,
                    f"{1e6 * seconds:.1f}",
                    f"{100 * seconds / total_time:.1f} %",
                    f"{100 * flops / total_flops:.1f} %" if total_flops else "–",
                    f"{flops / seconds / 1e12:.2f}" if flops else "–",
                ]
            )
        outer = ", ".join(f"{name} {1e6 * s:.1f} µs" for name, s in line["outer_s"].items())
        parts += [
            "",
            (
                f"## {line['model_size']}, {line['dtype_mode']}, {line['side']} px, "
                f"batch size {line['batch_size']}"
            ),
            "",
            _table(["Part", "Time (µs)", "Time share", "FLOP share", "TFLOP/s"], rows),
            "",
            f"Parts outside the blocks: {outer}.",
        ]
    if skipped:
        parts += ["", "## Skipped", ""]
        parts += [f"- {s['model_size']} {s['dtype_mode']}: {s['reason']}" for s in skipped]
    parts.append("")
    return "\n".join(parts)


def main() -> None:
    """Write the two results tables from the JSON lines."""
    BASELINE_TABLE.write_text(write_baseline_table(read_lines(BASELINE_LINES)))
    print(f"wrote {BASELINE_TABLE}")
    if PROFILE_LINES.exists():
        PROFILE_TABLE.write_text(write_profile_table(read_lines(PROFILE_LINES)))
        print(f"wrote {PROFILE_TABLE}")


if __name__ == "__main__":
    main()
