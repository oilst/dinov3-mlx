"""Write results/speedup-table.md from the Step 6 result files (Table 13 of
specs/06-speed/plan.html).

The functions of src/dinov3_apple/speedup.py do the work: ADR 0004 on both sides, the pass rule
of D21 and the compile check. A reader of paired lines keys each line with its backend.

Usage:
    uv run python scripts/speedup_table.py
    uv run python scripts/speedup_table.py --results-dir results --out <folder>
"""

import argparse
import json
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np

from dinov3_apple.bench import DTYPE_MODE_NAMES
from dinov3_apple.flops import PART_NAMES
from dinov3_apple.hub import MODEL_IDS
from dinov3_apple.speedup import (
    NOISE,
    autocast_over_cast,
    compile_check,
    matrix_cells,
    pair_rows,
    speedup_rows,
)

REPO_ROOT = Path(__file__).parents[1]
RESULTS_DIR = REPO_ROOT / "results"
FILES = {
    "changes": "speed-changes.jsonl",
    "matrix": "speed-matrix.jsonl",
    "pairs": "speed-pairs.jsonl",
    "sweep": "speed-sweep.jsonl",
    "learn": "speed-learn.jsonl",
    "profile": "speed-profile-p1.jsonl",
    "step3": "baseline-torch-mps.jsonl",
    "block_profile": "block-profile-torch-mps.jsonl",
}
CHANGE_NAMES = {
    "C1": "Compiled forward pass (E → A)",
    "C2": "Concat-free RoPE (A → R)",
    "C3": "Autocast / cast after compile",
    "C4": "fp16 patch embeddings (R-fp32 → R-fp16)",
    "C5": "Quantized weights / unquantized",
    "C6": "Preprocessing in MLX / CPU path",
    "C7": "Fix of the P1 profile",
}
PRIORITY_OUTPUT = {"P1": "frame", "P2": "patch"}


def read_lines(path: Path) -> list[dict]:
    """Return the JSON lines of a file, or [] if it does not exist."""
    if not path.exists():
        return []
    return [json.loads(text) for text in path.read_text().splitlines() if text.strip()]


def _f(value: float | None, digits: int = 3) -> str:
    return "–" if value is None else f"{value:.{digits}f}"


def _ips(value: float | None) -> str:
    return "–" if value is None else f"{value:.1f}"


def _table(header: list[str], rows: list[list[object]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return lines


def pairs_section(pairs: list[dict]) -> list[list[object]]:
    """Section 1: the 12 verdict rows of the paired runs at P1 and P2."""
    rows = []
    for row in pair_rows(pairs):
        compiled = "" if row.torch_compile is None else (" (compile)" if row.torch_compile else "")
        rows.append(
            [
                row.priority,
                row.dtype,
                row.output,
                row.mlx_mode or "–",
                f"{row.torch_mode or '–'}{compiled}",
                *[_f(m) for m in (row.run_medians + [None, None])[:2]],
                _f(row.pooled_median),
                _f(row.spread),
                row.verdict or "no valid rows",
                row.ratios,
            ]
        )
    return rows


def matrix_section(lines: list[dict]) -> list[list[object]]:
    """Section 2: the 324 cells of the matrix."""
    rows = []
    for c in matrix_cells(lines):
        torch_mode = (
            "–"
            if c.torch_mode is None
            else c.torch_mode + (" (compile)" if c.torch_compile else " (eager)")
        )
        rows.append(
            [
                c.model_size,
                c.side,
                c.batch_size,
                c.output,
                c.dtype,
                c.mlx_mode or "–",
                _ips(c.mlx_ips),
                torch_mode,
                _ips(c.torch_ips),
                _f(c.ratio),
                c.verdict or "–",
                c.mark or "",
                c.reason,
            ]
        )
    return rows


def quantized_section(lines: list[dict]) -> list[list[object]]:
    """Section 3: the quantized rows."""
    rows = []
    for r in speedup_rows(lines):
        if r.bits is None:
            continue
        parity = r.parity or {}
        rows.append(
            [
                r.model_size,
                r.side,
                r.batch_size,
                r.output,
                r.mlx_mode,
                r.bits,
                "no" if r.verdict == "not valid" else ("yes" if r.ratio is not None else "–"),
                _f(parity.get("frame"), 6),
                _f(parity.get("patch_mean"), 6),
                _f(parity.get("patch_p01"), 6),
                _ips(r.mlx_ips),
                _ips(r.torch_ips),
                _f(r.ratio),
                _f(r.unquantized_ratio),
                r.verdict or "–",
                r.reason,
            ]
        )
    return rows


def _summary(values: list[float]) -> str:
    if not values:
        return "–"
    return f"{float(np.median(values)):.3f} [{min(values):.3f}–{max(values):.3f}]"


def changes_section(lines: list[dict]) -> list[list[object]]:
    """Section 4: the change ratios C1 to C7: median [min–max] over the rounds."""
    rows = []
    for line in lines:
        if line.get("kind") == "change" and line.get("status") == "done":
            output = PRIORITY_OUTPUT[line["priority"]]
            equal = "yes" if line.get("frame_equal") and line.get("patches_equal") else "no"
            if line["change"] == "C4":
                equal = "frame yes" if line.get("frame_equal") else "frame no"
            rows.append(
                [
                    line["change"],
                    CHANGE_NAMES[line["change"]],
                    line["priority"],
                    line["dtype_mode"],
                    " → ".join(line["forms"]),
                    output,
                    _summary(line["ratios"][output]),
                    equal,
                ]
            )
    # C3 and C5: one subprocess for each configuration and round. A change ratio measures the
    # speed, so a 4-bit row that fails THRESHOLDS[mode] still gives its ratio.
    groups: dict[tuple, dict[tuple, float]] = defaultdict(dict)
    for line in lines:
        if line.get("change") in ("C3", "C5") and line.get("status") == "done":
            output = PRIORITY_OUTPUT[line["priority"]]
            key = (line["change"], line["priority"], line["dtype_mode"].split("-")[0])
            if line["change"] == "C5":
                key = (line["change"], line["priority"], line["dtype_mode"])
            groups[key][(line["round"], line["form"])] = line[output]["throughput_ips"]
    for (change, priority, name), values in sorted(groups.items()):
        output = PRIORITY_OUTPUT[priority]
        rounds = sorted({r for r, _ in values})
        if change == "C3":
            pairs = [(f"{name}-cast", f"{name}-autocast")]
        else:
            pairs = [("unquantized", "8-bit"), ("unquantized", "4-bit")]
        for old, new in pairs:
            ratios = [
                values[(r, new)] / values[(r, old)]
                for r in rounds
                if (r, old) in values and (r, new) in values
            ]
            rows.append(
                [
                    change,
                    CHANGE_NAMES[change],
                    priority,
                    name,
                    f"{old} → {new}",
                    output,
                    _summary(ratios),
                    "no (two configurations)",
                ]
            )
    for line in lines:
        if line.get("kind") == "preprocess" and line.get("status") == "done":
            for side, values in line["preprocess"].items():
                rows.append(
                    [
                        "C6",
                        CHANGE_NAMES["C6"],
                        f"{side} px",
                        "fp32",
                        f"CPU {values['cpu_median_ms']:.2f} ms → MLX {values['mlx_median_ms']:.2f} ms per frame",
                        "pixels",
                        _summary(values["ratios"]),
                        "T52 covers the values",
                    ]
                )
    return rows


def best_batch_section(
    lines: list[dict], sweep: list[dict]
) -> tuple[list[list[object]], list[list[object]]]:
    """Section 5: the best batch size of each model size, input resolution and dtype mode (mlx,
    compiled, unquantized), and the sweep."""
    found: dict[tuple, list[dict]] = defaultdict(list)
    for line in lines:
        mlx = line["backend"] == "mlx" and not line.get("bits")
        if mlx and line.get("status") == "done" and line.get("phase", "matrix") == "matrix":
            found[(line["model_size"], line["side"], line["dtype_mode"])].append(line)
    rows = []
    for model_size in MODEL_IDS:
        for side in (224, 448, 512):
            for mode in DTYPE_MODE_NAMES:
                group = sorted(
                    found.get((model_size, side, mode), []), key=lambda l: l["batch_size"]
                )
                if not group:
                    continue
                best = max(group, key=lambda l: l["frame"]["throughput_ips"])
                cells = "; ".join(
                    f"b{l['batch_size']}: {l['frame']['throughput_ips']:.1f} img/s, "
                    f"{1e3 * l['frame']['median_s']:.1f} / {1e3 * l['frame']['p90_s']:.1f} ms"
                    for l in group
                )
                rows.append([model_size, side, mode, best["batch_size"], cells])
    by_point: dict[tuple, list[dict]] = defaultdict(list)
    for line in sweep:
        if line.get("status") == "done":
            by_point[(line["model_size"], line["side"], line["batch_size"])].append(line)
    sweep_rows = []
    for (model_size, side, batch_size), group in sorted(
        by_point.items(), key=lambda i: (list(MODEL_IDS).index(i[0][0]), i[0][2])
    ):
        output = "frame" if side == 224 else "patch"
        values = [l[output]["throughput_ips"] for l in group]
        median = [1e3 * l[output]["median_s"] for l in group]
        p90 = [1e3 * l[output]["p90_s"] for l in group]
        sweep_rows.append(
            [
                model_size,
                side,
                batch_size,
                output,
                len(group),
                _summary(values),
                f"{np.median(median):.2f}",
                f"{np.median(p90):.2f}",
            ]
        )
    return rows, sweep_rows


def profile_section(profile: list[dict], block_profile: list[dict]) -> list[str]:
    """Section 7, part: the P1 profile against the eager torch-mps parts of Step 3."""
    text: list[str] = []
    torch_line = next(
        (
            l
            for l in block_profile
            if (l.get("model_size"), l.get("dtype_mode"), l.get("side"), l.get("batch_size"))
            == ("ViT-S/16", "fp16-cast", 224, 8)
            and l.get("status", "done") == "done"
        ),
        None,
    )
    for line in profile:
        if line.get("status") != "done":
            text.append(f"- {line['dtype_mode']}: {line['status']}, {line['reason']}")
            continue
        host = line["host"]
        text.append(
            f"- {line['dtype_mode']}: compiled call {1e3 * host['call_median_s']:.3f} ms, call to the end of mx.eval "
            f"{1e3 * host['eval_median_s']:.3f} ms, gap {1e3 * host['gap_median_s']:.3f} ms, host share "
            f"{host['host_share']:.3f}; kernels: compiled {line['kernels']['compiled']}, eager "
            f"{line['kernels']['eager']}; batch size 8: {line['batch_sizes']['8']['throughput_ips']:.1f} img/s, "
            f"batch size 32: {line['batch_sizes']['32']['throughput_ips']:.1f} img/s; block 1 compiled "
            f"{1e3 * line['block_s']:.3f} ms."
        )
    rows = []
    for name in PART_NAMES:
        row = [name]
        for line in profile:
            row.append(_f(1e3 * line["parts_s"][name], 4) if line.get("status") == "done" else "–")
        row.append(_f(1e3 * torch_line["parts_s"][name], 4) if torch_line else "–")
        rows.append(row)
    modes = [f"mlx {l['dtype_mode']} (ms)" for l in profile]
    text += ["", *_table(["Part", *modes, "torch-mps fp16-cast eager, Step 3 (ms)"], rows)]
    return text


def write_table(files: dict[str, list[dict]]) -> str:
    """Return the Markdown text of results/speedup-table.md."""
    matrix = files["matrix"]
    text = [
        "# Speedup of mlx against torch-mps",
        "",
        (
            f"Made by `scripts/speedup_table.py` on {date.today().isoformat()} from the Step 6 "
            "result files. Speedup = mlx throughput / torch-mps throughput of a paired run "
            "(ADR 0004, ADR 0009). Each side takes the fastest valid row of its dtype and output. "
            f'A ratio within 1 ± {NOISE} gets the mark "inside the noise". For an even number of '
            "values the median is the lower of the two middle values."
        ),
        "",
        "## 1 P1 and P2: paired runs",
        "",
        *_table(
            [
                "Priority",
                "Dtype",
                "Output",
                "mlx mode",
                "torch-mps mode",
                "Run 1 median",
                "Run 2 median",
                "Pooled median",
                "Spread between runs",
                "Verdict (D21)",
                "Ratios",
            ],
            pairs_section(files["pairs"]),
        ),
        "",
        "Table 1. The paired speedup runs at P1 and P2 (D7, D21).",
        "",
        "## 2 Matrix",
        "",
        *_table(
            [
                "Model size",
                "Input resolution",
                "Batch size",
                "Output",
                "Dtype",
                "mlx mode",
                "mlx (img/s)",
                "torch-mps mode",
                "torch-mps (img/s)",
                "Paired ratio",
                "Verdict",
                "Mark",
                "Reason of an empty cell",
            ],
            matrix_section(matrix),
        ),
        "",
        "Table 2. The 324 cells of the benchmark matrix, one paired round (D20).",
        "",
        "## 3 Quantized rows",
        "",
        *_table(
            [
                "Model size",
                "Input resolution",
                "Batch size",
                "Output",
                "Dtype mode",
                "Bits",
                "Valid",
                "Frame",
                "Patch mean",
                "Patch 1st percentile",
                "mlx (img/s)",
                "torch-mps (img/s)",
                "Paired ratio",
                "Ratio to unquantized",
                "Verdict",
                "Reason",
            ],
            quantized_section(matrix),
        ),
        "",
        "Table 3. The quantized rows (D18, D22, ADR 0011): valid only if image 0 passes THRESHOLDS[mode].",
        "",
        "## 4 Change ratios",
        "",
        *_table(
            [
                "Change",
                "Name",
                "Priority",
                "Dtype mode",
                "Forms",
                "Output",
                "Median [min–max]",
                "Bit equality",
            ],
            changes_section(files["changes"]),
        ),
        "",
        "Table 4. The change ratios (Table 23 of the plan), 5 rounds each.",
        "",
    ]
    best, sweep = best_batch_section(matrix, files["sweep"])
    text += [
        "## 5 Best batch size",
        "",
        *_table(
            [
                "Model size",
                "Input resolution",
                "Dtype mode",
                "Best batch size",
                "Frame: throughput, median / p90 latency",
            ],
            best,
        ),
        "",
        "Table 5. The best batch size of the compiled mlx rows of the matrix (D10 option a).",
        "",
        *_table(
            [
                "Model size",
                "Input resolution",
                "Batch size",
                "Output",
                "Rounds",
                "Throughput median [min–max]",
                "Median latency (ms)",
                "p90 latency (ms)",
            ],
            sweep,
        ),
        "",
        "Table 6. The batch-size sweep, compiled mlx in bf16-cast (D10 option c).",
        "",
        "## 6 Autocast / cast",
        "",
    ]
    ratios = autocast_over_cast(files["pairs"])
    text += _table(
        ["Priority", "Dtype", "Output", "Autocast / cast (mlx, compiled)"],
        [[p, d, o, _f(v)] for (p, d, o), v in sorted(ratios.items())],
    )
    text += [
        "",
        "Table 7. Autocast / cast on mlx after compile, median over the rounds of the paired runs (C3, R30).",
        "",
    ]
    listed = compile_check(matrix, files["step3"])
    marks = sorted(
        {
            (c.model_size, c.side, c.batch_size, c.dtype, c.mark)
            for c in matrix_cells(matrix)
            if c.mark
        }
    )
    text += [
        "## 7 Compile check, marks and the P1 profile",
        "",
        "Groups where the compiled torch-mps row is below the Step 3 eager median of its mode:",
        "",
        *([f"- {s} {side} px b{b} {m}" for s, side, b, m in listed] or ["- none"]),
        "",
        "Eager fallbacks and other marks:",
        "",
        *([f"- {s} {side} px b{b} {d}: {mark}" for s, side, b, d, mark in marks] or ["- none"]),
        "",
        "The P1 profile (Table 12 of the plan), ViT-S/16, 224 px, batch size 8:",
        "",
        *profile_section(files["profile"], files["block_profile"]),
        "",
        "Table 8. The parts of block 1 at P1, each alone. mlx: compiled, 3 warm-up and 20 calls. torch-mps: eager, Step 3.",
        "",
    ]
    learn = files["learn"]
    if learn:
        text += ["## 8 Learn data", ""]
        rows = []
        for line in learn:
            if line.get("phase") == "learn" and line.get("kind") == "bench":
                rows.append(
                    [
                        line["model_size"],
                        line["dtype_mode"],
                        line["side"],
                        line["batch_size"],
                        line["status"],
                        line.get("finite"),
                        _ips((line.get("frame") or {}).get("throughput_ips")),
                    ]
                )
            elif line.get("kind") == "token-range":
                rows.append(
                    [
                        line["model_size"],
                        line["dtype_mode"],
                        line["side"],
                        1,
                        line["status"],
                        line.get("finite"),
                        f"largest value {line.get('largest_value', 0):.1f}",
                    ]
                )
            elif line.get("phase") == "nax":
                rows.append(
                    [
                        line["model_size"],
                        f"{line['dtype_mode']} NAX {line.get('nax')}",
                        line["side"],
                        line["batch_size"],
                        line["status"],
                        line.get("finite"),
                        _ips((line.get("frame") or {}).get("throughput_ips")),
                    ]
                )
        text += _table(
            [
                "Model size",
                "Dtype mode",
                "Input resolution",
                "Batch size",
                "Status",
                "Finite",
                "Throughput or value",
            ],
            rows,
        )
        text += [
            "",
            "Table 9. The Learn data above 512 px and with the Neural Accelerators off (D11, D19).",
            "",
        ]
    return "\n".join(text)


def main(argv: list[str] | None = None) -> None:
    """Write results/speedup-table.md, or speedup-table.md in the folder of --out."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR)
    args = parser.parse_args(argv)
    files = {name: read_lines(args.results_dir / file) for name, file in FILES.items()}
    if not files["step3"]:
        files["step3"] = read_lines(RESULTS_DIR / FILES["step3"])
    if not files["block_profile"]:
        files["block_profile"] = read_lines(RESULTS_DIR / FILES["block_profile"])
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "speedup-table.md"
    path.write_text(write_table(files))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
