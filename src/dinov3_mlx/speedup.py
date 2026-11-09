"""The speedup of each configuration: ADR 0004 on both sides, the pass rule and the compile check
(Section 3.15 of specs/06-speed/plan.html, ADR 0009).

A reader of paired lines keeps the backend in each key: line_key has no backend, so latest_lines
alone would drop one of two lines with the same key on two backends. No torch, no mlx import.
"""

from collections.abc import Iterable
from dataclasses import dataclass, field

from dinov3_mlx.bench import (
    BATCH_SIZES,
    DTYPE_OF_MODE,
    OUTPUTS,
    QUANTIZED_MODES,
    SIDES,
    line_key,
)
from dinov3_mlx.hub import MODEL_IDS
from dinov3_mlx.memory import NO_FP32_WEIGHTS

NOISE = 0.03  # the spread inside one arm of the probe paired-noise
MODEL_SIZES = list(MODEL_IDS)
DTYPES = ["fp32", "fp16", "bf16"]
QUANTIZED_BITS = (8, 4)
NO_FP32_REASON = "ADR 0008: ViT-7B/16 never loads its fp32 weights"
# The phases of the matrix lines that join the choice of ADR 0004 (the gate J13 stays out).
MATRIX_PHASES = ("matrix", "compile-check")


def _lower_median(values: list[float]) -> float:
    """Return the median; for an even count, the lower of the two middle values (ADR 0004)."""
    ordered = sorted(values)
    return ordered[(len(ordered) - 1) // 2]


def pass_verdict(run_medians: list[float], pooled_median: float) -> str:
    """Return the verdict of a speedup at P1 or P2 (D21, option b).

    "above 1" only if the pooled median of all paired ratios and the median of each run are above
    1. Two runs that disagree in sign give "not above 1". A pooled median within 1 ± NOISE gets the
    mark "inside the noise" (Section 3.15 of the plan; finding V5 of verify.html).
    """
    above = pooled_median > 1 and all(median > 1 for median in run_medians)
    verdict = "above 1" if above else "not above 1"
    return f"{verdict}, inside the noise" if abs(pooled_median - 1) <= NOISE else verdict


def single_verdict(ratio: float) -> str:
    """Return the verdict of one paired ratio outside P1 and P2 (D20, D21). A ratio within
    1 ± NOISE gets the mark "inside the noise"."""
    verdict = "above 1" if ratio > 1 else "not above 1"
    return f"{verdict}, inside the noise" if abs(ratio - 1) <= NOISE else verdict


def spread_between_runs(run_medians: list[float]) -> float:
    """Return (largest run median − smallest run median) / smallest run median."""
    return (max(run_medians) - min(run_medians)) / min(run_medians)


def _ips(line: dict, output: str) -> float:
    return line[output]["throughput_ips"]


def _usable(line: dict) -> bool:
    return line.get("status") == "done" and bool(line.get("valid"))


def _cell(line: dict) -> tuple[str, int, int]:
    return line["model_size"], line["side"], line["batch_size"]


def _latest(lines: Iterable[dict]) -> list[dict]:
    """Return the last line of each (phase, backend, key), in the order of first appearance."""
    latest: dict[tuple, dict] = {}
    for line in lines:
        if line.get("phase", "matrix") in MATRIX_PHASES:
            latest[(line.get("phase", "matrix"), line["backend"], line_key(line))] = line
    return list(latest.values())


def _reasons(lines: list[dict]) -> str:
    """The reasons of the lines of one side without a usable line."""
    found = []
    for line in lines:
        if line.get("status") != "done":
            found.append(
                f"{line['backend']} {line['dtype_mode']}: {line['status']}, {line['reason']}"
            )
        elif not line.get("valid"):
            note = line.get("validity_note", "")
            found.append(f"{line['backend']} {line['dtype_mode']}: not valid, {note}")
    return "; ".join(found)


@dataclass(frozen=True)
class SpeedupRow:
    """The speedup of one model size, input resolution, batch size, output and dtype.

    Attributes:
        bits: None for the main row; 8 or 4 for a quantized row (D22).
        mlx_mode: The dtype mode of the mlx row (ADR 0004), or the mode of a quantized row.
        torch_mode: The dtype mode of the torch-mps row (ADR 0004).
        torch_compile: True if the torch-mps row ran torch.compile.
        ratio: mlx throughput / torch-mps throughput, or None for an empty cell.
        unquantized_ratio: A quantized row only: its throughput / the unquantized mlx row of the
            same dtype mode.
        verdict: "above 1", "not above 1" (with ", inside the noise"), "not valid" or "".
        mark: A mark of the torch-mps row, for example an eager fallback.
        reason: Why a cell is empty or not valid.
        parity: The parity of image 0 of a quantized row.
    """

    model_size: str
    side: int
    batch_size: int
    output: str
    dtype: str
    bits: int | None = None
    mlx_mode: str | None = None
    mlx_ips: float | None = None
    torch_mode: str | None = None
    torch_ips: float | None = None
    torch_compile: bool | None = None
    ratio: float | None = None
    unquantized_ratio: float | None = None
    verdict: str = ""
    mark: str = ""
    reason: str = ""
    parity: dict | None = None


def _fastest(lines: list[dict], output: str) -> dict | None:
    usable = [line for line in lines if _usable(line)]
    return max(usable, key=lambda line: _ips(line, output)) if usable else None


def _row(
    cell: tuple[str, int, int], output: str, dtype: str, mlx: list[dict], torch: list[dict]
) -> SpeedupRow:
    """The main row of one cell, output and dtype from the candidate lines of both sides.

    Both sides come from the group (phase) of the fastest torch-mps row, so the ratio pairs rows
    of one job (finding V10 of specs/06-speed/verify.html). A group without a usable mlx row of
    the dtype takes the fastest mlx row of the cell.
    """
    model_size, side, batch_size = cell
    if dtype == "fp32" and model_size in NO_FP32_WEIGHTS:
        return SpeedupRow(model_size, side, batch_size, output, dtype, reason=NO_FP32_REASON)
    best_torch = _fastest(torch, output)
    phase = best_torch.get("phase", "matrix") if best_torch else None
    same_group = [line for line in mlx if line.get("phase", "matrix") == phase]
    best_mlx = _fastest(same_group, output) or _fastest(mlx, output)
    if best_mlx is None or best_torch is None:
        reasons = [_reasons(side_lines) for side_lines in (mlx, torch)]
        reason = "; ".join(r for r in reasons if r) or "not measured"
        return SpeedupRow(model_size, side, batch_size, output, dtype, reason=reason)
    ratio = _ips(best_mlx, output) / _ips(best_torch, output)
    return SpeedupRow(
        model_size,
        side,
        batch_size,
        output,
        dtype,
        mlx_mode=best_mlx["dtype_mode"],
        mlx_ips=_ips(best_mlx, output),
        torch_mode=best_torch["dtype_mode"],
        torch_ips=_ips(best_torch, output),
        torch_compile=bool(best_torch["compile"]),
        ratio=ratio,
        verdict=single_verdict(ratio),
        mark=best_torch.get("mark", ""),
    )


def _parity_text(parity: dict | None) -> str:
    if not parity:
        return "no parity"
    return (
        f"frame {parity['frame']:.6g}, patch mean {parity['patch_mean']:.6g}, "
        f"patch 1st percentile {parity['patch_p01']:.6g}"
    )


def _quantized_rows(main: SpeedupRow, mlx: list[dict]) -> list[SpeedupRow]:
    """The quantized rows of one main row: one for each bits and dtype mode with a line."""
    rows = []
    for bits in QUANTIZED_BITS:
        for mode in QUANTIZED_MODES:
            if DTYPE_OF_MODE[mode] != main.dtype:
                continue
            found = [
                line for line in mlx if line.get("bits") == bits and line["dtype_mode"] == mode
            ]
            if not found:
                continue
            line = found[-1]
            base = {
                "model_size": main.model_size,
                "side": main.side,
                "batch_size": main.batch_size,
                "output": main.output,
                "dtype": main.dtype,
                "bits": bits,
                "mlx_mode": mode,
                "torch_mode": main.torch_mode,
                "torch_ips": main.torch_ips,
                "torch_compile": main.torch_compile,
                "parity": line.get("parity"),
            }
            if line.get("status") != "done":
                rows.append(SpeedupRow(**base, reason=f"{line['status']}, {line['reason']}"))
                continue
            ips = _ips(line, main.output)
            unquantized = [
                other
                for other in mlx
                if other.get("bits") is None and other["dtype_mode"] == mode and _usable(other)
            ]
            unquantized_ratio = ips / _ips(unquantized[-1], main.output) if unquantized else None
            ratio = ips / main.torch_ips if main.torch_ips else None
            if not line.get("valid"):
                reason = f"fails THRESHOLDS[{mode}]: {_parity_text(line.get('parity'))}"
                verdict = "not valid"
            else:
                reason = "" if ratio is not None else main.reason
                verdict = single_verdict(ratio) if ratio is not None else ""
            rows.append(
                SpeedupRow(
                    **base,
                    mlx_ips=ips,
                    ratio=ratio,
                    unquantized_ratio=unquantized_ratio,
                    verdict=verdict,
                    reason=reason,
                    mark=main.mark,
                )
            )
    return rows


def _by_cell(lines: list[dict]) -> dict[tuple[str, int, int], list[dict]]:
    cells: dict[tuple[str, int, int], list[dict]] = {}
    for line in _latest(lines):
        cells.setdefault(_cell(line), []).append(line)
    return cells


def _cell_rows(cell: tuple[str, int, int], group: list[dict], quantized: bool) -> list[SpeedupRow]:
    rows = []
    for output in OUTPUTS:
        for dtype in DTYPES:
            same = [line for line in group if DTYPE_OF_MODE[line["dtype_mode"]] == dtype]
            mlx = [line for line in same if line["backend"] == "mlx"]
            torch = [line for line in same if line["backend"] == "torch-mps"]
            main = _row(cell, output, dtype, [line for line in mlx if not line.get("bits")], torch)
            rows.append(main)
            if quantized:
                rows += _quantized_rows(main, mlx)
    return rows


def speedup_rows(lines: list[dict]) -> list[SpeedupRow]:
    """Return the main rows and the quantized rows of each cell that has a line.

    ADR 0004 on both sides: the mlx side takes the fastest valid unquantized mode of the dtype;
    the torch-mps side the fastest valid row of the dtype, from the matrix or from a follow-up
    group of the compile check. Quantized rows never join the choice (D22).

    Args:
        lines: The lines of results/speed-matrix.jsonl. The last line of each (phase, backend,
            key) counts, so a rerun of a skipped worker replaces its skip line.
    """
    rows = []
    for cell, group in _by_cell(lines).items():
        rows += _cell_rows(cell, group, quantized=True)
    return rows


def matrix_cells(lines: list[dict]) -> list[SpeedupRow]:
    """Return the 324 main cells of the matrix (6 model sizes × 3 input resolutions × 3 batch
    sizes × 2 outputs × 3 dtypes). A cell without lines says "not measured"."""
    cells = _by_cell(lines)
    rows = []
    for model_size in MODEL_SIZES:
        for side in SIDES:
            for batch_size in BATCH_SIZES:
                cell = (model_size, side, batch_size)
                rows += _cell_rows(cell, cells.get(cell, []), quantized=False)
    return rows


@dataclass(frozen=True)
class PairRow:
    """The speedup of one priority configuration, dtype and output from the paired runs (D21).

    Attributes:
        run_medians: The median of the paired ratios of each run, in the order of the run ids.
        pooled_median: The median of all paired ratios of all runs.
        spread: The spread between runs, or None for one run.
        ratios: The number of paired ratios.
    """

    priority: str
    dtype: str
    output: str
    mlx_mode: str | None
    torch_mode: str | None
    torch_compile: bool | None
    run_medians: list[float] = field(default_factory=list)
    pooled_median: float | None = None
    spread: float | None = None
    verdict: str = ""
    ratios: int = 0
    run_ids: list[str] = field(default_factory=list)


def _mode_by_median(lines: list[dict], output: str) -> str | None:
    """The fastest valid mode by the lower median over all its lines (ADR 0004)."""
    by_mode: dict[str, list[float]] = {}
    for line in lines:
        if _usable(line):
            by_mode.setdefault(line["dtype_mode"], []).append(_ips(line, output))
    if not by_mode:
        return None
    return max(by_mode, key=lambda mode: _lower_median(by_mode[mode]))


def pair_rows(lines: list[dict]) -> list[PairRow]:
    """Return the 12 verdict rows of the paired runs at P1 and P2 (Section 3.11 of the plan).

    For each priority configuration, dtype and output: the mode on each side by ADR 0004 over all
    rounds of all runs, one paired ratio for each run and round, the run medians, the pooled
    median, the spread between runs and the verdict of D21.

    Args:
        lines: The lines of results/speed-pairs.jsonl, with run_id, round and priority.
    """
    rows = []
    for priority in ("P1", "P2"):
        mine = [line for line in lines if line.get("priority") == priority]
        for dtype in DTYPES:
            same = [line for line in mine if DTYPE_OF_MODE[line["dtype_mode"]] == dtype]
            for output in OUTPUTS:
                mlx = [line for line in same if line["backend"] == "mlx"]
                torch = [line for line in same if line["backend"] == "torch-mps"]
                mlx_mode, torch_mode = _mode_by_median(mlx, output), _mode_by_median(torch, output)
                if mlx_mode is None or torch_mode is None:
                    rows.append(PairRow(priority, dtype, output, mlx_mode, torch_mode, None))
                    continue
                pick = {}
                for line in mlx + torch:
                    mode = mlx_mode if line["backend"] == "mlx" else torch_mode
                    if line["dtype_mode"] == mode and _usable(line):
                        pick[(str(line["run_id"]), line["round"], line["backend"])] = line
                by_run: dict[str, list[float]] = {}
                for (run_id, round_, backend), line in sorted(pick.items()):
                    other = pick.get((run_id, round_, "torch-mps"))
                    if backend == "mlx" and other is not None:
                        ratio = _ips(line, output) / _ips(other, output)
                        by_run.setdefault(run_id, []).append(ratio)
                run_ids = sorted(by_run)
                medians = [_lower_median(by_run[run_id]) for run_id in run_ids]
                pooled = _lower_median([r for run_id in run_ids for r in by_run[run_id]])
                compiled = {line["compile"] for line in torch if line["dtype_mode"] == torch_mode}
                rows.append(
                    PairRow(
                        priority,
                        dtype,
                        output,
                        mlx_mode,
                        torch_mode,
                        all(compiled),
                        medians,
                        pooled,
                        spread_between_runs(medians) if len(medians) > 1 else None,
                        pass_verdict(medians, pooled),
                        sum(len(values) for values in by_run.values()),
                        run_ids,
                    )
                )
    return rows


def autocast_over_cast(lines: list[dict]) -> dict[tuple[str, str, str], float]:
    """Return the median over the rounds of the paired runs of autocast / cast on mlx (C3, R30),
    for each priority configuration, dtype and output."""
    found: dict[tuple[str, str, str], float] = {}
    for priority in ("P1", "P2"):
        for dtype in ("fp16", "bf16"):
            for output in OUTPUTS:
                pick: dict[tuple[str, int, str], float] = {}
                for line in lines:
                    if line.get("priority") != priority or line["backend"] != "mlx":
                        continue
                    if line["dtype_mode"] in (f"{dtype}-cast", f"{dtype}-autocast") and _usable(
                        line
                    ):
                        key = (str(line["run_id"]), line["round"], line["dtype_mode"])
                        pick[key] = _ips(line, output)
                ratios = [
                    value / pick[(run_id, round_, f"{dtype}-cast")]
                    for (run_id, round_, mode), value in pick.items()
                    if mode.endswith("-autocast") and (run_id, round_, f"{dtype}-cast") in pick
                ]
                if ratios:
                    found[(priority, dtype, output)] = _lower_median(ratios)
    return found


def compile_check(
    matrix_lines: list[dict], step3_lines: list[dict]
) -> list[tuple[str, int, int, str]]:
    """Return each (model size, input resolution, batch size, dtype mode) whose compiled torch-mps
    row of the matrix is below the Step 3 eager median of the same mode, for the frame or the
    patch output (Section 3.8 of the plan). Job J16 runs a follow-up group for each.

    Args:
        matrix_lines: The lines of results/speed-matrix.jsonl.
        step3_lines: The lines of results/baseline-torch-mps.jsonl.
    """
    eager: dict[tuple, list[dict]] = {}
    for line in step3_lines:
        if (
            line.get("backend", "torch-mps") == "torch-mps"
            and not line["compile"]
            and _usable(line)
        ):
            eager.setdefault((*_cell(line), line["dtype_mode"]), []).append(line)
    listed = set()
    for line in _latest(matrix_lines):
        if line["backend"] != "torch-mps" or not line["compile"] or not _usable(line):
            continue
        if line.get("phase", "matrix") != "matrix":
            continue
        key = (*_cell(line), line["dtype_mode"])
        if key not in eager:
            continue
        for output in OUTPUTS:
            median = _lower_median([_ips(other, output) for other in eager[key]])
            if _ips(line, output) < median:
                listed.add(key)
    order = {size: index for index, size in enumerate(MODEL_SIZES)}
    return sorted(listed, key=lambda key: (order[key[0]], *key[1:]))
