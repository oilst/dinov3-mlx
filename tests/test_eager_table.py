"""Tests of scripts/eager_table.py: the paired ratios of compiled mlx against eager torch-mps
(W11 of specs/07-release/plan.html)."""

import eager_table


def bench_line(backend: str, run_id: str, round_: int, ips: float) -> dict:
    """Return a done, valid bench line of P1 in fp32 with one throughput for both outputs."""
    return {
        "backend": backend,
        "compile": backend == "mlx",
        "dtype_mode": "fp32",
        "priority": "P1",
        "run_id": run_id,
        "round": round_,
        "status": "done",
        "valid": True,
        "frame": {"throughput_ips": ips},
        "patch": {"throughput_ips": ips},
    }


def test_eager_table_gives_the_ratio_without_compile_mark() -> None:
    """The torch-mps mode has no "(compile)" mark; the pooled median is the lower median."""
    lines = [
        line
        for run_id in ("1", "2")
        for r, mlx_ips in enumerate((300.0, 200.0), start=1)
        for line in (
            bench_line("mlx", run_id, r, mlx_ips),
            bench_line("torch-mps", run_id, r, 100.0),
        )
    ]
    text = eager_table.write_table(lines)
    row = next(line for line in text.splitlines() if line.startswith("| P1 | fp32 | frame"))
    assert "(compile)" not in row
    assert "| 2.000 | 2.000 | 2.000 |" in row  # run 1, run 2, pooled: lower medians of 3 and 2
    assert "Source: `results/speed-pairs-eager.jsonl`" in text
