"""Tests of scripts/bench_streams.py: the probe D2 C of specs/07b-neural-engine-backend/explore.html
(two threads in one process against two processes)."""

import bench_streams
import pytest


def case_line(round_: int, case: str, ips: dict[str, float]) -> dict:
    """Return a done line of one case with the throughput of each side."""
    return {
        "round": round_,
        "case": case,
        "status": "done",
        "sides": {side: {"throughput_ips": value} for side, value in ips.items()},
    }


def test_schedule_rotates_the_cases() -> None:
    schedule = bench_streams.schedule(3)
    assert schedule[0] == bench_streams.CASES
    assert schedule[1] == bench_streams.CASES[1:] + bench_streams.CASES[:1]
    assert all(sorted(cases) == sorted(bench_streams.CASES) for cases in schedule)


def test_stream_ratios_divide_by_mlx_alone_of_the_round() -> None:
    lines = [
        case_line(0, "mlx", {"mlx": 1000.0}),
        case_line(0, "threads", {"mlx": 900.0, "ne": 400.0}),
        case_line(0, "processes", {"mlx": 950.0, "ne": 400.0}),
        case_line(1, "mlx", {"mlx": 500.0}),
        case_line(1, "threads", {"mlx": 500.0, "ne": 100.0}),
        case_line(1, "processes", {"mlx": 500.0, "ne": 150.0}),
    ]
    ratios = bench_streams.stream_ratios(lines)
    assert ratios["threads"] == pytest.approx([1.3, 1.2])
    assert ratios["processes"] == pytest.approx([1.35, 1.3])


def test_stream_ratios_skip_a_round_without_mlx_alone() -> None:
    lines = [
        case_line(0, "threads", {"mlx": 900.0, "ne": 400.0}),
        {"round": 0, "case": "mlx", "status": "error"},
    ]
    assert bench_streams.stream_ratios(lines) == {"threads": [], "processes": []}


@pytest.mark.parametrize(
    ("threads", "processes", "choice"),
    [([1.30, 1.28, 1.29], [1.32, 1.30, 1.31], "threads"), ([1.10], [1.30], "process")],
)
def test_choice_takes_threads_at_095_of_processes(
    threads: list[float], processes: list[float], choice: str
) -> None:
    assert bench_streams.choice({"threads": threads, "processes": processes}) == choice


def test_choice_without_ratios_raises() -> None:
    with pytest.raises(ValueError, match="no stream ratio"):
        bench_streams.choice({"threads": [], "processes": [1.3]})
