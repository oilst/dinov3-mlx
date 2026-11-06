from baseline_table import write_baseline_table, write_profile_table

from dinov3_apple.flops import PART_NAMES

MACHINE = {
    "power_source": "AC Power",
    "low_power_mode": "not reported",
    "thermal": "No thermal warning level has been recorded",
    "available_bytes": 11_000_000_000,
    "chip": "Apple M5 Max",
    "memory_bytes": 68_719_476_736,
    "versions": {"macos": "26.5.2", "torch": "2.15.0", "transformers": "5.22.1"},
}


def _line(dtype_mode: str, throughput: float, **fields: object) -> dict:
    line = {
        "run": "B1",
        "model_size": "ViT-S/16",
        "backend": "torch-mps",
        "dtype_mode": dtype_mode,
        "side": 224,
        "batch_size": 8,
        "compile": False,
        "repeat": 0,
        "status": "done",
        "reason": "",
        "valid": True,
        "validity_note": "parity passes",
        "frame": {"throughput_ips": throughput, "median_s": 0.01, "p90_s": 0.011},
        "patch": {"throughput_ips": throughput / 2, "median_s": 0.02, "p90_s": 0.021},
        "device": {"throughput_ips": throughput, "median_s": 0.01, "p90_s": 0.011},
        "timed_batches": 1000,
        "timed_s": 10.0,
        "warm_up_batches": 200,
        "warm_up_s": 2.0,
        "need_bytes": 6_000_000_000,
        "available_bytes": 11_000_000_000,
        "peak_bytes": 1_000_000_000,
        "load_peak_bytes": 900_000_000,
        "machine": MACHINE,
        "time": "2026-11-06T18:00:00+00:00",
    }
    return line | fields


def test_baseline_table_from_lines() -> None:
    lines = [
        _line("fp32", 100.0),
        _line("fp16-cast", 300.0),
        _line("fp16-autocast", 200.0),
        _line("bf16-cast", 150.0, valid=False, validity_note="non-finite output"),
        _line("bf16-autocast", 250.0),
        _line(
            "fp32",
            0.0,
            side=512,
            batch_size=32,
            status="skipped",
            reason="needs 6.7 GB, 6.4 GB available",
            need_bytes=6_700_000_000,
            available_bytes=6_400_000_000,
        ),
        _line("fp16-cast", 0.0, model_size="ViT-7B/16", status="deferred", reason="D8"),
    ]
    text = write_baseline_table(lines)
    assert "| ViT-S/16 | 224 | 8 | fp16 | fp16-cast | 300.0 |" in text
    assert "| ViT-S/16 | 224 | 8 | bf16 | bf16-autocast | 250.0 |" in text
    assert "## Skipped, deferred and error lines" in text
    assert "needs 6.7 GB, 6.4 GB available" in text
    assert "6.7" in text and "6.4" in text
    assert "deferred" in text
    assert "AC Power" in text


def test_profile_table_from_lines() -> None:
    parts = {name: 1e-4 for name in PART_NAMES}
    line = {
        "model_size": "ViT-S/16",
        "dtype_mode": "fp32",
        "side": 224,
        "batch_size": 1,
        "parts_s": parts,
        "part_flops": {name: 1e6 for name in PART_NAMES},
        "block_s": 1.05e-3,
        "outer_s": {"patch_embedding": 1e-4, "rope_tables": 1e-4, "final_norm": 1e-5},
        "forward_s": 1.4e-2,
        "blocks": 12,
    }
    text = write_profile_table([line])
    assert "sdpa" in text
    assert "Sum of parts / block" in text
    assert "1.048" in text  # 11 parts x 0.1 ms / 1.05 ms
