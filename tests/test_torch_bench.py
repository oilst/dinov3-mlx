from pathlib import Path

import pytest
import torch

from dinov3_apple.bench import LINE_FIELDS, BenchConfig, TimeBudget, result_line
from dinov3_apple.hub import WeightsNotFoundError
from dinov3_apple.machine import read_machine_state
from dinov3_apple.memory import InsufficientMemoryError
from dinov3_apple.reference import INPUTS_FILE, reference_path

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no torch-mps")

SHORT = TimeBudget(min_batches=3, min_seconds=0.0)


def test_worker_writes_all_fields(tmp_path: Path) -> None:
    from dinov3_apple.torch_bench import run_worker

    if not INPUTS_FILE.exists() or not reference_path("ViT-S/16").exists():
        pytest.skip("stored pixels or the ViT-S/16 reference file are missing")
    config = BenchConfig("B1", "ViT-S/16", "fp32", 224, 1)
    try:
        worker = run_worker(config, warm_up=SHORT, timed=SHORT, bench_dir=tmp_path)
    except (WeightsNotFoundError, InsufficientMemoryError) as error:
        pytest.skip(str(error))
    assert worker["status"] == "done", worker.get("reason")
    line = result_line(
        config, worker, need_bytes=1, available_bytes=2, machine=read_machine_state()
    )
    assert set(LINE_FIELDS) <= set(line)
    assert line["valid"] is True
    assert line["parity"]["frame"] > 0.9999
    assert line["timed_batches"] >= 3
    assert line["warm_up_batches"] >= 3
    assert line["output_dtype"] == "float32"
    for output in ("frame", "patch", "device"):
        assert line[output]["throughput_ips"] > 0
        assert line[output]["median_s"] <= line[output]["p90_s"]
    assert line["patch"]["median_s"] >= line["device"]["median_s"]
    assert line["peak_bytes"] >= line["load_peak_bytes"] >= line["footprint_base_bytes"]
    assert line["run_peak_bytes"] is None or line["run_peak_bytes"] == line["peak_bytes"]
    assert line["mps_driver_max_bytes"] > 0
    assert line["attn_implementation"] == "sdpa"
    assert len(list(tmp_path.glob("*.json"))) == 1


def test_worker_refuses_fp32_weights_for_7b() -> None:
    from dinov3_apple.torch_bench import run_worker

    with pytest.raises(ValueError, match="ViT-7B/16"):
        run_worker(BenchConfig("B4", "ViT-7B/16", "fp16-autocast", 224, 1))
