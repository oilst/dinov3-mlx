# Overhead of the public API

Made by `scripts/bench_api.py` on 2026-11-09 from `results/api-overhead.jsonl`. Benchmark B1 of `specs/07-release/plan.html`: `Embedder` against the direct path (preprocess_frames, the compiled forward pass, the cast of the output to the host) on the same uint8 batch of 1280 × 720 px frames, fp16-autocast. Ratio = api ÷ direct throughput of one round; the result is the lower median of 5 rounds (ADR 0004). The absolute throughput is report only (V2 of Step 6).

| Configuration | Model size | Input resolution | Batch size | Output | Frames per call | api img/s (median) | direct img/s (median) | Round ratios | Lower median | Target | Result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P1 | ViT-S/16 | 224 | 8 | frame | 8 | 1196.4 | 1192.0 | 1.069, 1.009, 0.996, 0.958, 0.995 | 0.996 | ≥ 0.97 | pass |
| P2 | ViT-H+/16 | 512 | 8 | patch | 8 | 16.6 | 16.0 | 1.009, 0.961, 1.040, 0.960, 1.023 | 1.009 | ≥ 0.97 | pass |
| P1-pad | ViT-S/16 | 224 | 8 | frame | 13 | 743.1 | 913.3 | 0.820, 0.819, 0.814, 0.828, 0.804 | 0.819 | report only | report only |

Table 1. The API overhead (AC9). P1-pad calls the API with 13 frames, so the second batch has 3 padded rows; its direct form runs 8 + 5 frames without padding.
