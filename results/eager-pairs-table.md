# Speedup of compiled mlx against eager torch-mps

W11 of `specs/07-release/plan.html`. Ratio = mlx throughput ÷ torch-mps throughput in the
same round of a paired run. The mlx side runs the compiled forward pass; the torch-mps
side runs eager, as Hugging Face transformers does without torch.compile. Each side takes
its fastest valid dtype mode of each dtype (ADR 0004). P1 = ViT-S/16, 224 px; P2 =
ViT-H+/16, 512 px; batch size 8.

| Priority | Dtype | Output | mlx mode | torch-mps mode | Run 1 median | Run 2 median | Pooled median | Spread between runs | Verdict (D21) | Ratios |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P1 | fp32 | frame | fp32 | fp32 | 1.392 | 1.394 | 1.392 | 0.002 | above 1 | 10 |
| P1 | fp32 | patch | fp32 | fp32 | 1.395 | 1.397 | 1.395 | 0.001 | above 1 | 10 |
| P1 | fp16 | frame | fp16-autocast | fp16-cast | 1.671 | 1.642 | 1.642 | 0.018 | above 1 | 10 |
| P1 | fp16 | patch | fp16-cast | fp16-cast | 1.683 | 1.681 | 1.681 | 0.002 | above 1 | 10 |
| P1 | bf16 | frame | bf16-cast | bf16-cast | 1.684 | 1.644 | 1.646 | 0.024 | above 1 | 10 |
| P1 | bf16 | patch | bf16-cast | bf16-cast | 1.692 | 1.652 | 1.654 | 0.024 | above 1 | 10 |
| P2 | fp32 | frame | fp32 | fp32 | 1.171 | 1.173 | 1.171 | 0.001 | above 1 | 10 |
| P2 | fp32 | patch | fp32 | fp32 | 1.171 | 1.172 | 1.171 | 0.001 | above 1 | 10 |
| P2 | fp16 | frame | fp16-cast | fp16-cast | 1.277 | 1.276 | 1.276 | 0.001 | above 1 | 10 |
| P2 | fp16 | patch | fp16-cast | fp16-cast | 1.278 | 1.276 | 1.276 | 0.001 | above 1 | 10 |
| P2 | bf16 | frame | bf16-cast | bf16-cast | 1.268 | 1.280 | 1.276 | 0.009 | above 1 | 10 |
| P2 | bf16 | patch | bf16-cast | bf16-cast | 1.269 | 1.280 | 1.276 | 0.009 | above 1 | 10 |

Source: `results/speed-pairs-eager.jsonl` (`scripts/bench_speed.py pairs --torch-eager`).
