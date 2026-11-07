# Baseline: Hugging Face backbone on torch-mps

Written by `scripts/baseline_table.py` on 2026-11-07 from `results/baseline-torch-mps.jsonl` (273 lines, 2026-11-06T21:43:26+00:00 to 2026-11-06T23:26:49+00:00). Spec: `specs/03-baseline/plan.html`. The latest line of each configuration counts.

Throughput is images per second, host to host (D3). Peak memory is the lifetime maximum of the macOS process footprint of the worker (D5). Latency is the median and the p90 of the batch times.

Repeatability: over 3 to 3 runs, the throughput of P1 and P2 spreads by 0.4 to 23.2 % (Section 4, limit 5 %). The fastest and the slowest run of one configuration differ by up to 1.3x. So one run can be far off on this Mac (finding F1 of `specs/03-baseline/verify.html`). A baseline row with 1 run is one sample.

## Checks

- AC1: 243 of 243 configurations of B1 and B4 have a line: {'done': 225, 'deferred': 18}.
- AC3: 12 done lines of B1 and B4 are not valid. Not expected: none. Expected but valid or not run: none.
- AC4: 0 done lines miss the time budget of Table 5.
- AC7: 0 done lines have a peak memory above the need. Largest peak / need: 0.61.

## 1. Baseline rows: frame embedding

The fastest valid dtype mode of each dtype (D2, Section 3.7 of the plan). Each dtype mode counts with the median of its valid runs; Runs gives their number (finding F1 of `specs/03-baseline/verify.html`). Compile rows count since the decision of the user on 2026-11-06; only P1 and P2 have them.

| Model size | Input resolution | Batch size | Dtype | Dtype mode | Runs | Throughput (img/s) | Median (ms) | p90 (ms) | Peak memory (GB) | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ViT-S/16 | 224 | 1 | fp32 | fp32 | 1 | 130.7 | 7.99 | 9.59 | 0.57 | parity passes |
| ViT-S/16 | 224 | 1 | fp16 | fp16-cast | 1 | 129.8 | 8.01 | 9.35 | 0.56 | parity passes |
| ViT-S/16 | 224 | 1 | bf16 | bf16-cast | 1 | 153.3 | 5.42 | 9.22 | 0.56 | parity passes |
| ViT-S/16 | 224 | 8 | fp32 | fp32 + compile | 1 | 642.3 | 13.02 | 13.73 | 0.77 | parity passes |
| ViT-S/16 | 224 | 8 | fp16 | fp16-cast + compile | 1 | 1860.0 | 4.49 | 4.74 | 0.72 | parity passes |
| ViT-S/16 | 224 | 8 | bf16 | bf16-cast + compile | 1 | 1848.4 | 4.56 | 4.78 | 0.73 | parity passes |
| ViT-S/16 | 224 | 32 | fp32 | fp32 | 1 | 765.5 | 41.30 | 43.38 | 1.69 | parity passes |
| ViT-S/16 | 224 | 32 | fp16 | fp16-cast | 1 | 1275.3 | 24.97 | 25.96 | 1.74 | parity passes |
| ViT-S/16 | 224 | 32 | bf16 | bf16-cast | 1 | 1243.2 | 25.67 | 26.62 | 1.74 | parity passes |
| ViT-S/16 | 448 | 1 | fp32 | fp32 | 1 | 104.0 | 9.71 | 10.07 | 0.60 | parity passes |
| ViT-S/16 | 448 | 1 | fp16 | fp16-cast | 1 | 107.4 | 9.47 | 10.17 | 0.60 | parity passes |
| ViT-S/16 | 448 | 1 | bf16 | bf16-cast | 1 | 110.0 | 9.37 | 10.10 | 0.60 | parity passes |
| ViT-S/16 | 448 | 8 | fp32 | fp32 | 1 | 158.5 | 49.98 | 52.78 | 1.69 | parity passes |
| ViT-S/16 | 448 | 8 | fp16 | fp16-cast | 1 | 303.9 | 26.15 | 27.44 | 1.74 | parity passes |
| ViT-S/16 | 448 | 8 | bf16 | bf16-cast | 1 | 301.9 | 26.48 | 27.61 | 1.74 | parity passes |
| ViT-S/16 | 448 | 32 | fp32 | fp32 | 1 | 159.7 | 199.00 | 215.52 | 1.83 | parity passes |
| ViT-S/16 | 448 | 32 | fp16 | fp16-cast | 1 | 298.2 | 109.79 | 112.40 | 1.80 | parity passes |
| ViT-S/16 | 448 | 32 | bf16 | bf16-cast | 1 | 300.2 | 108.86 | 112.26 | 1.80 | parity passes |
| ViT-S/16 | 512 | 1 | fp32 | fp32 | 1 | 82.2 | 12.12 | 12.66 | 0.61 | parity passes |
| ViT-S/16 | 512 | 1 | fp16 | fp16-autocast | 1 | 115.3 | 8.68 | 9.00 | 0.61 | parity passes |
| ViT-S/16 | 512 | 1 | bf16 | bf16-cast | 1 | 109.2 | 9.47 | 10.27 | 0.61 | parity passes |
| ViT-S/16 | 512 | 8 | fp32 | fp32 | 1 | 112.9 | 68.84 | 77.73 | 1.70 | parity passes |
| ViT-S/16 | 512 | 8 | fp16 | fp16-cast | 1 | 221.4 | 35.92 | 37.72 | 1.68 | parity passes |
| ViT-S/16 | 512 | 8 | bf16 | bf16-cast | 1 | 228.8 | 34.91 | 36.51 | 1.68 | parity passes |
| ViT-S/16 | 512 | 32 | fp32 | fp32 | 1 | 96.2 | 319.92 | 372.13 | 3.04 | parity passes |
| ViT-S/16 | 512 | 32 | fp16 | fp16-cast | 1 | 212.6 | 151.52 | 158.77 | 1.86 | parity passes |
| ViT-S/16 | 512 | 32 | bf16 | bf16-cast | 1 | 223.3 | 144.82 | 154.24 | 1.85 | parity passes |
| ViT-S+/16 | 224 | 1 | fp32 | fp32 | 1 | 113.3 | 9.08 | 10.20 | 0.60 | parity passes |
| ViT-S+/16 | 224 | 1 | fp16 | fp16-cast | 1 | 121.6 | 8.51 | 10.14 | 0.61 | parity passes |
| ViT-S+/16 | 224 | 1 | bf16 | bf16-cast | 1 | 122.3 | 8.54 | 10.23 | 0.61 | parity passes |
| ViT-S+/16 | 224 | 8 | fp32 | fp32 | 1 | 414.5 | 19.21 | 20.25 | 0.67 | parity passes |
| ViT-S+/16 | 224 | 8 | fp16 | fp16-cast | 1 | 760.2 | 10.50 | 11.10 | 0.64 | parity passes |
| ViT-S+/16 | 224 | 8 | bf16 | bf16-cast | 1 | 757.1 | 10.49 | 11.23 | 0.64 | parity passes |
| ViT-S+/16 | 224 | 32 | fp32 | fp32 | 1 | 464.3 | 70.00 | 73.18 | 1.72 | parity passes |
| ViT-S+/16 | 224 | 32 | fp16 | fp16-cast | 1 | 937.2 | 34.54 | 36.12 | 1.78 | parity passes |
| ViT-S+/16 | 224 | 32 | bf16 | bf16-cast | 1 | 946.3 | 34.20 | 35.68 | 1.78 | parity passes |
| ViT-S+/16 | 448 | 1 | fp32 | fp32 | 1 | 83.7 | 12.00 | 12.78 | 0.64 | parity passes |
| ViT-S+/16 | 448 | 1 | fp16 | fp16-cast | 1 | 106.6 | 9.60 | 10.45 | 0.61 | parity passes |
| ViT-S+/16 | 448 | 1 | bf16 | bf16-cast | 1 | 102.0 | 9.89 | 10.63 | 0.61 | parity passes |
| ViT-S+/16 | 448 | 8 | fp32 | fp32 | 1 | 104.5 | 78.38 | 80.93 | 1.72 | parity passes |
| ViT-S+/16 | 448 | 8 | fp16 | fp16-cast | 1 | 281.6 | 29.33 | 32.90 | 1.78 | parity passes |
| ViT-S+/16 | 448 | 8 | bf16 | bf16-cast | 1 | 233.7 | 34.24 | 35.41 | 1.78 | parity passes |
| ViT-S+/16 | 448 | 32 | fp32 | fp32 | 1 | 94.6 | 340.54 | 353.75 | 1.86 | parity passes |
| ViT-S+/16 | 448 | 32 | fp16 | fp16-cast | 1 | 259.2 | 127.14 | 128.51 | 1.85 | parity passes |
| ViT-S+/16 | 448 | 32 | bf16 | bf16-cast | 1 | 224.0 | 143.68 | 155.63 | 1.85 | parity passes |
| ViT-S+/16 | 512 | 1 | fp32 | fp32 | 1 | 63.0 | 16.25 | 17.00 | 0.64 | parity passes |
| ViT-S+/16 | 512 | 1 | fp16 | fp16-cast | 1 | 101.4 | 9.95 | 10.56 | 0.62 | parity passes |
| ViT-S+/16 | 512 | 1 | bf16 | bf16-cast | 1 | 100.1 | 10.02 | 10.62 | 0.62 | parity passes |
| ViT-S+/16 | 512 | 8 | fp32 | fp32 | 1 | 70.7 | 114.87 | 121.96 | 1.74 | parity passes |
| ViT-S+/16 | 512 | 8 | fp16 | fp16-cast | 1 | 196.8 | 41.63 | 48.93 | 1.80 | parity passes |
| ViT-S+/16 | 512 | 8 | bf16 | bf16-cast | 1 | 178.5 | 44.83 | 46.34 | 1.80 | parity passes |
| ViT-S+/16 | 512 | 32 | fp32 | fp32 | 1 | 58.4 | 544.47 | 570.45 | 3.08 | parity passes |
| ViT-S+/16 | 512 | 32 | fp16 | fp16-cast | 1 | 155.9 | 209.61 | 216.67 | 1.90 | parity passes |
| ViT-S+/16 | 512 | 32 | bf16 | bf16-cast | 1 | 171.8 | 185.84 | 199.69 | 1.90 | parity passes |
| ViT-B/16 | 224 | 1 | fp32 | fp32 | 1 | 103.6 | 9.62 | 9.97 | 0.85 | parity passes |
| ViT-B/16 | 224 | 1 | fp16 | fp16-cast | 1 | 107.9 | 9.48 | 10.00 | 0.82 | parity passes |
| ViT-B/16 | 224 | 1 | bf16 | bf16-cast | 1 | 107.6 | 9.59 | 10.06 | 0.82 | parity passes |
| ViT-B/16 | 224 | 8 | fp32 | fp32 | 1 | 197.7 | 41.04 | 46.15 | 2.00 | parity passes |
| ViT-B/16 | 224 | 8 | fp16 | fp16-cast | 1 | 396.3 | 19.98 | 21.39 | 0.86 | parity passes |
| ViT-B/16 | 224 | 8 | bf16 | bf16-cast | 1 | 415.4 | 18.93 | 20.67 | 0.86 | parity passes |
| ViT-B/16 | 224 | 32 | fp32 | fp32 | 1 | 178.1 | 182.82 | 190.67 | 1.99 | parity passes |
| ViT-B/16 | 224 | 32 | fp16 | fp16-cast | 1 | 392.1 | 82.50 | 86.19 | 2.08 | parity passes |
| ViT-B/16 | 224 | 32 | bf16 | bf16-cast | 1 | 435.7 | 73.14 | 77.45 | 2.08 | parity passes |
| ViT-B/16 | 448 | 1 | fp32 | fp32 | 1 | 39.5 | 25.47 | 26.29 | 0.89 | parity passes |
| ViT-B/16 | 448 | 1 | fp16 | fp16-cast | 1 | 92.6 | 10.99 | 11.88 | 0.84 | parity passes |
| ViT-B/16 | 448 | 1 | bf16 | bf16-cast | 1 | 96.6 | 10.09 | 11.37 | 0.84 | parity passes |
| ViT-B/16 | 448 | 8 | fp32 | fp32 | 1 | 39.3 | 205.52 | 211.58 | 2.00 | parity passes |
| ViT-B/16 | 448 | 8 | fp16 | fp16-cast | 1 | 100.5 | 79.53 | 84.78 | 2.06 | parity passes |
| ViT-B/16 | 448 | 8 | bf16 | bf16-cast | 1 | 106.5 | 75.48 | 79.71 | 2.06 | parity passes |
| ViT-B/16 | 448 | 32 | fp32 | fp32 | 1 | 35.5 | 897.10 | 938.21 | 3.30 | parity passes |
| ViT-B/16 | 448 | 32 | fp16 | fp16-cast | 1 | 89.2 | 359.13 | 378.12 | 2.10 | parity passes |
| ViT-B/16 | 448 | 32 | bf16 | bf16-cast | 1 | 96.6 | 336.25 | 354.54 | 2.10 | parity passes |
| ViT-B/16 | 512 | 1 | fp32 | fp32 | 1 | 28.1 | 35.97 | 37.21 | 1.97 | parity passes |
| ViT-B/16 | 512 | 1 | fp16 | fp16-cast | 1 | 70.4 | 14.20 | 15.49 | 0.88 | parity passes |
| ViT-B/16 | 512 | 1 | bf16 | bf16-cast | 1 | 73.9 | 13.51 | 14.75 | 0.88 | parity passes |
| ViT-B/16 | 512 | 8 | fp32 | fp32 | 1 | 27.6 | 292.20 | 310.30 | 2.03 | parity passes |
| ViT-B/16 | 512 | 8 | fp16 | fp16-cast | 1 | 74.2 | 108.99 | 113.57 | 1.97 | parity passes |
| ViT-B/16 | 512 | 8 | bf16 | bf16-cast | 1 | 80.3 | 99.44 | 106.18 | 1.97 | parity passes |
| ViT-B/16 | 512 | 32 | fp32 | fp32 | 1 | 23.5 | 1355.87 | 1417.66 | 4.66 | parity passes |
| ViT-B/16 | 512 | 32 | fp16 | fp16-cast | 1 | 63.5 | 505.32 | 541.24 | 3.23 | parity passes |
| ViT-B/16 | 512 | 32 | bf16 | bf16-cast | 1 | 66.9 | 475.66 | 505.15 | 3.23 | parity passes |
| ViT-L/16 | 224 | 1 | fp32 | fp32 | 1 | 38.5 | 26.28 | 28.40 | 2.67 | parity passes |
| ViT-L/16 | 224 | 1 | fp16 | fp16-autocast | 1 | 54.9 | 18.05 | 20.54 | 2.67 | parity passes |
| ViT-L/16 | 224 | 1 | bf16 | bf16-cast | 1 | 81.2 | 12.20 | 12.92 | 1.72 | parity passes |
| ViT-L/16 | 224 | 8 | fp32 | fp32 | 1 | 54.9 | 151.76 | 155.43 | 2.68 | parity passes |
| ViT-L/16 | 224 | 8 | fp16 | fp16-autocast | 1 | 111.5 | 72.14 | 77.71 | 2.68 | parity passes |
| ViT-L/16 | 224 | 8 | bf16 | bf16-cast | 1 | 156.7 | 51.95 | 54.96 | 2.83 | parity passes |
| ViT-L/16 | 224 | 32 | fp32 | fp32 | 1 | 50.8 | 634.36 | 647.10 | 2.75 | parity passes |
| ViT-L/16 | 224 | 32 | fp16 | fp16-autocast | 1 | 122.8 | 262.18 | 266.99 | 2.75 | parity passes |
| ViT-L/16 | 224 | 32 | bf16 | bf16-cast | 1 | 148.2 | 215.81 | 230.88 | 2.86 | parity passes |
| ViT-L/16 | 448 | 1 | fp32 | fp32 | 1 | 11.8 | 86.44 | 88.52 | 2.68 | parity passes |
| ViT-L/16 | 448 | 1 | fp16 | fp16-autocast | 1 | 26.5 | 38.01 | 40.77 | 2.69 | parity passes |
| ViT-L/16 | 448 | 1 | bf16 | bf16-cast | 1 | 34.8 | 28.70 | 31.35 | 1.76 | parity passes |
| ViT-L/16 | 448 | 8 | fp32 | fp32 | 1 | 11.6 | 694.16 | 706.14 | 2.77 | parity passes |
| ViT-L/16 | 448 | 8 | fp16 | fp16-autocast | 1 | 29.6 | 273.30 | 291.08 | 2.78 | parity passes |
| ViT-L/16 | 448 | 8 | bf16 | bf16-cast | 1 | 35.0 | 232.65 | 241.99 | 2.87 | parity passes |
| ViT-L/16 | 448 | 32 | fp32 | fp32 | 1 | 10.4 | 3093.15 | 3135.33 | 5.23 | parity passes |
| ViT-L/16 | 448 | 32 | fp16 | fp16-autocast | 1 | 27.1 | 1157.49 | 1208.83 | 4.15 | parity passes |
| ViT-L/16 | 448 | 32 | bf16 | bf16-cast | 1 | 29.5 | 1079.48 | 1126.51 | 4.15 | parity passes |
| ViT-L/16 | 512 | 1 | fp32 | fp32 | 1 | 8.5 | 119.59 | 123.03 | 2.69 | parity passes |
| ViT-L/16 | 512 | 1 | fp16 | fp16-autocast | 1 | 20.3 | 49.98 | 55.71 | 2.70 | parity passes |
| ViT-L/16 | 512 | 1 | bf16 | bf16-cast | 1 | 26.0 | 38.54 | 41.60 | 1.78 | parity passes |
| ViT-L/16 | 512 | 8 | fp32 | fp32 | 1 | 7.9 | 1009.76 | 1045.33 | 2.81 | parity passes |
| ViT-L/16 | 512 | 8 | fp16 | fp16-autocast | 1 | 19.1 | 420.91 | 452.29 | 2.82 | parity passes |
| ViT-L/16 | 512 | 8 | bf16 | bf16-cast | 1 | 25.1 | 322.42 | 335.25 | 2.89 | parity passes |
| ViT-L/16 | 512 | 32 | fp32 | fp32 | 1 | 7.3 | 4417.04 | 4460.80 | 5.62 | parity passes |
| ViT-L/16 | 512 | 32 | fp16 | fp16-autocast | 1 | 17.1 | 1881.89 | 1938.18 | 5.35 | parity passes |
| ViT-L/16 | 512 | 32 | bf16 | bf16-cast | 1 | 21.0 | 1517.50 | 1579.47 | 4.25 | parity passes |
| ViT-H+/16 | 224 | 1 | fp32 | fp32 | 1 | 16.3 | 62.44 | 65.35 | 4.82 | parity passes |
| ViT-H+/16 | 224 | 1 | fp16 | fp16-autocast | 1 | 26.8 | 37.57 | 41.35 | 4.82 | parity passes |
| ViT-H+/16 | 224 | 1 | bf16 | bf16-cast | 1 | 39.5 | 25.60 | 27.86 | 3.69 | parity passes |
| ViT-H+/16 | 224 | 8 | fp32 | fp32 | 1 | 23.3 | 347.50 | 350.15 | 4.83 | parity passes |
| ViT-H+/16 | 224 | 8 | fp16 | fp16-autocast | 1 | 48.3 | 163.79 | 180.50 | 4.84 | parity passes |
| ViT-H+/16 | 224 | 8 | bf16 | bf16-cast | 1 | 59.5 | 133.67 | 143.87 | 3.69 | parity passes |
| ViT-H+/16 | 224 | 32 | fp32 | fp32 | 1 | 21.4 | 1456.29 | 1624.25 | 4.94 | parity passes |
| ViT-H+/16 | 224 | 32 | fp16 | fp16-autocast | 1 | 46.4 | 685.44 | 712.38 | 4.92 | parity passes |
| ViT-H+/16 | 224 | 32 | bf16 | bf16-cast | 1 | 53.9 | 595.78 | 622.36 | 3.69 | parity passes |
| ViT-H+/16 | 448 | 1 | fp32 | fp32 | 1 | 5.1 | 199.66 | 207.91 | 4.84 | parity passes |
| ViT-H+/16 | 448 | 1 | fp16 | fp16-cast | 1 | 14.3 | 69.84 | 75.08 | 3.69 | parity passes |
| ViT-H+/16 | 448 | 1 | bf16 | bf16-cast | 1 | 14.1 | 70.74 | 75.58 | 3.69 | parity passes |
| ViT-H+/16 | 448 | 8 | fp32 | fp32 | 1 | 4.7 | 1703.34 | 1769.10 | 4.97 | parity passes |
| ViT-H+/16 | 448 | 8 | fp16 | fp16-cast | 1 | 13.5 | 588.51 | 610.10 | 3.69 | parity passes |
| ViT-H+/16 | 448 | 8 | bf16 | bf16-cast | 1 | 13.8 | 578.57 | 613.58 | 3.69 | parity passes |
| ViT-H+/16 | 448 | 32 | fp32 | fp32 | 1 | 4.1 | 7814.99 | 7932.98 | 7.48 | parity passes |
| ViT-H+/16 | 448 | 32 | fp16 | fp16-cast | 1 | 11.8 | 2699.35 | 2738.54 | 4.50 | parity passes |
| ViT-H+/16 | 448 | 32 | bf16 | bf16-cast | 1 | 13.0 | 2527.14 | 2709.95 | 4.51 | parity passes |
| ViT-H+/16 | 512 | 1 | fp32 | fp32 | 1 | 3.4 | 292.03 | 304.02 | 4.86 | parity passes |
| ViT-H+/16 | 512 | 1 | fp16 | fp16-cast | 1 | 10.9 | 91.36 | 97.73 | 3.69 | parity passes |
| ViT-H+/16 | 512 | 1 | bf16 | bf16-cast | 1 | 11.0 | 90.83 | 96.08 | 3.69 | parity passes |
| ViT-H+/16 | 512 | 8 | fp32 | fp32 + compile | 1 | 3.8 | 2105.29 | 2164.06 | 6.24 | parity passes |
| ViT-H+/16 | 512 | 8 | fp16 | fp16-cast + compile | 1 | 12.2 | 657.15 | 688.46 | 3.69 | parity passes |
| ViT-H+/16 | 512 | 8 | bf16 | bf16-cast + compile | 1 | 12.0 | 658.82 | 693.79 | 3.69 | parity passes |
| ViT-H+/16 | 512 | 32 | fp32 | fp32 | 1 | 3.1 | 10312.28 | 10388.94 | 9.27 | parity passes |
| ViT-H+/16 | 512 | 32 | fp16 | fp16-cast | 1 | 8.7 | 3684.47 | 3775.57 | 5.69 | parity passes |
| ViT-H+/16 | 512 | 32 | bf16 | bf16-cast | 1 | 9.7 | 3507.97 | 3594.97 | 5.68 | parity passes |
| ViT-7B/16 | 224 | 1 | fp16 | – | – | – | – | – | – | empty: fp16-cast: deferred, ViT-7B/16 runs in its own session (D8) |
| ViT-7B/16 | 224 | 1 | bf16 | – | – | – | – | – | – | empty: bf16-cast: deferred, ViT-7B/16 runs in its own session (D8) |
| ViT-7B/16 | 224 | 8 | fp16 | – | – | – | – | – | – | empty: fp16-cast: deferred, ViT-7B/16 runs in its own session (D8) |
| ViT-7B/16 | 224 | 8 | bf16 | – | – | – | – | – | – | empty: bf16-cast: deferred, ViT-7B/16 runs in its own session (D8) |
| ViT-7B/16 | 224 | 32 | fp16 | – | – | – | – | – | – | empty: fp16-cast: deferred, ViT-7B/16 runs in its own session (D8) |
| ViT-7B/16 | 224 | 32 | bf16 | – | – | – | – | – | – | empty: bf16-cast: deferred, ViT-7B/16 runs in its own session (D8) |
| ViT-7B/16 | 448 | 1 | fp16 | – | – | – | – | – | – | empty: fp16-cast: deferred, ViT-7B/16 runs in its own session (D8) |
| ViT-7B/16 | 448 | 1 | bf16 | – | – | – | – | – | – | empty: bf16-cast: deferred, ViT-7B/16 runs in its own session (D8) |
| ViT-7B/16 | 448 | 8 | fp16 | – | – | – | – | – | – | empty: fp16-cast: deferred, ViT-7B/16 runs in its own session (D8) |
| ViT-7B/16 | 448 | 8 | bf16 | – | – | – | – | – | – | empty: bf16-cast: deferred, ViT-7B/16 runs in its own session (D8) |
| ViT-7B/16 | 448 | 32 | fp16 | – | – | – | – | – | – | empty: fp16-cast: deferred, ViT-7B/16 runs in its own session (D8) |
| ViT-7B/16 | 448 | 32 | bf16 | – | – | – | – | – | – | empty: bf16-cast: deferred, ViT-7B/16 runs in its own session (D8) |
| ViT-7B/16 | 512 | 1 | fp16 | – | – | – | – | – | – | empty: fp16-cast: deferred, ViT-7B/16 runs in its own session (D8) |
| ViT-7B/16 | 512 | 1 | bf16 | – | – | – | – | – | – | empty: bf16-cast: deferred, ViT-7B/16 runs in its own session (D8) |
| ViT-7B/16 | 512 | 8 | fp16 | – | – | – | – | – | – | empty: fp16-cast: deferred, ViT-7B/16 runs in its own session (D8) |
| ViT-7B/16 | 512 | 8 | bf16 | – | – | – | – | – | – | empty: bf16-cast: deferred, ViT-7B/16 runs in its own session (D8) |
| ViT-7B/16 | 512 | 32 | fp16 | – | – | – | – | – | – | empty: fp16-cast: deferred, ViT-7B/16 runs in its own session (D8) |
| ViT-7B/16 | 512 | 32 | bf16 | – | – | – | – | – | – | empty: bf16-cast: deferred, ViT-7B/16 runs in its own session (D8) |

## 2. Baseline rows: patch embeddings

The fastest valid dtype mode of each dtype (D2, Section 3.7 of the plan). Each dtype mode counts with the median of its valid runs; Runs gives their number (finding F1 of `specs/03-baseline/verify.html`). Compile rows count since the decision of the user on 2026-11-06; only P1 and P2 have them.

| Model size | Input resolution | Batch size | Dtype | Dtype mode | Runs | Throughput (img/s) | Median (ms) | p90 (ms) | Peak memory (GB) | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ViT-S/16 | 224 | 1 | fp32 | fp32 | 1 | 131.4 | 7.95 | 9.55 | 0.57 | parity passes |
| ViT-S/16 | 224 | 1 | fp16 | fp16-cast | 1 | 130.6 | 7.96 | 9.30 | 0.56 | parity passes |
| ViT-S/16 | 224 | 1 | bf16 | bf16-cast | 1 | 153.9 | 5.41 | 9.18 | 0.56 | parity passes |
| ViT-S/16 | 224 | 8 | fp32 | fp32 + compile | 1 | 642.6 | 13.04 | 13.73 | 0.77 | parity passes |
| ViT-S/16 | 224 | 8 | fp16 | fp16-cast + compile | 1 | 1854.9 | 4.50 | 4.75 | 0.72 | parity passes |
| ViT-S/16 | 224 | 8 | bf16 | bf16-cast + compile | 1 | 1841.8 | 4.57 | 4.79 | 0.73 | parity passes |
| ViT-S/16 | 224 | 32 | fp32 | fp32 | 1 | 763.7 | 41.42 | 43.57 | 1.69 | parity passes |
| ViT-S/16 | 224 | 32 | fp16 | fp16-cast | 1 | 1273.0 | 24.99 | 25.99 | 1.74 | parity passes |
| ViT-S/16 | 224 | 32 | bf16 | bf16-cast | 1 | 1239.8 | 25.72 | 26.77 | 1.74 | parity passes |
| ViT-S/16 | 448 | 1 | fp32 | fp32 | 1 | 104.3 | 9.68 | 10.04 | 0.60 | parity passes |
| ViT-S/16 | 448 | 1 | fp16 | fp16-cast | 1 | 107.9 | 9.42 | 10.12 | 0.60 | parity passes |
| ViT-S/16 | 448 | 1 | bf16 | bf16-cast | 1 | 110.6 | 9.32 | 10.04 | 0.60 | parity passes |
| ViT-S/16 | 448 | 8 | fp32 | fp32 | 1 | 158.1 | 50.11 | 52.92 | 1.69 | parity passes |
| ViT-S/16 | 448 | 8 | fp16 | fp16-cast | 1 | 302.9 | 26.18 | 27.58 | 1.74 | parity passes |
| ViT-S/16 | 448 | 8 | bf16 | bf16-cast | 1 | 301.2 | 26.52 | 27.73 | 1.74 | parity passes |
| ViT-S/16 | 448 | 32 | fp32 | fp32 | 1 | 159.0 | 199.77 | 216.66 | 1.83 | parity passes |
| ViT-S/16 | 448 | 32 | fp16 | fp16-cast | 1 | 297.2 | 110.13 | 112.75 | 1.80 | parity passes |
| ViT-S/16 | 448 | 32 | bf16 | bf16-cast | 1 | 299.1 | 109.20 | 112.64 | 1.80 | parity passes |
| ViT-S/16 | 512 | 1 | fp32 | fp32 | 1 | 82.5 | 12.08 | 12.63 | 0.61 | parity passes |
| ViT-S/16 | 512 | 1 | fp16 | fp16-autocast | 1 | 115.5 | 8.68 | 8.98 | 0.61 | parity passes |
| ViT-S/16 | 512 | 1 | bf16 | bf16-cast | 1 | 109.7 | 9.42 | 10.23 | 0.61 | parity passes |
| ViT-S/16 | 512 | 8 | fp32 | fp32 | 1 | 112.7 | 69.01 | 77.97 | 1.70 | parity passes |
| ViT-S/16 | 512 | 8 | fp16 | fp16-cast | 1 | 221.0 | 35.98 | 37.85 | 1.68 | parity passes |
| ViT-S/16 | 512 | 8 | bf16 | bf16-cast | 1 | 228.6 | 34.90 | 36.67 | 1.68 | parity passes |
| ViT-S/16 | 512 | 32 | fp32 | fp32 | 1 | 95.9 | 321.00 | 373.22 | 3.04 | parity passes |
| ViT-S/16 | 512 | 32 | fp16 | fp16-cast | 1 | 211.8 | 151.99 | 159.19 | 1.86 | parity passes |
| ViT-S/16 | 512 | 32 | bf16 | bf16-cast | 1 | 222.6 | 145.23 | 154.59 | 1.85 | parity passes |
| ViT-S+/16 | 224 | 1 | fp32 | fp32 | 1 | 113.9 | 9.03 | 10.15 | 0.60 | parity passes |
| ViT-S+/16 | 224 | 1 | fp16 | fp16-cast | 1 | 122.2 | 8.46 | 10.07 | 0.61 | parity passes |
| ViT-S+/16 | 224 | 1 | bf16 | bf16-cast | 1 | 123.0 | 8.49 | 10.18 | 0.61 | parity passes |
| ViT-S+/16 | 224 | 8 | fp32 | fp32 | 1 | 414.6 | 19.19 | 20.27 | 0.67 | parity passes |
| ViT-S+/16 | 224 | 8 | fp16 | fp16-cast | 1 | 760.7 | 10.50 | 11.11 | 0.64 | parity passes |
| ViT-S+/16 | 224 | 8 | bf16 | bf16-cast | 1 | 757.5 | 10.48 | 11.26 | 0.64 | parity passes |
| ViT-S+/16 | 224 | 32 | fp32 | fp32 | 1 | 463.3 | 70.20 | 73.39 | 1.72 | parity passes |
| ViT-S+/16 | 224 | 32 | fp16 | fp16-cast | 1 | 936.4 | 34.59 | 36.16 | 1.78 | parity passes |
| ViT-S+/16 | 224 | 32 | bf16 | bf16-cast | 1 | 945.5 | 34.21 | 35.68 | 1.78 | parity passes |
| ViT-S+/16 | 448 | 1 | fp32 | fp32 | 1 | 84.0 | 11.95 | 12.75 | 0.64 | parity passes |
| ViT-S+/16 | 448 | 1 | fp16 | fp16-cast | 1 | 107.2 | 9.54 | 10.41 | 0.61 | parity passes |
| ViT-S+/16 | 448 | 1 | bf16 | bf16-cast | 1 | 102.5 | 9.84 | 10.58 | 0.61 | parity passes |
| ViT-S+/16 | 448 | 8 | fp32 | fp32 | 1 | 104.4 | 78.46 | 81.20 | 1.72 | parity passes |
| ViT-S+/16 | 448 | 8 | fp16 | fp16-cast | 1 | 281.1 | 29.38 | 32.91 | 1.78 | parity passes |
| ViT-S+/16 | 448 | 8 | bf16 | bf16-cast | 1 | 233.3 | 34.30 | 35.46 | 1.78 | parity passes |
| ViT-S+/16 | 448 | 32 | fp32 | fp32 | 1 | 94.4 | 341.34 | 354.70 | 1.86 | parity passes |
| ViT-S+/16 | 448 | 32 | fp16 | fp16-cast | 1 | 258.5 | 127.42 | 128.76 | 1.85 | parity passes |
| ViT-S+/16 | 448 | 32 | bf16 | bf16-cast | 1 | 223.4 | 143.99 | 156.00 | 1.85 | parity passes |
| ViT-S+/16 | 512 | 1 | fp32 | fp32 | 1 | 63.2 | 16.20 | 16.95 | 0.64 | parity passes |
| ViT-S+/16 | 512 | 1 | fp16 | fp16-cast | 1 | 101.8 | 9.89 | 10.52 | 0.62 | parity passes |
| ViT-S+/16 | 512 | 1 | bf16 | bf16-cast | 1 | 100.5 | 9.97 | 10.57 | 0.62 | parity passes |
| ViT-S+/16 | 512 | 8 | fp32 | fp32 | 1 | 70.5 | 115.00 | 122.18 | 1.74 | parity passes |
| ViT-S+/16 | 512 | 8 | fp16 | fp16-cast | 1 | 196.8 | 41.64 | 48.94 | 1.80 | parity passes |
| ViT-S+/16 | 512 | 8 | bf16 | bf16-cast | 1 | 178.4 | 44.84 | 46.35 | 1.80 | parity passes |
| ViT-S+/16 | 512 | 32 | fp32 | fp32 | 1 | 58.3 | 545.58 | 571.69 | 3.08 | parity passes |
| ViT-S+/16 | 512 | 32 | fp16 | fp16-cast | 1 | 155.5 | 209.85 | 217.14 | 1.90 | parity passes |
| ViT-S+/16 | 512 | 32 | bf16 | bf16-cast | 1 | 171.3 | 186.31 | 200.28 | 1.90 | parity passes |
| ViT-B/16 | 224 | 1 | fp32 | fp32 | 1 | 104.1 | 9.58 | 9.96 | 0.85 | parity passes |
| ViT-B/16 | 224 | 1 | fp16 | fp16-cast | 1 | 108.4 | 9.46 | 9.95 | 0.82 | parity passes |
| ViT-B/16 | 224 | 1 | bf16 | bf16-cast | 1 | 108.1 | 9.54 | 10.02 | 0.82 | parity passes |
| ViT-B/16 | 224 | 8 | fp32 | fp32 | 1 | 197.6 | 41.00 | 46.17 | 2.00 | parity passes |
| ViT-B/16 | 224 | 8 | fp16 | fp16-cast | 1 | 396.2 | 19.99 | 21.38 | 0.86 | parity passes |
| ViT-B/16 | 224 | 8 | bf16 | bf16-cast | 1 | 415.3 | 18.93 | 20.70 | 0.86 | parity passes |
| ViT-B/16 | 224 | 32 | fp32 | fp32 | 1 | 177.8 | 183.13 | 190.98 | 1.99 | parity passes |
| ViT-B/16 | 224 | 32 | fp16 | fp16-cast | 1 | 391.0 | 82.67 | 86.88 | 2.08 | parity passes |
| ViT-B/16 | 224 | 32 | bf16 | bf16-cast | 1 | 434.8 | 73.24 | 77.72 | 2.08 | parity passes |
| ViT-B/16 | 448 | 1 | fp32 | fp32 | 1 | 39.5 | 25.43 | 26.28 | 0.89 | parity passes |
| ViT-B/16 | 448 | 1 | fp16 | fp16-cast | 1 | 92.9 | 10.94 | 11.86 | 0.84 | parity passes |
| ViT-B/16 | 448 | 1 | bf16 | bf16-cast | 1 | 96.9 | 10.05 | 11.35 | 0.84 | parity passes |
| ViT-B/16 | 448 | 8 | fp32 | fp32 | 1 | 39.2 | 205.89 | 212.11 | 2.00 | parity passes |
| ViT-B/16 | 448 | 8 | fp16 | fp16-cast | 1 | 100.4 | 79.58 | 84.80 | 2.06 | parity passes |
| ViT-B/16 | 448 | 8 | bf16 | bf16-cast | 1 | 106.3 | 75.54 | 80.05 | 2.06 | parity passes |
| ViT-B/16 | 448 | 32 | fp32 | fp32 | 1 | 35.4 | 899.11 | 940.21 | 3.30 | parity passes |
| ViT-B/16 | 448 | 32 | fp16 | fp16-cast | 1 | 88.9 | 359.98 | 378.86 | 2.10 | parity passes |
| ViT-B/16 | 448 | 32 | bf16 | bf16-cast | 1 | 96.3 | 337.14 | 355.53 | 2.10 | parity passes |
| ViT-B/16 | 512 | 1 | fp32 | fp32 | 1 | 28.2 | 35.94 | 37.20 | 1.97 | parity passes |
| ViT-B/16 | 512 | 1 | fp16 | fp16-cast | 1 | 70.6 | 14.15 | 15.47 | 0.88 | parity passes |
| ViT-B/16 | 512 | 1 | bf16 | bf16-cast | 1 | 74.1 | 13.47 | 14.76 | 0.88 | parity passes |
| ViT-B/16 | 512 | 8 | fp32 | fp32 | 1 | 27.5 | 292.66 | 310.71 | 2.03 | parity passes |
| ViT-B/16 | 512 | 8 | fp16 | fp16-cast | 1 | 74.1 | 109.27 | 113.75 | 1.97 | parity passes |
| ViT-B/16 | 512 | 8 | bf16 | bf16-cast | 1 | 80.2 | 99.53 | 106.36 | 1.97 | parity passes |
| ViT-B/16 | 512 | 32 | fp32 | fp32 | 1 | 23.4 | 1358.29 | 1420.61 | 4.66 | parity passes |
| ViT-B/16 | 512 | 32 | fp16 | fp16-cast | 1 | 63.3 | 506.55 | 542.27 | 3.23 | parity passes |
| ViT-B/16 | 512 | 32 | bf16 | bf16-cast | 1 | 66.7 | 477.35 | 506.03 | 3.23 | parity passes |
| ViT-L/16 | 224 | 1 | fp32 | fp32 | 1 | 38.5 | 26.25 | 28.35 | 2.67 | parity passes |
| ViT-L/16 | 224 | 1 | fp16 | fp16-autocast | 1 | 55.0 | 18.04 | 20.54 | 2.67 | parity passes |
| ViT-L/16 | 224 | 1 | bf16 | bf16-cast | 1 | 81.4 | 12.16 | 12.91 | 1.72 | parity passes |
| ViT-L/16 | 224 | 8 | fp32 | fp32 | 1 | 54.8 | 151.80 | 155.46 | 2.68 | parity passes |
| ViT-L/16 | 224 | 8 | fp16 | fp16-autocast | 1 | 111.3 | 72.20 | 77.78 | 2.68 | parity passes |
| ViT-L/16 | 224 | 8 | bf16 | bf16-cast | 1 | 156.7 | 51.96 | 55.00 | 2.83 | parity passes |
| ViT-L/16 | 224 | 32 | fp32 | fp32 | 1 | 50.7 | 634.91 | 647.66 | 2.75 | parity passes |
| ViT-L/16 | 224 | 32 | fp16 | fp16-autocast | 1 | 122.5 | 262.75 | 267.49 | 2.75 | parity passes |
| ViT-L/16 | 224 | 32 | bf16 | bf16-cast | 1 | 148.0 | 216.13 | 231.26 | 2.86 | parity passes |
| ViT-L/16 | 448 | 1 | fp32 | fp32 | 1 | 11.8 | 86.39 | 88.58 | 2.68 | parity passes |
| ViT-L/16 | 448 | 1 | fp16 | fp16-autocast | 1 | 26.5 | 38.05 | 40.78 | 2.69 | parity passes |
| ViT-L/16 | 448 | 1 | bf16 | bf16-cast | 1 | 34.8 | 28.65 | 31.36 | 1.76 | parity passes |
| ViT-L/16 | 448 | 8 | fp32 | fp32 | 1 | 11.5 | 694.72 | 706.68 | 2.77 | parity passes |
| ViT-L/16 | 448 | 8 | fp16 | fp16-autocast | 1 | 29.5 | 273.86 | 291.71 | 2.78 | parity passes |
| ViT-L/16 | 448 | 8 | bf16 | bf16-cast | 1 | 35.0 | 233.21 | 242.27 | 2.87 | parity passes |
| ViT-L/16 | 448 | 32 | fp32 | fp32 | 1 | 10.4 | 3093.79 | 3136.13 | 5.23 | parity passes |
| ViT-L/16 | 448 | 32 | fp16 | fp16-autocast | 1 | 27.1 | 1160.39 | 1210.52 | 4.15 | parity passes |
| ViT-L/16 | 448 | 32 | bf16 | bf16-cast | 1 | 29.5 | 1080.21 | 1127.57 | 4.15 | parity passes |
| ViT-L/16 | 512 | 1 | fp32 | fp32 | 1 | 8.5 | 119.57 | 122.90 | 2.69 | parity passes |
| ViT-L/16 | 512 | 1 | fp16 | fp16-autocast | 1 | 20.3 | 49.95 | 55.75 | 2.70 | parity passes |
| ViT-L/16 | 512 | 1 | bf16 | bf16-cast | 1 | 26.0 | 38.52 | 41.59 | 1.78 | parity passes |
| ViT-L/16 | 512 | 8 | fp32 | fp32 | 1 | 7.9 | 1010.40 | 1045.99 | 2.81 | parity passes |
| ViT-L/16 | 512 | 8 | fp16 | fp16-autocast | 1 | 19.1 | 421.64 | 452.95 | 2.82 | parity passes |
| ViT-L/16 | 512 | 8 | bf16 | bf16-cast | 1 | 25.1 | 322.75 | 335.85 | 2.89 | parity passes |
| ViT-L/16 | 512 | 32 | fp32 | fp32 | 1 | 7.3 | 4418.54 | 4462.10 | 5.62 | parity passes |
| ViT-L/16 | 512 | 32 | fp16 | fp16-autocast | 1 | 17.1 | 1885.14 | 1941.40 | 5.35 | parity passes |
| ViT-L/16 | 512 | 32 | bf16 | bf16-cast | 1 | 21.0 | 1517.90 | 1581.07 | 4.25 | parity passes |
| ViT-H+/16 | 224 | 1 | fp32 | fp32 | 1 | 16.3 | 62.40 | 65.33 | 4.82 | parity passes |
| ViT-H+/16 | 224 | 1 | fp16 | fp16-autocast | 1 | 26.8 | 37.52 | 41.34 | 4.82 | parity passes |
| ViT-H+/16 | 224 | 1 | bf16 | bf16-cast | 1 | 39.6 | 25.55 | 27.79 | 3.69 | parity passes |
| ViT-H+/16 | 224 | 8 | fp32 | fp32 | 1 | 23.3 | 347.72 | 350.27 | 4.83 | parity passes |
| ViT-H+/16 | 224 | 8 | fp16 | fp16-autocast | 1 | 48.2 | 164.00 | 180.51 | 4.84 | parity passes |
| ViT-H+/16 | 224 | 8 | bf16 | bf16-cast | 1 | 59.5 | 133.77 | 143.89 | 3.69 | parity passes |
| ViT-H+/16 | 224 | 32 | fp32 | fp32 | 1 | 21.4 | 1457.17 | 1623.88 | 4.94 | parity passes |
| ViT-H+/16 | 224 | 32 | fp16 | fp16-autocast | 1 | 46.4 | 686.79 | 713.00 | 4.92 | parity passes |
| ViT-H+/16 | 224 | 32 | bf16 | bf16-cast | 1 | 53.8 | 596.02 | 622.56 | 3.69 | parity passes |
| ViT-H+/16 | 448 | 1 | fp32 | fp32 | 1 | 5.1 | 199.63 | 207.89 | 4.84 | parity passes |
| ViT-H+/16 | 448 | 1 | fp16 | fp16-cast | 1 | 14.3 | 69.81 | 75.02 | 3.69 | parity passes |
| ViT-H+/16 | 448 | 1 | bf16 | bf16-cast | 1 | 14.1 | 70.68 | 75.50 | 3.69 | parity passes |
| ViT-H+/16 | 448 | 8 | fp32 | fp32 | 1 | 4.7 | 1702.99 | 1768.58 | 4.97 | parity passes |
| ViT-H+/16 | 448 | 8 | fp16 | fp16-cast | 1 | 13.5 | 589.05 | 611.98 | 3.69 | parity passes |
| ViT-H+/16 | 448 | 8 | bf16 | bf16-cast | 1 | 13.8 | 578.90 | 613.92 | 3.69 | parity passes |
| ViT-H+/16 | 448 | 32 | fp32 | fp32 | 1 | 4.1 | 7816.57 | 7936.58 | 7.48 | parity passes |
| ViT-H+/16 | 448 | 32 | fp16 | fp16-cast | 1 | 11.8 | 2699.19 | 2738.72 | 4.50 | parity passes |
| ViT-H+/16 | 448 | 32 | bf16 | bf16-cast | 1 | 13.0 | 2528.22 | 2710.15 | 4.51 | parity passes |
| ViT-H+/16 | 512 | 1 | fp32 | fp32 | 1 | 3.4 | 292.04 | 304.04 | 4.86 | parity passes |
| ViT-H+/16 | 512 | 1 | fp16 | fp16-cast | 1 | 10.9 | 91.31 | 97.72 | 3.69 | parity passes |
| ViT-H+/16 | 512 | 1 | bf16 | bf16-cast | 1 | 11.0 | 90.81 | 96.04 | 3.69 | parity passes |
| ViT-H+/16 | 512 | 8 | fp32 | fp32 + compile | 1 | 3.8 | 2106.48 | 2164.90 | 6.24 | parity passes |
| ViT-H+/16 | 512 | 8 | fp16 | fp16-cast + compile | 1 | 12.2 | 658.31 | 688.91 | 3.69 | parity passes |
| ViT-H+/16 | 512 | 8 | bf16 | bf16-cast + compile | 1 | 12.0 | 659.22 | 694.31 | 3.69 | parity passes |
| ViT-H+/16 | 512 | 32 | fp32 | fp32 | 1 | 3.1 | 10323.46 | 10396.21 | 9.27 | parity passes |
| ViT-H+/16 | 512 | 32 | fp16 | fp16-cast | 1 | 8.7 | 3684.49 | 3776.13 | 5.69 | parity passes |
| ViT-H+/16 | 512 | 32 | bf16 | bf16-cast | 1 | 9.7 | 3508.44 | 3595.48 | 5.68 | parity passes |
| ViT-7B/16 | 224 | 1 | fp16 | – | – | – | – | – | – | empty: fp16-cast: deferred, ViT-7B/16 runs in its own session (D8) |
| ViT-7B/16 | 224 | 1 | bf16 | – | – | – | – | – | – | empty: bf16-cast: deferred, ViT-7B/16 runs in its own session (D8) |
| ViT-7B/16 | 224 | 8 | fp16 | – | – | – | – | – | – | empty: fp16-cast: deferred, ViT-7B/16 runs in its own session (D8) |
| ViT-7B/16 | 224 | 8 | bf16 | – | – | – | – | – | – | empty: bf16-cast: deferred, ViT-7B/16 runs in its own session (D8) |
| ViT-7B/16 | 224 | 32 | fp16 | – | – | – | – | – | – | empty: fp16-cast: deferred, ViT-7B/16 runs in its own session (D8) |
| ViT-7B/16 | 224 | 32 | bf16 | – | – | – | – | – | – | empty: bf16-cast: deferred, ViT-7B/16 runs in its own session (D8) |
| ViT-7B/16 | 448 | 1 | fp16 | – | – | – | – | – | – | empty: fp16-cast: deferred, ViT-7B/16 runs in its own session (D8) |
| ViT-7B/16 | 448 | 1 | bf16 | – | – | – | – | – | – | empty: bf16-cast: deferred, ViT-7B/16 runs in its own session (D8) |
| ViT-7B/16 | 448 | 8 | fp16 | – | – | – | – | – | – | empty: fp16-cast: deferred, ViT-7B/16 runs in its own session (D8) |
| ViT-7B/16 | 448 | 8 | bf16 | – | – | – | – | – | – | empty: bf16-cast: deferred, ViT-7B/16 runs in its own session (D8) |
| ViT-7B/16 | 448 | 32 | fp16 | – | – | – | – | – | – | empty: fp16-cast: deferred, ViT-7B/16 runs in its own session (D8) |
| ViT-7B/16 | 448 | 32 | bf16 | – | – | – | – | – | – | empty: bf16-cast: deferred, ViT-7B/16 runs in its own session (D8) |
| ViT-7B/16 | 512 | 1 | fp16 | – | – | – | – | – | – | empty: fp16-cast: deferred, ViT-7B/16 runs in its own session (D8) |
| ViT-7B/16 | 512 | 1 | bf16 | – | – | – | – | – | – | empty: bf16-cast: deferred, ViT-7B/16 runs in its own session (D8) |
| ViT-7B/16 | 512 | 8 | fp16 | – | – | – | – | – | – | empty: fp16-cast: deferred, ViT-7B/16 runs in its own session (D8) |
| ViT-7B/16 | 512 | 8 | bf16 | – | – | – | – | – | – | empty: bf16-cast: deferred, ViT-7B/16 runs in its own session (D8) |
| ViT-7B/16 | 512 | 32 | fp16 | – | – | – | – | – | – | empty: fp16-cast: deferred, ViT-7B/16 runs in its own session (D8) |
| ViT-7B/16 | 512 | 32 | bf16 | – | – | – | – | – | – | empty: bf16-cast: deferred, ViT-7B/16 runs in its own session (D8) |

## 3. All dtype modes

| Model size | Input resolution | Batch size | Dtype mode | Valid | Frame (img/s) | Patch (img/s) | Device (img/s) | Frame median (ms) | Frame p90 (ms) | Peak (GB) | Load peak (GB) | MPS driver (GB) | Need (GB) | Timed batches |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ViT-S/16 | 224 | 1 | fp32 | yes | 130.7 | 131.4 | 135.0 | 7.99 | 9.59 | 0.57 | 0.52 | 0.10 | 4.11 | 1315 |
| ViT-S/16 | 224 | 8 | fp32 | yes | 587.8 | 587.5 | 599.7 | 13.61 | 14.12 | 0.64 | 0.52 | 0.17 | 4.17 | 735 |
| ViT-S/16 | 224 | 32 | fp32 | yes | 765.5 | 763.7 | 772.6 | 41.30 | 43.38 | 1.69 | 0.52 | 1.18 | 4.36 | 239 |
| ViT-S/16 | 448 | 1 | fp32 | yes | 104.0 | 104.3 | 106.8 | 9.71 | 10.07 | 0.60 | 0.52 | 0.14 | 4.16 | 1044 |
| ViT-S/16 | 448 | 8 | fp32 | yes | 158.5 | 158.1 | 159.7 | 49.98 | 52.78 | 1.69 | 0.52 | 1.18 | 4.54 | 198 |
| ViT-S/16 | 448 | 32 | fp32 | yes | 159.7 | 159.0 | 160.0 | 199.00 | 215.52 | 1.83 | 0.52 | 1.18 | 5.84 | 50 |
| ViT-S/16 | 512 | 1 | fp32 | yes | 82.2 | 82.5 | 84.1 | 12.12 | 12.66 | 0.61 | 0.52 | 0.14 | 4.19 | 825 |
| ViT-S/16 | 512 | 8 | fp32 | yes | 112.9 | 112.7 | 113.6 | 68.84 | 77.73 | 1.70 | 0.52 | 1.18 | 4.76 | 141 |
| ViT-S/16 | 512 | 32 | fp32 | yes | 96.2 | 95.9 | 96.3 | 319.92 | 372.13 | 3.04 | 0.52 | 2.33 | 6.74 | 31 |
| ViT-S/16 | 224 | 1 | fp16-cast | yes | 129.8 | 130.6 | 134.1 | 8.01 | 9.35 | 0.56 | 0.51 | 0.06 | 4.06 | 1306 |
| ViT-S/16 | 224 | 8 | fp16-cast | yes | 847.4 | 848.0 | 870.7 | 9.62 | 10.27 | 0.60 | 0.51 | 0.09 | 4.08 | 1060 |
| ViT-S/16 | 224 | 32 | fp16-cast | yes | 1275.3 | 1273.0 | 1290.5 | 24.97 | 25.96 | 1.74 | 0.51 | 1.19 | 4.18 | 398 |
| ViT-S/16 | 448 | 1 | fp16-cast | yes | 107.4 | 107.9 | 110.5 | 9.47 | 10.17 | 0.60 | 0.51 | 0.10 | 4.08 | 1080 |
| ViT-S/16 | 448 | 8 | fp16-cast | yes | 303.9 | 302.9 | 307.3 | 26.15 | 27.44 | 1.74 | 0.51 | 1.19 | 4.27 | 379 |
| ViT-S/16 | 448 | 32 | fp16-cast | yes | 298.2 | 297.2 | 299.3 | 109.79 | 112.40 | 1.80 | 0.51 | 1.13 | 4.92 | 93 |
| ViT-S/16 | 512 | 1 | fp16-cast | yes | 107.0 | 107.5 | 110.0 | 9.48 | 10.18 | 0.61 | 0.51 | 0.10 | 4.09 | 1075 |
| ViT-S/16 | 512 | 8 | fp16-cast | yes | 221.4 | 221.0 | 223.7 | 35.92 | 37.72 | 1.68 | 0.51 | 1.13 | 4.38 | 277 |
| ViT-S/16 | 512 | 32 | fp16-cast | yes | 212.6 | 211.8 | 213.1 | 151.52 | 158.77 | 1.86 | 0.51 | 1.13 | 5.37 | 67 |
| ViT-S/16 | 224 | 1 | bf16-cast | yes | 153.3 | 153.9 | 158.7 | 5.42 | 9.22 | 0.56 | 0.51 | 0.06 | 4.06 | 1540 |
| ViT-S/16 | 224 | 8 | bf16-cast | yes | 818.7 | 819.4 | 840.7 | 9.78 | 10.27 | 0.59 | 0.51 | 0.09 | 4.08 | 1025 |
| ViT-S/16 | 224 | 32 | bf16-cast | yes | 1243.2 | 1239.8 | 1257.5 | 25.67 | 26.62 | 1.74 | 0.51 | 1.19 | 4.18 | 388 |
| ViT-S/16 | 448 | 1 | bf16-cast | yes | 110.0 | 110.6 | 113.3 | 9.37 | 10.10 | 0.60 | 0.51 | 0.10 | 4.08 | 1107 |
| ViT-S/16 | 448 | 8 | bf16-cast | yes | 301.9 | 301.2 | 305.3 | 26.48 | 27.61 | 1.74 | 0.51 | 1.19 | 4.27 | 377 |
| ViT-S/16 | 448 | 32 | bf16-cast | yes | 300.2 | 299.1 | 301.4 | 108.86 | 112.26 | 1.80 | 0.51 | 1.13 | 4.92 | 94 |
| ViT-S/16 | 512 | 1 | bf16-cast | yes | 109.2 | 109.7 | 112.4 | 9.47 | 10.27 | 0.61 | 0.51 | 0.10 | 4.09 | 1097 |
| ViT-S/16 | 512 | 8 | bf16-cast | yes | 228.8 | 228.6 | 231.0 | 34.91 | 36.51 | 1.68 | 0.51 | 1.13 | 4.38 | 286 |
| ViT-S/16 | 512 | 32 | bf16-cast | yes | 223.3 | 222.6 | 223.9 | 144.82 | 154.24 | 1.85 | 0.51 | 1.13 | 5.37 | 70 |
| ViT-S/16 | 224 | 1 | fp16-autocast | yes | 116.8 | 117.4 | 120.3 | 8.99 | 10.26 | 0.57 | 0.52 | 0.11 | 4.11 | 1175 |
| ViT-S/16 | 224 | 8 | fp16-autocast | yes | 733.7 | 733.3 | 751.8 | 10.84 | 11.59 | 0.60 | 0.52 | 0.14 | 4.17 | 917 |
| ViT-S/16 | 224 | 32 | fp16-autocast | yes | 998.8 | 995.0 | 1010.3 | 33.06 | 34.31 | 1.69 | 0.52 | 1.18 | 4.36 | 311 |
| ViT-S/16 | 448 | 1 | fp16-autocast | yes | 103.0 | 103.5 | 106.0 | 9.82 | 10.43 | 0.60 | 0.52 | 0.14 | 4.16 | 1035 |
| ViT-S/16 | 448 | 8 | fp16-autocast | yes | 233.9 | 232.9 | 236.4 | 34.16 | 35.08 | 1.69 | 0.52 | 1.18 | 4.54 | 292 |
| ViT-S/16 | 448 | 32 | fp16-autocast | yes | 232.4 | 231.0 | 233.2 | 140.95 | 146.73 | 1.83 | 0.52 | 1.18 | 5.84 | 73 |
| ViT-S/16 | 512 | 1 | fp16-autocast | yes | 115.3 | 115.5 | 118.7 | 8.68 | 9.00 | 0.61 | 0.52 | 0.15 | 4.19 | 1156 |
| ViT-S/16 | 512 | 8 | fp16-autocast | yes | 200.0 | 199.3 | 202.0 | 38.80 | 45.41 | 1.70 | 0.52 | 1.18 | 4.76 | 250 |
| ViT-S/16 | 512 | 32 | fp16-autocast | yes | 173.5 | 172.6 | 173.9 | 189.22 | 196.30 | 1.89 | 0.52 | 1.18 | 6.74 | 54 |
| ViT-S/16 | 224 | 1 | bf16-autocast | yes | 115.7 | 116.3 | 119.2 | 9.03 | 10.26 | 0.57 | 0.52 | 0.11 | 4.11 | 1164 |
| ViT-S/16 | 224 | 8 | bf16-autocast | yes | 718.4 | 718.1 | 735.8 | 11.03 | 11.88 | 0.60 | 0.52 | 0.14 | 4.17 | 898 |
| ViT-S/16 | 224 | 32 | bf16-autocast | yes | 919.6 | 915.9 | 928.5 | 35.74 | 37.74 | 1.69 | 0.52 | 1.18 | 4.36 | 287 |
| ViT-S/16 | 448 | 1 | bf16-autocast | yes | 104.6 | 105.0 | 107.5 | 9.73 | 10.39 | 0.60 | 0.52 | 0.14 | 4.16 | 1050 |
| ViT-S/16 | 448 | 8 | bf16-autocast | yes | 220.4 | 219.7 | 222.7 | 36.16 | 37.61 | 1.69 | 0.52 | 1.18 | 4.54 | 275 |
| ViT-S/16 | 448 | 32 | bf16-autocast | yes | 211.9 | 210.8 | 212.5 | 152.80 | 161.61 | 1.83 | 0.52 | 1.18 | 5.84 | 66 |
| ViT-S/16 | 512 | 1 | bf16-autocast | yes | 101.1 | 101.5 | 104.0 | 9.90 | 10.44 | 0.61 | 0.52 | 0.15 | 4.19 | 1016 |
| ViT-S/16 | 512 | 8 | bf16-autocast | yes | 163.9 | 163.4 | 165.2 | 48.61 | 50.79 | 1.70 | 0.52 | 1.18 | 4.76 | 205 |
| ViT-S/16 | 512 | 32 | bf16-autocast | yes | 157.4 | 156.5 | 157.7 | 204.94 | 214.26 | 1.89 | 0.52 | 1.18 | 6.74 | 49 |
| ViT-S+/16 | 224 | 1 | fp32 | yes | 113.3 | 113.9 | 116.7 | 9.08 | 10.20 | 0.60 | 0.56 | 0.14 | 4.15 | 1140 |
| ViT-S+/16 | 224 | 8 | fp32 | yes | 414.5 | 414.6 | 420.8 | 19.21 | 20.25 | 0.67 | 0.56 | 0.20 | 4.20 | 519 |
| ViT-S+/16 | 224 | 32 | fp32 | yes | 464.3 | 463.3 | 466.7 | 70.00 | 73.18 | 1.72 | 0.56 | 1.21 | 4.40 | 145 |
| ViT-S+/16 | 448 | 1 | fp32 | yes | 83.7 | 84.0 | 85.8 | 12.00 | 12.78 | 0.64 | 0.56 | 0.17 | 4.19 | 841 |
| ViT-S+/16 | 448 | 8 | fp32 | yes | 104.5 | 104.4 | 105.0 | 78.38 | 80.93 | 1.72 | 0.56 | 1.21 | 4.57 | 131 |
| ViT-S+/16 | 448 | 32 | fp32 | yes | 94.6 | 94.4 | 94.7 | 340.54 | 353.75 | 1.86 | 0.56 | 1.21 | 5.87 | 30 |
| ViT-S+/16 | 512 | 1 | fp32 | yes | 63.0 | 63.2 | 64.2 | 16.25 | 17.00 | 0.64 | 0.56 | 0.17 | 4.22 | 633 |
| ViT-S+/16 | 512 | 8 | fp32 | yes | 70.7 | 70.5 | 70.9 | 114.87 | 121.96 | 1.74 | 0.56 | 1.21 | 4.80 | 89 |
| ViT-S+/16 | 512 | 32 | fp32 | yes | 58.4 | 58.3 | 58.4 | 544.47 | 570.45 | 3.08 | 0.56 | 2.36 | 6.78 | 30 |
| ViT-S+/16 | 224 | 1 | fp16-cast | yes | 121.6 | 122.2 | 125.2 | 8.51 | 10.14 | 0.61 | 0.56 | 0.10 | 4.07 | 1223 |
| ViT-S+/16 | 224 | 8 | fp16-cast | yes | 760.2 | 760.7 | 780.5 | 10.50 | 11.10 | 0.64 | 0.56 | 0.12 | 4.10 | 951 |
| ViT-S+/16 | 224 | 32 | fp16-cast | yes | 937.2 | 936.4 | 945.6 | 34.54 | 36.12 | 1.78 | 0.56 | 1.23 | 4.20 | 293 |
| ViT-S+/16 | 448 | 1 | fp16-cast | yes | 106.6 | 107.2 | 109.7 | 9.60 | 10.45 | 0.61 | 0.56 | 0.10 | 4.10 | 1073 |
| ViT-S+/16 | 448 | 8 | fp16-cast | yes | 281.6 | 281.1 | 284.4 | 29.33 | 32.90 | 1.78 | 0.56 | 1.23 | 4.29 | 352 |
| ViT-S+/16 | 448 | 32 | fp16-cast | yes | 259.2 | 258.5 | 260.0 | 127.14 | 128.51 | 1.85 | 0.56 | 1.16 | 4.93 | 81 |
| ViT-S+/16 | 512 | 1 | fp16-cast | yes | 101.4 | 101.8 | 104.1 | 9.95 | 10.56 | 0.62 | 0.56 | 0.10 | 4.11 | 1019 |
| ViT-S+/16 | 512 | 8 | fp16-cast | yes | 196.8 | 196.8 | 198.5 | 41.63 | 48.93 | 1.80 | 0.56 | 1.23 | 4.40 | 247 |
| ViT-S+/16 | 512 | 32 | fp16-cast | yes | 155.9 | 155.5 | 156.2 | 209.61 | 216.67 | 1.90 | 0.56 | 1.16 | 5.39 | 49 |
| ViT-S+/16 | 224 | 1 | bf16-cast | yes | 122.3 | 123.0 | 126.0 | 8.54 | 10.23 | 0.61 | 0.56 | 0.10 | 4.07 | 1231 |
| ViT-S+/16 | 224 | 8 | bf16-cast | yes | 757.1 | 757.5 | 776.8 | 10.49 | 11.23 | 0.64 | 0.56 | 0.12 | 4.10 | 947 |
| ViT-S+/16 | 224 | 32 | bf16-cast | yes | 946.3 | 945.5 | 954.7 | 34.20 | 35.68 | 1.78 | 0.56 | 1.23 | 4.20 | 296 |
| ViT-S+/16 | 448 | 1 | bf16-cast | yes | 102.0 | 102.5 | 104.7 | 9.89 | 10.63 | 0.61 | 0.56 | 0.10 | 4.10 | 1025 |
| ViT-S+/16 | 448 | 8 | bf16-cast | yes | 233.7 | 233.3 | 235.8 | 34.24 | 35.41 | 1.78 | 0.56 | 1.23 | 4.29 | 292 |
| ViT-S+/16 | 448 | 32 | bf16-cast | yes | 224.0 | 223.4 | 224.6 | 143.68 | 155.63 | 1.85 | 0.56 | 1.16 | 4.93 | 70 |
| ViT-S+/16 | 512 | 1 | bf16-cast | yes | 100.1 | 100.5 | 102.7 | 10.02 | 10.62 | 0.62 | 0.56 | 0.10 | 4.11 | 1006 |
| ViT-S+/16 | 512 | 8 | bf16-cast | yes | 178.5 | 178.4 | 179.8 | 44.83 | 46.34 | 1.80 | 0.56 | 1.23 | 4.40 | 223 |
| ViT-S+/16 | 512 | 32 | bf16-cast | yes | 171.8 | 171.3 | 172.2 | 185.84 | 199.69 | 1.90 | 0.56 | 1.16 | 5.39 | 54 |
| ViT-S+/16 | 224 | 1 | fp16-autocast | yes | 107.6 | 108.2 | 110.6 | 9.52 | 10.95 | 0.61 | 0.56 | 0.15 | 4.15 | 1083 |
| ViT-S+/16 | 224 | 8 | fp16-autocast | yes | 630.5 | 630.2 | 644.6 | 12.64 | 13.38 | 0.64 | 0.56 | 0.17 | 4.20 | 788 |
| ViT-S+/16 | 224 | 32 | fp16-autocast | yes | 820.3 | 817.5 | 828.3 | 39.66 | 42.02 | 1.72 | 0.56 | 1.21 | 4.40 | 256 |
| ViT-S+/16 | 448 | 1 | fp16-autocast | yes | 98.8 | 99.1 | 101.4 | 10.23 | 10.86 | 0.64 | 0.56 | 0.18 | 4.19 | 992 |
| ViT-S+/16 | 448 | 8 | fp16-autocast | yes | 200.6 | 200.0 | 202.5 | 39.90 | 41.35 | 1.72 | 0.56 | 1.21 | 4.57 | 251 |
| ViT-S+/16 | 448 | 32 | fp16-autocast | yes | 189.6 | 188.8 | 190.0 | 169.66 | 188.81 | 1.86 | 0.56 | 1.21 | 5.87 | 59 |
| ViT-S+/16 | 512 | 1 | fp16-autocast | yes | 96.0 | 96.3 | 98.5 | 10.37 | 10.84 | 0.65 | 0.56 | 0.18 | 4.22 | 963 |
| ViT-S+/16 | 512 | 8 | fp16-autocast | yes | 167.6 | 167.6 | 169.2 | 47.82 | 56.54 | 1.74 | 0.56 | 1.21 | 4.80 | 210 |
| ViT-S+/16 | 512 | 32 | fp16-autocast | yes | 140.1 | 139.2 | 140.3 | 233.46 | 243.01 | 1.92 | 0.56 | 1.21 | 6.78 | 44 |
| ViT-S+/16 | 224 | 1 | bf16-autocast | yes | 107.4 | 107.9 | 110.3 | 9.67 | 10.64 | 0.61 | 0.56 | 0.15 | 4.15 | 1080 |
| ViT-S+/16 | 224 | 8 | bf16-autocast | yes | 623.9 | 623.4 | 637.4 | 12.77 | 13.49 | 0.64 | 0.56 | 0.17 | 4.20 | 780 |
| ViT-S+/16 | 224 | 32 | bf16-autocast | yes | 787.6 | 785.4 | 795.2 | 41.36 | 43.51 | 1.72 | 0.56 | 1.21 | 4.40 | 246 |
| ViT-S+/16 | 448 | 1 | bf16-autocast | yes | 97.6 | 97.9 | 100.2 | 10.27 | 10.79 | 0.64 | 0.56 | 0.18 | 4.19 | 980 |
| ViT-S+/16 | 448 | 8 | bf16-autocast | yes | 189.0 | 188.5 | 190.7 | 42.18 | 43.90 | 1.72 | 0.56 | 1.21 | 4.57 | 236 |
| ViT-S+/16 | 448 | 32 | bf16-autocast | yes | 177.6 | 177.0 | 178.0 | 180.52 | 197.89 | 1.86 | 0.56 | 1.21 | 5.87 | 56 |
| ViT-S+/16 | 512 | 1 | bf16-autocast | yes | 96.9 | 97.1 | 99.4 | 10.31 | 10.81 | 0.65 | 0.56 | 0.18 | 4.22 | 972 |
| ViT-S+/16 | 512 | 8 | bf16-autocast | yes | 154.3 | 154.2 | 155.5 | 55.96 | 58.83 | 1.74 | 0.56 | 1.21 | 4.80 | 193 |
| ViT-S+/16 | 512 | 32 | bf16-autocast | yes | 137.8 | 137.1 | 138.0 | 235.32 | 246.66 | 1.93 | 0.56 | 1.21 | 6.78 | 43 |
| ViT-B/16 | 224 | 1 | fp32 | yes | 103.6 | 104.1 | 106.5 | 9.62 | 9.97 | 0.85 | 0.80 | 0.39 | 4.43 | 1041 |
| ViT-B/16 | 224 | 8 | fp32 | yes | 197.7 | 197.6 | 199.1 | 41.04 | 46.15 | 2.00 | 0.80 | 1.52 | 4.54 | 247 |
| ViT-B/16 | 224 | 32 | fp32 | yes | 178.1 | 177.8 | 178.5 | 182.82 | 190.67 | 1.99 | 0.80 | 1.46 | 4.93 | 56 |
| ViT-B/16 | 448 | 1 | fp32 | yes | 39.5 | 39.5 | 40.0 | 25.47 | 26.29 | 0.89 | 0.80 | 0.42 | 4.52 | 396 |
| ViT-B/16 | 448 | 8 | fp32 | yes | 39.3 | 39.2 | 39.4 | 205.52 | 211.58 | 2.00 | 0.80 | 1.46 | 5.28 | 50 |
| ViT-B/16 | 448 | 32 | fp32 | yes | 35.5 | 35.4 | 35.5 | 897.10 | 938.21 | 3.30 | 0.80 | 2.53 | 7.87 | 30 |
| ViT-B/16 | 512 | 1 | fp32 | yes | 28.1 | 28.2 | 28.4 | 35.97 | 37.21 | 1.97 | 0.80 | 1.49 | 4.58 | 282 |
| ViT-B/16 | 512 | 8 | fp32 | yes | 27.6 | 27.5 | 27.6 | 292.20 | 310.30 | 2.03 | 0.80 | 1.46 | 5.73 | 35 |
| ViT-B/16 | 512 | 32 | fp32 | yes | 23.5 | 23.4 | 23.5 | 1355.87 | 1417.66 | 4.66 | 0.80 | 3.75 | 9.69 | 30 |
| ViT-B/16 | 224 | 1 | fp16-cast | yes | 107.9 | 108.4 | 110.8 | 9.48 | 10.00 | 0.82 | 0.80 | 0.21 | 4.21 | 1085 |
| ViT-B/16 | 224 | 8 | fp16-cast | yes | 396.3 | 396.2 | 401.8 | 19.98 | 21.39 | 0.86 | 0.80 | 0.25 | 4.27 | 496 |
| ViT-B/16 | 224 | 32 | fp16-cast | yes | 392.1 | 391.0 | 393.6 | 82.50 | 86.19 | 2.08 | 0.80 | 1.42 | 4.47 | 123 |
| ViT-B/16 | 448 | 1 | fp16-cast | yes | 92.6 | 92.9 | 94.9 | 10.99 | 11.88 | 0.84 | 0.80 | 0.21 | 4.26 | 929 |
| ViT-B/16 | 448 | 8 | fp16-cast | yes | 100.5 | 100.4 | 100.9 | 79.53 | 84.78 | 2.06 | 0.80 | 1.39 | 4.64 | 126 |
| ViT-B/16 | 448 | 32 | fp16-cast | yes | 89.2 | 88.9 | 89.3 | 359.13 | 378.12 | 2.10 | 0.80 | 1.29 | 5.94 | 30 |
| ViT-B/16 | 512 | 1 | fp16-cast | yes | 70.4 | 70.6 | 71.8 | 14.20 | 15.49 | 0.88 | 0.80 | 0.25 | 4.29 | 706 |
| ViT-B/16 | 512 | 8 | fp16-cast | yes | 74.2 | 74.1 | 74.4 | 108.99 | 113.57 | 1.97 | 0.80 | 1.29 | 4.87 | 93 |
| ViT-B/16 | 512 | 32 | fp16-cast | yes | 63.5 | 63.3 | 63.5 | 505.32 | 541.24 | 3.23 | 0.80 | 2.36 | 6.84 | 30 |
| ViT-B/16 | 224 | 1 | bf16-cast | yes | 107.6 | 108.1 | 110.6 | 9.59 | 10.06 | 0.82 | 0.80 | 0.21 | 4.21 | 1081 |
| ViT-B/16 | 224 | 8 | bf16-cast | yes | 415.4 | 415.3 | 421.4 | 18.93 | 20.67 | 0.86 | 0.80 | 0.25 | 4.27 | 520 |
| ViT-B/16 | 224 | 32 | bf16-cast | yes | 435.7 | 434.8 | 437.4 | 73.14 | 77.45 | 2.08 | 0.80 | 1.42 | 4.47 | 136 |
| ViT-B/16 | 448 | 1 | bf16-cast | yes | 96.6 | 96.9 | 99.1 | 10.09 | 11.37 | 0.84 | 0.80 | 0.21 | 4.26 | 970 |
| ViT-B/16 | 448 | 8 | bf16-cast | yes | 106.5 | 106.3 | 106.9 | 75.48 | 79.71 | 2.06 | 0.80 | 1.39 | 4.64 | 133 |
| ViT-B/16 | 448 | 32 | bf16-cast | yes | 96.6 | 96.3 | 96.7 | 336.25 | 354.54 | 2.10 | 0.80 | 1.29 | 5.94 | 31 |
| ViT-B/16 | 512 | 1 | bf16-cast | yes | 73.9 | 74.1 | 75.4 | 13.51 | 14.75 | 0.88 | 0.80 | 0.25 | 4.29 | 741 |
| ViT-B/16 | 512 | 8 | bf16-cast | yes | 80.3 | 80.2 | 80.6 | 99.44 | 106.18 | 1.97 | 0.80 | 1.29 | 4.87 | 101 |
| ViT-B/16 | 512 | 32 | bf16-cast | yes | 66.9 | 66.7 | 66.9 | 475.66 | 505.15 | 3.23 | 0.80 | 2.36 | 6.84 | 30 |
| ViT-B/16 | 224 | 1 | fp16-autocast | yes | 104.7 | 105.2 | 107.7 | 9.70 | 10.25 | 0.85 | 0.80 | 0.39 | 4.43 | 1052 |
| ViT-B/16 | 224 | 8 | fp16-autocast | yes | 332.2 | 331.5 | 336.1 | 23.64 | 26.22 | 0.93 | 0.80 | 0.45 | 4.54 | 415 |
| ViT-B/16 | 224 | 32 | fp16-autocast | yes | 347.8 | 346.3 | 349.3 | 92.63 | 97.30 | 1.99 | 0.80 | 1.45 | 4.93 | 109 |
| ViT-B/16 | 448 | 1 | fp16-autocast | yes | 74.1 | 74.1 | 75.6 | 13.57 | 15.13 | 0.89 | 0.80 | 0.41 | 4.52 | 742 |
| ViT-B/16 | 448 | 8 | fp16-autocast | yes | 85.8 | 85.5 | 86.2 | 93.22 | 99.12 | 2.01 | 0.80 | 1.45 | 5.28 | 107 |
| ViT-B/16 | 448 | 32 | fp16-autocast | yes | 76.9 | 76.6 | 77.0 | 417.91 | 444.30 | 3.30 | 0.80 | 2.53 | 7.87 | 30 |
| ViT-B/16 | 512 | 1 | fp16-autocast | yes | 57.9 | 57.9 | 58.9 | 17.30 | 19.29 | 0.93 | 0.80 | 0.45 | 4.58 | 580 |
| ViT-B/16 | 512 | 8 | fp16-autocast | yes | 61.9 | 61.7 | 62.1 | 130.07 | 139.21 | 2.03 | 0.80 | 1.45 | 5.73 | 78 |
| ViT-B/16 | 512 | 32 | fp16-autocast | yes | 54.3 | 54.1 | 54.4 | 590.03 | 611.76 | 3.40 | 0.80 | 2.53 | 9.69 | 30 |
| ViT-B/16 | 224 | 1 | bf16-autocast | yes | 106.3 | 106.8 | 109.3 | 9.70 | 10.29 | 0.85 | 0.80 | 0.39 | 4.43 | 1068 |
| ViT-B/16 | 224 | 8 | bf16-autocast | yes | 323.0 | 322.3 | 326.6 | 24.70 | 26.55 | 0.93 | 0.80 | 0.45 | 4.54 | 403 |
| ViT-B/16 | 224 | 32 | bf16-autocast | yes | 347.1 | 345.5 | 348.4 | 92.38 | 100.39 | 1.99 | 0.80 | 1.45 | 4.93 | 108 |
| ViT-B/16 | 448 | 1 | bf16-autocast | yes | 73.2 | 73.3 | 74.7 | 13.71 | 15.12 | 0.89 | 0.80 | 0.41 | 4.52 | 733 |
| ViT-B/16 | 448 | 8 | bf16-autocast | yes | 84.4 | 84.2 | 84.7 | 95.13 | 100.90 | 2.01 | 0.80 | 1.45 | 5.28 | 106 |
| ViT-B/16 | 448 | 32 | bf16-autocast | yes | 75.3 | 75.0 | 75.4 | 427.28 | 446.87 | 3.30 | 0.80 | 2.53 | 7.87 | 30 |
| ViT-B/16 | 512 | 1 | bf16-autocast | yes | 57.2 | 57.2 | 58.1 | 17.49 | 19.36 | 0.93 | 0.80 | 0.45 | 4.58 | 573 |
| ViT-B/16 | 512 | 8 | bf16-autocast | yes | 60.7 | 60.5 | 60.9 | 132.65 | 142.14 | 2.04 | 0.80 | 1.45 | 5.73 | 76 |
| ViT-B/16 | 512 | 32 | bf16-autocast | yes | 54.1 | 53.9 | 54.1 | 592.58 | 614.98 | 3.40 | 0.80 | 2.53 | 9.69 | 30 |
| ViT-L/16 | 224 | 1 | fp32 | yes | 38.5 | 38.5 | 38.9 | 26.28 | 28.40 | 2.67 | 2.61 | 2.20 | 5.48 | 386 |
| ViT-L/16 | 224 | 8 | fp32 | yes | 54.9 | 54.8 | 55.0 | 151.76 | 155.43 | 2.68 | 2.61 | 2.19 | 5.63 | 69 |
| ViT-L/16 | 224 | 32 | fp32 | yes | 50.8 | 50.7 | 50.8 | 634.36 | 647.10 | 2.75 | 2.61 | 2.19 | 6.15 | 30 |
| ViT-L/16 | 448 | 1 | fp32 | yes | 11.8 | 11.8 | 11.9 | 86.44 | 88.52 | 2.68 | 2.61 | 2.19 | 5.60 | 119 |
| ViT-L/16 | 448 | 8 | fp32 | yes | 11.6 | 11.5 | 11.6 | 694.16 | 706.14 | 2.77 | 2.61 | 2.19 | 6.61 | 30 |
| ViT-L/16 | 448 | 32 | fp32 | yes | 10.4 | 10.4 | 10.4 | 3093.15 | 3135.33 | 5.23 | 2.61 | 4.34 | 10.07 | 20 |
| ViT-L/16 | 512 | 1 | fp32 | yes | 8.5 | 8.5 | 8.5 | 119.59 | 123.03 | 2.69 | 2.61 | 2.19 | 5.67 | 85 |
| ViT-L/16 | 512 | 8 | fp32 | yes | 7.9 | 7.9 | 7.9 | 1009.76 | 1045.33 | 2.81 | 2.61 | 2.19 | 7.21 | 30 |
| ViT-L/16 | 512 | 32 | fp32 | yes | 7.3 | 7.3 | 7.3 | 4417.04 | 4460.80 | 5.62 | 2.61 | 4.58 | 12.49 | 14 |
| ViT-L/16 | 224 | 1 | fp16-cast | no: non-finite output | 75.8 | 76.1 | 77.4 | 13.11 | 14.62 | 1.71 | 1.67 | 0.65 | 4.74 | 761 |
| ViT-L/16 | 224 | 8 | fp16-cast | no: non-finite output | 137.9 | 137.9 | 138.6 | 58.45 | 61.13 | 2.83 | 1.67 | 1.76 | 4.81 | 173 |
| ViT-L/16 | 224 | 32 | fp16-cast | no: non-finite output | 127.5 | 127.3 | 127.7 | 256.15 | 270.18 | 2.85 | 1.67 | 1.72 | 5.07 | 40 |
| ViT-L/16 | 448 | 1 | fp16-cast | no: non-finite output | 32.2 | 32.3 | 32.5 | 30.92 | 33.38 | 1.76 | 1.67 | 0.68 | 4.80 | 323 |
| ViT-L/16 | 448 | 8 | fp16-cast | no: non-finite output | 32.0 | 32.0 | 32.0 | 255.27 | 264.21 | 2.86 | 1.67 | 1.72 | 5.30 | 40 |
| ViT-L/16 | 448 | 32 | fp16-cast | no: non-finite output | 28.1 | 28.1 | 28.1 | 1130.35 | 1186.27 | 4.15 | 1.67 | 2.80 | 7.04 | 30 |
| ViT-L/16 | 512 | 1 | fp16-cast | no: non-finite output | 26.4 | 26.4 | 26.6 | 38.01 | 40.25 | 1.76 | 1.67 | 0.68 | 4.84 | 264 |
| ViT-L/16 | 512 | 8 | fp16-cast | no: non-finite output | 25.3 | 25.2 | 25.3 | 319.51 | 333.39 | 2.88 | 1.67 | 1.72 | 5.61 | 32 |
| ViT-L/16 | 512 | 32 | fp16-cast | no: non-finite output | 22.3 | 22.3 | 22.3 | 1444.18 | 1495.19 | 4.22 | 1.67 | 2.80 | 8.24 | 30 |
| ViT-L/16 | 224 | 1 | bf16-cast | yes | 81.2 | 81.4 | 82.8 | 12.20 | 12.92 | 1.72 | 1.67 | 0.65 | 4.74 | 815 |
| ViT-L/16 | 224 | 8 | bf16-cast | yes | 156.7 | 156.7 | 157.6 | 51.95 | 54.96 | 2.83 | 1.67 | 1.76 | 4.81 | 196 |
| ViT-L/16 | 224 | 32 | bf16-cast | yes | 148.2 | 148.0 | 148.4 | 215.81 | 230.88 | 2.86 | 1.67 | 1.72 | 5.07 | 47 |
| ViT-L/16 | 448 | 1 | bf16-cast | yes | 34.8 | 34.8 | 35.1 | 28.70 | 31.35 | 1.76 | 1.67 | 0.68 | 4.80 | 348 |
| ViT-L/16 | 448 | 8 | bf16-cast | yes | 35.0 | 35.0 | 35.1 | 232.65 | 241.99 | 2.87 | 1.67 | 1.72 | 5.30 | 44 |
| ViT-L/16 | 448 | 32 | bf16-cast | yes | 29.5 | 29.5 | 29.5 | 1079.48 | 1126.51 | 4.15 | 1.67 | 2.80 | 7.04 | 30 |
| ViT-L/16 | 512 | 1 | bf16-cast | yes | 26.0 | 26.0 | 26.2 | 38.54 | 41.60 | 1.78 | 1.67 | 0.68 | 4.84 | 261 |
| ViT-L/16 | 512 | 8 | bf16-cast | yes | 25.1 | 25.1 | 25.2 | 322.42 | 335.25 | 2.89 | 1.67 | 1.72 | 5.61 | 32 |
| ViT-L/16 | 512 | 32 | bf16-cast | yes | 21.0 | 21.0 | 21.1 | 1517.50 | 1579.47 | 4.25 | 1.67 | 2.80 | 8.24 | 30 |
| ViT-L/16 | 224 | 1 | fp16-autocast | yes | 54.9 | 55.0 | 55.7 | 18.05 | 20.54 | 2.67 | 2.61 | 2.20 | 5.48 | 550 |
| ViT-L/16 | 224 | 8 | fp16-autocast | yes | 111.5 | 111.3 | 111.9 | 72.14 | 77.71 | 2.68 | 2.61 | 2.19 | 5.63 | 140 |
| ViT-L/16 | 224 | 32 | fp16-autocast | yes | 122.8 | 122.5 | 122.9 | 262.18 | 266.99 | 2.75 | 2.61 | 2.19 | 6.15 | 39 |
| ViT-L/16 | 448 | 1 | fp16-autocast | yes | 26.5 | 26.5 | 26.7 | 38.01 | 40.77 | 2.69 | 2.61 | 2.19 | 5.60 | 265 |
| ViT-L/16 | 448 | 8 | fp16-autocast | yes | 29.6 | 29.5 | 29.6 | 273.30 | 291.08 | 2.78 | 2.61 | 2.19 | 6.61 | 37 |
| ViT-L/16 | 448 | 32 | fp16-autocast | yes | 27.1 | 27.1 | 27.2 | 1157.49 | 1208.83 | 4.15 | 2.61 | 3.27 | 10.07 | 30 |
| ViT-L/16 | 512 | 1 | fp16-autocast | yes | 20.3 | 20.3 | 20.4 | 49.98 | 55.71 | 2.70 | 2.61 | 2.19 | 5.67 | 203 |
| ViT-L/16 | 512 | 8 | fp16-autocast | yes | 19.1 | 19.1 | 19.1 | 420.91 | 452.29 | 2.82 | 2.61 | 2.19 | 7.21 | 30 |
| ViT-L/16 | 512 | 32 | fp16-autocast | yes | 17.1 | 17.1 | 17.1 | 1881.89 | 1938.18 | 5.35 | 2.61 | 4.34 | 12.49 | 30 |
| ViT-L/16 | 224 | 1 | bf16-autocast | yes | 54.0 | 54.1 | 54.8 | 18.25 | 20.63 | 2.67 | 2.61 | 2.20 | 5.48 | 541 |
| ViT-L/16 | 224 | 8 | bf16-autocast | yes | 109.9 | 109.8 | 110.4 | 73.69 | 80.10 | 2.68 | 2.61 | 2.19 | 5.63 | 138 |
| ViT-L/16 | 224 | 32 | bf16-autocast | yes | 106.8 | 106.6 | 106.9 | 304.41 | 318.86 | 2.76 | 2.61 | 2.19 | 6.15 | 34 |
| ViT-L/16 | 448 | 1 | bf16-autocast | yes | 25.0 | 25.0 | 25.1 | 40.35 | 43.71 | 2.69 | 2.61 | 2.19 | 5.60 | 250 |
| ViT-L/16 | 448 | 8 | bf16-autocast | yes | 29.6 | 29.5 | 29.6 | 267.51 | 282.78 | 2.78 | 2.61 | 2.19 | 6.61 | 37 |
| ViT-L/16 | 448 | 32 | bf16-autocast | yes | 25.1 | 25.0 | 25.1 | 1329.73 | 1370.93 | 4.15 | 2.61 | 3.27 | 10.07 | 30 |
| ViT-L/16 | 512 | 1 | bf16-autocast | yes | 21.5 | 21.5 | 21.6 | 46.34 | 49.17 | 2.70 | 2.61 | 2.19 | 5.67 | 216 |
| ViT-L/16 | 512 | 8 | bf16-autocast | yes | 22.9 | 22.9 | 23.0 | 352.76 | 360.09 | 2.82 | 2.61 | 2.19 | 7.21 | 30 |
| ViT-L/16 | 512 | 32 | bf16-autocast | yes | 17.7 | 17.7 | 17.7 | 1794.67 | 1886.87 | 5.35 | 2.61 | 4.34 | 12.49 | 30 |
| ViT-H+/16 | 224 | 1 | fp32 | yes | 16.3 | 16.3 | 16.3 | 62.44 | 65.35 | 4.82 | 4.76 | 4.35 | 8.06 | 163 |
| ViT-H+/16 | 224 | 8 | fp32 | yes | 23.3 | 23.3 | 23.3 | 347.50 | 350.15 | 4.83 | 4.76 | 4.34 | 8.25 | 30 |
| ViT-H+/16 | 224 | 32 | fp32 | yes | 21.4 | 21.4 | 21.4 | 1456.29 | 1624.25 | 4.94 | 4.76 | 4.34 | 8.90 | 30 |
| ViT-H+/16 | 448 | 1 | fp32 | yes | 5.1 | 5.1 | 5.1 | 199.66 | 207.91 | 4.84 | 4.77 | 4.34 | 8.22 | 51 |
| ViT-H+/16 | 448 | 8 | fp32 | yes | 4.7 | 4.7 | 4.7 | 1703.34 | 1769.10 | 4.97 | 4.76 | 4.34 | 9.48 | 30 |
| ViT-H+/16 | 448 | 32 | fp32 | yes | 4.1 | 4.1 | 4.1 | 7814.99 | 7932.98 | 7.48 | 4.76 | 6.49 | 13.81 | 10 |
| ViT-H+/16 | 512 | 1 | fp32 | yes | 3.4 | 3.4 | 3.4 | 292.03 | 304.02 | 4.86 | 4.76 | 4.34 | 8.31 | 35 |
| ViT-H+/16 | 512 | 8 | fp32 | yes | 3.2 | 3.2 | 3.2 | 2549.56 | 2602.64 | 6.08 | 4.76 | 5.42 | 10.23 | 24 |
| ViT-H+/16 | 512 | 32 | fp32 | yes | 3.1 | 3.1 | 3.1 | 10312.28 | 10388.94 | 9.27 | 4.76 | 8.12 | 16.83 | 10 |
| ViT-H+/16 | 224 | 1 | fp16-cast | no: non-finite output | 39.1 | 39.2 | 39.5 | 25.62 | 28.30 | 3.69 | 3.69 | 2.23 | 6.03 | 392 |
| ViT-H+/16 | 224 | 8 | fp16-cast | no: non-finite output | 58.3 | 58.3 | 58.4 | 137.31 | 145.33 | 3.69 | 3.69 | 2.23 | 6.13 | 73 |
| ViT-H+/16 | 224 | 32 | fp16-cast | no: non-finite output | 53.3 | 53.3 | 53.4 | 596.94 | 623.34 | 3.69 | 3.69 | 2.23 | 6.45 | 30 |
| ViT-H+/16 | 448 | 1 | fp16-cast | yes | 14.3 | 14.3 | 14.3 | 69.84 | 75.08 | 3.69 | 3.69 | 2.23 | 6.11 | 143 |
| ViT-H+/16 | 448 | 8 | fp16-cast | yes | 13.5 | 13.5 | 13.5 | 588.51 | 610.10 | 3.69 | 3.69 | 2.23 | 6.74 | 30 |
| ViT-H+/16 | 448 | 32 | fp16-cast | yes | 11.8 | 11.8 | 11.9 | 2699.35 | 2738.54 | 4.50 | 3.69 | 3.30 | 8.90 | 23 |
| ViT-H+/16 | 512 | 1 | fp16-cast | yes | 10.9 | 10.9 | 10.9 | 91.36 | 97.73 | 3.69 | 3.69 | 2.23 | 6.15 | 109 |
| ViT-H+/16 | 512 | 8 | fp16-cast | yes | 9.9 | 9.9 | 9.9 | 795.82 | 842.93 | 4.31 | 3.69 | 3.30 | 7.12 | 30 |
| ViT-H+/16 | 512 | 32 | fp16-cast | yes | 8.7 | 8.7 | 8.7 | 3684.47 | 3775.57 | 5.69 | 3.69 | 4.37 | 10.41 | 17 |
| ViT-H+/16 | 224 | 1 | bf16-cast | yes | 39.5 | 39.6 | 39.9 | 25.60 | 27.86 | 3.69 | 3.69 | 2.23 | 6.03 | 396 |
| ViT-H+/16 | 224 | 8 | bf16-cast | yes | 59.5 | 59.5 | 59.6 | 133.67 | 143.87 | 3.69 | 3.69 | 2.23 | 6.13 | 75 |
| ViT-H+/16 | 224 | 32 | bf16-cast | yes | 53.9 | 53.8 | 53.9 | 595.78 | 622.36 | 3.69 | 3.69 | 2.23 | 6.45 | 30 |
| ViT-H+/16 | 448 | 1 | bf16-cast | yes | 14.1 | 14.1 | 14.2 | 70.74 | 75.58 | 3.69 | 3.69 | 2.23 | 6.11 | 142 |
| ViT-H+/16 | 448 | 8 | bf16-cast | yes | 13.8 | 13.8 | 13.8 | 578.57 | 613.58 | 3.69 | 3.69 | 2.23 | 6.74 | 30 |
| ViT-H+/16 | 448 | 32 | bf16-cast | yes | 13.0 | 13.0 | 13.0 | 2527.14 | 2709.95 | 4.51 | 3.69 | 3.30 | 8.90 | 25 |
| ViT-H+/16 | 512 | 1 | bf16-cast | yes | 11.0 | 11.0 | 11.0 | 90.83 | 96.08 | 3.69 | 3.69 | 2.23 | 6.15 | 110 |
| ViT-H+/16 | 512 | 8 | bf16-cast | yes | 10.1 | 10.1 | 10.1 | 781.23 | 829.00 | 4.31 | 3.69 | 3.30 | 7.12 | 30 |
| ViT-H+/16 | 512 | 32 | bf16-cast | yes | 9.7 | 9.7 | 9.7 | 3507.97 | 3594.97 | 5.68 | 3.69 | 4.37 | 10.41 | 19 |
| ViT-H+/16 | 224 | 1 | fp16-autocast | yes | 26.8 | 26.8 | 27.0 | 37.57 | 41.35 | 4.82 | 4.76 | 4.35 | 8.06 | 268 |
| ViT-H+/16 | 224 | 8 | fp16-autocast | yes | 48.3 | 48.2 | 48.4 | 163.79 | 180.50 | 4.84 | 4.76 | 4.34 | 8.25 | 61 |
| ViT-H+/16 | 224 | 32 | fp16-autocast | yes | 46.4 | 46.4 | 46.5 | 685.44 | 712.38 | 4.92 | 4.76 | 4.34 | 8.90 | 30 |
| ViT-H+/16 | 448 | 1 | fp16-autocast | yes | 11.0 | 11.0 | 11.1 | 90.96 | 97.64 | 4.85 | 4.76 | 4.34 | 8.22 | 111 |
| ViT-H+/16 | 448 | 8 | fp16-autocast | yes | 11.4 | 11.4 | 11.4 | 703.89 | 719.57 | 4.96 | 4.76 | 4.34 | 9.48 | 30 |
| ViT-H+/16 | 448 | 32 | fp16-autocast | yes | 10.3 | 10.3 | 10.3 | 3109.29 | 3155.43 | 7.46 | 4.76 | 6.49 | 13.81 | 20 |
| ViT-H+/16 | 512 | 1 | fp16-autocast | yes | 8.4 | 8.4 | 8.4 | 118.85 | 128.78 | 4.86 | 4.76 | 4.34 | 8.31 | 84 |
| ViT-H+/16 | 512 | 8 | fp16-autocast | yes | 8.4 | 8.4 | 8.4 | 955.63 | 978.68 | 5.00 | 4.76 | 4.34 | 10.23 | 30 |
| ViT-H+/16 | 512 | 32 | fp16-autocast | yes | 7.7 | 7.7 | 7.7 | 4181.13 | 4208.33 | 8.69 | 4.76 | 7.56 | 16.83 | 15 |
| ViT-H+/16 | 224 | 1 | bf16-autocast | yes | 26.2 | 26.2 | 26.3 | 38.55 | 42.25 | 4.82 | 4.76 | 4.35 | 8.06 | 262 |
| ViT-H+/16 | 224 | 8 | bf16-autocast | yes | 47.6 | 47.6 | 47.7 | 167.57 | 178.89 | 4.84 | 4.76 | 4.34 | 8.25 | 60 |
| ViT-H+/16 | 224 | 32 | bf16-autocast | yes | 46.1 | 46.1 | 46.2 | 689.87 | 733.20 | 4.92 | 4.76 | 4.34 | 8.90 | 30 |
| ViT-H+/16 | 448 | 1 | bf16-autocast | yes | 11.1 | 11.1 | 11.1 | 90.91 | 97.00 | 4.85 | 4.76 | 4.34 | 8.22 | 111 |
| ViT-H+/16 | 448 | 8 | bf16-autocast | yes | 11.6 | 11.6 | 11.6 | 687.04 | 726.88 | 4.96 | 4.76 | 4.34 | 9.48 | 30 |
| ViT-H+/16 | 448 | 32 | bf16-autocast | yes | 10.6 | 10.6 | 10.6 | 3011.35 | 3111.12 | 7.46 | 4.76 | 6.49 | 13.81 | 20 |
| ViT-H+/16 | 512 | 1 | bf16-autocast | yes | 8.7 | 8.7 | 8.7 | 115.67 | 122.05 | 4.86 | 4.76 | 4.34 | 8.31 | 87 |
| ViT-H+/16 | 512 | 8 | bf16-autocast | yes | 8.7 | 8.7 | 8.7 | 913.80 | 937.08 | 5.00 | 4.76 | 4.34 | 10.23 | 30 |
| ViT-H+/16 | 512 | 32 | bf16-autocast | yes | 8.0 | 7.9 | 8.0 | 4035.76 | 4056.09 | 8.69 | 4.76 | 7.56 | 16.83 | 15 |

## 4. Repeats and spread

Spread = (largest − smallest) / median of the throughputs of the main run and the repeats. Limit: 5 %.

| Priority | Output | Dtype mode | Throughputs (img/s) | Spread | Result |
| --- | --- | --- | --- | --- | --- |
| P1 | frame embedding | fp32 | 587.8, 492.1, 516.0 | 18.53 % | above 5 % |
| P1 | frame embedding | fp16-cast | 847.4, 810.6, 820.7 | 4.48 % | pass |
| P1 | frame embedding | bf16-cast | 818.7, 817.1, 820.5 | 0.42 % | pass |
| P1 | frame embedding | fp16-autocast | 733.7, 713.0, 734.3 | 2.91 % | pass |
| P1 | frame embedding | bf16-autocast | 718.4, 747.3, 775.0 | 7.58 % | above 5 % |
| P2 | patch embeddings | fp32 | 3.2, 4.0, 3.5 | 23.21 % | above 5 % |
| P2 | patch embeddings | fp16-cast | 9.9, 11.1, 10.0 | 12.23 % | above 5 % |
| P2 | patch embeddings | bf16-cast | 10.1, 10.5, 10.1 | 4.81 % | pass |
| P2 | patch embeddings | fp16-autocast | 8.4, 8.8, 8.6 | 5.05 % | above 5 % |
| P2 | patch embeddings | bf16-autocast | 8.7, 8.5, 8.9 | 3.53 % | pass |

## 5. Compile rows

9 of 10 compile rows are more than 10 % faster than the median of their eager runs. On 2026-11-06 the user decided that the compile rows join the baseline (D7).

| Priority | Dtype mode | Status | Compiled (img/s) | Eager median (img/s) | Gain | Note |
| --- | --- | --- | --- | --- | --- | --- |
| P1 | fp32 | done | 642.3 | 516.0 (3 runs) | +24.5 % | first call 1.2 s |
| P1 | fp16-cast | done | 1860.0 | 820.7 (3 runs) | +126.7 % | first call 1.0 s |
| P1 | bf16-cast | done | 1848.4 | 818.7 (3 runs) | +125.8 % | first call 1.0 s |
| P1 | fp16-autocast | done | 1502.1 | 733.7 (3 runs) | +104.7 % | first call 1.1 s |
| P1 | bf16-autocast | done | 1497.1 | 747.3 (3 runs) | +100.3 % | first call 1.1 s |
| P2 | fp32 | done | 3.8 | 3.5 (3 runs) | +9.5 % | first call 3.2 s |
| P2 | fp16-cast | done | 12.2 | 10.0 (3 runs) | +22.4 % | first call 2.4 s |
| P2 | bf16-cast | done | 12.0 | 10.1 (3 runs) | +18.8 % | first call 2.4 s |
| P2 | fp16-autocast | done | 11.1 | 8.6 (3 runs) | +30.0 % | first call 2.4 s |
| P2 | bf16-autocast | done | 11.1 | 8.7 (3 runs) | +27.2 % | first call 2.4 s |

## Skipped, deferred and error lines

| Model size | Dtype mode | Input resolution | Batch size | Compile | Repeat | Status | Need (GB) | Available (GB) | Reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ViT-7B/16 | fp16-cast | 224 | 1 | no | 0 | deferred | 20.15 | 33.77 | ViT-7B/16 runs in its own session (D8) |
| ViT-7B/16 | fp16-cast | 224 | 8 | no | 0 | deferred | 20.34 | 33.77 | ViT-7B/16 runs in its own session (D8) |
| ViT-7B/16 | fp16-cast | 224 | 32 | no | 0 | deferred | 21.02 | 33.77 | ViT-7B/16 runs in its own session (D8) |
| ViT-7B/16 | fp16-cast | 448 | 1 | no | 0 | deferred | 20.29 | 33.77 | ViT-7B/16 runs in its own session (D8) |
| ViT-7B/16 | fp16-cast | 448 | 8 | no | 0 | deferred | 21.48 | 33.77 | ViT-7B/16 runs in its own session (D8) |
| ViT-7B/16 | fp16-cast | 448 | 32 | no | 0 | deferred | 25.56 | 33.77 | ViT-7B/16 runs in its own session (D8) |
| ViT-7B/16 | fp16-cast | 512 | 1 | no | 0 | deferred | 20.37 | 33.77 | ViT-7B/16 runs in its own session (D8) |
| ViT-7B/16 | fp16-cast | 512 | 8 | no | 0 | deferred | 22.15 | 33.77 | ViT-7B/16 runs in its own session (D8) |
| ViT-7B/16 | fp16-cast | 512 | 32 | no | 0 | deferred | 28.23 | 33.77 | ViT-7B/16 runs in its own session (D8) |
| ViT-7B/16 | bf16-cast | 224 | 1 | no | 0 | deferred | 20.15 | 33.77 | ViT-7B/16 runs in its own session (D8) |
| ViT-7B/16 | bf16-cast | 224 | 8 | no | 0 | deferred | 20.34 | 33.77 | ViT-7B/16 runs in its own session (D8) |
| ViT-7B/16 | bf16-cast | 224 | 32 | no | 0 | deferred | 21.02 | 33.77 | ViT-7B/16 runs in its own session (D8) |
| ViT-7B/16 | bf16-cast | 448 | 1 | no | 0 | deferred | 20.29 | 33.77 | ViT-7B/16 runs in its own session (D8) |
| ViT-7B/16 | bf16-cast | 448 | 8 | no | 0 | deferred | 21.48 | 33.77 | ViT-7B/16 runs in its own session (D8) |
| ViT-7B/16 | bf16-cast | 448 | 32 | no | 0 | deferred | 25.56 | 33.77 | ViT-7B/16 runs in its own session (D8) |
| ViT-7B/16 | bf16-cast | 512 | 1 | no | 0 | deferred | 20.37 | 33.77 | ViT-7B/16 runs in its own session (D8) |
| ViT-7B/16 | bf16-cast | 512 | 8 | no | 0 | deferred | 22.15 | 33.77 | ViT-7B/16 runs in its own session (D8) |
| ViT-7B/16 | bf16-cast | 512 | 32 | no | 0 | deferred | 28.23 | 33.77 | ViT-7B/16 runs in its own session (D8) |

## Machine state and versions

| Item | Values |
| --- | --- |
| power_source | AC Power |
| adapter_watts | 94 |
| battery_percent | 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89 |
| low_power_mode | off |
| thermal | No thermal warning level has been recorded; No performance warning level has been recorded; No CPU power status has been recorded |
| chip | Apple M5 Max |
| memory | 64 GiB |
| macos | 26.5.2 |
| python | 3.12.13 |
| torch | 2.15.0 |
| transformers | 5.22.1 |
| mlx | 0.32.4 |
| numpy | 2.5.4 |
| safetensors | 0.8.0 |
