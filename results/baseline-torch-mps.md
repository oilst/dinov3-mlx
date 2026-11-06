# Baseline: Hugging Face backbone on torch-mps

Written by `scripts/baseline_table.py` on 2026-11-06 from `results/baseline-torch-mps.jsonl` (276 lines, 2026-11-06T17:10:21+00:00 to 2026-11-06T21:12:53+00:00). Spec: `specs/03-baseline/plan.html`. The latest line of each configuration counts.

Throughput is images per second, host to host (D3). Peak memory is the lifetime maximum of the macOS process footprint of the worker (D5). Latency is the median and the p90 of the batch times.

## Checks

- AC1: 243 of 243 configurations of B1 and B4 have a line: {'done': 225, 'deferred': 18}.
- AC3: 12 done lines of B1 and B4 are not valid. Not expected: none. Expected but valid or not run: none.
- AC4: 0 done lines miss the time budget of Table 5.
- AC7: 0 done lines have a peak memory above the need. Largest peak / need: 0.62.

## 1. Baseline rows: frame embedding

The fastest valid dtype mode of each dtype, without repeats (D2, Section 3.7 of the plan). Compile rows count since the decision of the user on 2026-11-06; only P1 and P2 have them.

| Model size | Input resolution | Batch size | Dtype | Dtype mode | Throughput (img/s) | Median (ms) | p90 (ms) | Peak memory (GB) | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ViT-S/16 | 224 | 1 | fp32 | fp32 | 121.4 | 9.19 | 9.91 | 0.56 | parity passes |
| ViT-S/16 | 224 | 1 | fp16 | fp16-autocast | 112.4 | 9.47 | 10.23 | 0.57 | parity passes |
| ViT-S/16 | 224 | 1 | bf16 | bf16-cast | 129.6 | 7.86 | 9.83 | 0.56 | parity passes |
| ViT-S/16 | 224 | 8 | fp32 | fp32 | 608.4 | 13.07 | 13.48 | 0.64 | parity passes |
| ViT-S/16 | 224 | 8 | fp16 | fp16-cast + compile | 1333.1 | 6.06 | 6.99 | 0.77 | parity passes |
| ViT-S/16 | 224 | 8 | bf16 | bf16-cast + compile | 1377.2 | 6.08 | 6.76 | 0.77 | parity passes |
| ViT-S/16 | 224 | 32 | fp32 | fp32 | 755.1 | 41.99 | 45.18 | 1.69 | parity passes |
| ViT-S/16 | 224 | 32 | fp16 | fp16-autocast | 654.4 | 48.77 | 52.07 | 1.69 | parity passes |
| ViT-S/16 | 224 | 32 | bf16 | bf16-cast | 764.1 | 41.71 | 44.97 | 1.74 | parity passes |
| ViT-S/16 | 448 | 1 | fp32 | fp32 | 100.3 | 9.84 | 10.42 | 0.60 | parity passes |
| ViT-S/16 | 448 | 1 | fp16 | fp16-autocast | 104.8 | 9.85 | 10.39 | 0.60 | parity passes |
| ViT-S/16 | 448 | 1 | bf16 | bf16-cast | 108.9 | 9.71 | 10.16 | 0.60 | parity passes |
| ViT-S/16 | 448 | 8 | fp32 | fp32 | 122.5 | 65.19 | 68.55 | 1.69 | parity passes |
| ViT-S/16 | 448 | 8 | fp16 | fp16-autocast | 160.3 | 51.70 | 58.00 | 1.69 | parity passes |
| ViT-S/16 | 448 | 8 | bf16 | bf16-cast | 198.1 | 41.11 | 43.99 | 1.74 | parity passes |
| ViT-S/16 | 448 | 32 | fp32 | fp32 | 104.5 | 306.45 | 336.00 | 1.83 | parity passes |
| ViT-S/16 | 448 | 32 | fp16 | fp16-autocast | 128.7 | 246.26 | 271.86 | 1.83 | parity passes |
| ViT-S/16 | 448 | 32 | bf16 | bf16-cast | 187.4 | 169.15 | 183.85 | 1.80 | parity passes |
| ViT-S/16 | 512 | 1 | fp32 | fp32 | 54.7 | 18.64 | 20.86 | 0.60 | parity passes |
| ViT-S/16 | 512 | 1 | fp16 | fp16-autocast | 91.6 | 10.77 | 12.03 | 0.61 | parity passes |
| ViT-S/16 | 512 | 1 | bf16 | bf16-cast | 105.9 | 9.81 | 10.18 | 0.61 | parity passes |
| ViT-S/16 | 512 | 8 | fp32 | fp32 | 49.9 | 155.41 | 204.88 | 1.70 | parity passes |
| ViT-S/16 | 512 | 8 | fp16 | fp16-autocast | 102.3 | 77.25 | 85.97 | 1.70 | parity passes |
| ViT-S/16 | 512 | 8 | bf16 | bf16-cast | 165.3 | 49.46 | 54.47 | 1.69 | parity passes |
| ViT-S/16 | 512 | 32 | fp32 | fp32 | 41.2 | 749.55 | 932.63 | 3.05 | parity passes |
| ViT-S/16 | 512 | 32 | fp16 | fp16-autocast | 104.8 | 297.99 | 343.15 | 1.89 | parity passes |
| ViT-S/16 | 512 | 32 | bf16 | bf16-cast | 148.0 | 213.40 | 231.81 | 1.85 | parity passes |
| ViT-S+/16 | 224 | 1 | fp32 | fp32 | 113.8 | 9.00 | 10.23 | 0.60 | parity passes |
| ViT-S+/16 | 224 | 1 | fp16 | fp16-cast | 126.2 | 7.83 | 9.91 | 0.61 | parity passes |
| ViT-S+/16 | 224 | 1 | bf16 | bf16-cast | 117.7 | 9.30 | 10.37 | 0.61 | parity passes |
| ViT-S+/16 | 224 | 8 | fp32 | fp32 | 327.2 | 25.45 | 27.43 | 0.67 | parity passes |
| ViT-S+/16 | 224 | 8 | fp16 | fp16-cast | 785.9 | 10.18 | 10.54 | 0.64 | parity passes |
| ViT-S+/16 | 224 | 8 | bf16 | bf16-cast | 782.8 | 10.24 | 10.64 | 0.64 | parity passes |
| ViT-S+/16 | 224 | 32 | fp32 | fp32 | 274.4 | 113.04 | 134.48 | 1.72 | parity passes |
| ViT-S+/16 | 224 | 32 | fp16 | fp16-cast | 738.8 | 44.01 | 46.94 | 1.78 | parity passes |
| ViT-S+/16 | 224 | 32 | bf16 | bf16-cast | 789.5 | 42.39 | 46.08 | 1.78 | parity passes |
| ViT-S+/16 | 448 | 1 | fp32 | fp32 | 63.6 | 15.68 | 16.58 | 0.64 | parity passes |
| ViT-S+/16 | 448 | 1 | fp16 | fp16-cast | 101.1 | 10.04 | 10.47 | 0.61 | parity passes |
| ViT-S+/16 | 448 | 1 | bf16 | bf16-cast | 102.0 | 10.01 | 10.39 | 0.61 | parity passes |
| ViT-S+/16 | 448 | 8 | fp32 | fp32 | 71.7 | 110.23 | 120.22 | 1.72 | parity passes |
| ViT-S+/16 | 448 | 8 | fp16 | fp16-cast | 194.8 | 43.23 | 46.95 | 1.78 | parity passes |
| ViT-S+/16 | 448 | 8 | bf16 | bf16-cast | 193.1 | 44.23 | 46.74 | 1.78 | parity passes |
| ViT-S+/16 | 448 | 32 | fp32 | fp32 | 65.4 | 471.33 | 549.58 | 1.87 | parity passes |
| ViT-S+/16 | 448 | 32 | fp16 | fp16-cast | 158.8 | 203.76 | 214.10 | 1.85 | parity passes |
| ViT-S+/16 | 448 | 32 | bf16 | bf16-cast | 156.3 | 203.14 | 220.32 | 1.85 | parity passes |
| ViT-S+/16 | 512 | 1 | fp32 | fp32 | 49.0 | 20.24 | 21.86 | 0.64 | parity passes |
| ViT-S+/16 | 512 | 1 | fp16 | fp16-cast | 102.0 | 10.15 | 10.57 | 0.62 | parity passes |
| ViT-S+/16 | 512 | 1 | bf16 | bf16-cast | 100.1 | 10.18 | 10.59 | 0.62 | parity passes |
| ViT-S+/16 | 512 | 8 | fp32 | fp32 | 51.4 | 152.49 | 166.21 | 1.74 | parity passes |
| ViT-S+/16 | 512 | 8 | fp16 | fp16-cast | 128.6 | 62.02 | 65.73 | 1.80 | parity passes |
| ViT-S+/16 | 512 | 8 | bf16 | bf16-cast | 119.5 | 66.89 | 71.43 | 1.80 | parity passes |
| ViT-S+/16 | 512 | 32 | fp32 | fp32 | 44.9 | 704.15 | 748.32 | 3.08 | parity passes |
| ViT-S+/16 | 512 | 32 | fp16 | fp16-cast | 114.8 | 277.54 | 294.93 | 1.90 | parity passes |
| ViT-S+/16 | 512 | 32 | bf16 | bf16-cast | 110.5 | 291.08 | 309.62 | 1.90 | parity passes |
| ViT-B/16 | 224 | 1 | fp32 | fp32 | 82.2 | 12.19 | 13.28 | 0.85 | parity passes |
| ViT-B/16 | 224 | 1 | fp16 | fp16-cast | 121.9 | 8.39 | 9.76 | 0.82 | parity passes |
| ViT-B/16 | 224 | 1 | bf16 | bf16-cast | 104.9 | 9.76 | 10.09 | 0.82 | parity passes |
| ViT-B/16 | 224 | 8 | fp32 | fp32 | 133.3 | 59.80 | 63.60 | 2.00 | parity passes |
| ViT-B/16 | 224 | 8 | fp16 | fp16-cast | 327.1 | 25.80 | 28.16 | 0.86 | parity passes |
| ViT-B/16 | 224 | 8 | bf16 | bf16-cast | 375.8 | 23.10 | 24.92 | 0.86 | parity passes |
| ViT-B/16 | 224 | 32 | fp32 | fp32 | 123.8 | 256.74 | 277.63 | 1.99 | parity passes |
| ViT-B/16 | 224 | 32 | fp16 | fp16-cast | 292.4 | 109.36 | 115.71 | 2.08 | parity passes |
| ViT-B/16 | 224 | 32 | bf16 | bf16-cast | 341.2 | 94.23 | 99.74 | 2.08 | parity passes |
| ViT-B/16 | 448 | 1 | fp32 | fp32 | 28.9 | 34.43 | 36.36 | 0.89 | parity passes |
| ViT-B/16 | 448 | 1 | fp16 | fp16-cast | 69.2 | 14.23 | 15.93 | 0.84 | parity passes |
| ViT-B/16 | 448 | 1 | bf16 | bf16-cast | 76.3 | 12.94 | 14.15 | 0.84 | parity passes |
| ViT-B/16 | 448 | 8 | fp32 | fp32 | 28.6 | 279.22 | 296.79 | 2.00 | parity passes |
| ViT-B/16 | 448 | 8 | fp16 | fp16-cast | 73.3 | 106.86 | 120.32 | 2.06 | parity passes |
| ViT-B/16 | 448 | 8 | bf16 | bf16-cast | 82.2 | 97.00 | 103.17 | 2.06 | parity passes |
| ViT-B/16 | 448 | 32 | fp32 | fp32 | 27.2 | 1180.54 | 1228.29 | 3.32 | parity passes |
| ViT-B/16 | 448 | 32 | fp16 | fp16-cast | 67.5 | 466.64 | 495.41 | 2.10 | parity passes |
| ViT-B/16 | 448 | 32 | bf16 | bf16-cast | 74.2 | 425.93 | 454.03 | 2.10 | parity passes |
| ViT-B/16 | 512 | 1 | fp32 | fp32 | 21.4 | 46.21 | 49.72 | 1.97 | parity passes |
| ViT-B/16 | 512 | 1 | fp16 | fp16-cast | 53.6 | 18.47 | 20.17 | 0.88 | parity passes |
| ViT-B/16 | 512 | 1 | bf16 | bf16-cast | 59.4 | 16.65 | 17.81 | 0.88 | parity passes |
| ViT-B/16 | 512 | 8 | fp32 | fp32 | 20.0 | 379.92 | 438.23 | 2.03 | parity passes |
| ViT-B/16 | 512 | 8 | fp16 | fp16-cast | 53.9 | 146.69 | 157.62 | 1.98 | parity passes |
| ViT-B/16 | 512 | 8 | bf16 | bf16-cast | 60.7 | 129.96 | 141.11 | 1.97 | parity passes |
| ViT-B/16 | 512 | 32 | fp32 | fp32 | 18.8 | 1691.32 | 1755.18 | 4.64 | parity passes |
| ViT-B/16 | 512 | 32 | fp16 | fp16-cast | 50.1 | 625.07 | 713.26 | 3.23 | parity passes |
| ViT-B/16 | 512 | 32 | bf16 | bf16-cast | 54.5 | 586.84 | 647.61 | 3.23 | parity passes |
| ViT-L/16 | 224 | 1 | fp32 | fp32 | 27.6 | 35.44 | 40.52 | 2.67 | parity passes |
| ViT-L/16 | 224 | 1 | fp16 | fp16-autocast | 48.8 | 20.11 | 22.79 | 2.67 | parity passes |
| ViT-L/16 | 224 | 1 | bf16 | bf16-cast | 77.6 | 12.84 | 14.02 | 1.72 | parity passes |
| ViT-L/16 | 224 | 8 | fp32 | fp32 | 41.1 | 195.80 | 202.69 | 2.68 | parity passes |
| ViT-L/16 | 224 | 8 | fp16 | fp16-autocast | 90.4 | 85.76 | 97.91 | 2.68 | parity passes |
| ViT-L/16 | 224 | 8 | bf16 | bf16-cast | 149.6 | 53.49 | 55.37 | 2.83 | parity passes |
| ViT-L/16 | 224 | 32 | fp32 | fp32 | 38.7 | 822.67 | 853.73 | 2.75 | parity passes |
| ViT-L/16 | 224 | 32 | fp16 | fp16-autocast | 84.6 | 372.18 | 421.76 | 2.76 | parity passes |
| ViT-L/16 | 224 | 32 | bf16 | bf16-cast | 147.1 | 215.00 | 228.38 | 2.86 | parity passes |
| ViT-L/16 | 448 | 1 | fp32 | fp32 | 9.5 | 105.39 | 108.93 | 2.69 | parity passes |
| ViT-L/16 | 448 | 1 | fp16 | fp16-autocast | 20.2 | 48.78 | 53.92 | 2.69 | parity passes |
| ViT-L/16 | 448 | 1 | bf16 | bf16-cast | 31.6 | 31.22 | 35.41 | 1.76 | parity passes |
| ViT-L/16 | 448 | 8 | fp32 | fp32 | 9.0 | 883.46 | 925.22 | 2.77 | parity passes |
| ViT-L/16 | 448 | 8 | fp16 | fp16-autocast | 20.5 | 388.61 | 412.02 | 2.78 | parity passes |
| ViT-L/16 | 448 | 8 | bf16 | bf16-cast | 35.7 | 223.43 | 233.19 | 2.87 | parity passes |
| ViT-L/16 | 448 | 32 | fp32 | fp32 | 8.0 | 3959.21 | 4103.39 | 5.22 | parity passes |
| ViT-L/16 | 448 | 32 | fp16 | fp16-autocast | 17.9 | 1791.30 | 1811.02 | 4.14 | parity passes |
| ViT-L/16 | 448 | 32 | bf16 | bf16-cast | 30.3 | 1008.75 | 1224.03 | 4.15 | parity passes |
| ViT-L/16 | 512 | 1 | fp32 | fp32 | 6.2 | 161.24 | 175.10 | 2.69 | parity passes |
| ViT-L/16 | 512 | 1 | fp16 | fp16-autocast | 15.7 | 62.26 | 70.71 | 2.70 | parity passes |
| ViT-L/16 | 512 | 1 | bf16 | bf16-cast | 25.7 | 39.25 | 40.51 | 1.78 | parity passes |
| ViT-L/16 | 512 | 8 | fp32 | fp32 | 6.2 | 1273.02 | 1435.58 | 2.83 | parity passes |
| ViT-L/16 | 512 | 8 | fp16 | fp16-autocast | 14.8 | 538.70 | 579.60 | 2.82 | parity passes |
| ViT-L/16 | 512 | 8 | bf16 | bf16-cast | 21.3 | 377.98 | 413.61 | 2.89 | parity passes |
| ViT-L/16 | 512 | 32 | fp32 | fp32 | 6.0 | 5358.27 | 5408.86 | 5.61 | parity passes |
| ViT-L/16 | 512 | 32 | fp16 | fp16-autocast | 13.3 | 2395.98 | 2448.17 | 5.35 | parity passes |
| ViT-L/16 | 512 | 32 | bf16 | bf16-cast | 17.3 | 1839.79 | 1908.45 | 4.24 | parity passes |
| ViT-H+/16 | 224 | 1 | fp32 | fp32 | 12.9 | 77.09 | 87.87 | 4.82 | parity passes |
| ViT-H+/16 | 224 | 1 | fp16 | fp16-autocast | 19.8 | 49.27 | 60.96 | 4.82 | parity passes |
| ViT-H+/16 | 224 | 1 | bf16 | bf16-cast | 31.3 | 30.68 | 36.46 | 3.69 | parity passes |
| ViT-H+/16 | 224 | 8 | fp32 | fp32 | 14.3 | 548.87 | 615.94 | 4.83 | parity passes |
| ViT-H+/16 | 224 | 8 | fp16 | fp16-autocast | 34.9 | 225.61 | 260.50 | 4.84 | parity passes |
| ViT-H+/16 | 224 | 8 | bf16 | bf16-cast | 47.1 | 168.26 | 182.66 | 3.69 | parity passes |
| ViT-H+/16 | 224 | 32 | fp32 | fp32 | 14.9 | 2128.12 | 2254.17 | 4.92 | parity passes |
| ViT-H+/16 | 224 | 32 | fp16 | fp16-autocast | 34.2 | 932.46 | 980.78 | 4.93 | parity passes |
| ViT-H+/16 | 224 | 32 | bf16 | bf16-cast | 45.0 | 701.85 | 743.42 | 3.69 | parity passes |
| ViT-H+/16 | 448 | 1 | fp32 | fp32 | 3.7 | 265.07 | 303.66 | 4.84 | parity passes |
| ViT-H+/16 | 448 | 1 | fp16 | fp16-cast | 10.3 | 95.43 | 108.96 | 3.69 | parity passes |
| ViT-H+/16 | 448 | 1 | bf16 | bf16-cast | 11.2 | 88.38 | 96.73 | 3.69 | parity passes |
| ViT-H+/16 | 448 | 8 | fp32 | fp32 | 3.6 | 2215.48 | 2253.68 | 4.95 | parity passes |
| ViT-H+/16 | 448 | 8 | fp16 | fp16-cast | 10.4 | 757.62 | 837.28 | 3.69 | parity passes |
| ViT-H+/16 | 448 | 8 | bf16 | bf16-cast | 11.1 | 712.36 | 734.89 | 3.69 | parity passes |
| ViT-H+/16 | 448 | 32 | fp32 | fp32 | 3.3 | 9576.36 | 9632.07 | 7.46 | parity passes |
| ViT-H+/16 | 448 | 32 | fp16 | fp16-cast | 9.4 | 3409.55 | 3458.78 | 4.50 | parity passes |
| ViT-H+/16 | 448 | 32 | bf16 | bf16-cast | 10.1 | 3168.81 | 3197.16 | 4.50 | parity passes |
| ViT-H+/16 | 512 | 1 | fp32 | fp32 | 2.6 | 379.73 | 425.09 | 4.86 | parity passes |
| ViT-H+/16 | 512 | 1 | fp16 | fp16-cast | 7.8 | 125.35 | 144.52 | 3.69 | parity passes |
| ViT-H+/16 | 512 | 1 | bf16 | bf16-cast | 8.8 | 114.32 | 120.23 | 3.69 | parity passes |
| ViT-H+/16 | 512 | 8 | fp32 | fp32 + compile | 2.9 | 2849.32 | 2913.59 | 6.35 | parity passes |
| ViT-H+/16 | 512 | 8 | fp16 | fp16-cast + compile | 10.5 | 726.15 | 892.38 | 3.69 | parity passes |
| ViT-H+/16 | 512 | 8 | bf16 | bf16-autocast + compile | 10.1 | 771.26 | 832.01 | 5.33 | parity passes |
| ViT-H+/16 | 512 | 32 | fp32 | fp32 | 2.1 | 14990.00 | 15312.99 | 9.30 | parity passes |
| ViT-H+/16 | 512 | 32 | fp16 | fp16-cast | 7.4 | 4339.55 | 4350.27 | 5.68 | parity passes |
| ViT-H+/16 | 512 | 32 | bf16 | bf16-cast | 6.7 | 4721.68 | 5215.99 | 5.73 | parity passes |
| ViT-7B/16 | 224 | 1 | fp16 | – | – | – | – | – | empty: fp16-cast: deferred, ViT-7B/16 runs in its own session (D8) |
| ViT-7B/16 | 224 | 1 | bf16 | – | – | – | – | – | empty: bf16-cast: deferred, ViT-7B/16 runs in its own session (D8) |
| ViT-7B/16 | 224 | 8 | fp16 | – | – | – | – | – | empty: fp16-cast: deferred, ViT-7B/16 runs in its own session (D8) |
| ViT-7B/16 | 224 | 8 | bf16 | – | – | – | – | – | empty: bf16-cast: deferred, ViT-7B/16 runs in its own session (D8) |
| ViT-7B/16 | 224 | 32 | fp16 | – | – | – | – | – | empty: fp16-cast: deferred, ViT-7B/16 runs in its own session (D8) |
| ViT-7B/16 | 224 | 32 | bf16 | – | – | – | – | – | empty: bf16-cast: deferred, ViT-7B/16 runs in its own session (D8) |
| ViT-7B/16 | 448 | 1 | fp16 | – | – | – | – | – | empty: fp16-cast: deferred, ViT-7B/16 runs in its own session (D8) |
| ViT-7B/16 | 448 | 1 | bf16 | – | – | – | – | – | empty: bf16-cast: deferred, ViT-7B/16 runs in its own session (D8) |
| ViT-7B/16 | 448 | 8 | fp16 | – | – | – | – | – | empty: fp16-cast: deferred, ViT-7B/16 runs in its own session (D8) |
| ViT-7B/16 | 448 | 8 | bf16 | – | – | – | – | – | empty: bf16-cast: deferred, ViT-7B/16 runs in its own session (D8) |
| ViT-7B/16 | 448 | 32 | fp16 | – | – | – | – | – | empty: fp16-cast: deferred, ViT-7B/16 runs in its own session (D8) |
| ViT-7B/16 | 448 | 32 | bf16 | – | – | – | – | – | empty: bf16-cast: deferred, ViT-7B/16 runs in its own session (D8) |
| ViT-7B/16 | 512 | 1 | fp16 | – | – | – | – | – | empty: fp16-cast: deferred, ViT-7B/16 runs in its own session (D8) |
| ViT-7B/16 | 512 | 1 | bf16 | – | – | – | – | – | empty: bf16-cast: deferred, ViT-7B/16 runs in its own session (D8) |
| ViT-7B/16 | 512 | 8 | fp16 | – | – | – | – | – | empty: fp16-cast: deferred, ViT-7B/16 runs in its own session (D8) |
| ViT-7B/16 | 512 | 8 | bf16 | – | – | – | – | – | empty: bf16-cast: deferred, ViT-7B/16 runs in its own session (D8) |
| ViT-7B/16 | 512 | 32 | fp16 | – | – | – | – | – | empty: fp16-cast: deferred, ViT-7B/16 runs in its own session (D8) |
| ViT-7B/16 | 512 | 32 | bf16 | – | – | – | – | – | empty: bf16-cast: deferred, ViT-7B/16 runs in its own session (D8) |

## 2. Baseline rows: patch embeddings

The fastest valid dtype mode of each dtype, without repeats (D2, Section 3.7 of the plan). Compile rows count since the decision of the user on 2026-11-06; only P1 and P2 have them.

| Model size | Input resolution | Batch size | Dtype | Dtype mode | Throughput (img/s) | Median (ms) | p90 (ms) | Peak memory (GB) | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ViT-S/16 | 224 | 1 | fp32 | fp32 | 122.0 | 9.15 | 9.87 | 0.56 | parity passes |
| ViT-S/16 | 224 | 1 | fp16 | fp16-autocast | 113.0 | 9.43 | 10.16 | 0.57 | parity passes |
| ViT-S/16 | 224 | 1 | bf16 | bf16-cast | 130.5 | 7.80 | 9.75 | 0.56 | parity passes |
| ViT-S/16 | 224 | 8 | fp32 | fp32 | 607.5 | 13.08 | 13.52 | 0.64 | parity passes |
| ViT-S/16 | 224 | 8 | fp16 | fp16-cast + compile | 1333.5 | 6.06 | 6.98 | 0.77 | parity passes |
| ViT-S/16 | 224 | 8 | bf16 | bf16-cast + compile | 1376.5 | 6.08 | 6.76 | 0.77 | parity passes |
| ViT-S/16 | 224 | 32 | fp32 | fp32 | 752.4 | 42.16 | 45.30 | 1.69 | parity passes |
| ViT-S/16 | 224 | 32 | fp16 | fp16-autocast | 652.4 | 48.87 | 52.24 | 1.69 | parity passes |
| ViT-S/16 | 224 | 32 | bf16 | bf16-cast | 763.5 | 41.75 | 45.03 | 1.74 | parity passes |
| ViT-S/16 | 448 | 1 | fp32 | fp32 | 100.7 | 9.83 | 10.38 | 0.60 | parity passes |
| ViT-S/16 | 448 | 1 | fp16 | fp16-autocast | 105.2 | 9.81 | 10.32 | 0.60 | parity passes |
| ViT-S/16 | 448 | 1 | bf16 | bf16-cast | 109.4 | 9.67 | 10.12 | 0.60 | parity passes |
| ViT-S/16 | 448 | 8 | fp32 | fp32 | 122.0 | 65.32 | 68.78 | 1.69 | parity passes |
| ViT-S/16 | 448 | 8 | fp16 | fp16-autocast | 159.7 | 51.81 | 58.18 | 1.69 | parity passes |
| ViT-S/16 | 448 | 8 | bf16 | bf16-cast | 197.9 | 41.15 | 44.01 | 1.74 | parity passes |
| ViT-S/16 | 448 | 32 | fp32 | fp32 | 103.5 | 307.74 | 337.37 | 1.83 | parity passes |
| ViT-S/16 | 448 | 32 | fp16 | fp16-autocast | 128.0 | 247.78 | 273.26 | 1.83 | parity passes |
| ViT-S/16 | 448 | 32 | bf16 | bf16-cast | 186.9 | 169.61 | 184.33 | 1.80 | parity passes |
| ViT-S/16 | 512 | 1 | fp32 | fp32 | 54.8 | 18.59 | 20.83 | 0.60 | parity passes |
| ViT-S/16 | 512 | 1 | fp16 | fp16-autocast | 91.9 | 10.74 | 12.00 | 0.61 | parity passes |
| ViT-S/16 | 512 | 1 | bf16 | bf16-cast | 106.3 | 9.78 | 10.15 | 0.61 | parity passes |
| ViT-S/16 | 512 | 8 | fp32 | fp32 | 49.7 | 155.56 | 206.05 | 1.70 | parity passes |
| ViT-S/16 | 512 | 8 | fp16 | fp16-autocast | 102.0 | 77.66 | 86.18 | 1.70 | parity passes |
| ViT-S/16 | 512 | 8 | bf16 | bf16-cast | 165.0 | 49.52 | 54.55 | 1.69 | parity passes |
| ViT-S/16 | 512 | 32 | fp32 | fp32 | 41.0 | 752.82 | 937.28 | 3.05 | parity passes |
| ViT-S/16 | 512 | 32 | fp16 | fp16-autocast | 104.3 | 299.11 | 346.02 | 1.89 | parity passes |
| ViT-S/16 | 512 | 32 | bf16 | bf16-cast | 147.5 | 214.14 | 232.34 | 1.85 | parity passes |
| ViT-S+/16 | 224 | 1 | fp32 | fp32 | 114.4 | 8.95 | 10.17 | 0.60 | parity passes |
| ViT-S+/16 | 224 | 1 | fp16 | fp16-cast | 126.9 | 7.79 | 9.86 | 0.61 | parity passes |
| ViT-S+/16 | 224 | 1 | bf16 | bf16-cast | 118.4 | 9.24 | 10.31 | 0.61 | parity passes |
| ViT-S+/16 | 224 | 8 | fp32 | fp32 | 327.5 | 25.47 | 27.42 | 0.67 | parity passes |
| ViT-S+/16 | 224 | 8 | fp16 | fp16-cast | 786.7 | 10.17 | 10.52 | 0.64 | parity passes |
| ViT-S+/16 | 224 | 8 | bf16 | bf16-cast | 783.1 | 10.22 | 10.64 | 0.64 | parity passes |
| ViT-S+/16 | 224 | 32 | fp32 | fp32 | 273.9 | 113.23 | 134.62 | 1.72 | parity passes |
| ViT-S+/16 | 224 | 32 | fp16 | fp16-cast | 738.2 | 44.02 | 47.00 | 1.78 | parity passes |
| ViT-S+/16 | 224 | 32 | bf16 | bf16-cast | 788.6 | 42.42 | 46.10 | 1.78 | parity passes |
| ViT-S+/16 | 448 | 1 | fp32 | fp32 | 63.8 | 15.63 | 16.53 | 0.64 | parity passes |
| ViT-S+/16 | 448 | 1 | fp16 | fp16-cast | 101.6 | 9.98 | 10.43 | 0.61 | parity passes |
| ViT-S+/16 | 448 | 1 | bf16 | bf16-cast | 102.5 | 9.97 | 10.36 | 0.61 | parity passes |
| ViT-S+/16 | 448 | 8 | fp32 | fp32 | 71.5 | 110.42 | 120.45 | 1.72 | parity passes |
| ViT-S+/16 | 448 | 8 | fp16 | fp16-cast | 194.6 | 43.30 | 46.99 | 1.78 | parity passes |
| ViT-S+/16 | 448 | 8 | bf16 | bf16-cast | 193.1 | 44.25 | 46.73 | 1.78 | parity passes |
| ViT-S+/16 | 448 | 32 | fp32 | fp32 | 65.2 | 472.26 | 550.65 | 1.87 | parity passes |
| ViT-S+/16 | 448 | 32 | fp16 | fp16-cast | 158.4 | 204.27 | 214.53 | 1.85 | parity passes |
| ViT-S+/16 | 448 | 32 | bf16 | bf16-cast | 156.0 | 203.46 | 220.83 | 1.85 | parity passes |
| ViT-S+/16 | 512 | 1 | fp32 | fp32 | 49.1 | 20.18 | 21.81 | 0.64 | parity passes |
| ViT-S+/16 | 512 | 1 | fp16 | fp16-cast | 102.4 | 10.10 | 10.54 | 0.62 | parity passes |
| ViT-S+/16 | 512 | 1 | bf16 | bf16-cast | 100.6 | 10.13 | 10.55 | 0.62 | parity passes |
| ViT-S+/16 | 512 | 8 | fp32 | fp32 | 51.3 | 152.74 | 166.62 | 1.74 | parity passes |
| ViT-S+/16 | 512 | 8 | fp16 | fp16-cast | 128.5 | 62.00 | 65.97 | 1.80 | parity passes |
| ViT-S+/16 | 512 | 8 | bf16 | bf16-cast | 119.4 | 67.03 | 71.62 | 1.80 | parity passes |
| ViT-S+/16 | 512 | 32 | fp32 | fp32 | 44.8 | 705.47 | 750.14 | 3.08 | parity passes |
| ViT-S+/16 | 512 | 32 | fp16 | fp16-cast | 114.5 | 278.19 | 295.49 | 1.90 | parity passes |
| ViT-S+/16 | 512 | 32 | bf16 | bf16-cast | 110.3 | 291.54 | 310.14 | 1.90 | parity passes |
| ViT-B/16 | 224 | 1 | fp32 | fp32 | 82.5 | 12.14 | 13.24 | 0.85 | parity passes |
| ViT-B/16 | 224 | 1 | fp16 | fp16-cast | 122.5 | 8.33 | 9.71 | 0.82 | parity passes |
| ViT-B/16 | 224 | 1 | bf16 | bf16-cast | 105.4 | 9.71 | 10.05 | 0.82 | parity passes |
| ViT-B/16 | 224 | 8 | fp32 | fp32 | 133.2 | 59.87 | 63.61 | 2.00 | parity passes |
| ViT-B/16 | 224 | 8 | fp16 | fp16-cast | 327.1 | 25.75 | 28.15 | 0.86 | parity passes |
| ViT-B/16 | 224 | 8 | bf16 | bf16-cast | 375.8 | 23.12 | 24.92 | 0.86 | parity passes |
| ViT-B/16 | 224 | 32 | fp32 | fp32 | 123.6 | 257.26 | 278.12 | 1.99 | parity passes |
| ViT-B/16 | 224 | 32 | fp16 | fp16-cast | 292.2 | 109.45 | 115.80 | 2.08 | parity passes |
| ViT-B/16 | 224 | 32 | bf16 | bf16-cast | 341.0 | 94.26 | 99.83 | 2.08 | parity passes |
| ViT-B/16 | 448 | 1 | fp32 | fp32 | 29.0 | 34.38 | 36.39 | 0.89 | parity passes |
| ViT-B/16 | 448 | 1 | fp16 | fp16-cast | 69.4 | 14.19 | 15.89 | 0.84 | parity passes |
| ViT-B/16 | 448 | 1 | bf16 | bf16-cast | 76.6 | 12.90 | 14.11 | 0.84 | parity passes |
| ViT-B/16 | 448 | 8 | fp32 | fp32 | 28.6 | 279.57 | 297.04 | 2.00 | parity passes |
| ViT-B/16 | 448 | 8 | fp16 | fp16-cast | 73.2 | 107.05 | 120.58 | 2.06 | parity passes |
| ViT-B/16 | 448 | 8 | bf16 | bf16-cast | 82.2 | 97.08 | 103.28 | 2.06 | parity passes |
| ViT-B/16 | 448 | 32 | fp32 | fp32 | 27.1 | 1182.59 | 1230.37 | 3.32 | parity passes |
| ViT-B/16 | 448 | 32 | fp16 | fp16-cast | 67.3 | 467.51 | 496.11 | 2.10 | parity passes |
| ViT-B/16 | 448 | 32 | bf16 | bf16-cast | 74.0 | 426.89 | 454.83 | 2.10 | parity passes |
| ViT-B/16 | 512 | 1 | fp32 | fp32 | 21.4 | 46.17 | 49.70 | 1.97 | parity passes |
| ViT-B/16 | 512 | 1 | fp16 | fp16-cast | 53.7 | 18.42 | 20.13 | 0.88 | parity passes |
| ViT-B/16 | 512 | 1 | bf16 | bf16-cast | 59.5 | 16.60 | 17.78 | 0.88 | parity passes |
| ViT-B/16 | 512 | 8 | fp32 | fp32 | 20.0 | 380.40 | 438.62 | 2.03 | parity passes |
| ViT-B/16 | 512 | 8 | fp16 | fp16-cast | 53.8 | 146.88 | 157.72 | 1.98 | parity passes |
| ViT-B/16 | 512 | 8 | bf16 | bf16-cast | 60.6 | 130.24 | 141.34 | 1.97 | parity passes |
| ViT-B/16 | 512 | 32 | fp32 | fp32 | 18.8 | 1693.08 | 1756.28 | 4.64 | parity passes |
| ViT-B/16 | 512 | 32 | fp16 | fp16-cast | 50.0 | 626.41 | 714.47 | 3.23 | parity passes |
| ViT-B/16 | 512 | 32 | bf16 | bf16-cast | 54.4 | 588.30 | 649.22 | 3.23 | parity passes |
| ViT-L/16 | 224 | 1 | fp32 | fp32 | 27.6 | 35.38 | 40.52 | 2.67 | parity passes |
| ViT-L/16 | 224 | 1 | fp16 | fp16-autocast | 49.0 | 20.07 | 22.75 | 2.67 | parity passes |
| ViT-L/16 | 224 | 1 | bf16 | bf16-cast | 77.7 | 12.84 | 13.99 | 1.72 | parity passes |
| ViT-L/16 | 224 | 8 | fp32 | fp32 | 41.1 | 195.80 | 202.65 | 2.68 | parity passes |
| ViT-L/16 | 224 | 8 | fp16 | fp16-autocast | 90.4 | 85.92 | 98.01 | 2.68 | parity passes |
| ViT-L/16 | 224 | 8 | bf16 | bf16-cast | 149.5 | 53.48 | 55.41 | 2.83 | parity passes |
| ViT-L/16 | 224 | 32 | fp32 | fp32 | 38.7 | 823.20 | 854.28 | 2.75 | parity passes |
| ViT-L/16 | 224 | 32 | fp16 | fp16-autocast | 84.4 | 372.82 | 422.32 | 2.76 | parity passes |
| ViT-L/16 | 224 | 32 | bf16 | bf16-cast | 146.9 | 215.23 | 228.62 | 2.86 | parity passes |
| ViT-L/16 | 448 | 1 | fp32 | fp32 | 9.5 | 105.35 | 108.88 | 2.69 | parity passes |
| ViT-L/16 | 448 | 1 | fp16 | fp16-autocast | 20.2 | 48.76 | 53.80 | 2.69 | parity passes |
| ViT-L/16 | 448 | 1 | bf16 | bf16-cast | 31.6 | 31.23 | 35.34 | 1.76 | parity passes |
| ViT-L/16 | 448 | 8 | fp32 | fp32 | 9.0 | 883.99 | 925.73 | 2.77 | parity passes |
| ViT-L/16 | 448 | 8 | fp16 | fp16-autocast | 20.4 | 389.21 | 412.66 | 2.78 | parity passes |
| ViT-L/16 | 448 | 8 | bf16 | bf16-cast | 35.6 | 223.75 | 233.42 | 2.87 | parity passes |
| ViT-L/16 | 448 | 32 | fp32 | fp32 | 8.0 | 3960.13 | 4104.19 | 5.22 | parity passes |
| ViT-L/16 | 448 | 32 | fp16 | fp16-autocast | 17.9 | 1792.29 | 1812.41 | 4.14 | parity passes |
| ViT-L/16 | 448 | 32 | bf16 | bf16-cast | 30.3 | 1009.83 | 1225.05 | 4.15 | parity passes |
| ViT-L/16 | 512 | 1 | fp32 | fp32 | 6.2 | 161.20 | 175.11 | 2.69 | parity passes |
| ViT-L/16 | 512 | 1 | fp16 | fp16-autocast | 15.8 | 62.30 | 70.44 | 2.70 | parity passes |
| ViT-L/16 | 512 | 1 | bf16 | bf16-cast | 25.7 | 39.24 | 40.54 | 1.78 | parity passes |
| ViT-L/16 | 512 | 8 | fp32 | fp32 | 6.2 | 1273.61 | 1435.28 | 2.83 | parity passes |
| ViT-L/16 | 512 | 8 | fp16 | fp16-autocast | 14.8 | 539.51 | 580.38 | 2.82 | parity passes |
| ViT-L/16 | 512 | 8 | bf16 | bf16-cast | 21.3 | 378.44 | 414.11 | 2.89 | parity passes |
| ViT-L/16 | 512 | 32 | fp32 | fp32 | 6.0 | 5360.23 | 5409.59 | 5.61 | parity passes |
| ViT-L/16 | 512 | 32 | fp16 | fp16-autocast | 13.3 | 2399.15 | 2451.90 | 5.35 | parity passes |
| ViT-L/16 | 512 | 32 | bf16 | bf16-cast | 17.3 | 1840.61 | 1909.77 | 4.24 | parity passes |
| ViT-H+/16 | 224 | 1 | fp32 | fp32 | 12.9 | 77.12 | 84.77 | 4.82 | parity passes |
| ViT-H+/16 | 224 | 1 | fp16 | fp16-autocast | 19.8 | 49.24 | 60.89 | 4.82 | parity passes |
| ViT-H+/16 | 224 | 1 | bf16 | bf16-cast | 31.4 | 30.61 | 36.40 | 3.69 | parity passes |
| ViT-H+/16 | 224 | 8 | fp32 | fp32 | 14.3 | 549.17 | 616.29 | 4.83 | parity passes |
| ViT-H+/16 | 224 | 8 | fp16 | fp16-autocast | 34.8 | 226.06 | 260.93 | 4.84 | parity passes |
| ViT-H+/16 | 224 | 8 | bf16 | bf16-cast | 47.0 | 168.46 | 182.70 | 3.69 | parity passes |
| ViT-H+/16 | 224 | 32 | fp32 | fp32 | 14.9 | 2126.98 | 2254.27 | 4.92 | parity passes |
| ViT-H+/16 | 224 | 32 | fp16 | fp16-autocast | 34.1 | 933.33 | 981.54 | 4.93 | parity passes |
| ViT-H+/16 | 224 | 32 | bf16 | bf16-cast | 44.9 | 703.30 | 744.11 | 3.69 | parity passes |
| ViT-H+/16 | 448 | 1 | fp32 | fp32 | 3.7 | 265.10 | 303.74 | 4.84 | parity passes |
| ViT-H+/16 | 448 | 1 | fp16 | fp16-cast | 10.3 | 95.38 | 108.89 | 3.69 | parity passes |
| ViT-H+/16 | 448 | 1 | bf16 | bf16-cast | 11.3 | 88.35 | 96.23 | 3.69 | parity passes |
| ViT-H+/16 | 448 | 8 | fp32 | fp32 | 3.6 | 2214.21 | 2253.28 | 4.95 | parity passes |
| ViT-H+/16 | 448 | 8 | fp16 | fp16-cast | 10.4 | 758.02 | 838.09 | 3.69 | parity passes |
| ViT-H+/16 | 448 | 8 | bf16 | bf16-cast | 11.1 | 712.74 | 735.30 | 3.69 | parity passes |
| ViT-H+/16 | 448 | 32 | fp32 | fp32 | 3.3 | 9577.50 | 9634.37 | 7.46 | parity passes |
| ViT-H+/16 | 448 | 32 | fp16 | fp16-cast | 9.4 | 3409.56 | 3458.83 | 4.50 | parity passes |
| ViT-H+/16 | 448 | 32 | bf16 | bf16-cast | 10.1 | 3168.23 | 3197.59 | 4.50 | parity passes |
| ViT-H+/16 | 512 | 1 | fp32 | fp32 | 2.6 | 379.78 | 425.13 | 4.86 | parity passes |
| ViT-H+/16 | 512 | 1 | fp16 | fp16-cast | 7.8 | 125.32 | 144.46 | 3.69 | parity passes |
| ViT-H+/16 | 512 | 1 | bf16 | bf16-cast | 8.8 | 114.35 | 120.23 | 3.69 | parity passes |
| ViT-H+/16 | 512 | 8 | fp32 | fp32 + compile | 2.9 | 2850.29 | 2914.61 | 6.35 | parity passes |
| ViT-H+/16 | 512 | 8 | fp16 | fp16-cast + compile | 10.5 | 726.69 | 892.90 | 3.69 | parity passes |
| ViT-H+/16 | 512 | 8 | bf16 | bf16-autocast + compile | 10.1 | 772.26 | 833.05 | 5.33 | parity passes |
| ViT-H+/16 | 512 | 32 | fp32 | fp32 | 2.1 | 14993.44 | 15316.10 | 9.30 | parity passes |
| ViT-H+/16 | 512 | 32 | fp16 | fp16-cast | 7.4 | 4339.17 | 4350.32 | 5.68 | parity passes |
| ViT-H+/16 | 512 | 32 | bf16 | bf16-cast | 6.7 | 4722.95 | 5217.09 | 5.73 | parity passes |
| ViT-7B/16 | 224 | 1 | fp16 | – | – | – | – | – | empty: fp16-cast: deferred, ViT-7B/16 runs in its own session (D8) |
| ViT-7B/16 | 224 | 1 | bf16 | – | – | – | – | – | empty: bf16-cast: deferred, ViT-7B/16 runs in its own session (D8) |
| ViT-7B/16 | 224 | 8 | fp16 | – | – | – | – | – | empty: fp16-cast: deferred, ViT-7B/16 runs in its own session (D8) |
| ViT-7B/16 | 224 | 8 | bf16 | – | – | – | – | – | empty: bf16-cast: deferred, ViT-7B/16 runs in its own session (D8) |
| ViT-7B/16 | 224 | 32 | fp16 | – | – | – | – | – | empty: fp16-cast: deferred, ViT-7B/16 runs in its own session (D8) |
| ViT-7B/16 | 224 | 32 | bf16 | – | – | – | – | – | empty: bf16-cast: deferred, ViT-7B/16 runs in its own session (D8) |
| ViT-7B/16 | 448 | 1 | fp16 | – | – | – | – | – | empty: fp16-cast: deferred, ViT-7B/16 runs in its own session (D8) |
| ViT-7B/16 | 448 | 1 | bf16 | – | – | – | – | – | empty: bf16-cast: deferred, ViT-7B/16 runs in its own session (D8) |
| ViT-7B/16 | 448 | 8 | fp16 | – | – | – | – | – | empty: fp16-cast: deferred, ViT-7B/16 runs in its own session (D8) |
| ViT-7B/16 | 448 | 8 | bf16 | – | – | – | – | – | empty: bf16-cast: deferred, ViT-7B/16 runs in its own session (D8) |
| ViT-7B/16 | 448 | 32 | fp16 | – | – | – | – | – | empty: fp16-cast: deferred, ViT-7B/16 runs in its own session (D8) |
| ViT-7B/16 | 448 | 32 | bf16 | – | – | – | – | – | empty: bf16-cast: deferred, ViT-7B/16 runs in its own session (D8) |
| ViT-7B/16 | 512 | 1 | fp16 | – | – | – | – | – | empty: fp16-cast: deferred, ViT-7B/16 runs in its own session (D8) |
| ViT-7B/16 | 512 | 1 | bf16 | – | – | – | – | – | empty: bf16-cast: deferred, ViT-7B/16 runs in its own session (D8) |
| ViT-7B/16 | 512 | 8 | fp16 | – | – | – | – | – | empty: fp16-cast: deferred, ViT-7B/16 runs in its own session (D8) |
| ViT-7B/16 | 512 | 8 | bf16 | – | – | – | – | – | empty: bf16-cast: deferred, ViT-7B/16 runs in its own session (D8) |
| ViT-7B/16 | 512 | 32 | fp16 | – | – | – | – | – | empty: fp16-cast: deferred, ViT-7B/16 runs in its own session (D8) |
| ViT-7B/16 | 512 | 32 | bf16 | – | – | – | – | – | empty: bf16-cast: deferred, ViT-7B/16 runs in its own session (D8) |

## 3. All dtype modes

| Model size | Input resolution | Batch size | Dtype mode | Valid | Frame (img/s) | Patch (img/s) | Device (img/s) | Frame median (ms) | Frame p90 (ms) | Peak (GB) | Load peak (GB) | MPS driver (GB) | Need (GB) | Timed batches |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ViT-S/16 | 224 | 1 | fp32 | yes | 121.4 | 122.0 | 125.2 | 9.19 | 9.91 | 0.56 | 0.52 | 0.10 | 4.11 | 1221 |
| ViT-S/16 | 224 | 8 | fp32 | yes | 608.4 | 607.5 | 619.7 | 13.07 | 13.48 | 0.64 | 0.52 | 0.17 | 4.17 | 760 |
| ViT-S/16 | 224 | 32 | fp32 | yes | 755.1 | 752.4 | 761.7 | 41.99 | 45.18 | 1.69 | 0.52 | 1.18 | 4.36 | 236 |
| ViT-S/16 | 448 | 1 | fp32 | yes | 100.3 | 100.7 | 103.3 | 9.84 | 10.42 | 0.60 | 0.52 | 0.14 | 4.16 | 1007 |
| ViT-S/16 | 448 | 8 | fp32 | yes | 122.5 | 122.0 | 123.2 | 65.19 | 68.55 | 1.69 | 0.52 | 1.18 | 4.54 | 153 |
| ViT-S/16 | 448 | 32 | fp32 | yes | 104.5 | 103.5 | 104.7 | 306.45 | 336.00 | 1.83 | 0.52 | 1.18 | 5.84 | 33 |
| ViT-S/16 | 512 | 1 | fp32 | yes | 54.7 | 54.8 | 55.6 | 18.64 | 20.86 | 0.60 | 0.52 | 0.14 | 4.19 | 548 |
| ViT-S/16 | 512 | 8 | fp32 | yes | 49.9 | 49.7 | 50.1 | 155.41 | 204.88 | 1.70 | 0.52 | 1.18 | 4.76 | 63 |
| ViT-S/16 | 512 | 32 | fp32 | yes | 41.2 | 41.0 | 41.2 | 749.55 | 932.63 | 3.05 | 0.52 | 2.33 | 6.74 | 30 |
| ViT-S/16 | 224 | 1 | fp16-cast | yes | 104.7 | 105.5 | 108.3 | 9.52 | 12.12 | 0.56 | 0.51 | 0.06 | 4.06 | 1055 |
| ViT-S/16 | 224 | 8 | fp16-cast | yes | 382.9 | 383.5 | 390.3 | 18.99 | 27.65 | 0.59 | 0.51 | 0.09 | 4.08 | 480 |
| ViT-S/16 | 224 | 32 | fp16-cast | yes | 367.3 | 367.5 | 371.0 | 66.14 | 146.35 | 1.74 | 0.51 | 1.19 | 4.18 | 115 |
| ViT-S/16 | 448 | 1 | fp16-cast | yes | 72.6 | 73.0 | 74.4 | 12.44 | 18.57 | 0.60 | 0.51 | 0.10 | 4.08 | 730 |
| ViT-S/16 | 448 | 8 | fp16-cast | yes | 91.5 | 91.5 | 92.1 | 75.38 | 97.72 | 1.74 | 0.51 | 1.19 | 4.27 | 115 |
| ViT-S/16 | 448 | 32 | fp16-cast | yes | 86.4 | 86.3 | 86.5 | 309.29 | 472.42 | 1.80 | 0.51 | 1.13 | 4.92 | 30 |
| ViT-S/16 | 512 | 1 | fp16-cast | yes | 61.5 | 61.8 | 63.1 | 15.17 | 20.60 | 0.61 | 0.51 | 0.10 | 4.09 | 618 |
| ViT-S/16 | 512 | 8 | fp16-cast | yes | 80.6 | 80.5 | 81.0 | 97.23 | 124.00 | 1.68 | 0.51 | 1.13 | 4.38 | 101 |
| ViT-S/16 | 512 | 32 | fp16-cast | yes | 82.0 | 81.8 | 82.1 | 385.14 | 497.96 | 1.86 | 0.51 | 1.13 | 5.37 | 30 |
| ViT-S/16 | 224 | 1 | bf16-cast | yes | 129.6 | 130.5 | 134.2 | 7.86 | 9.83 | 0.56 | 0.51 | 0.06 | 4.06 | 1306 |
| ViT-S/16 | 224 | 8 | bf16-cast | yes | 487.4 | 487.6 | 496.4 | 15.71 | 20.05 | 0.59 | 0.51 | 0.09 | 4.08 | 610 |
| ViT-S/16 | 224 | 32 | bf16-cast | yes | 764.1 | 763.5 | 769.4 | 41.71 | 44.97 | 1.74 | 0.51 | 1.19 | 4.18 | 239 |
| ViT-S/16 | 448 | 1 | bf16-cast | yes | 108.9 | 109.4 | 111.9 | 9.71 | 10.16 | 0.60 | 0.51 | 0.10 | 4.08 | 1094 |
| ViT-S/16 | 448 | 8 | bf16-cast | yes | 198.1 | 197.9 | 199.5 | 41.11 | 43.99 | 1.74 | 0.51 | 1.19 | 4.27 | 248 |
| ViT-S/16 | 448 | 32 | bf16-cast | yes | 187.4 | 186.9 | 187.9 | 169.15 | 183.85 | 1.80 | 0.51 | 1.13 | 4.92 | 59 |
| ViT-S/16 | 512 | 1 | bf16-cast | yes | 105.9 | 106.3 | 108.8 | 9.81 | 10.18 | 0.61 | 0.51 | 0.10 | 4.09 | 1064 |
| ViT-S/16 | 512 | 8 | bf16-cast | yes | 165.3 | 165.0 | 166.3 | 49.46 | 54.47 | 1.69 | 0.51 | 1.13 | 4.38 | 207 |
| ViT-S/16 | 512 | 32 | bf16-cast | yes | 148.0 | 147.5 | 148.2 | 213.40 | 231.81 | 1.85 | 0.51 | 1.13 | 5.37 | 47 |
| ViT-S/16 | 224 | 1 | fp16-autocast | yes | 112.4 | 113.0 | 115.8 | 9.47 | 10.23 | 0.57 | 0.52 | 0.11 | 4.11 | 1131 |
| ViT-S/16 | 224 | 8 | fp16-autocast | yes | 702.5 | 701.8 | 719.4 | 10.79 | 13.68 | 0.60 | 0.52 | 0.14 | 4.17 | 878 |
| ViT-S/16 | 224 | 32 | fp16-autocast | yes | 654.4 | 652.4 | 659.4 | 48.77 | 52.07 | 1.69 | 0.52 | 1.18 | 4.36 | 204 |
| ViT-S/16 | 448 | 1 | fp16-autocast | yes | 104.8 | 105.2 | 107.8 | 9.85 | 10.39 | 0.60 | 0.52 | 0.14 | 4.16 | 1053 |
| ViT-S/16 | 448 | 8 | fp16-autocast | yes | 160.3 | 159.7 | 161.5 | 51.70 | 58.00 | 1.69 | 0.52 | 1.18 | 4.54 | 200 |
| ViT-S/16 | 448 | 32 | fp16-autocast | yes | 128.7 | 128.0 | 128.9 | 246.26 | 271.86 | 1.83 | 0.52 | 1.18 | 5.84 | 41 |
| ViT-S/16 | 512 | 1 | fp16-autocast | yes | 91.6 | 91.9 | 94.2 | 10.77 | 12.03 | 0.61 | 0.52 | 0.15 | 4.19 | 920 |
| ViT-S/16 | 512 | 8 | fp16-autocast | yes | 102.3 | 102.0 | 102.9 | 77.25 | 85.97 | 1.70 | 0.52 | 1.18 | 4.76 | 128 |
| ViT-S/16 | 512 | 32 | fp16-autocast | yes | 104.8 | 104.3 | 105.0 | 297.99 | 343.15 | 1.89 | 0.52 | 1.18 | 6.74 | 33 |
| ViT-S/16 | 224 | 1 | bf16-autocast | yes | 109.9 | 110.6 | 113.2 | 9.51 | 10.42 | 0.57 | 0.52 | 0.11 | 4.11 | 1106 |
| ViT-S/16 | 224 | 8 | bf16-autocast | yes | 608.8 | 608.8 | 622.5 | 12.85 | 16.93 | 0.60 | 0.52 | 0.14 | 4.17 | 762 |
| ViT-S/16 | 224 | 32 | bf16-autocast | yes | 530.6 | 529.4 | 534.0 | 59.45 | 65.21 | 1.69 | 0.52 | 1.18 | 4.36 | 166 |
| ViT-S/16 | 448 | 1 | bf16-autocast | yes | 100.8 | 101.1 | 103.7 | 10.06 | 11.07 | 0.61 | 0.53 | 0.14 | 4.16 | 1012 |
| ViT-S/16 | 448 | 8 | bf16-autocast | yes | 132.2 | 132.0 | 133.0 | 59.64 | 67.09 | 1.69 | 0.52 | 1.18 | 4.54 | 165 |
| ViT-S/16 | 448 | 32 | bf16-autocast | yes | 129.3 | 128.8 | 129.6 | 240.25 | 281.31 | 1.83 | 0.52 | 1.18 | 5.84 | 41 |
| ViT-S/16 | 512 | 1 | bf16-autocast | yes | 90.0 | 90.3 | 92.5 | 11.09 | 12.54 | 0.61 | 0.52 | 0.15 | 4.19 | 904 |
| ViT-S/16 | 512 | 8 | bf16-autocast | yes | 102.3 | 102.1 | 102.9 | 76.82 | 86.14 | 1.70 | 0.52 | 1.18 | 4.76 | 128 |
| ViT-S/16 | 512 | 32 | bf16-autocast | yes | 99.5 | 99.1 | 99.6 | 319.11 | 354.77 | 1.89 | 0.52 | 1.18 | 6.74 | 31 |
| ViT-S+/16 | 224 | 1 | fp32 | yes | 113.8 | 114.4 | 117.2 | 9.00 | 10.23 | 0.60 | 0.56 | 0.14 | 4.15 | 1145 |
| ViT-S+/16 | 224 | 8 | fp32 | yes | 327.2 | 327.5 | 331.4 | 25.45 | 27.43 | 0.67 | 0.56 | 0.20 | 4.20 | 410 |
| ViT-S+/16 | 224 | 32 | fp32 | yes | 274.4 | 273.9 | 275.3 | 113.04 | 134.48 | 1.72 | 0.56 | 1.21 | 4.40 | 86 |
| ViT-S+/16 | 448 | 1 | fp32 | yes | 63.6 | 63.8 | 64.8 | 15.68 | 16.58 | 0.64 | 0.56 | 0.17 | 4.19 | 639 |
| ViT-S+/16 | 448 | 8 | fp32 | yes | 71.7 | 71.5 | 71.9 | 110.23 | 120.22 | 1.72 | 0.56 | 1.21 | 4.57 | 90 |
| ViT-S+/16 | 448 | 32 | fp32 | yes | 65.4 | 65.2 | 65.4 | 471.33 | 549.58 | 1.87 | 0.56 | 1.21 | 5.87 | 30 |
| ViT-S+/16 | 512 | 1 | fp32 | yes | 49.0 | 49.1 | 49.7 | 20.24 | 21.86 | 0.64 | 0.56 | 0.17 | 4.22 | 491 |
| ViT-S+/16 | 512 | 8 | fp32 | yes | 51.4 | 51.3 | 51.5 | 152.49 | 166.21 | 1.74 | 0.56 | 1.21 | 4.80 | 65 |
| ViT-S+/16 | 512 | 32 | fp32 | yes | 44.9 | 44.8 | 45.0 | 704.15 | 748.32 | 3.08 | 0.56 | 2.36 | 6.78 | 30 |
| ViT-S+/16 | 224 | 1 | fp16-cast | yes | 126.2 | 126.9 | 130.1 | 7.83 | 9.91 | 0.61 | 0.56 | 0.10 | 4.07 | 1269 |
| ViT-S+/16 | 224 | 8 | fp16-cast | yes | 785.9 | 786.7 | 805.7 | 10.18 | 10.54 | 0.64 | 0.56 | 0.12 | 4.10 | 984 |
| ViT-S+/16 | 224 | 32 | fp16-cast | yes | 738.8 | 738.2 | 744.0 | 44.01 | 46.94 | 1.78 | 0.56 | 1.23 | 4.20 | 231 |
| ViT-S+/16 | 448 | 1 | fp16-cast | yes | 101.1 | 101.6 | 103.9 | 10.04 | 10.47 | 0.61 | 0.56 | 0.10 | 4.10 | 1017 |
| ViT-S+/16 | 448 | 8 | fp16-cast | yes | 194.8 | 194.6 | 196.3 | 43.23 | 46.95 | 1.78 | 0.56 | 1.23 | 4.29 | 244 |
| ViT-S+/16 | 448 | 32 | fp16-cast | yes | 158.8 | 158.4 | 159.1 | 203.76 | 214.10 | 1.85 | 0.56 | 1.16 | 4.93 | 50 |
| ViT-S+/16 | 512 | 1 | fp16-cast | yes | 102.0 | 102.4 | 104.8 | 10.15 | 10.57 | 0.62 | 0.56 | 0.10 | 4.11 | 1025 |
| ViT-S+/16 | 512 | 8 | fp16-cast | yes | 128.6 | 128.5 | 129.3 | 62.02 | 65.73 | 1.80 | 0.56 | 1.23 | 4.40 | 161 |
| ViT-S+/16 | 512 | 32 | fp16-cast | yes | 114.8 | 114.5 | 115.0 | 277.54 | 294.93 | 1.90 | 0.56 | 1.16 | 5.39 | 36 |
| ViT-S+/16 | 224 | 1 | bf16-cast | yes | 117.7 | 118.4 | 121.3 | 9.30 | 10.37 | 0.61 | 0.56 | 0.10 | 4.07 | 1184 |
| ViT-S+/16 | 224 | 8 | bf16-cast | yes | 782.8 | 783.1 | 802.4 | 10.24 | 10.64 | 0.64 | 0.56 | 0.12 | 4.10 | 979 |
| ViT-S+/16 | 224 | 32 | bf16-cast | yes | 789.5 | 788.6 | 795.4 | 42.39 | 46.08 | 1.78 | 0.56 | 1.23 | 4.20 | 247 |
| ViT-S+/16 | 448 | 1 | bf16-cast | yes | 102.0 | 102.5 | 104.7 | 10.01 | 10.39 | 0.61 | 0.56 | 0.10 | 4.10 | 1026 |
| ViT-S+/16 | 448 | 8 | bf16-cast | yes | 193.1 | 193.1 | 194.6 | 44.23 | 46.74 | 1.78 | 0.56 | 1.23 | 4.29 | 242 |
| ViT-S+/16 | 448 | 32 | bf16-cast | yes | 156.3 | 156.0 | 156.6 | 203.14 | 220.32 | 1.85 | 0.56 | 1.16 | 4.93 | 49 |
| ViT-S+/16 | 512 | 1 | bf16-cast | yes | 100.1 | 100.6 | 102.8 | 10.18 | 10.59 | 0.62 | 0.56 | 0.10 | 4.11 | 1006 |
| ViT-S+/16 | 512 | 8 | bf16-cast | yes | 119.5 | 119.4 | 120.1 | 66.89 | 71.43 | 1.80 | 0.56 | 1.23 | 4.40 | 150 |
| ViT-S+/16 | 512 | 32 | bf16-cast | yes | 110.5 | 110.3 | 110.6 | 291.08 | 309.62 | 1.90 | 0.56 | 1.16 | 5.39 | 35 |
| ViT-S+/16 | 224 | 1 | fp16-autocast | yes | 108.3 | 108.9 | 111.4 | 9.87 | 10.77 | 0.61 | 0.56 | 0.15 | 4.15 | 1089 |
| ViT-S+/16 | 224 | 8 | fp16-autocast | yes | 550.3 | 550.4 | 561.8 | 13.84 | 17.89 | 0.64 | 0.56 | 0.17 | 4.20 | 688 |
| ViT-S+/16 | 224 | 32 | fp16-autocast | yes | 490.6 | 490.0 | 493.5 | 64.94 | 70.10 | 1.72 | 0.56 | 1.21 | 4.40 | 154 |
| ViT-S+/16 | 448 | 1 | fp16-autocast | yes | 92.4 | 92.8 | 95.0 | 10.72 | 12.02 | 0.64 | 0.56 | 0.18 | 4.19 | 928 |
| ViT-S+/16 | 448 | 8 | fp16-autocast | yes | 122.3 | 122.2 | 123.0 | 64.95 | 69.91 | 1.72 | 0.56 | 1.21 | 4.57 | 153 |
| ViT-S+/16 | 448 | 32 | fp16-autocast | yes | 115.5 | 115.0 | 115.6 | 269.62 | 311.65 | 1.86 | 0.56 | 1.21 | 5.87 | 36 |
| ViT-S+/16 | 512 | 1 | fp16-autocast | yes | 80.3 | 80.6 | 82.3 | 12.54 | 13.50 | 0.65 | 0.56 | 0.18 | 4.22 | 806 |
| ViT-S+/16 | 512 | 8 | fp16-autocast | yes | 94.1 | 94.0 | 94.6 | 83.21 | 94.26 | 1.74 | 0.56 | 1.21 | 4.80 | 118 |
| ViT-S+/16 | 512 | 32 | fp16-autocast | yes | 87.8 | 87.5 | 87.9 | 357.74 | 387.96 | 1.92 | 0.56 | 1.21 | 6.78 | 30 |
| ViT-S+/16 | 224 | 1 | bf16-autocast | yes | 106.6 | 107.2 | 109.6 | 9.58 | 10.86 | 0.61 | 0.56 | 0.15 | 4.15 | 1072 |
| ViT-S+/16 | 224 | 8 | bf16-autocast | yes | 553.2 | 552.9 | 563.6 | 14.66 | 16.94 | 0.64 | 0.56 | 0.17 | 4.20 | 692 |
| ViT-S+/16 | 224 | 32 | bf16-autocast | yes | 509.5 | 509.2 | 512.5 | 61.75 | 69.49 | 1.72 | 0.56 | 1.21 | 4.40 | 160 |
| ViT-S+/16 | 448 | 1 | bf16-autocast | yes | 98.3 | 98.6 | 100.9 | 10.18 | 10.77 | 0.64 | 0.56 | 0.18 | 4.19 | 987 |
| ViT-S+/16 | 448 | 8 | bf16-autocast | yes | 130.3 | 130.2 | 131.1 | 60.95 | 64.84 | 1.72 | 0.56 | 1.21 | 4.57 | 163 |
| ViT-S+/16 | 448 | 32 | bf16-autocast | yes | 114.4 | 114.0 | 114.5 | 277.42 | 293.26 | 1.86 | 0.56 | 1.21 | 5.87 | 36 |
| ViT-S+/16 | 512 | 1 | bf16-autocast | yes | 82.9 | 83.1 | 84.9 | 12.04 | 13.99 | 0.65 | 0.56 | 0.18 | 4.22 | 832 |
| ViT-S+/16 | 512 | 8 | bf16-autocast | yes | 98.4 | 98.3 | 99.0 | 80.77 | 87.81 | 1.74 | 0.56 | 1.21 | 4.80 | 123 |
| ViT-S+/16 | 512 | 32 | bf16-autocast | yes | 94.2 | 93.8 | 94.3 | 329.54 | 393.51 | 1.92 | 0.56 | 1.21 | 6.78 | 30 |
| ViT-B/16 | 224 | 1 | fp32 | yes | 82.2 | 82.5 | 84.2 | 12.19 | 13.28 | 0.85 | 0.80 | 0.39 | 4.43 | 826 |
| ViT-B/16 | 224 | 8 | fp32 | yes | 133.3 | 133.2 | 134.0 | 59.80 | 63.60 | 2.00 | 0.80 | 1.52 | 4.54 | 167 |
| ViT-B/16 | 224 | 32 | fp32 | yes | 123.8 | 123.6 | 124.0 | 256.74 | 277.63 | 1.99 | 0.80 | 1.46 | 4.93 | 39 |
| ViT-B/16 | 448 | 1 | fp32 | yes | 28.9 | 29.0 | 29.2 | 34.43 | 36.36 | 0.89 | 0.80 | 0.42 | 4.52 | 290 |
| ViT-B/16 | 448 | 8 | fp32 | yes | 28.6 | 28.6 | 28.6 | 279.22 | 296.79 | 2.00 | 0.80 | 1.46 | 5.28 | 36 |
| ViT-B/16 | 448 | 32 | fp32 | yes | 27.2 | 27.1 | 27.2 | 1180.54 | 1228.29 | 3.32 | 0.80 | 2.53 | 7.87 | 30 |
| ViT-B/16 | 512 | 1 | fp32 | yes | 21.4 | 21.4 | 21.6 | 46.21 | 49.72 | 1.97 | 0.80 | 1.49 | 4.58 | 215 |
| ViT-B/16 | 512 | 8 | fp32 | yes | 20.0 | 20.0 | 20.0 | 379.92 | 438.23 | 2.03 | 0.80 | 1.46 | 5.73 | 30 |
| ViT-B/16 | 512 | 32 | fp32 | yes | 18.8 | 18.8 | 18.8 | 1691.32 | 1755.18 | 4.64 | 0.80 | 3.75 | 9.69 | 30 |
| ViT-B/16 | 224 | 1 | fp16-cast | yes | 121.9 | 122.5 | 125.6 | 8.39 | 9.76 | 0.82 | 0.80 | 0.21 | 4.21 | 1226 |
| ViT-B/16 | 224 | 8 | fp16-cast | yes | 327.1 | 327.1 | 331.0 | 25.80 | 28.16 | 0.86 | 0.80 | 0.25 | 4.27 | 409 |
| ViT-B/16 | 224 | 32 | fp16-cast | yes | 292.4 | 292.2 | 293.4 | 109.36 | 115.71 | 2.08 | 0.80 | 1.42 | 4.47 | 92 |
| ViT-B/16 | 448 | 1 | fp16-cast | yes | 69.2 | 69.4 | 70.6 | 14.23 | 15.93 | 0.84 | 0.80 | 0.21 | 4.26 | 695 |
| ViT-B/16 | 448 | 8 | fp16-cast | yes | 73.3 | 73.2 | 73.6 | 106.86 | 120.32 | 2.06 | 0.80 | 1.39 | 4.64 | 92 |
| ViT-B/16 | 448 | 32 | fp16-cast | yes | 67.5 | 67.3 | 67.5 | 466.64 | 495.41 | 2.10 | 0.80 | 1.29 | 5.94 | 30 |
| ViT-B/16 | 512 | 1 | fp16-cast | yes | 53.6 | 53.7 | 54.5 | 18.47 | 20.17 | 0.88 | 0.80 | 0.25 | 4.29 | 538 |
| ViT-B/16 | 512 | 8 | fp16-cast | yes | 53.9 | 53.8 | 54.0 | 146.69 | 157.62 | 1.98 | 0.80 | 1.29 | 4.87 | 68 |
| ViT-B/16 | 512 | 32 | fp16-cast | yes | 50.1 | 50.0 | 50.1 | 625.07 | 713.26 | 3.23 | 0.80 | 2.36 | 6.84 | 30 |
| ViT-B/16 | 224 | 1 | bf16-cast | yes | 104.9 | 105.4 | 107.8 | 9.76 | 10.09 | 0.82 | 0.80 | 0.21 | 4.21 | 1055 |
| ViT-B/16 | 224 | 8 | bf16-cast | yes | 375.8 | 375.8 | 380.8 | 23.10 | 24.92 | 0.86 | 0.80 | 0.25 | 4.27 | 470 |
| ViT-B/16 | 224 | 32 | bf16-cast | yes | 341.2 | 341.0 | 342.5 | 94.23 | 99.74 | 2.08 | 0.80 | 1.42 | 4.47 | 107 |
| ViT-B/16 | 448 | 1 | bf16-cast | yes | 76.3 | 76.6 | 78.1 | 12.94 | 14.15 | 0.84 | 0.80 | 0.21 | 4.26 | 766 |
| ViT-B/16 | 448 | 8 | bf16-cast | yes | 82.2 | 82.2 | 82.5 | 97.00 | 103.17 | 2.06 | 0.80 | 1.39 | 4.64 | 103 |
| ViT-B/16 | 448 | 32 | bf16-cast | yes | 74.2 | 74.0 | 74.3 | 425.93 | 454.03 | 2.10 | 0.80 | 1.29 | 5.94 | 30 |
| ViT-B/16 | 512 | 1 | bf16-cast | yes | 59.4 | 59.5 | 60.5 | 16.65 | 17.81 | 0.88 | 0.80 | 0.25 | 4.29 | 596 |
| ViT-B/16 | 512 | 8 | bf16-cast | yes | 60.7 | 60.6 | 60.9 | 129.96 | 141.11 | 1.97 | 0.80 | 1.29 | 4.87 | 76 |
| ViT-B/16 | 512 | 32 | bf16-cast | yes | 54.5 | 54.4 | 54.5 | 586.84 | 647.61 | 3.23 | 0.80 | 2.36 | 6.84 | 30 |
| ViT-B/16 | 224 | 1 | fp16-autocast | yes | 105.8 | 106.3 | 108.8 | 9.58 | 10.24 | 0.85 | 0.80 | 0.39 | 4.43 | 1063 |
| ViT-B/16 | 224 | 8 | fp16-autocast | yes | 298.9 | 298.6 | 302.3 | 28.07 | 31.92 | 0.93 | 0.80 | 0.45 | 4.54 | 374 |
| ViT-B/16 | 224 | 32 | fp16-autocast | yes | 272.4 | 271.9 | 273.3 | 117.31 | 122.60 | 1.99 | 0.80 | 1.45 | 4.93 | 85 |
| ViT-B/16 | 448 | 1 | fp16-autocast | yes | 60.8 | 60.9 | 62.0 | 16.19 | 17.75 | 0.89 | 0.80 | 0.41 | 4.52 | 610 |
| ViT-B/16 | 448 | 8 | fp16-autocast | yes | 66.5 | 66.4 | 66.7 | 120.61 | 127.29 | 2.01 | 0.80 | 1.45 | 5.28 | 83 |
| ViT-B/16 | 448 | 32 | fp16-autocast | yes | 60.8 | 60.6 | 60.8 | 519.37 | 560.32 | 3.30 | 0.80 | 2.53 | 7.87 | 30 |
| ViT-B/16 | 512 | 1 | fp16-autocast | yes | 46.8 | 46.8 | 47.5 | 20.87 | 23.57 | 0.93 | 0.80 | 0.45 | 4.58 | 469 |
| ViT-B/16 | 512 | 8 | fp16-autocast | yes | 47.7 | 47.6 | 47.8 | 166.73 | 178.35 | 2.03 | 0.80 | 1.45 | 5.73 | 60 |
| ViT-B/16 | 512 | 32 | fp16-autocast | yes | 45.4 | 45.2 | 45.4 | 702.96 | 732.96 | 3.40 | 0.80 | 2.53 | 9.69 | 30 |
| ViT-B/16 | 224 | 1 | bf16-autocast | yes | 104.5 | 105.0 | 107.6 | 9.72 | 10.31 | 0.85 | 0.80 | 0.39 | 4.43 | 1051 |
| ViT-B/16 | 224 | 8 | bf16-autocast | yes | 287.7 | 286.9 | 290.7 | 28.41 | 35.41 | 0.93 | 0.80 | 0.45 | 4.54 | 359 |
| ViT-B/16 | 224 | 32 | bf16-autocast | yes | 268.9 | 268.4 | 269.7 | 119.64 | 124.17 | 1.99 | 0.80 | 1.45 | 4.93 | 84 |
| ViT-B/16 | 448 | 1 | bf16-autocast | yes | 60.6 | 60.6 | 61.7 | 16.34 | 17.82 | 0.89 | 0.80 | 0.41 | 4.52 | 607 |
| ViT-B/16 | 448 | 8 | bf16-autocast | yes | 64.4 | 64.2 | 64.6 | 120.95 | 136.57 | 2.01 | 0.80 | 1.45 | 5.28 | 81 |
| ViT-B/16 | 448 | 32 | bf16-autocast | yes | 60.8 | 60.6 | 60.8 | 520.69 | 558.69 | 3.30 | 0.80 | 2.53 | 7.87 | 30 |
| ViT-B/16 | 512 | 1 | bf16-autocast | yes | 41.6 | 41.7 | 42.2 | 23.57 | 27.93 | 0.93 | 0.80 | 0.45 | 4.58 | 417 |
| ViT-B/16 | 512 | 8 | bf16-autocast | yes | 43.4 | 43.3 | 43.5 | 176.50 | 207.35 | 2.03 | 0.80 | 1.45 | 5.73 | 55 |
| ViT-B/16 | 512 | 32 | bf16-autocast | yes | 42.0 | 41.9 | 42.1 | 747.49 | 792.39 | 3.40 | 0.80 | 2.53 | 9.69 | 30 |
| ViT-L/16 | 224 | 1 | fp32 | yes | 27.6 | 27.6 | 27.8 | 35.44 | 40.52 | 2.67 | 2.61 | 2.20 | 5.48 | 277 |
| ViT-L/16 | 224 | 8 | fp32 | yes | 41.1 | 41.1 | 41.2 | 195.80 | 202.69 | 2.68 | 2.61 | 2.19 | 5.63 | 52 |
| ViT-L/16 | 224 | 32 | fp32 | yes | 38.7 | 38.7 | 38.8 | 822.67 | 853.73 | 2.75 | 2.61 | 2.19 | 6.15 | 30 |
| ViT-L/16 | 448 | 1 | fp32 | yes | 9.5 | 9.5 | 9.6 | 105.39 | 108.93 | 2.69 | 2.62 | 2.19 | 5.60 | 96 |
| ViT-L/16 | 448 | 8 | fp32 | yes | 9.0 | 9.0 | 9.0 | 883.46 | 925.22 | 2.77 | 2.61 | 2.19 | 6.61 | 30 |
| ViT-L/16 | 448 | 32 | fp32 | yes | 8.0 | 8.0 | 8.0 | 3959.21 | 4103.39 | 5.22 | 2.61 | 4.34 | 10.07 | 16 |
| ViT-L/16 | 512 | 1 | fp32 | yes | 6.2 | 6.2 | 6.2 | 161.24 | 175.10 | 2.69 | 2.61 | 2.19 | 5.67 | 63 |
| ViT-L/16 | 512 | 8 | fp32 | yes | 6.2 | 6.2 | 6.2 | 1273.02 | 1435.58 | 2.83 | 2.61 | 2.19 | 7.21 | 30 |
| ViT-L/16 | 512 | 32 | fp32 | yes | 6.0 | 6.0 | 6.0 | 5358.27 | 5408.86 | 5.61 | 2.61 | 4.58 | 12.49 | 12 |
| ViT-L/16 | 224 | 1 | fp16-cast | no: non-finite output | 62.3 | 62.4 | 63.3 | 16.31 | 17.57 | 1.71 | 1.67 | 0.65 | 4.74 | 625 |
| ViT-L/16 | 224 | 8 | fp16-cast | no: non-finite output | 110.1 | 110.0 | 110.5 | 72.84 | 76.24 | 2.83 | 1.67 | 1.76 | 4.81 | 138 |
| ViT-L/16 | 224 | 32 | fp16-cast | no: non-finite output | 104.0 | 103.9 | 104.1 | 308.89 | 345.24 | 2.85 | 1.67 | 1.72 | 5.07 | 33 |
| ViT-L/16 | 448 | 1 | fp16-cast | no: non-finite output | 25.2 | 25.2 | 25.4 | 39.48 | 42.88 | 1.76 | 1.67 | 0.68 | 4.80 | 252 |
| ViT-L/16 | 448 | 8 | fp16-cast | no: non-finite output | 24.9 | 24.9 | 25.0 | 321.77 | 356.10 | 2.86 | 1.67 | 1.72 | 5.30 | 32 |
| ViT-L/16 | 448 | 32 | fp16-cast | no: non-finite output | 22.9 | 22.9 | 23.0 | 1398.33 | 1515.35 | 4.13 | 1.67 | 2.80 | 7.04 | 30 |
| ViT-L/16 | 512 | 1 | fp16-cast | no: non-finite output | 20.6 | 20.6 | 20.7 | 46.95 | 55.36 | 1.76 | 1.67 | 0.68 | 4.84 | 206 |
| ViT-L/16 | 512 | 8 | fp16-cast | no: non-finite output | 20.2 | 20.2 | 20.2 | 407.33 | 440.34 | 2.88 | 1.67 | 1.72 | 5.61 | 30 |
| ViT-L/16 | 512 | 32 | fp16-cast | no: non-finite output | 22.3 | 22.3 | 22.4 | 1413.04 | 1560.46 | 4.23 | 1.67 | 2.80 | 8.24 | 30 |
| ViT-L/16 | 224 | 1 | bf16-cast | yes | 77.6 | 77.7 | 79.1 | 12.84 | 14.02 | 1.72 | 1.67 | 0.65 | 4.74 | 778 |
| ViT-L/16 | 224 | 8 | bf16-cast | yes | 149.6 | 149.5 | 150.3 | 53.49 | 55.37 | 2.83 | 1.67 | 1.76 | 4.81 | 187 |
| ViT-L/16 | 224 | 32 | bf16-cast | yes | 147.1 | 146.9 | 147.3 | 215.00 | 228.38 | 2.86 | 1.67 | 1.72 | 5.07 | 46 |
| ViT-L/16 | 448 | 1 | bf16-cast | yes | 31.6 | 31.6 | 31.9 | 31.22 | 35.41 | 1.76 | 1.67 | 0.68 | 4.80 | 317 |
| ViT-L/16 | 448 | 8 | bf16-cast | yes | 35.7 | 35.6 | 35.7 | 223.43 | 233.19 | 2.87 | 1.67 | 1.72 | 5.30 | 45 |
| ViT-L/16 | 448 | 32 | bf16-cast | yes | 30.3 | 30.3 | 30.3 | 1008.75 | 1224.03 | 4.15 | 1.67 | 2.80 | 7.04 | 30 |
| ViT-L/16 | 512 | 1 | bf16-cast | yes | 25.7 | 25.7 | 25.9 | 39.25 | 40.51 | 1.78 | 1.67 | 0.68 | 4.84 | 257 |
| ViT-L/16 | 512 | 8 | bf16-cast | yes | 21.3 | 21.3 | 21.3 | 377.98 | 413.61 | 2.89 | 1.67 | 1.72 | 5.61 | 30 |
| ViT-L/16 | 512 | 32 | bf16-cast | yes | 17.3 | 17.3 | 17.4 | 1839.79 | 1908.45 | 4.24 | 1.67 | 2.80 | 8.24 | 30 |
| ViT-L/16 | 224 | 1 | fp16-autocast | yes | 48.8 | 49.0 | 49.6 | 20.11 | 22.79 | 2.67 | 2.61 | 2.20 | 5.48 | 490 |
| ViT-L/16 | 224 | 8 | fp16-autocast | yes | 90.4 | 90.4 | 90.8 | 85.76 | 97.91 | 2.68 | 2.61 | 2.19 | 5.63 | 113 |
| ViT-L/16 | 224 | 32 | fp16-autocast | yes | 84.6 | 84.4 | 84.6 | 372.18 | 421.76 | 2.76 | 2.61 | 2.19 | 6.15 | 30 |
| ViT-L/16 | 448 | 1 | fp16-autocast | yes | 20.2 | 20.2 | 20.3 | 48.78 | 53.92 | 2.69 | 2.61 | 2.19 | 5.60 | 203 |
| ViT-L/16 | 448 | 8 | fp16-autocast | yes | 20.5 | 20.4 | 20.5 | 388.61 | 412.02 | 2.78 | 2.61 | 2.19 | 6.61 | 30 |
| ViT-L/16 | 448 | 32 | fp16-autocast | yes | 17.9 | 17.9 | 17.9 | 1791.30 | 1811.02 | 4.14 | 2.62 | 3.27 | 10.07 | 30 |
| ViT-L/16 | 512 | 1 | fp16-autocast | yes | 15.7 | 15.8 | 15.8 | 62.26 | 70.71 | 2.70 | 2.62 | 2.19 | 5.67 | 158 |
| ViT-L/16 | 512 | 8 | fp16-autocast | yes | 14.8 | 14.8 | 14.8 | 538.70 | 579.60 | 2.82 | 2.61 | 2.19 | 7.21 | 30 |
| ViT-L/16 | 512 | 32 | fp16-autocast | yes | 13.3 | 13.3 | 13.3 | 2395.98 | 2448.17 | 5.35 | 2.61 | 4.34 | 12.49 | 25 |
| ViT-L/16 | 224 | 1 | bf16-autocast | yes | 45.3 | 45.5 | 46.1 | 21.48 | 25.25 | 2.67 | 2.61 | 2.20 | 5.48 | 455 |
| ViT-L/16 | 224 | 8 | bf16-autocast | yes | 87.6 | 87.4 | 87.8 | 89.21 | 102.55 | 2.68 | 2.62 | 2.19 | 5.63 | 110 |
| ViT-L/16 | 224 | 32 | bf16-autocast | yes | 83.9 | 83.7 | 84.0 | 377.31 | 417.17 | 2.76 | 2.61 | 2.19 | 6.15 | 30 |
| ViT-L/16 | 448 | 1 | bf16-autocast | yes | 19.9 | 19.9 | 20.1 | 48.55 | 56.37 | 2.69 | 2.61 | 2.19 | 5.60 | 200 |
| ViT-L/16 | 448 | 8 | bf16-autocast | yes | 20.4 | 20.4 | 20.4 | 379.11 | 427.27 | 2.78 | 2.61 | 2.19 | 6.61 | 30 |
| ViT-L/16 | 448 | 32 | bf16-autocast | yes | 18.3 | 18.2 | 18.3 | 1758.75 | 1803.83 | 4.14 | 2.61 | 3.27 | 10.07 | 30 |
| ViT-L/16 | 512 | 1 | bf16-autocast | yes | 16.1 | 16.2 | 16.2 | 60.12 | 68.66 | 2.70 | 2.62 | 2.19 | 5.67 | 162 |
| ViT-L/16 | 512 | 8 | bf16-autocast | yes | 15.4 | 15.4 | 15.4 | 517.54 | 554.95 | 2.82 | 2.61 | 2.19 | 7.21 | 30 |
| ViT-L/16 | 512 | 32 | bf16-autocast | yes | 14.0 | 13.9 | 14.0 | 2293.42 | 2352.03 | 5.36 | 2.62 | 4.34 | 12.49 | 27 |
| ViT-H+/16 | 224 | 1 | fp32 | yes | 12.9 | 12.9 | 12.9 | 77.09 | 87.87 | 4.82 | 4.77 | 4.35 | 8.06 | 129 |
| ViT-H+/16 | 224 | 8 | fp32 | yes | 14.3 | 14.3 | 14.4 | 548.87 | 615.94 | 4.83 | 4.77 | 4.34 | 8.25 | 30 |
| ViT-H+/16 | 224 | 32 | fp32 | yes | 14.9 | 14.9 | 14.9 | 2128.12 | 2254.17 | 4.92 | 4.77 | 4.34 | 8.90 | 28 |
| ViT-H+/16 | 448 | 1 | fp32 | yes | 3.7 | 3.7 | 3.7 | 265.07 | 303.66 | 4.84 | 4.76 | 4.34 | 8.22 | 37 |
| ViT-H+/16 | 448 | 8 | fp32 | yes | 3.6 | 3.6 | 3.6 | 2215.48 | 2253.68 | 4.95 | 4.76 | 4.34 | 9.48 | 27 |
| ViT-H+/16 | 448 | 32 | fp32 | yes | 3.3 | 3.3 | 3.3 | 9576.36 | 9632.07 | 7.46 | 4.77 | 6.49 | 13.81 | 10 |
| ViT-H+/16 | 512 | 1 | fp32 | yes | 2.6 | 2.6 | 2.6 | 379.73 | 425.09 | 4.86 | 4.77 | 4.34 | 8.31 | 30 |
| ViT-H+/16 | 512 | 8 | fp32 | yes | 2.6 | 2.6 | 2.6 | 3112.72 | 3142.07 | 6.08 | 4.77 | 5.42 | 10.23 | 20 |
| ViT-H+/16 | 512 | 32 | fp32 | yes | 2.1 | 2.1 | 2.1 | 14990.00 | 15312.99 | 9.30 | 4.76 | 8.12 | 16.83 | 10 |
| ViT-H+/16 | 224 | 1 | fp16-cast | no: non-finite output | 29.1 | 29.1 | 29.4 | 33.14 | 40.02 | 3.69 | 3.69 | 2.23 | 6.03 | 292 |
| ViT-H+/16 | 224 | 8 | fp16-cast | no: non-finite output | 43.2 | 43.2 | 43.3 | 182.12 | 202.61 | 3.69 | 3.69 | 2.23 | 6.13 | 54 |
| ViT-H+/16 | 224 | 32 | fp16-cast | no: non-finite output | 41.9 | 41.9 | 41.9 | 753.79 | 804.96 | 3.69 | 3.69 | 2.23 | 6.45 | 30 |
| ViT-H+/16 | 448 | 1 | fp16-cast | yes | 10.3 | 10.3 | 10.3 | 95.43 | 108.96 | 3.69 | 3.69 | 2.23 | 6.11 | 103 |
| ViT-H+/16 | 448 | 8 | fp16-cast | yes | 10.4 | 10.4 | 10.4 | 757.62 | 837.28 | 3.69 | 3.69 | 2.23 | 6.74 | 30 |
| ViT-H+/16 | 448 | 32 | fp16-cast | yes | 9.4 | 9.4 | 9.4 | 3409.55 | 3458.78 | 4.50 | 3.69 | 3.30 | 8.90 | 18 |
| ViT-H+/16 | 512 | 1 | fp16-cast | yes | 7.8 | 7.8 | 7.8 | 125.35 | 144.52 | 3.69 | 3.69 | 2.23 | 6.15 | 78 |
| ViT-H+/16 | 512 | 8 | fp16-cast | yes | 8.2 | 8.2 | 8.2 | 970.99 | 1019.97 | 4.32 | 3.69 | 3.30 | 7.12 | 30 |
| ViT-H+/16 | 512 | 32 | fp16-cast | yes | 7.4 | 7.4 | 7.4 | 4339.55 | 4350.27 | 5.68 | 3.69 | 4.37 | 10.41 | 14 |
| ViT-H+/16 | 224 | 1 | bf16-cast | yes | 31.3 | 31.4 | 31.7 | 30.68 | 36.46 | 3.69 | 3.69 | 2.23 | 6.03 | 315 |
| ViT-H+/16 | 224 | 8 | bf16-cast | yes | 47.1 | 47.0 | 47.2 | 168.26 | 182.66 | 3.69 | 3.69 | 2.23 | 6.13 | 59 |
| ViT-H+/16 | 224 | 32 | bf16-cast | yes | 45.0 | 44.9 | 45.0 | 701.85 | 743.42 | 3.69 | 3.69 | 2.23 | 6.45 | 30 |
| ViT-H+/16 | 448 | 1 | bf16-cast | yes | 11.2 | 11.3 | 11.3 | 88.38 | 96.73 | 3.69 | 3.69 | 2.23 | 6.11 | 113 |
| ViT-H+/16 | 448 | 8 | bf16-cast | yes | 11.1 | 11.1 | 11.1 | 712.36 | 734.89 | 3.69 | 3.69 | 2.23 | 6.74 | 30 |
| ViT-H+/16 | 448 | 32 | bf16-cast | yes | 10.1 | 10.1 | 10.1 | 3168.81 | 3197.16 | 4.50 | 3.69 | 3.30 | 8.90 | 19 |
| ViT-H+/16 | 512 | 1 | bf16-cast | yes | 8.8 | 8.8 | 8.9 | 114.32 | 120.23 | 3.69 | 3.69 | 2.23 | 6.15 | 89 |
| ViT-H+/16 | 512 | 8 | bf16-cast | yes | 8.3 | 8.3 | 8.3 | 958.61 | 1043.99 | 4.31 | 3.69 | 3.30 | 7.12 | 30 |
| ViT-H+/16 | 512 | 32 | bf16-cast | yes | 6.7 | 6.7 | 6.7 | 4721.68 | 5215.99 | 5.73 | 3.69 | 4.37 | 10.41 | 13 |
| ViT-H+/16 | 224 | 1 | fp16-autocast | yes | 19.8 | 19.8 | 19.9 | 49.27 | 60.96 | 4.82 | 4.76 | 4.35 | 8.06 | 198 |
| ViT-H+/16 | 224 | 8 | fp16-autocast | yes | 34.9 | 34.8 | 34.9 | 225.61 | 260.50 | 4.84 | 4.76 | 4.34 | 8.25 | 44 |
| ViT-H+/16 | 224 | 32 | fp16-autocast | yes | 34.2 | 34.1 | 34.2 | 932.46 | 980.78 | 4.93 | 4.76 | 4.34 | 8.90 | 30 |
| ViT-H+/16 | 448 | 1 | fp16-autocast | yes | 7.9 | 7.9 | 8.0 | 125.40 | 134.80 | 4.85 | 4.76 | 4.34 | 8.22 | 80 |
| ViT-H+/16 | 448 | 8 | fp16-autocast | yes | 8.4 | 8.4 | 8.4 | 948.49 | 980.91 | 4.96 | 4.76 | 4.34 | 9.48 | 30 |
| ViT-H+/16 | 448 | 32 | fp16-autocast | yes | 7.6 | 7.6 | 7.6 | 4207.30 | 4266.19 | 7.46 | 4.76 | 6.49 | 13.81 | 15 |
| ViT-H+/16 | 512 | 1 | fp16-autocast | yes | 6.0 | 6.0 | 6.1 | 163.17 | 179.73 | 4.86 | 4.76 | 4.34 | 8.31 | 61 |
| ViT-H+/16 | 512 | 8 | fp16-autocast | yes | 6.2 | 6.2 | 6.2 | 1276.95 | 1341.18 | 5.02 | 4.77 | 4.34 | 10.23 | 30 |
| ViT-H+/16 | 512 | 32 | fp16-autocast | yes | 5.7 | 5.7 | 5.7 | 5562.42 | 8018.56 | 8.69 | 4.77 | 7.56 | 16.83 | 12 |
| ViT-H+/16 | 224 | 1 | bf16-autocast | yes | 18.6 | 18.6 | 18.7 | 50.92 | 66.92 | 4.82 | 4.76 | 4.35 | 8.06 | 186 |
| ViT-H+/16 | 224 | 8 | bf16-autocast | yes | 33.6 | 33.6 | 33.7 | 236.99 | 252.91 | 4.84 | 4.76 | 4.34 | 8.25 | 42 |
| ViT-H+/16 | 224 | 32 | bf16-autocast | yes | 33.7 | 33.7 | 33.7 | 940.73 | 990.19 | 4.93 | 4.76 | 4.34 | 8.90 | 30 |
| ViT-H+/16 | 448 | 1 | bf16-autocast | yes | 7.8 | 7.8 | 7.9 | 125.72 | 139.04 | 4.85 | 4.77 | 4.34 | 8.22 | 79 |
| ViT-H+/16 | 448 | 8 | bf16-autocast | yes | 8.4 | 8.4 | 8.4 | 943.70 | 990.19 | 4.96 | 4.76 | 4.34 | 9.48 | 30 |
| ViT-H+/16 | 448 | 32 | bf16-autocast | yes | 7.7 | 7.7 | 7.7 | 4129.67 | 4222.51 | 7.47 | 4.76 | 6.49 | 13.81 | 15 |
| ViT-H+/16 | 512 | 1 | bf16-autocast | yes | 6.1 | 6.1 | 6.2 | 162.01 | 176.45 | 4.86 | 4.77 | 4.34 | 8.31 | 62 |
| ViT-H+/16 | 512 | 8 | bf16-autocast | yes | 6.3 | 6.3 | 6.3 | 1260.19 | 1324.64 | 5.01 | 4.76 | 4.34 | 10.23 | 30 |
| ViT-H+/16 | 512 | 32 | bf16-autocast | yes | 5.7 | 5.7 | 5.7 | 5580.86 | 5981.40 | 8.70 | 4.76 | 7.56 | 16.83 | 11 |

## 4. Repeats and spread

Spread = (largest − smallest) / median of the throughputs of the main run and the repeats. Limit: 5 %.

| Priority | Output | Dtype mode | Throughputs (img/s) | Spread | Result |
| --- | --- | --- | --- | --- | --- |
| P1 | frame embedding | fp32 | 608.4, 319.6, 334.1 | 86.44 % | above 5 % |
| P1 | frame embedding | fp16-cast | 382.9, 671.4, 704.6 | 47.91 % | above 5 % |
| P1 | frame embedding | bf16-cast | 487.4, 658.4, 706.9 | 33.34 % | above 5 % |
| P1 | frame embedding | fp16-autocast | 702.5, 519.0, 536.4 | 34.20 % | above 5 % |
| P1 | frame embedding | bf16-autocast | 608.8, 515.2, 531.7 | 17.60 % | above 5 % |
| P2 | patch embeddings | fp32 | 2.6, 2.4, 2.6 | 7.87 % | above 5 % |
| P2 | patch embeddings | fp16-cast | 8.2, 6.9, 7.6 | 16.04 % | above 5 % |
| P2 | patch embeddings | bf16-cast | 8.3, 6.9, 7.6 | 18.01 % | above 5 % |
| P2 | patch embeddings | fp16-autocast | 6.2, 5.9, 7.6 | 27.59 % | above 5 % |
| P2 | patch embeddings | bf16-autocast | 6.3, 5.9, 7.7 | 28.34 % | above 5 % |

## 5. Compile rows

10 of 10 compile rows are more than 10 % faster than the median of their eager runs. On 2026-11-06 the user decided that the compile rows join the baseline (D7).

| Priority | Dtype mode | Status | Compiled (img/s) | Eager median (img/s) | Gain | Note |
| --- | --- | --- | --- | --- | --- | --- |
| P1 | fp32 | done | 471.1 | 334.1 (3 runs) | +41.0 % | first call 4.9 s |
| P1 | fp16-cast | done | 1333.1 | 671.4 (3 runs) | +98.6 % | first call 4.4 s |
| P1 | bf16-cast | done | 1377.2 | 658.4 (3 runs) | +109.2 % | first call 4.3 s |
| P1 | fp16-autocast | done | 1108.9 | 536.4 (3 runs) | +106.8 % | first call 5.1 s |
| P1 | bf16-autocast | done | 1342.4 | 531.7 (3 runs) | +152.5 % | first call 4.9 s |
| P2 | fp32 | done | 2.9 | 2.6 (3 runs) | +11.3 % | first call 11.7 s |
| P2 | fp16-cast | done | 10.5 | 7.6 (3 runs) | +37.4 % | first call 9.6 s |
| P2 | bf16-cast | done | 9.0 | 7.6 (3 runs) | +19.0 % | first call 10.0 s |
| P2 | fp16-autocast | done | 10.4 | 6.2 (3 runs) | +67.9 % | first call 12.6 s |
| P2 | bf16-autocast | done | 10.1 | 6.3 (3 runs) | +60.0 % | first call 12.7 s |

## Skipped, deferred and error lines

| Model size | Dtype mode | Input resolution | Batch size | Compile | Repeat | Status | Need (GB) | Available (GB) | Reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ViT-7B/16 | fp16-cast | 224 | 1 | no | 0 | deferred | 20.15 | 20.60 | ViT-7B/16 runs in its own session (D8) |
| ViT-7B/16 | fp16-cast | 224 | 8 | no | 0 | deferred | 20.34 | 20.60 | ViT-7B/16 runs in its own session (D8) |
| ViT-7B/16 | fp16-cast | 224 | 32 | no | 0 | deferred | 21.02 | 20.60 | ViT-7B/16 runs in its own session (D8) |
| ViT-7B/16 | fp16-cast | 448 | 1 | no | 0 | deferred | 20.29 | 20.60 | ViT-7B/16 runs in its own session (D8) |
| ViT-7B/16 | fp16-cast | 448 | 8 | no | 0 | deferred | 21.48 | 20.60 | ViT-7B/16 runs in its own session (D8) |
| ViT-7B/16 | fp16-cast | 448 | 32 | no | 0 | deferred | 25.56 | 20.60 | ViT-7B/16 runs in its own session (D8) |
| ViT-7B/16 | fp16-cast | 512 | 1 | no | 0 | deferred | 20.37 | 20.60 | ViT-7B/16 runs in its own session (D8) |
| ViT-7B/16 | fp16-cast | 512 | 8 | no | 0 | deferred | 22.15 | 20.60 | ViT-7B/16 runs in its own session (D8) |
| ViT-7B/16 | fp16-cast | 512 | 32 | no | 0 | deferred | 28.23 | 20.60 | ViT-7B/16 runs in its own session (D8) |
| ViT-7B/16 | bf16-cast | 224 | 1 | no | 0 | deferred | 20.15 | 20.60 | ViT-7B/16 runs in its own session (D8) |
| ViT-7B/16 | bf16-cast | 224 | 8 | no | 0 | deferred | 20.34 | 20.60 | ViT-7B/16 runs in its own session (D8) |
| ViT-7B/16 | bf16-cast | 224 | 32 | no | 0 | deferred | 21.02 | 20.60 | ViT-7B/16 runs in its own session (D8) |
| ViT-7B/16 | bf16-cast | 448 | 1 | no | 0 | deferred | 20.29 | 20.60 | ViT-7B/16 runs in its own session (D8) |
| ViT-7B/16 | bf16-cast | 448 | 8 | no | 0 | deferred | 21.48 | 20.60 | ViT-7B/16 runs in its own session (D8) |
| ViT-7B/16 | bf16-cast | 448 | 32 | no | 0 | deferred | 25.56 | 20.60 | ViT-7B/16 runs in its own session (D8) |
| ViT-7B/16 | bf16-cast | 512 | 1 | no | 0 | deferred | 20.37 | 20.60 | ViT-7B/16 runs in its own session (D8) |
| ViT-7B/16 | bf16-cast | 512 | 8 | no | 0 | deferred | 22.15 | 20.60 | ViT-7B/16 runs in its own session (D8) |
| ViT-7B/16 | bf16-cast | 512 | 32 | no | 0 | deferred | 28.23 | 20.60 | ViT-7B/16 runs in its own session (D8) |

## Machine state and versions

| Item | Values |
| --- | --- |
| power_source | AC Power |
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
