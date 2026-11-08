# Core ML on the Neural Engine

Made by `scripts/bench_coreml.py table` from `results/coreml.jsonl` on 2026-11-08. Step 6c (specs/06c-neural-engine/plan.html). Core ML packages run with fp16 compute precision; the mlx side runs the compiled forward pass in the fastest valid fp16 dtype mode of Step 6. Parity: frame / patch mean / patch 1st percentile, the worst of the 3 test images (parity) or image 0 (speed).

## 1 Conversion

| Package | Status | Trace s | Convert s | Save s | Compile s | Size MB | Ops | Peak footprint GB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| vits16-224-b1-ne-s256-r1-ct9.0.mlpackage | done | 0.4 | 2.2 | 0.0 | 0.1 | 43.5 | 1325 | 0.52 |
| vits16-448-b1-ne-s256-r1-ct9.0.mlpackage | done | 0.6 | 2.2 | 0.0 | 0.1 | 43.6 | 1325 | 0.57 |
| vits16-512-b1-ne-s256-r1-ct9.0.mlpackage | done | 0.8 | 2.2 | 0.0 | 0.1 | 43.7 | 1325 | 0.62 |
| vits16-224-b8-ne-s256-r1-ct9.0.mlpackage | done | 0.7 | 2.2 | 0.0 | 0.1 | 43.5 | 1325 | 0.60 |
| vits16-512-b8-ne-s256-r1-ct9.0.mlpackage | done | 3.9 | 2.2 | 0.0 | 0.1 | 43.7 | 1325 | 1.55 |
| vits16plus-224-b1-ne-s256-r1-ct9.0.mlpackage | done | 0.4 | 2.3 | 0.0 | 0.1 | 57.7 | 1361 | 0.54 |
| vits16plus-448-b1-ne-s256-r1-ct9.0.mlpackage | done | 0.6 | 2.2 | 0.0 | 0.1 | 57.8 | 1361 | 0.59 |
| vits16plus-512-b1-ne-s256-r1-ct9.0.mlpackage | done | 0.8 | 2.3 | 0.0 | 0.1 | 57.9 | 1361 | 0.65 |
| vits16plus-224-b8-ne-s256-r1-ct9.0.mlpackage | done | 0.8 | 2.3 | 0.0 | 0.1 | 57.7 | 1361 | 0.63 |
| vits16plus-512-b8-ne-s256-r1-ct9.0.mlpackage | done | 4.3 | 2.3 | 0.1 | 0.1 | 57.9 | 1361 | 1.60 |
| vitb16-224-b1-ne-s256-r1-ct9.0.mlpackage | done | 0.5 | 2.9 | 0.0 | 0.1 | 171.6 | 1325 | 0.88 |
| vitb16-448-b1-ne-s256-r1-ct9.0.mlpackage | done | 1.1 | 3.0 | 0.0 | 0.1 | 171.7 | 1325 | 0.96 |
| vitb16-512-b1-ne-s256-r1-ct9.0.mlpackage | done | 1.4 | 3.0 | 0.0 | 0.1 | 171.8 | 1325 | 1.06 |
| vitb16-224-b8-ne-s256-r1-ct9.0.mlpackage | done | 1.5 | 3.2 | 0.0 | 0.1 | 171.6 | 1325 | 0.94 |
| vitb16-512-b8-ne-s256-r1-ct9.0.mlpackage | done | 8.8 | 3.0 | 0.0 | 0.1 | 171.8 | 1325 | 2.80 |
| vitl16-224-b1-ne-s256-r8-ct9.0.mlpackage | done | 1.9 | 8.3 | 0.1 | 0.2 | 606.7 | 2623 | 2.20 |
| vitl16-448-b1-ne-s256-r8-ct9.0.mlpackage | done | 3.1 | 7.7 | 0.1 | 0.1 | 606.8 | 2623 | 2.21 |
| vitl16-512-b1-ne-s256-r8-ct9.0.mlpackage | done | 4.1 | 7.9 | 0.1 | 0.2 | 606.9 | 2623 | 2.20 |
| vitl16-224-b8-ne-s256-r8-ct9.0.mlpackage | done | 4.4 | 7.9 | 0.1 | 0.1 | 606.7 | 2623 | 2.21 |
| vitl16-512-b8-ne-s256-r8-ct9.0.mlpackage | done | 26.0 | 7.6 | 0.1 | 0.2 | 606.9 | 2623 | 3.82 |
| vith16plus-224-b1-ne-s256-r8-ct9.0.mlpackage | done | 2.4 | 17.5 | 0.6 | 0.2 | 1,681.7 | 3583 | 4.93 |
| vith16plus-448-b1-ne-s256-r8-ct9.0.mlpackage | done | 6.6 | 16.7 | 0.5 | 0.2 | 1,681.9 | 3583 | 4.96 |
| vith16plus-512-b1-ne-s256-r8-ct9.0.mlpackage | done | 8.8 | 17.0 | 0.4 | 0.2 | 1,681.9 | 3583 | 5.00 |
| vith16plus-224-b8-ne-s256-r8-ct9.0.mlpackage | done | 10.7 | 16.6 | 0.3 | 0.2 | 1,681.8 | 3583 | 4.99 |
| vith16plus-512-b8-ne-s256-r8-ct9.0.mlpackage | done | 60.6 | 16.9 | 0.4 | 0.2 | 1,682.0 | 3583 | 5.79 |
| vits16-224-b1-plain-ct9.0.mlpackage | done | 0.4 | 2.0 | 0.0 | 0.1 | 43.4 | 1229 | 0.49 |
| ViT-7B/16 | deferred: ViT-7B/16 is deferred (D6 of explore.html) | – | – | – | – | – | – | – |

Table 1. The conversion of each package (job J1).

## 2 Parity

| Model size | Input resolution | Backend | Status | Valid | Parity | Largest token value | First load s | Cached load s | Peak footprint GB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ViT-S/16 | 224 | coreml-ne | done | True | 0.9999 / 0.9999 / 0.9996 | 9.1 | 0.07 | 0.022 | 0.37 |
| ViT-S/16 | 224 | coreml-gpu | done | True | 1.0000 / 1.0000 / 1.0000 | 9.2 | 0.16 | 0.024 | 0.57 |
| ViT-S/16 | 224 | coreml-all | done | True | 1.0000 / 1.0000 / 1.0000 | 9.2 | 0.40 | 0.028 | 0.57 |
| ViT-S/16 | 448 | coreml-ne | done | False | -0.0147 / -0.0686 / -0.2907 | 19.6 | 2.58 | 0.024 | 0.41 |
| ViT-S/16 | 448 | coreml-gpu | done | True | 1.0000 / 1.0000 / 1.0000 | 9.2 | 0.17 | 0.026 | 0.61 |
| ViT-S/16 | 448 | coreml-all | done | True | 1.0000 / 1.0000 / 1.0000 | 9.2 | 0.40 | 0.026 | 0.61 |
| ViT-S/16 | 512 | coreml-ne | done | False | 0.0669 / 0.3480 / -0.1178 | 9.2 | 2.88 | 0.025 | 0.42 |
| ViT-S/16 | 512 | coreml-gpu | done | True | 1.0000 / 1.0000 / 1.0000 | 9.2 | 0.18 | 0.025 | 0.64 |
| ViT-S/16 | 512 | coreml-all | done | True | 1.0000 / 1.0000 / 1.0000 | 9.2 | 0.40 | 0.025 | 0.64 |
| ViT-S+/16 | 224 | coreml-ne | done | True | 0.9997 / 0.9997 / 0.9988 | 10.9 | 1.68 | 0.023 | 0.38 |
| ViT-S+/16 | 224 | coreml-gpu | done | True | 1.0000 / 1.0000 / 0.9999 | 10.9 | 0.19 | 0.025 | 0.59 |
| ViT-S+/16 | 224 | coreml-all | done | True | 1.0000 / 1.0000 / 0.9999 | 10.9 | 0.42 | 0.027 | 0.59 |
| ViT-S+/16 | 448 | coreml-ne | done | False | -0.0123 / -0.0122 / -0.1145 | 24.9 | 2.57 | 0.024 | 0.41 |
| ViT-S+/16 | 448 | coreml-gpu | done | True | 1.0000 / 1.0000 / 1.0000 | 11.0 | 0.18 | 0.027 | 0.63 |
| ViT-S+/16 | 448 | coreml-all | done | True | 1.0000 / 1.0000 / 1.0000 | 11.0 | 0.42 | 0.029 | 0.63 |
| ViT-S+/16 | 512 | coreml-ne | done | False | -0.0035 / 0.0507 / -0.0238 | 10.8 | 2.84 | 0.023 | 0.42 |
| ViT-S+/16 | 512 | coreml-gpu | done | True | 1.0000 / 1.0000 / 1.0000 | 11.0 | 0.19 | 0.029 | 0.65 |
| ViT-S+/16 | 512 | coreml-all | done | True | 1.0000 / 1.0000 / 1.0000 | 11.0 | 0.43 | 0.027 | 0.65 |
| ViT-B/16 | 224 | coreml-ne | done | False | 0.0503 / -0.0260 / -0.0992 | 40.0 | 2.60 | 0.024 | 0.41 |
| ViT-B/16 | 224 | coreml-gpu | done | True | 1.0000 / 1.0000 / 1.0000 | 17.9 | 0.22 | 0.035 | 0.73 |
| ViT-B/16 | 224 | coreml-all | done | True | 1.0000 / 1.0000 / 1.0000 | 17.9 | 0.47 | 0.037 | 0.73 |
| ViT-B/16 | 448 | coreml-ne | done | False | -0.0517 / -0.0818 / -0.1505 | 27.2 | 3.35 | 0.024 | 0.45 |
| ViT-B/16 | 448 | coreml-gpu | done | True | 1.0000 / 1.0000 / 1.0000 | 14.8 | 0.24 | 0.036 | 0.79 |
| ViT-B/16 | 448 | coreml-all | done | True | 1.0000 / 1.0000 / 1.0000 | 14.8 | 0.52 | 0.036 | 0.80 |
| ViT-B/16 | 512 | coreml-ne | done | False | -0.0350 / -0.0034 / -0.0835 | 18.1 | 3.74 | 0.027 | 0.47 |
| ViT-B/16 | 512 | coreml-gpu | done | True | 1.0000 / 1.0000 / 1.0000 | 14.4 | 0.22 | 0.036 | 0.84 |
| ViT-B/16 | 512 | coreml-all | done | True | 1.0000 / 1.0000 / 1.0000 | 14.4 | 0.46 | 0.037 | 0.84 |
| ViT-L/16 | 224 | coreml-ne | done | False | -0.0253 / 0.0122 / -0.0662 | 16.7 | 7.09 | 0.041 | 0.40 |
| ViT-L/16 | 224 | coreml-gpu | done | False | 0.9740 / 0.9711 / 0.8979 | 16.9 | 0.54 | 0.093 | 1.16 |
| ViT-L/16 | 224 | coreml-all | done | False | 0.9740 / 0.9711 / 0.8979 | 16.9 | 1.13 | 0.106 | 1.16 |
| ViT-L/16 | 448 | coreml-ne | done | False | -0.0067 / 0.0979 / -0.0313 | 15.9 | 9.26 | 0.047 | 0.47 |
| ViT-L/16 | 448 | coreml-gpu | done | False | 0.9140 / 0.9525 / 0.8379 | 9.5 | 0.52 | 0.094 | 1.26 |
| ViT-L/16 | 448 | coreml-all | done | False | 0.9140 / 0.9525 / 0.8379 | 9.5 | 1.17 | 0.099 | 1.26 |
| ViT-L/16 | 512 | coreml-ne | done | False | -0.0381 / 0.0719 / -0.0338 | 11.4 | 9.95 | 0.049 | 0.50 |
| ViT-L/16 | 512 | coreml-gpu | done | False | 0.8879 / 0.9581 / 0.8555 | 9.5 | 0.63 | 0.092 | 1.32 |
| ViT-L/16 | 512 | coreml-all | done | False | 0.8879 / 0.9581 / 0.8555 | 9.5 | 1.11 | 0.105 | 1.32 |
| ViT-H+/16 | 224 | coreml-ne | done | False | -0.0068 / 0.0208 / -0.0548 | 9.2 | 16.42 | 0.062 | 0.44 |
| ViT-H+/16 | 224 | coreml-gpu | done | False | 0.9978 / 0.9835 / 0.8291 | 12.3 | 1.34 | 0.230 | 2.27 |
| ViT-H+/16 | 224 | coreml-all | done | False | 0.9978 / 0.9835 / 0.8291 | 12.3 | 2.33 | 0.258 | 2.27 |
| ViT-H+/16 | 448 | coreml-ne | done | False | -0.0085 / 0.0131 / -0.0616 | 8.8 | 18.06 | 0.071 | 0.52 |
| ViT-H+/16 | 448 | coreml-gpu | done | False | 0.9982 / 0.9975 / 0.9833 | 9.7 | 1.45 | 0.249 | 2.39 |
| ViT-H+/16 | 448 | coreml-all | done | False | 0.9982 / 0.9975 / 0.9833 | 9.7 | 2.44 | 0.238 | 2.39 |
| ViT-H+/16 | 512 | coreml-ne | done | False | -0.0300 / 0.0006 / -0.1082 | 26.9 | 18.74 | 0.088 | 0.55 |
| ViT-H+/16 | 512 | coreml-gpu | done | False | 0.9982 / 0.9980 / 0.9851 | 9.9 | 1.75 | 0.236 | 2.47 |
| ViT-H+/16 | 512 | coreml-all | done | False | 0.9982 / 0.9980 / 0.9851 | 9.9 | 2.68 | 0.259 | 2.47 |
| ViT-S/16 (plain graph) | 224 | coreml-ne | done | False | 0.0345 / 0.1373 / 0.0654 | 4.0 | 0.07 | 0.021 | 0.37 |
| ViT-S/16 (plain graph) | 224 | coreml-gpu | done | True | 1.0000 / 1.0000 / 1.0000 | 9.2 | 0.10 | 0.025 | 0.56 |
| ViT-S/16 (plain graph) | 224 | coreml-all | done | True | 1.0000 / 1.0000 / 1.0000 | 9.2 | 0.43 | 0.024 | 0.56 |

Table 2. The parity of each package on each compute unit at batch size 1 (job J2). Thresholds of ADR 0003 for fp16: 0.999 / 0.999 / 0.99.

## 3 Compute plan

| Package | Ops | Ops per device | Neural Engine cost share | Ops off the Neural Engine |
| --- | --- | --- | --- | --- |
| vits16-224-b1-ne-s256-r1-ct9.0.mlpackage | 571 | CPU 46, NeuralEngine 525 | 0.754 | ios18.mul 9, ios18.reshape 5, ios18.transpose 5, ios18.concat 5, ios18.linear 4, ios18.slice_by_index 4, ios18.add 3, split 2, ios18.matmul 2, ios18.cast 1, ios18.conv 1, ios18.layer_norm 1, ios16.reduce_max 1, ios18.sub 1, ios18.clip 1, ios18.softmax 1 |
| vits16-448-b1-ne-s256-r1-ct9.0.mlpackage | 571 | CPU 4, NeuralEngine 567 | 0.991 | ios18.reshape 3, ios18.add 1 |
| vits16-512-b1-ne-s256-r1-ct9.0.mlpackage | 571 | CPU 4, NeuralEngine 567 | 0.992 | ios18.reshape 3, ios18.add 1 |
| vits16-224-b8-ne-s256-r1-ct9.0.mlpackage | 571 | CPU 4, NeuralEngine 567 | 0.987 | ios18.reshape 3, ios18.add 1 |
| vits16-512-b8-ne-s256-r1-ct9.0.mlpackage | 571 | CPU 4, NeuralEngine 567 | 0.992 | ios18.reshape 3, ios18.add 1 |
| vits16plus-224-b1-ne-s256-r1-ct9.0.mlpackage | 595 | CPU 46, NeuralEngine 549 | 0.764 | ios18.mul 9, ios18.reshape 5, ios18.transpose 5, ios18.concat 5, ios18.linear 4, ios18.slice_by_index 4, ios18.add 3, split 2, ios18.matmul 2, ios18.cast 1, ios18.conv 1, ios18.layer_norm 1, ios16.reduce_max 1, ios18.sub 1, ios18.clip 1, ios18.softmax 1 |
| vits16plus-448-b1-ne-s256-r1-ct9.0.mlpackage | 595 | CPU 4, NeuralEngine 591 | 0.992 | ios18.reshape 3, ios18.add 1 |
| vits16plus-512-b1-ne-s256-r1-ct9.0.mlpackage | 595 | CPU 4, NeuralEngine 591 | 0.993 | ios18.reshape 3, ios18.add 1 |
| vits16plus-224-b8-ne-s256-r1-ct9.0.mlpackage | 595 | CPU 4, NeuralEngine 591 | 0.988 | ios18.reshape 3, ios18.add 1 |
| vits16plus-512-b8-ne-s256-r1-ct9.0.mlpackage | 595 | CPU 4, NeuralEngine 591 | 0.993 | ios18.reshape 3, ios18.add 1 |
| vitb16-224-b1-ne-s256-r1-ct9.0.mlpackage | 571 | CPU 4, NeuralEngine 567 | 0.991 | ios18.reshape 3, ios18.add 1 |
| vitb16-448-b1-ne-s256-r1-ct9.0.mlpackage | 571 | CPU 4, NeuralEngine 567 | 0.993 | ios18.reshape 3, ios18.add 1 |
| vitb16-512-b1-ne-s256-r1-ct9.0.mlpackage | 571 | CPU 4, NeuralEngine 567 | 0.994 | ios18.reshape 3, ios18.add 1 |
| vitb16-224-b8-ne-s256-r1-ct9.0.mlpackage | 571 | CPU 4, NeuralEngine 567 | 0.991 | ios18.reshape 3, ios18.add 1 |
| vitb16-512-b8-ne-s256-r1-ct9.0.mlpackage | 571 | CPU 4, NeuralEngine 567 | 0.994 | ios18.reshape 3, ios18.add 1 |
| vitl16-224-b1-ne-s256-r8-ct9.0.mlpackage | 1136 | CPU 4, NeuralEngine 1132 | 0.996 | ios18.reshape 3, ios18.add 1 |
| vitl16-448-b1-ne-s256-r8-ct9.0.mlpackage | 1136 | CPU 4, NeuralEngine 1132 | 0.997 | ios18.reshape 3, ios18.add 1 |
| vitl16-512-b1-ne-s256-r8-ct9.0.mlpackage | 1136 | CPU 4, NeuralEngine 1132 | 0.997 | ios18.reshape 3, ios18.add 1 |
| vitl16-224-b8-ne-s256-r8-ct9.0.mlpackage | 1136 | CPU 4, NeuralEngine 1132 | 0.996 | ios18.reshape 3, ios18.add 1 |
| vitl16-512-b8-ne-s256-r8-ct9.0.mlpackage | 1136 | CPU 4, NeuralEngine 1132 | 0.997 | ios18.reshape 3, ios18.add 1 |
| vith16plus-224-b1-ne-s256-r8-ct9.0.mlpackage | 1576 | CPU 4, NeuralEngine 1572 | 0.998 | ios18.reshape 3, ios18.add 1 |
| vith16plus-448-b1-ne-s256-r8-ct9.0.mlpackage | 1576 | CPU 4, NeuralEngine 1572 | 0.998 | ios18.reshape 3, ios18.add 1 |
| vith16plus-512-b1-ne-s256-r8-ct9.0.mlpackage | 1576 | CPU 4, NeuralEngine 1572 | 0.998 | ios18.reshape 3, ios18.add 1 |
| vith16plus-224-b8-ne-s256-r8-ct9.0.mlpackage | 1576 | CPU 4, NeuralEngine 1572 | 0.998 | ios18.reshape 3, ios18.add 1 |

Table 3. The compute plan of each package on CPU_AND_NE (job J2).

## 4 Matrix

| Model size | Input resolution | Batch size | Backend | Status | Valid | Parity (image 0) | img/s | Ratio to mlx | mlx dtype mode | mlx img/s | Reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

Table 4. The matrix: patch embeddings, 1 round (job J3). The ratio is Core ML / mlx of the same paired group.

## 5 Paired runs at P1 and P2

| Group | Backend | Run medians | Pooled median | Spread between runs | Verdict D21 | Core ML img/s (median) | mlx img/s (median) | Rounds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |

Table 5. The paired ratio Core ML / mlx at P1 (frame) and P2 (patch), median over the rounds of each run (run 1: build, run 2: verify). Report only: Step 6c has no speed target.

## 6 Two-stream run

| Group | Run | Neural Engine batch size | ne alone img/s | mlx alone img/s | both: ne img/s | both: mlx img/s | both: total img/s | Stream ratio | Rounds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

Table 6. The two-stream run: Core ML on CPU_AND_NE and compiled mlx on the GPU, alone and at the same time, 10 s per case. Stream ratio = (ne + mlx, both) / (mlx alone), median over the rounds.
