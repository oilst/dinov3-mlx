# Core ML on the Neural Engine: rewrite ne3

Made by `scripts/bench_coreml.py table` from `results/coreml-6d.jsonl` on 2026-11-09. Step 6d (specs/06d-neural-engine-fault/plan.html). Core ML packages run with fp16 compute precision; the mlx side runs the compiled forward pass in the fastest valid fp16 dtype mode of Step 6. Parity: frame / patch mean / patch 1st percentile, the worst of the 3 test images (parity) or image 0 (speed).

## 1 Conversion

| Package | Status | Trace s | Convert s | Save s | Compile s | Size MB | Ops | Constant concats | Peak footprint GB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| vits16-224-b1-ne3-s256-r1-ct9.0.mlpackage | done | 0.4 | 2.2 | 0.0 | 0.1 | 43.5 | 1324 | 0 | 0.53 |
| vits16-448-b1-ne3-s256-r1-ct9.0.mlpackage | done | 0.6 | 2.2 | 0.0 | 0.1 | 43.6 | 1324 | 0 | 0.57 |
| vits16-512-b1-ne3-s256-r1-ct9.0.mlpackage | done | 0.7 | 2.2 | 0.0 | 0.1 | 43.7 | 1324 | 0 | 0.62 |
| vits16-224-b8-ne3-s256-r1-ct9.0.mlpackage | done | 0.7 | 2.2 | 0.0 | 0.1 | 43.5 | 1324 | 0 | 0.60 |
| vits16-512-b8-ne3-s256-r1-ct9.0.mlpackage | done | 3.8 | 2.4 | 0.0 | 0.1 | 43.7 | 1324 | 0 | 1.55 |
| vits16plus-224-b1-ne3-s256-r1-ct9.0.mlpackage | done | 0.5 | 2.3 | 0.0 | 0.1 | 57.7 | 1360 | 0 | 0.55 |
| vits16plus-448-b1-ne3-s256-r1-ct9.0.mlpackage | done | 0.6 | 2.3 | 0.0 | 0.1 | 57.8 | 1360 | 0 | 0.60 |
| vits16plus-512-b1-ne3-s256-r1-ct9.0.mlpackage | done | 0.8 | 2.3 | 0.0 | 0.1 | 57.9 | 1360 | 0 | 0.67 |
| vits16plus-224-b8-ne3-s256-r1-ct9.0.mlpackage | done | 0.8 | 2.2 | 0.0 | 0.1 | 57.7 | 1360 | 0 | 0.63 |
| vits16plus-512-b8-ne3-s256-r1-ct9.0.mlpackage | done | 4.4 | 2.3 | 0.0 | 0.1 | 57.9 | 1360 | 0 | 1.60 |
| vitb16-224-b1-ne3-s256-r1-ct9.0.mlpackage | done | 0.8 | 2.8 | 0.0 | 0.1 | 171.6 | 1324 | 0 | 0.88 |
| vitb16-448-b1-ne3-s256-r1-ct9.0.mlpackage | done | 1.1 | 2.8 | 0.0 | 0.1 | 171.7 | 1324 | 0 | 0.96 |
| vitb16-512-b1-ne3-s256-r1-ct9.0.mlpackage | done | 1.4 | 2.8 | 0.0 | 0.1 | 171.8 | 1324 | 0 | 1.06 |
| vitb16-224-b8-ne3-s256-r1-ct9.0.mlpackage | done | 1.5 | 2.8 | 0.0 | 0.1 | 171.6 | 1324 | 0 | 0.94 |
| vitb16-512-b8-ne3-s256-r1-ct9.0.mlpackage | done | 8.7 | 2.8 | 0.0 | 0.1 | 171.8 | 1324 | 0 | 2.80 |
| vitl16-224-b1-ne3-s256-r8-ct9.0.mlpackage | done | 1.9 | 7.2 | 0.1 | 0.1 | 606.7 | 2622 | 0 | 2.20 |
| vitl16-448-b1-ne3-s256-r8-ct9.0.mlpackage | done | 3.0 | 6.9 | 0.1 | 0.2 | 606.8 | 2622 | 0 | 2.21 |
| vitl16-512-b1-ne3-s256-r8-ct9.0.mlpackage | done | 4.2 | 7.0 | 0.1 | 0.1 | 606.9 | 2622 | 0 | 2.20 |
| vitl16-224-b8-ne3-s256-r8-ct9.0.mlpackage | done | 4.4 | 7.1 | 0.1 | 0.1 | 606.7 | 2622 | 0 | 2.21 |
| vitl16-512-b8-ne3-s256-r8-ct9.0.mlpackage | done | 25.8 | 7.3 | 0.1 | 0.1 | 606.9 | 2622 | 0 | 4.31 |
| vith16plus-224-b1-ne3-s256-r8-ct9.0.mlpackage | done | 6.5 | 17.5 | 0.3 | 0.2 | 1,681.7 | 3582 | 0 | 4.94 |
| vith16plus-448-b1-ne3-s256-r8-ct9.0.mlpackage | done | 6.5 | 16.0 | 0.5 | 0.2 | 1,681.9 | 3582 | 0 | 4.96 |
| vith16plus-512-b1-ne3-s256-r8-ct9.0.mlpackage | done | 8.9 | 14.3 | 0.4 | 0.2 | 1,681.9 | 3582 | 0 | 5.00 |
| vith16plus-224-b8-ne3-s256-r8-ct9.0.mlpackage | done | 10.5 | 14.1 | 0.4 | 0.2 | 1,681.8 | 3582 | 0 | 4.99 |
| vith16plus-512-b8-ne3-s256-r8-ct9.0.mlpackage | done | 60.8 | 14.7 | 0.4 | 0.2 | 1,682.0 | 3582 | 0 | 5.64 |

Table 1. The conversion of each package (job J1). Constant concats: the concats with two or more constant inputs (the concat fault); convert_module refuses more than 0.

## 2 Parity

| Model size | Input resolution | Backend | Status | Valid | Parity | Largest token value | First load s | Cached load s | Peak footprint GB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ViT-S/16 | 224 | coreml-ne | done | True | 0.9999 / 1.0000 / 0.9999 | 9.1 | 1.73 | 0.038 | 0.38 |
| ViT-S/16 | 224 | coreml-gpu | done | True | 1.0000 / 1.0000 / 1.0000 | 9.2 | 0.16 | 0.024 | 0.57 |
| ViT-S/16 | 224 | coreml-all | done | True | 1.0000 / 1.0000 / 1.0000 | 9.2 | 0.40 | 0.031 | 0.58 |
| ViT-S/16 | 448 | coreml-ne | done | True | 0.9999 / 1.0000 / 0.9999 | 9.2 | 2.61 | 0.043 | 0.39 |
| ViT-S/16 | 448 | coreml-gpu | done | True | 1.0000 / 1.0000 / 1.0000 | 9.2 | 0.17 | 0.024 | 0.61 |
| ViT-S/16 | 448 | coreml-all | done | True | 1.0000 / 1.0000 / 1.0000 | 9.2 | 0.39 | 0.024 | 0.61 |
| ViT-S/16 | 512 | coreml-ne | done | True | 0.9999 / 1.0000 / 0.9999 | 9.2 | 2.90 | 0.041 | 0.40 |
| ViT-S/16 | 512 | coreml-gpu | done | True | 1.0000 / 1.0000 / 1.0000 | 9.2 | 0.16 | 0.024 | 0.64 |
| ViT-S/16 | 512 | coreml-all | done | True | 1.0000 / 1.0000 / 1.0000 | 9.2 | 0.40 | 0.023 | 0.64 |
| ViT-S+/16 | 224 | coreml-ne | done | True | 0.9998 / 0.9999 / 0.9995 | 10.9 | 1.78 | 0.040 | 0.38 |
| ViT-S+/16 | 224 | coreml-gpu | done | True | 1.0000 / 1.0000 / 0.9999 | 10.9 | 0.17 | 0.026 | 0.59 |
| ViT-S+/16 | 224 | coreml-all | done | True | 1.0000 / 1.0000 / 0.9999 | 10.9 | 0.42 | 0.026 | 0.59 |
| ViT-S+/16 | 448 | coreml-ne | done | True | 0.9999 / 0.9999 / 0.9998 | 11.0 | 2.52 | 0.043 | 0.40 |
| ViT-S+/16 | 448 | coreml-gpu | done | True | 1.0000 / 1.0000 / 1.0000 | 11.0 | 0.17 | 0.025 | 0.63 |
| ViT-S+/16 | 448 | coreml-all | done | True | 1.0000 / 1.0000 / 1.0000 | 11.0 | 0.42 | 0.026 | 0.63 |
| ViT-S+/16 | 512 | coreml-ne | done | True | 0.9999 / 0.9999 / 0.9999 | 11.0 | 2.82 | 0.040 | 0.40 |
| ViT-S+/16 | 512 | coreml-gpu | done | True | 1.0000 / 1.0000 / 1.0000 | 11.0 | 0.17 | 0.026 | 0.65 |
| ViT-S+/16 | 512 | coreml-all | done | True | 1.0000 / 1.0000 / 1.0000 | 11.0 | 0.41 | 0.026 | 0.65 |
| ViT-B/16 | 224 | coreml-ne | done | True | 0.9999 / 0.9999 / 0.9998 | 17.9 | 2.50 | 0.042 | 0.40 |
| ViT-B/16 | 224 | coreml-gpu | done | True | 1.0000 / 1.0000 / 1.0000 | 17.9 | 0.21 | 0.033 | 0.73 |
| ViT-B/16 | 224 | coreml-all | done | True | 1.0000 / 1.0000 / 1.0000 | 17.9 | 0.44 | 0.035 | 0.73 |
| ViT-B/16 | 448 | coreml-ne | done | True | 1.0000 / 1.0000 / 0.9999 | 14.8 | 3.27 | 0.043 | 0.42 |
| ViT-B/16 | 448 | coreml-gpu | done | True | 1.0000 / 1.0000 / 1.0000 | 14.8 | 0.20 | 0.033 | 0.80 |
| ViT-B/16 | 448 | coreml-all | done | True | 1.0000 / 1.0000 / 1.0000 | 14.8 | 0.49 | 0.033 | 0.79 |
| ViT-B/16 | 512 | coreml-ne | done | True | 1.0000 / 1.0000 / 0.9999 | 14.4 | 3.62 | 0.045 | 0.43 |
| ViT-B/16 | 512 | coreml-gpu | done | True | 1.0000 / 1.0000 / 1.0000 | 14.4 | 0.27 | 0.033 | 0.83 |
| ViT-B/16 | 512 | coreml-all | done | True | 1.0000 / 1.0000 / 1.0000 | 14.4 | 0.50 | 0.033 | 0.84 |
| ViT-L/16 | 224 | coreml-ne | done | True | 0.9999 / 0.9999 / 0.9993 | 15.7 | 6.33 | 0.058 | 0.39 |
| ViT-L/16 | 224 | coreml-gpu | done | True | 1.0000 / 1.0000 / 1.0000 | 15.7 | 0.56 | 0.090 | 1.16 |
| ViT-L/16 | 224 | coreml-all | done | True | 1.0000 / 1.0000 / 1.0000 | 15.7 | 1.06 | 0.088 | 1.16 |
| ViT-L/16 | 448 | coreml-ne | done | True | 0.9999 / 0.9999 / 0.9998 | 15.7 | 7.83 | 0.060 | 0.43 |
| ViT-L/16 | 448 | coreml-gpu | done | True | 1.0000 / 1.0000 / 1.0000 | 15.7 | 0.51 | 0.091 | 1.26 |
| ViT-L/16 | 448 | coreml-all | done | True | 1.0000 / 1.0000 / 1.0000 | 15.7 | 1.29 | 0.088 | 1.26 |
| ViT-L/16 | 512 | coreml-ne | done | True | 0.9998 / 0.9999 / 0.9998 | 15.7 | 8.67 | 0.062 | 0.44 |
| ViT-L/16 | 512 | coreml-gpu | done | True | 1.0000 / 1.0000 / 1.0000 | 15.7 | 0.52 | 0.091 | 1.32 |
| ViT-L/16 | 512 | coreml-all | done | True | 1.0000 / 1.0000 / 1.0000 | 15.7 | 1.32 | 0.092 | 1.32 |
| ViT-H+/16 | 224 | coreml-ne | done | True | 0.9999 / 0.9996 / 0.9938 | 12.3 | 16.88 | 0.080 | 0.42 |
| ViT-H+/16 | 224 | coreml-gpu | done | True | 1.0000 / 1.0000 / 0.9995 | 12.3 | 1.22 | 0.218 | 2.27 |
| ViT-H+/16 | 224 | coreml-all | done | True | 1.0000 / 1.0000 / 0.9995 | 12.3 | 2.20 | 0.241 | 2.28 |
| ViT-H+/16 | 448 | coreml-ne | done | True | 1.0000 / 1.0000 / 0.9997 | 9.7 | 16.36 | 0.085 | 0.46 |
| ViT-H+/16 | 448 | coreml-gpu | done | True | 1.0000 / 1.0000 / 1.0000 | 9.8 | 1.09 | 0.249 | 2.39 |
| ViT-H+/16 | 448 | coreml-all | done | True | 1.0000 / 1.0000 / 1.0000 | 9.8 | 2.38 | 0.232 | 2.39 |
| ViT-H+/16 | 512 | coreml-ne | done | True | 1.0000 / 1.0000 / 0.9998 | 9.8 | 15.56 | 0.078 | 0.48 |
| ViT-H+/16 | 512 | coreml-gpu | done | True | 1.0000 / 1.0000 / 1.0000 | 9.8 | 1.44 | 0.233 | 2.47 |
| ViT-H+/16 | 512 | coreml-all | done | True | 1.0000 / 1.0000 / 1.0000 | 9.8 | 1.97 | 0.220 | 2.47 |

Table 2. The parity of each package on each compute unit at batch size 1 (job J2). Thresholds of ADR 0003 for fp16: 0.999 / 0.999 / 0.99.

## 3 Compute plan

| Package | Ops | Ops per device | Concat devices | Neural Engine cost share | Ops off the Neural Engine |
| --- | --- | --- | --- | --- | --- |
| vits16-224-b1-ne3-s256-r1-ct9.0.mlpackage | 571 | NeuralEngine 571 | NeuralEngine 49 | 1.000 | – |
| vits16-448-b1-ne3-s256-r1-ct9.0.mlpackage | 571 | NeuralEngine 571 | NeuralEngine 49 | 1.000 | – |
| vits16-512-b1-ne3-s256-r1-ct9.0.mlpackage | 571 | NeuralEngine 571 | NeuralEngine 49 | 1.000 | – |
| vits16-224-b8-ne3-s256-r1-ct9.0.mlpackage | 571 | NeuralEngine 571 | NeuralEngine 49 | 1.000 | – |
| vits16-512-b8-ne3-s256-r1-ct9.0.mlpackage | 571 | NeuralEngine 571 | NeuralEngine 49 | 1.000 | – |
| vits16plus-224-b1-ne3-s256-r1-ct9.0.mlpackage | 595 | NeuralEngine 595 | NeuralEngine 49 | 1.000 | – |
| vits16plus-448-b1-ne3-s256-r1-ct9.0.mlpackage | 595 | NeuralEngine 595 | NeuralEngine 49 | 1.000 | – |
| vits16plus-512-b1-ne3-s256-r1-ct9.0.mlpackage | 595 | NeuralEngine 595 | NeuralEngine 49 | 1.000 | – |
| vits16plus-224-b8-ne3-s256-r1-ct9.0.mlpackage | 595 | NeuralEngine 595 | NeuralEngine 49 | 1.000 | – |
| vits16plus-512-b8-ne3-s256-r1-ct9.0.mlpackage | 595 | NeuralEngine 595 | NeuralEngine 49 | 1.000 | – |
| vitb16-224-b1-ne3-s256-r1-ct9.0.mlpackage | 571 | NeuralEngine 571 | NeuralEngine 49 | 1.000 | – |
| vitb16-448-b1-ne3-s256-r1-ct9.0.mlpackage | 571 | NeuralEngine 571 | NeuralEngine 49 | 1.000 | – |
| vitb16-512-b1-ne3-s256-r1-ct9.0.mlpackage | 571 | NeuralEngine 571 | NeuralEngine 49 | 1.000 | – |
| vitb16-224-b8-ne3-s256-r1-ct9.0.mlpackage | 571 | NeuralEngine 571 | NeuralEngine 49 | 1.000 | – |
| vitb16-512-b8-ne3-s256-r1-ct9.0.mlpackage | 571 | NeuralEngine 571 | NeuralEngine 49 | 1.000 | – |
| vitl16-224-b1-ne3-s256-r8-ct9.0.mlpackage | 1136 | NeuralEngine 1136 | NeuralEngine 97 | 1.000 | – |
| vitl16-448-b1-ne3-s256-r8-ct9.0.mlpackage | 1136 | NeuralEngine 1136 | NeuralEngine 97 | 1.000 | – |
| vitl16-512-b1-ne3-s256-r8-ct9.0.mlpackage | 1136 | NeuralEngine 1136 | NeuralEngine 97 | 1.000 | – |
| vitl16-224-b8-ne3-s256-r8-ct9.0.mlpackage | 1136 | NeuralEngine 1136 | NeuralEngine 97 | 1.000 | – |
| vitl16-512-b8-ne3-s256-r8-ct9.0.mlpackage | 1136 | NeuralEngine 1136 | NeuralEngine 97 | 1.000 | – |
| vith16plus-224-b1-ne3-s256-r8-ct9.0.mlpackage | 1576 | NeuralEngine 1576 | NeuralEngine 129 | 1.000 | – |
| vith16plus-448-b1-ne3-s256-r8-ct9.0.mlpackage | 1576 | NeuralEngine 1576 | NeuralEngine 129 | 1.000 | – |
| vith16plus-512-b1-ne3-s256-r8-ct9.0.mlpackage | 1576 | NeuralEngine 1576 | NeuralEngine 129 | 1.000 | – |
| vith16plus-224-b8-ne3-s256-r8-ct9.0.mlpackage | 1576 | NeuralEngine 1576 | NeuralEngine 129 | 1.000 | – |
| vith16plus-512-b8-ne3-s256-r8-ct9.0.mlpackage | 1576 | NeuralEngine 1576 | NeuralEngine 129 | 1.000 | – |

Table 3. The compute plan of each package on CPU_AND_NE (job J2).

## 4 Matrix

| Model size | Input resolution | Batch size | Backend | Status | Valid | Parity (image 0) | img/s | Ratio to mlx | mlx dtype mode | mlx img/s | Reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ViT-S/16 | 224 | 1 | coreml-ne | done | True | 0.9999 / 1.0000 / 0.9999 | 621.8 | 1.112 | fp16-cast | 559.1 |  |
| ViT-S/16 | 224 | 1 | coreml-gpu | done | True | 1.0000 / 1.0000 / 1.0000 | 445.8 | 0.797 | fp16-cast | 559.1 |  |
| ViT-S/16 | 224 | 8 | coreml-gpu | done | True | 1.0000 / 1.0000 / 1.0000 | 1,732.9 | 0.986 | fp16-cast | 1,757.2 |  |
| ViT-S/16 | 224 | 8 | coreml-ne (b8 package) | done | True | 0.9999 / 1.0000 / 0.9999 | 493.6 | 0.281 | fp16-cast | 1,757.2 |  |
| ViT-S/16 | 224 | 8 | coreml-ne (batch API) | done | True | 0.9999 / 1.0000 / 0.9999 | 666.0 | 0.379 | fp16-cast | 1,757.2 |  |
| ViT-S/16 | 512 | 1 | coreml-ne | done | True | 0.9999 / 1.0000 / 0.9999 | 35.7 | 0.121 | fp16-cast | 295.1 |  |
| ViT-S/16 | 512 | 1 | coreml-gpu | done | True | 1.0000 / 1.0000 / 1.0000 | 225.3 | 0.763 | fp16-cast | 295.1 |  |
| ViT-S/16 | 512 | 8 | coreml-ne (b8 package) | done | True | 0.9999 / 1.0000 / 0.9999 | 35.9 | 0.157 | fp16-cast | 228.8 |  |
| ViT-S/16 | 512 | 8 | coreml-ne (batch API) | done | True | 0.9999 / 1.0000 / 0.9999 | 36.2 | 0.158 | fp16-cast | 228.8 |  |
| ViT-S/16 | 512 | 8 | coreml-gpu | done | True | 1.0000 / 1.0000 / 1.0000 | 147.3 | 0.644 | fp16-cast | 228.8 |  |
| ViT-S+/16 | 224 | 1 | coreml-gpu | done | True | 1.0000 / 1.0000 / 1.0000 | 384.7 | 0.846 | fp16-cast | 454.7 |  |
| ViT-S+/16 | 224 | 1 | coreml-ne | done | True | 0.9999 / 0.9999 / 0.9998 | 551.3 | 1.213 | fp16-cast | 454.7 |  |
| ViT-S+/16 | 224 | 8 | coreml-gpu | done | True | 1.0000 / 1.0000 / 1.0000 | 1,300.9 | 1.100 | fp16-cast | 1,182.4 |  |
| ViT-S+/16 | 224 | 8 | coreml-ne (b8 package) | done | True | 0.9999 / 0.9999 / 0.9998 | 431.4 | 0.365 | fp16-cast | 1,182.4 |  |
| ViT-S+/16 | 224 | 8 | coreml-ne (batch API) | done | True | 0.9999 / 0.9999 / 0.9998 | 585.4 | 0.495 | fp16-cast | 1,182.4 |  |
| ViT-S+/16 | 512 | 1 | coreml-ne | done | True | 0.9999 / 0.9999 / 0.9999 | 34.0 | 0.162 | fp16-cast | 210.5 |  |
| ViT-S+/16 | 512 | 1 | coreml-gpu | done | True | 1.0000 / 1.0000 / 1.0000 | 162.0 | 0.770 | fp16-cast | 210.5 |  |
| ViT-S+/16 | 512 | 8 | coreml-ne (b8 package) | done | True | 0.9999 / 0.9999 / 0.9999 | 34.3 | 0.095 | fp16-cast | 360.2 |  |
| ViT-S+/16 | 512 | 8 | coreml-ne (batch API) | done | True | 0.9999 / 0.9999 / 0.9999 | 34.5 | 0.096 | fp16-cast | 360.2 |  |
| ViT-S+/16 | 512 | 8 | coreml-gpu | done | True | 1.0000 / 1.0000 / 1.0000 | 273.0 | 0.758 | fp16-cast | 360.2 |  |
| ViT-B/16 | 224 | 1 | coreml-ne | done | True | 1.0000 / 1.0000 / 0.9999 | 240.7 | 0.602 | fp16-cast | 399.8 |  |
| ViT-B/16 | 224 | 1 | coreml-gpu | done | True | 1.0000 / 1.0000 / 1.0000 | 327.5 | 0.819 | fp16-cast | 399.8 |  |
| ViT-B/16 | 224 | 8 | coreml-gpu | done | True | 1.0000 / 1.0000 / 1.0000 | 739.1 | 0.899 | fp16-cast | 822.0 |  |
| ViT-B/16 | 224 | 8 | coreml-ne (b8 package) | done | True | 1.0000 / 1.0000 / 0.9999 | 181.4 | 0.221 | fp16-cast | 822.0 |  |
| ViT-B/16 | 224 | 8 | coreml-ne (batch API) | done | True | 1.0000 / 1.0000 / 0.9999 | 248.1 | 0.302 | fp16-cast | 822.0 |  |
| ViT-B/16 | 512 | 1 | coreml-gpu | done | True | 1.0000 / 1.0000 / 1.0000 | 101.6 | 0.787 | fp16-cast | 129.1 |  |
| ViT-B/16 | 512 | 1 | coreml-ne | done | True | 1.0000 / 1.0000 / 0.9999 | 16.5 | 0.128 | fp16-cast | 129.1 |  |
| ViT-B/16 | 512 | 8 | coreml-ne (b8 package) | done | True | 1.0000 / 1.0000 / 0.9999 | 16.3 | 0.100 | fp16-cast | 163.0 |  |
| ViT-B/16 | 512 | 8 | coreml-ne (batch API) | done | True | 1.0000 / 1.0000 / 0.9999 | 16.6 | 0.102 | fp16-cast | 163.0 |  |
| ViT-B/16 | 512 | 8 | coreml-gpu | done | True | 1.0000 / 1.0000 / 1.0000 | 110.6 | 0.679 | fp16-cast | 163.0 |  |
| ViT-L/16 | 224 | 1 | coreml-ne | done | True | 0.9999 / 0.9999 / 0.9993 | 65.2 | 0.416 | fp16-autocast | 156.9 |  |
| ViT-L/16 | 224 | 1 | coreml-gpu | done | True | 1.0000 / 1.0000 / 1.0000 | 146.7 | 0.935 | fp16-autocast | 156.9 |  |
| ViT-L/16 | 224 | 8 | coreml-gpu | done | True | 1.0000 / 1.0000 / 1.0000 | 258.8 | 1.043 | fp16-autocast | 248.1 |  |
| ViT-L/16 | 224 | 8 | coreml-ne (b8 package) | done | True | 0.9999 / 0.9999 / 0.9993 | 54.6 | 0.220 | fp16-autocast | 248.1 |  |
| ViT-L/16 | 224 | 8 | coreml-ne (batch API) | done | True | 0.9999 / 0.9999 / 0.9993 | 65.9 | 0.266 | fp16-autocast | 248.1 |  |
| ViT-L/16 | 512 | 1 | coreml-ne | done | True | 0.9999 / 0.9999 / 0.9999 | 5.6 | 0.130 | fp16-autocast | 43.2 |  |
| ViT-L/16 | 512 | 1 | coreml-gpu | done | True | 1.0000 / 1.0000 / 1.0000 | 37.0 | 0.855 | fp16-autocast | 43.2 |  |
| ViT-L/16 | 512 | 8 | coreml-ne (b8 package) | done | True | 0.9999 / 0.9999 / 0.9999 | 5.5 | 0.114 | fp16-autocast | 48.3 |  |
| ViT-L/16 | 512 | 8 | coreml-ne (batch API) | done | True | 0.9999 / 0.9999 / 0.9999 | 5.6 | 0.117 | fp16-autocast | 48.3 |  |
| ViT-L/16 | 512 | 8 | coreml-gpu | done | True | 1.0000 / 1.0000 / 1.0000 | 40.5 | 0.839 | fp16-autocast | 48.3 |  |
| ViT-H+/16 | 224 | 1 | coreml-gpu | done | True | 1.0000 / 1.0000 / 0.9995 | 77.7 | 1.114 | fp16-autocast | 69.7 |  |
| ViT-H+/16 | 224 | 1 | coreml-ne | done | True | 1.0000 / 0.9996 / 0.9938 | 24.6 | 0.353 | fp16-autocast | 69.7 |  |
| ViT-H+/16 | 224 | 8 | coreml-gpu | done | True | 1.0000 / 1.0000 / 0.9995 | 116.8 | 1.133 | fp16-autocast | 103.1 |  |
| ViT-H+/16 | 224 | 8 | coreml-ne (b8 package) | done | True | 1.0000 / 0.9996 / 0.9938 | 23.5 | 0.228 | fp16-autocast | 103.1 |  |
| ViT-H+/16 | 224 | 8 | coreml-ne (batch API) | done | True | 1.0000 / 0.9996 / 0.9938 | 24.7 | 0.240 | fp16-autocast | 103.1 |  |
| ViT-H+/16 | 512 | 1 | coreml-ne | done | True | 1.0000 / 1.0000 / 0.9999 | 2.8 | 0.154 | fp16-autocast | 18.3 |  |
| ViT-H+/16 | 512 | 1 | coreml-gpu | done | True | 1.0000 / 1.0000 / 1.0000 | 17.5 | 0.955 | fp16-autocast | 18.3 |  |
| ViT-H+/16 | 512 | 8 | coreml-ne (b8 package) | done | True | 1.0000 / 1.0000 / 0.9999 | 2.7 | 0.133 | fp16-cast | 20.5 |  |
| ViT-H+/16 | 512 | 8 | coreml-ne (batch API) | done | True | 1.0000 / 1.0000 / 0.9999 | 2.8 | 0.137 | fp16-cast | 20.5 |  |
| ViT-H+/16 | 512 | 8 | coreml-gpu | done | True | 1.0000 / 1.0000 / 1.0000 | 17.4 | 0.849 | fp16-cast | 20.5 |  |

Table 4. The matrix: patch embeddings, 1 round (job J3). The ratio is Core ML / mlx of the same paired group.

## 5 Paired runs at P1, P3 and P2

| Group | Backend | Run medians | Pooled median | Spread between runs | Verdict D21 | Core ML img/s (median) | mlx img/s (median) | Rounds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P1-b1 | coreml-gpu | 0.864 | 0.864 | – | not above 1 | 468.2 | 541.3 | 5 |
| P1-b1 | coreml-ne | 1.168 | 1.168 | – | above 1 | 633.2 | 541.3 | 5 |
| P1-b8 | coreml-gpu | 1.009, 1.024 | 1.011 | 0.015 | above 1, inside the noise | 1,872.8 | 1,827.7 | 10 |
| P1-b8 | coreml-ne (batch API) | 0.370, 0.373 | 0.372 | 0.009 | not above 1 | 678.0 | 1,827.7 | 10 |
| P2-b8 | coreml-gpu | 0.802, 0.800 | 0.801 | 0.002 | not above 1 | 16.6 | 19.5 | 10 |
| P2-b8 | coreml-ne (batch API) | 0.135, 0.150 | 0.145 | 0.108 | not above 1 | 2.8 | 19.5 | 10 |
| P3-b8 | coreml-gpu | 1.031, 1.048 | 1.044 | 0.017 | above 1 | 260.6 | 244.8 | 10 |
| P3-b8 | coreml-ne (batch API) | 0.262, 0.272 | 0.267 | 0.037 | not above 1 | 65.6 | 244.8 | 10 |

Table 5. The paired ratio Core ML / mlx at P1 (ViT-S/16 224 px, frame), P3 (ViT-L/16 224 px, patch) and P2 (ViT-H+/16 512 px, patch), median over the rounds of each run (run 1: build, run 2: verify). Above batch size 1 coreml-ne runs the faster valid form of the matrix. Report only: no speed target (AC10).


## 6 Two-stream run

| Group | Run | Neural Engine batch size | Neural Engine form | Neural Engine output valid | ne alone img/s | mlx alone img/s | both: ne img/s | both: mlx img/s | both: total img/s | Stream ratio | Rounds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P1-b8 | 1 | 8 | batch-api | yes | 677.3 | 1,543.4 | 670.0 | 1,374.1 | 2,044.0 | 1.324 | 3 |
| P1-b8 | 2 | 8 | batch-api | yes | 676.4 | 1,939.9 | 655.8 | 1,938.5 | 2,583.1 | 1.318 | 3 |
| P2-b8 | 1 | 8 | batch-api | yes | 2.8 | 19.7 | 2.8 | 18.7 | 21.5 | 1.081 | 3 |
| P3-b8 | 1 | 8 | batch-api | yes | 65.7 | 233.6 | 64.6 | 198.0 | 263.2 | 1.122 | 3 |
| P3-b8 | 2 | 8 | batch-api | yes | 65.4 | 246.9 | 55.3 | 238.5 | 292.2 | 1.184 | 3 |

Table 6. The two-stream run: Core ML on CPU_AND_NE and compiled mlx on the GPU, alone and at the same time, 10 s per case. Stream ratio = (ne + mlx, both) / (mlx alone), median over the rounds. The stream worker checks no output; the column «Neural Engine output valid» comes from the coreml-ne lines of the matrix and the paired runs of the same batch size and form (finding V2 of the Step 6c verify.html).
