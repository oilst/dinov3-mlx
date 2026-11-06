# Block profile: Hugging Face backbone on torch-mps

Written by `scripts/baseline_table.py` on 2026-11-06 from `results/block-profile-torch-mps.jsonl`. Spec: Section 3.11 of `specs/03-baseline/plan.html` (D6).

Each part of block 1 runs alone in a loop with one synchronize at the end. The time is the median of five loops. The FLOP share comes from `dinov3_apple.flops`. Limits of the checks: 15 %.

The forward pass runs before and after the parts; the check uses the mean of the two. A large change between the two shows a change of the speed of the Mac during the profile. At batch size 1 a block alone takes longer than a block inside the forward pass, so Forward / L gives the time of one block inside the forward pass (finding F4 of `specs/03-baseline/verify.html`). Lines from before F4 have only the time after the parts.

## Summary

| Model size | Dtype mode | Input resolution | Batch size | Block (ms) | Forward / L (ms) | Sum of parts / block | Check | Forward pass, before → after (ms) | (L × block + outer) / forward | Check |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ViT-S/16 | fp32 | 224 | 1 | 0.43 | 0.32 | 0.872 | pass | 3.87 | 1.358 | outside 15 % |
| ViT-S/16 | fp32 | 224 | 8 | 1.39 | 1.65 | 0.987 | pass | 19.76 | 0.868 | pass |
| ViT-S/16 | fp32 | 512 | 1 | 1.31 | 1.44 | 0.997 | pass | 17.30 | 0.926 | pass |
| ViT-S/16 | fp32 | 512 | 8 | 10.43 | 10.54 | 0.967 | pass | 126.52 | 1.006 | pass |
| ViT-S/16 | fp16-cast | 224 | 1 | 0.29 | 0.18 | 0.933 | pass | 2.22 | 1.643 | outside 15 % |
| ViT-S/16 | fp16-cast | 224 | 8 | 0.71 | 0.69 | 1.055 | pass | 8.32 | 1.076 | pass |
| ViT-S/16 | fp16-cast | 512 | 1 | 0.55 | 0.51 | 1.081 | pass | 6.07 | 1.132 | pass |
| ViT-S/16 | fp16-cast | 512 | 8 | 4.03 | 4.18 | 0.948 | pass | 50.14 | 0.997 | pass |
| ViT-H+/16 | fp32 | 224 | 1 | 2.15 | 2.13 | 1.066 | pass | 68.29 | 1.010 | pass |
| ViT-H+/16 | fp32 | 224 | 8 | 16.08 | 16.50 | 0.982 | pass | 528.10 | 0.978 | pass |
| ViT-H+/16 | fp32 | 512 | 1 | 11.50 | 11.42 | 0.950 | pass | 365.52 | 1.009 | pass |
| ViT-H+/16 | fp32 | 512 | 8 | 95.97 | 99.72 | 1.001 | pass | 3190.96 | 0.965 | pass |
| ViT-H+/16 | fp16-cast | 224 | 1 | 0.94 | 0.88 | 0.874 | pass | 28.19 | 1.073 | pass |
| ViT-H+/16 | fp16-cast | 224 | 8 | 6.14 | 6.15 | 0.884 | pass | 196.68 | 1.006 | pass |
| ViT-H+/16 | fp16-cast | 512 | 1 | 3.90 | 3.85 | 1.012 | pass | 123.06 | 1.020 | pass |
| ViT-H+/16 | fp16-cast | 512 | 8 | 33.31 | 36.61 | 1.021 | pass | 1171.64 | 0.915 | pass |

## ViT-S/16, fp32, 224 px, batch size 1

| Part | Time (µs) | Time share | FLOP share | TFLOP/s |
| --- | --- | --- | --- | --- |
| norm1 | 5.6 | 1.5 % | 0.0 % | – |
| qkv_proj | 77.7 | 20.9 % | 23.0 % | 2.29 |
| rope | 70.0 | 18.8 % | 0.0 % | – |
| sdpa | 28.5 | 7.7 % | 8.0 % | 2.18 |
| o_proj | 35.9 | 9.7 % | 7.7 % | 1.65 |
| layer_scale1 | 8.9 | 2.4 % | 0.0 % | – |
| norm2 | 5.8 | 1.6 % | 0.0 % | – |
| ff_in | 55.1 | 14.8 % | 30.7 % | 4.30 |
| activation | 4.3 | 1.2 % | 0.0 % | – |
| ff_out | 71.7 | 19.3 % | 30.7 % | 3.31 |
| layer_scale2 | 8.5 | 2.3 % | 0.0 % | – |

Parts outside the blocks: patch_embedding 104.5 µs, rope_tables 26.0 µs, final_norm 5.7 µs.

## ViT-S/16, fp32, 224 px, batch size 8

| Part | Time (µs) | Time share | FLOP share | TFLOP/s |
| --- | --- | --- | --- | --- |
| norm1 | 12.1 | 0.9 % | 0.0 % | – |
| qkv_proj | 245.5 | 17.8 % | 23.0 % | 5.80 |
| rope | 208.2 | 15.1 % | 0.0 % | – |
| sdpa | 112.2 | 8.2 % | 8.0 % | 4.43 |
| o_proj | 94.1 | 6.8 % | 7.7 % | 5.04 |
| layer_scale1 | 43.5 | 3.2 % | 0.0 % | – |
| norm2 | 11.8 | 0.9 % | 0.0 % | – |
| ff_in | 275.1 | 20.0 % | 30.7 % | 6.89 |
| activation | 28.8 | 2.1 % | 0.0 % | – |
| ff_out | 303.7 | 22.1 % | 30.7 % | 6.25 |
| layer_scale2 | 41.4 | 3.0 % | 0.0 % | – |

Parts outside the blocks: patch_embedding 390.7 µs, rope_tables 27.0 µs, final_norm 12.4 µs.

## ViT-S/16, fp32, 512 px, batch size 1

| Part | Time (µs) | Time share | FLOP share | TFLOP/s |
| --- | --- | --- | --- | --- |
| norm1 | 9.9 | 0.8 % | 0.0 % | – |
| qkv_proj | 197.9 | 15.1 % | 17.3 % | 4.60 |
| rope | 143.3 | 10.9 % | 0.0 % | – |
| sdpa | 336.6 | 25.7 % | 30.9 % | 4.83 |
| o_proj | 74.2 | 5.7 % | 5.8 % | 4.09 |
| layer_scale1 | 30.4 | 2.3 % | 0.0 % | – |
| norm2 | 9.4 | 0.7 % | 0.0 % | – |
| ff_in | 216.6 | 16.5 % | 23.0 % | 5.61 |
| activation | 20.3 | 1.5 % | 0.0 % | – |
| ff_out | 240.8 | 18.4 % | 23.0 % | 5.04 |
| layer_scale2 | 29.2 | 2.2 % | 0.0 % | – |

Parts outside the blocks: patch_embedding 235.0 µs, rope_tables 28.4 µs, final_norm 9.0 µs.

## ViT-S/16, fp32, 512 px, batch size 8

| Part | Time (µs) | Time share | FLOP share | TFLOP/s |
| --- | --- | --- | --- | --- |
| norm1 | 66.6 | 0.7 % | 0.0 % | – |
| qkv_proj | 1245.3 | 12.3 % | 17.3 % | 5.85 |
| rope | 1438.1 | 14.2 % | 0.0 % | – |
| sdpa | 2580.2 | 25.6 % | 30.9 % | 5.04 |
| o_proj | 484.5 | 4.8 % | 5.8 % | 5.01 |
| layer_scale1 | 347.5 | 3.4 % | 0.0 % | – |
| norm2 | 64.4 | 0.6 % | 0.0 % | – |
| ff_in | 1585.6 | 15.7 % | 23.0 % | 6.12 |
| activation | 317.1 | 3.1 % | 0.0 % | – |
| ff_out | 1626.7 | 16.1 % | 23.0 % | 5.97 |
| layer_scale2 | 338.4 | 3.4 % | 0.0 % | – |

Parts outside the blocks: patch_embedding 2019.4 µs, rope_tables 27.6 µs, final_norm 67.9 µs.

## ViT-S/16, fp16-cast, 224 px, batch size 1

| Part | Time (µs) | Time share | FLOP share | TFLOP/s |
| --- | --- | --- | --- | --- |
| norm1 | 5.5 | 2.0 % | 0.0 % | – |
| qkv_proj | 66.6 | 24.7 % | 23.0 % | 2.67 |
| rope | 61.5 | 22.8 % | 0.0 % | – |
| sdpa | 10.5 | 3.9 % | 8.0 % | 5.91 |
| o_proj | 31.0 | 11.5 % | 7.7 % | 1.91 |
| layer_scale1 | 8.4 | 3.1 % | 0.0 % | – |
| norm2 | 5.6 | 2.1 % | 0.0 % | – |
| ff_in | 29.1 | 10.8 % | 30.7 % | 8.16 |
| activation | 4.8 | 1.8 % | 0.0 % | – |
| ff_out | 37.8 | 14.0 % | 30.7 % | 6.27 |
| layer_scale2 | 9.2 | 3.4 % | 0.0 % | – |

Parts outside the blocks: patch_embedding 139.2 µs, rope_tables 25.1 µs, final_norm 5.5 µs.

## ViT-S/16, fp16-cast, 224 px, batch size 8

| Part | Time (µs) | Time share | FLOP share | TFLOP/s |
| --- | --- | --- | --- | --- |
| norm1 | 13.8 | 1.8 % | 0.0 % | – |
| qkv_proj | 95.0 | 12.6 % | 23.0 % | 14.97 |
| rope | 249.9 | 33.2 % | 0.0 % | – |
| sdpa | 39.9 | 5.3 % | 8.0 % | 12.43 |
| o_proj | 43.4 | 5.8 % | 7.7 % | 10.93 |
| layer_scale1 | 46.4 | 6.2 % | 0.0 % | – |
| norm2 | 11.7 | 1.6 % | 0.0 % | – |
| ff_in | 100.5 | 13.4 % | 30.7 % | 18.87 |
| activation | 22.5 | 3.0 % | 0.0 % | – |
| ff_out | 87.5 | 11.6 % | 30.7 % | 21.68 |
| layer_scale2 | 42.3 | 5.6 % | 0.0 % | – |

Parts outside the blocks: patch_embedding 349.4 µs, rope_tables 27.5 µs, final_norm 10.9 µs.

## ViT-S/16, fp16-cast, 512 px, batch size 1

| Part | Time (µs) | Time share | FLOP share | TFLOP/s |
| --- | --- | --- | --- | --- |
| norm1 | 9.5 | 1.6 % | 0.0 % | – |
| qkv_proj | 98.0 | 16.5 % | 17.3 % | 9.29 |
| rope | 140.0 | 23.6 % | 0.0 % | – |
| sdpa | 87.6 | 14.8 % | 30.9 % | 18.56 |
| o_proj | 38.1 | 6.4 % | 5.8 % | 7.96 |
| layer_scale1 | 28.4 | 4.8 % | 0.0 % | – |
| norm2 | 9.0 | 1.5 % | 0.0 % | – |
| ff_in | 78.8 | 13.3 % | 23.0 % | 15.40 |
| activation | 12.1 | 2.0 % | 0.0 % | – |
| ff_out | 63.3 | 10.7 % | 23.0 % | 19.17 |
| layer_scale2 | 28.1 | 4.7 % | 0.0 % | – |

Parts outside the blocks: patch_embedding 253.8 µs, rope_tables 29.9 µs, final_norm 8.3 µs.

## ViT-S/16, fp16-cast, 512 px, batch size 8

| Part | Time (µs) | Time share | FLOP share | TFLOP/s |
| --- | --- | --- | --- | --- |
| norm1 | 49.3 | 1.3 % | 0.0 % | – |
| qkv_proj | 350.8 | 9.2 % | 17.3 % | 20.76 |
| rope | 1035.5 | 27.1 % | 0.0 % | – |
| sdpa | 682.5 | 17.9 % | 30.9 % | 19.06 |
| o_proj | 152.5 | 4.0 % | 5.8 % | 15.92 |
| layer_scale1 | 254.3 | 6.7 % | 0.0 % | – |
| norm2 | 48.7 | 1.3 % | 0.0 % | – |
| ff_in | 427.9 | 11.2 % | 23.0 % | 22.69 |
| activation | 139.2 | 3.6 % | 0.0 % | – |
| ff_out | 432.5 | 11.3 % | 23.0 % | 22.45 |
| layer_scale2 | 249.1 | 6.5 % | 0.0 % | – |

Parts outside the blocks: patch_embedding 1551.6 µs, rope_tables 30.0 µs, final_norm 50.9 µs.

## ViT-H+/16, fp32, 224 px, batch size 1

| Part | Time (µs) | Time share | FLOP share | TFLOP/s |
| --- | --- | --- | --- | --- |
| norm1 | 7.2 | 0.3 % | 0.0 % | – |
| qkv_proj | 438.2 | 19.1 % | 18.4 % | 4.51 |
| rope | 106.3 | 4.6 % | 0.0 % | – |
| sdpa | 56.0 | 2.4 % | 1.9 % | 3.69 |
| o_proj | 143.3 | 6.3 % | 6.1 % | 4.60 |
| layer_scale1 | 17.6 | 0.8 % | 0.0 % | – |
| norm2 | 8.3 | 0.4 % | 0.0 % | – |
| ff_in | 1003.5 | 43.9 % | 49.0 % | 5.25 |
| activation | 28.7 | 1.3 % | 0.0 % | – |
| ff_out | 462.8 | 20.2 % | 24.5 % | 5.69 |
| layer_scale2 | 16.3 | 0.7 % | 0.0 % | – |

Parts outside the blocks: patch_embedding 205.4 µs, rope_tables 22.6 µs, final_norm 7.4 µs.

## ViT-H+/16, fp32, 224 px, batch size 8

| Part | Time (µs) | Time share | FLOP share | TFLOP/s |
| --- | --- | --- | --- | --- |
| norm1 | 41.9 | 0.3 % | 0.0 % | – |
| qkv_proj | 2626.7 | 16.6 % | 18.4 % | 6.02 |
| rope | 866.1 | 5.5 % | 0.0 % | – |
| sdpa | 408.4 | 2.6 % | 1.9 % | 4.05 |
| o_proj | 856.5 | 5.4 % | 6.1 % | 6.15 |
| layer_scale1 | 179.4 | 1.1 % | 0.0 % | – |
| norm2 | 42.5 | 0.3 % | 0.0 % | – |
| ff_in | 6648.1 | 42.1 % | 49.0 % | 6.34 |
| activation | 490.6 | 3.1 % | 0.0 % | – |
| ff_out | 3461.4 | 21.9 % | 24.5 % | 6.09 |
| layer_scale2 | 163.7 | 1.0 % | 0.0 % | – |

Parts outside the blocks: patch_embedding 1709.0 µs, rope_tables 25.9 µs, final_norm 44.2 µs.

## ViT-H+/16, fp32, 512 px, batch size 1

| Part | Time (µs) | Time share | FLOP share | TFLOP/s |
| --- | --- | --- | --- | --- |
| norm1 | 26.9 | 0.2 % | 0.0 % | – |
| qkv_proj | 1661.4 | 15.2 % | 17.0 % | 6.09 |
| rope | 470.0 | 4.3 % | 0.0 % | – |
| sdpa | 1144.8 | 10.5 % | 9.1 % | 4.74 |
| o_proj | 558.5 | 5.1 % | 5.7 % | 6.04 |
| layer_scale1 | 107.5 | 1.0 % | 0.0 % | – |
| norm2 | 28.0 | 0.3 % | 0.0 % | – |
| ff_in | 4492.0 | 41.1 % | 45.4 % | 6.01 |
| activation | 291.1 | 2.7 % | 0.0 % | – |
| ff_out | 2036.1 | 18.6 % | 22.7 % | 6.62 |
| layer_scale2 | 105.4 | 1.0 % | 0.0 % | – |

Parts outside the blocks: patch_embedding 821.7 µs, rope_tables 28.0 µs, final_norm 28.4 µs.

## ViT-H+/16, fp32, 512 px, batch size 8

| Part | Time (µs) | Time share | FLOP share | TFLOP/s |
| --- | --- | --- | --- | --- |
| norm1 | 295.9 | 0.3 % | 0.0 % | – |
| qkv_proj | 14592.8 | 15.2 % | 17.0 % | 5.55 |
| rope | 5305.1 | 5.5 % | 0.0 % | – |
| sdpa | 9521.0 | 9.9 % | 9.1 % | 4.56 |
| o_proj | 4901.6 | 5.1 % | 5.7 % | 5.50 |
| layer_scale1 | 1350.6 | 1.4 % | 0.0 % | – |
| norm2 | 299.3 | 0.3 % | 0.0 % | – |
| ff_in | 36093.3 | 37.6 % | 45.4 % | 5.98 |
| activation | 2602.8 | 2.7 % | 0.0 % | – |
| ff_out | 19882.5 | 20.7 % | 22.7 % | 5.43 |
| layer_scale2 | 1266.1 | 1.3 % | 0.0 % | – |

Parts outside the blocks: patch_embedding 6833.1 µs, rope_tables 27.7 µs, final_norm 290.6 µs.

## ViT-H+/16, fp16-cast, 224 px, batch size 1

| Part | Time (µs) | Time share | FLOP share | TFLOP/s |
| --- | --- | --- | --- | --- |
| norm1 | 7.1 | 0.9 % | 0.0 % | – |
| qkv_proj | 130.5 | 15.9 % | 18.4 % | 15.15 |
| rope | 103.3 | 12.6 % | 0.0 % | – |
| sdpa | 18.3 | 2.2 % | 1.9 % | 11.31 |
| o_proj | 47.9 | 5.9 % | 6.1 % | 13.74 |
| layer_scale1 | 16.8 | 2.1 % | 0.0 % | – |
| norm2 | 7.2 | 0.9 % | 0.0 % | – |
| ff_in | 306.8 | 37.5 % | 49.0 % | 17.17 |
| activation | 14.2 | 1.7 % | 0.0 % | – |
| ff_out | 149.5 | 18.3 % | 24.5 % | 17.62 |
| layer_scale2 | 17.4 | 2.1 % | 0.0 % | – |

Parts outside the blocks: patch_embedding 229.7 µs, rope_tables 29.3 µs, final_norm 7.0 µs.

## ViT-H+/16, fp16-cast, 224 px, batch size 8

| Part | Time (µs) | Time share | FLOP share | TFLOP/s |
| --- | --- | --- | --- | --- |
| norm1 | 31.5 | 0.6 % | 0.0 % | – |
| qkv_proj | 708.6 | 13.1 % | 18.4 % | 22.31 |
| rope | 733.1 | 13.5 % | 0.0 % | – |
| sdpa | 126.0 | 2.3 % | 1.9 % | 13.13 |
| o_proj | 257.2 | 4.7 % | 6.1 % | 20.49 |
| layer_scale1 | 147.2 | 2.7 % | 0.0 % | – |
| norm2 | 31.3 | 0.6 % | 0.0 % | – |
| ff_in | 1908.6 | 35.2 % | 49.0 % | 22.09 |
| activation | 216.0 | 4.0 % | 0.0 % | – |
| ff_out | 1102.6 | 20.3 % | 24.5 % | 19.12 |
| layer_scale2 | 164.8 | 3.0 % | 0.0 % | – |

Parts outside the blocks: patch_embedding 1411.4 µs, rope_tables 33.9 µs, final_norm 37.4 µs.

## ViT-H+/16, fp16-cast, 512 px, batch size 1

| Part | Time (µs) | Time share | FLOP share | TFLOP/s |
| --- | --- | --- | --- | --- |
| norm1 | 24.0 | 0.6 % | 0.0 % | – |
| qkv_proj | 576.3 | 14.6 % | 17.0 % | 17.55 |
| rope | 410.7 | 10.4 % | 0.0 % | – |
| sdpa | 367.0 | 9.3 % | 9.1 % | 14.77 |
| o_proj | 198.8 | 5.0 % | 5.7 % | 16.96 |
| layer_scale1 | 91.3 | 2.3 % | 0.0 % | – |
| norm2 | 22.3 | 0.6 % | 0.0 % | – |
| ff_in | 1355.3 | 34.4 % | 45.4 % | 19.90 |
| activation | 123.2 | 3.1 % | 0.0 % | – |
| ff_out | 685.1 | 17.4 % | 22.7 % | 19.69 |
| layer_scale2 | 89.9 | 2.3 % | 0.0 % | – |

Parts outside the blocks: patch_embedding 792.4 µs, rope_tables 32.0 µs, final_norm 23.5 µs.

## ViT-H+/16, fp16-cast, 512 px, batch size 8

| Part | Time (µs) | Time share | FLOP share | TFLOP/s |
| --- | --- | --- | --- | --- |
| norm1 | 204.7 | 0.6 % | 0.0 % | – |
| qkv_proj | 4097.1 | 12.0 % | 17.0 % | 19.75 |
| rope | 4397.6 | 12.9 % | 0.0 % | – |
| sdpa | 2785.6 | 8.2 % | 9.1 % | 15.57 |
| o_proj | 1563.5 | 4.6 % | 5.7 % | 17.25 |
| layer_scale1 | 1062.5 | 3.1 % | 0.0 % | – |
| norm2 | 207.2 | 0.6 % | 0.0 % | – |
| ff_in | 10393.2 | 30.5 % | 45.4 % | 20.76 |
| activation | 1390.1 | 4.1 % | 0.0 % | – |
| ff_out | 6898.7 | 20.3 % | 22.7 % | 15.64 |
| layer_scale2 | 1028.6 | 3.0 % | 0.0 % | – |

Parts outside the blocks: patch_embedding 5851.7 µs, rope_tables 31.6 µs, final_norm 216.6 µs.
