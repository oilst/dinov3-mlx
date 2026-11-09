# Block profile: Hugging Face backbone on torch-mps

Written by `scripts/baseline_table.py` on 2026-11-07 from `results/block-profile-torch-mps.jsonl`. Spec: Section 3.11 of `specs/03-baseline/plan.html` (D6).

Each part of block 1 runs alone in a loop with one synchronize at the end. The time is the median of five loops. The FLOP share comes from `dinov3_mlx.flops`. Limits of the checks: 15 %.

The forward pass runs before and after the parts; the check uses the mean of the two. A large change between the two shows a change of the speed of the Mac during the profile. At batch size 1 a block alone takes longer than a block inside the forward pass, so Forward / L gives the time of one block inside the forward pass (finding F4 of `specs/03-baseline/verify.html`). Lines from before F4 have only the time after the parts.

## Summary

| Model size | Dtype mode | Input resolution | Batch size | Block (ms) | Forward / L (ms) | Sum of parts / block | Check | Forward pass, before → after (ms) | (L × block + outer) / forward | Check |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ViT-S/16 | fp32 | 224 | 1 | 0.36 | 0.23 | 0.967 | pass | 2.74 → 2.75 | 1.611 | outside 15 % |
| ViT-S/16 | fp32 | 224 | 8 | 1.30 | 1.29 | 0.980 | pass | 15.49 → 15.45 | 1.031 | pass |
| ViT-S/16 | fp32 | 512 | 1 | 1.06 | 1.00 | 1.005 | pass | 11.78 → 12.14 | 1.080 | pass |
| ViT-S/16 | fp32 | 512 | 8 | 7.25 | 7.52 | 1.065 | pass | 96.56 → 83.93 | 0.979 | pass |
| ViT-S/16 | fp16-cast | 224 | 1 | 0.29 | 0.18 | 0.926 | pass | 2.13 → 2.12 | 1.690 | outside 15 % |
| ViT-S/16 | fp16-cast | 224 | 8 | 0.58 | 0.45 | 0.998 | pass | 4.96 → 5.80 | 1.342 | outside 15 % |
| ViT-S/16 | fp16-cast | 512 | 1 | 0.47 | 0.39 | 1.003 | pass | 4.62 → 4.69 | 1.252 | outside 15 % |
| ViT-S/16 | fp16-cast | 512 | 8 | 3.07 | 3.04 | 0.933 | pass | 35.43 → 37.47 | 1.046 | pass |
| ViT-H+/16 | fp32 | 224 | 1 | 2.00 | 1.77 | 0.933 | pass | 53.82 → 59.36 | 1.133 | pass |
| ViT-H+/16 | fp32 | 224 | 8 | 12.60 | 12.44 | 0.960 | pass | 394.44 → 401.81 | 1.016 | pass |
| ViT-H+/16 | fp32 | 512 | 1 | 8.68 | 8.58 | 0.984 | pass | 276.56 → 272.32 | 1.014 | pass |
| ViT-H+/16 | fp32 | 512 | 8 | 70.38 | 72.23 | 1.053 | pass | 2312.29 → 2310.37 | 0.977 | pass |
| ViT-H+/16 | fp16-cast | 224 | 1 | 0.74 | 0.64 | 0.935 | pass | 20.66 → 20.12 | 1.170 | outside 15 % |
| ViT-H+/16 | fp16-cast | 224 | 8 | 3.98 | 3.91 | 0.972 | pass | 125.26 → 124.94 | 1.026 | pass |
| ViT-H+/16 | fp16-cast | 512 | 1 | 2.75 | 2.69 | 0.954 | pass | 85.43 → 87.04 | 1.028 | pass |
| ViT-H+/16 | fp16-cast | 512 | 8 | 23.84 | 24.00 | 0.987 | pass | 759.68 → 776.37 | 0.999 | pass |

## ViT-S/16, fp32, 224 px, batch size 1

| Part | Time (µs) | Time share | FLOP share | TFLOP/s |
| --- | --- | --- | --- | --- |
| norm1 | 5.8 | 1.7 % | 0.0 % | – |
| qkv_proj | 78.8 | 22.7 % | 23.0 % | 2.26 |
| rope | 62.6 | 18.0 % | 0.0 % | – |
| sdpa | 22.5 | 6.5 % | 8.0 % | 2.75 |
| o_proj | 35.7 | 10.3 % | 7.7 % | 1.66 |
| layer_scale1 | 8.3 | 2.4 % | 0.0 % | – |
| norm2 | 5.8 | 1.7 % | 0.0 % | – |
| ff_in | 48.6 | 14.0 % | 30.7 % | 4.88 |
| activation | 4.2 | 1.2 % | 0.0 % | – |
| ff_out | 67.0 | 19.3 % | 30.7 % | 3.54 |
| layer_scale2 | 8.0 | 2.3 % | 0.0 % | – |

Parts outside the blocks: patch_embedding 90.7 µs, rope_tables 21.9 µs, final_norm 5.9 µs.

## ViT-S/16, fp32, 224 px, batch size 8

| Part | Time (µs) | Time share | FLOP share | TFLOP/s |
| --- | --- | --- | --- | --- |
| norm1 | 11.5 | 0.9 % | 0.0 % | – |
| qkv_proj | 233.1 | 18.3 % | 23.0 % | 6.10 |
| rope | 190.9 | 15.0 % | 0.0 % | – |
| sdpa | 98.1 | 7.7 % | 8.0 % | 5.06 |
| o_proj | 88.5 | 7.0 % | 7.7 % | 5.36 |
| layer_scale1 | 39.0 | 3.1 % | 0.0 % | – |
| norm2 | 11.6 | 0.9 % | 0.0 % | – |
| ff_in | 255.9 | 20.1 % | 30.7 % | 7.41 |
| activation | 24.9 | 2.0 % | 0.0 % | – |
| ff_out | 281.3 | 22.1 % | 30.7 % | 6.74 |
| layer_scale2 | 37.8 | 3.0 % | 0.0 % | – |

Parts outside the blocks: patch_embedding 330.0 µs, rope_tables 21.8 µs, final_norm 10.3 µs.

## ViT-S/16, fp32, 512 px, batch size 1

| Part | Time (µs) | Time share | FLOP share | TFLOP/s |
| --- | --- | --- | --- | --- |
| norm1 | 9.0 | 0.8 % | 0.0 % | – |
| qkv_proj | 166.9 | 15.7 % | 17.3 % | 5.45 |
| rope | 116.3 | 10.9 % | 0.0 % | – |
| sdpa | 265.9 | 25.0 % | 30.9 % | 6.12 |
| o_proj | 65.1 | 6.1 % | 5.8 % | 4.66 |
| layer_scale1 | 24.7 | 2.3 % | 0.0 % | – |
| norm2 | 8.7 | 0.8 % | 0.0 % | – |
| ff_in | 176.3 | 16.6 % | 23.0 % | 6.88 |
| activation | 16.6 | 1.6 % | 0.0 % | – |
| ff_out | 189.0 | 17.8 % | 23.0 % | 6.42 |
| layer_scale2 | 24.6 | 2.3 % | 0.0 % | – |

Parts outside the blocks: patch_embedding 188.7 µs, rope_tables 22.5 µs, final_norm 8.5 µs.

## ViT-S/16, fp32, 512 px, batch size 8

| Part | Time (µs) | Time share | FLOP share | TFLOP/s |
| --- | --- | --- | --- | --- |
| norm1 | 49.4 | 0.6 % | 0.0 % | – |
| qkv_proj | 983.9 | 12.7 % | 17.3 % | 7.40 |
| rope | 1052.7 | 13.6 % | 0.0 % | – |
| sdpa | 1996.8 | 25.8 % | 30.9 % | 6.52 |
| o_proj | 372.6 | 4.8 % | 5.8 % | 6.52 |
| layer_scale1 | 259.7 | 3.4 % | 0.0 % | – |
| norm2 | 48.1 | 0.6 % | 0.0 % | – |
| ff_in | 1223.5 | 15.8 % | 23.0 % | 7.94 |
| activation | 213.4 | 2.8 % | 0.0 % | – |
| ff_out | 1277.8 | 16.5 % | 23.0 % | 7.60 |
| layer_scale2 | 249.6 | 3.2 % | 0.0 % | – |

Parts outside the blocks: patch_embedding 1267.1 µs, rope_tables 22.4 µs, final_norm 39.9 µs.

## ViT-S/16, fp16-cast, 224 px, batch size 1

| Part | Time (µs) | Time share | FLOP share | TFLOP/s |
| --- | --- | --- | --- | --- |
| norm1 | 5.6 | 2.1 % | 0.0 % | – |
| qkv_proj | 67.4 | 25.2 % | 23.0 % | 2.64 |
| rope | 61.9 | 23.2 % | 0.0 % | – |
| sdpa | 10.3 | 3.9 % | 8.0 % | 6.01 |
| o_proj | 31.1 | 11.6 % | 7.7 % | 1.91 |
| layer_scale1 | 7.9 | 3.0 % | 0.0 % | – |
| norm2 | 5.6 | 2.1 % | 0.0 % | – |
| ff_in | 28.3 | 10.6 % | 30.7 % | 8.39 |
| activation | 4.0 | 1.5 % | 0.0 % | – |
| ff_out | 37.4 | 14.0 % | 30.7 % | 6.35 |
| layer_scale2 | 7.9 | 2.9 % | 0.0 % | – |

Parts outside the blocks: patch_embedding 99.4 µs, rope_tables 24.8 µs, final_norm 5.5 µs.

## ViT-S/16, fp16-cast, 224 px, batch size 8

| Part | Time (µs) | Time share | FLOP share | TFLOP/s |
| --- | --- | --- | --- | --- |
| norm1 | 10.4 | 1.8 % | 0.0 % | – |
| qkv_proj | 83.7 | 14.5 % | 23.0 % | 17.00 |
| rope | 175.7 | 30.5 % | 0.0 % | – |
| sdpa | 29.4 | 5.1 % | 8.0 % | 16.91 |
| o_proj | 40.2 | 7.0 % | 7.7 % | 11.80 |
| layer_scale1 | 33.6 | 5.8 % | 0.0 % | – |
| norm2 | 10.3 | 1.8 % | 0.0 % | – |
| ff_in | 73.4 | 12.8 % | 30.7 % | 25.84 |
| activation | 15.0 | 2.6 % | 0.0 % | – |
| ff_out | 70.3 | 12.2 % | 30.7 % | 26.98 |
| layer_scale2 | 33.5 | 5.8 % | 0.0 % | – |

Parts outside the blocks: patch_embedding 268.2 µs, rope_tables 24.8 µs, final_norm 10.0 µs.

## ViT-S/16, fp16-cast, 512 px, batch size 1

| Part | Time (µs) | Time share | FLOP share | TFLOP/s |
| --- | --- | --- | --- | --- |
| norm1 | 8.3 | 1.8 % | 0.0 % | – |
| qkv_proj | 76.1 | 16.2 % | 17.3 % | 11.96 |
| rope | 109.6 | 23.4 % | 0.0 % | – |
| sdpa | 67.2 | 14.3 % | 30.9 % | 24.20 |
| o_proj | 35.3 | 7.5 % | 5.8 % | 8.61 |
| layer_scale1 | 21.5 | 4.6 % | 0.0 % | – |
| norm2 | 8.2 | 1.7 % | 0.0 % | – |
| ff_in | 56.0 | 11.9 % | 23.0 % | 21.69 |
| activation | 9.8 | 2.1 % | 0.0 % | – |
| ff_out | 53.7 | 11.5 % | 23.0 % | 22.60 |
| layer_scale2 | 22.8 | 4.9 % | 0.0 % | – |

Parts outside the blocks: patch_embedding 194.8 µs, rope_tables 25.0 µs, final_norm 8.1 µs.

## ViT-S/16, fp16-cast, 512 px, batch size 8

| Part | Time (µs) | Time share | FLOP share | TFLOP/s |
| --- | --- | --- | --- | --- |
| norm1 | 38.8 | 1.4 % | 0.0 % | – |
| qkv_proj | 266.0 | 9.3 % | 17.3 % | 27.38 |
| rope | 780.2 | 27.2 % | 0.0 % | – |
| sdpa | 510.5 | 17.8 % | 30.9 % | 25.49 |
| o_proj | 119.8 | 4.2 % | 5.8 % | 20.27 |
| layer_scale1 | 184.8 | 6.4 % | 0.0 % | – |
| norm2 | 38.8 | 1.4 % | 0.0 % | – |
| ff_in | 313.4 | 10.9 % | 23.0 % | 30.98 |
| activation | 102.1 | 3.6 % | 0.0 % | – |
| ff_out | 316.8 | 11.0 % | 23.0 % | 30.65 |
| layer_scale2 | 195.9 | 6.8 % | 0.0 % | – |

Parts outside the blocks: patch_embedding 1185.3 µs, rope_tables 25.6 µs, final_norm 37.7 µs.

## ViT-H+/16, fp32, 224 px, batch size 1

| Part | Time (µs) | Time share | FLOP share | TFLOP/s |
| --- | --- | --- | --- | --- |
| norm1 | 7.5 | 0.4 % | 0.0 % | – |
| qkv_proj | 332.1 | 17.8 % | 18.4 % | 5.95 |
| rope | 93.3 | 5.0 % | 0.0 % | – |
| sdpa | 44.4 | 2.4 % | 1.9 % | 4.66 |
| o_proj | 113.2 | 6.1 % | 6.1 % | 5.82 |
| layer_scale1 | 15.3 | 0.8 % | 0.0 % | – |
| norm2 | 7.4 | 0.4 % | 0.0 % | – |
| ff_in | 781.4 | 41.9 % | 49.0 % | 6.74 |
| activation | 25.3 | 1.4 % | 0.0 % | – |
| ff_out | 427.6 | 23.0 % | 24.5 % | 6.16 |
| layer_scale2 | 15.4 | 0.8 % | 0.0 % | – |

Parts outside the blocks: patch_embedding 198.1 µs, rope_tables 22.0 µs, final_norm 7.4 µs.

## ViT-H+/16, fp32, 224 px, batch size 8

| Part | Time (µs) | Time share | FLOP share | TFLOP/s |
| --- | --- | --- | --- | --- |
| norm1 | 31.4 | 0.3 % | 0.0 % | – |
| qkv_proj | 2003.0 | 16.6 % | 18.4 % | 7.89 |
| rope | 690.5 | 5.7 % | 0.0 % | – |
| sdpa | 319.2 | 2.6 % | 1.9 % | 5.18 |
| o_proj | 677.6 | 5.6 % | 6.1 % | 7.78 |
| layer_scale1 | 136.4 | 1.1 % | 0.0 % | – |
| norm2 | 32.4 | 0.3 % | 0.0 % | – |
| ff_in | 5085.3 | 42.0 % | 49.0 % | 8.29 |
| activation | 321.5 | 2.7 % | 0.0 % | – |
| ff_out | 2670.4 | 22.1 % | 24.5 % | 7.89 |
| layer_scale2 | 133.6 | 1.1 % | 0.0 % | – |

Parts outside the blocks: patch_embedding 1388.8 µs, rope_tables 22.2 µs, final_norm 31.0 µs.

## ViT-H+/16, fp32, 512 px, batch size 1

| Part | Time (µs) | Time share | FLOP share | TFLOP/s |
| --- | --- | --- | --- | --- |
| norm1 | 22.9 | 0.3 % | 0.0 % | – |
| qkv_proj | 1328.3 | 15.6 % | 17.0 % | 7.62 |
| rope | 351.6 | 4.1 % | 0.0 % | – |
| sdpa | 838.0 | 9.8 % | 9.1 % | 6.47 |
| o_proj | 451.4 | 5.3 % | 5.7 % | 7.47 |
| layer_scale1 | 84.9 | 1.0 % | 0.0 % | – |
| norm2 | 22.6 | 0.3 % | 0.0 % | – |
| ff_in | 3424.4 | 40.1 % | 45.4 % | 7.88 |
| activation | 179.0 | 2.1 % | 0.0 % | – |
| ff_out | 1755.8 | 20.6 % | 22.7 % | 7.68 |
| layer_scale2 | 80.7 | 0.9 % | 0.0 % | – |

Parts outside the blocks: patch_embedding 630.0 µs, rope_tables 22.7 µs, final_norm 19.8 µs.

## ViT-H+/16, fp32, 512 px, batch size 8

| Part | Time (µs) | Time share | FLOP share | TFLOP/s |
| --- | --- | --- | --- | --- |
| norm1 | 224.6 | 0.3 % | 0.0 % | – |
| qkv_proj | 10649.2 | 14.4 % | 17.0 % | 7.60 |
| rope | 3986.6 | 5.4 % | 0.0 % | – |
| sdpa | 7257.4 | 9.8 % | 9.1 % | 5.98 |
| o_proj | 3626.8 | 4.9 % | 5.7 % | 7.44 |
| layer_scale1 | 982.6 | 1.3 % | 0.0 % | – |
| norm2 | 222.1 | 0.3 % | 0.0 % | – |
| ff_in | 27544.7 | 37.2 % | 45.4 % | 7.83 |
| activation | 1608.4 | 2.2 % | 0.0 % | – |
| ff_out | 17011.5 | 23.0 % | 22.7 % | 6.34 |
| layer_scale2 | 975.3 | 1.3 % | 0.0 % | – |

Parts outside the blocks: patch_embedding 5219.3 µs, rope_tables 22.4 µs, final_norm 160.5 µs.

## ViT-H+/16, fp16-cast, 224 px, batch size 1

| Part | Time (µs) | Time share | FLOP share | TFLOP/s |
| --- | --- | --- | --- | --- |
| norm1 | 7.0 | 1.0 % | 0.0 % | – |
| qkv_proj | 118.3 | 17.1 % | 18.4 % | 16.70 |
| rope | 97.9 | 14.2 % | 0.0 % | – |
| sdpa | 16.7 | 2.4 % | 1.9 % | 12.38 |
| o_proj | 46.3 | 6.7 % | 6.1 % | 14.24 |
| layer_scale1 | 15.5 | 2.2 % | 0.0 % | – |
| norm2 | 6.9 | 1.0 % | 0.0 % | – |
| ff_in | 235.0 | 34.0 % | 49.0 % | 22.42 |
| activation | 11.9 | 1.7 % | 0.0 % | – |
| ff_out | 120.6 | 17.4 % | 24.5 % | 21.85 |
| layer_scale2 | 15.0 | 2.2 % | 0.0 % | – |

Parts outside the blocks: patch_embedding 181.8 µs, rope_tables 25.3 µs, final_norm 7.0 µs.

## ViT-H+/16, fp16-cast, 224 px, batch size 8

| Part | Time (µs) | Time share | FLOP share | TFLOP/s |
| --- | --- | --- | --- | --- |
| norm1 | 26.1 | 0.7 % | 0.0 % | – |
| qkv_proj | 533.8 | 13.8 % | 18.4 % | 29.61 |
| rope | 533.9 | 13.8 % | 0.0 % | – |
| sdpa | 96.2 | 2.5 % | 1.9 % | 17.20 |
| o_proj | 197.7 | 5.1 % | 6.1 % | 26.65 |
| layer_scale1 | 110.3 | 2.9 % | 0.0 % | – |
| norm2 | 26.0 | 0.7 % | 0.0 % | – |
| ff_in | 1371.0 | 35.4 % | 49.0 % | 30.75 |
| activation | 148.2 | 3.8 % | 0.0 % | – |
| ff_out | 719.7 | 18.6 % | 24.5 % | 29.29 |
| layer_scale2 | 105.1 | 2.7 % | 0.0 % | – |

Parts outside the blocks: patch_embedding 955.0 µs, rope_tables 24.7 µs, final_norm 22.7 µs.

## ViT-H+/16, fp16-cast, 512 px, batch size 1

| Part | Time (µs) | Time share | FLOP share | TFLOP/s |
| --- | --- | --- | --- | --- |
| norm1 | 18.0 | 0.7 % | 0.0 % | – |
| qkv_proj | 368.6 | 14.0 % | 17.0 % | 27.45 |
| rope | 291.9 | 11.1 % | 0.0 % | – |
| sdpa | 221.3 | 8.4 % | 9.1 % | 24.50 |
| o_proj | 129.4 | 4.9 % | 5.7 % | 26.06 |
| layer_scale1 | 65.2 | 2.5 % | 0.0 % | – |
| norm2 | 17.5 | 0.7 % | 0.0 % | – |
| ff_in | 899.2 | 34.3 % | 45.4 % | 30.00 |
| activation | 79.7 | 3.0 % | 0.0 % | – |
| ff_out | 470.1 | 17.9 % | 22.7 % | 28.69 |
| layer_scale2 | 63.5 | 2.4 % | 0.0 % | – |

Parts outside the blocks: patch_embedding 573.5 µs, rope_tables 25.4 µs, final_norm 16.6 µs.

## ViT-H+/16, fp16-cast, 512 px, batch size 8

| Part | Time (µs) | Time share | FLOP share | TFLOP/s |
| --- | --- | --- | --- | --- |
| norm1 | 132.0 | 0.6 % | 0.0 % | – |
| qkv_proj | 2815.5 | 12.0 % | 17.0 % | 28.74 |
| rope | 3031.2 | 12.9 % | 0.0 % | – |
| sdpa | 1926.9 | 8.2 % | 9.1 % | 22.51 |
| o_proj | 1003.9 | 4.3 % | 5.7 % | 26.87 |
| layer_scale1 | 752.1 | 3.2 % | 0.0 % | – |
| norm2 | 133.7 | 0.6 % | 0.0 % | – |
| ff_in | 7325.4 | 31.1 % | 45.4 % | 29.46 |
| activation | 955.3 | 4.1 % | 0.0 % | – |
| ff_out | 4707.2 | 20.0 % | 22.7 % | 22.92 |
| layer_scale2 | 751.0 | 3.2 % | 0.0 % | – |

Parts outside the blocks: patch_embedding 4217.1 µs, rope_tables 25.9 µs, final_norm 137.4 µs.
