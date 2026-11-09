# Parity of mlx

Made by `scripts/parity_table.py` on 2026-11-07 from the parity tests (mlx 0.32.4). The reference output is torch-cpu in fp32. Each value is the minimum over the configurations of the row. The patch minimum is report only. Section 3 gives the thresholds.

## 1 Parity against the reference output

Each row covers 12 configurations: 3 test images × 4 input resolutions.

| Model size | Dtype mode | Measured | Non-finite | Frame | Patch mean | Patch 1st percentile | Patch minimum | Result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ViT-S/16 | fp32 | 12 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | pass |
| ViT-S/16 | fp16-cast | 12 | 0 | 0.999995 | 0.999997 | 0.999986 | 0.999944 | pass |
| ViT-S/16 | bf16-cast | 12 | 0 | 0.999545 | 0.999779 | 0.999560 | 0.998911 | pass |
| ViT-S/16 | fp16-autocast | 12 | 0 | 0.999997 | 0.999998 | 0.999995 | 0.999981 | pass |
| ViT-S/16 | bf16-autocast | 12 | 0 | 0.999862 | 0.999924 | 0.999650 | 0.999079 | pass |
| ViT-S+/16 | fp32 | 12 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | pass |
| ViT-S+/16 | fp16-cast | 12 | 0 | 0.999973 | 0.999920 | 0.999808 | 0.999714 | pass |
| ViT-S+/16 | bf16-cast | 12 | 0 | 0.998847 | 0.995816 | 0.990381 | 0.983124 | pass |
| ViT-S+/16 | fp16-autocast | 12 | 0 | 0.999986 | 0.999958 | 0.999904 | 0.999679 | pass |
| ViT-S+/16 | bf16-autocast | 12 | 0 | 0.998387 | 0.995136 | 0.988882 | 0.981135 | pass |
| ViT-B/16 | fp32 | 12 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | pass |
| ViT-B/16 | fp16-cast | 12 | 0 | 0.999995 | 0.999997 | 0.999991 | 0.999984 | pass |
| ViT-B/16 | bf16-cast | 12 | 0 | 0.999711 | 0.999821 | 0.999478 | 0.998675 | pass |
| ViT-B/16 | fp16-autocast | 12 | 0 | 0.999997 | 0.999998 | 0.999993 | 0.999978 | pass |
| ViT-B/16 | bf16-autocast | 12 | 0 | 0.999873 | 0.999899 | 0.999528 | 0.998174 | pass |
| ViT-L/16 | fp32 | 12 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | pass |
| ViT-L/16 | fp16-cast | 0 | 12 | – | – | – | – | expected failure (fp16 overflow) |
| ViT-L/16 | bf16-cast | 12 | 0 | 0.999767 | 0.999558 | 0.996394 | 0.978968 | pass |
| ViT-L/16 | fp16-autocast | 12 | 0 | 0.999998 | 0.999996 | 0.999966 | 0.999926 | pass |
| ViT-L/16 | bf16-autocast | 12 | 0 | 0.999924 | 0.999791 | 0.997288 | 0.990439 | pass |
| ViT-H+/16 | fp32 | 12 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | pass |
| ViT-H+/16 | fp16-cast | 6 | 6 | 0.999999 | 0.999998 | 0.999993 | 0.999969 | pass, 6 expected failures (fp16 overflow) |
| ViT-H+/16 | bf16-cast | 12 | 0 | 0.999833 | 0.999061 | 0.986725 | 0.958567 | pass |
| ViT-H+/16 | fp16-autocast | 12 | 0 | 0.999999 | 0.999981 | 0.999679 | 0.998518 | pass |
| ViT-H+/16 | bf16-autocast | 12 | 0 | 0.999947 | 0.999279 | 0.985938 | 0.956203 | pass |
| ViT-7B/16 | fp32 | 0 | 0 | – | – | – | – | skipped: ViT-7B/16 mlx fp32: ViT-7B/16 runs with fp16, bf16 or quantized weights only (CLAUDE.md run rules) |
| ViT-7B/16 | fp16-cast | 12 | 0 | 0.999990 | 0.999981 | 0.999654 | 0.998525 | pass |
| ViT-7B/16 | bf16-cast | 12 | 0 | 0.999739 | 0.999119 | 0.982649 | 0.912368 | pass |
| ViT-7B/16 | fp16-autocast | 12 | 0 | 0.999990 | 0.999986 | 0.999762 | 0.998173 | pass |
| ViT-7B/16 | bf16-autocast | 12 | 0 | 0.999898 | 0.999229 | 0.983737 | 0.918577 | pass |

Table 1. Parity of mlx against the reference output.

## 2 Batch consistency

A batch of 8 at 224 × 224 px against 8 runs with batch size 1, on mlx in the same dtype mode.

| Model size | Dtype mode | Measured | Non-finite | Frame | Patch mean | Patch 1st percentile | Patch minimum | Result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ViT-S/16 | fp32 | 1 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | pass |
| ViT-S/16 | fp16-cast | 1 | 0 | 0.999998 | 0.999999 | 0.999997 | 0.999997 | pass |
| ViT-S/16 | bf16-cast | 1 | 0 | 0.999864 | 0.999918 | 0.999821 | 0.999800 | pass |
| ViT-S/16 | fp16-autocast | 1 | 0 | 0.999999 | 0.999999 | 0.999999 | 0.999997 | pass |
| ViT-S/16 | bf16-autocast | 1 | 0 | 0.999949 | 0.999969 | 0.999914 | 0.999908 | pass |
| ViT-S+/16 | fp32 | 1 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | pass |
| ViT-S+/16 | fp16-cast | 1 | 0 | 0.999998 | 0.999998 | 0.999997 | 0.999996 | pass |
| ViT-S+/16 | bf16-cast | 1 | 0 | 0.999897 | 0.999913 | 0.999716 | 0.999541 | pass |
| ViT-S+/16 | fp16-autocast | 1 | 0 | 0.999999 | 0.999999 | 0.999998 | 0.999998 | pass |
| ViT-S+/16 | bf16-autocast | 1 | 0 | 0.999943 | 0.999960 | 0.999879 | 0.999759 | pass |
| ViT-B/16 | fp32 | 1 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | pass |
| ViT-B/16 | fp16-cast | 1 | 0 | 0.999999 | 0.999999 | 0.999996 | 0.999995 | pass |
| ViT-B/16 | bf16-cast | 1 | 0 | 0.999916 | 0.999913 | 0.999793 | 0.999511 | pass |
| ViT-B/16 | fp16-autocast | 1 | 0 | 0.999999 | 0.999999 | 0.999998 | 0.999995 | pass |
| ViT-B/16 | bf16-autocast | 1 | 0 | 0.999957 | 0.999964 | 0.999868 | 0.999197 | pass |
| ViT-L/16 | fp32 | 1 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | pass |
| ViT-L/16 | fp16-cast | 0 | 1 | – | – | – | – | expected failure (fp16 overflow) |
| ViT-L/16 | bf16-cast | 1 | 0 | 0.999907 | 0.999799 | 0.999107 | 0.997752 | pass |
| ViT-L/16 | fp16-autocast | 1 | 0 | 1.000000 | 0.999999 | 0.999997 | 0.999996 | pass |
| ViT-L/16 | bf16-autocast | 1 | 0 | 0.999979 | 0.999951 | 0.999819 | 0.999692 | pass |
| ViT-H+/16 | fp32 | 1 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | pass |
| ViT-H+/16 | fp16-cast | 0 | 1 | – | – | – | – | expected failure (fp16 overflow) |
| ViT-H+/16 | bf16-cast | 1 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | pass |
| ViT-H+/16 | fp16-autocast | 1 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | pass |
| ViT-H+/16 | bf16-autocast | 1 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | pass |
| ViT-7B/16 | fp32 | 0 | 0 | – | – | – | – | skipped: ViT-7B/16 mlx fp32: ViT-7B/16 runs with fp16, bf16 or quantized weights only (CLAUDE.md run rules) |
| ViT-7B/16 | fp16-cast | 1 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | pass |
| ViT-7B/16 | bf16-cast | 1 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | pass |
| ViT-7B/16 | fp16-autocast | 1 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | pass |
| ViT-7B/16 | bf16-autocast | 1 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | pass |

Table 2. Batch consistency.

## 3 Thresholds

| Dtype mode | Frame | Patch mean | Patch 1st percentile |
| --- | --- | --- | --- |
| fp32 | 0.9999 | 0.9999 | 0.9999 |
| fp16-cast | 0.999 | 0.999 | 0.99 |
| bf16-cast | 0.998 | 0.995 | 0.96 |
| fp16-autocast | 0.999 | 0.999 | 0.99 |
| bf16-autocast | 0.998 | 0.995 | 0.96 |

Table 3. Thresholds of `src/dinov3_mlx/parity.py`. On 2026-11-06 the user gave bf16 its own row: bf16 activations give lower parity on torch-mps and on torch-cpu alike.

## 4 Stored hidden states

Each check runs one part on the stored input of iss-kibo-lab/224x224 and compares with the stored output: the embeddings, each block alone, the final layer norm and the L + 1 hidden states in a chain (L is the number of blocks). In the cast modes the stored inputs are first rounded to the stream dtype. The fp32 rows assert; the four fp16 and bf16 dtype modes are report only. The minimum token cosine runs in fp64.

| Model size | Dtype mode | Check | Part | 1 − minimum token cosine | Largest difference | Largest relative difference |
| --- | --- | --- | --- | --- | --- | --- |
| ViT-S/16 | fp32 | stored-embeddings | iss-kibo-lab/224x224 | 9.46e-13 | 1.24e-05 | 1.162e-06 |
| ViT-S/16 | fp32 | stored-block | block-0 | 1.68e-11 | 0.0001755 | 3.691e-06 |
| ViT-S/16 | fp32 | stored-block | block-1 | 3.18e-13 | 2.289e-05 | 5.689e-07 |
| ViT-S/16 | fp32 | stored-block | block-2 | 2.78e-13 | 6.104e-05 | 2.421e-07 |
| ViT-S/16 | fp32 | stored-block | block-3 | 1.83e-13 | 1.717e-05 | 6.954e-08 |
| ViT-S/16 | fp32 | stored-block | block-4 | 9.59e-13 | 3.147e-05 | 1.05e-07 |
| ViT-S/16 | fp32 | stored-block | block-5 | 5.33e-13 | 6.104e-05 | 1.977e-07 |
| ViT-S/16 | fp32 | stored-block | block-6 | 1.38e-12 | 0.0004883 | 5.044e-07 |
| ViT-S/16 | fp32 | stored-block | block-7 | 1.59e-12 | 0.0001221 | 1.264e-07 |
| ViT-S/16 | fp32 | stored-block | block-8 | 1.36e-12 | 8.011e-05 | 8.119e-08 |
| ViT-S/16 | fp32 | stored-block | block-9 | 1.86e-12 | 0.0002441 | 2.102e-07 |
| ViT-S/16 | fp32 | stored-block | block-10 | 8.13e-13 | 0.0004883 | 2.858e-07 |
| ViT-S/16 | fp32 | stored-block | block-11 | 8.33e-13 | 0.0005493 | 3.724e-07 |
| ViT-S/16 | fp32 | stored-final-norm | iss-kibo-lab/224x224 | 8.77e-15 | 2.384e-06 | 2.603e-07 |
| ViT-S/16 | fp32 | hidden-state | state-0 | 9.46e-13 | 1.24e-05 | 1.162e-06 |
| ViT-S/16 | fp32 | hidden-state | state-1 | 3.03e-11 | 0.0003738 | 7.863e-06 |
| ViT-S/16 | fp32 | hidden-state | state-2 | 3.10e-11 | 0.000124 | 3.081e-06 |
| ViT-S/16 | fp32 | hidden-state | state-3 | 2.46e-11 | 5.436e-05 | 2.156e-07 |
| ViT-S/16 | fp32 | hidden-state | state-4 | 1.73e-11 | 4.959e-05 | 2.009e-07 |
| ViT-S/16 | fp32 | hidden-state | state-5 | 2.86e-11 | 5.341e-05 | 1.782e-07 |
| ViT-S/16 | fp32 | hidden-state | state-6 | 2.72e-11 | 6.104e-05 | 1.977e-07 |
| ViT-S/16 | fp32 | hidden-state | state-7 | 4.28e-11 | 0.0003662 | 3.783e-07 |
| ViT-S/16 | fp32 | hidden-state | state-8 | 4.12e-11 | 0.0004883 | 5.057e-07 |
| ViT-S/16 | fp32 | hidden-state | state-9 | 5.89e-11 | 0.0004272 | 4.33e-07 |
| ViT-S/16 | fp32 | hidden-state | state-10 | 6.81e-11 | 0.0006104 | 5.254e-07 |
| ViT-S/16 | fp32 | hidden-state | state-11 | 1.07e-10 | 0.0006104 | 3.573e-07 |
| ViT-S/16 | fp32 | hidden-state | state-12 | 7.46e-12 | 0.001221 | 8.276e-07 |
| ViT-S/16 | fp16-cast | stored-embeddings | iss-kibo-lab/224x224 | 9.71e-07 | 0.003759 | 0.0003524 |
| ViT-S/16 | fp16-cast | stored-block | block-0 | 3.47e-06 | 0.06728 | 0.001415 |
| ViT-S/16 | fp16-cast | stored-block | block-1 | 7.17e-07 | 0.04717 | 0.001172 |
| ViT-S/16 | fp16-cast | stored-block | block-2 | 8.12e-07 | 0.2165 | 0.0008587 |
| ViT-S/16 | fp16-cast | stored-block | block-3 | 2.57e-07 | 0.02535 | 0.0001027 |
| ViT-S/16 | fp16-cast | stored-block | block-4 | 3.46e-07 | 0.06915 | 0.0002308 |
| ViT-S/16 | fp16-cast | stored-block | block-5 | 2.96e-07 | 0.05603 | 0.0001815 |
| ViT-S/16 | fp16-cast | stored-block | block-6 | 4.24e-07 | 1.02 | 0.001054 |
| ViT-S/16 | fp16-cast | stored-block | block-7 | 4.48e-07 | 0.5342 | 0.0005533 |
| ViT-S/16 | fp16-cast | stored-block | block-8 | 7.00e-07 | 0.4134 | 0.0004189 |
| ViT-S/16 | fp16-cast | stored-block | block-9 | 4.15e-07 | 0.4418 | 0.0003803 |
| ViT-S/16 | fp16-cast | stored-block | block-10 | 6.82e-07 | 1.067 | 0.0006246 |
| ViT-S/16 | fp16-cast | stored-block | block-11 | 3.28e-07 | 0.51 | 0.0003458 |
| ViT-S/16 | fp16-cast | stored-final-norm | iss-kibo-lab/224x224 | 2.80e-07 | 0.003802 | 0.0004152 |
| ViT-S/16 | fp16-cast | hidden-state | state-0 | 9.71e-07 | 0.003759 | 0.0003524 |
| ViT-S/16 | fp16-cast | hidden-state | state-1 | 1.72e-05 | 0.2213 | 0.004655 |
| ViT-S/16 | fp16-cast | hidden-state | state-2 | 1.38e-05 | 0.05012 | 0.001246 |
| ViT-S/16 | fp16-cast | hidden-state | state-3 | 1.29e-05 | 0.2835 | 0.001125 |
| ViT-S/16 | fp16-cast | hidden-state | state-4 | 8.29e-06 | 0.2643 | 0.001071 |
| ViT-S/16 | fp16-cast | hidden-state | state-5 | 8.91e-06 | 0.3192 | 0.001065 |
| ViT-S/16 | fp16-cast | hidden-state | state-6 | 1.08e-05 | 0.306 | 0.0009914 |
| ViT-S/16 | fp16-cast | hidden-state | state-7 | 1.80e-05 | 0.2668 | 0.0002757 |
| ViT-S/16 | fp16-cast | hidden-state | state-8 | 2.10e-05 | 0.5342 | 0.0005533 |
| ViT-S/16 | fp16-cast | hidden-state | state-9 | 3.09e-05 | 0.4134 | 0.0004189 |
| ViT-S/16 | fp16-cast | hidden-state | state-10 | 4.22e-05 | 0.3366 | 0.0002898 |
| ViT-S/16 | fp16-cast | hidden-state | state-11 | 3.79e-05 | 1.067 | 0.0006246 |
| ViT-S/16 | fp16-cast | hidden-state | state-12 | 3.29e-06 | 0.9675 | 0.0006559 |
| ViT-S/16 | bf16-cast | stored-embeddings | iss-kibo-lab/224x224 | 5.78e-05 | 0.02974 | 0.002788 |
| ViT-S/16 | bf16-cast | stored-block | block-0 | 1.51e-04 | 0.8942 | 0.01881 |
| ViT-S/16 | bf16-cast | stored-block | block-1 | 5.36e-05 | 0.2542 | 0.006319 |
| ViT-S/16 | bf16-cast | stored-block | block-2 | 7.44e-05 | 1.091 | 0.00433 |
| ViT-S/16 | bf16-cast | stored-block | block-3 | 2.62e-05 | 0.3378 | 0.001368 |
| ViT-S/16 | bf16-cast | stored-block | block-4 | 1.85e-05 | 0.4942 | 0.001649 |
| ViT-S/16 | bf16-cast | stored-block | block-5 | 2.26e-05 | 1.306 | 0.004231 |
| ViT-S/16 | bf16-cast | stored-block | block-6 | 3.37e-05 | 5.33 | 0.005506 |
| ViT-S/16 | bf16-cast | stored-block | block-7 | 2.41e-05 | 2.466 | 0.002554 |
| ViT-S/16 | bf16-cast | stored-block | block-8 | 3.77e-05 | 3.628 | 0.003677 |
| ViT-S/16 | bf16-cast | stored-block | block-9 | 2.40e-05 | 3.558 | 0.003063 |
| ViT-S/16 | bf16-cast | stored-block | block-10 | 2.75e-05 | 6.933 | 0.004059 |
| ViT-S/16 | bf16-cast | stored-block | block-11 | 2.48e-05 | 4.968 | 0.003368 |
| ViT-S/16 | bf16-cast | stored-final-norm | iss-kibo-lab/224x224 | 2.44e-05 | 0.0316 | 0.00345 |
| ViT-S/16 | bf16-cast | hidden-state | state-0 | 5.78e-05 | 0.02974 | 0.002788 |
| ViT-S/16 | bf16-cast | hidden-state | state-1 | 1.53e-04 | 0.8942 | 0.01881 |
| ViT-S/16 | bf16-cast | hidden-state | state-2 | 1.40e-04 | 0.1917 | 0.004765 |
| ViT-S/16 | bf16-cast | hidden-state | state-3 | 1.93e-04 | 0.9085 | 0.003604 |
| ViT-S/16 | bf16-cast | hidden-state | state-4 | 3.28e-04 | 1.139 | 0.004615 |
| ViT-S/16 | bf16-cast | hidden-state | state-5 | 4.34e-04 | 0.5792 | 0.001933 |
| ViT-S/16 | bf16-cast | hidden-state | state-6 | 7.49e-04 | 1.306 | 0.004231 |
| ViT-S/16 | bf16-cast | hidden-state | state-7 | 2.04e-03 | 5.33 | 0.005506 |
| ViT-S/16 | bf16-cast | hidden-state | state-8 | 2.08e-03 | 2.466 | 0.002554 |
| ViT-S/16 | bf16-cast | hidden-state | state-9 | 3.19e-03 | 1.531 | 0.001552 |
| ViT-S/16 | bf16-cast | hidden-state | state-10 | 2.73e-03 | 1.713 | 0.001474 |
| ViT-S/16 | bf16-cast | hidden-state | state-11 | 2.11e-03 | 6.933 | 0.004059 |
| ViT-S/16 | bf16-cast | hidden-state | state-12 | 1.81e-04 | 9.321 | 0.006319 |
| ViT-S/16 | fp16-autocast | stored-embeddings | iss-kibo-lab/224x224 | 9.71e-07 | 0.003759 | 0.0003524 |
| ViT-S/16 | fp16-autocast | stored-block | block-0 | 5.86e-05 | 0.09877 | 0.002078 |
| ViT-S/16 | fp16-autocast | stored-block | block-1 | 2.03e-07 | 0.009245 | 0.0002298 |
| ViT-S/16 | fp16-autocast | stored-block | block-2 | 5.70e-07 | 0.02339 | 9.279e-05 |
| ViT-S/16 | fp16-autocast | stored-block | block-3 | 2.07e-07 | 0.008789 | 3.56e-05 |
| ViT-S/16 | fp16-autocast | stored-block | block-4 | 1.50e-07 | 0.00764 | 2.549e-05 |
| ViT-S/16 | fp16-autocast | stored-block | block-5 | 1.06e-07 | 0.0134 | 4.34e-05 |
| ViT-S/16 | fp16-autocast | stored-block | block-6 | 1.96e-07 | 0.2573 | 0.0002658 |
| ViT-S/16 | fp16-autocast | stored-block | block-7 | 2.32e-07 | 0.158 | 0.0001636 |
| ViT-S/16 | fp16-autocast | stored-block | block-8 | 2.05e-07 | 0.02356 | 2.388e-05 |
| ViT-S/16 | fp16-autocast | stored-block | block-9 | 4.45e-07 | 0.08777 | 7.555e-05 |
| ViT-S/16 | fp16-autocast | stored-block | block-10 | 3.08e-07 | 0.06213 | 3.637e-05 |
| ViT-S/16 | fp16-autocast | stored-block | block-11 | 1.55e-07 | 0.2227 | 0.000151 |
| ViT-S/16 | fp16-autocast | stored-final-norm | iss-kibo-lab/224x224 | 8.77e-15 | 2.384e-06 | 2.603e-07 |
| ViT-S/16 | fp16-autocast | hidden-state | state-0 | 9.71e-07 | 0.003759 | 0.0003524 |
| ViT-S/16 | fp16-autocast | hidden-state | state-1 | 1.03e-06 | 0.0426 | 0.0008961 |
| ViT-S/16 | fp16-autocast | hidden-state | state-2 | 1.70e-06 | 0.03068 | 0.0007626 |
| ViT-S/16 | fp16-autocast | hidden-state | state-3 | 2.05e-06 | 0.0278 | 0.0001103 |
| ViT-S/16 | fp16-autocast | hidden-state | state-4 | 3.13e-06 | 0.02464 | 9.983e-05 |
| ViT-S/16 | fp16-autocast | hidden-state | state-5 | 6.62e-06 | 0.02393 | 7.984e-05 |
| ViT-S/16 | fp16-autocast | hidden-state | state-6 | 8.48e-06 | 0.04671 | 0.0001513 |
| ViT-S/16 | fp16-autocast | hidden-state | state-7 | 1.26e-05 | 0.2565 | 0.0002649 |
| ViT-S/16 | fp16-autocast | hidden-state | state-8 | 1.30e-05 | 0.2576 | 0.0002668 |
| ViT-S/16 | fp16-autocast | hidden-state | state-9 | 2.43e-05 | 0.2576 | 0.000261 |
| ViT-S/16 | fp16-autocast | hidden-state | state-10 | 3.74e-05 | 0.2573 | 0.0002215 |
| ViT-S/16 | fp16-autocast | hidden-state | state-11 | 5.80e-05 | 0.3677 | 0.0002152 |
| ViT-S/16 | fp16-autocast | hidden-state | state-12 | 3.00e-06 | 0.2891 | 0.000196 |
| ViT-S/16 | bf16-autocast | stored-embeddings | iss-kibo-lab/224x224 | 5.78e-05 | 0.02974 | 0.002788 |
| ViT-S/16 | bf16-autocast | stored-block | block-0 | 1.38e-04 | 0.7297 | 0.01535 |
| ViT-S/16 | bf16-autocast | stored-block | block-1 | 1.37e-05 | 0.1039 | 0.002582 |
| ViT-S/16 | bf16-autocast | stored-block | block-2 | 2.21e-05 | 0.6061 | 0.002404 |
| ViT-S/16 | bf16-autocast | stored-block | block-3 | 1.83e-05 | 0.06189 | 0.0002507 |
| ViT-S/16 | bf16-autocast | stored-block | block-4 | 1.15e-05 | 0.1021 | 0.0003406 |
| ViT-S/16 | bf16-autocast | stored-block | block-5 | 5.94e-06 | 0.1464 | 0.0004743 |
| ViT-S/16 | bf16-autocast | stored-block | block-6 | 1.30e-05 | 3.593 | 0.003712 |
| ViT-S/16 | bf16-autocast | stored-block | block-7 | 2.37e-05 | 0.5419 | 0.0005612 |
| ViT-S/16 | bf16-autocast | stored-block | block-8 | 1.39e-05 | 0.4405 | 0.0004464 |
| ViT-S/16 | bf16-autocast | stored-block | block-9 | 1.35e-05 | 0.2089 | 0.0001799 |
| ViT-S/16 | bf16-autocast | stored-block | block-10 | 1.46e-05 | 2.527 | 0.001479 |
| ViT-S/16 | bf16-autocast | stored-block | block-11 | 6.94e-06 | 1.628 | 0.001103 |
| ViT-S/16 | bf16-autocast | stored-final-norm | iss-kibo-lab/224x224 | 8.77e-15 | 2.384e-06 | 2.603e-07 |
| ViT-S/16 | bf16-autocast | hidden-state | state-0 | 5.78e-05 | 0.02974 | 0.002788 |
| ViT-S/16 | bf16-autocast | hidden-state | state-1 | 1.38e-04 | 0.7372 | 0.01551 |
| ViT-S/16 | bf16-autocast | hidden-state | state-2 | 1.71e-04 | 0.2151 | 0.005346 |
| ViT-S/16 | bf16-autocast | hidden-state | state-3 | 1.77e-04 | 0.7032 | 0.00279 |
| ViT-S/16 | bf16-autocast | hidden-state | state-4 | 3.60e-04 | 0.7002 | 0.002836 |
| ViT-S/16 | bf16-autocast | hidden-state | state-5 | 5.74e-04 | 0.8023 | 0.002677 |
| ViT-S/16 | bf16-autocast | hidden-state | state-6 | 8.03e-04 | 0.5768 | 0.001868 |
| ViT-S/16 | bf16-autocast | hidden-state | state-7 | 1.96e-03 | 3.577 | 0.003696 |
| ViT-S/16 | bf16-autocast | hidden-state | state-8 | 2.08e-03 | 3.552 | 0.003679 |
| ViT-S/16 | bf16-autocast | hidden-state | state-9 | 3.15e-03 | 3.574 | 0.003622 |
| ViT-S/16 | bf16-autocast | hidden-state | state-10 | 2.44e-03 | 3.552 | 0.003058 |
| ViT-S/16 | bf16-autocast | hidden-state | state-11 | 1.84e-03 | 3.579 | 0.002095 |
| ViT-S/16 | bf16-autocast | hidden-state | state-12 | 1.67e-04 | 3.717 | 0.00252 |
| ViT-S+/16 | fp32 | stored-embeddings | iss-kibo-lab/224x224 | 1.41e-12 | 5.484e-06 | 7.94e-07 |
| ViT-S+/16 | fp32 | stored-block | block-0 | 1.26e-09 | 0.0002778 | 1.573e-05 |
| ViT-S+/16 | fp32 | stored-block | block-1 | 5.50e-13 | 2.67e-05 | 5.759e-07 |
| ViT-S+/16 | fp32 | stored-block | block-2 | 6.24e-13 | 0.002441 | 9.88e-07 |
| ViT-S+/16 | fp32 | stored-block | block-3 | 5.27e-13 | 1.526e-05 | 6.201e-09 |
| ViT-S+/16 | fp32 | stored-block | block-4 | 6.93e-13 | 6.104e-05 | 2.485e-08 |
| ViT-S+/16 | fp32 | stored-block | block-5 | 6.78e-13 | 0.008789 | 1.831e-06 |
| ViT-S+/16 | fp32 | stored-block | block-6 | 6.54e-13 | 4.673e-05 | 9.752e-09 |
| ViT-S+/16 | fp32 | stored-block | block-7 | 5.26e-13 | 6.104e-05 | 1.278e-08 |
| ViT-S+/16 | fp32 | stored-block | block-8 | 1.12e-12 | 0.0001259 | 2.659e-08 |
| ViT-S+/16 | fp32 | stored-block | block-9 | 1.76e-12 | 0.0002556 | 5.407e-08 |
| ViT-S+/16 | fp32 | stored-block | block-10 | 7.89e-13 | 0.0009766 | 1.856e-07 |
| ViT-S+/16 | fp32 | stored-block | block-11 | 3.83e-13 | 0.001099 | 2.538e-07 |
| ViT-S+/16 | fp32 | stored-final-norm | iss-kibo-lab/224x224 | 1.37e-14 | 1.431e-06 | 1.311e-07 |
| ViT-S+/16 | fp32 | hidden-state | state-0 | 1.41e-12 | 5.484e-06 | 7.94e-07 |
| ViT-S+/16 | fp32 | hidden-state | state-1 | 8.93e-10 | 0.0002575 | 1.459e-05 |
| ViT-S+/16 | fp32 | hidden-state | state-2 | 8.27e-11 | 8.154e-05 | 1.759e-06 |
| ViT-S+/16 | fp32 | hidden-state | state-3 | 7.81e-11 | 0.004395 | 1.778e-06 |
| ViT-S+/16 | fp32 | hidden-state | state-4 | 1.03e-10 | 0.004395 | 1.786e-06 |
| ViT-S+/16 | fp32 | hidden-state | state-5 | 1.38e-10 | 0.004395 | 1.789e-06 |
| ViT-S+/16 | fp32 | hidden-state | state-6 | 1.18e-10 | 0.007324 | 1.526e-06 |
| ViT-S+/16 | fp32 | hidden-state | state-7 | 9.73e-11 | 0.007324 | 1.528e-06 |
| ViT-S+/16 | fp32 | hidden-state | state-8 | 1.03e-10 | 0.007324 | 1.534e-06 |
| ViT-S+/16 | fp32 | hidden-state | state-9 | 1.09e-10 | 0.007324 | 1.547e-06 |
| ViT-S+/16 | fp32 | hidden-state | state-10 | 1.43e-10 | 0.007324 | 1.55e-06 |
| ViT-S+/16 | fp32 | hidden-state | state-11 | 1.28e-10 | 0.006836 | 1.299e-06 |
| ViT-S+/16 | fp32 | hidden-state | state-12 | 8.61e-12 | 0.007324 | 1.692e-06 |
| ViT-S+/16 | fp16-cast | stored-embeddings | iss-kibo-lab/224x224 | 1.78e-06 | 0.002796 | 0.0004048 |
| ViT-S+/16 | fp16-cast | stored-block | block-0 | 3.32e-04 | 0.1995 | 0.0113 |
| ViT-S+/16 | fp16-cast | stored-block | block-1 | 7.62e-07 | 0.01945 | 0.0004195 |
| ViT-S+/16 | fp16-cast | stored-block | block-2 | 4.09e-07 | 0.8845 | 0.0003579 |
| ViT-S+/16 | fp16-cast | stored-block | block-3 | 5.58e-07 | 1.453 | 0.0005906 |
| ViT-S+/16 | fp16-cast | stored-block | block-4 | 4.47e-07 | 0.5869 | 0.0002389 |
| ViT-S+/16 | fp16-cast | stored-block | block-5 | 7.50e-07 | 3.98 | 0.0008293 |
| ViT-S+/16 | fp16-cast | stored-block | block-6 | 5.95e-07 | 0.5344 | 0.0001115 |
| ViT-S+/16 | fp16-cast | stored-block | block-7 | 6.79e-07 | 1.431 | 0.0002996 |
| ViT-S+/16 | fp16-cast | stored-block | block-8 | 4.56e-07 | 2.924 | 0.0006175 |
| ViT-S+/16 | fp16-cast | stored-block | block-9 | 7.11e-07 | 1.813 | 0.0003835 |
| ViT-S+/16 | fp16-cast | stored-block | block-10 | 1.53e-06 | 3.846 | 0.0007308 |
| ViT-S+/16 | fp16-cast | stored-block | block-11 | 3.97e-07 | 3.24 | 0.0007485 |
| ViT-S+/16 | fp16-cast | stored-final-norm | iss-kibo-lab/224x224 | 2.44e-07 | 0.004972 | 0.0004557 |
| ViT-S+/16 | fp16-cast | hidden-state | state-0 | 1.78e-06 | 0.002796 | 0.0004048 |
| ViT-S+/16 | fp16-cast | hidden-state | state-1 | 3.88e-04 | 0.223 | 0.01263 |
| ViT-S+/16 | fp16-cast | hidden-state | state-2 | 5.25e-05 | 0.04642 | 0.001001 |
| ViT-S+/16 | fp16-cast | hidden-state | state-3 | 5.35e-05 | 0.8845 | 0.0003579 |
| ViT-S+/16 | fp16-cast | hidden-state | state-4 | 6.70e-05 | 1.453 | 0.0005906 |
| ViT-S+/16 | fp16-cast | hidden-state | state-5 | 8.41e-05 | 1.413 | 0.0005752 |
| ViT-S+/16 | fp16-cast | hidden-state | state-6 | 7.33e-05 | 2.521 | 0.0005252 |
| ViT-S+/16 | fp16-cast | hidden-state | state-7 | 6.08e-05 | 2.534 | 0.0005289 |
| ViT-S+/16 | fp16-cast | hidden-state | state-8 | 6.73e-05 | 2.673 | 0.0005597 |
| ViT-S+/16 | fp16-cast | hidden-state | state-9 | 5.74e-05 | 2.924 | 0.0006175 |
| ViT-S+/16 | fp16-cast | hidden-state | state-10 | 6.15e-05 | 2.815 | 0.0005956 |
| ViT-S+/16 | fp16-cast | hidden-state | state-11 | 6.84e-05 | 2.672 | 0.0005078 |
| ViT-S+/16 | fp16-cast | hidden-state | state-12 | 4.71e-06 | 4.672 | 0.001079 |
| ViT-S+/16 | bf16-cast | stored-embeddings | iss-kibo-lab/224x224 | 1.27e-04 | 0.02068 | 0.002994 |
| ViT-S+/16 | bf16-cast | stored-block | block-0 | 2.93e-02 | 1.431 | 0.08107 |
| ViT-S+/16 | bf16-cast | stored-block | block-1 | 4.49e-05 | 0.207 | 0.004463 |
| ViT-S+/16 | bf16-cast | stored-block | block-2 | 3.15e-05 | 24.88 | 0.01007 |
| ViT-S+/16 | bf16-cast | stored-block | block-3 | 2.54e-05 | 12.55 | 0.005099 |
| ViT-S+/16 | bf16-cast | stored-block | block-4 | 2.81e-05 | 7.413 | 0.003018 |
| ViT-S+/16 | bf16-cast | stored-block | block-5 | 6.84e-05 | 32.02 | 0.006671 |
| ViT-S+/16 | bf16-cast | stored-block | block-6 | 3.71e-05 | 9.776 | 0.00204 |
| ViT-S+/16 | bf16-cast | stored-block | block-7 | 3.34e-05 | 24 | 0.005025 |
| ViT-S+/16 | bf16-cast | stored-block | block-8 | 2.92e-05 | 10.14 | 0.002141 |
| ViT-S+/16 | bf16-cast | stored-block | block-9 | 4.38e-05 | 9.185 | 0.001943 |
| ViT-S+/16 | bf16-cast | stored-block | block-10 | 3.52e-05 | 17.94 | 0.003408 |
| ViT-S+/16 | bf16-cast | stored-block | block-11 | 2.67e-05 | 40.76 | 0.009416 |
| ViT-S+/16 | bf16-cast | stored-final-norm | iss-kibo-lab/224x224 | 1.47e-05 | 0.04013 | 0.003678 |
| ViT-S+/16 | bf16-cast | hidden-state | state-0 | 1.27e-04 | 0.02068 | 0.002994 |
| ViT-S+/16 | bf16-cast | hidden-state | state-1 | 2.98e-02 | 1.441 | 0.08162 |
| ViT-S+/16 | bf16-cast | hidden-state | state-2 | 3.36e-03 | 0.4934 | 0.01064 |
| ViT-S+/16 | bf16-cast | hidden-state | state-3 | 2.04e-03 | 24.88 | 0.01007 |
| ViT-S+/16 | bf16-cast | hidden-state | state-4 | 3.81e-03 | 19.45 | 0.007906 |
| ViT-S+/16 | bf16-cast | hidden-state | state-5 | 5.82e-03 | 23.41 | 0.009531 |
| ViT-S+/16 | bf16-cast | hidden-state | state-6 | 7.68e-03 | 32.02 | 0.006671 |
| ViT-S+/16 | bf16-cast | hidden-state | state-7 | 9.48e-03 | 24.53 | 0.00512 |
| ViT-S+/16 | bf16-cast | hidden-state | state-8 | 1.43e-02 | 24.67 | 0.005166 |
| ViT-S+/16 | bf16-cast | hidden-state | state-9 | 1.30e-02 | 25.83 | 0.005456 |
| ViT-S+/16 | bf16-cast | hidden-state | state-10 | 1.00e-02 | 32.76 | 0.00693 |
| ViT-S+/16 | bf16-cast | hidden-state | state-11 | 6.20e-03 | 35.56 | 0.006757 |
| ViT-S+/16 | bf16-cast | hidden-state | state-12 | 7.37e-04 | 40.76 | 0.009416 |
| ViT-S+/16 | fp16-autocast | stored-embeddings | iss-kibo-lab/224x224 | 1.78e-06 | 0.002796 | 0.0004048 |
| ViT-S+/16 | fp16-autocast | stored-block | block-0 | 3.57e-04 | 0.1765 | 0.009999 |
| ViT-S+/16 | fp16-autocast | stored-block | block-1 | 1.92e-07 | 0.01313 | 0.0002833 |
| ViT-S+/16 | fp16-autocast | stored-block | block-2 | 1.98e-07 | 0.3198 | 0.0001294 |
| ViT-S+/16 | fp16-autocast | stored-block | block-3 | 2.16e-07 | 0.02313 | 9.401e-06 |
| ViT-S+/16 | fp16-autocast | stored-block | block-4 | 3.60e-07 | 0.03282 | 1.336e-05 |
| ViT-S+/16 | fp16-autocast | stored-block | block-5 | 5.37e-07 | 2.632 | 0.0005483 |
| ViT-S+/16 | fp16-autocast | stored-block | block-6 | 1.64e-07 | 0.01061 | 2.213e-06 |
| ViT-S+/16 | fp16-autocast | stored-block | block-7 | 2.86e-07 | 0.02647 | 5.542e-06 |
| ViT-S+/16 | fp16-autocast | stored-block | block-8 | 3.08e-07 | 0.03192 | 6.742e-06 |
| ViT-S+/16 | fp16-autocast | stored-block | block-9 | 3.63e-07 | 0.1545 | 3.269e-05 |
| ViT-S+/16 | fp16-autocast | stored-block | block-10 | 3.03e-07 | 1.945 | 0.0003697 |
| ViT-S+/16 | fp16-autocast | stored-block | block-11 | 1.28e-07 | 0.6926 | 0.00016 |
| ViT-S+/16 | fp16-autocast | stored-final-norm | iss-kibo-lab/224x224 | 1.37e-14 | 1.431e-06 | 1.311e-07 |
| ViT-S+/16 | fp16-autocast | hidden-state | state-0 | 1.78e-06 | 0.002796 | 0.0004048 |
| ViT-S+/16 | fp16-autocast | hidden-state | state-1 | 3.38e-04 | 0.1755 | 0.009939 |
| ViT-S+/16 | fp16-autocast | hidden-state | state-2 | 3.86e-05 | 0.04398 | 0.0009485 |
| ViT-S+/16 | fp16-autocast | hidden-state | state-3 | 3.78e-05 | 0.3162 | 0.0001279 |
| ViT-S+/16 | fp16-autocast | hidden-state | state-4 | 4.93e-05 | 0.3113 | 0.0001265 |
| ViT-S+/16 | fp16-autocast | hidden-state | state-5 | 6.40e-05 | 0.3154 | 0.0001284 |
| ViT-S+/16 | fp16-autocast | hidden-state | state-6 | 5.71e-05 | 2.625 | 0.000547 |
| ViT-S+/16 | fp16-autocast | hidden-state | state-7 | 4.46e-05 | 2.616 | 0.0005459 |
| ViT-S+/16 | fp16-autocast | hidden-state | state-8 | 5.04e-05 | 2.626 | 0.0005498 |
| ViT-S+/16 | fp16-autocast | hidden-state | state-9 | 4.72e-05 | 2.635 | 0.0005565 |
| ViT-S+/16 | fp16-autocast | hidden-state | state-10 | 5.22e-05 | 2.57 | 0.0005438 |
| ViT-S+/16 | fp16-autocast | hidden-state | state-11 | 5.66e-05 | 1.995 | 0.0003792 |
| ViT-S+/16 | fp16-autocast | hidden-state | state-12 | 4.46e-06 | 1.981 | 0.0004577 |
| ViT-S+/16 | bf16-autocast | stored-embeddings | iss-kibo-lab/224x224 | 1.27e-04 | 0.02068 | 0.002994 |
| ViT-S+/16 | bf16-autocast | stored-block | block-0 | 2.90e-02 | 1.422 | 0.08053 |
| ViT-S+/16 | bf16-autocast | stored-block | block-1 | 1.35e-05 | 0.2034 | 0.004386 |
| ViT-S+/16 | bf16-autocast | stored-block | block-2 | 1.82e-05 | 17.04 | 0.006897 |
| ViT-S+/16 | bf16-autocast | stored-block | block-3 | 1.58e-05 | 0.2743 | 0.0001115 |
| ViT-S+/16 | bf16-autocast | stored-block | block-4 | 1.29e-05 | 0.302 | 0.0001229 |
| ViT-S+/16 | bf16-autocast | stored-block | block-5 | 4.61e-05 | 7.057 | 0.00147 |
| ViT-S+/16 | bf16-autocast | stored-block | block-6 | 1.52e-05 | 0.1453 | 3.031e-05 |
| ViT-S+/16 | bf16-autocast | stored-block | block-7 | 1.30e-05 | 0.1368 | 2.864e-05 |
| ViT-S+/16 | bf16-autocast | stored-block | block-8 | 1.40e-05 | 0.2405 | 5.08e-05 |
| ViT-S+/16 | bf16-autocast | stored-block | block-9 | 1.85e-05 | 0.9743 | 0.0002061 |
| ViT-S+/16 | bf16-autocast | stored-block | block-10 | 2.17e-05 | 16.05 | 0.00305 |
| ViT-S+/16 | bf16-autocast | stored-block | block-11 | 8.85e-06 | 6.124 | 0.001415 |
| ViT-S+/16 | bf16-autocast | stored-final-norm | iss-kibo-lab/224x224 | 1.37e-14 | 1.431e-06 | 1.311e-07 |
| ViT-S+/16 | bf16-autocast | hidden-state | state-0 | 1.27e-04 | 0.02068 | 0.002994 |
| ViT-S+/16 | bf16-autocast | hidden-state | state-1 | 2.96e-02 | 1.439 | 0.08149 |
| ViT-S+/16 | bf16-autocast | hidden-state | state-2 | 3.03e-03 | 0.4694 | 0.01012 |
| ViT-S+/16 | bf16-autocast | hidden-state | state-3 | 1.90e-03 | 16.99 | 0.006874 |
| ViT-S+/16 | bf16-autocast | hidden-state | state-4 | 3.56e-03 | 16.96 | 0.006892 |
| ViT-S+/16 | bf16-autocast | hidden-state | state-5 | 5.44e-03 | 16.93 | 0.006892 |
| ViT-S+/16 | bf16-autocast | hidden-state | state-6 | 7.15e-03 | 16.92 | 0.003526 |
| ViT-S+/16 | bf16-autocast | hidden-state | state-7 | 8.61e-03 | 16.89 | 0.003525 |
| ViT-S+/16 | bf16-autocast | hidden-state | state-8 | 1.36e-02 | 16.91 | 0.003541 |
| ViT-S+/16 | bf16-autocast | hidden-state | state-9 | 1.24e-02 | 16.92 | 0.003573 |
| ViT-S+/16 | bf16-autocast | hidden-state | state-10 | 9.60e-03 | 17.13 | 0.003624 |
| ViT-S+/16 | bf16-autocast | hidden-state | state-11 | 5.64e-03 | 22.24 | 0.004226 |
| ViT-S+/16 | bf16-autocast | hidden-state | state-12 | 7.04e-04 | 26.19 | 0.006051 |
| ViT-B/16 | fp32 | stored-embeddings | iss-kibo-lab/224x224 | 1.20e-12 | 1.049e-05 | 1.394e-06 |
| ViT-B/16 | fp32 | stored-block | block-0 | 2.76e-11 | 8.738e-05 | 2.926e-06 |
| ViT-B/16 | fp32 | stored-block | block-1 | 2.46e-12 | 7.629e-05 | 1.677e-06 |
| ViT-B/16 | fp32 | stored-block | block-2 | 4.00e-12 | 0.004883 | 5.083e-07 |
| ViT-B/16 | fp32 | stored-block | block-3 | 7.65e-12 | 9.155e-05 | 9.547e-09 |
| ViT-B/16 | fp32 | stored-block | block-4 | 2.52e-11 | 0.0001926 | 2.009e-08 |
| ViT-B/16 | fp32 | stored-block | block-5 | 8.88e-12 | 0.0004883 | 5.093e-08 |
| ViT-B/16 | fp32 | stored-block | block-6 | 1.34e-11 | 0.0002441 | 2.547e-08 |
| ViT-B/16 | fp32 | stored-block | block-7 | 6.50e-12 | 0.001099 | 1.146e-07 |
| ViT-B/16 | fp32 | stored-block | block-8 | 4.86e-12 | 0.001099 | 1.146e-07 |
| ViT-B/16 | fp32 | stored-block | block-9 | 6.28e-12 | 0.001953 | 2.036e-07 |
| ViT-B/16 | fp32 | stored-block | block-10 | 9.53e-12 | 0.002686 | 1.799e-07 |
| ViT-B/16 | fp32 | stored-block | block-11 | 7.71e-13 | 0.001709 | 1.163e-07 |
| ViT-B/16 | fp32 | stored-final-norm | iss-kibo-lab/224x224 | 1.31e-14 | 2.384e-06 | 1.337e-07 |
| ViT-B/16 | fp32 | hidden-state | state-0 | 1.20e-12 | 1.049e-05 | 1.394e-06 |
| ViT-B/16 | fp32 | hidden-state | state-1 | 2.03e-11 | 8.893e-05 | 2.977e-06 |
| ViT-B/16 | fp32 | hidden-state | state-2 | 3.79e-11 | 8.392e-05 | 1.845e-06 |
| ViT-B/16 | fp32 | hidden-state | state-3 | 3.34e-11 | 0.02441 | 2.542e-06 |
| ViT-B/16 | fp32 | hidden-state | state-4 | 3.78e-11 | 0.02441 | 2.546e-06 |
| ViT-B/16 | fp32 | hidden-state | state-5 | 6.17e-11 | 0.02441 | 2.546e-06 |
| ViT-B/16 | fp32 | hidden-state | state-6 | 7.99e-11 | 0.02441 | 2.547e-06 |
| ViT-B/16 | fp32 | hidden-state | state-7 | 1.06e-10 | 0.02441 | 2.547e-06 |
| ViT-B/16 | fp32 | hidden-state | state-8 | 1.13e-10 | 0.02441 | 2.546e-06 |
| ViT-B/16 | fp32 | hidden-state | state-9 | 9.00e-11 | 0.02441 | 2.546e-06 |
| ViT-B/16 | fp32 | hidden-state | state-10 | 1.04e-10 | 0.02441 | 2.545e-06 |
| ViT-B/16 | fp32 | hidden-state | state-11 | 1.35e-10 | 0.02441 | 1.636e-06 |
| ViT-B/16 | fp32 | hidden-state | state-12 | 2.33e-11 | 0.02441 | 1.661e-06 |
| ViT-B/16 | fp16-cast | stored-embeddings | iss-kibo-lab/224x224 | 7.29e-07 | 0.003532 | 0.0004693 |
| ViT-B/16 | fp16-cast | stored-block | block-0 | 1.22e-05 | 0.06498 | 0.002176 |
| ViT-B/16 | fp16-cast | stored-block | block-1 | 5.10e-07 | 0.02549 | 0.0005603 |
| ViT-B/16 | fp16-cast | stored-block | block-2 | 6.88e-07 | 2.116 | 0.0002203 |
| ViT-B/16 | fp16-cast | stored-block | block-3 | 5.41e-07 | 2.03 | 0.0002117 |
| ViT-B/16 | fp16-cast | stored-block | block-4 | 1.57e-06 | 4.219 | 0.00044 |
| ViT-B/16 | fp16-cast | stored-block | block-5 | 1.97e-06 | 3.029 | 0.000316 |
| ViT-B/16 | fp16-cast | stored-block | block-6 | 1.11e-06 | 3.027 | 0.0003158 |
| ViT-B/16 | fp16-cast | stored-block | block-7 | 9.56e-07 | 3.776 | 0.0003939 |
| ViT-B/16 | fp16-cast | stored-block | block-8 | 6.63e-07 | 4.731 | 0.0004934 |
| ViT-B/16 | fp16-cast | stored-block | block-9 | 9.83e-07 | 4.317 | 0.0004501 |
| ViT-B/16 | fp16-cast | stored-block | block-10 | 1.23e-06 | 2.286 | 0.0001532 |
| ViT-B/16 | fp16-cast | stored-block | block-11 | 4.16e-07 | 6.56 | 0.0004463 |
| ViT-B/16 | fp16-cast | stored-final-norm | iss-kibo-lab/224x224 | 2.71e-07 | 0.008039 | 0.0004507 |
| ViT-B/16 | fp16-cast | hidden-state | state-0 | 7.29e-07 | 0.003532 | 0.0004693 |
| ViT-B/16 | fp16-cast | hidden-state | state-1 | 1.23e-05 | 0.06498 | 0.002176 |
| ViT-B/16 | fp16-cast | hidden-state | state-2 | 2.13e-05 | 0.05898 | 0.001297 |
| ViT-B/16 | fp16-cast | hidden-state | state-3 | 1.29e-05 | 2.116 | 0.0002203 |
| ViT-B/16 | fp16-cast | hidden-state | state-4 | 1.32e-05 | 2.144 | 0.0002236 |
| ViT-B/16 | fp16-cast | hidden-state | state-5 | 2.05e-05 | 4.219 | 0.00044 |
| ViT-B/16 | fp16-cast | hidden-state | state-6 | 2.32e-05 | 4.971 | 0.0005185 |
| ViT-B/16 | fp16-cast | hidden-state | state-7 | 2.43e-05 | 4.973 | 0.0005187 |
| ViT-B/16 | fp16-cast | hidden-state | state-8 | 2.29e-05 | 4.282 | 0.0004466 |
| ViT-B/16 | fp16-cast | hidden-state | state-9 | 2.39e-05 | 5.743 | 0.0005989 |
| ViT-B/16 | fp16-cast | hidden-state | state-10 | 2.69e-05 | 8.317 | 0.0008671 |
| ViT-B/16 | fp16-cast | hidden-state | state-11 | 3.12e-05 | 7.358 | 0.000493 |
| ViT-B/16 | fp16-cast | hidden-state | state-12 | 3.95e-06 | 8.87 | 0.0006035 |
| ViT-B/16 | bf16-cast | stored-embeddings | iss-kibo-lab/224x224 | 4.87e-05 | 0.02921 | 0.003882 |
| ViT-B/16 | bf16-cast | stored-block | block-0 | 1.22e-04 | 0.2573 | 0.008613 |
| ViT-B/16 | bf16-cast | stored-block | block-1 | 3.42e-05 | 0.1426 | 0.003135 |
| ViT-B/16 | bf16-cast | stored-block | block-2 | 3.79e-05 | 6.11 | 0.0006361 |
| ViT-B/16 | bf16-cast | stored-block | block-3 | 3.81e-05 | 10.03 | 0.001046 |
| ViT-B/16 | bf16-cast | stored-block | block-4 | 7.82e-05 | 20.94 | 0.002184 |
| ViT-B/16 | bf16-cast | stored-block | block-5 | 7.12e-05 | 12.97 | 0.001353 |
| ViT-B/16 | bf16-cast | stored-block | block-6 | 1.01e-04 | 13.51 | 0.00141 |
| ViT-B/16 | bf16-cast | stored-block | block-7 | 4.50e-05 | 12.28 | 0.001281 |
| ViT-B/16 | bf16-cast | stored-block | block-8 | 6.00e-05 | 21.74 | 0.002268 |
| ViT-B/16 | bf16-cast | stored-block | block-9 | 4.78e-05 | 23.68 | 0.002469 |
| ViT-B/16 | bf16-cast | stored-block | block-10 | 4.99e-05 | 78 | 0.005226 |
| ViT-B/16 | bf16-cast | stored-block | block-11 | 1.72e-05 | 41.44 | 0.00282 |
| ViT-B/16 | bf16-cast | stored-final-norm | iss-kibo-lab/224x224 | 2.41e-05 | 0.06713 | 0.003764 |
| ViT-B/16 | bf16-cast | hidden-state | state-0 | 4.87e-05 | 0.02921 | 0.003882 |
| ViT-B/16 | bf16-cast | hidden-state | state-1 | 1.55e-04 | 0.2626 | 0.008792 |
| ViT-B/16 | bf16-cast | hidden-state | state-2 | 3.31e-04 | 0.1899 | 0.004174 |
| ViT-B/16 | bf16-cast | hidden-state | state-3 | 6.09e-04 | 18.2 | 0.001895 |
| ViT-B/16 | bf16-cast | hidden-state | state-4 | 1.10e-03 | 17.86 | 0.001862 |
| ViT-B/16 | bf16-cast | hidden-state | state-5 | 1.61e-03 | 20.94 | 0.002184 |
| ViT-B/16 | bf16-cast | hidden-state | state-6 | 2.83e-03 | 26.92 | 0.002808 |
| ViT-B/16 | bf16-cast | hidden-state | state-7 | 2.88e-03 | 19.87 | 0.002072 |
| ViT-B/16 | bf16-cast | hidden-state | state-8 | 3.22e-03 | 24.22 | 0.002527 |
| ViT-B/16 | bf16-cast | hidden-state | state-9 | 3.65e-03 | 30.07 | 0.003136 |
| ViT-B/16 | bf16-cast | hidden-state | state-10 | 3.66e-03 | 26.55 | 0.002768 |
| ViT-B/16 | bf16-cast | hidden-state | state-11 | 2.16e-03 | 50 | 0.00335 |
| ViT-B/16 | bf16-cast | hidden-state | state-12 | 1.72e-04 | 55.35 | 0.003766 |
| ViT-B/16 | fp16-autocast | stored-embeddings | iss-kibo-lab/224x224 | 7.29e-07 | 0.003532 | 0.0004693 |
| ViT-B/16 | fp16-autocast | stored-block | block-0 | 6.13e-07 | 0.01536 | 0.0005142 |
| ViT-B/16 | fp16-autocast | stored-block | block-1 | 2.14e-07 | 0.008228 | 0.0001809 |
| ViT-B/16 | fp16-autocast | stored-block | block-2 | 1.64e-07 | 2.497 | 0.0002599 |
| ViT-B/16 | fp16-autocast | stored-block | block-3 | 2.72e-07 | 0.01416 | 1.476e-06 |
| ViT-B/16 | fp16-autocast | stored-block | block-4 | 6.02e-07 | 0.03857 | 4.023e-06 |
| ViT-B/16 | fp16-autocast | stored-block | block-5 | 1.00e-06 | 0.1458 | 1.521e-05 |
| ViT-B/16 | fp16-autocast | stored-block | block-6 | 6.97e-07 | 0.162 | 1.69e-05 |
| ViT-B/16 | fp16-autocast | stored-block | block-7 | 3.20e-07 | 0.3298 | 3.44e-05 |
| ViT-B/16 | fp16-autocast | stored-block | block-8 | 3.19e-07 | 0.8451 | 8.813e-05 |
| ViT-B/16 | fp16-autocast | stored-block | block-9 | 3.04e-07 | 0.6173 | 6.436e-05 |
| ViT-B/16 | fp16-autocast | stored-block | block-10 | 3.32e-07 | 2.323 | 0.0001557 |
| ViT-B/16 | fp16-autocast | stored-block | block-11 | 1.28e-07 | 0.4803 | 3.268e-05 |
| ViT-B/16 | fp16-autocast | stored-final-norm | iss-kibo-lab/224x224 | 1.31e-14 | 2.384e-06 | 1.337e-07 |
| ViT-B/16 | fp16-autocast | hidden-state | state-0 | 7.29e-07 | 0.003532 | 0.0004693 |
| ViT-B/16 | fp16-autocast | hidden-state | state-1 | 6.52e-07 | 0.01536 | 0.0005142 |
| ViT-B/16 | fp16-autocast | hidden-state | state-2 | 2.35e-06 | 0.01156 | 0.0002541 |
| ViT-B/16 | fp16-autocast | hidden-state | state-3 | 7.24e-06 | 4.358 | 0.0004537 |
| ViT-B/16 | fp16-autocast | hidden-state | state-4 | 1.13e-05 | 4.368 | 0.0004555 |
| ViT-B/16 | fp16-autocast | hidden-state | state-5 | 1.81e-05 | 4.367 | 0.0004555 |
| ViT-B/16 | fp16-autocast | hidden-state | state-6 | 1.75e-05 | 4.368 | 0.0004556 |
| ViT-B/16 | fp16-autocast | hidden-state | state-7 | 1.88e-05 | 4.368 | 0.0004556 |
| ViT-B/16 | fp16-autocast | hidden-state | state-8 | 1.67e-05 | 4.368 | 0.0004556 |
| ViT-B/16 | fp16-autocast | hidden-state | state-9 | 1.78e-05 | 4.369 | 0.0004557 |
| ViT-B/16 | fp16-autocast | hidden-state | state-10 | 2.30e-05 | 4.369 | 0.0004555 |
| ViT-B/16 | fp16-autocast | hidden-state | state-11 | 1.59e-05 | 4.38 | 0.0002934 |
| ViT-B/16 | fp16-autocast | hidden-state | state-12 | 1.56e-06 | 4.321 | 0.000294 |
| ViT-B/16 | bf16-autocast | stored-embeddings | iss-kibo-lab/224x224 | 4.87e-05 | 0.02921 | 0.003882 |
| ViT-B/16 | bf16-autocast | stored-block | block-0 | 5.32e-05 | 0.1143 | 0.003826 |
| ViT-B/16 | bf16-autocast | stored-block | block-1 | 1.41e-05 | 0.06182 | 0.001359 |
| ViT-B/16 | bf16-autocast | stored-block | block-2 | 1.13e-05 | 12.94 | 0.001347 |
| ViT-B/16 | bf16-autocast | stored-block | block-3 | 1.93e-05 | 0.09124 | 9.514e-06 |
| ViT-B/16 | bf16-autocast | stored-block | block-4 | 4.51e-05 | 0.2264 | 2.362e-05 |
| ViT-B/16 | bf16-autocast | stored-block | block-5 | 3.93e-05 | 0.6632 | 6.918e-05 |
| ViT-B/16 | bf16-autocast | stored-block | block-6 | 2.46e-05 | 3.084 | 0.0003216 |
| ViT-B/16 | bf16-autocast | stored-block | block-7 | 1.88e-05 | 2.609 | 0.0002721 |
| ViT-B/16 | bf16-autocast | stored-block | block-8 | 1.78e-05 | 3.231 | 0.000337 |
| ViT-B/16 | bf16-autocast | stored-block | block-9 | 1.67e-05 | 10.05 | 0.001047 |
| ViT-B/16 | bf16-autocast | stored-block | block-10 | 1.89e-05 | 39 | 0.002613 |
| ViT-B/16 | bf16-autocast | stored-block | block-11 | 5.71e-06 | 4.562 | 0.0003104 |
| ViT-B/16 | bf16-autocast | stored-final-norm | iss-kibo-lab/224x224 | 1.31e-14 | 2.384e-06 | 1.337e-07 |
| ViT-B/16 | bf16-autocast | hidden-state | state-0 | 4.87e-05 | 0.02921 | 0.003882 |
| ViT-B/16 | bf16-autocast | hidden-state | state-1 | 9.05e-05 | 0.1183 | 0.003961 |
| ViT-B/16 | bf16-autocast | hidden-state | state-2 | 2.32e-04 | 0.09604 | 0.002111 |
| ViT-B/16 | bf16-autocast | hidden-state | state-3 | 5.38e-04 | 8.387 | 0.0008731 |
| ViT-B/16 | bf16-autocast | hidden-state | state-4 | 1.03e-03 | 8.409 | 0.0008769 |
| ViT-B/16 | bf16-autocast | hidden-state | state-5 | 1.61e-03 | 8.412 | 0.0008773 |
| ViT-B/16 | bf16-autocast | hidden-state | state-6 | 2.56e-03 | 8.408 | 0.000877 |
| ViT-B/16 | bf16-autocast | hidden-state | state-7 | 2.91e-03 | 8.351 | 0.0008711 |
| ViT-B/16 | bf16-autocast | hidden-state | state-8 | 3.03e-03 | 8.338 | 0.0008696 |
| ViT-B/16 | bf16-autocast | hidden-state | state-9 | 3.03e-03 | 8.196 | 0.0008548 |
| ViT-B/16 | bf16-autocast | hidden-state | state-10 | 3.00e-03 | 9.214 | 0.0009606 |
| ViT-B/16 | bf16-autocast | hidden-state | state-11 | 1.84e-03 | 45.52 | 0.00305 |
| ViT-B/16 | bf16-autocast | hidden-state | state-12 | 1.48e-04 | 46.76 | 0.003181 |
| ViT-L/16 | fp32 | stored-embeddings | iss-kibo-lab/224x224 | 6.70e-13 | 2.861e-05 | 1.882e-06 |
| ViT-L/16 | fp32 | stored-block | block-0 | 1.27e-12 | 0.04688 | 3.006e-07 |
| ViT-L/16 | fp32 | stored-block | block-1 | 1.31e-12 | 0.0009766 | 6.264e-09 |
| ViT-L/16 | fp32 | stored-block | block-2 | 2.89e-12 | 0.0009766 | 6.264e-09 |
| ViT-L/16 | fp32 | stored-block | block-3 | 2.10e-12 | 3.052e-05 | 1.958e-10 |
| ViT-L/16 | fp32 | stored-block | block-4 | 1.48e-12 | 3.052e-05 | 1.958e-10 |
| ViT-L/16 | fp32 | stored-block | block-5 | 1.88e-12 | 8.392e-05 | 5.383e-10 |
| ViT-L/16 | fp32 | stored-block | block-6 | 1.61e-12 | 6.104e-05 | 3.915e-10 |
| ViT-L/16 | fp32 | stored-block | block-7 | 2.83e-12 | 0.0001011 | 6.485e-10 |
| ViT-L/16 | fp32 | stored-block | block-8 | 1.61e-12 | 8.392e-05 | 5.384e-10 |
| ViT-L/16 | fp32 | stored-block | block-9 | 2.25e-12 | 0.0002441 | 1.566e-09 |
| ViT-L/16 | fp32 | stored-block | block-10 | 2.10e-12 | 0.0001335 | 8.566e-10 |
| ViT-L/16 | fp32 | stored-block | block-11 | 3.73e-12 | 0.0002136 | 1.371e-09 |
| ViT-L/16 | fp32 | stored-block | block-12 | 5.75e-12 | 0.0002403 | 1.542e-09 |
| ViT-L/16 | fp32 | stored-block | block-13 | 1.20e-11 | 0.0004139 | 2.656e-09 |
| ViT-L/16 | fp32 | stored-block | block-14 | 1.52e-11 | 0.0004234 | 2.718e-09 |
| ViT-L/16 | fp32 | stored-block | block-15 | 1.10e-11 | 0.0004883 | 3.134e-09 |
| ViT-L/16 | fp32 | stored-block | block-16 | 6.05e-12 | 0.0004044 | 2.595e-09 |
| ViT-L/16 | fp32 | stored-block | block-17 | 4.00e-12 | 0.0004578 | 2.938e-09 |
| ViT-L/16 | fp32 | stored-block | block-18 | 6.50e-12 | 0.0008392 | 5.385e-09 |
| ViT-L/16 | fp32 | stored-block | block-19 | 7.11e-12 | 0.0009079 | 5.822e-09 |
| ViT-L/16 | fp32 | stored-block | block-20 | 9.05e-12 | 0.00145 | 9.288e-09 |
| ViT-L/16 | fp32 | stored-block | block-21 | 1.33e-11 | 0.002563 | 1.641e-08 |
| ViT-L/16 | fp32 | stored-block | block-22 | 7.92e-12 | 0.003906 | 2.486e-08 |
| ViT-L/16 | fp32 | stored-block | block-23 | 1.45e-12 | 0.007568 | 4.888e-08 |
| ViT-L/16 | fp32 | stored-final-norm | iss-kibo-lab/224x224 | 1.19e-14 | 9.537e-07 | 6.089e-08 |
| ViT-L/16 | fp32 | hidden-state | state-0 | 6.70e-13 | 2.861e-05 | 1.882e-06 |
| ViT-L/16 | fp32 | hidden-state | state-1 | 1.77e-12 | 0.04688 | 3.006e-07 |
| ViT-L/16 | fp32 | hidden-state | state-2 | 2.91e-12 | 0.04688 | 3.007e-07 |
| ViT-L/16 | fp32 | hidden-state | state-3 | 3.87e-12 | 0.04688 | 3.007e-07 |
| ViT-L/16 | fp32 | hidden-state | state-4 | 4.12e-12 | 0.04688 | 3.007e-07 |
| ViT-L/16 | fp32 | hidden-state | state-5 | 1.26e-11 | 0.04688 | 3.007e-07 |
| ViT-L/16 | fp32 | hidden-state | state-6 | 5.70e-12 | 0.04688 | 3.007e-07 |
| ViT-L/16 | fp32 | hidden-state | state-7 | 5.37e-12 | 0.04688 | 3.007e-07 |
| ViT-L/16 | fp32 | hidden-state | state-8 | 7.96e-12 | 0.04688 | 3.007e-07 |
| ViT-L/16 | fp32 | hidden-state | state-9 | 9.53e-12 | 0.04688 | 3.007e-07 |
| ViT-L/16 | fp32 | hidden-state | state-10 | 9.75e-12 | 0.04688 | 3.007e-07 |
| ViT-L/16 | fp32 | hidden-state | state-11 | 1.46e-11 | 0.04688 | 3.007e-07 |
| ViT-L/16 | fp32 | hidden-state | state-12 | 7.42e-11 | 0.04688 | 3.008e-07 |
| ViT-L/16 | fp32 | hidden-state | state-13 | 1.34e-10 | 0.04688 | 3.008e-07 |
| ViT-L/16 | fp32 | hidden-state | state-14 | 2.55e-10 | 0.04688 | 3.008e-07 |
| ViT-L/16 | fp32 | hidden-state | state-15 | 3.24e-10 | 0.04688 | 3.008e-07 |
| ViT-L/16 | fp32 | hidden-state | state-16 | 3.91e-10 | 0.04688 | 3.009e-07 |
| ViT-L/16 | fp32 | hidden-state | state-17 | 4.30e-10 | 0.04688 | 3.008e-07 |
| ViT-L/16 | fp32 | hidden-state | state-18 | 4.43e-10 | 0.04688 | 3.009e-07 |
| ViT-L/16 | fp32 | hidden-state | state-19 | 4.82e-10 | 0.04688 | 3.008e-07 |
| ViT-L/16 | fp32 | hidden-state | state-20 | 5.06e-10 | 0.04688 | 3.006e-07 |
| ViT-L/16 | fp32 | hidden-state | state-21 | 4.83e-10 | 0.04688 | 3.004e-07 |
| ViT-L/16 | fp32 | hidden-state | state-22 | 3.87e-10 | 0.04688 | 3.001e-07 |
| ViT-L/16 | fp32 | hidden-state | state-23 | 3.38e-10 | 0.04688 | 2.983e-07 |
| ViT-L/16 | fp32 | hidden-state | state-24 | 4.49e-11 | 0.04688 | 3.028e-07 |
| ViT-L/16 | fp16-cast | stored-embeddings | iss-kibo-lab/224x224 | 1.59e-07 | 0.005273 | 0.0003468 |
| ViT-L/16 | fp16-cast | stored-block | block-0 | nan | inf | inf |
| ViT-L/16 | fp16-cast | stored-block | block-1 | nan | nan | nan |
| ViT-L/16 | fp16-cast | stored-block | block-2 | nan | nan | nan |
| ViT-L/16 | fp16-cast | stored-block | block-3 | nan | nan | nan |
| ViT-L/16 | fp16-cast | stored-block | block-4 | nan | nan | nan |
| ViT-L/16 | fp16-cast | stored-block | block-5 | nan | nan | nan |
| ViT-L/16 | fp16-cast | stored-block | block-6 | nan | nan | nan |
| ViT-L/16 | fp16-cast | stored-block | block-7 | nan | nan | nan |
| ViT-L/16 | fp16-cast | stored-block | block-8 | nan | nan | nan |
| ViT-L/16 | fp16-cast | stored-block | block-9 | nan | nan | nan |
| ViT-L/16 | fp16-cast | stored-block | block-10 | nan | nan | nan |
| ViT-L/16 | fp16-cast | stored-block | block-11 | nan | nan | nan |
| ViT-L/16 | fp16-cast | stored-block | block-12 | nan | nan | nan |
| ViT-L/16 | fp16-cast | stored-block | block-13 | nan | nan | nan |
| ViT-L/16 | fp16-cast | stored-block | block-14 | nan | nan | nan |
| ViT-L/16 | fp16-cast | stored-block | block-15 | nan | nan | nan |
| ViT-L/16 | fp16-cast | stored-block | block-16 | nan | nan | nan |
| ViT-L/16 | fp16-cast | stored-block | block-17 | nan | nan | nan |
| ViT-L/16 | fp16-cast | stored-block | block-18 | nan | nan | nan |
| ViT-L/16 | fp16-cast | stored-block | block-19 | nan | nan | nan |
| ViT-L/16 | fp16-cast | stored-block | block-20 | nan | nan | nan |
| ViT-L/16 | fp16-cast | stored-block | block-21 | nan | nan | nan |
| ViT-L/16 | fp16-cast | stored-block | block-22 | nan | nan | nan |
| ViT-L/16 | fp16-cast | stored-block | block-23 | nan | nan | nan |
| ViT-L/16 | fp16-cast | stored-final-norm | iss-kibo-lab/224x224 | nan | nan | nan |
| ViT-L/16 | fp16-cast | hidden-state | state-0 | 1.59e-07 | 0.005273 | 0.0003468 |
| ViT-L/16 | fp16-cast | hidden-state | state-1 | nan | inf | inf |
| ViT-L/16 | fp16-cast | hidden-state | state-2 | nan | nan | nan |
| ViT-L/16 | fp16-cast | hidden-state | state-3 | nan | nan | nan |
| ViT-L/16 | fp16-cast | hidden-state | state-4 | nan | nan | nan |
| ViT-L/16 | fp16-cast | hidden-state | state-5 | nan | nan | nan |
| ViT-L/16 | fp16-cast | hidden-state | state-6 | nan | nan | nan |
| ViT-L/16 | fp16-cast | hidden-state | state-7 | nan | nan | nan |
| ViT-L/16 | fp16-cast | hidden-state | state-8 | nan | nan | nan |
| ViT-L/16 | fp16-cast | hidden-state | state-9 | nan | nan | nan |
| ViT-L/16 | fp16-cast | hidden-state | state-10 | nan | nan | nan |
| ViT-L/16 | fp16-cast | hidden-state | state-11 | nan | nan | nan |
| ViT-L/16 | fp16-cast | hidden-state | state-12 | nan | nan | nan |
| ViT-L/16 | fp16-cast | hidden-state | state-13 | nan | nan | nan |
| ViT-L/16 | fp16-cast | hidden-state | state-14 | nan | nan | nan |
| ViT-L/16 | fp16-cast | hidden-state | state-15 | nan | nan | nan |
| ViT-L/16 | fp16-cast | hidden-state | state-16 | nan | nan | nan |
| ViT-L/16 | fp16-cast | hidden-state | state-17 | nan | nan | nan |
| ViT-L/16 | fp16-cast | hidden-state | state-18 | nan | nan | nan |
| ViT-L/16 | fp16-cast | hidden-state | state-19 | nan | nan | nan |
| ViT-L/16 | fp16-cast | hidden-state | state-20 | nan | nan | nan |
| ViT-L/16 | fp16-cast | hidden-state | state-21 | nan | nan | nan |
| ViT-L/16 | fp16-cast | hidden-state | state-22 | nan | nan | nan |
| ViT-L/16 | fp16-cast | hidden-state | state-23 | nan | nan | nan |
| ViT-L/16 | fp16-cast | hidden-state | state-24 | nan | nan | nan |
| ViT-L/16 | bf16-cast | stored-embeddings | iss-kibo-lab/224x224 | 1.03e-05 | 0.03235 | 0.002128 |
| ViT-L/16 | bf16-cast | stored-block | block-0 | 3.51e-05 | 748.6 | 0.004801 |
| ViT-L/16 | bf16-cast | stored-block | block-1 | 3.48e-05 | 246.8 | 0.001583 |
| ViT-L/16 | bf16-cast | stored-block | block-2 | 2.69e-05 | 242.6 | 0.001556 |
| ViT-L/16 | bf16-cast | stored-block | block-3 | 1.60e-05 | 242.4 | 0.001555 |
| ViT-L/16 | bf16-cast | stored-block | block-4 | 1.32e-05 | 242.2 | 0.001553 |
| ViT-L/16 | bf16-cast | stored-block | block-5 | 1.35e-05 | 242.2 | 0.001554 |
| ViT-L/16 | bf16-cast | stored-block | block-6 | 1.26e-05 | 241.5 | 0.001549 |
| ViT-L/16 | bf16-cast | stored-block | block-7 | 2.50e-05 | 241 | 0.001546 |
| ViT-L/16 | bf16-cast | stored-block | block-8 | 1.32e-05 | 238.6 | 0.00153 |
| ViT-L/16 | bf16-cast | stored-block | block-9 | 1.40e-05 | 227.5 | 0.001459 |
| ViT-L/16 | bf16-cast | stored-block | block-10 | 1.74e-05 | 222.2 | 0.001426 |
| ViT-L/16 | bf16-cast | stored-block | block-11 | 2.00e-05 | 208.7 | 0.001339 |
| ViT-L/16 | bf16-cast | stored-block | block-12 | 2.92e-05 | 185.9 | 0.001193 |
| ViT-L/16 | bf16-cast | stored-block | block-13 | 5.06e-05 | 161.3 | 0.001035 |
| ViT-L/16 | bf16-cast | stored-block | block-14 | 2.39e-05 | 163.9 | 0.001052 |
| ViT-L/16 | bf16-cast | stored-block | block-15 | 4.72e-05 | 136.3 | 0.0008748 |
| ViT-L/16 | bf16-cast | stored-block | block-16 | 3.53e-05 | 180.5 | 0.001158 |
| ViT-L/16 | bf16-cast | stored-block | block-17 | 2.87e-05 | 154.9 | 0.0009942 |
| ViT-L/16 | bf16-cast | stored-block | block-18 | 2.38e-05 | 199.9 | 0.001283 |
| ViT-L/16 | bf16-cast | stored-block | block-19 | 3.48e-05 | 295 | 0.001891 |
| ViT-L/16 | bf16-cast | stored-block | block-20 | 5.68e-05 | 414.7 | 0.002657 |
| ViT-L/16 | bf16-cast | stored-block | block-21 | 4.55e-05 | 556.4 | 0.003562 |
| ViT-L/16 | bf16-cast | stored-block | block-22 | 5.65e-05 | 565.6 | 0.003599 |
| ViT-L/16 | bf16-cast | stored-block | block-23 | 1.51e-05 | 1224 | 0.007906 |
| ViT-L/16 | bf16-cast | stored-final-norm | iss-kibo-lab/224x224 | 1.55e-05 | 0.03761 | 0.002401 |
| ViT-L/16 | bf16-cast | hidden-state | state-0 | 1.03e-05 | 0.03235 | 0.002128 |
| ViT-L/16 | bf16-cast | hidden-state | state-1 | 4.34e-05 | 748.6 | 0.004801 |
| ViT-L/16 | bf16-cast | hidden-state | state-2 | 6.24e-05 | 777.2 | 0.004986 |
| ViT-L/16 | bf16-cast | hidden-state | state-3 | 1.18e-04 | 781.4 | 0.005013 |
| ViT-L/16 | bf16-cast | hidden-state | state-4 | 1.67e-04 | 781.6 | 0.005014 |
| ViT-L/16 | bf16-cast | hidden-state | state-5 | 2.12e-04 | 781.8 | 0.005015 |
| ViT-L/16 | bf16-cast | hidden-state | state-6 | 4.40e-04 | 781.8 | 0.005015 |
| ViT-L/16 | bf16-cast | hidden-state | state-7 | 7.72e-04 | 782.5 | 0.005019 |
| ViT-L/16 | bf16-cast | hidden-state | state-8 | 7.28e-04 | 783 | 0.005023 |
| ViT-L/16 | bf16-cast | hidden-state | state-9 | 8.87e-04 | 785.4 | 0.005038 |
| ViT-L/16 | bf16-cast | hidden-state | state-10 | 1.01e-03 | 796.5 | 0.00511 |
| ViT-L/16 | bf16-cast | hidden-state | state-11 | 1.59e-03 | 801.8 | 0.005144 |
| ViT-L/16 | bf16-cast | hidden-state | state-12 | 2.78e-03 | 815.3 | 0.005231 |
| ViT-L/16 | bf16-cast | hidden-state | state-13 | 5.59e-03 | 838.1 | 0.005378 |
| ViT-L/16 | bf16-cast | hidden-state | state-14 | 8.15e-03 | 862.7 | 0.005537 |
| ViT-L/16 | bf16-cast | hidden-state | state-15 | 8.86e-03 | 860.1 | 0.00552 |
| ViT-L/16 | bf16-cast | hidden-state | state-16 | 9.33e-03 | 887.7 | 0.005698 |
| ViT-L/16 | bf16-cast | hidden-state | state-17 | 9.29e-03 | 843.5 | 0.005413 |
| ViT-L/16 | bf16-cast | hidden-state | state-18 | 9.44e-03 | 869.1 | 0.005578 |
| ViT-L/16 | bf16-cast | hidden-state | state-19 | 8.07e-03 | 824.1 | 0.005288 |
| ViT-L/16 | bf16-cast | hidden-state | state-20 | 8.60e-03 | 729 | 0.004675 |
| ViT-L/16 | bf16-cast | hidden-state | state-21 | 6.97e-03 | 609.3 | 0.003904 |
| ViT-L/16 | bf16-cast | hidden-state | state-22 | 8.09e-03 | 467.6 | 0.002993 |
| ViT-L/16 | bf16-cast | hidden-state | state-23 | 6.21e-03 | 565.6 | 0.003599 |
| ViT-L/16 | bf16-cast | hidden-state | state-24 | 5.09e-04 | 200 | 0.001292 |
| ViT-L/16 | fp16-autocast | stored-embeddings | iss-kibo-lab/224x224 | 1.59e-07 | 0.005273 | 0.0003468 |
| ViT-L/16 | fp16-autocast | stored-block | block-0 | 2.34e-07 | 27.82 | 0.0001784 |
| ViT-L/16 | fp16-autocast | stored-block | block-1 | 1.16e-06 | 0.03058 | 1.962e-07 |
| ViT-L/16 | fp16-autocast | stored-block | block-2 | 8.39e-08 | 0.02051 | 1.316e-07 |
| ViT-L/16 | fp16-autocast | stored-block | block-3 | 2.34e-08 | 0.01367 | 8.77e-08 |
| ViT-L/16 | fp16-autocast | stored-block | block-4 | 4.11e-08 | 0.006678 | 4.284e-08 |
| ViT-L/16 | fp16-autocast | stored-block | block-5 | 6.14e-08 | 0.0116 | 7.439e-08 |
| ViT-L/16 | fp16-autocast | stored-block | block-6 | 4.69e-08 | 0.07693 | 4.935e-07 |
| ViT-L/16 | fp16-autocast | stored-block | block-7 | 7.58e-08 | 0.01335 | 8.565e-08 |
| ViT-L/16 | fp16-autocast | stored-block | block-8 | 4.70e-08 | 0.01562 | 1.002e-07 |
| ViT-L/16 | fp16-autocast | stored-block | block-9 | 7.13e-08 | 0.04297 | 2.757e-07 |
| ViT-L/16 | fp16-autocast | stored-block | block-10 | 1.25e-07 | 0.02098 | 1.346e-07 |
| ViT-L/16 | fp16-autocast | stored-block | block-11 | 1.11e-07 | 0.02868 | 1.84e-07 |
| ViT-L/16 | fp16-autocast | stored-block | block-12 | 1.05e-07 | 0.05859 | 3.76e-07 |
| ViT-L/16 | fp16-autocast | stored-block | block-13 | 1.28e-07 | 0.03337 | 2.142e-07 |
| ViT-L/16 | fp16-autocast | stored-block | block-14 | 1.54e-07 | 0.0421 | 2.702e-07 |
| ViT-L/16 | fp16-autocast | stored-block | block-15 | 3.80e-07 | 0.05565 | 3.572e-07 |
| ViT-L/16 | fp16-autocast | stored-block | block-16 | 1.43e-07 | 0.04688 | 3.008e-07 |
| ViT-L/16 | fp16-autocast | stored-block | block-17 | 2.18e-07 | 0.07812 | 5.014e-07 |
| ViT-L/16 | fp16-autocast | stored-block | block-18 | 1.98e-07 | 0.08097 | 5.196e-07 |
| ViT-L/16 | fp16-autocast | stored-block | block-19 | 1.96e-07 | 0.1049 | 6.726e-07 |
| ViT-L/16 | fp16-autocast | stored-block | block-20 | 3.04e-07 | 0.2419 | 1.55e-06 |
| ViT-L/16 | fp16-autocast | stored-block | block-21 | 3.69e-07 | 0.3489 | 2.234e-06 |
| ViT-L/16 | fp16-autocast | stored-block | block-22 | 4.91e-07 | 0.4326 | 2.753e-06 |
| ViT-L/16 | fp16-autocast | stored-block | block-23 | 1.08e-07 | 1.388 | 8.963e-06 |
| ViT-L/16 | fp16-autocast | stored-final-norm | iss-kibo-lab/224x224 | 1.19e-14 | 9.537e-07 | 6.089e-08 |
| ViT-L/16 | fp16-autocast | hidden-state | state-0 | 1.59e-07 | 0.005273 | 0.0003468 |
| ViT-L/16 | fp16-autocast | hidden-state | state-1 | 4.15e-07 | 27.82 | 0.0001784 |
| ViT-L/16 | fp16-autocast | hidden-state | state-2 | 1.37e-06 | 27.82 | 0.0001784 |
| ViT-L/16 | fp16-autocast | hidden-state | state-3 | 1.87e-06 | 27.82 | 0.0001784 |
| ViT-L/16 | fp16-autocast | hidden-state | state-4 | 2.29e-06 | 27.82 | 0.0001784 |
| ViT-L/16 | fp16-autocast | hidden-state | state-5 | 3.58e-06 | 27.81 | 0.0001784 |
| ViT-L/16 | fp16-autocast | hidden-state | state-6 | 6.12e-06 | 27.81 | 0.0001784 |
| ViT-L/16 | fp16-autocast | hidden-state | state-7 | 8.62e-06 | 27.8 | 0.0001784 |
| ViT-L/16 | fp16-autocast | hidden-state | state-8 | 9.47e-06 | 27.8 | 0.0001783 |
| ViT-L/16 | fp16-autocast | hidden-state | state-9 | 1.15e-05 | 27.81 | 0.0001784 |
| ViT-L/16 | fp16-autocast | hidden-state | state-10 | 1.78e-05 | 27.86 | 0.0001787 |
| ViT-L/16 | fp16-autocast | hidden-state | state-11 | 2.91e-05 | 27.86 | 0.0001788 |
| ViT-L/16 | fp16-autocast | hidden-state | state-12 | 4.51e-05 | 27.88 | 0.0001789 |
| ViT-L/16 | fp16-autocast | hidden-state | state-13 | 7.51e-05 | 27.79 | 0.0001783 |
| ViT-L/16 | fp16-autocast | hidden-state | state-14 | 1.20e-04 | 27.77 | 0.0001782 |
| ViT-L/16 | fp16-autocast | hidden-state | state-15 | 1.61e-04 | 27.76 | 0.0001782 |
| ViT-L/16 | fp16-autocast | hidden-state | state-16 | 1.67e-04 | 27.78 | 0.0001783 |
| ViT-L/16 | fp16-autocast | hidden-state | state-17 | 1.60e-04 | 27.79 | 0.0001784 |
| ViT-L/16 | fp16-autocast | hidden-state | state-18 | 1.62e-04 | 27.8 | 0.0001784 |
| ViT-L/16 | fp16-autocast | hidden-state | state-19 | 1.50e-04 | 27.77 | 0.0001782 |
| ViT-L/16 | fp16-autocast | hidden-state | state-20 | 1.03e-04 | 27.75 | 0.0001779 |
| ViT-L/16 | fp16-autocast | hidden-state | state-21 | 6.37e-05 | 27.77 | 0.0001779 |
| ViT-L/16 | fp16-autocast | hidden-state | state-22 | 7.11e-05 | 27.8 | 0.000178 |
| ViT-L/16 | fp16-autocast | hidden-state | state-23 | 6.98e-05 | 27.7 | 0.0001763 |
| ViT-L/16 | fp16-autocast | hidden-state | state-24 | 5.80e-06 | 27.64 | 0.0001786 |
| ViT-L/16 | bf16-autocast | stored-embeddings | iss-kibo-lab/224x224 | 1.03e-05 | 0.03235 | 0.002128 |
| ViT-L/16 | bf16-autocast | stored-block | block-0 | 2.28e-05 | 328.2 | 0.002105 |
| ViT-L/16 | bf16-autocast | stored-block | block-1 | 3.05e-05 | 0.1892 | 1.214e-06 |
| ViT-L/16 | bf16-autocast | stored-block | block-2 | 3.01e-06 | 0.08085 | 5.187e-07 |
| ViT-L/16 | bf16-autocast | stored-block | block-3 | 1.30e-06 | 0.043 | 2.758e-07 |
| ViT-L/16 | bf16-autocast | stored-block | block-4 | 2.42e-06 | 0.05014 | 3.216e-07 |
| ViT-L/16 | bf16-autocast | stored-block | block-5 | 2.73e-06 | 0.07265 | 4.66e-07 |
| ViT-L/16 | bf16-autocast | stored-block | block-6 | 2.75e-06 | 1.286 | 8.252e-06 |
| ViT-L/16 | bf16-autocast | stored-block | block-7 | 6.79e-06 | 0.1067 | 6.846e-07 |
| ViT-L/16 | bf16-autocast | stored-block | block-8 | 3.76e-06 | 0.09766 | 6.265e-07 |
| ViT-L/16 | bf16-autocast | stored-block | block-9 | 3.18e-06 | 0.4028 | 2.584e-06 |
| ViT-L/16 | bf16-autocast | stored-block | block-10 | 6.52e-06 | 0.2568 | 1.647e-06 |
| ViT-L/16 | bf16-autocast | stored-block | block-11 | 8.78e-06 | 0.5312 | 3.409e-06 |
| ViT-L/16 | bf16-autocast | stored-block | block-12 | 7.70e-06 | 0.6914 | 4.437e-06 |
| ViT-L/16 | bf16-autocast | stored-block | block-13 | 8.83e-06 | 0.5703 | 3.66e-06 |
| ViT-L/16 | bf16-autocast | stored-block | block-14 | 1.16e-05 | 0.3738 | 2.399e-06 |
| ViT-L/16 | bf16-autocast | stored-block | block-15 | 1.19e-05 | 0.392 | 2.516e-06 |
| ViT-L/16 | bf16-autocast | stored-block | block-16 | 9.31e-06 | 0.4115 | 2.641e-06 |
| ViT-L/16 | bf16-autocast | stored-block | block-17 | 1.46e-05 | 0.5952 | 3.82e-06 |
| ViT-L/16 | bf16-autocast | stored-block | block-18 | 1.29e-05 | 0.6801 | 4.364e-06 |
| ViT-L/16 | bf16-autocast | stored-block | block-19 | 1.31e-05 | 0.7713 | 4.946e-06 |
| ViT-L/16 | bf16-autocast | stored-block | block-20 | 1.63e-05 | 2.105 | 1.349e-05 |
| ViT-L/16 | bf16-autocast | stored-block | block-21 | 2.76e-05 | 2.307 | 1.477e-05 |
| ViT-L/16 | bf16-autocast | stored-block | block-22 | 1.94e-05 | 3.679 | 2.342e-05 |
| ViT-L/16 | bf16-autocast | stored-block | block-23 | 5.13e-06 | 8.181 | 5.284e-05 |
| ViT-L/16 | bf16-autocast | stored-final-norm | iss-kibo-lab/224x224 | 1.19e-14 | 9.537e-07 | 6.089e-08 |
| ViT-L/16 | bf16-autocast | hidden-state | state-0 | 1.03e-05 | 0.03235 | 0.002128 |
| ViT-L/16 | bf16-autocast | hidden-state | state-1 | 3.02e-05 | 328.2 | 0.002105 |
| ViT-L/16 | bf16-autocast | hidden-state | state-2 | 5.28e-05 | 328.2 | 0.002105 |
| ViT-L/16 | bf16-autocast | hidden-state | state-3 | 6.29e-05 | 328.2 | 0.002105 |
| ViT-L/16 | bf16-autocast | hidden-state | state-4 | 1.30e-04 | 328.2 | 0.002105 |
| ViT-L/16 | bf16-autocast | hidden-state | state-5 | 1.84e-04 | 328.2 | 0.002105 |
| ViT-L/16 | bf16-autocast | hidden-state | state-6 | 4.11e-04 | 328.2 | 0.002105 |
| ViT-L/16 | bf16-autocast | hidden-state | state-7 | 6.19e-04 | 328.2 | 0.002105 |
| ViT-L/16 | bf16-autocast | hidden-state | state-8 | 6.45e-04 | 328.2 | 0.002105 |
| ViT-L/16 | bf16-autocast | hidden-state | state-9 | 8.41e-04 | 328.2 | 0.002105 |
| ViT-L/16 | bf16-autocast | hidden-state | state-10 | 9.97e-04 | 328.2 | 0.002105 |
| ViT-L/16 | bf16-autocast | hidden-state | state-11 | 2.09e-03 | 328.2 | 0.002105 |
| ViT-L/16 | bf16-autocast | hidden-state | state-12 | 3.92e-03 | 328.1 | 0.002105 |
| ViT-L/16 | bf16-autocast | hidden-state | state-13 | 6.53e-03 | 328.1 | 0.002105 |
| ViT-L/16 | bf16-autocast | hidden-state | state-14 | 1.05e-02 | 327.9 | 0.002104 |
| ViT-L/16 | bf16-autocast | hidden-state | state-15 | 1.06e-02 | 327.9 | 0.002105 |
| ViT-L/16 | bf16-autocast | hidden-state | state-16 | 1.18e-02 | 328 | 0.002105 |
| ViT-L/16 | bf16-autocast | hidden-state | state-17 | 1.19e-02 | 328.3 | 0.002107 |
| ViT-L/16 | bf16-autocast | hidden-state | state-18 | 1.22e-02 | 328.4 | 0.002108 |
| ViT-L/16 | bf16-autocast | hidden-state | state-19 | 9.52e-03 | 328.1 | 0.002105 |
| ViT-L/16 | bf16-autocast | hidden-state | state-20 | 8.73e-03 | 327.7 | 0.002101 |
| ViT-L/16 | bf16-autocast | hidden-state | state-21 | 7.35e-03 | 324.8 | 0.002081 |
| ViT-L/16 | bf16-autocast | hidden-state | state-22 | 9.71e-03 | 323.6 | 0.002072 |
| ViT-L/16 | bf16-autocast | hidden-state | state-23 | 6.38e-03 | 329.8 | 0.002099 |
| ViT-L/16 | bf16-autocast | hidden-state | state-24 | 5.31e-04 | 327.5 | 0.002115 |
| ViT-H+/16 | fp32 | stored-embeddings | iss-kibo-lab/224x224 | 5.56e-13 | 1.204e-05 | 7.716e-07 |
| ViT-H+/16 | fp32 | stored-block | block-0 | 1.10e-11 | 0.0002937 | 4.004e-06 |
| ViT-H+/16 | fp32 | stored-block | block-1 | 1.80e-11 | 0.0003433 | 2.468e-06 |
| ViT-H+/16 | fp32 | stored-block | block-2 | 8.22e-13 | 0.0001221 | 4.603e-07 |
| ViT-H+/16 | fp32 | stored-block | block-3 | 3.38e-13 | 7.629e-05 | 2.778e-07 |
| ViT-H+/16 | fp32 | stored-block | block-4 | 8.93e-13 | 0.0002289 | 9.412e-07 |
| ViT-H+/16 | fp32 | stored-block | block-5 | 1.10e-13 | 0.0001678 | 2.784e-07 |
| ViT-H+/16 | fp32 | stored-block | block-6 | 4.77e-14 | 0.0001526 | 2.492e-07 |
| ViT-H+/16 | fp32 | stored-block | block-7 | 2.12e-14 | 4.578e-05 | 7.538e-08 |
| ViT-H+/16 | fp32 | stored-block | block-8 | 2.21e-14 | 6.104e-05 | 1.009e-07 |
| ViT-H+/16 | fp32 | stored-block | block-9 | 8.27e-14 | 0.0001678 | 3.058e-07 |
| ViT-H+/16 | fp32 | stored-block | block-10 | 5.27e-14 | 9.155e-05 | 1.746e-07 |
| ViT-H+/16 | fp32 | stored-block | block-11 | 8.46e-14 | 0.0007935 | 1.925e-07 |
| ViT-H+/16 | fp32 | stored-block | block-12 | 7.75e-14 | 0.0001221 | 2.974e-08 |
| ViT-H+/16 | fp32 | stored-block | block-13 | 1.19e-13 | 0.0002747 | 6.684e-08 |
| ViT-H+/16 | fp32 | stored-block | block-14 | 1.72e-13 | 0.0001831 | 4.438e-08 |
| ViT-H+/16 | fp32 | stored-block | block-15 | 1.23e-13 | 0.0001831 | 4.451e-08 |
| ViT-H+/16 | fp32 | stored-block | block-16 | 1.07e-13 | 0.0002747 | 6.628e-08 |
| ViT-H+/16 | fp32 | stored-block | block-17 | 2.86e-13 | 0.0003967 | 9.613e-08 |
| ViT-H+/16 | fp32 | stored-block | block-18 | 2.25e-13 | 0.0003357 | 8.227e-08 |
| ViT-H+/16 | fp32 | stored-block | block-19 | 3.81e-13 | 0.0004272 | 1.068e-07 |
| ViT-H+/16 | fp32 | stored-block | block-20 | 4.18e-13 | 0.0004425 | 1.143e-07 |
| ViT-H+/16 | fp32 | stored-block | block-21 | 3.00e-13 | 0.000351 | 9.478e-08 |
| ViT-H+/16 | fp32 | stored-block | block-22 | 3.91e-13 | 0.07031 | 5.362e-07 |
| ViT-H+/16 | fp32 | stored-block | block-23 | 3.68e-13 | 0.0003357 | 2.556e-09 |
| ViT-H+/16 | fp32 | stored-block | block-24 | 5.79e-13 | 0.0007629 | 5.81e-09 |
| ViT-H+/16 | fp32 | stored-block | block-25 | 6.58e-13 | 0.007812 | 5.951e-08 |
| ViT-H+/16 | fp32 | stored-block | block-26 | 6.20e-13 | 0.007812 | 5.948e-08 |
| ViT-H+/16 | fp32 | stored-block | block-27 | 6.47e-13 | 0.001343 | 1.023e-08 |
| ViT-H+/16 | fp32 | stored-block | block-28 | 7.23e-13 | 0.001953 | 1.487e-08 |
| ViT-H+/16 | fp32 | stored-block | block-29 | 7.35e-13 | 0.007812 | 5.943e-08 |
| ViT-H+/16 | fp32 | stored-block | block-30 | 2.11e-12 | 0.005859 | 4.464e-08 |
| ViT-H+/16 | fp32 | stored-block | block-31 | 1.87e-12 | 0.04004 | 3.053e-07 |
| ViT-H+/16 | fp32 | stored-final-norm | iss-kibo-lab/224x224 | 1.27e-14 | 1.907e-06 | 1.55e-07 |
| ViT-H+/16 | fp32 | hidden-state | state-0 | 5.56e-13 | 1.204e-05 | 7.716e-07 |
| ViT-H+/16 | fp32 | hidden-state | state-1 | 1.08e-11 | 0.0002937 | 4.004e-06 |
| ViT-H+/16 | fp32 | hidden-state | state-2 | 2.57e-11 | 0.000412 | 2.961e-06 |
| ViT-H+/16 | fp32 | hidden-state | state-3 | 1.44e-11 | 0.0002441 | 9.206e-07 |
| ViT-H+/16 | fp32 | hidden-state | state-4 | 2.16e-12 | 0.0001831 | 6.667e-07 |
| ViT-H+/16 | fp32 | hidden-state | state-5 | 1.51e-12 | 0.0002403 | 9.883e-07 |
| ViT-H+/16 | fp32 | hidden-state | state-6 | 1.42e-12 | 0.0003204 | 5.316e-07 |
| ViT-H+/16 | fp32 | hidden-state | state-7 | 9.63e-13 | 0.0002899 | 4.735e-07 |
| ViT-H+/16 | fp32 | hidden-state | state-8 | 1.35e-12 | 0.0003204 | 5.277e-07 |
| ViT-H+/16 | fp32 | hidden-state | state-9 | 1.70e-12 | 0.0003052 | 5.046e-07 |
| ViT-H+/16 | fp32 | hidden-state | state-10 | 2.32e-12 | 0.000351 | 6.394e-07 |
| ViT-H+/16 | fp32 | hidden-state | state-11 | 4.03e-12 | 0.0002975 | 5.675e-07 |
| ViT-H+/16 | fp32 | hidden-state | state-12 | 4.86e-12 | 0.003418 | 8.294e-07 |
| ViT-H+/16 | fp32 | hidden-state | state-13 | 1.16e-11 | 0.003418 | 8.328e-07 |
| ViT-H+/16 | fp32 | hidden-state | state-14 | 3.95e-11 | 0.003418 | 8.318e-07 |
| ViT-H+/16 | fp32 | hidden-state | state-15 | 6.75e-11 | 0.003418 | 8.284e-07 |
| ViT-H+/16 | fp32 | hidden-state | state-16 | 1.11e-10 | 0.003418 | 8.308e-07 |
| ViT-H+/16 | fp32 | hidden-state | state-17 | 1.53e-10 | 0.003418 | 8.248e-07 |
| ViT-H+/16 | fp32 | hidden-state | state-18 | 3.37e-10 | 0.003365 | 8.153e-07 |
| ViT-H+/16 | fp32 | hidden-state | state-19 | 5.20e-10 | 0.003613 | 8.856e-07 |
| ViT-H+/16 | fp32 | hidden-state | state-20 | 1.12e-09 | 0.005291 | 1.322e-06 |
| ViT-H+/16 | fp32 | hidden-state | state-21 | 1.55e-09 | 0.006084 | 1.572e-06 |
| ViT-H+/16 | fp32 | hidden-state | state-22 | 2.73e-09 | 0.0089 | 2.404e-06 |
| ViT-H+/16 | fp32 | hidden-state | state-23 | 3.30e-09 | 0.08398 | 6.405e-07 |
| ViT-H+/16 | fp32 | hidden-state | state-24 | 5.61e-09 | 0.08398 | 6.395e-07 |
| ViT-H+/16 | fp32 | hidden-state | state-25 | 6.17e-09 | 0.08398 | 6.396e-07 |
| ViT-H+/16 | fp32 | hidden-state | state-26 | 7.26e-09 | 0.08203 | 6.248e-07 |
| ViT-H+/16 | fp32 | hidden-state | state-27 | 7.63e-09 | 0.08203 | 6.246e-07 |
| ViT-H+/16 | fp32 | hidden-state | state-28 | 8.34e-09 | 0.08398 | 6.397e-07 |
| ViT-H+/16 | fp32 | hidden-state | state-29 | 1.05e-08 | 0.08594 | 6.542e-07 |
| ViT-H+/16 | fp32 | hidden-state | state-30 | 1.62e-08 | 0.09375 | 7.132e-07 |
| ViT-H+/16 | fp32 | hidden-state | state-31 | 1.65e-08 | 0.09961 | 7.589e-07 |
| ViT-H+/16 | fp32 | hidden-state | state-32 | 2.26e-09 | 0.1383 | 1.054e-06 |
| ViT-H+/16 | fp16-cast | stored-embeddings | iss-kibo-lab/224x224 | 2.87e-07 | 0.00395 | 0.0002531 |
| ViT-H+/16 | fp16-cast | stored-block | block-0 | 3.64e-07 | 0.04206 | 0.0005733 |
| ViT-H+/16 | fp16-cast | stored-block | block-1 | 6.65e-06 | 0.2206 | 0.001586 |
| ViT-H+/16 | fp16-cast | stored-block | block-2 | 2.76e-07 | 0.1984 | 0.0007481 |
| ViT-H+/16 | fp16-cast | stored-block | block-3 | 2.27e-07 | 0.08987 | 0.0003272 |
| ViT-H+/16 | fp16-cast | stored-block | block-4 | 1.04e-07 | 0.1058 | 0.0004351 |
| ViT-H+/16 | fp16-cast | stored-block | block-5 | 9.51e-08 | 0.1903 | 0.0003157 |
| ViT-H+/16 | fp16-cast | stored-block | block-6 | 9.82e-08 | 0.2261 | 0.0003694 |
| ViT-H+/16 | fp16-cast | stored-block | block-7 | 7.56e-08 | 0.2673 | 0.0004402 |
| ViT-H+/16 | fp16-cast | stored-block | block-8 | 6.48e-08 | 0.1837 | 0.0003037 |
| ViT-H+/16 | fp16-cast | stored-block | block-9 | 7.81e-08 | 0.1285 | 0.0002342 |
| ViT-H+/16 | fp16-cast | stored-block | block-10 | 8.67e-08 | 0.1768 | 0.0003371 |
| ViT-H+/16 | fp16-cast | stored-block | block-11 | 7.79e-08 | 3.019 | 0.0007325 |
| ViT-H+/16 | fp16-cast | stored-block | block-12 | 8.89e-08 | 0.4653 | 0.0001134 |
| ViT-H+/16 | fp16-cast | stored-block | block-13 | 1.26e-07 | 2.923 | 0.0007114 |
| ViT-H+/16 | fp16-cast | stored-block | block-14 | 1.60e-07 | 1.771 | 0.0004291 |
| ViT-H+/16 | fp16-cast | stored-block | block-15 | 1.43e-07 | 1.825 | 0.0004436 |
| ViT-H+/16 | fp16-cast | stored-block | block-16 | 1.30e-07 | 0.3872 | 9.344e-05 |
| ViT-H+/16 | fp16-cast | stored-block | block-17 | 1.68e-07 | 1.009 | 0.0002446 |
| ViT-H+/16 | fp16-cast | stored-block | block-18 | 2.21e-07 | 1.634 | 0.0004003 |
| ViT-H+/16 | fp16-cast | stored-block | block-19 | 2.83e-07 | 0.7023 | 0.0001755 |
| ViT-H+/16 | fp16-cast | stored-block | block-20 | 3.29e-07 | 1.119 | 0.000289 |
| ViT-H+/16 | fp16-cast | stored-block | block-21 | 2.24e-07 | 0.6243 | 0.0001686 |
| ViT-H+/16 | fp16-cast | stored-block | block-22 | nan | inf | inf |
| ViT-H+/16 | fp16-cast | stored-block | block-23 | nan | nan | nan |
| ViT-H+/16 | fp16-cast | stored-block | block-24 | nan | nan | nan |
| ViT-H+/16 | fp16-cast | stored-block | block-25 | nan | nan | nan |
| ViT-H+/16 | fp16-cast | stored-block | block-26 | nan | nan | nan |
| ViT-H+/16 | fp16-cast | stored-block | block-27 | nan | nan | nan |
| ViT-H+/16 | fp16-cast | stored-block | block-28 | nan | nan | nan |
| ViT-H+/16 | fp16-cast | stored-block | block-29 | nan | nan | nan |
| ViT-H+/16 | fp16-cast | stored-block | block-30 | nan | nan | nan |
| ViT-H+/16 | fp16-cast | stored-block | block-31 | nan | nan | nan |
| ViT-H+/16 | fp16-cast | stored-final-norm | iss-kibo-lab/224x224 | nan | nan | nan |
| ViT-H+/16 | fp16-cast | hidden-state | state-0 | 2.87e-07 | 0.00395 | 0.0002531 |
| ViT-H+/16 | fp16-cast | hidden-state | state-1 | 4.24e-07 | 0.04206 | 0.0005733 |
| ViT-H+/16 | fp16-cast | hidden-state | state-2 | 6.54e-06 | 0.2206 | 0.001586 |
| ViT-H+/16 | fp16-cast | hidden-state | state-3 | 1.73e-06 | 0.1984 | 0.0007481 |
| ViT-H+/16 | fp16-cast | hidden-state | state-4 | 1.20e-06 | 0.1601 | 0.000583 |
| ViT-H+/16 | fp16-cast | hidden-state | state-5 | 3.74e-06 | 0.169 | 0.0006951 |
| ViT-H+/16 | fp16-cast | hidden-state | state-6 | 2.69e-06 | 0.1903 | 0.0003157 |
| ViT-H+/16 | fp16-cast | hidden-state | state-7 | 2.96e-06 | 0.2381 | 0.0003889 |
| ViT-H+/16 | fp16-cast | hidden-state | state-8 | 4.14e-06 | 0.2673 | 0.0004402 |
| ViT-H+/16 | fp16-cast | hidden-state | state-9 | 5.26e-06 | 0.3328 | 0.0005502 |
| ViT-H+/16 | fp16-cast | hidden-state | state-10 | 6.99e-06 | 0.4354 | 0.0007933 |
| ViT-H+/16 | fp16-cast | hidden-state | state-11 | 8.50e-06 | 0.3232 | 0.0006165 |
| ViT-H+/16 | fp16-cast | hidden-state | state-12 | 7.77e-06 | 3.019 | 0.0007325 |
| ViT-H+/16 | fp16-cast | hidden-state | state-13 | 1.58e-05 | 3.681 | 0.0008968 |
| ViT-H+/16 | fp16-cast | hidden-state | state-14 | 4.47e-05 | 6.923 | 0.001685 |
| ViT-H+/16 | fp16-cast | hidden-state | state-15 | 6.14e-05 | 6.229 | 0.00151 |
| ViT-H+/16 | fp16-cast | hidden-state | state-16 | 1.39e-04 | 6.175 | 0.001501 |
| ViT-H+/16 | fp16-cast | hidden-state | state-17 | 2.04e-04 | 8.13 | 0.001962 |
| ViT-H+/16 | fp16-cast | hidden-state | state-18 | 3.20e-04 | 9.009 | 0.002183 |
| ViT-H+/16 | fp16-cast | hidden-state | state-19 | 5.88e-04 | 9.634 | 0.002361 |
| ViT-H+/16 | fp16-cast | hidden-state | state-20 | 1.00e-03 | 9.977 | 0.002493 |
| ViT-H+/16 | fp16-cast | hidden-state | state-21 | 1.29e-03 | 9.409 | 0.002431 |
| ViT-H+/16 | fp16-cast | hidden-state | state-22 | 1.34e-03 | 9.376 | 0.002532 |
| ViT-H+/16 | fp16-cast | hidden-state | state-23 | nan | inf | inf |
| ViT-H+/16 | fp16-cast | hidden-state | state-24 | nan | nan | nan |
| ViT-H+/16 | fp16-cast | hidden-state | state-25 | nan | nan | nan |
| ViT-H+/16 | fp16-cast | hidden-state | state-26 | nan | nan | nan |
| ViT-H+/16 | fp16-cast | hidden-state | state-27 | nan | nan | nan |
| ViT-H+/16 | fp16-cast | hidden-state | state-28 | nan | nan | nan |
| ViT-H+/16 | fp16-cast | hidden-state | state-29 | nan | nan | nan |
| ViT-H+/16 | fp16-cast | hidden-state | state-30 | nan | nan | nan |
| ViT-H+/16 | fp16-cast | hidden-state | state-31 | nan | nan | nan |
| ViT-H+/16 | fp16-cast | hidden-state | state-32 | nan | nan | nan |
| ViT-H+/16 | bf16-cast | stored-embeddings | iss-kibo-lab/224x224 | 1.82e-05 | 0.03274 | 0.002098 |
| ViT-H+/16 | bf16-cast | stored-block | block-0 | 1.17e-05 | 0.3649 | 0.004974 |
| ViT-H+/16 | bf16-cast | stored-block | block-1 | 3.76e-04 | 2.002 | 0.01439 |
| ViT-H+/16 | bf16-cast | stored-block | block-2 | 1.29e-05 | 1.198 | 0.004519 |
| ViT-H+/16 | bf16-cast | stored-block | block-3 | 1.03e-05 | 0.6601 | 0.002403 |
| ViT-H+/16 | bf16-cast | stored-block | block-4 | 1.22e-05 | 1.169 | 0.004807 |
| ViT-H+/16 | bf16-cast | stored-block | block-5 | 7.58e-06 | 2.81 | 0.004661 |
| ViT-H+/16 | bf16-cast | stored-block | block-6 | 7.96e-06 | 3.774 | 0.006164 |
| ViT-H+/16 | bf16-cast | stored-block | block-7 | 3.68e-06 | 0.9603 | 0.001581 |
| ViT-H+/16 | bf16-cast | stored-block | block-8 | 4.59e-06 | 3.167 | 0.005237 |
| ViT-H+/16 | bf16-cast | stored-block | block-9 | 3.22e-06 | 0.8724 | 0.00159 |
| ViT-H+/16 | bf16-cast | stored-block | block-10 | 4.29e-06 | 2.243 | 0.004277 |
| ViT-H+/16 | bf16-cast | stored-block | block-11 | 5.56e-06 | 7.019 | 0.001703 |
| ViT-H+/16 | bf16-cast | stored-block | block-12 | 1.31e-05 | 23.68 | 0.00577 |
| ViT-H+/16 | bf16-cast | stored-block | block-13 | 6.51e-06 | 13.08 | 0.003182 |
| ViT-H+/16 | bf16-cast | stored-block | block-14 | 7.04e-06 | 2.972 | 0.0007204 |
| ViT-H+/16 | bf16-cast | stored-block | block-15 | 1.10e-05 | 14.18 | 0.003446 |
| ViT-H+/16 | bf16-cast | stored-block | block-16 | 9.66e-06 | 16.13 | 0.003893 |
| ViT-H+/16 | bf16-cast | stored-block | block-17 | 7.48e-06 | 1.908 | 0.0004624 |
| ViT-H+/16 | bf16-cast | stored-block | block-18 | 9.89e-06 | 3.138 | 0.000769 |
| ViT-H+/16 | bf16-cast | stored-block | block-19 | 1.54e-05 | 2.7 | 0.0006747 |
| ViT-H+/16 | bf16-cast | stored-block | block-20 | 1.54e-05 | 14.59 | 0.00377 |
| ViT-H+/16 | bf16-cast | stored-block | block-21 | 2.05e-05 | 8.134 | 0.002197 |
| ViT-H+/16 | bf16-cast | stored-block | block-22 | 1.47e-05 | 464.1 | 0.003539 |
| ViT-H+/16 | bf16-cast | stored-block | block-23 | 1.76e-05 | 254.6 | 0.001939 |
| ViT-H+/16 | bf16-cast | stored-block | block-24 | 1.73e-05 | 272.8 | 0.002078 |
| ViT-H+/16 | bf16-cast | stored-block | block-25 | 1.56e-05 | 218.5 | 0.001665 |
| ViT-H+/16 | bf16-cast | stored-block | block-26 | 1.49e-05 | 264.9 | 0.002017 |
| ViT-H+/16 | bf16-cast | stored-block | block-27 | 1.48e-05 | 223.3 | 0.001701 |
| ViT-H+/16 | bf16-cast | stored-block | block-28 | 1.71e-05 | 287.6 | 0.002189 |
| ViT-H+/16 | bf16-cast | stored-block | block-29 | 2.16e-05 | 375.7 | 0.002858 |
| ViT-H+/16 | bf16-cast | stored-block | block-30 | 4.36e-05 | 346.1 | 0.002637 |
| ViT-H+/16 | bf16-cast | stored-block | block-31 | 2.18e-05 | 137 | 0.001044 |
| ViT-H+/16 | bf16-cast | stored-final-norm | iss-kibo-lab/224x224 | 1.03e-05 | 0.01249 | 0.001015 |
| ViT-H+/16 | bf16-cast | hidden-state | state-0 | 1.82e-05 | 0.03274 | 0.002098 |
| ViT-H+/16 | bf16-cast | hidden-state | state-1 | 2.22e-05 | 0.3608 | 0.004917 |
| ViT-H+/16 | bf16-cast | hidden-state | state-2 | 2.58e-04 | 1.762 | 0.01266 |
| ViT-H+/16 | bf16-cast | hidden-state | state-3 | 7.15e-05 | 1.198 | 0.004519 |
| ViT-H+/16 | bf16-cast | hidden-state | state-4 | 9.62e-05 | 1.018 | 0.003706 |
| ViT-H+/16 | bf16-cast | hidden-state | state-5 | 2.72e-04 | 1.169 | 0.004807 |
| ViT-H+/16 | bf16-cast | hidden-state | state-6 | 2.84e-04 | 1.754 | 0.00291 |
| ViT-H+/16 | bf16-cast | hidden-state | state-7 | 3.30e-04 | 3.774 | 0.006164 |
| ViT-H+/16 | bf16-cast | hidden-state | state-8 | 3.61e-04 | 2.292 | 0.003774 |
| ViT-H+/16 | bf16-cast | hidden-state | state-9 | 3.84e-04 | 3.167 | 0.005237 |
| ViT-H+/16 | bf16-cast | hidden-state | state-10 | 4.17e-04 | 3.128 | 0.005698 |
| ViT-H+/16 | bf16-cast | hidden-state | state-11 | 4.66e-04 | 3.677 | 0.007012 |
| ViT-H+/16 | bf16-cast | hidden-state | state-12 | 4.71e-04 | 24.98 | 0.006062 |
| ViT-H+/16 | bf16-cast | hidden-state | state-13 | 6.83e-04 | 8.319 | 0.002027 |
| ViT-H+/16 | bf16-cast | hidden-state | state-14 | 1.29e-03 | 13.08 | 0.003182 |
| ViT-H+/16 | bf16-cast | hidden-state | state-15 | 1.62e-03 | 10.66 | 0.002584 |
| ViT-H+/16 | bf16-cast | hidden-state | state-16 | 2.10e-03 | 16.08 | 0.003909 |
| ViT-H+/16 | bf16-cast | hidden-state | state-17 | 2.71e-03 | 16.13 | 0.003893 |
| ViT-H+/16 | bf16-cast | hidden-state | state-18 | 3.96e-03 | 8.273 | 0.002005 |
| ViT-H+/16 | bf16-cast | hidden-state | state-19 | 7.45e-03 | 10.93 | 0.002678 |
| ViT-H+/16 | bf16-cast | hidden-state | state-20 | 1.36e-02 | 14.57 | 0.003641 |
| ViT-H+/16 | bf16-cast | hidden-state | state-21 | 1.51e-02 | 17.55 | 0.004534 |
| ViT-H+/16 | bf16-cast | hidden-state | state-22 | 2.21e-02 | 26.24 | 0.007086 |
| ViT-H+/16 | bf16-cast | hidden-state | state-23 | 2.62e-02 | 464.1 | 0.003539 |
| ViT-H+/16 | bf16-cast | hidden-state | state-24 | 3.49e-02 | 311.3 | 0.00237 |
| ViT-H+/16 | bf16-cast | hidden-state | state-25 | 4.54e-02 | 244.6 | 0.001862 |
| ViT-H+/16 | bf16-cast | hidden-state | state-26 | 5.15e-02 | 218.5 | 0.001665 |
| ViT-H+/16 | bf16-cast | hidden-state | state-27 | 5.65e-02 | 264.9 | 0.002017 |
| ViT-H+/16 | bf16-cast | hidden-state | state-28 | 5.32e-02 | 223.3 | 0.001701 |
| ViT-H+/16 | bf16-cast | hidden-state | state-29 | 5.00e-02 | 287.6 | 0.002189 |
| ViT-H+/16 | bf16-cast | hidden-state | state-30 | 4.59e-02 | 375.7 | 0.002858 |
| ViT-H+/16 | bf16-cast | hidden-state | state-31 | 2.81e-02 | 346.1 | 0.002637 |
| ViT-H+/16 | bf16-cast | hidden-state | state-32 | 5.63e-03 | 501.4 | 0.003822 |
| ViT-H+/16 | fp16-autocast | stored-embeddings | iss-kibo-lab/224x224 | 2.87e-07 | 0.00395 | 0.0002531 |
| ViT-H+/16 | fp16-autocast | stored-block | block-0 | 9.40e-08 | 0.01823 | 0.0002485 |
| ViT-H+/16 | fp16-autocast | stored-block | block-1 | 1.95e-06 | 0.1151 | 0.0008276 |
| ViT-H+/16 | fp16-autocast | stored-block | block-2 | 7.84e-08 | 0.02095 | 7.901e-05 |
| ViT-H+/16 | fp16-autocast | stored-block | block-3 | 1.37e-08 | 0.01702 | 6.197e-05 |
| ViT-H+/16 | fp16-autocast | stored-block | block-4 | 1.67e-08 | 0.01942 | 7.988e-05 |
| ViT-H+/16 | fp16-autocast | stored-block | block-5 | 1.30e-08 | 0.1229 | 0.0002039 |
| ViT-H+/16 | fp16-autocast | stored-block | block-6 | 1.19e-08 | 0.0275 | 4.492e-05 |
| ViT-H+/16 | fp16-autocast | stored-block | block-7 | 5.32e-09 | 0.01608 | 2.648e-05 |
| ViT-H+/16 | fp16-autocast | stored-block | block-8 | 3.38e-09 | 0.01601 | 2.646e-05 |
| ViT-H+/16 | fp16-autocast | stored-block | block-9 | 5.47e-09 | 0.03691 | 6.725e-05 |
| ViT-H+/16 | fp16-autocast | stored-block | block-10 | 1.63e-08 | 0.03873 | 7.386e-05 |
| ViT-H+/16 | fp16-autocast | stored-block | block-11 | 2.12e-08 | 0.878 | 0.0002131 |
| ViT-H+/16 | fp16-autocast | stored-block | block-12 | 1.40e-08 | 0.06131 | 1.494e-05 |
| ViT-H+/16 | fp16-autocast | stored-block | block-13 | 2.15e-08 | 0.04694 | 1.142e-05 |
| ViT-H+/16 | fp16-autocast | stored-block | block-14 | 2.84e-08 | 0.03979 | 9.645e-06 |
| ViT-H+/16 | fp16-autocast | stored-block | block-15 | 2.97e-08 | 0.05414 | 1.316e-05 |
| ViT-H+/16 | fp16-autocast | stored-block | block-16 | 2.10e-08 | 0.05823 | 1.405e-05 |
| ViT-H+/16 | fp16-autocast | stored-block | block-17 | 2.47e-08 | 0.05792 | 1.404e-05 |
| ViT-H+/16 | fp16-autocast | stored-block | block-18 | 4.37e-08 | 0.07556 | 1.852e-05 |
| ViT-H+/16 | fp16-autocast | stored-block | block-19 | 5.77e-08 | 0.05914 | 1.478e-05 |
| ViT-H+/16 | fp16-autocast | stored-block | block-20 | 5.47e-08 | 0.1261 | 3.257e-05 |
| ViT-H+/16 | fp16-autocast | stored-block | block-21 | 5.82e-08 | 0.1124 | 3.036e-05 |
| ViT-H+/16 | fp16-autocast | stored-block | block-22 | 5.51e-08 | 69.47 | 0.0005298 |
| ViT-H+/16 | fp16-autocast | stored-block | block-23 | 8.04e-08 | 0.4844 | 3.688e-06 |
| ViT-H+/16 | fp16-autocast | stored-block | block-24 | 6.90e-08 | 0.2074 | 1.579e-06 |
| ViT-H+/16 | fp16-autocast | stored-block | block-25 | 6.12e-08 | 2.703 | 2.059e-05 |
| ViT-H+/16 | fp16-autocast | stored-block | block-26 | 6.84e-08 | 1.695 | 1.291e-05 |
| ViT-H+/16 | fp16-autocast | stored-block | block-27 | 1.15e-07 | 0.5547 | 4.225e-06 |
| ViT-H+/16 | fp16-autocast | stored-block | block-28 | 8.23e-08 | 1.117 | 8.505e-06 |
| ViT-H+/16 | fp16-autocast | stored-block | block-29 | 1.93e-07 | 3.883 | 2.954e-05 |
| ViT-H+/16 | fp16-autocast | stored-block | block-30 | 3.18e-07 | 2.276 | 1.734e-05 |
| ViT-H+/16 | fp16-autocast | stored-block | block-31 | 2.75e-07 | 9.208 | 7.02e-05 |
| ViT-H+/16 | fp16-autocast | stored-final-norm | iss-kibo-lab/224x224 | 1.27e-14 | 1.907e-06 | 1.55e-07 |
| ViT-H+/16 | fp16-autocast | hidden-state | state-0 | 2.87e-07 | 0.00395 | 0.0002531 |
| ViT-H+/16 | fp16-autocast | hidden-state | state-1 | 2.59e-07 | 0.01994 | 0.0002718 |
| ViT-H+/16 | fp16-autocast | hidden-state | state-2 | 5.19e-06 | 0.125 | 0.0008983 |
| ViT-H+/16 | fp16-autocast | hidden-state | state-3 | 1.79e-06 | 0.1418 | 0.0005348 |
| ViT-H+/16 | fp16-autocast | hidden-state | state-4 | 1.18e-06 | 0.1427 | 0.0005197 |
| ViT-H+/16 | fp16-autocast | hidden-state | state-5 | 3.81e-06 | 0.1313 | 0.0005399 |
| ViT-H+/16 | fp16-autocast | hidden-state | state-6 | 2.55e-06 | 0.1513 | 0.000251 |
| ViT-H+/16 | fp16-autocast | hidden-state | state-7 | 2.97e-06 | 0.1542 | 0.0002519 |
| ViT-H+/16 | fp16-autocast | hidden-state | state-8 | 4.06e-06 | 0.1534 | 0.0002526 |
| ViT-H+/16 | fp16-autocast | hidden-state | state-9 | 5.29e-06 | 0.1995 | 0.0003299 |
| ViT-H+/16 | fp16-autocast | hidden-state | state-10 | 7.56e-06 | 0.3477 | 0.0006334 |
| ViT-H+/16 | fp16-autocast | hidden-state | state-11 | 9.61e-06 | 0.2551 | 0.0004865 |
| ViT-H+/16 | fp16-autocast | hidden-state | state-12 | 8.06e-06 | 0.6333 | 0.0001537 |
| ViT-H+/16 | fp16-autocast | hidden-state | state-13 | 1.48e-05 | 0.6455 | 0.0001573 |
| ViT-H+/16 | fp16-autocast | hidden-state | state-14 | 4.14e-05 | 1.109 | 0.00027 |
| ViT-H+/16 | fp16-autocast | hidden-state | state-15 | 7.57e-05 | 2.111 | 0.0005116 |
| ViT-H+/16 | fp16-autocast | hidden-state | state-16 | 3.59e-04 | 2.444 | 0.0005942 |
| ViT-H+/16 | fp16-autocast | hidden-state | state-17 | 5.38e-04 | 1.74 | 0.00042 |
| ViT-H+/16 | fp16-autocast | hidden-state | state-18 | 9.01e-04 | 5.778 | 0.0014 |
| ViT-H+/16 | fp16-autocast | hidden-state | state-19 | 1.72e-03 | 11.43 | 0.002801 |
| ViT-H+/16 | fp16-autocast | hidden-state | state-20 | 3.08e-03 | 10.02 | 0.002504 |
| ViT-H+/16 | fp16-autocast | hidden-state | state-21 | 4.01e-03 | 12.18 | 0.003148 |
| ViT-H+/16 | fp16-autocast | hidden-state | state-22 | 3.71e-03 | 10.66 | 0.002879 |
| ViT-H+/16 | fp16-autocast | hidden-state | state-23 | 4.04e-03 | 34.31 | 0.0002617 |
| ViT-H+/16 | fp16-autocast | hidden-state | state-24 | 4.12e-03 | 34.33 | 0.0002614 |
| ViT-H+/16 | fp16-autocast | hidden-state | state-25 | 5.25e-03 | 34.36 | 0.0002617 |
| ViT-H+/16 | fp16-autocast | hidden-state | state-26 | 7.44e-03 | 34.44 | 0.0002623 |
| ViT-H+/16 | fp16-autocast | hidden-state | state-27 | 7.91e-03 | 34.48 | 0.0002626 |
| ViT-H+/16 | fp16-autocast | hidden-state | state-28 | 8.71e-03 | 34.52 | 0.0002629 |
| ViT-H+/16 | fp16-autocast | hidden-state | state-29 | 6.70e-03 | 36.95 | 0.0002813 |
| ViT-H+/16 | fp16-autocast | hidden-state | state-30 | 4.66e-03 | 52.48 | 0.0003992 |
| ViT-H+/16 | fp16-autocast | hidden-state | state-31 | 4.06e-03 | 61.84 | 0.0004711 |
| ViT-H+/16 | fp16-autocast | hidden-state | state-32 | 3.27e-04 | 93.03 | 0.0007093 |
| ViT-H+/16 | bf16-autocast | stored-embeddings | iss-kibo-lab/224x224 | 1.82e-05 | 0.03274 | 0.002098 |
| ViT-H+/16 | bf16-autocast | stored-block | block-0 | 4.28e-06 | 0.1726 | 0.002353 |
| ViT-H+/16 | bf16-autocast | stored-block | block-1 | 1.63e-04 | 1.123 | 0.008075 |
| ViT-H+/16 | bf16-autocast | stored-block | block-2 | 2.47e-06 | 0.2799 | 0.001055 |
| ViT-H+/16 | bf16-autocast | stored-block | block-3 | 1.62e-06 | 0.1269 | 0.000462 |
| ViT-H+/16 | bf16-autocast | stored-block | block-4 | 1.27e-06 | 0.2823 | 0.001161 |
| ViT-H+/16 | bf16-autocast | stored-block | block-5 | 1.05e-06 | 0.2451 | 0.0004065 |
| ViT-H+/16 | bf16-autocast | stored-block | block-6 | 1.51e-06 | 0.2411 | 0.0003938 |
| ViT-H+/16 | bf16-autocast | stored-block | block-7 | 3.20e-07 | 0.112 | 0.0001845 |
| ViT-H+/16 | bf16-autocast | stored-block | block-8 | 2.52e-07 | 0.1681 | 0.0002779 |
| ViT-H+/16 | bf16-autocast | stored-block | block-9 | 3.35e-07 | 0.2204 | 0.0004015 |
| ViT-H+/16 | bf16-autocast | stored-block | block-10 | 5.33e-07 | 0.2597 | 0.0004953 |
| ViT-H+/16 | bf16-autocast | stored-block | block-11 | 7.46e-07 | 16.42 | 0.003986 |
| ViT-H+/16 | bf16-autocast | stored-block | block-12 | 1.21e-06 | 0.434 | 0.0001057 |
| ViT-H+/16 | bf16-autocast | stored-block | block-13 | 2.14e-06 | 1.936 | 0.000471 |
| ViT-H+/16 | bf16-autocast | stored-block | block-14 | 1.36e-06 | 0.3879 | 9.401e-05 |
| ViT-H+/16 | bf16-autocast | stored-block | block-15 | 1.23e-06 | 0.5246 | 0.0001275 |
| ViT-H+/16 | bf16-autocast | stored-block | block-16 | 9.37e-07 | 0.445 | 0.0001074 |
| ViT-H+/16 | bf16-autocast | stored-block | block-17 | 1.57e-06 | 0.5188 | 0.0001257 |
| ViT-H+/16 | bf16-autocast | stored-block | block-18 | 2.00e-06 | 0.524 | 0.0001284 |
| ViT-H+/16 | bf16-autocast | stored-block | block-19 | 2.68e-06 | 0.5558 | 0.0001389 |
| ViT-H+/16 | bf16-autocast | stored-block | block-20 | 3.06e-06 | 0.8346 | 0.0002156 |
| ViT-H+/16 | bf16-autocast | stored-block | block-21 | 2.78e-06 | 0.8237 | 0.0002225 |
| ViT-H+/16 | bf16-autocast | stored-block | block-22 | 2.86e-06 | 149.1 | 0.001137 |
| ViT-H+/16 | bf16-autocast | stored-block | block-23 | 4.04e-06 | 10.06 | 7.662e-05 |
| ViT-H+/16 | bf16-autocast | stored-block | block-24 | 3.98e-06 | 1.531 | 1.166e-05 |
| ViT-H+/16 | bf16-autocast | stored-block | block-25 | 4.17e-06 | 10.62 | 8.093e-05 |
| ViT-H+/16 | bf16-autocast | stored-block | block-26 | 3.82e-06 | 12.35 | 9.404e-05 |
| ViT-H+/16 | bf16-autocast | stored-block | block-27 | 5.20e-06 | 13.83 | 0.0001053 |
| ViT-H+/16 | bf16-autocast | stored-block | block-28 | 5.84e-06 | 4.055 | 3.087e-05 |
| ViT-H+/16 | bf16-autocast | stored-block | block-29 | 6.52e-06 | 35.19 | 0.0002677 |
| ViT-H+/16 | bf16-autocast | stored-block | block-30 | 1.21e-05 | 14.08 | 0.0001073 |
| ViT-H+/16 | bf16-autocast | stored-block | block-31 | 1.23e-05 | 51.86 | 0.0003954 |
| ViT-H+/16 | bf16-autocast | stored-final-norm | iss-kibo-lab/224x224 | 1.27e-14 | 1.907e-06 | 1.55e-07 |
| ViT-H+/16 | bf16-autocast | hidden-state | state-0 | 1.82e-05 | 0.03274 | 0.002098 |
| ViT-H+/16 | bf16-autocast | hidden-state | state-1 | 1.43e-05 | 0.404 | 0.005507 |
| ViT-H+/16 | bf16-autocast | hidden-state | state-2 | 2.29e-04 | 1.2 | 0.008627 |
| ViT-H+/16 | bf16-autocast | hidden-state | state-3 | 7.07e-05 | 0.6212 | 0.002342 |
| ViT-H+/16 | bf16-autocast | hidden-state | state-4 | 1.05e-04 | 0.5347 | 0.001947 |
| ViT-H+/16 | bf16-autocast | hidden-state | state-5 | 2.88e-04 | 0.7055 | 0.002901 |
| ViT-H+/16 | bf16-autocast | hidden-state | state-6 | 2.88e-04 | 1.835 | 0.003044 |
| ViT-H+/16 | bf16-autocast | hidden-state | state-7 | 3.30e-04 | 1.875 | 0.003062 |
| ViT-H+/16 | bf16-autocast | hidden-state | state-8 | 3.46e-04 | 1.588 | 0.002615 |
| ViT-H+/16 | bf16-autocast | hidden-state | state-9 | 3.70e-04 | 2.477 | 0.004096 |
| ViT-H+/16 | bf16-autocast | hidden-state | state-10 | 4.07e-04 | 2.714 | 0.004944 |
| ViT-H+/16 | bf16-autocast | hidden-state | state-11 | 4.54e-04 | 1.663 | 0.003171 |
| ViT-H+/16 | bf16-autocast | hidden-state | state-12 | 3.91e-04 | 16 | 0.003883 |
| ViT-H+/16 | bf16-autocast | hidden-state | state-13 | 7.68e-04 | 15.51 | 0.00378 |
| ViT-H+/16 | bf16-autocast | hidden-state | state-14 | 1.75e-03 | 14.69 | 0.003574 |
| ViT-H+/16 | bf16-autocast | hidden-state | state-15 | 2.33e-03 | 14.04 | 0.003403 |
| ViT-H+/16 | bf16-autocast | hidden-state | state-16 | 3.38e-03 | 13.46 | 0.003271 |
| ViT-H+/16 | bf16-autocast | hidden-state | state-17 | 4.45e-03 | 12.02 | 0.002901 |
| ViT-H+/16 | bf16-autocast | hidden-state | state-18 | 5.93e-03 | 11.66 | 0.002824 |
| ViT-H+/16 | bf16-autocast | hidden-state | state-19 | 1.07e-02 | 18.18 | 0.004454 |
| ViT-H+/16 | bf16-autocast | hidden-state | state-20 | 2.01e-02 | 16.75 | 0.004185 |
| ViT-H+/16 | bf16-autocast | hidden-state | state-21 | 2.29e-02 | 22.22 | 0.00574 |
| ViT-H+/16 | bf16-autocast | hidden-state | state-22 | 3.32e-02 | 25.88 | 0.006989 |
| ViT-H+/16 | bf16-autocast | hidden-state | state-23 | 3.94e-02 | 151.8 | 0.001157 |
| ViT-H+/16 | bf16-autocast | hidden-state | state-24 | 4.70e-02 | 155.3 | 0.001182 |
| ViT-H+/16 | bf16-autocast | hidden-state | state-25 | 5.06e-02 | 154 | 0.001173 |
| ViT-H+/16 | bf16-autocast | hidden-state | state-26 | 6.34e-02 | 145.1 | 0.001105 |
| ViT-H+/16 | bf16-autocast | hidden-state | state-27 | 6.73e-02 | 161.1 | 0.001227 |
| ViT-H+/16 | bf16-autocast | hidden-state | state-28 | 8.61e-02 | 175 | 0.001333 |
| ViT-H+/16 | bf16-autocast | hidden-state | state-29 | 8.78e-02 | 163.9 | 0.001248 |
| ViT-H+/16 | bf16-autocast | hidden-state | state-30 | 8.56e-02 | 206.9 | 0.001574 |
| ViT-H+/16 | bf16-autocast | hidden-state | state-31 | 5.52e-02 | 344.8 | 0.002627 |
| ViT-H+/16 | bf16-autocast | hidden-state | state-32 | 1.05e-02 | 852.1 | 0.006496 |
| ViT-7B/16 | fp16-cast | stored-embeddings | iss-kibo-lab/224x224 | 9.37e-08 | 0.03497 | 0.0002942 |
| ViT-7B/16 | fp16-cast | stored-block | block-0 | 2.72e-07 | 2.071 | 0.0007107 |
| ViT-7B/16 | fp16-cast | stored-block | block-1 | 1.91e-07 | 0.4771 | 0.000119 |
| ViT-7B/16 | fp16-cast | stored-block | block-2 | 1.73e-07 | 2.266 | 0.0004734 |
| ViT-7B/16 | fp16-cast | stored-block | block-3 | 1.76e-07 | 2.963 | 0.0005511 |
| ViT-7B/16 | fp16-cast | stored-block | block-4 | 1.09e-07 | 0.2476 | 4.242e-05 |
| ViT-7B/16 | fp16-cast | stored-block | block-5 | 1.40e-07 | 0.5453 | 8.778e-05 |
| ViT-7B/16 | fp16-cast | stored-block | block-6 | 1.13e-07 | 1.507 | 0.0002312 |
| ViT-7B/16 | fp16-cast | stored-block | block-7 | 1.01e-07 | 1.792 | 0.0002664 |
| ViT-7B/16 | fp16-cast | stored-block | block-8 | 7.70e-08 | 0.7539 | 0.0001093 |
| ViT-7B/16 | fp16-cast | stored-block | block-9 | 8.57e-08 | 0.6201 | 8.873e-05 |
| ViT-7B/16 | fp16-cast | stored-block | block-10 | 7.71e-08 | 2.023 | 0.0002839 |
| ViT-7B/16 | fp16-cast | stored-block | block-11 | 5.19e-08 | 2.338 | 0.0003232 |
| ViT-7B/16 | fp16-cast | stored-block | block-12 | 7.72e-08 | 0.5007 | 6.845e-05 |
| ViT-7B/16 | fp16-cast | stored-block | block-13 | 9.65e-08 | 2.777 | 0.000376 |
| ViT-7B/16 | fp16-cast | stored-block | block-14 | 3.89e-08 | 0.254 | 3.411e-05 |
| ViT-7B/16 | fp16-cast | stored-block | block-15 | 3.27e-08 | 2 | 0.0002665 |
| ViT-7B/16 | fp16-cast | stored-block | block-16 | 4.96e-08 | 1.263 | 0.0001674 |
| ViT-7B/16 | fp16-cast | stored-block | block-17 | 3.94e-08 | 2.854 | 0.0003756 |
| ViT-7B/16 | fp16-cast | stored-block | block-18 | 5.23e-08 | 1.446 | 0.0001891 |
| ViT-7B/16 | fp16-cast | stored-block | block-19 | 7.08e-08 | 1.538 | 0.0001994 |
| ViT-7B/16 | fp16-cast | stored-block | block-20 | 5.95e-08 | 2.033 | 0.0002617 |
| ViT-7B/16 | fp16-cast | stored-block | block-21 | 5.02e-08 | 3.207 | 0.0004101 |
| ViT-7B/16 | fp16-cast | stored-block | block-22 | 9.10e-08 | 0.6606 | 8.372e-05 |
| ViT-7B/16 | fp16-cast | stored-block | block-23 | 6.10e-08 | 1.397 | 0.0001754 |
| ViT-7B/16 | fp16-cast | stored-block | block-24 | 1.84e-07 | 2.215 | 0.0002747 |
| ViT-7B/16 | fp16-cast | stored-block | block-25 | 1.36e-07 | 0.7266 | 8.881e-05 |
| ViT-7B/16 | fp16-cast | stored-block | block-26 | 1.19e-07 | 2.259 | 0.0002722 |
| ViT-7B/16 | fp16-cast | stored-block | block-27 | 1.70e-07 | 2.063 | 0.0002454 |
| ViT-7B/16 | fp16-cast | stored-block | block-28 | 1.99e-07 | 5.484 | 0.0006397 |
| ViT-7B/16 | fp16-cast | stored-block | block-29 | 2.87e-07 | 1.185 | 0.0001351 |
| ViT-7B/16 | fp16-cast | stored-block | block-30 | 1.58e-07 | 6.41 | 0.0007073 |
| ViT-7B/16 | fp16-cast | stored-block | block-31 | 2.20e-07 | 0.9362 | 9.985e-05 |
| ViT-7B/16 | fp16-cast | stored-block | block-32 | 3.00e-07 | 1.014 | 0.0001043 |
| ViT-7B/16 | fp16-cast | stored-block | block-33 | 2.59e-07 | 4.186 | 0.0004131 |
| ViT-7B/16 | fp16-cast | stored-block | block-34 | 2.33e-07 | 6.488 | 0.0006139 |
| ViT-7B/16 | fp16-cast | stored-block | block-35 | 2.38e-07 | 4.159 | 0.0003766 |
| ViT-7B/16 | fp16-cast | stored-block | block-36 | 2.10e-07 | 2.826 | 0.0002435 |
| ViT-7B/16 | fp16-cast | stored-block | block-37 | 2.34e-07 | 1.818 | 0.0001491 |
| ViT-7B/16 | fp16-cast | stored-block | block-38 | 1.66e-07 | 2.111 | 0.0001643 |
| ViT-7B/16 | fp16-cast | stored-block | block-39 | 1.85e-07 | 6.577 | 0.0004858 |
| ViT-7B/16 | fp16-cast | stored-final-norm | iss-kibo-lab/224x224 | 1.31e-07 | 0.003964 | 0.0005978 |
| ViT-7B/16 | fp16-cast | hidden-state | state-0 | 9.37e-08 | 0.03497 | 0.0002942 |
| ViT-7B/16 | fp16-cast | hidden-state | state-1 | 4.45e-07 | 2.071 | 0.0007107 |
| ViT-7B/16 | fp16-cast | hidden-state | state-2 | 5.57e-07 | 3.523 | 0.0008789 |
| ViT-7B/16 | fp16-cast | hidden-state | state-3 | 8.05e-07 | 1.734 | 0.0003624 |
| ViT-7B/16 | fp16-cast | hidden-state | state-4 | 1.05e-06 | 2.963 | 0.0005511 |
| ViT-7B/16 | fp16-cast | hidden-state | state-5 | 1.22e-06 | 4.159 | 0.0007126 |
| ViT-7B/16 | fp16-cast | hidden-state | state-6 | 1.83e-06 | 3.706 | 0.0005965 |
| ViT-7B/16 | fp16-cast | hidden-state | state-7 | 2.42e-06 | 2.493 | 0.0003825 |
| ViT-7B/16 | fp16-cast | hidden-state | state-8 | 3.66e-06 | 5.792 | 0.0008611 |
| ViT-7B/16 | fp16-cast | hidden-state | state-9 | 4.14e-06 | 4.754 | 0.000689 |
| ViT-7B/16 | fp16-cast | hidden-state | state-10 | 5.87e-06 | 3.38 | 0.0004836 |
| ViT-7B/16 | fp16-cast | hidden-state | state-11 | 5.46e-06 | 1.977 | 0.0002774 |
| ViT-7B/16 | fp16-cast | hidden-state | state-12 | 5.60e-06 | 2.338 | 0.0003232 |
| ViT-7B/16 | fp16-cast | hidden-state | state-13 | 5.88e-06 | 4.351 | 0.0005947 |
| ViT-7B/16 | fp16-cast | hidden-state | state-14 | 7.32e-06 | 6.777 | 0.0009176 |
| ViT-7B/16 | fp16-cast | hidden-state | state-15 | 6.76e-06 | 8.005 | 0.001075 |
| ViT-7B/16 | fp16-cast | hidden-state | state-16 | 7.09e-06 | 6 | 0.0007999 |
| ViT-7B/16 | fp16-cast | hidden-state | state-17 | 7.85e-06 | 6.737 | 0.0008929 |
| ViT-7B/16 | fp16-cast | hidden-state | state-18 | 1.15e-05 | 5.146 | 0.0006772 |
| ViT-7B/16 | fp16-cast | hidden-state | state-19 | 2.48e-05 | 5.446 | 0.0007122 |
| ViT-7B/16 | fp16-cast | hidden-state | state-20 | 4.62e-05 | 5.538 | 0.0007179 |
| ViT-7B/16 | fp16-cast | hidden-state | state-21 | 5.67e-05 | 6.033 | 0.0007768 |
| ViT-7B/16 | fp16-cast | hidden-state | state-22 | 3.92e-05 | 4.793 | 0.000613 |
| ViT-7B/16 | fp16-cast | hidden-state | state-23 | 6.00e-05 | 4.661 | 0.0005906 |
| ViT-7B/16 | fp16-cast | hidden-state | state-24 | 8.94e-05 | 5.397 | 0.0006775 |
| ViT-7B/16 | fp16-cast | hidden-state | state-25 | 1.21e-04 | 6.215 | 0.0007709 |
| ViT-7B/16 | fp16-cast | hidden-state | state-26 | 1.68e-04 | 7.273 | 0.0008891 |
| ViT-7B/16 | fp16-cast | hidden-state | state-27 | 1.98e-04 | 5.741 | 0.0006919 |
| ViT-7B/16 | fp16-cast | hidden-state | state-28 | 2.94e-04 | 5.937 | 0.0007059 |
| ViT-7B/16 | fp16-cast | hidden-state | state-29 | 4.35e-04 | 3.414 | 0.0003982 |
| ViT-7B/16 | fp16-cast | hidden-state | state-30 | 6.15e-04 | 5.565 | 0.0006346 |
| ViT-7B/16 | fp16-cast | hidden-state | state-31 | 8.03e-04 | 10.77 | 0.001188 |
| ViT-7B/16 | fp16-cast | hidden-state | state-32 | 9.43e-04 | 17.89 | 0.001908 |
| ViT-7B/16 | fp16-cast | hidden-state | state-33 | 9.48e-04 | 24.54 | 0.002522 |
| ViT-7B/16 | fp16-cast | hidden-state | state-34 | 7.56e-04 | 35.26 | 0.00348 |
| ViT-7B/16 | fp16-cast | hidden-state | state-35 | 7.83e-04 | 44.28 | 0.004189 |
| ViT-7B/16 | fp16-cast | hidden-state | state-36 | 7.41e-04 | 52.87 | 0.004787 |
| ViT-7B/16 | fp16-cast | hidden-state | state-37 | 7.96e-04 | 62.61 | 0.005395 |
| ViT-7B/16 | fp16-cast | hidden-state | state-38 | 6.71e-04 | 77.67 | 0.006371 |
| ViT-7B/16 | fp16-cast | hidden-state | state-39 | 6.45e-04 | 97.89 | 0.007615 |
| ViT-7B/16 | fp16-cast | hidden-state | state-40 | 1.54e-04 | 90.29 | 0.00667 |
| ViT-7B/16 | bf16-cast | stored-embeddings | iss-kibo-lab/224x224 | 5.70e-06 | 0.2338 | 0.001967 |
| ViT-7B/16 | bf16-cast | stored-block | block-0 | 2.04e-05 | 14.07 | 0.004829 |
| ViT-7B/16 | bf16-cast | stored-block | block-1 | 1.19e-05 | 7.523 | 0.001877 |
| ViT-7B/16 | bf16-cast | stored-block | block-2 | 1.05e-05 | 13.73 | 0.00287 |
| ViT-7B/16 | bf16-cast | stored-block | block-3 | 1.06e-05 | 1.927 | 0.0003585 |
| ViT-7B/16 | bf16-cast | stored-block | block-4 | 9.86e-06 | 11.84 | 0.002029 |
| ViT-7B/16 | bf16-cast | stored-block | block-5 | 9.70e-06 | 4.294 | 0.0006913 |
| ViT-7B/16 | bf16-cast | stored-block | block-6 | 5.83e-06 | 10.49 | 0.00161 |
| ViT-7B/16 | bf16-cast | stored-block | block-7 | 7.73e-06 | 6.208 | 0.000923 |
| ViT-7B/16 | bf16-cast | stored-block | block-8 | 1.03e-05 | 19.25 | 0.00279 |
| ViT-7B/16 | bf16-cast | stored-block | block-9 | 5.73e-06 | 19.38 | 0.002773 |
| ViT-7B/16 | bf16-cast | stored-block | block-10 | 3.29e-06 | 22.02 | 0.00309 |
| ViT-7B/16 | bf16-cast | stored-block | block-11 | 2.74e-06 | 2.923 | 0.0004041 |
| ViT-7B/16 | bf16-cast | stored-block | block-12 | 2.96e-06 | 19.65 | 0.002686 |
| ViT-7B/16 | bf16-cast | stored-block | block-13 | 3.16e-06 | 6.777 | 0.0009176 |
| ViT-7B/16 | bf16-cast | stored-block | block-14 | 2.19e-06 | 8.005 | 0.001075 |
| ViT-7B/16 | bf16-cast | stored-block | block-15 | 3.29e-06 | 18 | 0.002399 |
| ViT-7B/16 | bf16-cast | stored-block | block-16 | 5.68e-06 | 9.271 | 0.001229 |
| ViT-7B/16 | bf16-cast | stored-block | block-17 | 3.69e-06 | 17.15 | 0.002256 |
| ViT-7B/16 | bf16-cast | stored-block | block-18 | 3.08e-06 | 3.066 | 0.000401 |
| ViT-7B/16 | bf16-cast | stored-block | block-19 | 2.80e-06 | 2.462 | 0.0003191 |
| ViT-7B/16 | bf16-cast | stored-block | block-20 | 5.58e-06 | 10.03 | 0.001292 |
| ViT-7B/16 | bf16-cast | stored-block | block-21 | 5.97e-06 | 11.21 | 0.001433 |
| ViT-7B/16 | bf16-cast | stored-block | block-22 | 3.83e-06 | 19.34 | 0.002451 |
| ViT-7B/16 | bf16-cast | stored-block | block-23 | 6.42e-06 | 7.501 | 0.0009416 |
| ViT-7B/16 | bf16-cast | stored-block | block-24 | 5.46e-06 | 3.241 | 0.000402 |
| ViT-7B/16 | bf16-cast | stored-block | block-25 | 7.60e-06 | 20.73 | 0.002534 |
| ViT-7B/16 | bf16-cast | stored-block | block-26 | 7.29e-06 | 21.74 | 0.00262 |
| ViT-7B/16 | bf16-cast | stored-block | block-27 | 1.30e-05 | 37.94 | 0.004511 |
| ViT-7B/16 | bf16-cast | stored-block | block-28 | 1.07e-05 | 61.48 | 0.007171 |
| ViT-7B/16 | bf16-cast | stored-block | block-29 | 1.02e-05 | 4.145 | 0.0004727 |
| ViT-7B/16 | bf16-cast | stored-block | block-30 | 2.51e-05 | 38.41 | 0.004238 |
| ViT-7B/16 | bf16-cast | stored-block | block-31 | 1.12e-05 | 32.25 | 0.00344 |
| ViT-7B/16 | bf16-cast | stored-block | block-32 | 1.53e-05 | 7.64 | 0.0007853 |
| ViT-7B/16 | bf16-cast | stored-block | block-33 | 1.96e-05 | 20.19 | 0.001992 |
| ViT-7B/16 | bf16-cast | stored-block | block-34 | 1.73e-05 | 9.512 | 0.0008999 |
| ViT-7B/16 | bf16-cast | stored-block | block-35 | 1.29e-05 | 36.16 | 0.003274 |
| ViT-7B/16 | bf16-cast | stored-block | block-36 | 1.57e-05 | 42.83 | 0.00369 |
| ViT-7B/16 | bf16-cast | stored-block | block-37 | 1.08e-05 | 30.64 | 0.002513 |
| ViT-7B/16 | bf16-cast | stored-block | block-38 | 1.31e-05 | 54.17 | 0.004214 |
| ViT-7B/16 | bf16-cast | stored-block | block-39 | 1.84e-05 | 52.01 | 0.003842 |
| ViT-7B/16 | bf16-cast | stored-final-norm | iss-kibo-lab/224x224 | 7.87e-06 | 0.02809 | 0.004237 |
| ViT-7B/16 | bf16-cast | hidden-state | state-0 | 5.70e-06 | 0.2338 | 0.001967 |
| ViT-7B/16 | bf16-cast | hidden-state | state-1 | 2.29e-05 | 14.07 | 0.004829 |
| ViT-7B/16 | bf16-cast | hidden-state | state-2 | 4.01e-05 | 23.52 | 0.005868 |
| ViT-7B/16 | bf16-cast | hidden-state | state-3 | 4.83e-05 | 13.73 | 0.00287 |
| ViT-7B/16 | bf16-cast | hidden-state | state-4 | 6.20e-05 | 5.927 | 0.001102 |
| ViT-7B/16 | bf16-cast | hidden-state | state-5 | 1.01e-04 | 11.84 | 0.002029 |
| ViT-7B/16 | bf16-cast | hidden-state | state-6 | 1.26e-04 | 10.98 | 0.001767 |
| ViT-7B/16 | bf16-cast | hidden-state | state-7 | 1.64e-04 | 10.49 | 0.00161 |
| ViT-7B/16 | bf16-cast | hidden-state | state-8 | 2.59e-04 | 8.97 | 0.001334 |
| ViT-7B/16 | bf16-cast | hidden-state | state-9 | 2.97e-04 | 19.25 | 0.00279 |
| ViT-7B/16 | bf16-cast | hidden-state | state-10 | 3.81e-04 | 12.62 | 0.001806 |
| ViT-7B/16 | bf16-cast | hidden-state | state-11 | 3.99e-04 | 22.02 | 0.00309 |
| ViT-7B/16 | bf16-cast | hidden-state | state-12 | 4.69e-04 | 33.66 | 0.004654 |
| ViT-7B/16 | bf16-cast | hidden-state | state-13 | 5.73e-04 | 51.65 | 0.00706 |
| ViT-7B/16 | bf16-cast | hidden-state | state-14 | 5.92e-04 | 57.22 | 0.007748 |
| ViT-7B/16 | bf16-cast | hidden-state | state-15 | 5.54e-04 | 55.99 | 0.007518 |
| ViT-7B/16 | bf16-cast | hidden-state | state-16 | 1.17e-03 | 46 | 0.006132 |
| ViT-7B/16 | bf16-cast | hidden-state | state-17 | 1.58e-03 | 25.26 | 0.003348 |
| ViT-7B/16 | bf16-cast | hidden-state | state-18 | 2.09e-03 | 14.85 | 0.001955 |
| ViT-7B/16 | bf16-cast | hidden-state | state-19 | 3.09e-03 | 9.63 | 0.001259 |
| ViT-7B/16 | bf16-cast | hidden-state | state-20 | 3.04e-03 | 11.17 | 0.001448 |
| ViT-7B/16 | bf16-cast | hidden-state | state-21 | 4.26e-03 | 12.89 | 0.00166 |
| ViT-7B/16 | bf16-cast | hidden-state | state-22 | 6.70e-03 | 17.82 | 0.002279 |
| ViT-7B/16 | bf16-cast | hidden-state | state-23 | 9.64e-03 | 19.57 | 0.002479 |
| ViT-7B/16 | bf16-cast | hidden-state | state-24 | 1.20e-02 | 18.43 | 0.002313 |
| ViT-7B/16 | bf16-cast | hidden-state | state-25 | 1.48e-02 | 18.41 | 0.002283 |
| ViT-7B/16 | bf16-cast | hidden-state | state-26 | 2.31e-02 | 28.39 | 0.00347 |
| ViT-7B/16 | bf16-cast | hidden-state | state-27 | 2.70e-02 | 29.08 | 0.003505 |
| ViT-7B/16 | bf16-cast | hidden-state | state-28 | 3.65e-02 | 40.99 | 0.004874 |
| ViT-7B/16 | bf16-cast | hidden-state | state-29 | 4.80e-02 | 42.92 | 0.005007 |
| ViT-7B/16 | bf16-cast | hidden-state | state-30 | 5.50e-02 | 46.47 | 0.005299 |
| ViT-7B/16 | bf16-cast | hidden-state | state-31 | 6.45e-02 | 62.74 | 0.006924 |
| ViT-7B/16 | bf16-cast | hidden-state | state-32 | 6.29e-02 | 81.95 | 0.00874 |
| ViT-7B/16 | bf16-cast | hidden-state | state-33 | 7.15e-02 | 168.3 | 0.0173 |
| ViT-7B/16 | bf16-cast | hidden-state | state-34 | 7.25e-02 | 203.7 | 0.0201 |
| ViT-7B/16 | bf16-cast | hidden-state | state-35 | 8.02e-02 | 284 | 0.02687 |
| ViT-7B/16 | bf16-cast | hidden-state | state-36 | 6.80e-02 | 269.5 | 0.0244 |
| ViT-7B/16 | bf16-cast | hidden-state | state-37 | 5.28e-02 | 321.1 | 0.02767 |
| ViT-7B/16 | bf16-cast | hidden-state | state-38 | 5.97e-02 | 325.3 | 0.02668 |
| ViT-7B/16 | bf16-cast | hidden-state | state-39 | 7.78e-02 | 441.9 | 0.03438 |
| ViT-7B/16 | bf16-cast | hidden-state | state-40 | 1.25e-02 | 708.4 | 0.05233 |
| ViT-7B/16 | fp16-autocast | stored-embeddings | iss-kibo-lab/224x224 | 9.37e-08 | 0.03497 | 0.0002942 |
| ViT-7B/16 | fp16-autocast | stored-block | block-0 | 4.32e-08 | 0.353 | 0.0001212 |
| ViT-7B/16 | fp16-autocast | stored-block | block-1 | 1.63e-08 | 0.04053 | 1.011e-05 |
| ViT-7B/16 | fp16-autocast | stored-block | block-2 | 8.04e-09 | 0.06641 | 1.387e-05 |
| ViT-7B/16 | fp16-autocast | stored-block | block-3 | 6.76e-09 | 0.271 | 5.04e-05 |
| ViT-7B/16 | fp16-autocast | stored-block | block-4 | 3.28e-09 | 0.2578 | 4.418e-05 |
| ViT-7B/16 | fp16-autocast | stored-block | block-5 | 2.98e-09 | 0.1279 | 2.059e-05 |
| ViT-7B/16 | fp16-autocast | stored-block | block-6 | 5.32e-09 | 0.09277 | 1.423e-05 |
| ViT-7B/16 | fp16-autocast | stored-block | block-7 | 4.19e-09 | 0.127 | 1.887e-05 |
| ViT-7B/16 | fp16-autocast | stored-block | block-8 | 2.69e-09 | 0.05664 | 8.21e-06 |
| ViT-7B/16 | fp16-autocast | stored-block | block-9 | 2.16e-09 | 0.07275 | 1.041e-05 |
| ViT-7B/16 | fp16-autocast | stored-block | block-10 | 3.12e-09 | 0.06689 | 9.387e-06 |
| ViT-7B/16 | fp16-autocast | stored-block | block-11 | 2.03e-09 | 0.06152 | 8.505e-06 |
| ViT-7B/16 | fp16-autocast | stored-block | block-12 | 4.59e-09 | 0.06885 | 9.411e-06 |
| ViT-7B/16 | fp16-autocast | stored-block | block-13 | 1.84e-09 | 0.04004 | 5.422e-06 |
| ViT-7B/16 | fp16-autocast | stored-block | block-14 | 2.64e-09 | 0.01306 | 1.754e-06 |
| ViT-7B/16 | fp16-autocast | stored-block | block-15 | 2.87e-09 | 0.05566 | 7.42e-06 |
| ViT-7B/16 | fp16-autocast | stored-block | block-16 | 2.73e-09 | 0.02393 | 3.171e-06 |
| ViT-7B/16 | fp16-autocast | stored-block | block-17 | 2.84e-09 | 0.02341 | 3.08e-06 |
| ViT-7B/16 | fp16-autocast | stored-block | block-18 | 5.22e-09 | 0.07812 | 1.022e-05 |
| ViT-7B/16 | fp16-autocast | stored-block | block-19 | 4.77e-09 | 0.08203 | 1.063e-05 |
| ViT-7B/16 | fp16-autocast | stored-block | block-20 | 4.99e-09 | 0.03262 | 4.201e-06 |
| ViT-7B/16 | fp16-autocast | stored-block | block-21 | 5.40e-09 | 0.04178 | 5.343e-06 |
| ViT-7B/16 | fp16-autocast | stored-block | block-22 | 5.49e-09 | 0.03265 | 4.138e-06 |
| ViT-7B/16 | fp16-autocast | stored-block | block-23 | 8.02e-09 | 0.04004 | 5.026e-06 |
| ViT-7B/16 | fp16-autocast | stored-block | block-24 | 7.74e-09 | 0.0376 | 4.664e-06 |
| ViT-7B/16 | fp16-autocast | stored-block | block-25 | 9.88e-09 | 0.07318 | 8.946e-06 |
| ViT-7B/16 | fp16-autocast | stored-block | block-26 | 1.10e-08 | 0.0625 | 7.532e-06 |
| ViT-7B/16 | fp16-autocast | stored-block | block-27 | 1.22e-08 | 0.06494 | 7.722e-06 |
| ViT-7B/16 | fp16-autocast | stored-block | block-28 | 1.83e-08 | 0.07874 | 9.184e-06 |
| ViT-7B/16 | fp16-autocast | stored-block | block-29 | 1.62e-08 | 0.07904 | 9.013e-06 |
| ViT-7B/16 | fp16-autocast | stored-block | block-30 | 1.75e-08 | 0.1171 | 1.292e-05 |
| ViT-7B/16 | fp16-autocast | stored-block | block-31 | 2.30e-08 | 0.1507 | 1.607e-05 |
| ViT-7B/16 | fp16-autocast | stored-block | block-32 | 2.58e-08 | 0.1439 | 1.479e-05 |
| ViT-7B/16 | fp16-autocast | stored-block | block-33 | 3.39e-08 | 0.1724 | 1.701e-05 |
| ViT-7B/16 | fp16-autocast | stored-block | block-34 | 4.21e-08 | 0.204 | 1.93e-05 |
| ViT-7B/16 | fp16-autocast | stored-block | block-35 | 6.35e-08 | 0.3218 | 2.914e-05 |
| ViT-7B/16 | fp16-autocast | stored-block | block-36 | 6.94e-08 | 0.4621 | 3.982e-05 |
| ViT-7B/16 | fp16-autocast | stored-block | block-37 | 8.53e-08 | 0.4012 | 3.291e-05 |
| ViT-7B/16 | fp16-autocast | stored-block | block-38 | 7.58e-08 | 0.5065 | 3.94e-05 |
| ViT-7B/16 | fp16-autocast | stored-block | block-39 | 3.27e-08 | 2.385 | 0.0001762 |
| ViT-7B/16 | fp16-autocast | stored-final-norm | iss-kibo-lab/224x224 | 8.44e-15 | 1.431e-06 | 2.157e-07 |
| ViT-7B/16 | fp16-autocast | hidden-state | state-0 | 9.37e-08 | 0.03497 | 0.0002942 |
| ViT-7B/16 | fp16-autocast | hidden-state | state-1 | 3.14e-07 | 0.353 | 0.0001212 |
| ViT-7B/16 | fp16-autocast | hidden-state | state-2 | 4.48e-07 | 0.3857 | 9.623e-05 |
| ViT-7B/16 | fp16-autocast | hidden-state | state-3 | 6.11e-07 | 0.3062 | 6.396e-05 |
| ViT-7B/16 | fp16-autocast | hidden-state | state-4 | 7.18e-07 | 0.5894 | 0.0001096 |
| ViT-7B/16 | fp16-autocast | hidden-state | state-5 | 1.07e-06 | 0.8442 | 0.0001447 |
| ViT-7B/16 | fp16-autocast | hidden-state | state-6 | 1.65e-06 | 0.7163 | 0.0001153 |
| ViT-7B/16 | fp16-autocast | hidden-state | state-7 | 2.31e-06 | 0.8091 | 0.0001241 |
| ViT-7B/16 | fp16-autocast | hidden-state | state-8 | 3.42e-06 | 1.062 | 0.0001579 |
| ViT-7B/16 | fp16-autocast | hidden-state | state-9 | 4.10e-06 | 1.147 | 0.0001663 |
| ViT-7B/16 | fp16-autocast | hidden-state | state-10 | 5.40e-06 | 1.262 | 0.0001805 |
| ViT-7B/16 | fp16-autocast | hidden-state | state-11 | 5.18e-06 | 1.329 | 0.0001864 |
| ViT-7B/16 | fp16-autocast | hidden-state | state-12 | 5.68e-06 | 1.214 | 0.0001678 |
| ViT-7B/16 | fp16-autocast | hidden-state | state-13 | 5.83e-06 | 1.225 | 0.0001674 |
| ViT-7B/16 | fp16-autocast | hidden-state | state-14 | 7.59e-06 | 1.265 | 0.0001712 |
| ViT-7B/16 | fp16-autocast | hidden-state | state-15 | 7.07e-06 | 1.274 | 0.0001711 |
| ViT-7B/16 | fp16-autocast | hidden-state | state-16 | 8.44e-06 | 1.26 | 0.0001679 |
| ViT-7B/16 | fp16-autocast | hidden-state | state-17 | 8.37e-06 | 1.22 | 0.0001617 |
| ViT-7B/16 | fp16-autocast | hidden-state | state-18 | 1.22e-05 | 1.235 | 0.0001625 |
| ViT-7B/16 | fp16-autocast | hidden-state | state-19 | 2.64e-05 | 1.294 | 0.0001693 |
| ViT-7B/16 | fp16-autocast | hidden-state | state-20 | 4.83e-05 | 1.302 | 0.0001687 |
| ViT-7B/16 | fp16-autocast | hidden-state | state-21 | 5.76e-05 | 3.193 | 0.0004111 |
| ViT-7B/16 | fp16-autocast | hidden-state | state-22 | 7.60e-05 | 1.898 | 0.0002427 |
| ViT-7B/16 | fp16-autocast | hidden-state | state-23 | 1.19e-04 | 2.306 | 0.0002922 |
| ViT-7B/16 | fp16-autocast | hidden-state | state-24 | 1.82e-04 | 3.425 | 0.0004299 |
| ViT-7B/16 | fp16-autocast | hidden-state | state-25 | 2.49e-04 | 2.156 | 0.0002674 |
| ViT-7B/16 | fp16-autocast | hidden-state | state-26 | 3.48e-04 | 3.335 | 0.0004076 |
| ViT-7B/16 | fp16-autocast | hidden-state | state-27 | 3.84e-04 | 2.83 | 0.000341 |
| ViT-7B/16 | fp16-autocast | hidden-state | state-28 | 4.66e-04 | 2.278 | 0.0002709 |
| ViT-7B/16 | fp16-autocast | hidden-state | state-29 | 5.26e-04 | 4.065 | 0.0004741 |
| ViT-7B/16 | fp16-autocast | hidden-state | state-30 | 5.38e-04 | 5.238 | 0.0005973 |
| ViT-7B/16 | fp16-autocast | hidden-state | state-31 | 7.04e-04 | 10.31 | 0.001138 |
| ViT-7B/16 | fp16-autocast | hidden-state | state-32 | 8.28e-04 | 17.04 | 0.001817 |
| ViT-7B/16 | fp16-autocast | hidden-state | state-33 | 8.29e-04 | 22.73 | 0.002337 |
| ViT-7B/16 | fp16-autocast | hidden-state | state-34 | 6.63e-04 | 32.1 | 0.003168 |
| ViT-7B/16 | fp16-autocast | hidden-state | state-35 | 7.20e-04 | 39.9 | 0.003775 |
| ViT-7B/16 | fp16-autocast | hidden-state | state-36 | 1.19e-03 | 48.58 | 0.004399 |
| ViT-7B/16 | fp16-autocast | hidden-state | state-37 | 1.49e-03 | 56.97 | 0.004909 |
| ViT-7B/16 | fp16-autocast | hidden-state | state-38 | 1.23e-03 | 70.79 | 0.005807 |
| ViT-7B/16 | fp16-autocast | hidden-state | state-39 | 8.17e-04 | 89.13 | 0.006934 |
| ViT-7B/16 | fp16-autocast | hidden-state | state-40 | 1.70e-04 | 80.74 | 0.005964 |
| ViT-7B/16 | bf16-autocast | stored-embeddings | iss-kibo-lab/224x224 | 5.70e-06 | 0.2338 | 0.001967 |
| ViT-7B/16 | bf16-autocast | stored-block | block-0 | 3.49e-06 | 5.136 | 0.001762 |
| ViT-7B/16 | bf16-autocast | stored-block | block-1 | 6.82e-07 | 0.6477 | 0.0001616 |
| ViT-7B/16 | bf16-autocast | stored-block | block-2 | 4.79e-07 | 1.125 | 0.0002351 |
| ViT-7B/16 | bf16-autocast | stored-block | block-3 | 2.65e-07 | 0.8003 | 0.0001488 |
| ViT-7B/16 | bf16-autocast | stored-block | block-4 | 2.10e-07 | 1.545 | 0.0002647 |
| ViT-7B/16 | bf16-autocast | stored-block | block-5 | 2.18e-07 | 1.362 | 0.0002193 |
| ViT-7B/16 | bf16-autocast | stored-block | block-6 | 2.90e-07 | 1.938 | 0.0002974 |
| ViT-7B/16 | bf16-autocast | stored-block | block-7 | 1.73e-07 | 0.2695 | 4.007e-05 |
| ViT-7B/16 | bf16-autocast | stored-block | block-8 | 1.57e-07 | 0.9155 | 0.0001327 |
| ViT-7B/16 | bf16-autocast | stored-block | block-9 | 1.21e-07 | 0.05615 | 8.035e-06 |
| ViT-7B/16 | bf16-autocast | stored-block | block-10 | 1.39e-07 | 0.1812 | 2.542e-05 |
| ViT-7B/16 | bf16-autocast | stored-block | block-11 | 1.18e-07 | 0.2417 | 3.341e-05 |
| ViT-7B/16 | bf16-autocast | stored-block | block-12 | 1.02e-07 | 0.3931 | 5.373e-05 |
| ViT-7B/16 | bf16-autocast | stored-block | block-13 | 1.88e-07 | 0.2412 | 3.266e-05 |
| ViT-7B/16 | bf16-autocast | stored-block | block-14 | 1.67e-07 | 0.1453 | 1.951e-05 |
| ViT-7B/16 | bf16-autocast | stored-block | block-15 | 1.43e-07 | 0.1181 | 1.574e-05 |
| ViT-7B/16 | bf16-autocast | stored-block | block-16 | 1.78e-07 | 0.1205 | 1.596e-05 |
| ViT-7B/16 | bf16-autocast | stored-block | block-17 | 1.69e-07 | 0.1285 | 1.691e-05 |
| ViT-7B/16 | bf16-autocast | stored-block | block-18 | 2.00e-07 | 0.2384 | 3.117e-05 |
| ViT-7B/16 | bf16-autocast | stored-block | block-19 | 3.45e-07 | 0.5615 | 7.279e-05 |
| ViT-7B/16 | bf16-autocast | stored-block | block-20 | 2.51e-07 | 0.2587 | 3.331e-05 |
| ViT-7B/16 | bf16-autocast | stored-block | block-21 | 4.01e-07 | 0.4282 | 5.477e-05 |
| ViT-7B/16 | bf16-autocast | stored-block | block-22 | 5.92e-07 | 0.4293 | 5.44e-05 |
| ViT-7B/16 | bf16-autocast | stored-block | block-23 | 5.70e-07 | 0.3991 | 5.009e-05 |
| ViT-7B/16 | bf16-autocast | stored-block | block-24 | 4.41e-07 | 0.3602 | 4.468e-05 |
| ViT-7B/16 | bf16-autocast | stored-block | block-25 | 6.88e-07 | 0.5327 | 6.511e-05 |
| ViT-7B/16 | bf16-autocast | stored-block | block-26 | 6.95e-07 | 0.5139 | 6.192e-05 |
| ViT-7B/16 | bf16-autocast | stored-block | block-27 | 6.66e-07 | 0.5407 | 6.429e-05 |
| ViT-7B/16 | bf16-autocast | stored-block | block-28 | 9.18e-07 | 0.7733 | 9.02e-05 |
| ViT-7B/16 | bf16-autocast | stored-block | block-29 | 9.54e-07 | 0.6919 | 7.89e-05 |
| ViT-7B/16 | bf16-autocast | stored-block | block-30 | 1.50e-06 | 0.8636 | 9.53e-05 |
| ViT-7B/16 | bf16-autocast | stored-block | block-31 | 1.52e-06 | 1.384 | 0.0001476 |
| ViT-7B/16 | bf16-autocast | stored-block | block-32 | 1.97e-06 | 1.487 | 0.0001529 |
| ViT-7B/16 | bf16-autocast | stored-block | block-33 | 3.25e-06 | 1.417 | 0.0001399 |
| ViT-7B/16 | bf16-autocast | stored-block | block-34 | 2.88e-06 | 1.71 | 0.0001618 |
| ViT-7B/16 | bf16-autocast | stored-block | block-35 | 4.69e-06 | 2.329 | 0.0002108 |
| ViT-7B/16 | bf16-autocast | stored-block | block-36 | 4.23e-06 | 3.015 | 0.0002598 |
| ViT-7B/16 | bf16-autocast | stored-block | block-37 | 3.27e-06 | 3.825 | 0.0003138 |
| ViT-7B/16 | bf16-autocast | stored-block | block-38 | 6.34e-06 | 6.525 | 0.0005076 |
| ViT-7B/16 | bf16-autocast | stored-block | block-39 | 3.55e-06 | 23.55 | 0.001739 |
| ViT-7B/16 | bf16-autocast | stored-final-norm | iss-kibo-lab/224x224 | 8.44e-15 | 1.431e-06 | 2.157e-07 |
| ViT-7B/16 | bf16-autocast | hidden-state | state-0 | 5.70e-06 | 0.2338 | 0.001967 |
| ViT-7B/16 | bf16-autocast | hidden-state | state-1 | 1.71e-05 | 5.136 | 0.001762 |
| ViT-7B/16 | bf16-autocast | hidden-state | state-2 | 2.18e-05 | 5.846 | 0.001458 |
| ViT-7B/16 | bf16-autocast | hidden-state | state-3 | 3.06e-05 | 7.08 | 0.001479 |
| ViT-7B/16 | bf16-autocast | hidden-state | state-4 | 5.62e-05 | 9.926 | 0.001846 |
| ViT-7B/16 | bf16-autocast | hidden-state | state-5 | 9.76e-05 | 8.381 | 0.001436 |
| ViT-7B/16 | bf16-autocast | hidden-state | state-6 | 1.31e-04 | 7.019 | 0.00113 |
| ViT-7B/16 | bf16-autocast | hidden-state | state-7 | 1.67e-04 | 8.967 | 0.001376 |
| ViT-7B/16 | bf16-autocast | hidden-state | state-8 | 2.24e-04 | 7.214 | 0.001073 |
| ViT-7B/16 | bf16-autocast | hidden-state | state-9 | 3.01e-04 | 6.299 | 0.000913 |
| ViT-7B/16 | bf16-autocast | hidden-state | state-10 | 3.50e-04 | 6.576 | 0.0009409 |
| ViT-7B/16 | bf16-autocast | hidden-state | state-11 | 3.66e-04 | 7.147 | 0.001003 |
| ViT-7B/16 | bf16-autocast | hidden-state | state-12 | 4.22e-04 | 7.427 | 0.001027 |
| ViT-7B/16 | bf16-autocast | hidden-state | state-13 | 5.22e-04 | 7.735 | 0.001057 |
| ViT-7B/16 | bf16-autocast | hidden-state | state-14 | 5.78e-04 | 7.45 | 0.001009 |
| ViT-7B/16 | bf16-autocast | hidden-state | state-15 | 5.35e-04 | 7.349 | 0.0009867 |
| ViT-7B/16 | bf16-autocast | hidden-state | state-16 | 1.05e-03 | 7.356 | 0.0009806 |
| ViT-7B/16 | bf16-autocast | hidden-state | state-17 | 1.47e-03 | 7.252 | 0.0009612 |
| ViT-7B/16 | bf16-autocast | hidden-state | state-18 | 1.96e-03 | 7.19 | 0.0009462 |
| ViT-7B/16 | bf16-autocast | hidden-state | state-19 | 2.88e-03 | 10.55 | 0.00138 |
| ViT-7B/16 | bf16-autocast | hidden-state | state-20 | 2.87e-03 | 12.82 | 0.001662 |
| ViT-7B/16 | bf16-autocast | hidden-state | state-21 | 3.98e-03 | 16.16 | 0.002081 |
| ViT-7B/16 | bf16-autocast | hidden-state | state-22 | 6.38e-03 | 21.51 | 0.00275 |
| ViT-7B/16 | bf16-autocast | hidden-state | state-23 | 9.26e-03 | 22.43 | 0.002843 |
| ViT-7B/16 | bf16-autocast | hidden-state | state-24 | 1.17e-02 | 18.41 | 0.002312 |
| ViT-7B/16 | bf16-autocast | hidden-state | state-25 | 1.45e-02 | 20.77 | 0.002576 |
| ViT-7B/16 | bf16-autocast | hidden-state | state-26 | 2.20e-02 | 27.22 | 0.003328 |
| ViT-7B/16 | bf16-autocast | hidden-state | state-27 | 2.59e-02 | 26.83 | 0.003233 |
| ViT-7B/16 | bf16-autocast | hidden-state | state-28 | 3.52e-02 | 37.24 | 0.004428 |
| ViT-7B/16 | bf16-autocast | hidden-state | state-29 | 4.63e-02 | 36.24 | 0.004227 |
| ViT-7B/16 | bf16-autocast | hidden-state | state-30 | 5.28e-02 | 41.86 | 0.004773 |
| ViT-7B/16 | bf16-autocast | hidden-state | state-31 | 6.15e-02 | 60.07 | 0.006629 |
| ViT-7B/16 | bf16-autocast | hidden-state | state-32 | 5.99e-02 | 74.82 | 0.00798 |
| ViT-7B/16 | bf16-autocast | hidden-state | state-33 | 6.76e-02 | 155 | 0.01593 |
| ViT-7B/16 | bf16-autocast | hidden-state | state-34 | 6.77e-02 | 184.9 | 0.01825 |
| ViT-7B/16 | bf16-autocast | hidden-state | state-35 | 7.40e-02 | 273.7 | 0.0259 |
| ViT-7B/16 | bf16-autocast | hidden-state | state-36 | 6.25e-02 | 270.2 | 0.02446 |
| ViT-7B/16 | bf16-autocast | hidden-state | state-37 | 5.84e-02 | 315.4 | 0.02718 |
| ViT-7B/16 | bf16-autocast | hidden-state | state-38 | 7.32e-02 | 321.6 | 0.02638 |
| ViT-7B/16 | bf16-autocast | hidden-state | state-39 | 9.75e-02 | 434.1 | 0.03377 |
| ViT-7B/16 | bf16-autocast | hidden-state | state-40 | 1.66e-02 | 698.1 | 0.05157 |

Table 4. Checks of mlx against the stored hidden states.

## 5 Cross-check against torch

scripts/cross_check.py compares the mlx output with the torch output in the same dtype mode, without the reference output. Report only: no threshold applies.

| Model size | Dtype mode | Configuration | Torch backend | Status | Frame | Patch mean | Patch 1st percentile | Patch minimum |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ViT-7B/16 | bf16-cast | forest/224x224 | torch-mps | measured | 0.999945 | 0.999936 | 0.999578 | 0.999247 |
| ViT-7B/16 | bf16-cast | forest/224x304 | torch-mps | measured | 0.999954 | 0.999954 | 0.999849 | 0.999634 |
| ViT-7B/16 | bf16-cast | forest/448x448 | torch-mps | measured | 0.999953 | 0.999969 | 0.999919 | 0.999722 |
| ViT-7B/16 | bf16-cast | forest/512x512 | torch-mps | measured | 0.999945 | 0.999970 | 0.999924 | 0.999725 |
| ViT-7B/16 | bf16-cast | iss-conference/224x224 | torch-mps | measured | 0.999938 | 0.999830 | 0.998893 | 0.997460 |
| ViT-7B/16 | bf16-cast | iss-conference/304x224 | torch-mps | measured | 0.999945 | 0.999505 | 0.995725 | 0.942227 |
| ViT-7B/16 | bf16-cast | iss-conference/448x448 | torch-mps | measured | 0.999944 | 0.999918 | 0.999665 | 0.993909 |
| ViT-7B/16 | bf16-cast | iss-conference/512x512 | torch-mps | measured | 0.999942 | 0.999941 | 0.999758 | 0.999072 |
| ViT-7B/16 | bf16-cast | iss-kibo-lab/224x224 | torch-mps | measured | 0.999944 | 0.999852 | 0.999108 | 0.994850 |
| ViT-7B/16 | bf16-cast | iss-kibo-lab/224x400 | torch-mps | measured | 0.999925 | 0.999882 | 0.999250 | 0.992311 |
| ViT-7B/16 | bf16-cast | iss-kibo-lab/448x448 | torch-mps | measured | 0.999937 | 0.999932 | 0.999603 | 0.994370 |
| ViT-7B/16 | bf16-cast | iss-kibo-lab/512x512 | torch-mps | measured | 0.999951 | 0.999956 | 0.999854 | 0.998708 |

Table 5. Cross-check of mlx against torch.

## 6 Compiled forward pass against the eager forward pass

Each configuration runs the compiled forward pass and the eager forward pass of one backbone (W1 of specs/06-speed/plan.html). A non-finite configuration counts as equal when both forms are non-finite at the same positions.

| Model size | Dtype mode | Configurations | Bit for bit equal | Largest difference |
| --- | --- | --- | --- | --- |
| ViT-S/16 | fp32 | 12 | 12 | 0 |
| ViT-S/16 | fp16-cast | 12 | 12 | 0 |
| ViT-S/16 | bf16-cast | 12 | 12 | 0 |
| ViT-S/16 | fp16-autocast | 12 | 12 | 0 |
| ViT-S/16 | bf16-autocast | 12 | 12 | 0 |
| ViT-S+/16 | fp32 | 12 | 12 | 0 |
| ViT-S+/16 | fp16-cast | 12 | 12 | 0 |
| ViT-S+/16 | bf16-cast | 12 | 12 | 0 |
| ViT-S+/16 | fp16-autocast | 12 | 12 | 0 |
| ViT-S+/16 | bf16-autocast | 12 | 12 | 0 |
| ViT-B/16 | fp32 | 12 | 12 | 0 |
| ViT-B/16 | fp16-cast | 12 | 12 | 0 |
| ViT-B/16 | bf16-cast | 12 | 12 | 0 |
| ViT-B/16 | fp16-autocast | 12 | 12 | 0 |
| ViT-B/16 | bf16-autocast | 12 | 12 | 0 |
| ViT-L/16 | fp32 | 12 | 12 | 0 |
| ViT-L/16 | fp16-cast | 12 | 12 | 0 |
| ViT-L/16 | bf16-cast | 12 | 12 | 0 |
| ViT-L/16 | fp16-autocast | 12 | 12 | 0 |
| ViT-L/16 | bf16-autocast | 12 | 12 | 0 |
| ViT-H+/16 | fp32 | 12 | 12 | 0 |
| ViT-H+/16 | fp16-cast | 12 | 12 | 0 |
| ViT-H+/16 | bf16-cast | 12 | 12 | 0 |
| ViT-H+/16 | fp16-autocast | 12 | 12 | 0 |
| ViT-H+/16 | bf16-autocast | 12 | 12 | 0 |
| ViT-7B/16 | fp32 | 0 | 0 | – |
| ViT-7B/16 | fp16-cast | 12 | 12 | 0 |
| ViT-7B/16 | bf16-cast | 12 | 12 | 0 |
| ViT-7B/16 | fp16-autocast | 12 | 12 | 0 |
| ViT-7B/16 | bf16-autocast | 12 | 12 | 0 |

Table 6. The compiled forward pass against the eager forward pass.

## 7 Quantized weights

Report only, with a fault bound in the tests (ADR 0011). Each row has the minimum over its configurations, against the reference output and against the unquantized output of the same dtype mode. The column Pass counts the configurations that pass the thresholds of the dtype mode (Section 3).

| Model size | Dtype mode | Bits | Measured | Pass | Frame | Patch mean | Patch 1st percentile | Frame (unquantized) | Patch mean (unquantized) | Patch 1st percentile (unquantized) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ViT-S/16 | fp16-autocast | 4 | 12 | 0 | 0.861780 | 0.940549 | 0.903443 | 0.861770 | 0.940538 | 0.903631 |
| ViT-S/16 | fp16-autocast | 8 | 12 | 12 | 0.999392 | 0.999818 | 0.999441 | 0.999412 | 0.999819 | 0.999440 |
| ViT-S/16 | bf16-autocast | 4 | 12 | 0 | 0.863331 | 0.940645 | 0.903103 | 0.863758 | 0.940730 | 0.903360 |
| ViT-S/16 | bf16-autocast | 8 | 12 | 12 | 0.998484 | 0.999458 | 0.998767 | 0.998406 | 0.999478 | 0.998820 |
| ViT-H+/16 | fp16-autocast | 4 | 12 | 0 | 0.987205 | 0.961082 | 0.757351 | 0.987206 | 0.961042 | 0.752912 |
| ViT-H+/16 | fp16-autocast | 8 | 12 | 12 | 0.999954 | 0.999790 | 0.998918 | 0.999951 | 0.999805 | 0.998775 |
| ViT-H+/16 | bf16-autocast | 4 | 12 | 0 | 0.987078 | 0.960276 | 0.749771 | 0.987063 | 0.960893 | 0.759336 |
| ViT-H+/16 | bf16-autocast | 8 | 12 | 12 | 0.999877 | 0.998980 | 0.983976 | 0.999882 | 0.999345 | 0.995660 |
| ViT-7B/16 | fp16-autocast | 4 | 12 | 0 | 0.993235 | 0.987466 | 0.933929 | 0.993231 | 0.987451 | 0.934982 |
| ViT-7B/16 | fp16-autocast | 8 | 12 | 12 | 0.999967 | 0.999931 | 0.999337 | 0.999975 | 0.999948 | 0.999678 |
| ViT-7B/16 | bf16-autocast | 4 | 12 | 0 | 0.993112 | 0.986591 | 0.910137 | 0.993356 | 0.987231 | 0.939965 |
| ViT-7B/16 | bf16-autocast | 8 | 12 | 12 | 0.999893 | 0.998698 | 0.983754 | 0.999932 | 0.999271 | 0.998212 |

Table 7. Parity of the quantized weights.

## 8 fp16 patch embeddings

The patch embeddings of mlx_embeddings in fp16 (D13 of specs/06-speed/plan.html) against the reference output. The extra loss is the fp32-output parity minus the fp16-output parity of the patch mean and the patch 1st percentile.

| Model size | Dtype mode | Measured | Patch mean | Patch 1st percentile | Largest extra loss | Frame unchanged |
| --- | --- | --- | --- | --- | --- | --- |
| ViT-S/16 | fp16-cast | 12 | 0.999997 | 0.999986 | 0 | yes |
| ViT-S/16 | bf16-cast | 12 | 0.999779 | 0.999560 | 0 | yes |
| ViT-S/16 | fp16-autocast | 12 | 0.999998 | 0.999995 | 5.3e-08 | yes |
| ViT-S/16 | bf16-autocast | 12 | 0.999924 | 0.999650 | 3.4e-07 | yes |
| ViT-S+/16 | fp16-cast | 12 | 0.999920 | 0.999808 | 0 | yes |
| ViT-S+/16 | bf16-cast | 12 | 0.995816 | 0.990381 | 0 | yes |
| ViT-S+/16 | fp16-autocast | 12 | 0.999958 | 0.999904 | 1.5e-07 | yes |
| ViT-S+/16 | bf16-autocast | 12 | 0.995136 | 0.988886 | 2.1e-07 | yes |
| ViT-B/16 | fp16-cast | 12 | 0.999997 | 0.999991 | 0 | yes |
| ViT-B/16 | bf16-cast | 12 | 0.999821 | 0.999478 | 0 | yes |
| ViT-B/16 | fp16-autocast | 12 | 0.999998 | 0.999993 | 9.2e-08 | yes |
| ViT-B/16 | bf16-autocast | 12 | 0.999899 | 0.999528 | 1.8e-07 | yes |
| ViT-L/16 | fp16-cast | 0 | – | – | – | – |
| ViT-L/16 | bf16-cast | 12 | 0.999558 | 0.996394 | 0 | yes |
| ViT-L/16 | fp16-autocast | 12 | 0.999996 | 0.999966 | 8.3e-08 | yes |
| ViT-L/16 | bf16-autocast | 12 | 0.999791 | 0.997287 | 2.8e-07 | yes |
| ViT-H+/16 | fp16-cast | 6 | 0.999998 | 0.999993 | 0 | yes |
| ViT-H+/16 | bf16-cast | 12 | 0.999061 | 0.986725 | 3.7e-15 | yes |
| ViT-H+/16 | fp16-autocast | 12 | 0.999981 | 0.999679 | 2.5e-07 | yes |
| ViT-H+/16 | bf16-autocast | 12 | 0.999279 | 0.985938 | 1.3e-06 | yes |
| ViT-7B/16 | fp16-cast | 12 | 0.999981 | 0.999654 | 0 | yes |
| ViT-7B/16 | bf16-cast | 12 | 0.999119 | 0.982649 | 3.1e-15 | yes |
| ViT-7B/16 | fp16-autocast | 12 | 0.999986 | 0.999761 | 1e-07 | yes |
| ViT-7B/16 | bf16-autocast | 12 | 0.999229 | 0.983737 | 7.3e-07 | yes |

Table 8. Parity with the fp16 patch embeddings.

## 9 Non-finite configurations: mlx against torch-mps

Each row is a parity or batch configuration with a non-finite output on at least one backend. The last line says if the two sets are equal.

| Test | Model size | Dtype mode | Configuration | mlx | torch-mps |
| --- | --- | --- | --- | --- | --- |
| parity | ViT-L/16 | fp16-cast | forest/224x224 | non-finite | non-finite |
| parity | ViT-L/16 | fp16-cast | forest/224x304 | non-finite | non-finite |
| parity | ViT-L/16 | fp16-cast | forest/448x448 | non-finite | non-finite |
| parity | ViT-L/16 | fp16-cast | forest/512x512 | non-finite | non-finite |
| parity | ViT-L/16 | fp16-cast | iss-conference/224x224 | non-finite | non-finite |
| parity | ViT-L/16 | fp16-cast | iss-conference/304x224 | non-finite | non-finite |
| parity | ViT-L/16 | fp16-cast | iss-conference/448x448 | non-finite | non-finite |
| parity | ViT-L/16 | fp16-cast | iss-conference/512x512 | non-finite | non-finite |
| parity | ViT-L/16 | fp16-cast | iss-kibo-lab/224x224 | non-finite | non-finite |
| parity | ViT-L/16 | fp16-cast | iss-kibo-lab/224x400 | non-finite | non-finite |
| parity | ViT-L/16 | fp16-cast | iss-kibo-lab/448x448 | non-finite | non-finite |
| parity | ViT-L/16 | fp16-cast | iss-kibo-lab/512x512 | non-finite | non-finite |
| parity | ViT-H+/16 | fp16-cast | forest/224x224 | non-finite | non-finite |
| parity | ViT-H+/16 | fp16-cast | forest/224x304 | non-finite | non-finite |
| parity | ViT-H+/16 | fp16-cast | iss-conference/224x224 | non-finite | non-finite |
| parity | ViT-H+/16 | fp16-cast | iss-conference/304x224 | non-finite | non-finite |
| parity | ViT-H+/16 | fp16-cast | iss-kibo-lab/224x224 | non-finite | non-finite |
| parity | ViT-H+/16 | fp16-cast | iss-kibo-lab/224x400 | non-finite | non-finite |
| batch | ViT-L/16 | fp16-cast | batch-8-224x224 | non-finite | non-finite |
| batch | ViT-H+/16 | fp16-cast | batch-8-224x224 | non-finite | non-finite |

Table 9. Non-finite configurations of mlx and torch-mps.

equal
