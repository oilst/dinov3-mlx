# Parity of mlx

Made by `scripts/parity_table.py` on 2026-11-07 from the parity tests (mlx 0.32.4). The reference output is torch-cpu in fp32. Each value is the minimum over the configurations of the row. The patch minimum is report only. Section 3 gives the thresholds.

## 1 Parity against the reference output

Each row covers 12 configurations: 3 test images × 4 input resolutions.

| Model size | Dtype mode | Measured | Non-finite | Frame | Patch mean | Patch 1st percentile | Patch minimum | Result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ViT-S/16 | fp32 | 12 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | pass |

Table 1. Parity of mlx against the reference output.

## 2 Batch consistency

A batch of 8 at 224 × 224 px against 8 runs with batch size 1, on mlx in the same dtype mode.

| Model size | Dtype mode | Measured | Non-finite | Frame | Patch mean | Patch 1st percentile | Patch minimum | Result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ViT-S/16 | fp32 | 1 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | pass |

Table 2. Batch consistency.

## 3 Thresholds

| Dtype mode | Frame | Patch mean | Patch 1st percentile |
| --- | --- | --- | --- |
| fp32 | 0.9999 | 0.9999 | 0.9999 |
| fp16-cast | 0.999 | 0.999 | 0.99 |
| bf16-cast | 0.998 | 0.995 | 0.96 |
| fp16-autocast | 0.999 | 0.999 | 0.99 |
| bf16-autocast | 0.998 | 0.995 | 0.96 |

Table 3. Thresholds of `src/dinov3_apple/parity.py`. On 2026-11-06 the user gave bf16 its own row: bf16 activations give lower parity on torch-mps and on torch-cpu alike.

## 4 Stored hidden states

Each check runs one part on the stored input of iss-kibo-lab/224x224 and compares with the stored output: the embeddings, each block alone, the final layer norm and the 13 hidden states in a chain. The minimum token cosine runs in fp64.

| Check | Part | 1 − minimum token cosine | Largest difference | Largest relative difference |
| --- | --- | --- | --- | --- |
| stored-embeddings | iss-kibo-lab/224x224 | 9.46e-13 | 1.24e-05 | 1.162e-06 |
| stored-block | block-0 | 1.68e-11 | 0.0001755 | 3.691e-06 |
| stored-block | block-1 | 3.18e-13 | 2.289e-05 | 5.689e-07 |
| stored-block | block-2 | 2.78e-13 | 6.104e-05 | 2.421e-07 |
| stored-block | block-3 | 1.83e-13 | 1.717e-05 | 6.954e-08 |
| stored-block | block-4 | 9.59e-13 | 3.147e-05 | 1.05e-07 |
| stored-block | block-5 | 5.33e-13 | 6.104e-05 | 1.977e-07 |
| stored-block | block-6 | 1.38e-12 | 0.0004883 | 5.044e-07 |
| stored-block | block-7 | 1.59e-12 | 0.0001221 | 1.264e-07 |
| stored-block | block-8 | 1.36e-12 | 8.011e-05 | 8.119e-08 |
| stored-block | block-9 | 1.86e-12 | 0.0002441 | 2.102e-07 |
| stored-block | block-10 | 8.13e-13 | 0.0004883 | 2.858e-07 |
| stored-block | block-11 | 8.33e-13 | 0.0005493 | 3.724e-07 |
| stored-final-norm | iss-kibo-lab/224x224 | 8.77e-15 | 2.384e-06 | 2.603e-07 |
| hidden-state | state-0 | 9.46e-13 | 1.24e-05 | 1.162e-06 |
| hidden-state | state-1 | 3.03e-11 | 0.0003738 | 7.863e-06 |
| hidden-state | state-2 | 3.10e-11 | 0.000124 | 3.081e-06 |
| hidden-state | state-3 | 2.46e-11 | 5.436e-05 | 2.156e-07 |
| hidden-state | state-4 | 1.73e-11 | 4.959e-05 | 2.009e-07 |
| hidden-state | state-5 | 2.86e-11 | 5.341e-05 | 1.782e-07 |
| hidden-state | state-6 | 2.72e-11 | 6.104e-05 | 1.977e-07 |
| hidden-state | state-7 | 4.28e-11 | 0.0003662 | 3.783e-07 |
| hidden-state | state-8 | 4.12e-11 | 0.0004883 | 5.057e-07 |
| hidden-state | state-9 | 5.89e-11 | 0.0004272 | 4.33e-07 |
| hidden-state | state-10 | 6.81e-11 | 0.0006104 | 5.254e-07 |
| hidden-state | state-11 | 1.07e-10 | 0.0006104 | 3.573e-07 |
| hidden-state | state-12 | 7.46e-12 | 0.001221 | 8.276e-07 |

Table 4. Checks of mlx against the stored hidden states.
