# Neural Engine rewrite: score scale 256, a power-of-two residual-stream scale with ε / r², no LayerScale fold

Step 6c runs the Hugging Face backbone as a Core ML package on the Apple Neural Engine (`specs/06c-neural-engine/`). The Neural Engine stores every activation in fp16. Two value ranges of DINOv3 are larger than the largest fp16 value, 65,504:

- the attention scores q·kᵀ / √d: up to 2.5 million in block 1 of ViT-S/16 (finding F1 of `explore.html`);
- the residual stream with its register tokens: up to 158,433 in ViT-L/16 and 149,750 in ViT-H+/16 (`results/token-range-mlx.md`).

The plain fp16 package gives finite but wrong tokens on CPU_AND_NE (frame cosine 0.035 for ViT-S/16), and correct tokens on CPU_AND_GPU.

Decision: before the trace, the build changes the torch graph with the Neural Engine rewrite (`rewrite_for_neural_engine` in `src/dinov3_mlx/coreml.py`). It is exact in real numbers:

1. **Score scale.** `ne_attention` computes z = (q · (1/√d) / 256) · kᵀ, then softmax(clip(z − max z, −60,000 / 256) · 256). The row maximum comes before the multiplication, so no value above 0 enters the softmax. 256 is a power of two, so the division adds no rounding error.
2. **Residual-stream scale.** The embeddings are divided by r, and the weights and the bias of `o_proj` and `down_proj` are multiplied by 1/r. r is the smallest power of two with (largest token value) / r ≤ 32,768: 1 for ViT-S/16, ViT-S+/16 and ViT-B/16, 8 for ViT-L/16 and ViT-H+/16. `residual_scale` reads the value from the token-range table. The layer norms remove the factor. Each layer norm ε becomes ε / r², so each layer norm gives exactly its old output (plan amendment of 2026-11-08).
3. **No LayerScale fold.** LayerScale stays a separate multiplication.

Why:

- **The fold is wrong in fp16.** The LayerScale factors of DINOv3 go down to 3.6e-8. Folded into `o_proj`, they made 94 % of the weights of block 1 of ViT-S/16 subnormal in fp16 (63 to 78 % in ViT-L/16 and ViT-H+/16). With the fold, ViT-L/16 and ViT-H+/16 failed also on the GPU (patch 1st percentile 0.985). With only a power of two in the weights, a weight changes only below the smallest normal fp16 value.
- **One package serves every compute unit.** On CPU_AND_GPU and ALL the rewrite does not change the parity (0.999995 against 0.999996 for the plain graph of ViT-S/16).
- **ε / r² is needed.** With the Hugging Face ε, each layer norm acted as if ε were 64 times larger (r = 8). ViT-L/16 then failed also on CPU_AND_GPU (frame 0.888 to 0.974). With ε / r², every block of ViT-L/16 on CPU_AND_GPU has cosine ≥ 0.9999998, and ViT-L/16 and ViT-H+/16 pass the fp16 thresholds on CPU_AND_GPU and ALL. A larger r (64) is worse: λ · (branch / r) leaves the normal fp16 range.
- **Exact and testable.** Tests T1, T3 and T4 compare the rewrite with the Hugging Face graph in fp64 and fp32, and check that LayerScale and the power of two are bit for bit as planned.

We considered:

- The plain graph: fails on the Neural Engine for every model size (F1).
- Score scaling only: ViT-L/16 and ViT-H+/16 overflow in the residual stream.
- The layout of Apple's ml-ane-transformers ((B, C, 1, S) tensors, 1 × 1 convolutions, one attention for each head): a second rewrite of each block, with a gain that this step did not measure. It is open question Q1 of the plan.

Consequences:

- The margins are 2 times for the residual stream (32,768 against 65,504) and 6.6 times for the scores of ViT-S/16 (9,877 against 65,504) on the 3 test images. Another image with larger values can break the margin (risk R54). A new token-range table changes r, and r is part of the package name.
- On the Neural Engine each op rounds to fp16, so its parity is lower than on the GPU. The build measures it on the 3 test images for every model size and input resolution (`results/coreml-table.md`, Table 2).
- The rewrite does not make the Neural Engine correct in general. On CPU_AND_NE only the packages whose compute plan keeps the first ops (input cast, patch embedding, first attention branch) on the CPU pass: ViT-S/16 and ViT-S+/16 at 224 px, batch size 1. In every other package the output of the embeddings is already wrong (Q3 probe of the build: cosine 0.19 for ViT-S/16 at 448 px). A patch embedding as reshape + matmul (exact) did not change this; the orchestrator had approved it as a deviation from Q3 on the condition that it fixes the fault, so the build removed it again. The cause is open (follow-up).
- The package names carry the rewrite version: `ne2` (score scale, residual-stream scale, ε / r²).

Update 2026-11-09: Step 6d found the cause of the open fault above, the concat fault of the token concat. ADR 0016 gives the fix (one constant prefix, rewrite version `ne3`) and the guard.

Status: accepted on 2026-11-08 by the Step 6c build, from D2 of `specs/06c-neural-engine/explore.html` (the user asked the agent to decide, night of 2026-11-07).
