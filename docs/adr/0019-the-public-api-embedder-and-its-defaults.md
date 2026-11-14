# The public API: the class Embedder and its defaults

Steps 2 to 6 built the MLX port behind internal functions (`load_mlx_backbone`, `compile_backbone`, `preprocess_frames`, `mlx_embeddings`). Users need one stable entry point. Version 0.1.0 is the first contract; a change to it needs a new version.

Decision (D1 B and D2 of `specs/07-release/explore.html`, the user, 2026-11-09; the TF32 probe P1 of `plan.html`, the user, 2026-11-10):

1. **One class.** `Embedder(model_size="ViT-S/16", dtype_mode="fp16-autocast", batch_size=8, bits=None, backend="mlx")` loads and compiles once. `embed_frames`, `embed_patches` and `embed` give the frame embeddings (B, D) in fp32 and the patch embeddings (B, H/16, W/16, D), fp16 in the four half dtype modes.
2. **Input.** A uint8 RGB array (B, h, w, 3), or a list of PIL images or uint8 arrays of one size. Other dtypes raise `TypeError`, mixed sizes raise `ValueError`.
3. **Input resolution.** The default is 224 × 224 px, as the Hugging Face image processor of DINOv3. `size` takes an int or (H, W), each side a multiple of 16. The resize does not keep the aspect ratio.
4. **Default dtype mode fp16-autocast.** It is the only fast dtype mode that gives finite outputs and passes the fp16 thresholds of ADR 0003 for all six model sizes.
5. **A fixed batch size, padded.** The default is 8. The last batch gets zero frames, so each input resolution needs one trace only. The rows of the zero frames are dropped.
6. **Output.** NumPy by default; `as_numpy=False` keeps `mx.array` on the GPU. No L2 normalization.
7. **Weights.** A download on first use. A refused download raises `WeightsAccessError` with the model page, the DINOv3 License and `hf auth login`.
8. **TF32.** `import dinov3_mlx` sets `MLX_ENABLE_TF32=0` with `setdefault`. The constructor runs the TF32 probe `tf32_active`: one 64 × 64 fp32 matmul whose exact result needs more than the 10-bit mantissa of TF32. If TF32 is on, it raises `TF32ModeError`.
9. **Quantized weights.** `bits` takes only `None` or 8, and 8 only in fp16-autocast and bf16-autocast (ADR 0011). The constructor checks this, the run rule and the TF32 probe before any weight look-up or download (V1 and V8 of `specs/07-release/verify.html`, fixed at the Close).
10. **Side effects.** The import loads no mlx and no torch (PEP 562). The constructor sets the MLX memory limit of the process again after the load. It does not take the GPU lock of this repo.

Why:

- The API wraps the tested path: the compiled forward pass of ADR 0010 and the GPU preprocessing of Step 6. B1 measured the overhead of the API against this path: 0.996 at P1 and 1.009 at P2 (`results/api-table.md`).
- fp16-autocast costs 7 to 11 % of the throughput of bf16-cast, but bf16-cast has a worst patch 1st percentile of 0.9826, and fp16-cast overflows for ViT-L/16 and ViT-H+/16.
- MLX reads `MLX_ENABLE_TF32` once, at its first use. The variable alone does not show the state of a process that used mlx before the import, so the probe measures it.
- A fixed batch size keeps the compile cost to one trace for each input resolution. The best batch size of the speedup table rests on one round (V4 of Step 6), so the API does not use it.

We considered:

- **Functions instead of a class** (D1 A). Each call would load or cache the model in a hidden global.
- **The own-aspect size** (D2b b). It keeps the shape of a 16:9 frame, but Hugging Face users get 224 × 224.
- **bf16-cast as the default** (D2c b). Faster, with a lower patch parity.
- **Only an error for TF32, no `setdefault`** (D2e b). It fails for every user who imports mlx first.

Consequences:

- The import turns TF32 off for the fp32 matmuls of the user in the same process, if the user did not set the variable. The README states this.
- A padded last batch costs throughput: P1-pad (13 frames at batch size 8) gave 0.819 (`results/api-table.md`). The README tells users to give a multiple of `batch_size`.
- `backend` accepts only `"mlx"`. Step 7b adds `"mlx+ne"` behind the same contract.

Status: accepted on 2026-11-09 and 2026-11-10; written at the Step 7 Close on 2026-11-14.
