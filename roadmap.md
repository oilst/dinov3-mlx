# Roadmap

One row for each step. `docs/plan.html` describes each step in detail. A step gets its spec folder `specs/NN-slug/` when it starts.

| Step | Goal | DINOv3 topic that we learn | Spec folder | Status |
|---|---|---|---|---|
| 0 Workflow | Mission, roadmap, tech stack, glossary and the plan | The model family and the training method (overview) | none | done |
| 1 Scope | Prior art, repo license, Python project, access to the weights | What the other ports cover | `specs/01-scope/` | done (2026-11-06). All six model sizes are in the Hugging Face cache. |
| 2 Reference | Reference outputs and parity tests for all model sizes | Tokens and outputs: CLS token, register tokens, patch tokens | `specs/02-reference/` | done (2026-11-06). The ViT-7B/16 reference file is deferred: its memory check failed. |
| 3 Baseline | Benchmark harness and the PyTorch MPS baseline | Where the compute goes for each model size and input resolution | `specs/03-baseline/` | done (2026-11-07). Measured again on the 94 W adapter. AC5 (spread ≤ 5 %) fails in 5 of 10 rows: accepted by the user. ViT-7B/16 (B4) is deferred. ADR 0004. |
| 4 MLX port | ViT-S/16 in MLX and the weight converter | Each part of a block: patch embedding, RoPE, attention, LayerScale, feed-forward network | `specs/04-mlx-port/` | next |
| 5 All sizes | All model sizes in fp32, fp16 and bf16 | SwiGLU, model scaling, numeric range of the tokens | `specs/05-all-sizes/` | open |
| 6 Speed | Compiled graphs, batch sizes, input resolutions, quantized weights | Attention cost at high input resolution | `specs/06-speed/` | open |
| 7 Release | Python API, README with results, CI, first public push | none | `specs/07-release/` | open |
| 8 Fine-tuning | 8a: the reference fine-tuning setup, fast in MLX. 8b: DoRA and other adapters | DINO loss, iBOT loss, KoLeo regularizer, Gram anchoring | `specs/08-fine-tuning/` | open |
