# ViT-7B/16 holds the fp32 weights of one block at a time, only in the streamed reference

The run rule of CLAUDE.md says that ViT-7B/16 runs in fp16, bf16 or with quantized weights, because its fp32 weights alone need about 27 GB. We read this rule as a limit on the fp32 weights in memory, not as a ban on fp32 arithmetic:

- The fp32 weights of the whole model never load. The guards stay: `check_run_rule` in `load_torch_backbone` and in `load_mlx_backbone` raises `RunRuleError` before the memory check, and `scripts/make_reference.py` records a skip without `--streamed`. No full fp32 load on torch or `mlx`, no lazily memory-mapped full fp32 model and no autocast mode with fp32 weights.
- The streamed reference (`run_torch_streamed` in `src/dinov3_mlx/reference.py`) holds the fp32 weights of one block at a time. It reads the tensors of one block through `ShardIndex`, runs all configurations through the block, frees the block and then reads the next block. It is the only fp32 path of ViT-7B/16.
- The limit applies to the resident memory, not to the address space. `ShardIndex.read` maps the whole shard of each tensor (about 5 GB), and the tensors of `OuterParts` keep the maps of their shards open during the whole run. A mapped file is not a load (finding V4 of `specs/05-all-sizes/verify.html`).

Why:

- Memory. The largest block of ViT-7B/16 has 0.67 GB of fp32 weights, against 26.86 GB for the whole model. The memory check of the streamed run asks for 4.82 GB (1.2 × (largest block + outer parts) + 4 GB). The measured peak at the end of the streamed run is 1.75 GB (36.3 % of the need), and the lifetime peak of the subprocess is 1.93 GB (40.1 % of the need). A full fp32 load would ask for 36.24 GB (Figure 3 and Table 22 of `verify.html`).
- Thresholds. The parity thresholds of ADR 0003 are defined against an fp32 reference output. A bf16 reference has its own bf16 error. So only an fp32 reference checks the bf16 thresholds of ADR 0003 for ViT-7B/16.
- Bit equality. The streamed path uses the same Hugging Face classes, attention "sdpa", fp32 RoPE tables and batch size 1 as a full-model run. Against the five reference files of Step 2 (ViT-S/16 to ViT-H+/16, full-model runs), the largest difference is 0.0 for the tokens of 12 configurations and for the hidden states of each model size (R24, Table 22 of `verify.html`).

We considered three other options (D4 of `specs/05-all-sizes/explore.html`, Table 12):

- A bf16 reference on torch-cpu, full or streamed (Option B). It fits the run rule. But it does not measure parity against fp32, and it does not check the bf16 thresholds of ADR 0003 for ViT-7B/16.
- A cross-check of `mlx` bf16-cast against torch-mps bf16-cast (Option C). It fits the run rule, but it shows only that two backends agree. No thresholds exist for that. Path D4-N used it as a diagnostic: smallest frame 0.999925.
- Defer again (Option D). No new code and no memory, but no parity rows for ViT-7B/16.

Consequences:

- ViT-7B/16 has a reference file: `reference-vit7b16.safetensors`, with the header field `streamed` true. Its 48 `mlx` parity rows in fp16-cast, bf16-cast, fp16-autocast and bf16-autocast pass, and its 24 torch-mps rows in fp16-cast and bf16-cast pass (Table 23 of `verify.html`). The largest share of a threshold budget is 43.9 % (torch-mps bf16-cast, patch 1st percentile). ADR 0003 does not change.
- The token range of ViT-7B/16 in fp32 comes from the hidden states of the streamed file: 13,537.4 after block 40, 20.7 % of the fp16 maximum.
- The 12 fp32 rows of ViT-7B/16 on each backend stay skipped with the `RunRuleError` text, and so do the torch-mps rows in fp16-autocast and bf16-autocast (fp32 weights).
- The check of the ViT-7B/16 file against itself shows only that the streamed run repeats (finding V3). The independent evidence is the bit equality on five model sizes, the small model with the shape of ViT-7B/16 against `from_pretrained` (0.0) and the agreement of two backends in fp16.
- A new model size above the memory limits can use the same path. Add it to `NO_FP32_WEIGHTS`, and make its reference file with `scripts/make_reference.py --streamed`.
- The guard must stay in front of every full fp32 load. Each new loader or test of a model size in `NO_FP32_WEIGHTS` calls `check_run_rule` first. The tests of the guards (T7, T11 to T15) and the mutation probes G1 to G4 of `verify.html` pin this.
- Open: no test reads a block with a missing tensor (V1), and `make_reference.py --streamed` without model sizes writes the five full-model files again (V2).

Status: accepted on 2026-11-07 by the user's own answer "Yes, run it" to the question "May the streamed fp32 reference of ViT-7B/16 run, one block of fp32 weights at a time?" (Table 20 of `specs/05-all-sizes/plan.html`). The agent did not accept it on behalf of the user.
