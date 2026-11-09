# Neural Engine embeddings with one constant prefix (rewrite ne3) and a guard against the concat fault

Step 6c (finding V1 of `specs/06c-neural-engine/verify.html`) found that 13 of 15 Core ML packages of rewrite `ne2` give finite but wrong tokens on CPU_AND_NE. On CPU_AND_GPU and ALL the same packages pass. Step 6d (`specs/06d-neural-engine-fault/`) looked for the cause with probes E1 to E6 (macOS 26, coremltools 9.0, ML Program for macOS 15):

- The first wrong op is the token concat of the embeddings, `torch.cat([cls_token, register_tokens, patch_embeddings], dim=1)`. Up to the transpose the Neural Engine output has cosine 1.000000; after the concat it has 0.192437 and 788 of 789 wrong tokens (ViT-S/16, 448 px).
- Each wrong row is an exact copy of another row: the register tokens are missing, and the patch tokens start 3 rows too early. So it is a fault of the row offsets, not of the fp16 range.
- The converter makes the CLS token and the register tokens two separate constants. A concat with two or more constant inputs in front of the patch tokens was wrong in 3 of 5 forms; every concat with one constant input was right (probe E6). We did not find the exact rule: two forms with two constants were right.
- The two packages that passed in Step 6c (ViT-S/16 and ViT-S+/16, 224 px, batch size 1) ran the concat on the CPU.

We call this the concat fault (`GLOSSARY.md`).

Decision (option A of `explore.html`, picked by the user on 2026-11-08):

1. **One constant prefix.** `ne_embeddings` in `src/dinov3_mlx/coreml.py` joins the CLS token and the 4 register tokens first. The trace folds this join into one constant of 5 rows, and the token concat gets one constant input. In fp32 the output is bit-equal to the Hugging Face embeddings. `NeuralEngineBackbone.forward` calls it. The rest of the rewrite of ADR 0014 does not change.
2. **Rewrite version `ne3`.** `REWRITE_VERSION` is `"ne3"`, and it is part of each package name, so no `ne2` package is used again. The `ne2` packages were deleted before the conversion of Step 6d.
3. **A guard in the conversion.** `constant_concats` lists the concat ops with two or more constant inputs in the MIL program of `ct.convert`. `convert_module` raises `NeuralEngineGraphError` before the save when the list is not empty, and each conversion line has the field `constant_concats`.
4. **A part test.** Test T16 converts only the `ne3` embeddings of ViT-S/16 at 448 px and checks them on CPU_AND_NE, so a change of the fault in a new macOS or coremltools shows at once.

Why:

- **It fixes the fault and is exact.** With `ne3`, 15 of 15 CPU_AND_NE parity rows pass the fp16 thresholds of ADR 0003 (Step 6c: 2 of 15). The worst values are frame 0.99983 (ViT-S+/16, 224 px), patch mean 0.99959 and patch 1st percentile 0.99377 (ViT-H+/16, 224 px). 30 of 30 rows on CPU_AND_GPU and ALL still pass.
- **The whole graph runs on the Neural Engine.** The compute plan gives a Neural Engine cost share of 1.000 in 25 of 25 packages, with every concat on the Neural Engine.
- **The two streams add up.** The stream ratio (Core ML on the Neural Engine and compiled mlx on the GPU at the same time, against mlx alone) is 1.324 and 1.318 at P1 and 1.122 and 1.184 at P3 (runs 1 and 2), with valid outputs.
- **It is the smallest change.** One function and one guard. The other exact forms of probe E3 (pad and add, the concat before the transpose, a shift matmul) are also right but change more.

We considered:

- **Two packages in a chain** (option B): the embeddings on CPU_ONLY, the blocks on CPU_AND_NE. One more call and a token copy for each batch, two loads, and the CPU work can slow the mlx process. Not needed, because option A fixes the fault in one package.
- **Bound the fault, no fix** (option C): the Neural Engine path stays limited to ViT-S/16 and ViT-S+/16 at 224 px and batch size 1.
- **The layout of Apple's ml-ane-transformers**: not needed for the parity. It can help the Neural Engine speed at 448 and 512 px. Q2 of `explore.html` asked for a later step on this speed; the user said no (2026-11-09).

Consequences:

- The fix rests on a fault that we observed, not on a documented rule. A new macOS or coremltools can change the fault; T15 and T16 must run again after each upgrade (ADR 0015 already pins coremltools 9.0).
- `_is_constant` counts a value as constant when a `const` op makes it or when it has a value (`val`), also from a non-const op (finding V7 of the Step 6d `verify.html`, fixed on 2026-11-09). With coremltools 9.0 the prefix is a `const` op also at batch size 8, so the old check found the same concats; the whole `ne3` backbone has no flagged concat (T15).
- Every package of a new shape must be converted again with `ne3`. The first load of a new package compiles it for the Neural Engine and takes up to 118.8 s (ViT-H+/16, 512 px, batch size 8); a cached load takes at most 0.25 s. The bundle cache of Core ML grew by about 17 GB during the parity job of Step 6d.
- At 224 px the batch API (8 calls of the batch-size-1 package in one `predict`) stays faster than the batch-size-8 package (ViT-S/16: 666 against 494 img/s). At 512 px both forms are equal.

Status: accepted on 2026-11-09 (Step 6d Close), from option A of `specs/06d-neural-engine-fault/explore.html`.
