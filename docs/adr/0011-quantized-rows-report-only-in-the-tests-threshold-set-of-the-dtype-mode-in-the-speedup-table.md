# Quantized rows: report only in the tests, the threshold set of the dtype mode in the speedup table

ADR 0003 has no thresholds for quantized weights. Step 6 uses two rules, one for the tests and one for the speedup table:

- **Tests: report only, with a fault bound.** `test_quantized` (W2, `tests/test_mlx_quantized.py`) records the parity of each quantized configuration against the reference output and against the unquantized output of the same dtype mode, and "passes `THRESHOLDS[mode]`: yes or no". It asserts only the fault bound `FAULT_BOUND`: 8 bits frame 0.99, patch mean 0.99, patch 1st percentile 0.90; 4 bits frame 0.75, patch mean 0.75 (the patch 1st percentile of 4 bits is report only). A fault bound catches only faults, for example the F1 cast of Step 4 (frame cosine 0.716). It is not a threshold.
- **Speedup table: the threshold set of the dtype mode.** A quantized row is valid only if image 0 passes `THRESHOLDS[mode]` of its dtype mode (ADR 0003). A row that is not valid shows its parity values and no speedup verdict.
- **The key (D22).** A quantized row keeps the name of its dtype mode and adds the field `bits` (none for unquantized weights, 8 or 4) to `BenchConfig` and to each result line. `MLX_DTYPE_MODES`, `THRESHOLDS` and the dtype mode keep one meaning. Old result lines without `bits` read as unquantized. The memory need and the MLX limit of a quantized worker use the quantized bytes.
- **The `mlx` side of a speedup.** Quantized rows never join the ADR 0004 choice of the fastest valid dtype mode. Each quantized row gets its own speedup row next to the unquantized row of the same dtype mode.
- **The mission rule.** The quantized rows stay inside the rule "speedup above 1". If a quantized speedup is not above 1, or a quantized row is not valid, `verify.html` gives the reason. `mission.md` does not change.

ADR 0003 does not change.

Why:

- Only the user can set thresholds. Report only in the tests follows Table 9 of `docs/plan.html`, which said "`test_quantized`: report only" before Step 6, and keeps ADR 0003 as it is. The user can decide on own thresholds after the numbers exist.
- The speedup table needs a rule for a valid quantized row anyway. The set of the dtype mode needs no new numbers, and it applies the same quality bar as the unquantized rows. With the probe values, all 8-bit rows pass and no 4-bit row passes.
- A field `bits` keeps the name of a dtype mode for one thing: the matmul dtype and the stream dtype. A new mode name would mix the matmul dtype and the weight format, and the ADR 0004 choice on the `mlx` side would mix quantized and unquantized rows.

We considered these other options (D18 and D22 of `specs/06-speed/explore.html`):

- Report only everywhere (D18 option A alone). No guard against a slow loss of quantized parity, and no rule for the speedup table.
- The set of the dtype mode in the tests too (D18 option B alone). It applies ADR 0003 to a new kind of row in a test that fails. The 8-bit margins are small: the ViT-S/16 frame uses 76 % of the bf16 budget and 61 % of the fp16 budget.
- Own thresholds for quantized weights (D18 option C). 4-bit rows could count, but a 4-bit set would have to accept a patch 1st percentile of about 0.64 (ViT-S+/16). It changes ADR 0003 and `mission.md`.
- New dtype mode names, for example bf16-autocast-q8 (D22 option b). No new key field, but `MLX_DTYPE_MODES`, `THRESHOLDS` and this rule would need new keys.

Consequences:

- Parity in the tests (AC11 of `specs/06-speed/verify.html`). 144 rows, 0 below the fault bound. 8 bits: 72 of 72 pass `THRESHOLDS[mode]`; the smallest values are frame 0.998484, patch mean 0.998698 and patch 1st percentile 0.983754. 4 bits: 0 of 72 pass; frame 0.861780 to 0.994, patch 1st percentile down to 0.749771.
- The speedup table (AC10). 144 quantized cells (ViT-H+/16 and ViT-7B/16 in fp16-autocast and bf16-autocast, 8 and 4 bits). 8 bits: 72 valid, 16 of them above 1 and 56 not above 1. 4 bits: 72 not valid, for example "fails `THRESHOLDS[bf16-autocast]`: frame 0.994344, patch mean 0.996476, patch 1st percentile 0.990915". 0 quantized rows in the ADR 0004 choice of the `mlx` side.
- On this Mac, quantized weights save memory, not time. The 8-bit rows reach a median of 0.88 to 0.94 of the unquantized `mlx` row of the same dtype mode (change ratio C5 at P2: 0.874 and 0.884). The quantized matmul dequantizes each group of weights during the matmul, and at these shapes the GPU is compute bound. The 16 cells of 8 bits above 1 lie where the `torch-mps` side is slow.
- Memory. Load peak of ViT-7B/16: 7.33 GB with 8 bits and 3.98 GB with 4 bits (AC12). The benchmark need of ViT-7B/16 at 512 px, batch size 32, bf16-autocast drops from 27.67 GB to 20.12 GB (8 bits) and 16.10 GB (4 bits).
- One paired round does not decide a quantized cell near 1 (finding V17). Verify measured ViT-7B/16 at 512 px, batch size 8 again, and the fp16-autocast 8-bit cell changed from 0.990 to 1.068.
- The 4-bit rows stay in the results with their parity values. If a later step wants 4-bit weights to count, it needs own thresholds, which is a decision of the user and a change of ADR 0003.

Status: accepted on 2026-11-07. The user answered "Take all recommendations" to the questions of `specs/06-speed/explore.html`. Answer 1 of Table 41 of that page applies the proposal of D18: report only with a fault bound in the tests, the threshold set of the dtype mode in the speedup table. Question 5 (the quantized rows and the mission rule) had no recommendation. For it, the agent applied the option that keeps `mission.md` and the ADRs unchanged (Section 6.1 of `explore.html`). D22 option a is the recommended option of answer 6. Table 2 of `specs/06-speed/plan.html` applies them.
