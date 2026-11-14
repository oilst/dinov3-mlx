# Separate projections stay the compiled path, and the package keeps the tile sizes of MLX

The roofline of `docs/learn/06-5-roofline.html` shows that the five matmul parts of a ViT-H+/16 block are compute-bound and reach only part of the matmul roof. Step 6e (`specs/06e-kernels/`) tested two ways to close this gap:

- **Part A, joined projections (change C7).** One matmul for q, k and v, and one for gate and up. The forward pass joins the weights with `mx.concatenate` inside the trace, at each call. `set_joined_projections` in `src/dinov3_mlx/mlx_model.py` selects the form. The parameters keep their names.
- **Part C1, other tile sizes.** A source build of MLX 0.32.4 outside the repo (ADR 0013) with a tile override. 12 tile sizes for each of three targets: fp16 1,280 → 1,280 (T1), bf16 1,280 → 10,240 (T2) and the attention kernel (T3).

Decision:

1. **Both joins stay off.** The compiled forward pass of `compile_backbone`, the parity tests and the public API keep the separate projections of Step 6 (D6 of `specs/06-speed/explore.html` stands). `set_joined_projections` stays in the code for later measurements. No caller of the package turns it on.
2. **The package keeps the released MLX and its tile sizes.** The source build is a probe only. The package does not depend on it.
3. **No pull request to MLX now.** Q3 of `specs/06e-kernels/explore.html` stays with the user. The gains of C1 are small (below), so this step does not prepare one.

Why:

- **The D6 rule says off.** A join is on only if its pooled median at P2 (ViT-H+/16, 512 px, batch size 8) is at least 1.03 and the median of each run is above 1. In the five dtype modes, R-Jqkv gives 0.952 to 0.961, R-Jff 0.989 to 0.997 and R-J 0.940 to 0.954 (`results/joined-table.md`). Every joined form is slower than R. At batch size 1 and at P1 the joined forms are also slower.
- **The outputs are equal.** The first batch of each joined form is equal to R, bit for bit, in all 30 rows. So the choice is a speed choice only.
- **The joined forms need more memory.** The MLX peak of R-J at P2 is up to 546 MB above R (fp32) and 125 to 147 MB above R in the half dtype modes.
- **Other tile sizes gain little.** The sweep gives 1.6 % (T1), 3.6 % (T2) and 5.0 % (T3) on one kernel against the wheel (`results/kernels-c1.md`). Verify (V2 of `specs/06e-kernels/verify.html`) found a drift in the sweep: the T2 value is an artifact (1.001 and 0.997 in the cold rounds), T1 is inside the noise, and T3 needs a confirmation run. For block 1 in fp16-cast, T1 and T3 save about 0.8 % of the time: below the noise of one paired run. T2 is the joined gate and up, which decision 1 keeps off.

We considered:

- **Joined weights stored once** (D1 of the plan). A second copy of 1.15 GB at ViT-H+/16 in fp16, other parameter names, and a change of the strict load and of `nn.quantize`. The concat inside the trace costs about 1 % of block 1 at the memory roof, so a stored join cannot turn a loss of 4 to 6 % into a gain.
- **A join on for single dtype modes.** R-Jff at P2 lies at 0.989 to 0.997: no gain in any dtype mode.
- **A pull request to MLX with other tile sizes.** The largest gain is 5.0 % on the attention kernel alone. Three attention tile sizes with bk = 16 give wrong outputs, so any new tile size needs its own parity check.

Consequences:

- **The cause of the loss is not known.** Table 1 of the roofline page predicts a faster matmul for the joined form, and Table 5 predicted a saving of 8.5 % of block 1. The paired run measures only the whole forward pass. A profile of the parts of the joined form can show where the time goes.
- **The P1 rows of C7 read low.** The harness switches the flags before and after each batch (`_JoinedForm` in `src/dinov3_mlx/mlx_bench.py`). At P1 this costs about 0.15 ms per batch, about 4 % of a batch. ViT-S/16 has no gate, so its R-Jff runs the same graph as R and shows this cost. At P2 the cost is below 0.1 %. The D6 decision uses P2 only, so it does not change. A later change that switches a form at each call must count this cost.
- **AC8 does not apply.** No default changed, so the parity tests run the same path as before.
- **The source build stays outside the repo.** `~/.cache/dinov3-mlx/mlx-src/` holds the clone, the venv, the patch and the build log (422 MB). The user can delete it. A new probe needs the Metal Toolchain again.

Status: accepted on 2026-11-14 (Step 6e Close), from the D6 rule of `specs/06e-kernels/plan.html` and the measured values of B1 to B4.
