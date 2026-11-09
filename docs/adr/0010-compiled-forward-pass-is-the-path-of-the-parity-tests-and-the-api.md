# The compiled forward pass is the path of the parity tests and the API

The compiled forward pass is the forward pass that the parity tests and the benchmarks run. The API of Step 7 will run it too. `compile_backbone(model)` in `src/dinov3_apple/mlx_model.py` makes it:

- One `mx.compile` function for each backbone, over the whole forward pass. It has one trace for each input shape (batch size, height, width). It does not use `shapeless=True`.
- The weights enter as inputs (`inputs=model.state`). So the output follows a weight change after the compile.
- The RoPE tables enter as arguments. Python reads them from the RoPE cache, outside the trace.
- Call `compile_backbone` after the last weight change (the load, `nn.quantize`). The caller holds the function, never the model.
- `run_mlx(model, pixels, dtype_mode, forward)` runs the compiled forward pass when it gets `forward`. The parity tests give it the compiled forward pass of the model slot (`tests/conftest.py`).
- Two tests assert compiled equal to eager: `test_compiled_equals_eager` (W1, the 12 configurations of each model size and dtype mode) and `test_compiled_equals_eager_small` (T34, the small models). On the GPU the largest difference must be 0.0.
- The diagnostics stay on the eager forward pass: the hidden states (`run_mlx_hidden_states`), the token range (`run_mlx_token_range`) and the part tests.

Why:

- **The tests check what users run.** The benchmarks and the API of Step 7 run the compiled forward pass. If the parity tests ran only the eager forward pass, no parity row would cover that path.
- **Bit equality on the GPU.** In the probes, 360 of 360 parity rows and all P1 and P2 cells were bit for bit equal on the GPU (Section 2.2 of `specs/06-speed/explore.html`). So the test of compiled against eager keeps the eager diagnostics valid.
- **Speed.** The whole forward pass gives all fusions of the element-wise chains. At P1 the kernel count drops from 841 (eager) to 637 (compiled). The change ratio C1 (compiled / eager) at P1 is 1.051 in fp32 and 1.293 to 1.305 in the four fp16 and bf16 dtype modes (5 rounds, Table 5 of `specs/06-speed/verify.html`). The Python graph build drops from 0.26 to 0.11 ms (12 blocks) and from 0.77 to 0.32 ms (32 blocks).
- **Weights as inputs.** `inputs=model.state` follows a weight change for 0.010 to 0.026 ms per call. The same pattern serves the compiled training step of Step 8a.

We considered these other options (D1 and D4 of `specs/06-speed/explore.html`):

- Compiled element-wise helpers only (D1 option B). The same fused kernels and no stale-weight hazard. But the Python graph build stays: about 0.13 ms more per call than option A, about 5 % of a ViT-S/16 call at batch size 1.
- The whole forward pass with `shapeless=True` (D1 option C). It does not work in mlx 0.32.4: "Slice cannot infer output shapes" at the first call, and a reshape error in the patch embedding at a new shape. The retrace that it saves costs only 2 to 6 ms.
- One compiled function for each block (D1 option D). No fusion across blocks, and the Python loop over the blocks stays.
- Eager parity tests, plus one test of compiled against eager (D4 option a). No change of the parity tests, but the tests do not run the path that the benchmarks and the API run.
- Both forward passes in every parity row (D4 option c). Twice the forward passes, for little more information when the equality test passes.

Consequences:

- Compiled equals eager (AC1, AC2 of `verify.html`). GPU, small models: 0.0 in 15 of 15 cases. With the weights: 180 of 180 configurations of the default session with 0.0, and 330 finite configurations of the slow session with 0.0 (282 of ViT-S/16 to ViT-H+/16, 48 of ViT-7B/16). The 18 configurations of `EXPECTED_FP16_OVERFLOW` (ViT-L/16 and ViT-H+/16 in fp16-cast) are non-finite at the same positions in both forms.
- Parity on the compiled forward pass (AC3). 282 of 300 rows pass, 18 are expected failures (fp16 overflow), and ViT-7B/16 passes 48 of 48 (its 12 fp32 rows stay skipped by ADR 0008). The thresholds of ADR 0003 do not change.
- The CPU stream (`DINOV3_MLX_DEVICE=cpu`) is not bit for bit equal. The compiled CPU kernel rounds once for each fused chain, and the eager CPU path rounds after each operation. fp32: 3.1e-6, inside the tolerance 1e-5. The four fp16 and bf16 dtype modes miss 1e-5 (0.0024 to 0.047), so `test_compiled_equals_eager_small` makes them report only, with a fault bound for each mode (`CPU_FAULT_BOUND`: 0.03, 0.25, 0.015, 0.1, at least 5 times the measured value). A red case with the F1 fault of Step 4 gives 4.67 to 4.68, far above each bound. This is deviation 1 of the build. The planning session decided it, not the user. The user confirmed it on 2026-11-09 ("go with your suggestion"; finding V13 of `verify.html`, Section 14).
- Hazards of a compiled function (AC5). A weight change after the compile gives the new output (T35). A function stored on the model keeps the weights alive until `gc.collect()`, so the caller holds it; after `del` of the model and of the function, the memory is back within 1 MB (T36).
- Each new input shape costs one trace (a few ms, 1.5 to 6 ms in the probes) and each process one Metal compile. The API of Step 7 and the training step of Step 8a must hold one compiled function for each backbone and reuse it across batches.
- At P1 the compiled forward pass of `mlx` is still slower than `torch.compile` on `torch-mps` in fp16 and bf16 (pooled paired ratio 0.867 to 0.900, ADR 0009). The P1 profile gives a host share of 3.3 to 3.5 %, so the GPU kernels decide, not Python.
- A later mlx version can change `mx.compile`. The equality tests catch a difference on the GPU.

Status: accepted on 2026-11-07. The user answered "Take all recommendations" to the questions of `specs/06-speed/explore.html`. Answer 6 of Table 41 of that page applies the recommended option of each decision D1 to D23, here D1 option A and D4 option b (Table 2 of `specs/06-speed/plan.html`). The fault bounds of the CPU stream (deviation 1) wait for the confirmation of the user.
