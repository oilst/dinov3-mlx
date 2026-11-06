# Separate parity thresholds for bf16

The parity tests use one threshold set for each dtype. fp16 and bf16 have different sets:

| Dtype | Frame | Patch mean | Patch 1st percentile |
| --- | --- | --- | --- |
| fp32 | ≥ 0.9999 | ≥ 0.9999 | ≥ 0.9999 |
| fp16, cast mode and autocast mode | ≥ 0.999 | ≥ 0.999 | ≥ 0.99 |
| bf16, cast mode and autocast mode | ≥ 0.998 | ≥ 0.995 | ≥ 0.96 |

The patch minimum is report only. The thresholds hold for every backend and model size (`THRESHOLDS` in `src/dinov3_apple/parity.py`).

Step 2 first proposed one set for fp16 and bf16 together (0.999, 0.999 and 0.99). On `torch-mps`, 5 bf16 configurations failed it: the worst values were frame 0.998365, patch mean 0.995177 and patch 1st percentile 0.968878. A probe showed that `torch-cpu` in bf16 gives the same values, so the cause is bf16 arithmetic, not the backend. ViT-H+/16 at 224 px is sensitive to its input: in fp32, a rounding of only the input pixels to bf16 gives a patch 1st percentile of 0.9925. The bf16 set takes the worst measured values, rounded down. fp16 passes the first proposal with margin (worst frame 0.999986, patch 1st percentile 0.999781), so it keeps it.

We considered two other options:

- Keep one set and mark the 5 configurations as expected failures. This keeps tight limits elsewhere, but it needs a list of exceptions for each configuration, and `mlx` would need its own list.
- One threshold set for each model size (Option C of D6 in `specs/02-reference/explore.html`). The mission asks for one set for each dtype, and a new torch version can move many numbers.

Consequence: `mission.md` and Table 9 of `docs/plan.html` give the bf16 set. An `mlx` result in bf16 passes with less parity than an fp16 result. The ViT-7B/16 reference file does not exist yet; when it does, its bf16 values must also pass this set, or this ADR changes.

Status: accepted on 2026-11-06 (the user's decision after the bf16 probe of Step 2).
