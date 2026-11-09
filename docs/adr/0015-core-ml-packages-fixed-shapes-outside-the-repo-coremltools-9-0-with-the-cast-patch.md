# Core ML packages: fixed shapes, outside the repo, coremltools 9.0 with the _cast patch

Step 6c converts the rewritten backbone (ADR 0014) into Core ML packages (`specs/06c-neural-engine/`). Three choices are hard to reverse, because every package, result line and test depends on them.

Decision:

1. **Fixed shapes.** One package for each model size, input resolution and batch size. The input `pixels` is fp32 (B, 3, H, W); the output `tokens` is fp32 (B, 5 + N, D). The trace makes the RoPE tables constants of the one shape. The package is an ML Program with fp16 compute precision and the macOS 15 target.
2. **Outside the repo, with every setting in the name.** The compiled models (`.mlmodelc`) live in `~/.cache/dinov3-mlx/coreml/`. `convert_backbone` deletes the `.mlpackage` after the compile: it holds the same weights again (ViT-H+/16: 1.68 GB), and the workers load only the `.mlmodelc`. A full disk during the build (deviation 8 of the hand-over) showed the cost of two copies. `package_path` gives the name `<slug>-<side>-b<batch>-<rewrite version>-s256-r<r>-ct9.0.mlpackage`, for example `vits16-224-b1-ne2-s256-r1-ct9.0.mlpackage` (the plain graph: `-plain-ct9.0`). A change of the rewrite (`REWRITE_VERSION`) or of coremltools gives a new name, so a stale package never serves a new configuration. The build used this once: ε / r² changed the rewrite version from `ne` to `ne2`.
3. **coremltools 9.0 in the dependency group `coreml`, with the `_cast` patch.** coremltools 9.0 calls `int()` on a constant array with one value in the torch op `aten::Int`; numpy 2.5.4 refuses this. `patch_coremltools_cast` points the torch frontend at `_cast_with_item` at run time. It does not edit the installed package, and it raises `CoremltoolsVersionError` for any other coremltools version.

Why:

- **Enumerated or flexible shapes need RoPE tables for each shape in the graph**, and the Neural Engine compiles a static graph for each shape anyway (D5 of `explore.html`).
- **The packages are large and derived.** The 26 compiled models take about 12 GB (`results/coreml-table.md`, Table 1). The weights live outside the repo (CLAUDE.md); a package is a copy of the weights.
- **The patch is small and checked.** Test T13 checks the patch, the version guard and that `ops.py` keeps its SHA-256. Test T6 converts a small model without the patch (TypeError) and with it. A new coremltools release fails these tests at once (risk R55).

We considered:

- `torch.export` instead of `torch.jit.trace`: it works with the same ops and parity (probe p11). It is the fallback when a torch release breaks the trace.
- An edit of the installed `ops.py`: lost at the next `uv sync`, and not visible in the repo.
- A pin of numpy below 2.5: changes the version of every other step.

Consequences:

- Core ML keeps a compiled bundle of each model and process in `~/Library/Caches/python/com.apple.e5rt.e5bundlecache`; it grew to about 120 GB in the night of Steps 6b and 6c. The user decides about it.
- A new input resolution or batch size needs a new conversion: 2 s for ViT-S/16, about 82 s for ViT-H+/16 at 512 px batch size 8, plus the first load on the Neural Engine (up to about 2 min).
- `uv run --group coreml` installs coremltools; the tests of `tests/test_coreml.py` skip without it.
- Remove `_cast_with_item` and `patch_coremltools_cast` when a coremltools release after 9.0 fixes `_cast`.

Status: accepted on 2026-11-08 by the Step 6c build, from D1, D3 and D5 of `specs/06c-neural-engine/explore.html` (the user asked the agent to decide, night of 2026-11-07).
