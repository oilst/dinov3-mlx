# The name dinov3-mlx

Until Step 7 the package was `dinov3_apple`, the distribution `dinov3-apple`, and the cache folder `~/.cache/dinov3-apple/`. "Apple" is a trademark of Apple Inc. A public package with a trademark in its name can look like a product of Apple (Section 2.6 of `specs/07-release/explore.html`).

Decision (D4 B, the user, 2026-11-09):

1. The import name is `dinov3_mlx`, the distribution name `dinov3-mlx`, the repo `github.com/oilst/dinov3-mlx`, and the cache folder `~/.cache/dinov3-mlx/`.
2. The README says "on Apple silicon" in its description. This is a factual use of the name of the hardware, not a product name.
3. The rename moved the local cache folder once. No weights were converted again (W1).
4. Only `specs/07-release/` and this ADR may name the old names. Test T10 (`test_no_old_package_name`) scans every tracked file and exempts these records.

Why:

- The name says what the package is: DINOv3 in MLX. MLX runs only on Apple silicon, so the platform is clear.
- The rename is mechanical and has a test. It costs less before the first public push than after it.

We considered:

- **Keep `dinov3-apple`** (option A). No work, but the trademark stays in the name.
- **Another name** (option C). No name came up that says more than `dinov3-mlx`.

Consequences:

- The name is close to three other MLX ports of DINOv3 (`mlx_dinov3`, `mlx-dinov3`, `MLXDINOv3`). The README links the comparison with two of them.
- The legal check of the user (gate G1) covers the name too.
- The Step 7 plan reserved number 0017 for this ADR. Step 6e closed first and took 0017 (ADR 0017), so this ADR is 0018, and T10 exempts `docs/adr/0018-`.

Status: accepted on 2026-11-09 (D4 B of `specs/07-release/explore.html`); written at the Step 7 Close on 2026-11-14.
