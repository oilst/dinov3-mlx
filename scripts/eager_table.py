"""Write results/eager-pairs-table.md from results/speed-pairs-eager.jsonl: the paired ratio of
compiled mlx against eager torch-mps at P1 and P2 (W11 of specs/07-release/plan.html).

Eager torch-mps is the form of Hugging Face transformers without torch.compile. The rows use the
rules of Table 1 of results/speedup-table.md (pair_rows, ADR 0004, D21 of Step 6).

    uv run python scripts/eager_table.py
"""

import argparse
from pathlib import Path

from speedup_table import _table, pairs_section, read_lines

REPO_ROOT = Path(__file__).parents[1]
RESULTS = REPO_ROOT / "results" / "speed-pairs-eager.jsonl"
TABLE = REPO_ROOT / "results" / "eager-pairs-table.md"
HEADER = [
    "Priority",
    "Dtype",
    "Output",
    "mlx mode",
    "torch-mps mode",
    "Run 1 median",
    "Run 2 median",
    "Pooled median",
    "Spread between runs",
    "Verdict (D21)",
    "Ratios",
]


def write_table(lines: list[dict]) -> str:
    """Return the Markdown text of results/eager-pairs-table.md.

    Args:
        lines: The lines of results/speed-pairs-eager.jsonl.

    Returns:
        The text: the 12 verdict rows of P1 and P2 and the source line.
    """
    out = [
        "# Speedup of compiled mlx against eager torch-mps",
        "",
        "W11 of `specs/07-release/plan.html`. Ratio = mlx throughput ÷ torch-mps throughput in the",
        "same round of a paired run. The mlx side runs the compiled forward pass; the torch-mps",
        "side runs eager, as Hugging Face transformers does without torch.compile. Each side takes",
        "its fastest valid dtype mode of each dtype (ADR 0004). P1 = ViT-S/16, 224 px; P2 =",
        "ViT-H+/16, 512 px; batch size 8.",
        "",
        *_table(HEADER, pairs_section(lines)),
        "",
        "Source: `results/speed-pairs-eager.jsonl` (`scripts/bench_speed.py pairs --torch-eager`).",
    ]
    return "\n".join(out) + "\n"


def main() -> None:
    """Write the table from the results file."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--table", type=Path, default=TABLE)
    args = parser.parse_args()
    args.table.write_text(write_table(read_lines(args.results)))
    print(args.table)


if __name__ == "__main__":
    main()
