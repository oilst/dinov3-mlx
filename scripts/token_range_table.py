"""Turn the token-range lines of the mlx tests into results/token-range-mlx.md (D6 of
specs/05-all-sizes/explore.html).

Test W7 writes one line of kind "token-range" for each model size, dtype mode and configuration
to tests/fixtures/generated/parity/mlx.jsonl. For each configuration the last line counts. The
values are report only. A non-finite value is a result: it shows the first block that overflows.

Usage:
    uv run python scripts/token_range_table.py
"""

import argparse
import math
from datetime import date
from pathlib import Path

from parity_table import latest_lines, parity_log, table_order

from dinov3_apple.stored import configurations

REPO_ROOT = Path(__file__).parents[1]
RESULTS = REPO_ROOT / "results" / "token-range-mlx.md"
FP16_MAX = 65_504.0
BRANCHES = ["attention", "ff_hidden", "ff_out"]


def token_name(index: int) -> str:
    """Return the name of a token index: CLS, R1 to R4, or the patch number."""
    if index == 0:
        return "CLS"
    if index <= 4:
        return f"R{index}"
    return f"patch {index - 5}"


def _largest_finite(values: list[float]) -> tuple[float, int] | None:
    """Return the largest finite value and its index, or None if no value is finite."""
    finite = [(value, i) for i, value in enumerate(values) if math.isfinite(value)]
    return max(finite) if finite else None


def _number(value: float | None) -> str:
    return "–" if value is None else f"{value:,.1f}"


def _summary(line: dict) -> dict:
    """Return the values of one row: the largest finite token value, its block and token, the
    share of the largest fp16 value, the first non-finite block and the largest branch values."""
    values = line["values"]
    largest = _largest_finite(values["tokens"])
    first = next((i for i, ok in enumerate(values["tokens_finite"]) if not ok), None)
    branches = {}
    for name in BRANCHES:
        found = _largest_finite(values[name])
        branches[name] = None if found is None else found[0]
    return {
        "largest": None if largest is None else largest[0],
        "block": None if largest is None else largest[1],
        "token": None if largest is None else token_name(values["tokens_where"][largest[1]]),
        "first_non_finite": first,
        **branches,
    }


def _share(value: float | None) -> str:
    return "–" if value is None else f"{value / FP16_MAX:.1%}".replace("%", " %")


def _dash(value: object | None) -> str:
    return "–" if value is None else str(value)


def _sorted(lines: list[dict]) -> list[dict]:
    config_order = [config.id for config in configurations()]

    def key(line: dict) -> tuple[int, int, int, str]:
        config = line["config"]
        position = config_order.index(config) if config in config_order else len(config_order)
        return (*table_order(line["model_size"], line["dtype_mode"]), position, config)

    return sorted(lines, key=key)


def configuration_rows(lines: list[dict]) -> list[str]:
    """Return one Markdown table row for each configuration."""
    rows = []
    for line in _sorted(lines):
        s = _summary(line)
        rows.append(
            f"| {line['model_size']} | {line['dtype_mode']} | {line['config']} | "
            f"{_number(s['largest'])} | {_dash(s['block'])} | {_dash(s['token'])} | "
            f"{_share(s['largest'])} | {_dash(s['first_non_finite'])} | "
            + " | ".join(_number(s[name]) for name in BRANCHES)
            + " |"
        )
    return rows


def summary_rows(lines: list[dict]) -> list[str]:
    """Return one Markdown table row for each model size and dtype mode: the configuration with
    the largest finite token value, and the count of configurations with a non-finite value."""
    groups: dict[tuple[str, str], list[dict]] = {}
    for line in _sorted(lines):
        groups.setdefault((line["model_size"], line["dtype_mode"]), []).append(line)
    rows = []
    for (model_size, dtype_mode), group in groups.items():
        summaries = [(_summary(line), line["config"]) for line in group]
        finite = [pair for pair in summaries if pair[0]["largest"] is not None]
        firsts = [s["first_non_finite"] for s, _ in summaries if s["first_non_finite"] is not None]
        first = _dash(min(firsts) if firsts else None)
        if finite:
            s, config = max(finite, key=lambda pair: pair[0]["largest"])
            largest = f"{_number(s['largest'])} | {config} | {s['block']} | {s['token']}"
            share = _share(s["largest"])
        else:
            largest, share = "– | – | – | –", "–"
        rows.append(
            f"| {model_size} | {dtype_mode} | {largest} | {share} | "
            f"{len(firsts)} of {len(group)} | {first} |"
        )
    return rows


def write_table(lines: list[dict]) -> str:
    """Return the Markdown text of results/token-range-mlx.md.

    Args:
        lines: The JSON lines of the mlx tests. Only the lines of kind "token-range" count.

    Returns:
        The text with a summary for each model size and dtype mode and a row for each
        configuration.
    """
    lines = [line for line in lines if line["kind"] == "token-range" and line.get("values")]
    versions = sorted({line["mlx"] for line in lines if "mlx" in line})
    branch_heads = (
        "Largest attention branch output | Largest feed-forward hidden value | "
        "Largest feed-forward branch output"
    )
    return "\n".join(
        [
            "# Token range of mlx",
            "",
            (
                f"Made by `scripts/token_range_table.py` on {date.today().isoformat()} from test "
                f"W7 (mlx {', '.join(versions)}). Each value is the largest absolute finite value "
                "of image 0 of the configuration. Block 0 is the token sequence before block 1; "
                "block i is the token sequence after block i. Token 0 is the CLS token, R1 to R4 "
                "are the register tokens. The share compares the largest value with the largest "
                "fp16 value, 65,504. Report only."
            ),
            "",
            "## 1 Each model size and dtype mode",
            "",
            (
                "| Model size | Dtype mode | Largest token value | Configuration | Block | Token | "
                "Share of 65,504 | Configurations with a non-finite value | "
                "First non-finite block |"
            ),
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            *summary_rows(lines),
            "",
            "Table 1. The configuration with the largest finite token value.",
            "",
            "## 2 Each configuration",
            "",
            (
                "| Model size | Dtype mode | Configuration | Largest token value | Block | Token | "
                f"Share of 65,504 | First non-finite block | {branch_heads} |"
            ),
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            *configuration_rows(lines),
            "",
            (
                "Table 2. The token range of each configuration. The branch outputs are before "
                "LayerScale."
            ),
            "",
        ]
    )


def main() -> None:
    """Write results/token-range-mlx.md."""
    argparse.ArgumentParser(description=__doc__).parse_args()
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(write_table(latest_lines(parity_log("mlx"))))
    print(f"wrote {RESULTS}")


if __name__ == "__main__":
    main()
