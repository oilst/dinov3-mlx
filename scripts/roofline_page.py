"""Write docs/learn/06-5-roofline.html: the roofline of one block of ViT-H+/16 at 512 px.

The inputs are four result files: the part times of the P2 profile (scripts/bench_speed.py profile
--priority P2), the roofs of this Mac (scripts/probe_roofs.py), the paired run of the joined
projections (results/speed-joined.jsonl, Step 6e) and the C1 tile probe (results/kernels-c1.jsonl,
Step 6e). The FLOPs and the bytes of each part come from dinov3_mlx.flops. No GPU, no model.

    uv run python scripts/roofline_page.py
"""

import argparse
import html
import json
import math
from pathlib import Path

import joined_table
import numpy as np
import probe_kernels

from dinov3_mlx.flops import MODEL_SHAPES, PART_NAMES, image_flops, part_bytes, part_flops, tokens

ROOT = Path(__file__).resolve().parents[1]
MODEL_SIZE, SIDE, BATCH_SIZE, BYTES_PER_VALUE = "ViT-H+/16", 512, 8, 2
MODES = {"fp16-cast": ("fp16", "s1", "circle"), "bf16-cast": ("bf16", "s2", "square")}
MATMUL_PARTS = ["qkv_proj", "sdpa", "o_proj", "ff_in", "ff_out"]
POINTS = [*PART_NAMES, "block"]
# The configurations of B1 of Step 6e in the order of Table 6: (priority, batch size, dtype mode).
JOINED_CELLS = [
    *(("P2", 8, m) for m in ("fp32", "fp16-cast", "bf16-cast", "fp16-autocast", "bf16-autocast")),
    *(("P2", 1, m) for m in ("fp16-cast", "bf16-cast")),
    *(("P1", 8, m) for m in ("fp32", "fp16-cast", "bf16-cast")),
]
# The parts of block 1 in fp16-cast that run the shapes of the C1 targets T1 (1,280 → 1,280) and T3
# (SDPA). T2 is the joined gate and up, which the default form does not run (D6 of Step 6e).
TILE_PARTS = {"T1": ["qkv_proj", "o_proj"], "T3": ["sdpa"]}
JOINED_CODE = """\
# src/dinov3_mlx/mlx_model.py (Step 6e)
def _joined(layers: list[nn.Linear], x: mx.array) -> mx.array:
    weight = mx.concatenate([layer.weight for layer in layers])  # (3D, D): at each call (D1)
    return joined_linear(x, weight, _joined_bias(layers))  # zeros for a missing bias (D3)

def joined_linear(x: mx.array, weight: mx.array, bias: mx.array | None) -> mx.array:
    if bias is None:
        return x @ weight.T  # the call of nn.Linear without a bias
    return mx.addmm(bias, x, weight.T)  # the call of nn.Linear with a bias: bit equal (D5)

# Attention.qkv, joined form
y = _joined([self.q_proj, self.k_proj, self.v_proj], x)  # (B, T, 3D)
y = y.reshape(b, t, 3, self.heads, -1).transpose(2, 0, 3, 1, 4)  # (3, B, heads, T, head_dim)
return y[0], y[1], y[2]  # q, k, v

# SwiGLU.hidden, joined form
gate, up = mx.split(_joined([self.gate_proj, self.up_proj], x), 2, axis=-1)  # (B, T, F) each

# Select the form before compile_backbone: a trace reads the flags when it records (D2)
set_joined_projections(model, qkv=True, ff=True)  # ValueError for a quantized model (D4)
forward = compile_backbone(model)
"""
# Element-wise FLOPs for each value: an estimate for the parts that flops.py counts as 0 FLOPs.
# Layer norm: mean, variance, normalize, scale and bias. RoPE: 2 multiplies and 1 add on q and k.
# LayerScale: 1 multiply and the residual add. SwiGLU: silu(gate) · up.
ELEMENTWISE_PER_VALUE = {
    "norm1": 8,
    "norm2": 8,
    "rope": 6,
    "layer_scale1": 2,
    "layer_scale2": 2,
    "activation": 6,
}


def read_lines(path: Path) -> list[dict]:
    """Return the lines of a JSON-lines file."""
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def elementwise_flops(name: str, batch_size: int, num_tokens: int) -> int:
    """Return the estimated FLOPs of one element-wise part of one ViT-H+/16 block.

    Args:
        name: A part of ELEMENTWISE_PER_VALUE.
        batch_size: The batch size B.
        num_tokens: The token count T.

    Returns:
        The FLOPs: the per-value count times B T D, or times B T F for the activation.
    """
    shape = MODEL_SHAPES[MODEL_SIZE]
    width = shape.ff if name == "activation" else shape.dim
    return ELEMENTWISE_PER_VALUE[name] * batch_size * num_tokens * width


def block_points(profile: dict) -> list[dict]:
    """Return one point for each part of block 1 and one for the whole block, for one profile line.

    Args:
        profile: One line of the P2 profile, with parts_s and block_s.

    Returns:
        Dicts with part, mode, flops, bytes, seconds, intensity (FLOP/B), tflops, gbs and estimated
        (True when the FLOPs come from ELEMENTWISE_PER_VALUE).
    """
    shape, num_tokens = MODEL_SHAPES[MODEL_SIZE], tokens(SIDE)
    flops = part_flops(shape, BATCH_SIZE, num_tokens)
    nbytes = part_bytes(shape, BATCH_SIZE, num_tokens, BYTES_PER_VALUE)
    counted = {
        name: flops[name] or elementwise_flops(name, BATCH_SIZE, num_tokens) for name in PART_NAMES
    }
    seconds = dict(profile["parts_s"], block=profile["block_s"])
    counted["block"], nbytes = sum(counted.values()), dict(nbytes, block=sum(nbytes.values()))
    points = []
    for name in POINTS:
        f, b, s = counted[name], nbytes[name], seconds[name]
        points.append(
            {
                "part": name,
                "mode": profile["dtype_mode"],
                "flops": f,
                "bytes": b,
                "seconds": s,
                "intensity": f / b,
                "tflops": f / s / 1e12,
                "gbs": b / s / 1e9,
                "estimated": name in ELEMENTWISE_PER_VALUE,
            }
        )
    return points


def attainable(roofs: dict, mode: str, intensity: float) -> float:
    """Return the roof in TFLOP/s at an arithmetic intensity: min(matmul roof, memory roof · I)."""
    return min(roofs[MODES[mode][0]], roofs["memory_gbs"] * intensity / 1000)


def fmt(value: float, digits: int = 1) -> str:
    """Return value with thousands separators and a fixed number of digits."""
    return f"{value:,.{digits}f}"


class Axes:
    """The pixel mapping of one plot: log x, log or linear y."""

    def __init__(
        self,
        size: tuple[int, int],
        x_range: tuple[float, float],
        y_range: tuple[float, float],
        y_log: bool = True,
    ) -> None:
        self.w, self.h = size
        self.left, self.right, self.top, self.bottom = 70, 20, 20, 52
        self.x0, self.x1 = math.log10(x_range[0]), math.log10(x_range[1])
        self.y_log = y_log
        self.y0, self.y1 = (math.log10(v) for v in y_range) if y_log else y_range

    def x(self, value: float) -> float:
        """Return the pixel x of an intensity."""
        span = self.w - self.left - self.right
        return self.left + (math.log10(value) - self.x0) / (self.x1 - self.x0) * span

    def y(self, value: float) -> float:
        """Return the pixel y of a rate."""
        t = math.log10(value) if self.y_log else value
        return (
            self.h
            - self.bottom
            - (t - self.y0) / (self.y1 - self.y0) * (self.h - self.top - self.bottom)
        )

    def frame(
        self, x_ticks: list[float], y_ticks: list[float], x_title: str, y_title: str
    ) -> list[str]:
        """Return the grid, the ticks, the axes and the axis titles."""
        out = []
        for v in x_ticks:
            x = self.x(v)
            out.append(
                f'<line x1="{x:.1f}" y1="{self.top}" x2="{x:.1f}" y2="{self.h - self.bottom}" class="grid"/>'
            )
            out.append(
                f'<text x="{x:.1f}" y="{self.h - self.bottom + 18}" class="tick" text-anchor="middle">{v:g}</text>'
            )
        for v in y_ticks:
            y = self.y(v)
            out.append(
                f'<line x1="{self.left}" y1="{y:.1f}" x2="{self.w - self.right}" y2="{y:.1f}" class="grid"/>'
            )
            out.append(
                f'<text x="{self.left - 8}" y="{y + 4:.1f}" class="tick" text-anchor="end">{v:g}</text>'
            )
        bottom, middle = self.h - self.bottom, (self.top + self.h - self.bottom) / 2
        out.append(
            f'<line x1="{self.left}" y1="{bottom}" x2="{self.w - self.right}" y2="{bottom}" class="axis"/>'
        )
        out.append(
            f'<line x1="{self.left}" y1="{self.top}" x2="{self.left}" y2="{bottom}" class="axis"/>'
        )
        out.append(
            f'<text x="{(self.left + self.w - self.right) / 2:.1f}" y="{self.h - 10}" class="axt" text-anchor="middle">{x_title}</text>'
        )
        out.append(
            f'<text x="16" y="{middle:.1f}" class="axt" text-anchor="middle" transform="rotate(-90 16 {middle:.1f})">{y_title}</text>'
        )
        return out


def mark(axes: Axes, point: dict, roofs: dict, size: float, square: float) -> str:
    """Return the SVG mark of one point and its hit area, with the tooltip text in data-tip."""
    _, cls, form = MODES[point["mode"]]
    cx, cy = axes.x(point["intensity"]), axes.y(point["tflops"])
    share = point["tflops"] / attainable(roofs, point["mode"], point["intensity"]) * 100
    tip = (
        f"{point['part']} · {point['mode']}|{fmt(point['intensity'], 2)} FLOP/B|"
        f"{fmt(point['tflops'], 2)} TFLOP/s · {fmt(point['gbs'])} GB/s|"
        f"{fmt(point['seconds'] * 1e3, 3)} ms|{fmt(share)} % of the roof"
    )
    if point["estimated"]:
        tip += "|FLOPs: estimate"
    attrs = f'class="pt {cls}" data-tip="{tip}" tabindex="0"'
    if form == "circle":
        shape = f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{size}" {attrs}/>'
    else:
        shape = (
            f'<rect x="{cx - square / 2:.1f}" y="{cy - square / 2:.1f}" width="{square:.1f}" '
            f'height="{square:.1f}" rx="1.5" {attrs}/>'
        )
    return shape + f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="12" class="hit" data-tip="{tip}"/>'


def label(axes: Axes, point: dict, text: str, dx: float, dy: float, anchor: str) -> str:
    """Return a direct label next to a point."""
    x, y = axes.x(point["intensity"]) + dx, axes.y(point["tflops"]) + dy
    return f'<text x="{x:.1f}" y="{y:.1f}" class="lab" text-anchor="{anchor}">{text}</text>'


def figure_full(by: dict, roofs: dict) -> str:
    """Return the SVG of Figure 1: the full roofline, log-log."""
    axes = Axes((820, 500), (0.1, 3000), (0.01, 100))
    out = axes.frame(
        [0.1, 1, 10, 100, 1000],
        [0.01, 0.1, 1, 10, 100],
        "Arithmetic intensity (FLOP per byte moved, log scale)",
        "Reached TFLOP/s (log scale)",
    )
    bw = roofs["memory_gbs"]
    for dtype, cls in (("fp16", "s1"), ("bf16", "s2")):
        ridge = roofs[dtype] * 1000 / bw
        pts = [(0.1, bw * 0.1 / 1000), (ridge, roofs[dtype]), (3000, roofs[dtype])]
        path = " ".join(
            f"{'M' if i == 0 else 'L'}{axes.x(a):.1f},{axes.y(b):.1f}"
            for i, (a, b) in enumerate(pts)
        )
        out.append(f'<path d="{path}" class="roof {cls}"/>')
    ridge = roofs["fp16"] * 1000 / bw
    out.append(
        f'<line x1="{axes.x(ridge):.1f}" y1="{axes.y(roofs["fp16"]):.1f}" x2="{axes.x(ridge):.1f}" '
        f'y2="{axes.h - axes.bottom}" class="ridge"/>'
    )
    out.append(
        f'<text x="{axes.x(ridge) + 6:.1f}" y="{axes.h - axes.bottom - 8}" class="note">'
        f"ridge point {ridge:.0f} FLOP/B (fp16)</text>"
    )
    x0, x1 = axes.x(0.6), axes.x(25)
    angle = math.degrees(math.atan2(axes.y(bw * 25 / 1000) - axes.y(bw * 0.6 / 1000), x1 - x0))
    lx, ly = axes.x(3), axes.y(bw * 3 / 1000) - 8
    out.append(
        f'<text x="{lx:.1f}" y="{ly:.1f}" class="roofl" transform="rotate({angle:.1f} {lx:.1f} {ly:.1f})">'
        f"memory roof: {bw:.0f} GB/s</text>"
    )
    out.append(
        f'<text x="{axes.x(3000) - 4:.1f}" y="{axes.y(roofs["fp16"]) - 8:.1f}" class="roofl" text-anchor="end">'
        f"matmul roof: fp16 {roofs['fp16']:.1f}, bf16 {roofs['bf16']:.1f} TFLOP/s</text>"
    )
    zx0, zx1, zy0, zy1 = ZOOM
    out.append(
        f'<rect x="{axes.x(zx0):.1f}" y="{axes.y(zy1):.1f}" width="{axes.x(zx1) - axes.x(zx0):.1f}" '
        f'height="{axes.y(zy0) - axes.y(zy1):.1f}" class="zoom"/>'
    )
    out.append(
        f'<text x="{axes.x(zx1):.1f}" y="{axes.y(zy0) + 18:.1f}" class="lab" text-anchor="end">'
        "matmul parts: Figure 2</text>"
    )
    # bf16 squares first and larger, fp16 circles on top: both stay visible at the same intensity.
    for mode in ("bf16-cast", "fp16-cast"):
        for name in POINTS:
            out.append(mark(axes, by[(mode, name)], roofs, 3.5, 10))
    for name, text, dx, dy, anchor in (
        ("layer_scale1", "layer_scale1, layer_scale2", 0, 24, "middle"),
        ("rope", "rope", 8, -12, "start"),
        ("activation", "activation", -10, -12, "end"),
        ("norm1", "norm1, norm2", 10, -12, "start"),
    ):
        out.append(label(axes, by[("fp16-cast", name)], text, dx, dy, anchor))
    return "".join(out)


ZOOM = (330, 1150, 28, 66)  # the region of Figure 2: x from, x to, y from, y to


def figure_zoom(by: dict, roofs: dict) -> str:
    """Return the SVG of Figure 2: the compute-bound region, log x, linear y."""
    zx0, zx1, zy0, zy1 = ZOOM
    axes = Axes((820, 440), (zx0, zx1), (zy0, zy1), y_log=False)
    out = axes.frame(
        [400, 500, 600, 700, 800, 900, 1000],
        [30, 35, 40, 45, 50, 55, 60, 65],
        "Arithmetic intensity (FLOP per byte moved, log scale)",
        "Reached TFLOP/s (linear scale)",
    )
    for dtype, cls in (("fp16", "s1"), ("bf16", "s2")):
        y = axes.y(roofs[dtype])
        out.append(
            f'<line x1="{axes.left}" y1="{y:.1f}" x2="{axes.w - axes.right}" y2="{y:.1f}" class="roof {cls}"/>'
        )
    out.append(
        f'<text x="{axes.left + 8}" y="{axes.y(roofs["fp16"]) - 7:.1f}" class="roofl">'
        f"fp16 matmul roof {roofs['fp16']:.1f} TFLOP/s</text>"
    )
    out.append(
        f'<text x="{axes.left + 8}" y="{axes.y(roofs["bf16"]) + 16:.1f}" class="roofl">'
        f"bf16 matmul roof {roofs['bf16']:.1f} TFLOP/s</text>"
    )
    for mode in ("bf16-cast", "fp16-cast"):
        for name in [*MATMUL_PARTS, "block"]:
            out.append(mark(axes, by[(mode, name)], roofs, 5.5, 9.5))
    for name, text, dx, dy, anchor in (
        ("block", "block 1 (compiled)", 0, -16, "middle"),
        ("sdpa", "sdpa", -12, 5, "end"),
        ("o_proj", "o_proj", 12, 18, "start"),
        ("qkv_proj", "qkv_proj", -12, 5, "end"),
        ("ff_out", "ff_out", 14, 5, "start"),
        ("ff_in", "ff_in", 14, 5, "start"),
    ):
        out.append(label(axes, by[("fp16-cast", name)], text, dx, dy, anchor))
    return "".join(out)


def linear_time(line: dict, dtype: str, k: int, n: int) -> float:
    """Return the time of one nn.Linear of the roof probe."""
    return next(
        r["seconds"] for r in line["linear"] if (r["dtype"], r["in"], r["out"]) == (dtype, k, n)
    )


def linear_rate(line: dict, dtype: str, k: int, n: int) -> float:
    """Return the TFLOP/s of one nn.Linear of the roof probe."""
    return next(
        r["tflops"] for r in line["linear"] if (r["dtype"], r["in"], r["out"]) == (dtype, k, n)
    )


def joined_cells(lines: list[dict]) -> dict[tuple[str, int, str], dict[str, dict]]:
    """Return the rows of results/joined-table.md by configuration and joined form.

    Args:
        lines: The lines of results/speed-joined.jsonl.

    Returns:
        For each (priority, batch size, dtype mode) of B1, the row of joined_table.joined_rows of
        each joined form (R-J, R-Jqkv and R-Jff).
    """
    cells: dict[tuple[str, int, str], dict[str, dict]] = {}
    for row in joined_table.joined_rows(lines):
        priority, _, _, batch_size, mode = row["cell"]
        cells.setdefault((priority, batch_size, mode), {})[row["form"]] = row
    return cells


def tile_rows(lines: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    """Return the control rows (B3), the sweep rows (B4) and the best valid row of each target of
    results/kernels-c1.md, with the functions of scripts/probe_kernels.py."""
    control = probe_kernels.control_rows(lines)
    newest = {line["target"]: line for line in lines if line.get("kind") == "sweep"}
    rows = probe_kernels.sweep_rows(list(newest.values()), control)
    return control, rows, probe_kernels.best_rows(rows)


def flag_cost(lines: list[dict]) -> tuple[float, float]:
    """Return the cost of the flag switch of the C7 harness at P1 and the batch time of R there.

    ViT-S/16 has no gate, so at P1 the form R-Jff runs the same graph as R. The difference of their
    batch times is the cost outside the graph: _JoinedForm of mlx_bench.py calls
    set_joined_projections before and after each batch.

    Args:
        lines: The lines of results/speed-joined.jsonl.

    Returns:
        The median difference and the median batch time of R, in seconds, over the rounds of the
        P1 lines of both runs.
    """
    costs, batch_times = [], []
    for line in lines:
        if line.get("change") != "C7" or line["status"] != "done" or line["priority"] != "P1":
            continue
        b, throughput = line["batch_size"], line["throughput"]
        for r, jff in zip(throughput["R"]["frame"], throughput["R-Jff"]["frame"], strict=True):
            costs.append(b / jff - b / r)
            batch_times.append(b / r)
    return float(np.median(costs)), float(np.median(batch_times))


def concat_seconds(roofs: dict) -> float:
    """Return the time of the weight concat of both joins in one block in fp16-cast at the memory
    roof: the q, k and v weights (3 D²) and the gate and up weights (2 D F), each read once and
    written once (D1 of specs/06e-kernels/plan.html)."""
    shape = MODEL_SHAPES[MODEL_SIZE]
    values = 3 * shape.dim**2 + 2 * shape.dim * shape.ff
    return 2 * values * BYTES_PER_VALUE / (roofs["memory_gbs"] * 1e9)


def tile_gain(by: dict, best: list[dict]) -> float:
    """Return the time that the best valid tile sizes of T1 and T3 save in block 1 in fp16-cast.

    Each part time shrinks by the ratio of its target to the wheel. The part times come from the
    P2 profile, where each part runs alone, so the value is an estimate.

    Args:
        by: The points of block 1 by (dtype mode, part).
        best: The rows of probe_kernels.best_rows.

    Returns:
        The saved time in seconds.
    """
    ratios = {row["target"]: row["ratio_to_wheel"] for row in best}
    return sum(
        by[("fp16-cast", part)]["seconds"] * (1 - 1 / ratios[target])
        for target, parts in TILE_PARTS.items()
        for part in parts
    )


def _box(x: int, y: int, w: int, h: int, text: str, css: str = "box") -> str:
    """Return a box with one or two centered lines of text (split at |)."""
    lines = text.split("|")
    first = y + h / 2 - (len(lines) - 1) * 8 + 4
    labels = "".join(
        f'<text x="{x + w / 2:.0f}" y="{first + 16 * i:.0f}" class="lab" text-anchor="middle">{t}</text>'
        for i, t in enumerate(lines)
    )
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="5" class="{css}"/>{labels}'


def _flow(x1: int, y1: int, x2: int, y2: int) -> str:
    """Return an arrow from (x1, y1) to (x2, y2)."""
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" class="flow" marker-end="url(#head)"/>'


def figure_joined(line: dict) -> str:
    """Return Figure 3: the separate and the joined projection of q, k and v."""
    d = MODEL_SHAPES[MODEL_SIZE].dim
    one, three = linear_rate(line, "fp16", d, d), linear_rate(line, "fp16", d, 3 * d)
    parts = [
        (
            '<defs><marker id="head" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" '
            'markerHeight="7" orient="auto"><path d="M0,0 L8,4 L0,8 z" class="headk"/></marker>'
            "</defs>"
        ),
        '<text x="20" y="28" class="axt">R: separate projections (Step 6)</text>',
        _box(30, 125, 100, 40, "x|(B, T, D)"),
    ]
    for i, name in enumerate("qkv"):
        y = 55 + 70 * i
        parts += [
            _box(190, y, 130, 40, f"{name}_proj|D → D", "box mm"),
            _flow(130, 145, 188, y + 20),
            _flow(320, y + 20, 348, y + 20),
            f'<text x="356" y="{y + 25}" class="lab">{name}</text>',
        ]
    parts += [
        (
            f'<text x="20" y="266" class="note">3 matmuls of {d:,} → {d:,}: {one:.1f} TFLOP/s '
            "each (fp16, Table 1)</text>"
        ),
        '<line x1="410" y1="40" x2="410" y2="250" class="grid"/>',
        '<text x="430" y="28" class="axt">R-Jqkv: joined projection (Step 6e)</text>',
    ]
    for i, name in enumerate(("Wq", "Wk", "Wv")):
        y = 45 + 35 * i
        parts += [_box(440, y, 56, 26, name), _flow(496, y + 13, 528, 80 + 8 * i)]
    parts += [
        _box(530, 68, 130, 40, "concatenate|(3D, D)", "box dash"),
        _box(440, 185, 90, 40, "x|(B, T, D)"),
        _box(560, 180, 110, 50, "one matmul|D → 3D", "box mm"),
        _box(700, 180, 110, 50, "split into|q, k, v"),
        _flow(600, 108, 610, 178),
        _flow(530, 205, 558, 205),
        _flow(670, 205, 698, 205),
        (
            f'<text x="430" y="266" class="note">1 matmul of {d:,} → {3 * d:,}: {three:.1f} '
            "TFLOP/s (fp16, Table 1)</text>"
        ),
        '<text x="430" y="284" class="note">Dashed: inside the trace, at each call (D1)</text>',
    ]
    return "".join(parts)


def fusion_rows(by: dict, line: dict) -> list[dict]:
    """Return the projection fusion estimate of each dtype mode (D6 b of Step 6).

    qkv_proj runs three D → D matmuls and ff_in two D → F matmuls. The fused forms are one
    D → 3D and one D → 2F matmul. The estimate compares the profile time of the part with the probe
    time of the fused linear layer.
    """
    shape = MODEL_SHAPES[MODEL_SIZE]
    rows = []
    for mode, (dtype, _, _) in MODES.items():
        qkv, ff_in, block = (by[(mode, n)]["seconds"] for n in ("qkv_proj", "ff_in", "block"))
        qkv_fused = linear_time(line, dtype, shape.dim, 3 * shape.dim)
        ff_fused = linear_time(line, dtype, shape.dim, 2 * shape.ff)
        saved = (qkv - qkv_fused) + (ff_in - ff_fused)
        rows.append(
            {
                "mode": mode,
                "qkv": qkv,
                "qkv_fused": qkv_fused,
                "ff_in": ff_in,
                "ff_fused": ff_fused,
                "saved": saved,
                "block": block,
                "share": saved / block,
            }
        )
    return rows


STYLE = """
/* Explainer page. White page, black text, no dark mode. */
:root { --bg: #ffffff; --fg: #111414; --muted: #4a5655; --rule: #d6dedd; --panel: #f5f8f7; --accent: #057379; --s1: #2a78d6; --s2: #eb6834; color-scheme: light; }
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--fg); font: 16px/1.55 -apple-system, BlinkMacSystemFont, "Helvetica Neue", Arial, sans-serif; }
main { max-width: 62rem; margin: 0 auto; padding: 40px 20px 64px; display: grid; gap: 1.75rem; }
main > * { min-width: 0; }
h1 { font-size: 1.9rem; line-height: 1.15; margin: 0; }
h2 { font-size: 1.3rem; margin: 0; }
section { display: grid; gap: 0.8rem; }
p, ul, ol { margin: 0; max-width: 46rem; }
li + li { margin-top: 0.25rem; }
code, pre { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.86em; }
pre { margin: 0; background: var(--panel); border: 1px solid var(--rule); border-radius: 6px; padding: 0.8rem 1rem; overflow-x: auto; }
.table-wrap { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 0.9rem; }
th, td { text-align: left; vertical-align: top; padding: 0.45rem 0.7rem 0.45rem 0; border-bottom: 1px solid var(--rule); }
th { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.04em; color: var(--muted); border-bottom: 1.5px solid var(--fg); }
th.grp { border-bottom: 1px solid var(--rule); }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
tr.total td { font-weight: 600; border-top: 1.5px solid var(--fg); }
.caption, figcaption { font-size: 0.88rem; color: var(--muted); }
figure { margin: 0; display: grid; gap: 0.4rem; }
.fig { border: 1px solid var(--rule); border-radius: 6px; padding: 0.5rem; overflow-x: auto; }
.fig svg { display: block; width: 100%; height: auto; min-width: 640px; }
.note-box { border-left: 3px solid var(--accent); padding: 0.3rem 0 0.3rem 0.9rem; }
nav ol { columns: 2; }
a { color: var(--accent); }
.steps { font-size: 0.92rem; color: var(--muted); }
.legend { display: flex; flex-wrap: wrap; gap: 0.4rem 1.4rem; font-size: 0.88rem; }
.legend span { display: inline-flex; align-items: center; gap: 0.4rem; }
.key { width: 11px; height: 11px; display: inline-block; }
.key.s1 { background: var(--s1); border-radius: 50%; }
.key.s2 { background: var(--s2); border-radius: 1.5px; }
.key.roofk { width: 22px; height: 0; border-top: 2px solid var(--fg); }
.grid { stroke: var(--rule); stroke-width: 1; }
.axis { stroke: var(--muted); stroke-width: 1; }
.tick { fill: var(--muted); font-size: 12px; font-variant-numeric: tabular-nums; }
.axt { fill: var(--fg); font-size: 13px; }
.roof { fill: none; stroke-width: 2; }
.roof.s1 { stroke: var(--s1); }
.roof.s2 { stroke: var(--s2); stroke-dasharray: 6 4; }
.roofl, .lab { fill: var(--fg); font-size: 12.5px; }
.ridge { stroke: var(--muted); stroke-dasharray: 2 3; }
.note { fill: var(--muted); font-size: 12px; }
.zoom { fill: none; stroke: var(--muted); stroke-dasharray: 3 3; }
.pt { stroke: var(--bg); stroke-width: 2; }
.pt.s1 { fill: var(--s1); }
.pt.s2 { fill: var(--s2); }
.pt:focus { outline: none; stroke: var(--fg); }
.hit { fill: transparent; }
#tip { position: fixed; pointer-events: none; background: var(--bg); color: var(--fg); border: 1px solid var(--rule); border-radius: 6px; padding: 0.45rem 0.6rem; font-size: 0.82rem; line-height: 1.4; box-shadow: 0 2px 10px rgba(17, 20, 20, 0.12); font-variant-numeric: tabular-nums; z-index: 5; }
#tip b { display: block; }
.box { fill: var(--panel); stroke: var(--muted); stroke-width: 1; }
.box.mm { stroke: var(--accent); stroke-width: 1.5; }
.box.dash { stroke-dasharray: 4 3; }
.flow { stroke: var(--muted); stroke-width: 1.2; }
.headk { fill: var(--muted); }
"""

SCRIPT = """
(() => {
  const tip = document.getElementById("tip");
  const show = (el, x, y) => {
    const lines = el.dataset.tip.split("|");
    tip.replaceChildren();
    const head = document.createElement("b");
    head.textContent = lines[0];
    tip.append(head, ...lines.slice(1).flatMap((t, i) => (i ? [document.createElement("br"), t] : [t])));
    tip.hidden = false;
    const w = tip.offsetWidth, h = tip.offsetHeight;
    tip.style.left = Math.min(x + 14, window.innerWidth - w - 8) + "px";
    tip.style.top = (y + h + 20 > window.innerHeight ? y - h - 12 : y + 14) + "px";
  };
  document.querySelectorAll("[data-tip]").forEach((el) => {
    el.addEventListener("pointermove", (e) => show(el, e.clientX, e.clientY));
    el.addEventListener("pointerleave", () => (tip.hidden = true));
    el.addEventListener("focus", () => { const r = el.getBoundingClientRect(); show(el, r.right, r.top); });
    el.addEventListener("blur", () => (tip.hidden = true));
  });
})();
"""


def _d6(cell: dict[str, dict]) -> str:
    """Return the D6 decision of one configuration of Table 6: the forms that are on, or off."""
    rows = list(cell.values())
    if not rows[0]["rule_applies"]:
        return "–"
    on = [row["form"] for row in rows if row["on"]]
    return "on: " + ", ".join(on) if on else "off"


def page(profiles: list[dict], line: dict, joined: list[dict], kernels: list[dict]) -> str:
    """Return the explainer page.

    Args:
        profiles: The two lines of the P2 profile, fp16-cast and bf16-cast.
        line: The line of the roof probe.
        joined: The lines of results/speed-joined.jsonl (Step 6e, B1 and B2).
        kernels: The lines of results/kernels-c1.jsonl (Step 6e, B3 and B4).

    Returns:
        The HTML text.
    """
    roofs = line["roofs"]
    points = [p for profile in profiles for p in block_points(profile)]
    by = {(p["mode"], p["part"]): p for p in points}
    bw, ridge = roofs["memory_gbs"], roofs["fp16"] * 1000 / roofs["memory_gbs"]
    shape, num_tokens = MODEL_SHAPES[MODEL_SIZE], tokens(SIDE)
    ips = {p["dtype_mode"]: p["batch_sizes"]["8"]["throughput_ips"] for p in profiles}
    image_tflop = image_flops(shape, SIDE)[0] / 1e12
    matmul = [by[(m, n)] for m in MODES for n in MATMUL_PARTS]
    shares = [p["tflops"] / attainable(roofs, p["mode"], p["intensity"]) * 100 for p in matmul]
    elementwise = [by[(m, n)] for m in MODES for n in ELEMENTWISE_PER_VALUE]
    bw_shares = [p["gbs"] / bw * 100 for p in elementwise]
    block16 = by[("fp16-cast", "block")]
    matmul16 = sum(by[("fp16-cast", n)]["seconds"] for n in MATMUL_PARTS)
    parts16 = sum(by[("fp16-cast", n)]["seconds"] for n in PART_NAMES)
    rest16 = block16["seconds"] - matmul16
    ideal16 = sum(by[("fp16-cast", n)]["flops"] for n in MATMUL_PARTS) / (roofs["fp16"] * 1e12)
    ideal_ew = sum(by[("fp16-cast", n)]["bytes"] for n in ELEMENTWISE_PER_VALUE) / (bw * 1e9)
    fusion = fusion_rows(by, line)
    flops1, bytes1 = (
        part_flops(shape, 1, num_tokens),
        part_bytes(shape, 1, num_tokens, BYTES_PER_VALUE),
    )
    bs1 = {n: flops1[n] / bytes1[n] for n in MATMUL_PARTS}
    slow = {
        n: by[("bf16-cast", n)]["seconds"] / by[("fp16-cast", n)]["seconds"] - 1
        for n in ("ff_in", "ff_out")
    }

    rows = []
    for name in POINTS:
        a, b = by[("fp16-cast", name)], by[("bf16-cast", name)]
        kind = (
            "matmul" if name in MATMUL_PARTS else "all parts" if name == "block" else "element-wise"
        )
        cells = [
            f"<td>{'block 1' if name == 'block' else name}</td>",
            f"<td>{kind}</td>",
            f'<td class="num">{fmt(a["intensity"], 2)}{"*" if a["estimated"] else ""}</td>',
        ]
        for p in (a, b):
            share = p["tflops"] / attainable(roofs, p["mode"], p["intensity"]) * 100
            cells += [
                f'<td class="num">{fmt(p["seconds"] * 1e3, 3)}</td>',
                f'<td class="num">{fmt(p["tflops"], 2)}</td>',
                f'<td class="num">{fmt(p["gbs"])}</td>',
                f'<td class="num">{fmt(share)} %</td>',
            ]
        css = ' class="total"' if name == "block" else ""
        rows.append(f"<tr{css}>{''.join(cells)}</tr>")
    linear_rows = "".join(
        f'<tr><td>{r["dtype"]}</td><td class="num">{r["in"]:,} → {r["out"]:,}</td>'
        f'<td class="num">{fmt(r["seconds"] * 1e3, 3)}</td><td class="num">{fmt(r["tflops"])}</td></tr>'
        for r in line["linear"]
    )
    plain_rows = "".join(
        f'<tr><td>{r["dtype"]}</td><td class="num">{" × ".join(f"{v:,}" for v in r["shape"])}</td>'
        f'<td class="num">{fmt(r["seconds"] * 1e3, 3)}</td><td class="num">{fmt(r["tflops"])}</td></tr>'
        for r in line["plain"]
    )
    memory_rows = "".join(
        f"<tr><td>{r['dtype']}</td><td>{'x * 2 (read and write)' if r['kernel'] == 'scale' else 'mx.sum (read)'}</td>"
        f'<td class="num">{r["values"] // 2**20:,} Mi</td><td class="num">{fmt(r["gbs"])}</td></tr>'
        for r in line["memory"]
    )
    fusion_table = "".join(
        f'<tr><td>{r["mode"]}</td><td class="num">{fmt(r["qkv"] * 1e3, 3)}</td><td class="num">{fmt(r["qkv_fused"] * 1e3, 3)}</td>'
        f'<td class="num">{fmt(r["ff_in"] * 1e3, 3)}</td><td class="num">{fmt(r["ff_fused"] * 1e3, 3)}</td>'
        f'<td class="num">{fmt(r["saved"] * 1e3, 3)}</td><td class="num">{fmt(r["share"] * 100)} %</td></tr>'
        for r in fusion
    )
    plain16 = [r["tflops"] for r in line["plain"] if r["dtype"] == "fp16"]
    plain32 = [r["tflops"] for r in line["plain"] if r["dtype"] == "fp32"]
    f16, fb = fusion[0], fusion[1]

    cells = joined_cells(joined)
    joined_rows = [row for c in JOINED_CELLS for row in cells[c].values()]
    p2 = {
        form: [cells[c][form]["pooled"] for c in JOINED_CELLS[:5]] for form in joined_table.JOINED
    }
    peak_mb = max(row["peak_above_r_bytes"] for row in joined_rows) / 1e6
    all_equal = all(row["bit_equal"] for row in joined_rows)
    cost, p1_batch = flag_cost(joined)
    p2_batch = min(BATCH_SIZE / ips[m] for m in MODES)
    concat16 = concat_seconds(roofs)
    labels = {"P2": f"P2, {MODEL_SIZE}, {SIDE} px", "P1": "P1, ViT-S/16, 224 px"}
    joined_rows_html = "".join(
        f"<tr><td>{labels[p]}</td><td class='num'>{b}</td><td>{m}</td>"
        + "".join(
            f'<td class="num">{cells[(p, b, m)][f]["pooled"]:.3f}</td>' for f in joined_table.JOINED
        )
        + f'<td class="num">{cells[(p, b, m)]["R-J"]["peak_above_r_bytes"] / 1e6:,.0f}</td>'
        + f"<td>{'bit equal' if all(r['bit_equal'] for r in cells[(p, b, m)].values()) else 'differs'}</td>"
        + f"<td>{_d6(cells[(p, b, m)])}</td></tr>"
        for p, b, m in JOINED_CELLS
    )

    control, sweep, best = tile_rows(kernels)
    choice = {target: configs[0] for target, (_, _, configs) in probe_kernels.TARGETS.items()}
    noise = max(abs(1 - r["ratio"]) for r in sweep if r["config"] == choice[r["target"]])
    invalid = [r for r in sweep if not r["valid"]]
    saved_tiles = tile_gain(by, best)
    tile_rows_html = "".join(
        f"<tr><td>{r['target']}</td><td>{r['probe']}</td><td>{choice[r['target']]}</td>"
        f'<td>{r["config"]}</td><td class="num">{fmt(r["tflops"])}</td>'
        f'<td class="num">{r["ratio_to_wheel"]:.3f}</td>'
        f'<td class="num">{sum(s["valid"] for s in sweep if s["target"] == r["target"])} of '
        f"{sum(s['target'] == r['target'] for s in sweep)}</td></tr>"
        for r in best
    )
    best_by = {r["target"]: r for r in best}

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>The Roofline of a ViT-H+/16 Block</title>
<style>{STYLE}</style>
</head>
<body>
<main>
<header>
  <h1>The Roofline of a ViT-H+/16 Block</h1>
  <p>Explainer for roadmap Step 6. Date: 2026-11-10. Sections 7 and 8: Step 6e, 2026-11-14.</p>
  <p class="steps">Step 6 explainers: <a href="06-1-compiled-forward-pass.html">1 Compiled forward pass</a> · <a href="06-2-attention-and-batch-size.html">2 Attention and batch size</a> · <a href="06-3-quantized-weights.html">3 Quantized weights</a> · <a href="06-4-preprocessing.html">4 Preprocessing</a> · 5 Roofline (this page). Step 3: <a href="03-where-the-compute-goes.html">Where the compute goes</a>. Step 6e: <a href="#joined">Section 7</a> and <a href="#tiles">Section 8</a>.</p>
  <p class="steps">Previous: <a href="06-4-preprocessing.html">06-4 Preprocessing</a></p>
</header>

<nav>
  <h2>Contents</h2>
  <ol>
    <li><a href="#why">Why this explainer</a></li>
    <li><a href="#idea">Arithmetic intensity and the roofline</a></li>
    <li><a href="#roofs">The roofs of this Mac</a></li>
    <li><a href="#plot">The roofline of one block</a></li>
    <li><a href="#values">The values of each part</a></li>
    <li><a href="#fusion">What the roofline says about fusion</a></li>
    <li><a href="#joined">The joined projections, measured</a></li>
    <li><a href="#tiles">Other tile sizes in a source build</a></li>
    <li><a href="#count">How the page counts FLOPs and bytes</a></li>
    <li><a href="#measure">How we measured</a></li>
    <li><a href="#surprises">Surprises</a></li>
    <li><a href="#sources">Sources</a></li>
  </ol>
</nav>

<section id="why">
  <h2>1 Why this explainer</h2>
  <p>The speedup table of Step 6 gives the throughput of the compiled MLX port. It does not tell how close each part of a block is to the limits of this Mac. This page puts each part of block 1 of {MODEL_SIZE} at {SIDE} px (T = {num_tokens:,} tokens), batch size {BATCH_SIZE}, on a roofline. The configuration is P2 of <code>specs/06-speed/plan.html</code>, in fp16-cast and bf16-cast. The Mac is an Apple M5 Max with 40 GPU cores and 64 GB.</p>
</section>

<section id="idea">
  <h2>2 Arithmetic intensity and the roofline</h2>
  <p>The arithmetic intensity I of a part is its FLOPs divided by the bytes that it moves between memory and the GPU. Its unit is FLOP/B.</p>
  <pre>attainable rate = min(matmul roof, memory roof × I)</pre>
  <p>A part with a low intensity waits for memory: it is memory-bound, and its limit is the slanted roof. A part with a high intensity waits for the arithmetic units: it is compute-bound, and its limit is the flat roof. The ridge point is the intensity where the two roofs meet: matmul roof / memory roof.</p>
</section>

<section id="roofs">
  <h2>3 The roofs of this Mac</h2>
  <p>The matmul roof is {fmt(roofs["fp16"])} TFLOP/s in fp16 and {fmt(roofs["bf16"])} TFLOP/s in bf16. The memory roof is {bw:.0f} GB/s. So the ridge point is {ridge:.0f} FLOP/B in fp16. These are the highest rates of <code>scripts/probe_roofs.py</code>, not the peaks of a data sheet (Section 10).</p>
  <div class="table-wrap"><table>
    <thead><tr><th>Dtype</th><th class="num">nn.Linear, in → out</th><th class="num">ms</th><th class="num">TFLOP/s</th></tr></thead>
    <tbody>{linear_rows}</tbody>
  </table></div>
  <p class="caption">Table 1. Linear probe: one compiled <code>nn.Linear</code> with bias on an input (8, 1,029, in). 1,280 → 1,280 is the shape of q_proj, k_proj, v_proj and o_proj. 1,280 → 5,120 is gate_proj or up_proj, 5,120 → 1,280 is down_proj (ff_out). 1,280 → 3,840 and 1,280 → 10,240 are the fused forms of D6 b: q, k and v in one matmul, and gate and up in one matmul.</p>
  <div class="table-wrap"><table>
    <thead><tr><th>Dtype</th><th class="num">a @ b, m × k × n</th><th class="num">ms</th><th class="num">TFLOP/s</th></tr></thead>
    <tbody>{plain_rows}</tbody>
  </table></div>
  <p class="caption">Table 2. Plain probe: <code>a @ b</code> with b stored as (k, n). fp32 with TF32 off (ADR 0005).</p>
  <div class="table-wrap"><table>
    <thead><tr><th>Dtype</th><th>Kernel</th><th class="num">Values</th><th class="num">GB/s</th></tr></thead>
    <tbody>{memory_rows}</tbody>
  </table></div>
  <p class="caption">Table 3. Memory probe. GB/s counts the bytes read plus the bytes written.</p>
</section>

<section id="plot">
  <h2>4 The roofline of one block</h2>
  <figure>
    <div class="legend" aria-hidden="true"><span><i class="key s1"></i>fp16-cast</span><span><i class="key s2"></i>bf16-cast</span><span><i class="key roofk"></i>roof (solid fp16, dashed bf16)</span></div>
    <div class="fig"><svg viewBox="0 0 820 500" role="img" aria-label="Roofline of the 11 block parts and block 1">{figure_full(by, roofs)}</svg></div>
    <figcaption>Figure 1. Roofline of block 1 of {MODEL_SIZE} at {SIDE} px, batch size {BATCH_SIZE}. Each point is one part, timed alone and compiled. The dashed frame shows the region of Figure 2. Move the pointer over a point for its values.</figcaption>
  </figure>
  <figure>
    <div class="fig"><svg viewBox="0 0 820 440" role="img" aria-label="The compute-bound region, enlarged">{figure_zoom(by, roofs)}</svg></div>
    <figcaption>Figure 2. The compute-bound region of Figure 1, enlarged, with a linear y axis. The labels are at the fp16-cast points. The bf16-cast point of each part has the same intensity.</figcaption>
  </figure>
  <ul>
    <li>The five matmul parts have an intensity of {min(p["intensity"] for p in matmul):,.0f} to {max(p["intensity"] for p in matmul):,.0f} FLOP/B: {min(p["intensity"] for p in matmul) / ridge:.1f} to {max(p["intensity"] for p in matmul) / ridge:.1f} times the ridge point. They are compute-bound. They reach {min(shares):.0f} to {max(shares):.0f} % of the matmul roof.</li>
    <li>The six element-wise parts have an intensity of {min(p["intensity"] for p in elementwise):.2f} to {max(p["intensity"] for p in elementwise):.0f} FLOP/B. They are memory-bound. Timed alone, they reach {min(bw_shares):.0f} to {max(bw_shares):.0f} % of the memory roof.</li>
    <li>Block 1 as one compiled function reaches {fmt(block16["tflops"])} TFLOP/s in fp16-cast: {block16["tflops"] / roofs["fp16"] * 100:.0f} % of the fp16 matmul roof. The forward pass gives {fmt(ips["fp16-cast"])} img/s × {image_tflop * 1000:,.0f} GFLOP per image (<code>flops.image_flops</code>) = {fmt(ips["fp16-cast"] * image_tflop)} TFLOP/s.</li>
    <li>fp16-cast and bf16-cast are almost equal. bf16-cast needs {slow["ff_in"] * 100:.0f} % more time in ff_in and {slow["ff_out"] * 100:.0f} % more time in ff_out.</li>
  </ul>
</section>

<section id="values">
  <h2>5 The values of each part</h2>
  <div class="table-wrap"><table>
    <thead>
      <tr><th rowspan="2">Part</th><th rowspan="2">Kind</th><th rowspan="2" class="num">FLOP/B</th><th colspan="4" class="grp">fp16-cast</th><th colspan="4" class="grp">bf16-cast</th></tr>
      <tr><th class="num">ms</th><th class="num">TFLOP/s</th><th class="num">GB/s</th><th class="num">of roof</th><th class="num">ms</th><th class="num">TFLOP/s</th><th class="num">GB/s</th><th class="num">of roof</th></tr>
    </thead>
    <tbody>{"".join(rows)}</tbody>
  </table></div>
  <p class="caption">Table 4. Each part of block 1, timed alone (field <code>parts_s</code>), and block 1 as one compiled function (<code>block_s</code>). GB/s = bytes / time. "Of roof" = reached TFLOP/s / attainable rate. * The FLOPs of the element-wise parts are an estimate (Section 9). The 11 part times add up to {fmt(parts16 * 1e3, 2)} ms in fp16-cast; block 1 takes {fmt(block16["seconds"] * 1e3, 2)} ms, because the compiled block fuses the element-wise chains.</p>
</section>

<section id="fusion">
  <h2>6 What the roofline says about fusion</h2>
  <p>Fusion joins kernels, so that intermediate values stay on the GPU and do not go to memory. Thus fusion helps a memory-bound part. It does not raise the rate of a compute-bound part. The roofline gives an upper limit for each kind of fusion.</p>
  <ul>
    <li><b>Element-wise fusion.</b> In fp16-cast the five matmul parts take {fmt(matmul16 * 1e3, 2)} ms. Block 1 takes {fmt(block16["seconds"] * 1e3, 2)} ms. So the layer norms and the fused element-wise kernels take about {fmt(rest16 * 1e3, 2)} ms inside the block: {rest16 / block16["seconds"] * 100:.0f} % of the block time. At the memory roof the element-wise bytes need {fmt(ideal_ew * 1e3, 2)} ms. A fusion of these parts into the matmul kernels (an epilogue) can save at most the {rest16 / block16["seconds"] * 100:.0f} %. <code>mx.compile</code> already fuses the element-wise chains (<a href="06-1-compiled-forward-pass.html">06-1</a>). The layer norms stay their own kernels. An epilogue fusion needs a custom Metal kernel that is as fast as the matmul kernel of MLX.</li>
    <li><b>Projection fusion (D6 b).</b> This fusion does not reduce the bytes. It gives fewer and larger matmuls, and on this Mac a larger matmul reaches a higher rate (Table 1). Table 5 gives the estimate at P2.</li>
    <li><b>The matmul parts.</b> At the fp16 matmul roof the five matmul parts would take {fmt(ideal16 * 1e3, 2)} ms in place of {fmt(matmul16 * 1e3, 2)} ms. This gap is in the matmul and SDPA kernels of MLX, not in the port.</li>
  </ul>
  <div class="table-wrap"><table>
    <thead><tr><th>Dtype mode</th><th class="num">qkv_proj (ms)</th><th class="num">1,280 → 3,840 (ms)</th><th class="num">ff_in (ms)</th><th class="num">1,280 → 10,240 (ms)</th><th class="num">Saved (ms)</th><th class="num">of block 1</th></tr></thead>
    <tbody>{fusion_table}</tbody>
  </table></div>
  <p class="caption">Table 5. Estimate of the projection fusion at P2: the profile time of the part (three or two matmuls) against the probe time of one fused linear layer (Table 1). The two times come from different processes, so the estimate has the noise of two runs. Step 6 measured only 0 to 2 % with compile at 224 px and batch size 1 (D6 of <code>specs/06-speed/explore.html</code>).</p>
  <p class="note-box">The estimate is {fmt(f16["share"] * 100)} % of the block time in fp16-cast and {fmt(fb["share"] * 100)} % in bf16-cast. Only a paired run of both forms at P2 can show the real gain. Section 7 gives this paired run.</p>
</section>

<section id="joined">
  <h2>7 The joined projections, measured</h2>
  <p>Step 6e built the projection fusion of Table 5 as the joined projection and measured it in a paired run (change C7 of <code>specs/06e-kernels/plan.html</code>). Figure 3 shows the separate form R and the joined projection of q, k and v. The joined projection of gate and up has the same structure, with two weights.</p>
  <figure>
    <div class="fig"><svg viewBox="0 0 820 290" role="img" aria-label="The separate and the joined projection of q, k and v">{figure_joined(line)}</svg></div>
    <figcaption>Figure 3. The separate form R (left) and the joined projection of q, k and v (right). Teal: matmul kernels. The weights stay three parameters. The forward pass joins them inside the trace, at each call (D1).</figcaption>
  </figure>
  <figure>
    <pre><code>{html.escape(JOINED_CODE)}</code></pre>
    <figcaption>Listing 1. The joined form in <code>src/dinov3_mlx/mlx_model.py</code>, shortened. The D numbers are the decisions of <code>specs/06e-kernels/plan.html</code>.</figcaption>
  </figure>
  <ul>
    <li><code>set_joined_projections(model, qkv, ff)</code> sets the plain attribute <code>joined</code> of each <code>Attention</code> and <code>SwiGLU</code> module. A plain attribute is not a parameter. So <code>load_weights</code>, <code>set_dtype</code> and <code>nn.quantize</code> do not see it. A trace reads the attribute when it records, so call the function before <code>compile_backbone</code>. The function raises <code>ValueError</code> for a model with quantized weights, because packed weights, scales and offsets need their own join.</li>
    <li><code>_joined(layers, x)</code> joins the weights with <code>mx.concatenate</code> inside the trace, at each call. The weights stay inputs of the trace. So a weight change reaches the output, and no second copy of the weights stays in memory between calls.</li>
    <li><code>_joined_bias(layers)</code> joins the biases. A layer without a bias gets a bias of zeros. {MODEL_SIZE} has a bias on q_proj and v_proj, but not on k_proj.</li>
    <li><code>joined_linear(x, weight, bias)</code> makes the same calls as <code>nn.Linear</code>: <code>mx.addmm</code> with a bias, else a matmul.</li>
    <li><code>Attention.qkv</code> splits the joined output (B, T, 3D) into q, k and v with one reshape and one transpose. <code>SwiGLU.hidden</code> splits its joined output (B, T, 2F) with <code>mx.split</code>.</li>
  </ul>
  <div class="table-wrap"><table>
    <thead><tr><th>Configuration</th><th class="num">Batch size</th><th>Dtype mode</th><th class="num">R-J</th><th class="num">R-Jqkv</th><th class="num">R-Jff</th><th class="num">R-J peak − R (MB)</th><th>First batch</th><th>D6</th></tr></thead>
    <tbody>{joined_rows_html}</tbody>
  </table></div>
  <p class="caption">Table 6. The paired ratio of each joined form to R: the throughput of the form ÷ the throughput of R in the same round. Each value is the pooled median of two runs of 5 rounds (<code>results/joined-table.md</code>). Below 1: the joined form is slower. R-J joins both, R-Jqkv joins only q, k and v, R-Jff joins only gate and up. Peak: the MLX peak of the timed batches of R-J minus that of R. D6: a join is on only if its pooled median at P2 with batch size 8 is at least 1.03 and the median of each run is above 1.</p>
  <ul>
    <li><b>Equal outputs.</b> {"In all rows, the first batch of each joined form is equal to R, bit for bit." if all_equal else "Not all rows are bit equal (Table 6)."}</li>
    <li><b>Slower at P2.</b> At P2 with batch size 8, R-Jqkv gives {min(p2["R-Jqkv"]):.3f} to {max(p2["R-Jqkv"]):.3f}, R-Jff {min(p2["R-Jff"]):.3f} to {max(p2["R-Jff"]):.3f} and R-J {min(p2["R-J"]):.3f} to {max(p2["R-J"]):.3f}. No join reaches the limit 1.03 of D6. So both joins stay off in every dtype mode, and the compiled path of the package does not change.</li>
    <li><b>More memory.</b> The joined forms need up to {peak_mb:,.0f} MB more MLX peak memory than R.</li>
    <li><b>A cost outside the graph at P1.</b> ViT-S/16 has no gate. So at P1 the form R-Jff runs the same graph as R. Its ratio is still below 1, because the harness switches the flags before and after each batch (<code>_JoinedForm</code> in <code>mlx_bench.py</code>): <code>set_joined_projections</code> visits each block and checks the model for quantized weights. This costs {cost * 1e3:.2f} ms per batch: {cost / p1_batch * 100:.0f} % of a P1 batch of {p1_batch * 1e3:.1f} ms. All joined forms pay this cost at P1, so the P1 rows of Table 6 read low by about that share. At P2 a batch takes at least {p2_batch * 1e3:,.0f} ms, so the same cost stays below 0.1 %.</li>
  </ul>
  <p class="note-box">Why is the joined form slower at P2? Table 1 shows that one {shape.dim:,} → {3 * shape.dim:,} matmul is faster than three {shape.dim:,} → {shape.dim:,} matmuls, and Table 5 estimates a saving of {fmt(f16["share"] * 100)} % of the block time in fp16-cast. The concat of the weights of both joins moves {fmt(concat16 * 1e6)} µs of bytes per block at the memory roof: {fmt(concat16 / block16["seconds"] * 100, 2)} % of block 1. These effects predict a gain, not a loss of {(1 - max(p2["R-Jqkv"])) * 100:.0f} to {(1 - min(p2["R-Jqkv"])) * 100:.0f} % for R-Jqkv. The paired run measures only the whole forward pass. It does not show where the joined form loses the time. A profile of the parts of the joined form can show it.</p>
</section>

<section id="tiles">
  <h2>8 Other tile sizes in a source build</h2>
  <p>A matmul kernel computes its output in tiles. One threadgroup computes one tile of bm × bn output values and steps through the sum over K in parts of bk. wm × wn simdgroups share the tile. The attention kernel uses tiles of bq queries and bk keys. MLX selects the tile size from the device and the shape. MLX 0.32.4 does this in <code>steel_matmul_regular_axpby_nax</code> (<code>mlx/backend/metal/matmul.cpp</code>) and in <code>sdpa_full_self_attention_nax</code> (<code>mlx/backend/metal/scaled_dot_product_attention.cpp</code>).</p>
  <p>The probe C1 of Step 6e built MLX 0.32.4 from its source code, outside the repo, in its own venv (ADR 0013). A patch reads a tile size from an environment variable at each dispatch (the tile override) and adds the kernels of the other tile sizes. The package keeps the released MLX.</p>
  <ul>
    <li><b>Control.</b> The unchanged source build has the speed of the wheel: {sum(r["within"] for r in control)} of {len(control)} shapes are within ± 3 % (Table 1 of <code>results/kernels-c1.md</code>). So a ratio in the source build is also a ratio for the wheel.</li>
    <li><b>Noise.</b> Each target ran 12 tile sizes. One of them is the choice of MLX, the same tile size as the control. Its ratio to the control is within ± {noise * 100:.1f} %. This is the noise of one sweep.</li>
    <li><b>Wrong outputs.</b> {len(invalid)} attention tile sizes with bk = 16 run without an error, but give wrong outputs (smallest row cosine {min(r["cosine"] for r in invalid):.2f}). MLX does not ship these tile sizes; the patch added them. Only the cosine check found them.</li>
    <li><b>Effect on the block.</b> T1 and T3 are shapes of the default form. With the best tile sizes, qkv_proj, o_proj and sdpa of block 1 in fp16-cast would save about {fmt(saved_tiles * 1e3, 2)} ms of {fmt(block16["seconds"] * 1e3, 2)} ms: {saved_tiles / block16["seconds"] * 100:.1f} %. This is below the noise of one paired run (V17 of Step 6). T2 is the joined gate and up, which D6 keeps off.</li>
  </ul>
  <div class="table-wrap"><table>
    <thead><tr><th>Target</th><th>Probe</th><th>Choice of MLX</th><th>Best tile size</th><th class="num">TFLOP/s</th><th class="num">Ratio to the wheel</th><th class="num">Valid</th></tr></thead>
    <tbody>{tile_rows_html}</tbody>
  </table></div>
  <p class="caption">Table 7. The best valid tile size of each target of C1, at the shapes of {MODEL_SIZE}, {SIDE} px, batch size {BATCH_SIZE}. Matmul tile sizes: bm, bn, bk, wm, wn. Attention tile sizes: bq, bk, wm, wn. Ratio to the wheel: above 1 is faster. Valid: the smallest row cosine to a NumPy float32 result is at least 0.99999. Source: Table 3 of <code>results/kernels-c1.md</code>.</p>
  <p class="note-box">The tile sizes that MLX selects are close to the best for these shapes. Another tile size gains at most {(max(r["ratio_to_wheel"] for r in best) - 1) * 100:.1f} % on one kernel (T1: {(best_by["T1"]["ratio_to_wheel"] - 1) * 100:.1f} %, near the noise). So only a small part of the gap to the matmul roof (Section 6) is in the choice of the tile size. Verify found a drift in the sweep (V2 of <code>specs/06e-kernels/verify.html</code>): the T2 gain is an artifact, and T3 needs a confirmation run.</p>
</section>

<section id="count">
  <h2>9 How the page counts FLOPs and bytes</h2>
  <ul>
    <li>The FLOPs of the matmul parts come from <code>flops.part_flops</code>. One multiply-add counts as 2 FLOPs. SDPA counts 4 B T² D.</li>
    <li>The bytes come from <code>flops.part_bytes</code> with 2 bytes for each value: each part reads its inputs and its weights once and writes its outputs once. A real kernel can move more bytes, so the true intensity can be lower.</li>
    <li><code>flops.py</code> counts 0 FLOPs for the element-wise parts. A point with 0 FLOPs has no position on a log axis. So <code>roofline_page.elementwise_flops</code> uses an estimate for each value: layer norm 8, RoPE 6 (3 on q and 3 on k), LayerScale with the residual add 2, SwiGLU activation 6. The estimate moves only the six element-wise points. Their GB/s does not change.</li>
    <li>At batch size 1 the linear layers have a lower intensity: qkv_proj {bs1["qkv_proj"]:.0f}, o_proj {bs1["o_proj"]:.0f}, ff_in {bs1["ff_in"]:.0f} and ff_out {bs1["ff_out"]:.0f} FLOP/B. They are still above the ridge point. SDPA stays at {bs1["sdpa"]:.0f} FLOP/B.</li>
  </ul>
</section>

<section id="measure">
  <h2>10 How we measured</h2>
  <ul>
    <li><b>Part times.</b> <code>scripts/bench_speed.py profile --priority P2</code> runs <code>mlx_bench.run_profile_worker</code> in one subprocess for each dtype mode, through the GPU lock, on AC power (<code>results/speed-profile-p2.jsonl</code>). Each part runs alone and compiled: 3 warm-up calls, then the median of 20 calls with one <code>mx.eval</code> after each call.</li>
    <li><b>The fixed cost of each call.</b> At P1 each part takes at least about 0.17 ms, also the smallest part. So the element-wise times of this page include a fixed cost, and their GB/s values are lower limits.</li>
    <li><b>Roofs.</b> <code>scripts/probe_roofs.py</code> with <code>MLX_ENABLE_TF32=0</code> (<code>results/roofs.jsonl</code>). The values of this page come from the first form of the probe: two scripts with the same probes. That run did not record the machine state (field <code>note</code>).</li>
    <li>The plain <code>a @ b</code> reached only {min(plain16):.1f} to {max(plain16):.1f} TFLOP/s in fp16, and {min(plain32):.1f} to {max(plain32):.1f} TFLOP/s in fp32.</li>
    <li><b>Joined projections (Section 7).</b> <code>scripts/bench_speed.py change --change joined</code> runs the four forms R, R-J, R-Jqkv and R-Jff in one worker process for each configuration (<code>mlx_bench.run_change_worker</code>). <code>change_forms</code> gives each joined form its own compiled forward pass, which <code>_JoinedForm</code> wraps: it sets the flags with <code>set_joined_projections</code> before each batch and clears them after it. <code>run_forms</code> runs 5 rounds; the order of the forms rotates by one from round to round, and each form gets its own warm-up and timed batches in each round. The ratio of a form is its throughput ÷ the throughput of R in the same round. <code>first_batch_check</code> compares the first batch of each form with R, bit for bit, and records the largest differences. Two runs at different times give <code>results/speed-joined.jsonl</code>.</li>
    <li><b>Tile sizes (Section 8).</b> <code>scripts/probe_kernels.py</code> starts <code>scripts/probe_kernels_worker.py</code> with <code>python -I</code> in the venv of the wheel or of the source build, one worker at a time, through the GPU lock. Each probe time is the median of 20 calls with one <code>mx.eval</code> after each call. The control alternates the two venvs for 5 rounds; the sweep compares each tile size with the control build (<code>results/kernels-c1.jsonl</code>).</li>
  </ul>
</section>

<section id="surprises">
  <h2>11 Surprises</h2>
  <ul>
    <li>The first matmul roof came from a square 8,192 matmul. It was lower than the rate of the ff_in part. The linear layout, with the weight stored as (out, in), is faster on this Mac than <code>a @ b</code>. So the roof must come from the same layout as the model.</li>
    <li>In bf16 the fused 1,280 → 10,240 layer is not faster than two 1,280 → 5,120 layers (Table 1). In fp16 it is.</li>
    <li>The compiled block is faster than the sum of its parts. A part alone pays a fixed cost and writes its output to memory; inside the block the element-wise chains are fused.</li>
    <li>The joined projection is slower than R at P2, although its one matmul is faster than the separate matmuls alone (Sections 6 and 7).</li>
    <li>At P1 one Python call for each batch costs about {cost / p1_batch * 100:.0f} % of the throughput (Section 7). A harness that switches a form at each call must count this cost.</li>
    <li>{len(invalid)} attention tile sizes give wrong outputs without an error (Section 8).</li>
  </ul>
</section>

<section id="sources">
  <h2>12 Sources</h2>
  <ul>
    <li><code>scripts/roofline_page.py</code> writes this page from <code>results/speed-profile-p2.jsonl</code>, <code>results/roofs.jsonl</code>, <code>results/speed-joined.jsonl</code> and <code>results/kernels-c1.jsonl</code>.</li>
    <li>Step 6e: <code>specs/06e-kernels/plan.html</code>; the tables <code>results/joined-table.md</code> (<code>scripts/joined_table.py</code>) and <code>results/kernels-c1.md</code> (<code>scripts/probe_kernels.py</code>); <code>set_joined_projections</code> in <code>src/dinov3_mlx/mlx_model.py</code>.</li>
    <li><code>scripts/probe_roofs.py</code>: the roof probe. <code>src/dinov3_mlx/flops.py</code>: FLOPs and bytes. <code>src/dinov3_mlx/mlx_bench.py</code>: <code>run_profile_worker</code> and <code>mlx_block_parts</code>.</li>
    <li>S. Williams, A. Waterman, D. Patterson: <a href="https://doi.org/10.1145/1498765.1498785">Roofline: an insightful visual performance model for multicore architectures</a>, Communications of the ACM 52 (4), 2009.</li>
  </ul>
</section>
</main>
<div id="tip" hidden></div>
<script>{SCRIPT}</script>
</body>
</html>
"""


def main() -> None:
    """Write the page from the two result files."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profile", type=Path, default=ROOT / "results/speed-profile-p2.jsonl")
    parser.add_argument("--roofs", type=Path, default=ROOT / "results/roofs.jsonl")
    parser.add_argument("--joined", type=Path, default=joined_table.RESULTS)
    parser.add_argument("--kernels", type=Path, default=probe_kernels.RESULTS)
    parser.add_argument("--out", type=Path, default=ROOT / "docs/learn/06-5-roofline.html")
    args = parser.parse_args()
    profiles = [
        p for p in read_lines(args.profile) if p["dtype_mode"] in MODES and p["status"] == "done"
    ]
    text = page(
        profiles, read_lines(args.roofs)[-1], read_lines(args.joined), read_lines(args.kernels)
    )
    args.out.write_text(text)
    print(args.out)


if __name__ == "__main__":
    main()
