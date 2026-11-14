"""Tests T10 to T13 of specs/07-release/plan.html: the rename, the core dependencies, the site
and the README."""

import json
import re
import subprocess
import tomllib
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).parent.parent

# The old package name, the old distribution name and so the old cache folder. Without the first
# letter, so the bracket form of HEAVY_JOB_PATTERN ("[d]inov3_...") also counts.
OLD_NAMES = ("inov3" + "_apple", "inov3" + "-apple")
# The files that describe the rename keep the old name.
RENAME_RECORDS = ("specs/07-release/", "docs/adr/0018-")


def tracked_files() -> list[Path]:
    """The files that git tracks, without the files of the rename records."""
    names = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.splitlines()
    return [ROOT / name for name in names if not name.startswith(RENAME_RECORDS)]


def test_no_old_package_name() -> None:
    """T10: no tracked file names the old package, the old distribution or the old cache path."""
    hits = []
    for path in tracked_files():
        if not path.is_file():
            continue
        text = path.read_bytes().decode("utf-8", errors="ignore")
        hits += [f"{path.relative_to(ROOT)}: {name}" for name in OLD_NAMES if name in text]
    assert hits == []


def test_no_home_folder_path() -> None:
    """AC15 guard: no tracked file holds a home folder path of macOS. The rewrite check of
    scripts/rewrite_history.py refuses the whole history with one."""
    prefix = "/" + "Users/"
    hits = [
        str(path.relative_to(ROOT))
        for path in tracked_files()
        if path.is_file() and prefix in path.read_bytes().decode("utf-8", errors="ignore")
    ]
    assert hits == []


# The packages of the API path (D3 B of specs/07-release/explore.html).
CORE_PACKAGES = {"mlx", "numpy", "safetensors", "huggingface-hub", "pillow"}
REFERENCE_PACKAGES = {"torch", "transformers"}


def package_name(requirement: str) -> str:
    """The package name of a requirement string, for example "mlx" of "mlx>=0.32.4,<0.33"."""
    return re.split(r"[<>=!~\[; ]", requirement, maxsplit=1)[0].lower()


def test_core_dependencies() -> None:
    """T11: five core packages with ranges; torch and transformers in the extra and the dev group
    only; the lock file matches pyproject.toml."""
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    core = project["project"]["dependencies"]
    assert {package_name(r) for r in core} == CORE_PACKAGES
    assert all(">=" in r and "<" in r for r in core), core
    extra = project["project"]["optional-dependencies"]["reference"]
    assert {package_name(r) for r in extra} == REFERENCE_PACKAGES
    dev = {package_name(r) for r in project["dependency-groups"]["dev"] if isinstance(r, str)}
    assert REFERENCE_PACKAGES <= dev
    result = subprocess.run(
        ["uv", "lock", "--check"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


class Links(HTMLParser):
    """The href values and the ids of one HTML page."""

    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []
        self.ids: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if values.get("id"):
            self.ids.add(values["id"])
        if tag == "a" and values.get("name"):
            self.ids.add(values["name"])
        if tag == "a" and values.get("href"):
            self.hrefs.append(values["href"])


def page_links(path: Path) -> Links:
    """Parse one HTML page."""
    parser = Links()
    parser.feed(path.read_text())
    return parser


def broken_links(site: Path) -> list[str]:
    """Return each relative link of the HTML pages in site whose file or anchor is missing."""
    pages = {path: page_links(path) for path in site.rglob("*.html")}
    broken = []
    for path, links in pages.items():
        for href in links.hrefs:
            parts = urlsplit(href)
            if parts.scheme or parts.netloc or href.startswith("mailto:"):
                continue
            target = (path.parent / unquote(parts.path)).resolve() if parts.path else path
            if target.is_dir():
                target = target / "index.html"
            if not target.is_file() or not target.is_relative_to(site.resolve()):
                broken.append(f"{path.relative_to(site)}: {href}")
            elif parts.fragment and target.suffix == ".html":
                ids = pages[target].ids if target in pages else page_links(target).ids
                if unquote(parts.fragment) not in ids:
                    broken.append(f"{path.relative_to(site)}: {href}")
    return broken


def test_site_links_resolve(tmp_path: Path) -> None:
    """T12: make_site.py builds the site; its index links each explainer, each spec page and the
    plan; each relative link and anchor resolves."""
    import make_site

    site = tmp_path / "_site"
    make_site.build(site)
    index = page_links(site / "index.html")
    pages = [p.relative_to(ROOT).as_posix() for p in ROOT.glob("docs/learn/*.html")]
    pages += [p.relative_to(ROOT).as_posix() for p in ROOT.glob("specs/*/*.html")]
    pages.append("docs/plan.html")
    assert sorted(set(pages) - set(index.hrefs)) == []
    assert broken_links(site) == []


# The sections of Section 3.4 of the plan, as level-2 headings of the README.
README_SECTIONS = [
    "Quick start",
    "How it compares",
    "How far from the optimum",
    "What this repo holds",
    "Documentation",
    "Limits",
    "Tests",
    "License",
]


def readme() -> str:
    return (ROOT / "README.md").read_text()


def test_readme_sections() -> None:
    """T13: the README has its sections in order (the short form of 2026-11-14, after Section 3.4)
    and names V2, V4 and V17."""
    headings = re.findall(r"^## (.+)$", readme(), flags=re.MULTILINE)
    assert headings == README_SECTIONS
    for limit in ("V2", "V4", "V17"):
        assert re.search(rf"\b{limit}\b", readme()), limit


def test_readme_links_resolve() -> None:
    """T13: each relative link of the README resolves to a tracked file."""
    tracked = {path.relative_to(ROOT).as_posix() for path in tracked_files()}
    broken = []
    for target in re.findall(r"\]\(([^)\s]+)\)", readme()):
        parts = urlsplit(target)
        if parts.scheme or not parts.path:
            continue
        if unquote(parts.path).rstrip("/") not in tracked and not (ROOT / parts.path).is_dir():
            broken.append(target)
    assert broken == []


def test_readme_tables_name_their_source() -> None:
    """T13: each table of the README is followed by a line "Source: ..." with a results file."""
    lines = readme().splitlines()
    ends = [
        i
        for i, line in enumerate(lines)
        if line.startswith("|") and (i + 1 == len(lines) or not lines[i + 1].startswith("|"))
    ]
    assert ends, "the README has no table"
    for end in ends:
        after = "\n".join(lines[end + 1 : end + 3])
        sources = re.findall(r"Source: .*?\(?(results/[\w./-]+)", after)
        assert sources, f"the table that ends on line {end + 1} names no source"
        assert all((ROOT / source).is_file() for source in sources), sources


def test_bench_api_table(tmp_path: Path) -> None:
    """B1: the table gives the lower median of each configuration against the target 0.97."""
    import bench_api

    results = tmp_path / "api-overhead.jsonl"
    lines = []
    for name, ratios in (
        ("P1", [0.99, 0.98, 1.0, 0.97, 0.99]),
        ("P2", [0.95, 0.96, 0.9, 1.0, 0.9]),
    ):
        rounds = [{"api": 100 * r, "direct": 100.0, "ratio": r} for r in ratios]
        lines.append({"name": name, "status": "done", "rounds": rounds, "ratio": sorted(ratios)[2]})
    lines.append({"name": "P1-pad", "status": "skipped", "reason": "needs 99 GB"})
    results.write_text("".join(json.dumps(line) + "\n" for line in lines))
    table = tmp_path / "api-table.md"
    bench_api.write_table(results, table)
    rows = {row.split(" | ")[0].strip("| "): row for row in table.read_text().splitlines()}
    assert rows["P1"].endswith("| 0.990 | ≥ 0.97 | pass |")
    assert rows["P2"].endswith("| 0.950 | ≥ 0.97 | fail |")
    assert "skipped: needs 99 GB" in rows["P1-pad"]
