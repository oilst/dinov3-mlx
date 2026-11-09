"""Build the GitHub Pages site in _site/ (D8 A of specs/07-release/explore.html).

The site holds docs/ and specs/ as they are, plus an index page that links the plan, each
explainer, each spec page and each ADR. The pages stand alone (inline CSS and SVG), so a copy is
enough. No third-party package.

    uv run python scripts/make_site.py [--out _site]
"""

import argparse
import html
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO_URL = "https://github.com/oilst/dinov3-mlx"
STYLE = """
body { background: #fff; color: #000; font-family: -apple-system, "Helvetica Neue", Arial,
  sans-serif; line-height: 1.5; max-width: 860px; margin: 0 auto; padding: 24px 16px; }
h1 { font-size: 1.8em; margin-bottom: 0.2em; }
h2 { font-size: 1.25em; margin-top: 1.8em; border-bottom: 1px solid #ccc; }
a { color: #0645ad; }
ol, ul { padding-left: 1.4em; }
li { margin: 0.2em 0; }
code { font-family: "SF Mono", Menlo, monospace; font-size: 0.92em; }
"""


def page_title(path: Path) -> str:
    """Return the text of the <title> element of one HTML page, or its file name."""
    match = re.search(r"<title>(.*?)</title>", path.read_text(), flags=re.DOTALL)
    return html.unescape(match.group(1).strip()) if match else path.name


def adr_title(path: Path) -> str:
    """Return the first heading of one ADR file, or its file name."""
    for line in path.read_text().splitlines():
        if line.startswith("# "):
            return line.removeprefix("# ").strip()
    return path.name


def link_list(items: list[tuple[str, str]]) -> str:
    """Return an HTML list of (href, text) pairs."""
    rows = [f'  <li><a href="{href}">{html.escape(text)}</a></li>' for href, text in items]
    return "<ul>\n" + "\n".join(rows) + "\n</ul>"


def index_page(root: Path) -> str:
    """Return the index page of the site."""
    explainers = sorted(root.glob("docs/learn/*.html"))
    specs = sorted(p for p in root.glob("specs/*/*.html"))
    adrs = sorted(root.glob("docs/adr/*.md"))
    rel = lambda path: path.relative_to(root).as_posix()
    sections = [
        ("1 The plan", link_list([("docs/plan.html", page_title(root / "docs/plan.html"))])),
        ("2 Explainers", link_list([(rel(p), page_title(p)) for p in explainers])),
        ("3 Specs", link_list([(rel(p), page_title(p)) for p in specs])),
        ("4 Decisions (ADRs)", link_list([(rel(p), adr_title(p)) for p in adrs])),
    ]
    body = "\n".join(f"<h2>{title}</h2>\n{content}" for title, content in sections)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>dinov3-mlx</title>
<style>{STYLE}</style>
</head>
<body>
<h1>dinov3-mlx</h1>
<p>DINOv3 frame embeddings and patch embeddings for Apple silicon with MLX. By Ueli Peter.
The code and the README are in the repository: <a href="{REPO_URL}">{REPO_URL}</a>.</p>
<p>This site holds the plan, the explainers of each DINOv3 part and the spec of each step.</p>
{body}
</body>
</html>
"""


def build(out: Path, root: Path = ROOT) -> None:
    """Write the site into out. An old out folder is replaced.

    Args:
        out: The site folder, for example _site.
        root: The repo folder.
    """
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for folder in ("docs", "specs"):
        shutil.copytree(root / folder, out / folder)
    (out / "index.html").write_text(index_page(root))
    (out / ".nojekyll").write_text("")  # serve the files as they are


def main(argv: list[str] | None = None) -> None:
    """Parse the arguments and build the site."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "_site", help="the site folder")
    args = parser.parse_args(argv)
    build(args.out)
    print(f"site: {args.out}")


if __name__ == "__main__":
    main()
