"""Tests T10 to T13 of specs/07-release/plan.html: the rename, the core dependencies, the site
and the README."""

import re
import subprocess
import tomllib
from pathlib import Path

ROOT = Path(__file__).parent.parent

# The old package name, the old distribution name and so the old cache folder. Without the first
# letter, so the bracket form of HEAVY_JOB_PATTERN ("[d]inov3_...") also counts.
OLD_NAMES = ("inov3" + "_apple", "inov3" + "-apple")
# The files that describe the rename keep the old name.
RENAME_RECORDS = ("specs/07-release/", "docs/adr/0017-")


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
