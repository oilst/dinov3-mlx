import hashlib
import re
from pathlib import Path

import pytest
from PIL import Image

IMAGES_DIR = Path(__file__).parent / "fixtures" / "images"

# Table 1 of specs/02-reference/plan.html: file name and stored size (width, height).
TEST_IMAGES: dict[str, tuple[int, int]] = {
    "iss-kibo-lab.jpg": (1024, 576),
    "forest.jpg": (1024, 768),
    "iss-conference.jpg": (768, 1024),
}


def _documented_sha256() -> dict[str, str]:
    text = (IMAGES_DIR / "SOURCES.md").read_text()
    return dict(
        re.findall(r"^\| `([\w-]+\.jpg)` \|.*?\| `([0-9a-f]{64})` \|", text, flags=re.MULTILINE)
    )


@pytest.mark.parametrize("name", TEST_IMAGES)
def test_test_images_match_sources(name: str) -> None:
    path = IMAGES_DIR / name
    with Image.open(path) as image:
        assert image.format == "JPEG"
        assert image.size == TEST_IMAGES[name]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == _documented_sha256()[name]


def test_test_images_total_size_is_at_most_1_mb() -> None:
    assert sum((IMAGES_DIR / name).stat().st_size for name in TEST_IMAGES) <= 1_000_000
