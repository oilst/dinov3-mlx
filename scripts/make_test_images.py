"""Download the three public-domain test images and store them in tests/fixtures/images/.

Each source photo comes from Wikimedia Commons. The script checks the SHA-1 of each download
against the value of the Commons API and against the value that we checked on 2026-11-06. Then it
crops the center to the aspect ratio of the test image, resizes it with Pillow LANCZOS to 1024 px
on the long side and saves a JPEG with quality 95. Last, it writes SOURCES.md.
"""

import hashlib
import json
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps

IMAGES_DIR = Path(__file__).parents[1] / "tests" / "fixtures" / "images"
COMMONS_API = "https://commons.wikimedia.org/w/api.php"
COMMONS_PAGE = "https://commons.wikimedia.org/wiki/File:"
# A generic User-Agent. It names the tool and no person.
USER_AGENT = "dinov3-apple-test-images/0.1 (scripts/make_test_images.py)"
LONG_SIDE = 1024
JPEG_QUALITY = 92
MAX_TOTAL_BYTES = 1_000_000


@dataclass(frozen=True)
class TestImage:
    """One test image of Table 1 of specs/02-reference/plan.html."""

    name: str
    source: str
    aspect: tuple[int, int]
    sha1: str
    author: str
    license: str


TEST_IMAGES: list[TestImage] = [
    TestImage(
        name="iss-kibo-lab.jpg",
        source="ISS-22_crew_members_in_the_Kibo_lab.jpg",
        aspect=(16, 9),
        sha1="564aa44ebf0db3683e2b5b86d48622475336f496",
        author="NASA",
        license="Public domain (PD NASA)",
    ),
    TestImage(
        name="forest.jpg",
        source="South_Fork_Amercian_River_Cohesive_Strategy_Field_Trip_(36013609855).jpg",
        aspect=(4, 3),
        sha1="3116c98b7974dc25134658e1140394410962efcb",
        author="USDA Forest Service, Pacific Southwest Region 5",
        license="Public domain (PD USDA)",
    ),
    TestImage(
        name="iss-conference.jpg",
        source=(
            "Expedition_75_crew_members_gather_for_a_daily_planning_conference_(iss075e0162800).jpg"
        ),
        aspect=(3, 4),
        sha1="9396757b90e0e888ec02bdcd5614ac6c69496e7f",
        author="NASA Johnson Space Center",
        license="Public domain (PD NASA)",
    ),
]


class SourceChangedError(ValueError):
    """A source file on Wikimedia Commons is not the file that we checked."""


def _get(url: str, attempts: int = 5) -> bytes:
    """Return the body of a GET request. Wait and try again after HTTP 429 (too many requests)."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            if error.code != 429 or attempt == attempts - 1:
                raise
            time.sleep(int(error.headers.get("Retry-After", 0)) or 10 * (attempt + 1))
    raise AssertionError("unreachable")


def commons_image_info(source: str) -> dict:
    """Return the Commons API image info (url, sha1, width, height, license) of one file."""
    query = urllib.parse.urlencode(
        {
            "action": "query",
            "format": "json",
            "prop": "imageinfo",
            "iiprop": "url|sha1|size|extmetadata",
            "iiextmetadatafilter": "LicenseShortName",
            "titles": f"File:{source}",
        }
    )
    pages = json.loads(_get(f"{COMMONS_API}?{query}"))["query"]["pages"]
    return next(iter(pages.values()))["imageinfo"][0]


def center_crop(image: Image.Image, aspect: tuple[int, int]) -> Image.Image:
    """Crop the center of the image to the aspect ratio width : height."""
    width, height = image.size
    aspect_w, aspect_h = aspect
    if width * aspect_h > height * aspect_w:
        new_width, new_height = round(height * aspect_w / aspect_h), height
    else:
        new_width, new_height = width, round(width * aspect_h / aspect_w)
    left, top = (width - new_width) // 2, (height - new_height) // 2
    return image.crop((left, top, left + new_width, top + new_height))


def stored_size(aspect: tuple[int, int]) -> tuple[int, int]:
    """Return (width, height) with LONG_SIDE on the long side."""
    aspect_w, aspect_h = aspect
    if aspect_w >= aspect_h:
        return LONG_SIDE, round(LONG_SIDE * aspect_h / aspect_w)
    return round(LONG_SIDE * aspect_w / aspect_h), LONG_SIDE


def make_test_image(test_image: TestImage, download_dir: Path) -> dict[str, str]:
    """Download, check, crop, resize and save one test image.

    Returns:
        The values for its row in SOURCES.md.

    Raises:
        SourceChangedError: If the SHA-1 of the download differs from the API or the checked value.
    """
    info = commons_image_info(test_image.source)
    if info["sha1"] != test_image.sha1:
        raise SourceChangedError(f"{test_image.source}: API SHA-1 {info['sha1']} changed")
    if info["extmetadata"]["LicenseShortName"]["value"] != "Public domain":
        raise SourceChangedError(f"{test_image.source}: license is no longer public domain")
    download = download_dir / test_image.name
    download.write_bytes(_get(info["url"].split("?")[0]))
    sha1 = hashlib.sha1(download.read_bytes()).hexdigest()
    if sha1 != info["sha1"]:
        raise SourceChangedError(f"{test_image.source}: download SHA-1 {sha1} != {info['sha1']}")

    with Image.open(download) as source:
        upright = ImageOps.exif_transpose(source).convert("RGB")
    cropped = center_crop(upright, test_image.aspect)
    resized = cropped.resize(stored_size(test_image.aspect), Image.Resampling.LANCZOS)
    out = IMAGES_DIR / test_image.name
    resized.save(out, format="JPEG", quality=JPEG_QUALITY)

    crop = "none" if cropped.size == upright.size else f"center {cropped.width} × {cropped.height}"
    return {
        "name": test_image.name,
        "size": f"{resized.width} × {resized.height}",
        "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "page": f"{COMMONS_PAGE}{test_image.source}",
        "source_size": f"{upright.width} × {upright.height}",
        "sha1": sha1,
        "crop": crop,
        "author": test_image.author,
        "license": test_image.license,
    }


def write_sources(rows: list[dict[str, str]]) -> None:
    """Write SOURCES.md with one table row for each test image."""
    lines = [
        "# Test images",
        "",
        (
            "Each test image is a work of a US federal agency, so it is in the public domain in "
            "the US. We checked each license page on 2026-11-06. `scripts/make_test_images.py` "
            "makes the files: it downloads the source file, checks its SHA-1 against the "
            "Wikimedia Commons API, crops the center, resizes with Pillow LANCZOS to "
            f"{LONG_SIDE} px on the long side and saves a JPEG with quality {JPEG_QUALITY}. The "
            "script removes all metadata."
        ),
        "",
        (
            "The NASA media guidelines ask users not to suggest an endorsement by NASA. This "
            "repo uses the photos only as test inputs. NASA and the USDA Forest Service do not "
            "endorse it."
        ),
        "",
        (
            "| File | Stored size | SHA-256 of the stored file | Source page | Author | License "
            "| Source size | SHA-1 of the source | Crop |"
        ),
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['name']}` | {row['size']} | `{row['sha256']}` | {row['page']} "
            f"| {row['author']} | {row['license']} | {row['source_size']} | `{row['sha1']}` "
            f"| {row['crop']} |"
        )
    lines += ["", "Sizes are width × height in pixels.", ""]
    (IMAGES_DIR / "SOURCES.md").write_text("\n".join(lines))


def main() -> None:
    """Make the three test images and SOURCES.md."""
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as download_dir:
        rows = [make_test_image(image, Path(download_dir)) for image in TEST_IMAGES]
    write_sources(rows)
    total = sum((IMAGES_DIR / row["name"]).stat().st_size for row in rows)
    print(f"{len(rows)} test images, {total / 1e6:.2f} MB in {IMAGES_DIR}")
    if total > MAX_TOTAL_BYTES:
        raise SystemExit(f"The test images need {total} bytes, more than {MAX_TOTAL_BYTES}.")


if __name__ == "__main__":
    main()
