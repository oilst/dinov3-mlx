"""Download the weights of the DINOv3 model sizes into the Hugging Face cache, one at a time."""

import argparse
import fnmatch
import time
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.constants import HF_HUB_CACHE

from dinov3_apple.hub import MODEL_IDS, WEIGHT_PATTERNS, check_free_disk, download

KEEP_FREE_BYTES = 50 * 10**9


def download_size(model_id: str) -> int:
    """Return the size in bytes of the weight files of one model size on the Hub."""
    info = HfApi().model_info(model_id, files_metadata=True)
    return sum(
        sibling.size or 0
        for sibling in info.siblings or []
        if any(fnmatch.fnmatch(sibling.rfilename, pattern) for pattern in WEIGHT_PATTERNS)
    )


def main() -> None:
    """Download each model size that the command line names, or all six."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model_sizes",
        nargs="*",
        metavar="MODEL_SIZE",
        help=f"One or more of {', '.join(MODEL_IDS)}. Default: all six.",
    )
    args = parser.parse_args()
    unknown = sorted(set(args.model_sizes) - MODEL_IDS.keys())
    if unknown:
        parser.error(f"unknown model sizes: {', '.join(unknown)}")

    cache = Path(HF_HUB_CACHE)
    cache.mkdir(parents=True, exist_ok=True)
    for model_size in args.model_sizes or MODEL_IDS:
        model_id = MODEL_IDS[model_size]
        size = download_size(model_id)
        check_free_disk(size, KEEP_FREE_BYTES, cache)
        start = time.perf_counter()
        path = download(model_id)
        print(f"{model_size}: {size / 1e9:.2f} GB in {time.perf_counter() - start:.0f} s, {path}")


if __name__ == "__main__":
    main()
