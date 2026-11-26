"""
scripts/download_model.py
Download the trained weights from the GitHub Release into saved_models/.

Usage (from the project root):
    python scripts/download_model.py
    python scripts/download_model.py --url <custom-url>
"""
import argparse
import os
import sys
import urllib.request

DEFAULT_URL = (
    "https://github.com/shahipervez/Image-Caption-Generator/"
    "releases/download/v1.0.0/caption_model.weights.h5"
)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET = os.path.join(PROJECT_ROOT, "saved_models", "caption_model.weights.h5")


def progress(block_num: int, block_size: int, total_size: int) -> None:
    if total_size <= 0:
        return
    done = min(block_num * block_size, total_size)
    pct = done * 100 / total_size
    bar = "#" * int(pct // 4)
    sys.stdout.write(f"\r  [{bar:<25}] {pct:5.1f}%  {done / 1e6:6.1f}/{total_size / 1e6:.1f} MB")
    sys.stdout.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description="Download caption model weights.")
    parser.add_argument("--url", default=os.getenv("MODEL_WEIGHTS_URL", DEFAULT_URL))
    parser.add_argument("--force", action="store_true", help="Re-download even if the file exists")
    args = parser.parse_args()

    if os.path.exists(TARGET) and not args.force:
        print(f"Weights already present: {TARGET} ({os.path.getsize(TARGET) / 1e6:.1f} MB)")
        print("Use --force to download again.")
        return

    os.makedirs(os.path.dirname(TARGET), exist_ok=True)
    tmp_path = TARGET + ".part"
    print(f"Downloading weights from:\n  {args.url}")
    try:
        urllib.request.urlretrieve(args.url, tmp_path, reporthook=progress)
    except Exception as exc:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        sys.exit(f"\nDownload failed: {exc}")
    os.replace(tmp_path, TARGET)
    print(f"\nSaved to {TARGET} ({os.path.getsize(TARGET) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
