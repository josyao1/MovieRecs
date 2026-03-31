"""
Download MovieLens dataset.

Supports both 1M (legacy) and 25M (default).

Usage:
    python scripts/download_data.py          # downloads 25M
    python scripts/download_data.py --ml1m   # downloads 1M (legacy)

Downloads to data/raw/ and extracts files.
"""
import argparse
import urllib.request
import zipfile
from pathlib import Path

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"

DATASETS = {
    "25m": {
        "url":    "https://files.grouplens.org/datasets/movielens/ml-25m.zip",
        "folder": "ml-25m",
        "exts":   {".csv"},
    },
    "1m": {
        "url":    "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
        "folder": "ml-1m",
        "exts":   {".dat", ".txt"},
    },
}


def download(version: str = "25m") -> None:
    cfg = DATASETS[version]
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = RAW_DIR / f"ml-{version}.zip"

    print(f"Downloading MovieLens {version.upper()} ({cfg['url']}) ...")
    urllib.request.urlretrieve(cfg["url"], zip_path)
    print(f"Downloaded ({zip_path.stat().st_size / 1e6:.0f} MB)")

    print("Extracting ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.namelist():
            p = Path(member)
            if p.suffix in cfg["exts"] and p.parent.name == cfg["folder"]:
                dest = RAW_DIR / p.name
                dest.write_bytes(zf.read(member))
                print(f"  → {dest}")

    zip_path.unlink()
    print(f"Done. Raw data in {RAW_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ml1m", action="store_true", help="Download 1M instead of 25M")
    args = parser.parse_args()
    download("1m" if args.ml1m else "25m")
