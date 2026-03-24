"""
Download MovieLens 1M dataset.

Usage:
    python scripts/download_data.py

Downloads to data/raw/ and extracts .dat files.
"""
import urllib.request
import zipfile
from pathlib import Path

URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
RAW_DIR = Path(__file__).parent.parent / "data" / "raw"


def download_movielens_1m() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = RAW_DIR / "ml-1m.zip"

    print(f"Downloading MovieLens 1M from {URL} ...")
    urllib.request.urlretrieve(URL, zip_path)
    print(f"Downloaded to {zip_path}")

    print("Extracting ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.namelist():
            filename = Path(member).name
            if filename.endswith(".dat") or filename.endswith(".txt"):
                source = zf.open(member)
                dest = RAW_DIR / filename
                with open(dest, "wb") as f:
                    f.write(source.read())
                print(f"  Extracted: {dest}")

    zip_path.unlink()
    print("Done. Raw data in data/raw/")


if __name__ == "__main__":
    download_movielens_1m()
