"""
Fetch TMDB plot overviews for all movies in the dataset.

Saves movie_id,description to data/processed/movie_descriptions.csv.
Resumes automatically from an existing checkpoint.

Run from project root:
    python scripts/fetch_descriptions.py
    python scripts/fetch_descriptions.py --workers 20

Requires: TMDB_READ_ACCESS_TOKEN in .env
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

OUT_PATH  = ROOT / "data" / "processed" / "movie_descriptions.csv"
TMDB_BASE = "https://api.themoviedb.org/3"

TOKEN = os.getenv("TMDB_READ_ACCESS_TOKEN")
if not TOKEN:
    raise RuntimeError("TMDB_READ_ACCESS_TOKEN not set in .env")

HEADERS = {"Authorization": f"Bearer {TOKEN}"}


def clean_title(raw: str) -> str:
    title = re.sub(r"\s*\(\d{4}\)\s*$", "", raw).strip()
    title = re.sub(r"^(.*),\s*(The|A|An|Les|Le|La|L')$", r"\2 \1", title, flags=re.IGNORECASE)
    return title


def fetch_one(movie_id: int, title: str, year: int | None) -> tuple[int, str]:
    params = {"query": title, "include_adult": False}
    if year:
        params["year"] = year
    try:
        r = requests.get(f"{TMDB_BASE}/search/movie", headers=HEADERS,
                         params=params, timeout=10)
        r.raise_for_status()
        results = r.json().get("results", [])
        if not results and year:
            r2 = requests.get(f"{TMDB_BASE}/search/movie", headers=HEADERS,
                              params={"query": title, "include_adult": False}, timeout=10)
            r2.raise_for_status()
            results = r2.json().get("results", [])
        overview = results[0].get("overview", "") if results else ""
        return movie_id, overview or ""
    except Exception:
        return movie_id, ""


def main(workers: int = 10) -> None:
    from src.preprocessing.data_loader import load_movies
    movies = load_movies()

    # Load existing checkpoint
    existing: dict[int, str] = {}
    if OUT_PATH.exists():
        df = pd.read_csv(OUT_PATH)
        existing = {int(r.movie_id): str(r.description) for r in df.itertuples()}
        print(f"Resuming — {len(existing):,} already cached")

    todo = [
        (int(r.movie_id), clean_title(r.title),
         int(r.year) if pd.notna(r.year) else None)
        for r in movies.itertuples()
        if int(r.movie_id) not in existing
    ]

    if not todo:
        print(f"All done — {len(existing):,} descriptions cached.")
        return

    total = len(todo) + len(existing)
    done  = len(existing)
    save_every = max(200, workers * 10)
    results = dict(existing)

    print(f"Fetching {len(todo):,} remaining movies with {workers} workers …")
    start = time.time()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_one, mid, title, year): mid
                   for mid, title, year in todo}

        for i, future in enumerate(as_completed(futures), 1):
            mid, desc = future.result()
            results[mid] = desc
            done += 1

            if i % save_every == 0:
                _save(results)
                elapsed = time.time() - start
                rate    = i / elapsed
                eta     = (len(todo) - i) / max(rate, 0.001)
                print(f"  {done:,}/{total:,} ({done/total*100:.0f}%) — "
                      f"{rate:.0f} req/s — ETA {eta/60:.1f} min")

    _save(results)
    filled = sum(1 for v in results.values() if v)
    print(f"\nDone in {(time.time()-start)/60:.1f} min — "
          f"{filled:,}/{total:,} descriptions ({filled/total*100:.1f}%)")
    print(f"Saved → {OUT_PATH}")


def _save(results: dict[int, str]) -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["movie_id", "description"])
        for mid, desc in sorted(results.items()):
            w.writerow([mid, desc])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=10)
    main(workers=parser.parse_args().workers)
