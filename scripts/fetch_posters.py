"""
Fetch TMDB poster URLs for all movies in the dataset.

Saves {movie_id: poster_url} to artifacts/posters.json.
Resumes automatically from an existing checkpoint.

Run from project root:
    python scripts/fetch_posters.py
    python scripts/fetch_posters.py --workers 20   # more parallelism

Requires: TMDB_READ_ACCESS_TOKEN in .env
"""
from __future__ import annotations

import argparse
import json
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

POSTERS_PATH = ROOT / "artifacts" / "posters.json"
TMDB_BASE    = "https://api.themoviedb.org/3"
POSTER_BASE  = "https://image.tmdb.org/t/p/w500"

TOKEN = os.getenv("TMDB_READ_ACCESS_TOKEN")
if not TOKEN:
    raise RuntimeError("TMDB_READ_ACCESS_TOKEN not set in .env")

HEADERS = {"Authorization": f"Bearer {TOKEN}"}


def clean_title(raw: str) -> str:
    """'Matrix, The (1999)' → 'The Matrix'"""
    title = re.sub(r"\s*\(\d{4}\)\s*$", "", raw).strip()
    title = re.sub(r"^(.*),\s*(The|A|An|Les|Le|La|L')$", r"\2 \1", title, flags=re.IGNORECASE)
    return title


def fetch_one(movie_id: int, title: str, year: int | None) -> tuple[int, str | None]:
    """Fetch poster URL for one movie. Returns (movie_id, url_or_None)."""
    params = {"query": title, "include_adult": False}
    if year:
        params["year"] = year

    try:
        r = requests.get(f"{TMDB_BASE}/search/movie", headers=HEADERS,
                         params=params, timeout=10)
        r.raise_for_status()
        results = r.json().get("results", [])
        if not results and year:
            # Retry without year constraint
            r2 = requests.get(f"{TMDB_BASE}/search/movie", headers=HEADERS,
                              params={"query": title, "include_adult": False}, timeout=10)
            r2.raise_for_status()
            results = r2.json().get("results", [])
        poster = results[0].get("poster_path") if results else None
        return movie_id, (f"{POSTER_BASE}{poster}" if poster else None)
    except Exception:
        return movie_id, None


def main(workers: int = 10) -> None:
    from src.preprocessing.data_loader import load_movies
    movies = load_movies()

    # Load existing checkpoint
    if POSTERS_PATH.exists():
        with open(POSTERS_PATH) as f:
            posters: dict[str, str | None] = json.load(f)
        print(f"Resuming — {len(posters):,} already cached")
    else:
        posters = {}

    # Build work queue — movies not yet fetched
    todo = [
        (int(r.movie_id), clean_title(r.title),
         int(r.year) if pd.notna(r.year) else None)
        for r in movies.itertuples()
        if str(r.movie_id) not in posters
    ]

    if not todo:
        total_found = sum(1 for v in posters.values() if v)
        print(f"All done — {total_found:,}/{len(posters):,} posters in cache.")
        return

    total   = len(todo) + len(posters)
    found   = sum(1 for v in posters.values() if v)
    done    = len(posters)
    save_every = max(100, workers * 5)

    print(f"Fetching {len(todo):,} remaining movies with {workers} workers …")
    start = time.time()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_one, mid, title, year): mid
                   for mid, title, year in todo}

        for i, future in enumerate(as_completed(futures), 1):
            mid, url = future.result()
            posters[str(mid)] = url
            if url:
                found += 1
            done += 1

            if i % save_every == 0:
                with open(POSTERS_PATH, "w") as f:
                    json.dump(posters, f)
                elapsed = time.time() - start
                rate    = i / elapsed
                eta     = (len(todo) - i) / rate
                pct     = done / total * 100
                print(f"  {done:,}/{total:,} ({pct:.0f}%) — "
                      f"{found:,} posters — {rate:.0f} req/s — ETA {eta/60:.1f} min")

    # Final save
    with open(POSTERS_PATH, "w") as f:
        json.dump(posters, f)

    total_found = sum(1 for v in posters.values() if v)
    elapsed = time.time() - start
    print(f"\nDone in {elapsed/60:.1f} min — "
          f"{total_found:,}/{total:,} posters ({total_found/total*100:.1f}%)")
    print(f"Saved → {POSTERS_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=10,
                        help="Parallel TMDB requests (default 10, max ~40)")
    main(workers=parser.parse_args().workers)
