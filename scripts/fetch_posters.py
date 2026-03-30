"""
Fetch TMDB poster URLs for all MovieLens 1M movies.

Searches TMDB by cleaned title + year. Saves a mapping of
{movie_id: poster_url} to artifacts/posters.json.

Run from project root:
    python scripts/fetch_posters.py

Requires: TMDB_READ_ACCESS_TOKEN in .env
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import requests
from dotenv import load_dotenv
import os

load_dotenv()

ROOT = Path(__file__).parents[1]
POSTERS_PATH = ROOT / "artifacts" / "posters.json"
TMDB_BASE = "https://api.themoviedb.org/3"
POSTER_BASE = "https://image.tmdb.org/t/p/w500"

TOKEN = os.getenv("TMDB_READ_ACCESS_TOKEN")
if not TOKEN:
    raise RuntimeError("TMDB_READ_ACCESS_TOKEN not set in .env")

HEADERS = {"Authorization": f"Bearer {TOKEN}"}


def clean_title(raw: str) -> str:
    """Strip year and trailing articles moved to end, e.g. 'Matrix, The (1999)' -> 'The Matrix'."""
    title = re.sub(r"\s*\(\d{4}\)\s*$", "", raw).strip()
    # Handle "Title, The" → "The Title"
    title = re.sub(r"^(.*),\s*(The|A|An|Les|Le|La|L')$", r"\2 \1", title, flags=re.IGNORECASE)
    return title


def search_tmdb(title: str, year: int | None) -> str | None:
    params = {"query": title, "include_adult": False}
    if year:
        params["year"] = year
    r = requests.get(f"{TMDB_BASE}/search/movie", headers=HEADERS, params=params, timeout=10)
    if r.status_code != 200:
        return None
    results = r.json().get("results", [])
    if not results:
        # Retry without year constraint
        if year:
            return search_tmdb(title, None)
        return None
    poster = results[0].get("poster_path")
    return f"{POSTER_BASE}{poster}" if poster else None


def main():
    import sys
    sys.path.insert(0, str(ROOT))
    from src.preprocessing.data_loader import load_movies
    movies = load_movies()

    # Load existing cache if any
    if POSTERS_PATH.exists():
        with open(POSTERS_PATH) as f:
            posters: dict[str, str | None] = json.load(f)
        print(f"Resuming — {len(posters)} already cached")
    else:
        posters = {}

    total = len(movies)
    found = 0
    skipped = 0

    for i, row in enumerate(movies.itertuples(), 1):
        mid = str(row.movie_id)
        if mid in posters:
            skipped += 1
            continue

        title = clean_title(row.title)
        year = int(row.year) if row.year and not str(row.year) == "<NA>" else None

        url = search_tmdb(title, year)
        posters[mid] = url
        if url:
            found += 1

        if i % 100 == 0:
            # Save checkpoint
            with open(POSTERS_PATH, "w") as f:
                json.dump(posters, f)
            pct = (i + skipped) / total * 100
            print(f"  {i + skipped}/{total} ({pct:.0f}%) — found {found + sum(1 for v in posters.values() if v)} posters")

        # Respect TMDB rate limit (~40 req/s; stay conservative)
        time.sleep(0.05)

    # Final save
    with open(POSTERS_PATH, "w") as f:
        json.dump(posters, f)

    total_found = sum(1 for v in posters.values() if v)
    print(f"\nDone: {total_found}/{total} posters found ({total_found/total*100:.1f}%)")
    print(f"Saved to {POSTERS_PATH}")


if __name__ == "__main__":
    main()
