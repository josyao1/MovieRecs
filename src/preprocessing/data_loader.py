"""
Load and clean MovieLens data into normalized DataFrames.

Auto-detects dataset version by which files are present in data/raw/:
  ML-1M  (legacy):  ratings.dat, movies.dat          — "::" separator, latin-1
  ML-25M (default): ratings.csv, movies.csv, tags.csv — CSV with header row

Normalized output columns are identical regardless of source version:
  ratings → user_id, movie_id, rating, timestamp
  movies  → movie_id, title, genres (list), year
  tags    → user_id, movie_id, tag  (25M only, empty DataFrame for 1M)
"""
from __future__ import annotations

import pandas as pd
from pathlib import Path

RAW_DIR       = Path(__file__).parents[2] / "data" / "raw"
PROCESSED_DIR = Path(__file__).parents[2] / "data" / "processed"


def _detect_version(raw_dir: Path) -> str:
    if (raw_dir / "ratings.csv").exists():
        return "25m"
    if (raw_dir / "ratings.dat").exists():
        return "1m"
    raise FileNotFoundError(
        f"No MovieLens files found in {raw_dir}. "
        "Run: python scripts/download_data.py"
    )


def load_ratings(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Return ratings with columns: user_id, movie_id, rating, timestamp."""
    version = _detect_version(raw_dir)

    if version == "25m":
        df = pd.read_csv(raw_dir / "ratings.csv")
        df = df.rename(columns={"userId": "user_id", "movieId": "movie_id"})
    else:
        df = pd.read_csv(
            raw_dir / "ratings.dat",
            sep="::", engine="python",
            names=["user_id", "movie_id", "rating", "timestamp"],
        )

    return df[["user_id", "movie_id", "rating", "timestamp"]].astype({
        "user_id": int, "movie_id": int, "rating": float, "timestamp": int,
    })


def load_movies(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Return movies with columns: movie_id, title, genres (list), year."""
    version = _detect_version(raw_dir)

    if version == "25m":
        df = pd.read_csv(raw_dir / "movies.csv")
        df = df.rename(columns={"movieId": "movie_id"})
    else:
        df = pd.read_csv(
            raw_dir / "movies.dat",
            sep="::", engine="python",
            names=["movie_id", "title", "genres"],
            encoding="latin-1",
        )

    # Parse genres — handle "(no genres listed)" from ML-25M
    df["genres"] = df["genres"].apply(
        lambda g: [x for x in str(g).split("|") if x and x != "(no genres listed)"]
    )
    df["year"] = df["title"].str.extract(r"\((\d{4})\)").astype("Int64")
    df["movie_id"] = df["movie_id"].astype(int)

    print(f"Loaded {len(df):,} movies (ML-{version.upper()}) · "
          f"years {df['year'].min()}–{df['year'].max()}")
    return df[["movie_id", "title", "genres", "year"]]


def load_tags(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """
    Return user-generated tags (ML-25M only).
    Useful as richer content features than genres alone.
    Returns empty DataFrame for ML-1M.
    """
    version = _detect_version(raw_dir)
    if version != "25m" or not (raw_dir / "tags.csv").exists():
        return pd.DataFrame(columns=["user_id", "movie_id", "tag"])

    df = pd.read_csv(raw_dir / "tags.csv", usecols=["userId", "movieId", "tag"])
    df = df.rename(columns={"userId": "user_id", "movieId": "movie_id"})
    df["tag"] = df["tag"].str.lower().str.strip()
    df = df.dropna(subset=["tag"])
    print(f"Loaded {len(df):,} tags")
    return df[["user_id", "movie_id", "tag"]]


def save_processed(ratings: pd.DataFrame, movies: pd.DataFrame) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    ratings.to_csv(PROCESSED_DIR / "ratings.csv", index=False)
    movies.to_csv(PROCESSED_DIR / "movies.csv", index=False)
    print(f"Saved {len(ratings):,} ratings and {len(movies):,} movies to {PROCESSED_DIR}")
