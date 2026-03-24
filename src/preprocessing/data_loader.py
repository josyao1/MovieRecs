"""
Load and clean raw MovieLens 1M data into normalized DataFrames.

MovieLens 1M files:
  ratings.dat   :: UserID::MovieID::Rating::Timestamp
  movies.dat    :: MovieID::Title::Genres
  tags.dat      :: UserID::MovieID::Tag::Timestamp  (if using ML-10M or 20M)
"""
from __future__ import annotations

import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).parents[2] / "data" / "raw"
PROCESSED_DIR = Path(__file__).parents[2] / "data" / "processed"


def load_ratings(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Return ratings as a tidy DataFrame with columns:
    user_id, movie_id, rating, timestamp
    """
    path = raw_dir / "ratings.dat"
    df = pd.read_csv(
        path,
        sep="::",
        engine="python",
        names=["user_id", "movie_id", "rating", "timestamp"],
    )
    df = df.astype({"user_id": int, "movie_id": int, "rating": float, "timestamp": int})
    return df


def load_movies(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Return movie metadata as a DataFrame with columns:
    movie_id, title, genres (list), year
    """
    path = raw_dir / "movies.dat"
    df = pd.read_csv(
        path,
        sep="::",
        engine="python",
        names=["movie_id", "title", "genres"],
        encoding="latin-1",
    )
    df["genres"] = df["genres"].str.split("|")
    # Extract year from title, e.g. "Toy Story (1995)" -> 1995
    df["year"] = df["title"].str.extract(r"\((\d{4})\)").astype("Int64")
    df["movie_id"] = df["movie_id"].astype(int)
    return df


def save_processed(ratings: pd.DataFrame, movies: pd.DataFrame) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    ratings.to_csv(PROCESSED_DIR / "ratings.csv", index=False)
    movies.to_csv(PROCESSED_DIR / "movies.csv", index=False)
    print(f"Saved {len(ratings):,} ratings and {len(movies):,} movies to {PROCESSED_DIR}")
