"""
Load all trained artifacts at startup and expose a single AppState object.

Loading happens once when the server starts. All routes share the same loaded
models — no re-loading per request.

Artifacts expected:
  artifacts/models/cf_model.pkl
  artifacts/models/content_model.pkl
  artifacts/models/reranker.pkl
  artifacts/metrics/comparison.json
  data/processed/movies.csv
  data/processed/train.parquet   (for item stats + user history lookup)
"""
from __future__ import annotations

import ast
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parents[1]


class AppState:
    def __init__(self):
        print("Loading models and data...")

        # --- Movie metadata ---
        self.movies = pd.read_csv(ROOT / "data/processed/movies.csv")
        self.movies["genres"] = self.movies["genres"].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )
        self.movie_lookup: dict[int, dict] = {
            int(r.movie_id): {
                "movie_id": int(r.movie_id),
                "title": r.title,
                "year": int(r.year) if not pd.isna(r.year) else None,
                "genres": r.genres if isinstance(r.genres, list) else [],
            }
            for r in self.movies.itertuples()
        }

        # --- Training data (for item stats and user seen-sets) ---
        self.train = pd.read_parquet(ROOT / "data/processed/train.parquet")
        self.item_stats = (
            self.train.groupby("movie_id")["rating"]
            .agg(avg_rating="mean", rating_count="count")
            .assign(pop_score=lambda d: d["rating_count"] * d["avg_rating"])
        )
        self.user_train_count = (
            self.train.groupby("user_id")["movie_id"].count().to_dict()
        )
        self.genre_lookup: dict[int, set[str]] = {
            mid: set(info["genres"]) for mid, info in self.movie_lookup.items()
        }

        # --- Models ---
        with open(ROOT / "artifacts/models/cf_model.pkl", "rb") as f:
            self.cf = pickle.load(f)
        with open(ROOT / "artifacts/models/content_model.pkl", "rb") as f:
            self.cb = pickle.load(f)
        with open(ROOT / "artifacts/models/reranker.pkl", "rb") as f:
            rd = pickle.load(f)
        self.ranker = rd["ranker"]
        self.feature_importance = rd["importances"]

        # Popularity model (inline — lightweight)
        from src.models.popularity import PopularityModel
        self.pop = PopularityModel(score_mode="weighted")
        self.pop.fit(self.train)

        # --- Precomputed metrics ---
        with open(ROOT / "artifacts/metrics/comparison.json") as f:
            self.comparison = json.load(f)

        # --- Poster URLs (optional; fetched from TMDB) ---
        posters_path = ROOT / "artifacts" / "posters.json"
        if posters_path.exists():
            with open(posters_path) as f:
                raw = json.load(f)
            self.posters: dict[int, str | None] = {int(k): v for k, v in raw.items()}
            print(f"Posters: {sum(1 for v in self.posters.values() if v):,} loaded")
        else:
            self.posters = {}

        # --- Session store (in-memory; ephemeral users created via /onboard) ---
        # Maps session_user_id -> list of (movie_id, rating) tuples
        self.sessions: dict[str, list[tuple[int, float]]] = {}

        print(
            f"Ready: {len(self.movie_lookup):,} movies | "
            f"{self.train['user_id'].nunique():,} trained users"
        )


# Singleton — instantiated once on import
state = AppState()
