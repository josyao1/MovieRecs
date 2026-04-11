"""
Load all trained artifacts at startup and expose a single AppState object.

Loading happens once when the server starts. All routes share the same loaded
models — no re-loading per request.

Artifacts expected:
  artifacts/models/cf_model.pkl
  artifacts/models/content_model.pkl
  artifacts/models/popularity_model.pkl
  artifacts/models/reranker.pkl
  artifacts/metrics/comparison.json
  artifacts/precomputed/item_stats.parquet     (from scripts/slim_models.py)
  artifacts/precomputed/user_train_count.json  (from scripts/slim_models.py)
  artifacts/precomputed/user_seen.parquet      (from scripts/slim_models.py)
  data/processed/movies.csv

Falls back to building from train.parquet if precomputed files are absent
(backwards-compatible with pre-slim workflow).
"""
from __future__ import annotations

import ast
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parents[1]
COMPUTED = ROOT / "artifacts" / "precomputed"


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
        self.genre_lookup: dict[int, set[str]] = {
            mid: set(info["genres"]) for mid, info in self.movie_lookup.items()
        }

        # --- Item stats + user interaction counts ---
        if (COMPUTED / "item_stats.parquet").exists():
            _item_stats = pd.read_parquet(COMPUTED / "item_stats.parquet")
            self.item_stats = _item_stats.set_index("movie_id")
            print(f"  item_stats: {len(self.item_stats):,} movies (precomputed)")
        else:
            print("  Precomputed item_stats not found, building from train.parquet…")
            train = pd.read_parquet(ROOT / "data/processed/train.parquet")
            self.item_stats = (
                train.groupby("movie_id")["rating"]
                .agg(avg_rating="mean", rating_count="count")
                .assign(pop_score=lambda d: d["rating_count"] * d["avg_rating"])
            )

        if (COMPUTED / "user_train_count.json").exists():
            with open(COMPUTED / "user_train_count.json") as f:
                self.user_train_count: dict[int, int] = {
                    int(k): v for k, v in json.load(f).items()
                }
            print(f"  user_train_count: {len(self.user_train_count):,} users (precomputed)")
        else:
            if not hasattr(self, "_train_loaded"):
                train = pd.read_parquet(ROOT / "data/processed/train.parquet")
            self.user_train_count = (
                train.groupby("user_id")["movie_id"].count().to_dict()
            )

        # --- User seen sets (used to filter already-watched movies) ---
        if (COMPUTED / "user_seen.parquet").exists():
            _seen_df = pd.read_parquet(COMPUTED / "user_seen.parquet")
            user_seen: dict[int, set[int]] = (
                _seen_df.groupby("user_id")["movie_id"].apply(set).to_dict()
            )
            print(f"  user_seen: {len(user_seen):,} users (precomputed)")
        else:
            if not hasattr(self, "train"):
                train = pd.read_parquet(ROOT / "data/processed/train.parquet")
            user_seen = train.groupby("user_id")["movie_id"].apply(set).to_dict()

        # --- Models ---
        with open(ROOT / "artifacts/models/cf_model.pkl", "rb") as f:
            self.cf = pickle.load(f)
        # Inject user_seen if it was stripped for size
        if not getattr(self.cf, "_user_seen", None):
            self.cf._user_seen = user_seen

        with open(ROOT / "artifacts/models/content_model.pkl", "rb") as f:
            self.cb = pickle.load(f)
        if not getattr(self.cb, "_user_seen", None):
            self.cb._user_seen = user_seen
        # Old genre-based models don't have _user_genre_profiles — provide empty fallback
        if not hasattr(self.cb, "_user_genre_profiles"):
            self.cb._user_genre_profiles = {}

        with open(ROOT / "artifacts/models/reranker.pkl", "rb") as f:
            rd = pickle.load(f)
        self.ranker = rd["ranker"]
        self.feature_importance = rd["importances"]

        # Popularity model
        with open(ROOT / "artifacts/models/popularity_model.pkl", "rb") as f:
            self.pop = pickle.load(f)
        if not getattr(self.pop, "_user_seen", None):
            self.pop._user_seen = user_seen

        # --- Precomputed metrics ---
        with open(ROOT / "artifacts/metrics/comparison.json") as f:
            self.comparison = json.load(f)

        # --- Poster URLs ---
        posters_path = ROOT / "artifacts" / "posters.json"
        if posters_path.exists():
            with open(posters_path) as f:
                raw = json.load(f)
            self.posters: dict[int, str | None] = {int(k): v for k, v in raw.items()}
            print(f"  Posters: {sum(1 for v in self.posters.values() if v):,} loaded")
        else:
            self.posters = {}

        # --- Semantic search: query encoder + description lookup + filter mask ---
        try:
            from sentence_transformers import SentenceTransformer
            from src.models.content_based import EMBED_MODEL
            self.query_encoder = SentenceTransformer(EMBED_MODEL)
            print(f"  Query encoder loaded: {EMBED_MODEL}")
        except Exception as e:
            self.query_encoder = None
            print(f"  Query encoder failed to load ({e}) — semantic search disabled")

        desc_path = ROOT / "data/processed/movie_descriptions.csv"
        if desc_path.exists():
            desc_df = pd.read_csv(desc_path)
            self.description_lookup: dict[int, str] = {
                int(r.movie_id): (
                    r.description
                    if isinstance(r.description, str) and len(r.description) >= 50
                    else ""
                )
                for r in desc_df.itertuples()
            }
        else:
            self.description_lookup = {}
            print("  movie_descriptions.csv not found — semantic search disabled")

        # Boolean mask over the item vector matrix — True = eligible for semantic search
        n_items = len(self.cb._item_index)
        self.semantic_mask = np.zeros(n_items, dtype=bool)
        for movie_id, row_idx in self.cb._item_index.items():
            if self.description_lookup.get(movie_id, ""):
                self.semantic_mask[row_idx] = True
        print(f"  Semantic search: {self.semantic_mask.sum():,} eligible movies")

        # --- Session store ---
        self.sessions: dict[str, list[tuple[int, float]]] = {}

        print(
            f"Ready: {len(self.movie_lookup):,} movies | "
            f"{len(self.user_train_count):,} trained users"
        )


# Singleton — instantiated once on import
state = AppState()
