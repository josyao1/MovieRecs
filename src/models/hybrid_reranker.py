"""
Hybrid reranker.

Architecture:
  1. Candidate generation: union of top-K from CF and top-K from content-based
  2. Feature engineering: build a feature row for every (user, candidate) pair
  3. Reranking: LightGBM ranker predicts final relevance score

This is the production-style architecture used in real recommendation systems:
fast retrieval (CF + content-based) feeds a more expensive but accurate ranker.

The LightGBM ranker uses LambdaRank (objective='lambdarank') which optimizes
NDCG directly — exactly the metric we care about.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("Warning: lightgbm not installed. Run: pip install lightgbm")

from .popularity import PopularityModel
from .collaborative_filter import CollaborativeFilterModel
from .content_based import ContentBasedModel

ARTIFACTS_DIR = Path(__file__).parents[2] / "artifacts"
CANDIDATE_K = 50   # candidates per model before reranking
POSITIVE_THRESHOLD = 3.5


class HybridReranker:
    def __init__(self, n_estimators: int = 200, learning_rate: float = 0.05):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self._ranker = None
        self._cf: CollaborativeFilterModel | None = None
        self._cb: ContentBasedModel | None = None
        self._pop: PopularityModel | None = None
        self._item_stats: pd.DataFrame | None = None  # movie_id stats for features
        self._user_stats: pd.DataFrame | None = None
        self._genre_lookup: dict[int, set[str]] = {}

    def set_component_models(
        self,
        cf: CollaborativeFilterModel,
        cb: ContentBasedModel,
        pop: PopularityModel,
    ) -> None:
        self._cf = cf
        self._cb = cb
        self._pop = pop

    def _build_features(
        self,
        user_id: int,
        candidate_ids: list[int],
        cf_scores: dict[int, float],
        cb_scores: dict[int, float],
        user_genre_profile: dict[str, float],
    ) -> pd.DataFrame:
        """Build feature row for each (user, candidate) pair."""
        rows = []
        for movie_id in candidate_ids:
            pop_score = self._item_stats.loc[movie_id, "pop_score"] if movie_id in self._item_stats.index else 0.0
            avg_rating = self._item_stats.loc[movie_id, "avg_rating"] if movie_id in self._item_stats.index else 0.0
            rating_count = self._item_stats.loc[movie_id, "rating_count"] if movie_id in self._item_stats.index else 0
            item_genres = self._genre_lookup.get(movie_id, set())
            genre_overlap = sum(user_genre_profile.get(g, 0.0) for g in item_genres)

            rows.append({
                "cf_score": cf_scores.get(movie_id, 0.0),
                "content_score": cb_scores.get(movie_id, 0.0),
                "pop_score": pop_score,
                "avg_rating": avg_rating,
                "rating_count": np.log1p(rating_count),
                "genre_overlap": genre_overlap,
                "user_interaction_count": self._user_stats.get(user_id, 0),
            })
        return pd.DataFrame(rows)

    def fit(
        self,
        train: pd.DataFrame,
        val: pd.DataFrame,
        movies: pd.DataFrame,
    ) -> "HybridReranker":
        if not HAS_LGB:
            raise ImportError("pip install lightgbm")

        # Build item and user statistics
        self._item_stats = (
            train.groupby("movie_id")["rating"]
            .agg(avg_rating="mean", rating_count="count")
            .assign(pop_score=lambda d: d["rating_count"] * d["avg_rating"])
        )
        self._user_stats = train.groupby("user_id")["movie_id"].count().to_dict()
        self._genre_lookup = {
            row.movie_id: set(row.genres) if isinstance(row.genres, list) else set()
            for row in movies.itertuples()
        }

        # Build training data from validation set
        # Label: 1 if user interacted with movie in val with rating >= threshold
        val_positive = set(
            zip(val[val["rating"] >= POSITIVE_THRESHOLD]["user_id"],
                val[val["rating"] >= POSITIVE_THRESHOLD]["movie_id"])
        )

        X_rows, y_rows, groups = [], [], []
        sample_users = train["user_id"].unique()[:2000]  # cap for training speed

        for user_id in sample_users:
            cf_recs = dict(self._cf.recommend(user_id, CANDIDATE_K))
            cb_recs = dict(self._cb.recommend(user_id, CANDIDATE_K))
            candidates = list(set(cf_recs) | set(cb_recs))
            if not candidates:
                continue

            # User genre profile
            seen_genres = [
                g for m in train[train["user_id"] == user_id]["movie_id"]
                for g in self._genre_lookup.get(m, set())
            ]
            genre_counts = pd.Series(seen_genres).value_counts(normalize=True).to_dict()

            features = self._build_features(user_id, candidates, cf_recs, cb_recs, genre_counts)
            labels = [1 if (user_id, m) in val_positive else 0 for m in candidates]

            X_rows.append(features)
            y_rows.extend(labels)
            groups.append(len(candidates))

        X = pd.concat(X_rows, ignore_index=True)
        y = np.array(y_rows)

        self._ranker = lgb.LGBMRanker(
            objective="lambdarank",
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=31,
            verbose=-1,
        )
        self._ranker.fit(X, y, group=groups)
        print(f"Reranker trained on {len(groups):,} users, {len(X):,} candidate pairs")
        return self

    def recommend(self, user_id: int, top_k: int = 10) -> list[tuple[int, float]]:
        cf_recs = dict(self._cf.recommend(user_id, CANDIDATE_K))
        cb_recs = dict(self._cb.recommend(user_id, CANDIDATE_K))
        candidates = list(set(cf_recs) | set(cb_recs))

        if not candidates:
            return self._pop.recommend(user_id, top_k)

        genre_counts: dict[str, float] = {}
        features = self._build_features(user_id, candidates, cf_recs, cb_recs, genre_counts)
        scores = self._ranker.predict(features)

        ranked = sorted(zip(candidates, scores), key=lambda x: -x[1])
        return [(int(m), float(s)) for m, s in ranked[:top_k]]

    @property
    def feature_importance(self) -> pd.DataFrame:
        if self._ranker is None:
            raise ValueError("Model not trained yet")
        feature_names = [
            "cf_score", "content_score", "pop_score", "avg_rating",
            "rating_count", "genre_overlap", "user_interaction_count"
        ]
        return pd.DataFrame({
            "feature": feature_names,
            "importance": self._ranker.feature_importances_,
        }).sort_values("importance", ascending=False)

    def save(self, path: Path | None = None) -> None:
        path = path or (ARTIFACTS_DIR / "models" / "reranker.pkl")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path | None = None) -> "HybridReranker":
        path = path or (ARTIFACTS_DIR / "models" / "reranker.pkl")
        with open(path, "rb") as f:
            return pickle.load(f)
