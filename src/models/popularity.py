"""
Popularity baseline recommender.

Recommends globally popular unseen items for each user.
"Popular" = weighted score of interaction count and average rating.

This is the benchmark every other model must beat.
"""
from __future__ import annotations

import pandas as pd
import numpy as np


class PopularityModel:
    def __init__(self, score_mode: str = "weighted"):
        """
        score_mode:
          'count'    — rank by raw interaction count
          'rating'   — rank by average rating (min_count threshold applied)
          'weighted' — count * avg_rating (Bayesian-style blend)
        """
        self.score_mode = score_mode
        self._item_scores: pd.Series | None = None  # movie_id -> score

    def fit(self, train: pd.DataFrame) -> "PopularityModel":
        stats = (
            train.groupby("movie_id")["rating"]
            .agg(count="count", mean="mean")
            .reset_index()
        )

        if self.score_mode == "count":
            stats["score"] = stats["count"]
        elif self.score_mode == "rating":
            min_count = stats["count"].quantile(0.25)
            stats = stats[stats["count"] >= min_count]
            stats["score"] = stats["mean"]
        else:  # weighted
            stats["score"] = stats["count"] * stats["mean"]

        self._item_scores = stats.set_index("movie_id")["score"].sort_values(ascending=False)
        self._user_seen: dict[int, set[int]] = (
            train.groupby("user_id")["movie_id"].apply(set).to_dict()
        )
        return self

    def recommend(self, user_id: int, top_k: int = 10) -> list[tuple[int, float]]:
        """Return [(movie_id, score), ...] sorted by score descending."""
        seen = self._user_seen.get(user_id, set())
        recs = [
            (int(movie_id), float(score))
            for movie_id, score in self._item_scores.items()
            if movie_id not in seen
        ]
        return recs[:top_k]

    def recommend_batch(
        self, user_ids: list[int], top_k: int = 10
    ) -> dict[int, list[tuple[int, float]]]:
        return {uid: self.recommend(uid, top_k) for uid in user_ids}
