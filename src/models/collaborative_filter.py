"""
Collaborative Filtering via implicit Alternating Least Squares (ALS).

Uses the `implicit` library which implements efficient ALS for implicit feedback.
Ratings >= POSITIVE_THRESHOLD are treated as positive interactions;
the confidence weight scales with the rating value.

Why implicit feedback instead of explicit rating prediction?
The task is ranking, not rating prediction. We want to surface items the user
will engage with, not predict the exact star rating. Treating ratings as
confidence-weighted implicit feedback is more aligned with the ranking objective.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

try:
    import implicit
    HAS_IMPLICIT = True
except ImportError:
    HAS_IMPLICIT = False
    print("Warning: `implicit` not installed. Run: pip install implicit")

ARTIFACTS_DIR = Path(__file__).parents[2] / "artifacts"
POSITIVE_THRESHOLD = 3.5  # ratings below this are excluded
CONFIDENCE_ALPHA = 40     # confidence = 1 + alpha * rating (standard ALS param)


class CollaborativeFilterModel:
    def __init__(self, factors: int = 64, iterations: int = 20, regularization: float = 0.01):
        self.factors = factors
        self.iterations = iterations
        self.regularization = regularization
        self._model = None
        self._user_index: dict[int, int] = {}   # user_id -> matrix row
        self._item_index: dict[int, int] = {}   # movie_id -> matrix col
        self._index_user: dict[int, int] = {}   # matrix row -> user_id
        self._index_item: dict[int, int] = {}   # matrix col -> movie_id
        self._user_seen: dict[int, set[int]] = {}

    def fit(self, train: pd.DataFrame) -> "CollaborativeFilterModel":
        if not HAS_IMPLICIT:
            raise ImportError("pip install implicit")

        # Filter to positive interactions only
        pos = train[train["rating"] >= POSITIVE_THRESHOLD].copy()

        # Build index mappings
        users = sorted(pos["user_id"].unique())
        items = sorted(pos["movie_id"].unique())
        self._user_index = {u: i for i, u in enumerate(users)}
        self._item_index = {m: i for i, m in enumerate(items)}
        self._index_user = {i: u for u, i in self._user_index.items()}
        self._index_item = {i: m for m, i in self._item_index.items()}

        # Build user-item confidence matrix (CSR)
        rows = pos["user_id"].map(self._user_index)
        cols = pos["movie_id"].map(self._item_index)
        confidence = 1 + CONFIDENCE_ALPHA * pos["rating"]
        user_item = sp.csr_matrix(
            (confidence, (rows, cols)),
            shape=(len(users), len(items)),
        )

        self._model = implicit.als.AlternatingLeastSquares(
            factors=self.factors,
            iterations=self.iterations,
            regularization=self.regularization,
            use_gpu=False,
        )
        self._model.fit(user_item)

        # Store seen items per user
        self._user_seen = train.groupby("user_id")["movie_id"].apply(set).to_dict()

        return self

    def recommend(self, user_id: int, top_k: int = 10) -> list[tuple[int, float]]:
        if user_id not in self._user_index:
            return []  # cold-start user — fall back to content-based or popularity

        u_idx = self._user_index[user_id]
        seen = self._user_seen.get(user_id, set())

        # Score all items
        user_vec = self._model.user_factors[u_idx]
        scores = self._model.item_factors @ user_vec

        # Build ranked list, excluding seen
        item_scores = [
            (self._index_item[i], float(scores[i]))
            for i in np.argsort(-scores)
            if self._index_item[i] not in seen
        ]
        return item_scores[:top_k]

    @property
    def user_embeddings(self) -> np.ndarray:
        return self._model.user_factors

    @property
    def item_embeddings(self) -> np.ndarray:
        return self._model.item_factors

    def save(self, path: Path | None = None) -> None:
        path = path or (ARTIFACTS_DIR / "models" / "cf_model.pkl")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path | None = None) -> "CollaborativeFilterModel":
        path = path or (ARTIFACTS_DIR / "models" / "cf_model.pkl")
        with open(path, "rb") as f:
            return pickle.load(f)
