"""
Content-based recommender.

Item vectors are built from:
  - genre one-hot encoding
  - TF-IDF over tag text (if tags available, else skip)

User preference profile = mean of item vectors for movies the user rated positively.

Scoring: cosine similarity between user profile and candidate item vectors.

Strength: works for cold-start users where CF has no behavioral signal.
Weakness: prone to over-recommending within narrow genre bands (filter bubble).
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
import scipy.sparse as sp

ARTIFACTS_DIR = Path(__file__).parents[2] / "artifacts"
POSITIVE_THRESHOLD = 3.5


class ContentBasedModel:
    def __init__(self, use_tags: bool = True, tfidf_max_features: int = 500):
        self.use_tags = use_tags
        self.tfidf_max_features = tfidf_max_features
        self._item_vectors: sp.csr_matrix | None = None   # shape (n_items, n_features)
        self._item_index: dict[int, int] = {}             # movie_id -> row in item_vectors
        self._index_item: dict[int, int] = {}
        self._mlb: MultiLabelBinarizer | None = None
        self._tfidf: TfidfVectorizer | None = None
        self._user_seen: dict[int, set[int]] = {}

    def fit(
        self,
        train: pd.DataFrame,
        movies: pd.DataFrame,
        tags: pd.DataFrame | None = None,
    ) -> "ContentBasedModel":
        """Build item feature vectors and user preference profiles."""
        movies = movies.copy()

        # --- Genre features ---
        self._mlb = MultiLabelBinarizer()
        genre_matrix = self._mlb.fit_transform(movies["genres"].fillna("").apply(
            lambda x: x if isinstance(x, list) else x.split("|")
        ))

        # --- Tag features (optional) ---
        if self.use_tags and tags is not None:
            tag_text = tags.groupby("movie_id")["tag"].apply(" ".join).reset_index()
            movies = movies.merge(tag_text, on="movie_id", how="left")
            movies["tag"] = movies["tag"].fillna("")
            self._tfidf = TfidfVectorizer(max_features=self.tfidf_max_features)
            tag_matrix = self._tfidf.fit_transform(movies["tag"])
            feature_matrix = sp.hstack([genre_matrix, tag_matrix])
        else:
            feature_matrix = sp.csr_matrix(genre_matrix)

        # Build index
        movie_ids = movies["movie_id"].tolist()
        self._item_index = {m: i for i, m in enumerate(movie_ids)}
        self._index_item = {i: m for m, i in self._item_index.items()}
        self._item_vectors = feature_matrix  # shape: (n_movies, n_features)

        self._user_seen = train.groupby("user_id")["movie_id"].apply(set).to_dict()

        # Precompute and cache user profiles so we don't need to store _train
        pos = train[train["rating"] >= POSITIVE_THRESHOLD]
        self._user_profiles: dict[int, np.ndarray] = {}
        for user_id, group in pos.groupby("user_id"):
            indices = [self._item_index[m] for m in group["movie_id"] if m in self._item_index]
            if not indices:
                continue
            profile = np.asarray(self._item_vectors[indices].mean(axis=0)).flatten()
            norm = np.linalg.norm(profile)
            self._user_profiles[user_id] = profile / norm if norm > 0 else profile

        return self

    def _user_profile(self, user_id: int) -> np.ndarray | None:
        """Return precomputed user content preference vector."""
        return self._user_profiles.get(user_id)

    def recommend(self, user_id: int, top_k: int = 10) -> list[tuple[int, float]]:
        profile = self._user_profile(user_id)
        if profile is None:
            return []

        # Cosine similarity: profile is already L2-normalized
        scores = self._item_vectors.dot(profile)
        seen = self._user_seen.get(user_id, set())

        ranked = [
            (self._index_item[i], float(scores[i]))
            for i in np.argsort(-scores)
            if self._index_item[i] not in seen
        ]
        return ranked[:top_k]

    def item_vector(self, movie_id: int) -> np.ndarray | None:
        if movie_id not in self._item_index:
            return None
        return np.asarray(self._item_vectors[self._item_index[movie_id]].todense()).flatten()

    def save(self, path: Path | None = None) -> None:
        path = path or (ARTIFACTS_DIR / "models" / "content_model.pkl")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path | None = None) -> "ContentBasedModel":
        path = path or (ARTIFACTS_DIR / "models" / "content_model.pkl")
        with open(path, "rb") as f:
            return pickle.load(f)
