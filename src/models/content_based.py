"""
Content-based recommender using sentence embeddings.

Item vectors are built by encoding the following text per movie with
all-MiniLM-L6-v2 (384-dim):

    "{title}. {genres}. {description}"

If no description is available, only title + genres are used.

User preference profile = mean of item vectors for movies rated ≥ POSITIVE_THRESHOLD,
L2-normalized so that scoring is a plain dot product (cosine similarity).

A separate _user_genre_profiles dict stores the user's genre distribution (fraction
of liked movies per genre) — used as a feature by the hybrid reranker.

Strength: distinguishes movies within the same genre via plot semantics.
Weakness: embedding quality depends on description availability (~97% via TMDB).
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

ARTIFACTS_DIR      = Path(__file__).parents[2] / "artifacts"
POSITIVE_THRESHOLD = 3.5
EMBED_MODEL        = "all-MiniLM-L6-v2"


class ContentBasedModel:
    def __init__(self):
        self._item_vectors: np.ndarray | None = None   # shape (n_items, 384)
        self._item_index: dict[int, int]       = {}    # movie_id → row
        self._index_item: dict[int, int]       = {}    # row → movie_id
        self._user_seen:   dict[int, set[int]] = {}
        self._user_profiles:       dict[int, np.ndarray]       = {}
        self._user_genre_profiles: dict[int, dict[str, float]] = {}

    def fit(
        self,
        train: pd.DataFrame,
        movies: pd.DataFrame,
        descriptions: pd.DataFrame | None = None,
    ) -> "ContentBasedModel":
        """
        Build sentence-embedding item vectors and user preference profiles.

        Parameters
        ----------
        train        : ratings DataFrame with columns [user_id, movie_id, rating]
        movies       : movies DataFrame with columns [movie_id, title, genres]
        descriptions : optional DataFrame with columns [movie_id, description]
                       from scripts/fetch_descriptions.py
        """
        from sentence_transformers import SentenceTransformer

        movies = movies.copy()

        # Merge in descriptions if provided
        if descriptions is not None:
            movies = movies.merge(descriptions[["movie_id", "description"]],
                                  on="movie_id", how="left")
            movies["description"] = movies["description"].fillna("")
        else:
            movies["description"] = ""

        # Build one text string per movie: "Title. Genre1 Genre2. Description."
        def _text(row) -> str:
            genres = row.genres if isinstance(row.genres, list) else []
            genre_str = " ".join(genres)
            parts = [row.title]
            if genre_str:
                parts.append(genre_str)
            if row.description:
                parts.append(row.description)
            return ". ".join(parts)

        texts = [_text(r) for r in movies.itertuples()]
        movie_ids = movies["movie_id"].tolist()

        # Build index
        self._item_index = {m: i for i, m in enumerate(movie_ids)}
        self._index_item = {i: m for m, i in self._item_index.items()}

        # Encode with sentence transformer — returns L2-normalized 384-dim vectors
        print(f"  Encoding {len(texts):,} movies with {EMBED_MODEL}…")
        model = SentenceTransformer(EMBED_MODEL)
        embeddings = model.encode(
            texts,
            batch_size=512,
            show_progress_bar=True,
            normalize_embeddings=True,   # L2-norm → cosine sim = dot product
            convert_to_numpy=True,
        )
        self._item_vectors = embeddings.astype(np.float32)  # (n_movies, 384)

        # User seen sets
        self._user_seen = train.groupby("user_id")["movie_id"].apply(set).to_dict()

        # Precompute user profiles (embedding-based) and genre profiles
        pos = train[train["rating"] >= POSITIVE_THRESHOLD]
        genre_lookup: dict[int, list[str]] = {
            int(r.movie_id): (r.genres if isinstance(r.genres, list) else [])
            for r in movies.itertuples()
        }

        self._user_profiles       = {}
        self._user_genre_profiles = {}

        for user_id, group in pos.groupby("user_id"):
            liked_ids = group["movie_id"].tolist()

            # Embedding profile
            indices = [self._item_index[m] for m in liked_ids if m in self._item_index]
            if indices:
                profile = self._item_vectors[indices].mean(axis=0)
                norm = np.linalg.norm(profile)
                self._user_profiles[user_id] = profile / norm if norm > 0 else profile

            # Genre profile (fraction of liked movies per genre)
            all_genres = [g for mid in liked_ids for g in genre_lookup.get(mid, [])]
            if all_genres:
                counts: dict[str, int] = {}
                for g in all_genres:
                    counts[g] = counts.get(g, 0) + 1
                total = sum(counts.values())
                self._user_genre_profiles[user_id] = {
                    g: c / total for g, c in counts.items()
                }

        print(f"  Built profiles for {len(self._user_profiles):,} users")
        return self

    def recommend(self, user_id: int, top_k: int = 10) -> list[tuple[int, float]]:
        profile = self._user_profiles.get(user_id)
        if profile is None:
            return []

        scores = self._item_vectors.dot(profile)  # cosine sim (vectors are normalized)
        seen   = self._user_seen.get(user_id, set())

        ranked = [
            (self._index_item[i], float(scores[i]))
            for i in np.argsort(-scores)
            if self._index_item.get(i) not in seen
        ]
        return ranked[:top_k]

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
