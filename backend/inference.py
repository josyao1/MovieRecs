"""
Core inference pipeline: candidate generation → feature engineering → reranking.

This is shared by all recommendation endpoints. It handles:
  - Known users (in CF model): full hybrid pipeline
  - Session users (just onboarded): content-based + popularity fallback
  - Explanation tag generation
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CANDIDATE_K = 50
POSITIVE_THRESHOLD = 4.0


def _build_feature_row(
    user_id,
    movie_id: int,
    cf_score: float,
    cb_score: float,
    genre_profile: dict[str, float],
    state,
) -> dict:
    stats = state.item_stats.loc[movie_id] if movie_id in state.item_stats.index else None
    return {
        "cf_score": cf_score,
        "content_score": cb_score,
        "pop_score": float(stats["pop_score"]) if stats is not None else 0.0,
        "avg_rating": float(stats["avg_rating"]) if stats is not None else 0.0,
        "rating_count": float(np.log1p(stats["rating_count"])) if stats is not None else 0.0,
        "genre_overlap": sum(
            genre_profile.get(g, 0.0) for g in state.genre_lookup.get(movie_id, set())
        ),
        "user_interaction_count": float(
            np.log1p(state.user_train_count.get(user_id, 0))
        ),
    }


def _genre_profile_from_ratings(
    rated_movies: list[tuple[int, float]], state
) -> dict[str, float]:
    """Build genre preference profile from a list of (movie_id, rating) pairs."""
    liked = [mid for mid, r in rated_movies if r >= POSITIVE_THRESHOLD]
    all_genres = [g for mid in liked for g in state.genre_lookup.get(mid, set())]
    if not all_genres:
        return {}
    return pd.Series(all_genres).value_counts(normalize=True).to_dict()


def recommend_for_known_user(user_id: int, top_k: int, state) -> list[dict]:
    """Full hybrid pipeline for a user already in the CF model."""
    cf_recs = dict(state.cf.recommend(user_id, CANDIDATE_K))

    if not cf_recs:
        return recommend_popular(user_id, top_k, state)

    # CF-only candidates — content candidates are niche movies that don't appear
    # in user histories, so they only add noise. Content scores are still computed
    # for the CF candidates as a reranking feature.
    candidates = list(cf_recs.keys())
    cb_recs = state.cb.score_items(user_id, candidates)

    # Genre profile for genre_overlap feature — stored separately from embedding profile
    genre_profile = state.cb._user_genre_profiles.get(user_id, {})

    rows = [
        _build_feature_row(user_id, mid, cf_recs.get(mid, 0.0), cb_recs.get(mid, 0.0),
                           genre_profile, state)
        for mid in candidates
    ]
    scores = state.ranker.predict(pd.DataFrame(rows))
    ranked = sorted(zip(candidates, scores), key=lambda x: -x[1])

    return _format_results(ranked[:top_k], cf_recs, cb_recs, genre_profile, state)


def recommend_for_session_user(
    session_ratings: list[tuple[int, float]], top_k: int, state
) -> list[dict]:
    """
    Recommend for a brand-new user who just onboarded.
    No CF embedding exists yet — use content-based profile + popularity fallback.
    """
    genre_profile = _genre_profile_from_ratings(session_ratings, state)
    seen = {mid for mid, _ in session_ratings}

    # Candidate pool: content-based scores over all movies.
    # For session users we don't have a precomputed profile, so we build one on
    # the fly by averaging the sentence-embedding vectors of liked movies.
    cb_scores: dict[int, float] = {}
    liked_indices = [
        state.cb._item_index[mid]
        for mid, r in session_ratings
        if r >= POSITIVE_THRESHOLD and mid in state.cb._item_index
    ]
    if liked_indices:
        session_vec = state.cb._item_vectors[liked_indices].mean(axis=0)
        norm = np.linalg.norm(session_vec)
        if norm > 0:
            session_vec = session_vec / norm
        raw_scores = state.cb._item_vectors.dot(session_vec)
        for i, score in enumerate(raw_scores):
            mid = state.cb._index_item.get(i)
            if mid and mid not in seen:
                cb_scores[mid] = float(score)

    # Top candidates by content score + fill with popular items
    top_cb = sorted(cb_scores.items(), key=lambda x: -x[1])[:CANDIDATE_K]
    pop_fallback = [
        (mid, score) for mid, score in state.pop._item_scores.items()
        if mid not in seen and mid not in dict(top_cb)
    ][:CANDIDATE_K]

    cb_score_map = dict(top_cb)
    candidates = list(cb_score_map) + [mid for mid, _ in pop_fallback if mid not in cb_score_map]

    rows = [
        _build_feature_row("session", mid, 0.0, cb_score_map.get(mid, 0.0), genre_profile, state)
        for mid in candidates
    ]
    scores = state.ranker.predict(pd.DataFrame(rows))
    ranked = sorted(zip(candidates, scores), key=lambda x: -x[1])

    return _format_results(ranked[:top_k], {}, dict(top_cb), genre_profile, state)


def recommend_popular(user_id, top_k: int, state) -> list[dict]:
    recs = state.pop.recommend(user_id, top_k)
    return _format_results(recs, {}, {}, {}, state)


def _format_results(
    ranked: list[tuple[int, float]],
    cf_scores: dict[int, float],
    cb_scores: dict[int, float],
    genre_profile: dict[str, float],
    state,
) -> list[dict]:
    results = []
    for movie_id, score in ranked:
        info = state.movie_lookup.get(movie_id)
        if not info:
            continue
        results.append({
            **info,
            "poster_url": state.posters.get(movie_id),
            "score": round(float(score), 4),
            "explanation": _explain(movie_id, cf_scores, cb_scores, genre_profile, state),
        })
    return results


def semantic_search(query_vec: np.ndarray, top_k: int, state) -> list[dict]:
    """
    Cosine similarity search against the precomputed item embedding matrix.
    Only returns movies with descriptions >= 50 chars (enforced by semantic_mask).
    query_vec must already be L2-normalized (shape: [384,]).
    """
    scores = state.cb._item_vectors.dot(query_vec)
    masked = scores * state.semantic_mask  # zeroes ineligible rows

    # argpartition is O(n) — faster than full argsort for large n
    k = min(top_k, len(masked))
    top_indices = np.argpartition(-masked, k)[:k]
    top_indices = top_indices[np.argsort(-masked[top_indices])]

    results = []
    for row_idx in top_indices:
        movie_id = state.cb._index_item.get(row_idx)
        if movie_id is None:
            continue
        info = state.movie_lookup.get(movie_id)
        if not info:
            continue
        description = state.description_lookup.get(movie_id, "")
        snippet = description[:200] + "…" if len(description) > 200 else description
        results.append({
            **info,
            "poster_url": state.posters.get(movie_id),
            "similarity_score": round(float(masked[row_idx]), 4),
            "description_snippet": snippet,
        })
    return results


def _explain(
    movie_id: int,
    cf_scores: dict[int, float],
    cb_scores: dict[int, float],
    genre_profile: dict[str, float],
    state,
) -> str:
    """Generate a human-readable explanation tag for why a movie was recommended."""
    genres = state.genre_lookup.get(movie_id, set())
    top_genre = max(genre_profile, key=genre_profile.get) if genre_profile else None
    cf_s = cf_scores.get(movie_id, 0.0)
    cb_s = cb_scores.get(movie_id, 0.0)
    stats = state.item_stats.loc[movie_id] if movie_id in state.item_stats.index else None
    rating_count = int(stats["rating_count"]) if stats is not None else 0

    # Pick the most informative explanation based on dominant signal
    if cf_s > 0.9:
        return "Highly rated by users with similar taste"
    if top_genre and top_genre in genres:
        return f"Matches your preference for {top_genre}"
    if cf_s > 0.5 and cb_s > 0.3:
        return "Similar to movies you rated highly"
    if rating_count > 1000:
        return "Popular among users with your viewing history"
    if rating_count < 100 and cf_s > 0.4:
        return "Hidden gem — discovered through users like you"
    if cb_s > cf_s:
        return f"Strong match for your {top_genre or 'favorite'} genre taste"
    return "Recommended based on your watching history"
