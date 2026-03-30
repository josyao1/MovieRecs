"""
Search endpoint with optional personalized reranking.

GET /search?q=matrix&user_id=42&top_k=10

If user_id is provided, results are reranked by relevance to that user.
If not, results are ordered by text match quality + popularity.
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class SearchResult(BaseModel):
    movie_id: int
    title: str
    year: Optional[int]
    genres: list[str]
    poster_url: Optional[str] = None
    match_score: float        # text relevance score
    personalized_score: float # reranker score (0 if no user)
    explanation: str


@router.get("/search", response_model=list[SearchResult])
def search(q: str, user_id: Optional[int] = None, session_id: Optional[str] = None, top_k: int = 10):
    """
    Search movies by title or genre keywords.
    Optionally rerank by personalization if user_id or session_id is provided.

    This demonstrates the ranking platform aspect: same content, different orderings
    based on who is asking.
    """
    from backend.model_loader import state
    import numpy as np
    import pandas as pd
    from backend.inference import _genre_profile_from_ratings, _build_feature_row

    q_lower = q.lower().strip()

    # --- Text matching: title substring + genre keyword ---
    matches = []
    for mid, info in state.movie_lookup.items():
        title_score = 0.0
        genre_score = 0.0

        if q_lower in info["title"].lower():
            # Exact substring match weighted by position
            pos = info["title"].lower().find(q_lower)
            title_score = 1.0 if pos == 0 else 0.7

        for g in info["genres"]:
            if q_lower in g.lower():
                genre_score = 0.5
                break

        match_score = max(title_score, genre_score)
        if match_score > 0:
            matches.append((mid, match_score))

    if not matches:
        return []

    # --- Personalized reranking if user context provided ---
    if user_id is not None or session_id is not None:
        if user_id is not None and user_id in state.cf._user_index:
            cf_recs = dict(state.cf.recommend(user_id, 200))
            cb_recs = dict(state.cb.recommend(user_id, 200))
            user_train = state.train[state.train["user_id"] == user_id]
            rated = list(zip(user_train["movie_id"], user_train["rating"]))
            genre_profile = _genre_profile_from_ratings(rated, state)
            uid_key = user_id
        elif session_id and session_id in state.sessions:
            cf_recs = {}
            cb_recs = {}
            genre_profile = _genre_profile_from_ratings(state.sessions[session_id], state)
            uid_key = "session"
        else:
            cf_recs, cb_recs, genre_profile, uid_key = {}, {}, {}, None

        if uid_key is not None:
            candidate_ids = [mid for mid, _ in matches]
            rows = [
                _build_feature_row(uid_key, mid, cf_recs.get(mid, 0.0),
                                   cb_recs.get(mid, 0.0), genre_profile, state)
                for mid in candidate_ids
            ]
            p_scores = state.ranker.predict(pd.DataFrame(rows))
            personalized = dict(zip(candidate_ids, p_scores))
        else:
            personalized = {}
    else:
        personalized = {}

    # --- Combine and sort ---
    from backend.inference import _explain
    results = []
    for mid, match_score in matches:
        info = state.movie_lookup[mid]
        p_score = float(personalized.get(mid, 0.0))
        cf_recs_local = cf_recs if 'cf_recs' in dir() else {}
        cb_recs_local = cb_recs if 'cb_recs' in dir() else {}
        gp_local = genre_profile if 'genre_profile' in dir() else {}
        explanation = _explain(mid, cf_recs_local, cb_recs_local, gp_local, state) if personalized else "Matches your search"
        results.append(SearchResult(
            movie_id=mid,
            title=info["title"],
            year=info["year"],
            genres=info["genres"],
            poster_url=state.posters.get(mid),
            match_score=round(match_score, 4),
            personalized_score=round(p_score, 4),
            explanation=explanation,
        ))

    # Sort: if personalized, blend scores; else just match score
    if personalized:
        results.sort(key=lambda r: -(0.4 * r.match_score + 0.6 * r.personalized_score))
    else:
        results.sort(key=lambda r: -r.match_score)

    return results[:top_k]
