"""
Recommendation endpoints.

POST /onboard         — Accept initial ratings, create a session user
GET  /recommendations/{user_id}  — Ranked recs for a known CF user
GET  /session/{session_id}       — Ranked recs for an onboarded session user
"""
from __future__ import annotations

import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()


# --- Request / Response models ---

class RatingItem(BaseModel):
    movie_id: int
    rating: float  # 1.0 - 5.0


class OnboardRequest(BaseModel):
    ratings: list[RatingItem]


class OnboardResponse(BaseModel):
    session_id: str
    message: str


class RecommendationItem(BaseModel):
    movie_id: int
    title: str
    year: Optional[int]
    genres: list[str]
    score: float
    explanation: str


# --- Endpoints ---

@router.post("/onboard", response_model=OnboardResponse)
def onboard(body: OnboardRequest):
    """
    Accept a list of movie ratings from a new user.
    Returns a session_id to use with GET /session/{session_id}.

    This simulates cold-start: the user tells us their taste upfront,
    and we use content-based + reranker to generate immediate recommendations.
    """
    from backend.model_loader import state

    if not body.ratings:
        raise HTTPException(status_code=400, detail="Provide at least one rating.")

    session_id = str(uuid.uuid4())[:8]
    state.sessions[session_id] = [
        (r.movie_id, r.rating) for r in body.ratings
    ]
    return OnboardResponse(
        session_id=session_id,
        message=f"Session created. Use GET /session/{session_id} for recommendations.",
    )


@router.get("/session/{session_id}", response_model=list[RecommendationItem])
def session_recommendations(session_id: str, top_k: int = 10):
    """
    Recommendations for a user who onboarded via POST /onboard.
    Uses content-based profile + reranker (no CF embedding for new users).
    """
    from backend.model_loader import state
    from backend.inference import recommend_for_session_user

    if session_id not in state.sessions:
        raise HTTPException(status_code=404, detail="Session not found. Call POST /onboard first.")

    session_ratings = state.sessions[session_id]
    results = recommend_for_session_user(session_ratings, top_k=top_k, state=state)
    return results


@router.get("/recommendations/{user_id}", response_model=list[RecommendationItem])
def recommendations(user_id: int, top_k: int = 10):
    """
    Ranked recommendations for a known user (present in the CF model).
    Uses full hybrid pipeline: CF + content-based candidates → reranker.
    """
    from backend.model_loader import state
    from backend.inference import recommend_for_known_user, recommend_popular

    if user_id not in state.cf._user_index:
        # User not in CF model — fall back to popularity
        results = recommend_popular(user_id, top_k=top_k, state=state)
    else:
        results = recommend_for_known_user(user_id, top_k=top_k, state=state)

    return results
