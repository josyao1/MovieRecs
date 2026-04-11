"""
Semantic (natural-language) search endpoint.

GET /semantic-search?q=lone+hero+in+a+lawless+frontier&top_k=10

Encodes the query with all-MiniLM-L6-v2 (same model used to build item embeddings),
then performs cosine similarity search against the precomputed 62K-movie item matrix.
Only returns movies with descriptions >= 50 chars — ensures embedding quality.
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Query

router = APIRouter()


@router.get("/semantic-search")
def semantic_search_endpoint(
    q: str = Query(..., min_length=1, max_length=500),
    top_k: int = Query(default=10, ge=1, le=50),
    year_min: int = Query(default=None),
):
    from backend.model_loader import state
    from backend.inference import semantic_search

    query_vec = state.query_encoder.encode(
        q.strip(),
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return semantic_search(query_vec, top_k=top_k, state=state, year_min=year_min)
