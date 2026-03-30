"""
FastAPI application entry point.

Startup: loads all trained model artifacts once into AppState.
Routes:
  POST /onboard                    — create session user from initial ratings
  GET  /session/{session_id}       — recs for new session user
  GET  /recommendations/{user_id}  — recs for known CF user
  GET  /search                     — title/genre search + optional reranking
  GET  /item/{movie_id}            — single movie metadata
  GET  /movies                     — paginated movie list for onboarding
  GET  /metrics                    — precomputed model comparison data
  GET  /insights                   — structured insights page data
  GET  /health                     — liveness check

Run:
  uvicorn backend.main:app --reload --port 8000
"""
import sys
from pathlib import Path

# Make src importable from backend
sys.path.insert(0, str(Path(__file__).parents[1]))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routes.recommendations import router as rec_router
from backend.routes.search import router as search_router
from backend.routes.insights import router as insights_router

app = FastAPI(
    title="Hybrid Recommendation & Ranking Platform",
    description=(
        "ML-powered movie recommendation API. Trains and compares four models: "
        "popularity baseline, collaborative filtering (ALS), content-based, "
        "and a hybrid LightGBM reranker."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(rec_router, tags=["Recommendations"])
app.include_router(search_router, tags=["Search"])
app.include_router(insights_router, tags=["Insights & Metadata"])


@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok"}
