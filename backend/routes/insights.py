"""
Insights endpoints — serve precomputed model analysis data to the frontend.

GET /metrics     — full model comparison JSON
GET /insights    — structured narrative data for the insights page
GET /item/{id}   — single movie metadata
GET /movies      — paginated movie list (for onboarding picker)
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional

router = APIRouter()


@router.get("/metrics")
def metrics():
    """Return full model comparison metrics for all 4 models."""
    from backend.model_loader import state
    return state.comparison


@router.get("/insights")
def insights():
    """
    Return structured data for the insights page:
    - Model comparison table
    - Feature importance from reranker
    - Key findings narrative
    - Tradeoff analysis
    - Dataset stats
    """
    from backend.model_loader import state

    comp = state.comparison["models"]

    # Build comparison table rows
    table = []
    for model_name, data in comp.items():
        o = data["overall"]
        table.append({
            "model": model_name.replace("_", " ").title(),
            "precision_at_10": round(o["precision@10"], 4),
            "recall_at_10": round(o["recall@10"], 4),
            "ndcg_at_10": round(o["ndcg@10"], 4),
            "popularity_bias": round(data["popularity_bias"], 3),
            "genre_diversity": round(data["genre_diversity"], 3),
            "segments": {
                seg: {
                    "precision": round(v["precision@10"], 4),
                    "recall": round(v["recall@10"], 4),
                    "ndcg": round(v["ndcg@10"], 4),
                }
                for seg, v in data["segments"].items()
            }
        })

    # Feature importance (sorted)
    feature_importance = sorted(
        [{"feature": k, "importance": v}
         for k, v in state.comparison["feature_importance"].items()],
        key=lambda x: -x["importance"]
    )

    key_findings = [
        {
            "title": "Hybrid reranker beats CF by 23% on NDCG",
            "detail": (
                "The hybrid model achieves the best NDCG@10 (0.060), R@10 (0.075), and P@10 (0.037) "
                "across all four models — a 23% NDCG improvement over CF alone (0.048). "
                "By using CF to generate 50 candidates and then reranking with LightGBM on 7 "
                "engineered features, it surfaces more relevant items and orders them better "
                "than any single approach."
            ),
        },
        {
            "title": "Content-based model is near-useless standalone — but critical as a feature",
            "detail": (
                "Standalone, the content model scores NDCG=0.001 — essentially random. "
                "Sentence embeddings encode semantic similarity well, but user profiles built "
                "from a handful of liked movies are too noisy to produce reliable rankings "
                "across 62K items. However, content_score is the 3rd most important reranker "
                "feature (importance 995), meaning it adds signal when combined with CF candidates "
                "and genre overlap rather than running alone."
            ),
        },
        {
            "title": "CF score is the reranker's strongest signal",
            "detail": (
                "Despite being trained as a LambdaRank model over 7 features, the reranker "
                "assigned cf_score the highest importance (1198), confirming that ALS latent "
                "embeddings capture the most reliable preference signal. Genre overlap came "
                "second (1133), acting as an interpretable proxy for user taste. Together they "
                "account for 46% of total feature importance."
            ),
        },
        {
            "title": "Accuracy vs diversity tradeoff confirmed empirically",
            "detail": (
                "The hybrid reranker's NDCG gains come partly at the cost of diversity: "
                "popularity bias increased from 25.4% (CF) to 41.9% (hybrid). The reranker "
                "learned that avg_rating and pop_score are strong relevance predictors — "
                "which is statistically true — but this amplifies the tendency to over-recommend "
                "widely popular films at the expense of niche discoveries."
            ),
        },
        {
            "title": "71.5% failure rate — the honest number",
            "detail": (
                "In nearly three quarters of users, no model placed any relevant item in the "
                "top 10. The aggregate NDCG of 0.060 is pulled up by the minority of users "
                "where models work well. Root causes: small per-user test sets (3-6 items), "
                "strict ≥4.0 relevance threshold, and temporal drift — test interactions "
                "were held out chronologically, so user tastes at test time had shifted "
                "from the training signal."
            ),
        },
        {
            "title": "user_interaction_count as implicit confidence calibration",
            "detail": (
                "The reranker assigned 4th-highest importance to user_interaction_count. "
                "This means it learned to trust CF scores more for active users who have "
                "richer, better-trained ALS embeddings. For sparse users, the reranker "
                "downweights cf_score and leans more on genre overlap and global popularity "
                "signals — an emergent behavior from the training data, not an explicit rule."
            ),
        },
    ]

    tradeoffs = [
        {
            "axis_a": "Personalization",
            "axis_b": "Popularity bias",
            "observation": (
                "CF personalizes strongly (25.4% popularity bias vs 100% for the baseline) "
                "but the hybrid reranker's accuracy gains come partly from re-learning "
                "popularity patterns (41.9% bias). Achieving better NDCG without increasing "
                "popularity bias would require explicit de-biasing during reranker training."
            ),
        },
        {
            "axis_a": "Recall",
            "axis_b": "Precision",
            "observation": (
                "CF improves recall over popularity (0.064 vs 0.031) and also improves "
                "precision (0.030 vs 0.016). The hybrid reranker improves both further "
                "(P=0.037, R=0.075). This is unusually clean — the reranker found a way "
                "to improve ordering quality without the typical recall-precision tradeoff."
            ),
        },
        {
            "axis_a": "Model complexity",
            "axis_b": "Interpretability",
            "observation": (
                "The simplest model (popularity) is fully interpretable but useless "
                "for personalization. The hybrid reranker is still partially interpretable "
                "via feature importance — we can explain why a movie was ranked highly. "
                "Replacing LightGBM with a deep neural reranker would likely improve NDCG "
                "but would make per-recommendation explanations much harder."
            ),
        },
        {
            "axis_a": "Semantic similarity",
            "axis_b": "Collaborative signal",
            "observation": (
                "Content embeddings (all-MiniLM-L6-v2) capture plot and genre semantics "
                "but perform poorly standalone — too much noise across 62K items. "
                "ALS collaborative filtering exploits collective user behavior but can't "
                "generalize to items with no ratings. The hybrid gets the best of both: "
                "CF generates high-quality candidates, content scores refine them."
            ),
        },
    ]

    future_work = [
        "Session-based modeling: transformer over recent interaction sequences (BERT4Rec / SASRec)",
        "Neural two-tower reranker: replace LightGBM with a deep cross-attention model for user-item interactions",
        "Online learning: incremental ALS embedding updates from new interactions without full retrain",
        "Explicit diversity re-ranking: MMR or DPP post-processing to enforce genre variety in the final top-10",
        "De-biasing during reranker training: IPS weighting to correct for popularity-driven label noise",
        "A/B testing framework: compare models on live user traffic rather than offline held-out metrics",
    ]

    return {
        "comparison_table": table,
        "feature_importance": feature_importance,
        "key_findings": key_findings,
        "tradeoffs": tradeoffs,
        "future_work": future_work,
        "dataset_stats": state.comparison["dataset_stats"],
        "failure_rate": state.comparison["failure_rate"],
    }


@router.get("/item/{movie_id}")
def get_item(movie_id: int):
    """Return metadata for a single movie."""
    from backend.model_loader import state

    info = state.movie_lookup.get(movie_id)
    if not info:
        raise HTTPException(status_code=404, detail=f"Movie {movie_id} not found.")

    stats = state.item_stats.loc[movie_id] if movie_id in state.item_stats.index else None
    return {
        **info,
        "poster_url": state.posters.get(movie_id),
        "avg_rating": round(float(stats["avg_rating"]), 2) if stats is not None else None,
        "rating_count": int(stats["rating_count"]) if stats is not None else None,
    }


@router.get("/movies")
def list_movies(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    genre: Optional[str] = None,
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
):
    """
    Paginated movie list for the onboarding picker.
    Optionally filter by genre and/or year range. Ordered by popularity (most rated first).
    """
    from backend.model_loader import state

    pop_order = (
        state.item_stats["rating_count"]
        .sort_values(ascending=False)
        .index.tolist()
    )

    results = []
    for mid in pop_order:
        info = state.movie_lookup.get(mid)
        if not info:
            continue
        if genre and genre.lower() not in [g.lower() for g in info["genres"]]:
            continue
        year = info.get("year")
        if year_min is not None and (year is None or year < year_min):
            continue
        if year_max is not None and (year is None or year > year_max):
            continue
        stats = state.item_stats.loc[mid]
        results.append({
            **info,
            "poster_url": state.posters.get(mid),
            "avg_rating": round(float(stats["avg_rating"]), 2),
            "rating_count": int(stats["rating_count"]),
        })

    start = (page - 1) * page_size
    return {
        "total": len(results),
        "page": page,
        "page_size": page_size,
        "movies": results[start : start + page_size],
    }
