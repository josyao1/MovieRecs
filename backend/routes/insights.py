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
            "title": "Hybrid reranker wins on recall and NDCG",
            "detail": (
                "The hybrid model achieves the best R@10 (0.062) and NDCG@10 (0.054) "
                "across all four models. By combining CF candidate generation with a "
                "LightGBM reranker trained on engineered features, it surfaces more "
                "relevant items and ranks them better than any single model alone."
            ),
        },
        {
            "title": "Popularity baseline is harder to beat than expected",
            "detail": (
                "The simple popularity baseline (count × avg_rating) achieved P@10=0.041 "
                "and NDCG=0.052 — outperforming CF on both metrics. This is a well-known "
                "phenomenon: popular items have broad appeal, so recommending them is a "
                "safe strategy even without any personalization."
            ),
        },
        {
            "title": "Content-based model failed standalone — but contributed as a feature",
            "detail": (
                "Genre-only content features produced the weakest standalone model (NDCG=0.013). "
                "18 binary genre dimensions cannot distinguish users who like the same genres "
                "but with different tastes within them. However, the genre_overlap feature "
                "derived from the same content model was the single most important feature "
                "in the hybrid reranker — showing that weak standalone models can still "
                "contribute valuable signals to ensembles."
            ),
        },
        {
            "title": "Accuracy vs diversity tradeoff confirmed empirically",
            "detail": (
                "The hybrid reranker achieved better accuracy metrics (NDCG) than CF, "
                "but increased popularity bias from 32.2% to 49.7%. The reranker learned "
                "that avg_rating and pop_score are strong predictors of relevance — "
                "which is correct — but this amplifies the tendency to over-recommend "
                "widely popular films at the expense of niche discoveries."
            ),
        },
        {
            "title": "49% failure rate — the honest number",
            "detail": (
                "In nearly half of users, no model placed any relevant item in the top 10. "
                "The aggregate NDCG of 0.054 is pulled up by users where models work well. "
                "Root causes: small test sets (3-6 items per user), strict ≥4.0 relevance "
                "threshold, and temporal drift — user tastes at test time had shifted from "
                "the training signal."
            ),
        },
        {
            "title": "user_interaction_count as implicit calibration",
            "detail": (
                "The reranker assigned the 3rd highest importance to user_interaction_count. "
                "This means it learned that CF scores are more trustworthy for active users "
                "who have richer, better-trained embeddings. For sparse users, the reranker "
                "relies more on genre overlap and popularity signals instead."
            ),
        },
    ]

    tradeoffs = [
        {
            "axis_a": "Personalization",
            "axis_b": "Popularity bias",
            "observation": (
                "CF personalizes more than popularity (32.2% vs 99.9% popularity bias) "
                "but the hybrid reranker's accuracy gains come partly from re-learning "
                "popularity patterns. Better personalization may require explicit "
                "de-biasing techniques during training."
            ),
        },
        {
            "axis_a": "Recall",
            "axis_b": "Precision",
            "observation": (
                "CF improves recall over popularity (0.052 vs 0.045) but loses on "
                "precision (0.029 vs 0.041). The hybrid reranker partially reconciles "
                "this: it retains CF's broad recall while improving ordering quality. "
                "But some precision is still traded for recall."
            ),
        },
        {
            "axis_a": "Model complexity",
            "axis_b": "Interpretability",
            "observation": (
                "The simplest model (popularity) is fully interpretable but useless "
                "for personalization. The most complex model (hybrid reranker) is still "
                "partially interpretable via feature importance. Deeper neural rerankers "
                "would trade away that interpretability."
            ),
        },
    ]

    future_work = [
        "Richer content features: sentence embeddings from movie descriptions (e.g. all-MiniLM-L6-v2)",
        "Session-based modeling: transformer over recent interaction sequences (BERT4Rec)",
        "Neural two-tower reranker: replace LightGBM with a deep model for user-item interactions",
        "Online learning: incremental embedding updates from new interactions without full retrain",
        "Explicit diversity constraints: enforce genre diversity in final ranked list",
        "A/B testing framework: compare models on live traffic rather than offline metrics",
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
