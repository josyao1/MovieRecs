"""
Full retrain pipeline for ML-25M (or whichever dataset is in data/raw/).

Runs all phases in order:
  1. Preprocess + split
  2. Popularity baseline
  3. Collaborative filtering (ALS)
  4. Content-based
  5. Hybrid reranker
  6. Evaluation + save comparison.json

Run from project root:
    python scripts/retrain.py

Estimated time on a modern laptop:
  - Preprocessing:  ~1 min
  - CF (ALS):       ~10-15 min
  - Content:        ~1 min
  - Reranker:       ~5-10 min
  - Evaluation:     ~5 min  (sampled at 5,000 users)
  Total:            ~25-35 min
"""
from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parents[1]
ARTIFACTS = ROOT / "artifacts"
import sys; sys.path.insert(0, str(ROOT))

# Cap evaluation at this many users to keep it tractable on 25M-scale data
EVAL_SAMPLE = 5_000


def timed(label: str):
    class _ctx:
        def __enter__(self):
            print(f"\n{'─'*50}")
            print(f"▶  {label}")
            self._t = time.time()
        def __exit__(self, *_):
            print(f"✓  {label} — {time.time()-self._t:.0f}s")
    return _ctx()


def main():
    from src.preprocessing.data_loader import load_ratings, load_movies
    from src.preprocessing.splitter import split, save_splits
    from src.models.popularity import PopularityModel
    from src.models.collaborative_filter import CollaborativeFilterModel
    from src.models.content_based import ContentBasedModel
    from src.models.hybrid_reranker import HybridReranker
    from src.evaluation.metrics import evaluate_model, build_test_lookup

    # ── 1. Load & split ──────────────────────────────────────────────────────
    with timed("Load data"):
        ratings = load_ratings()
        movies  = load_movies()
        print(f"   {len(ratings):,} ratings | {len(movies):,} movies")

    with timed("Train/val/test split"):
        train, val, test, cold_users = split(ratings)
        save_splits(train, val, test, cold_users)

    from src.preprocessing.data_loader import save_processed
    save_processed(ratings, movies)

    test_lookup = build_test_lookup(test, min_rating=4.0)
    val_lookup  = build_test_lookup(val,  min_rating=4.0)

    # Sample a fixed subset for evaluation (full test set is 100K+ users)
    rng = np.random.default_rng(42)
    all_test_users = list(test_lookup.keys())
    if len(all_test_users) > EVAL_SAMPLE:
        eval_users = set(rng.choice(all_test_users, EVAL_SAMPLE, replace=False).tolist())
        print(f"\n   Evaluation capped at {EVAL_SAMPLE:,} / {len(all_test_users):,} test users")
    else:
        eval_users = set(all_test_users)
    eval_lookup = {uid: test_lookup[uid] for uid in eval_users}

    results = {}

    # ── 2. Popularity ─────────────────────────────────────────────────────────
    with timed("Popularity baseline"):
        pop = PopularityModel(score_mode="weighted")
        pop.fit(train)
        # Popularity returns the same list for every user — compute once
        _pop_list = [m for m, _ in pop.recommend(next(iter(eval_lookup)), 10)]
        pop_recs = {uid: _pop_list for uid in eval_lookup}
        results["popularity"] = evaluate_model(pop_recs, eval_lookup, k=10)
        print(f"   NDCG@10: {results['popularity']['ndcg@10']:.4f}")

    (ARTIFACTS / "models").mkdir(parents=True, exist_ok=True)
    with open(ARTIFACTS / "models" / "popularity_model.pkl", "wb") as f:
        pickle.dump(pop, f)

    # ── 3. Collaborative filtering ────────────────────────────────────────────
    with timed("Collaborative filtering (ALS)"):
        cf = CollaborativeFilterModel(factors=64, iterations=20, regularization=0.01)
        cf.fit(train)
        cf_recs = {}
        for uid in eval_lookup:
            try:
                cf_recs[uid] = [m for m, _ in cf.recommend(uid, 10)]
            except Exception:
                cf_recs[uid] = []
        results["collaborative_filter"] = evaluate_model(cf_recs, eval_lookup, k=10)
        print(f"   NDCG@10: {results['collaborative_filter']['ndcg@10']:.4f}")

    with open(ARTIFACTS / "models" / "cf_model.pkl", "wb") as f:
        pickle.dump(cf, f)

    # ── 4. Content-based ──────────────────────────────────────────────────────
    with timed("Content-based"):
        cb = ContentBasedModel()
        cb.fit(train, movies)
        cb_recs = {}
        for uid in eval_lookup:
            try:
                cb_recs[uid] = [m for m, _ in cb.recommend(uid, 10)]
            except Exception:
                cb_recs[uid] = []
        results["content_based"] = evaluate_model(cb_recs, eval_lookup, k=10)
        print(f"   NDCG@10: {results['content_based']['ndcg@10']:.4f}")

    with open(ARTIFACTS / "models" / "content_model.pkl", "wb") as f:
        pickle.dump(cb, f)

    # ── 5. Hybrid reranker ────────────────────────────────────────────────────
    with timed("Hybrid reranker (LightGBM LambdaRank)"):
        reranker = HybridReranker(n_estimators=200, learning_rate=0.05)
        reranker.set_component_models(cf, cb, pop)
        reranker.fit(train, val, movies)

        hybrid_recs = {}
        for uid in eval_lookup:
            try:
                hybrid_recs[uid] = [m for m, _ in reranker.recommend(uid, 10)]
            except Exception:
                hybrid_recs[uid] = []
        results["hybrid_reranker"] = evaluate_model(hybrid_recs, eval_lookup, k=10)
        print(f"   NDCG@10: {results['hybrid_reranker']['ndcg@10']:.4f}")

    with open(ARTIFACTS / "models" / "reranker.pkl", "wb") as f:
        pickle.dump({"ranker": reranker._ranker, "importances": {
            row.feature: int(row.importance)
            for row in reranker.feature_importance.itertuples()
        }}, f)

    # ── 6. Save comparison metrics ────────────────────────────────────────────
    with timed("Save metrics"):
        # Popularity bias = fraction of recs that are top-100 popular items
        top100 = set(
            train.groupby("movie_id")["rating"].count()
            .sort_values(ascending=False).head(100).index
        )

        def pop_bias(recs):
            all_recs = [m for lst in recs.values() for m in lst]
            return round(sum(1 for m in all_recs if m in top100) / max(len(all_recs), 1), 3)

        def genre_div(recs, movie_genres):
            genres_seen = set()
            total = 0
            for lst in recs.values():
                for m in lst:
                    genres_seen.update(movie_genres.get(m, []))
                    total += 1
            return round(len(genres_seen) / max(total, 1), 3)

        movie_genres = {int(r.movie_id): r.genres for r in movies.itertuples()}

        # Failure rate
        hybrid_hits = sum(
            1 for uid in eval_lookup
            if any(m in eval_lookup[uid] for m in hybrid_recs.get(uid, []))
        )
        failure_rate = round(1 - hybrid_hits / max(len(eval_lookup), 1), 3)

        recs_by_model = {
            "popularity":            pop_recs,
            "collaborative_filter":  cf_recs,
            "content_based":         cb_recs,
            "hybrid_reranker":       hybrid_recs,
        }

        comparison = {
            "models": {
                name: {
                    "overall": res,           # flat dict: precision@10, recall@10, ndcg@10
                    "segments": res.get("segments", {}),
                    "popularity_bias": pop_bias(recs_by_model[name]),
                    "genre_diversity": genre_div(recs_by_model[name], movie_genres),
                }
                for name, res in results.items()
            },
            "feature_importance": {
                row.feature: int(row.importance)
                for row in reranker.feature_importance.itertuples()
            },
            "dataset_stats": {
                "n_users":         int(ratings["user_id"].nunique()),
                "n_movies":        int(movies["movie_id"].nunique()),
                "n_ratings_train": int(len(train)),
                "sparsity":        round(1 - len(train) / (
                    ratings["user_id"].nunique() * movies["movie_id"].nunique()
                ), 4),
            },
            "failure_rate": failure_rate,
        }

        (ARTIFACTS / "metrics").mkdir(parents=True, exist_ok=True)
        with open(ARTIFACTS / "metrics" / "comparison.json", "w") as f:
            json.dump(comparison, f, indent=2)
        print(f"   Saved to artifacts/metrics/comparison.json")

    print(f"\n{'═'*50}")
    print("ALL DONE — restart backend to load new models")
    print("  uvicorn backend.main:app --reload --port 8000")
    print(f"{'═'*50}")


if __name__ == "__main__":
    main()
