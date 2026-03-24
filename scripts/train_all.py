"""
End-to-end training script.

Runs the full pipeline in order:
  1. Load and clean raw data
  2. Time-aware train/val/test split
  3. Train all models
  4. Evaluate and save comparison metrics

Usage:
    python scripts/train_all.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.preprocessing.data_loader import load_ratings, load_movies, save_processed
from src.preprocessing.splitter import split, save_splits
from src.models.popularity import PopularityModel
from src.models.collaborative_filter import CollaborativeFilterModel
from src.models.content_based import ContentBasedModel
from src.models.hybrid_reranker import HybridReranker
from src.evaluation.metrics import evaluate_model, build_test_lookup

ARTIFACTS_DIR = Path(__file__).parents[1] / "artifacts"
METRICS_DIR = ARTIFACTS_DIR / "metrics"
EMBEDDINGS_DIR = ARTIFACTS_DIR / "embeddings"

import numpy as np


def main():
    print("=== Step 1: Load and clean data ===")
    ratings = load_ratings()
    movies = load_movies()
    save_processed(ratings, movies)
    print(f"Ratings: {len(ratings):,}  Movies: {len(movies):,}")

    print("\n=== Step 2: Train/val/test split ===")
    train, val, test, cold_users = split(ratings)
    save_splits(train, val, test, cold_users)

    # Build test lookup for evaluation
    test_lookup = build_test_lookup(test)
    val_lookup = build_test_lookup(val)
    eval_users = list(test_lookup.keys())
    print(f"Evaluating on {len(eval_users):,} users with test interactions")

    comparison = {}

    print("\n=== Step 3: Popularity Baseline ===")
    pop = PopularityModel(score_mode="weighted")
    pop.fit(train)
    pop_recs = pop.recommend_batch(eval_users, top_k=10)
    pop_recs_list = {u: [m for m, _ in recs] for u, recs in pop_recs.items()}
    pop_metrics = evaluate_model(pop_recs_list, test_lookup, k=10)
    print("Popularity:", pop_metrics)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (METRICS_DIR / "popularity.json").write_text(json.dumps(pop_metrics, indent=2))
    comparison["popularity"] = pop_metrics

    print("\n=== Step 4: Collaborative Filtering ===")
    cf = CollaborativeFilterModel(factors=64, iterations=20)
    cf.fit(train)
    cf_recs = {u: [m for m, _ in cf.recommend(u, 10)] for u in eval_users}
    cf_metrics = evaluate_model(cf_recs, test_lookup, k=10)
    print("CF:", cf_metrics)
    (METRICS_DIR / "cf.json").write_text(json.dumps(cf_metrics, indent=2))
    comparison["collaborative_filtering"] = cf_metrics
    cf.save()

    # Save embeddings
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_DIR / "user_embeddings.npy", cf.user_embeddings)
    np.save(EMBEDDINGS_DIR / "item_embeddings.npy", cf.item_embeddings)

    print("\n=== Step 5: Content-Based ===")
    cb = ContentBasedModel()
    cb.fit(train, movies)
    cb_recs = {u: [m for m, _ in cb.recommend(u, 10)] for u in eval_users}
    cb_metrics = evaluate_model(cb_recs, test_lookup, k=10)
    print("Content-based:", cb_metrics)
    (METRICS_DIR / "content.json").write_text(json.dumps(cb_metrics, indent=2))
    comparison["content_based"] = cb_metrics
    cb.save()

    print("\n=== Step 6: Hybrid Reranker ===")
    reranker = HybridReranker(n_estimators=200)
    reranker.set_component_models(cf=cf, cb=cb, pop=pop)
    reranker.fit(train, val, movies)
    hybrid_recs = {u: [m for m, _ in reranker.recommend(u, 10)] for u in eval_users}
    hybrid_metrics = evaluate_model(hybrid_recs, test_lookup, k=10)
    print("Hybrid:", hybrid_metrics)
    (METRICS_DIR / "hybrid.json").write_text(json.dumps(hybrid_metrics, indent=2))
    comparison["hybrid_reranker"] = hybrid_metrics
    reranker.save()

    print("\n=== Step 7: Save comparison ===")
    (METRICS_DIR / "comparison.json").write_text(json.dumps(comparison, indent=2))
    print("Saved comparison.json")

    print("\n=== Summary ===")
    for model, metrics in comparison.items():
        print(f"  {model:<25} P@10={metrics['precision@10']:.4f}  R@10={metrics['recall@10']:.4f}  NDCG@10={metrics['ndcg@10']:.4f}")


if __name__ == "__main__":
    main()
