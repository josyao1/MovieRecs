"""
Ranking evaluation metrics: Precision@K, Recall@K, NDCG@K.

All functions operate on lists/sets and are averaged across users.
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np


def precision_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    top_k = recommended[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / k


def recall_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    top_k = recommended[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """NDCG@K with binary relevance."""
    top_k = recommended[:k]
    dcg = sum(
        1.0 / math.log2(i + 2)
        for i, item in enumerate(top_k)
        if item in relevant
    )
    # Ideal DCG: all relevant items placed at top
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_model(
    recommendations: dict[int, list[int]],
    test_interactions: dict[int, set[int]],
    k: int = 10,
    user_train_counts: dict[int, int] | None = None,
) -> dict[str, float]:
    """Evaluate a model's recommendations against test interactions.

    Args:
        recommendations: {user_id: [movie_id, ...]} sorted by predicted relevance
        test_interactions: {user_id: {movie_id, ...}} ground truth
        k: cutoff
        user_train_counts: optional {user_id: n_train_ratings} for cold/warm/hot breakdown

    Returns:
        dict with precision@k, recall@k, ndcg@k (macro-averaged over users),
        and optionally a 'segments' key with cold/warm/hot breakdowns.
    """
    precisions, recalls, ndcgs = [], [], []
    seg_buckets: dict[str, dict[str, list]] = {
        "cold": {"p": [], "r": [], "n": []},
        "warm": {"p": [], "r": [], "n": []},
        "hot":  {"p": [], "r": [], "n": []},
    }

    for user_id, relevant in test_interactions.items():
        if user_id not in recommendations:
            continue
        recs = recommendations[user_id]
        p = precision_at_k(recs, relevant, k)
        r = recall_at_k(recs, relevant, k)
        n = ndcg_at_k(recs, relevant, k)
        precisions.append(p)
        recalls.append(r)
        ndcgs.append(n)

        if user_train_counts is not None:
            count = user_train_counts.get(user_id, 0)
            seg = "cold" if count < 20 else ("warm" if count < 100 else "hot")
            seg_buckets[seg]["p"].append(p)
            seg_buckets[seg]["r"].append(r)
            seg_buckets[seg]["n"].append(n)

    result: dict = {
        f"precision@{k}": float(np.mean(precisions)),
        f"recall@{k}":    float(np.mean(recalls)),
        f"ndcg@{k}":      float(np.mean(ndcgs)),
        "n_users_evaluated": len(precisions),
    }

    if user_train_counts is not None:
        result["segments"] = {
            seg: {
                f"precision@{k}": float(np.mean(v["p"])) if v["p"] else 0.0,
                f"recall@{k}":    float(np.mean(v["r"])) if v["r"] else 0.0,
                f"ndcg@{k}":      float(np.mean(v["n"])) if v["n"] else 0.0,
                "n_users":        len(v["p"]),
            }
            for seg, v in seg_buckets.items()
        }

    return result


def build_test_lookup(test_df, min_rating: float = 4.0) -> dict[int, set[int]]:
    """Convert test DataFrame to {user_id: set of relevant movie_ids}.

    Items with rating >= min_rating are considered relevant.
    """
    lookup: dict[int, set[int]] = defaultdict(set)
    for row in test_df.itertuples():
        if row.rating >= min_rating:
            lookup[row.user_id].add(row.movie_id)
    return dict(lookup)
