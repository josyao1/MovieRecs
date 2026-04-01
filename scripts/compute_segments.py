"""
Compute cold/warm/hot segment metrics for all models and patch comparison.json.
Uses existing trained model artifacts — no retraining required.

Cold:  < 20 training ratings
Warm:  20–99 training ratings
Hot:   >= 100 training ratings

Run: python scripts/compute_segments.py
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

EVAL_SAMPLE = 5_000

print("Loading data...")
train = pd.read_parquet(ROOT / "data/processed/train.parquet")
test  = pd.read_parquet(ROOT / "data/processed/test.parquet")

user_train_counts = train.groupby("user_id")["movie_id"].count().to_dict()
user_seen = train.groupby("user_id")["movie_id"].apply(set).to_dict()

from src.evaluation.metrics import evaluate_model, build_test_lookup

test_lookup = build_test_lookup(test, min_rating=4.0)

rng = np.random.default_rng(42)
all_test_users = list(test_lookup.keys())
if len(all_test_users) > EVAL_SAMPLE:
    eval_users = set(rng.choice(all_test_users, EVAL_SAMPLE, replace=False).tolist())
else:
    eval_users = set(all_test_users)
eval_lookup = {uid: test_lookup[uid] for uid in eval_users}

print(f"Evaluating on {len(eval_lookup):,} users with segment breakdown")
print(f"  cold (<20):  {sum(1 for u in eval_lookup if user_train_counts.get(u,0) < 20):,}")
print(f"  warm (20-99):{sum(1 for u in eval_lookup if 20 <= user_train_counts.get(u,0) < 100):,}")
print(f"  hot (>=100): {sum(1 for u in eval_lookup if user_train_counts.get(u,0) >= 100):,}")

# Load models
print("\nLoading models...")
with open(ROOT / "artifacts/models/popularity_model.pkl", "rb") as f:
    pop = pickle.load(f)
with open(ROOT / "artifacts/models/cf_model.pkl", "rb") as f:
    cf = pickle.load(f)
with open(ROOT / "artifacts/models/content_model.pkl", "rb") as f:
    cb = pickle.load(f)
with open(ROOT / "artifacts/models/reranker.pkl", "rb") as f:
    rd = pickle.load(f)

pop._user_seen = user_seen
cf._user_seen  = user_seen
cb._user_seen  = user_seen

ranker = rd["ranker"]

# Load reranker features
item_stats = pd.read_parquet(ROOT / "artifacts/precomputed/item_stats.parquet").set_index("movie_id")
movies = pd.read_csv(ROOT / "data/processed/movies.csv")
import ast
movies["genres"] = movies["genres"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
genre_lookup = {
    int(r.movie_id): set(r.genres) if isinstance(r.genres, list) else set()
    for r in movies.itertuples()
}

CANDIDATE_K = 50

def build_feature_row(mid, cf_score, cb_score, genre_profile):
    stats = item_stats.loc[mid] if mid in item_stats.index else None
    return {
        "cf_score":               cf_score,
        "content_score":          cb_score,
        "pop_score":              float(stats["pop_score"])    if stats is not None else 0.0,
        "avg_rating":             float(stats["avg_rating"])   if stats is not None else 0.0,
        "rating_count":           float(np.log1p(stats["rating_count"])) if stats is not None else 0.0,
        "genre_overlap":          sum(genre_profile.get(g, 0.0) for g in genre_lookup.get(mid, set())),
        "user_interaction_count": float(np.log1p(user_train_counts.get(mid, 0))),
    }

def get_pop_recs(uid):
    return [m for m, _ in pop.recommend(uid, 10)]

def get_cf_recs(uid):
    try:
        return [m for m, _ in cf.recommend(uid, 10)]
    except Exception:
        return []

def get_cb_recs(uid):
    try:
        return [m for m, _ in cb.recommend(uid, 10)]
    except Exception:
        return []

def get_hybrid_recs(uid):
    try:
        cf_r = dict(cf.recommend(uid, CANDIDATE_K))
        if not cf_r:
            return get_pop_recs(uid)
        candidates = list(cf_r.keys())
        cb_r = cb.score_items(uid, candidates)
        genre_profile = cb._user_genre_profiles.get(uid, {})
        rows = [build_feature_row(mid, cf_r.get(mid, 0.0), cb_r.get(mid, 0.0), genre_profile) for mid in candidates]
        scores = ranker.predict(pd.DataFrame(rows))
        ranked = sorted(zip(candidates, scores), key=lambda x: -x[1])
        return [int(m) for m, _ in ranked[:10]]
    except Exception:
        return []

print("\nBuilding recommendations...")
_pop_list = get_pop_recs(next(iter(eval_lookup)))
pop_recs  = {uid: _pop_list for uid in eval_lookup}

cf_recs = {}
cb_recs = {}
hybrid_recs = {}

for i, uid in enumerate(eval_lookup):
    if i % 500 == 0:
        print(f"  {i}/{len(eval_lookup)}")
    cf_recs[uid]     = get_cf_recs(uid)
    cb_recs[uid]     = get_cb_recs(uid)
    hybrid_recs[uid] = get_hybrid_recs(uid)

print("\nEvaluating with segments...")
results = {
    "popularity":           evaluate_model(pop_recs,    eval_lookup, k=10, user_train_counts=user_train_counts),
    "collaborative_filter": evaluate_model(cf_recs,     eval_lookup, k=10, user_train_counts=user_train_counts),
    "content_based":        evaluate_model(cb_recs,     eval_lookup, k=10, user_train_counts=user_train_counts),
    "hybrid_reranker":      evaluate_model(hybrid_recs, eval_lookup, k=10, user_train_counts=user_train_counts),
}

for name, r in results.items():
    print(f"\n{name}:")
    print(f"  NDCG@10: {r['ndcg@10']:.4f}")
    for seg, sv in r.get("segments", {}).items():
        print(f"  {seg:4s} (n={sv['n_users']:4d}): NDCG={sv['ndcg@10']:.4f}")

# Patch comparison.json
comp_path = ROOT / "artifacts/metrics/comparison.json"
with open(comp_path) as f:
    comp = json.load(f)

for model_key, r in results.items():
    segs = r.get("segments", {})
    comp["models"][model_key]["segments"] = segs
    comp["models"][model_key]["overall"]["precision@10"] = round(r["precision@10"], 4)
    comp["models"][model_key]["overall"]["recall@10"]    = round(r["recall@10"], 4)
    comp["models"][model_key]["overall"]["ndcg@10"]      = round(r["ndcg@10"], 4)

with open(comp_path, "w") as f:
    json.dump(comp, f, indent=2)

print(f"\nPatched {comp_path}")
