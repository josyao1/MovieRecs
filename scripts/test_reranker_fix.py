"""
Quick validation: compare old (union candidates) vs new (CF-only candidates)
reranker training on a small subset of users — no re-encoding required.

Uses existing content_model.pkl and cf_model.pkl.

Run: USE_TF=0 python scripts/test_reranker_fix.py
Expected: new NDCG >= CF NDCG (0.0484)
"""
import pickle, sys, math
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

CANDIDATE_K   = 50
POSITIVE_THRESHOLD = 3.5
N_USERS       = 500   # small — fast test
N_ESTIMATORS  = 100

print("Loading models and data...")
with open(ROOT / "artifacts/models/cf_model.pkl", "rb") as f:
    cf = pickle.load(f)
with open(ROOT / "artifacts/models/content_model.pkl", "rb") as f:
    cb = pickle.load(f)

train = pd.read_parquet(ROOT / "data/processed/train.parquet")
val   = pd.read_parquet(ROOT / "data/processed/val.parquet")
test  = pd.read_parquet(ROOT / "data/processed/test.parquet")
movies = pd.read_csv(ROOT / "data/processed/movies.csv")

# Patch back _user_seen (stripped by slim_models)
user_seen = train.groupby("user_id")["movie_id"].apply(set).to_dict()
cf._user_seen = user_seen
cb._user_seen = user_seen

item_stats = (
    train.groupby("movie_id")["rating"]
    .agg(avg_rating="mean", rating_count="count")
    .assign(pop_score=lambda d: d["rating_count"] * d["avg_rating"])
)
user_stats = train.groupby("user_id")["movie_id"].count().to_dict()
genre_lookup = {
    row.movie_id: set(row.genres) if isinstance(row.genres, list) else set()
    for row in movies.itertuples()
}

val_positive = set(zip(
    val[val["rating"] >= POSITIVE_THRESHOLD]["user_id"],
    val[val["rating"] >= POSITIVE_THRESHOLD]["movie_id"]
))
test_lookup = (
    test[test["rating"] >= 4.0]
    .groupby("user_id")["movie_id"].apply(set).to_dict()
)

sample_users = [u for u in train["user_id"].unique()[:N_USERS] if u in cf._user_index]


def build_feature_row(user_id, movie_id, cf_score, cb_score, genre_profile):
    stats = item_stats.loc[movie_id] if movie_id in item_stats.index else None
    item_genres = genre_lookup.get(movie_id, set())
    return {
        "cf_score": cf_score,
        "content_score": cb_score,
        "pop_score": float(stats["pop_score"]) if stats is not None else 0.0,
        "avg_rating": float(stats["avg_rating"]) if stats is not None else 0.0,
        "rating_count": float(np.log1p(stats["rating_count"])) if stats is not None else 0.0,
        "genre_overlap": sum(genre_profile.get(g, 0.0) for g in item_genres),
        "user_interaction_count": float(np.log1p(user_stats.get(user_id, 0))),
    }


def ndcg_at_k(recs, relevant, k=10):
    top_k = recs[:k]
    dcg = sum(1.0 / math.log2(i + 2) for i, m in enumerate(top_k) if m in relevant)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal if ideal > 0 else 0.0


def train_and_eval(use_cf_only: bool, label: str):
    X_rows, y_rows, groups = [], [], []

    for user_id in sample_users:
        cf_recs = dict(cf.recommend(user_id, CANDIDATE_K))
        if not cf_recs:
            continue

        if use_cf_only:
            candidates = list(cf_recs.keys())
            cb_recs = cb.score_items(user_id, candidates)
        else:
            cb_recs_list = dict(cb.recommend(user_id, CANDIDATE_K))
            candidates = list(set(cf_recs) | set(cb_recs_list))
            cb_recs = cb_recs_list

        liked = [m for m, _ in train[train["user_id"] == user_id][["movie_id","rating"]].itertuples(index=False) if _ >= POSITIVE_THRESHOLD]
        genres = [g for m in liked for g in genre_lookup.get(m, set())]
        genre_profile = pd.Series(genres).value_counts(normalize=True).to_dict() if genres else {}

        features = [build_feature_row(user_id, m, cf_recs.get(m, 0.0), cb_recs.get(m, 0.0), genre_profile) for m in candidates]
        labels   = [1 if (user_id, m) in val_positive else 0 for m in candidates]

        X_rows.append(pd.DataFrame(features))
        y_rows.extend(labels)
        groups.append(len(candidates))

    X = pd.concat(X_rows, ignore_index=True)
    y = np.array(y_rows)

    ranker = lgb.LGBMRanker(objective="lambdarank", n_estimators=N_ESTIMATORS,
                             learning_rate=0.05, num_leaves=31, verbose=-1)
    ranker.fit(X, y, group=groups)

    # Eval on test set
    eval_users = [u for u in list(test_lookup.keys())[:1000] if u in cf._user_index]
    ndcgs = []
    for uid in eval_users:
        cf_r = dict(cf.recommend(uid, CANDIDATE_K))
        if not cf_r:
            continue
        if use_cf_only:
            cands = list(cf_r.keys())
            cb_r = cb.score_items(uid, cands)
        else:
            cb_r_list = dict(cb.recommend(uid, CANDIDATE_K))
            cands = list(set(cf_r) | set(cb_r_list))
            cb_r = cb_r_list

        gp = cb._user_genre_profiles.get(uid, {})
        feats = [build_feature_row(uid, m, cf_r.get(m,0.0), cb_r.get(m,0.0), gp) for m in cands]
        scores = ranker.predict(pd.DataFrame(feats))
        ranked = [m for m, _ in sorted(zip(cands, scores), key=lambda x: -x[1])]
        if uid in test_lookup:
            ndcgs.append(ndcg_at_k(ranked, test_lookup[uid]))

    result = float(np.mean(ndcgs))
    print(f"[{label}] NDCG@10: {result:.4f}  (n={len(ndcgs)} users)")
    return result


print(f"\nTraining on {len(sample_users)} users, evaluating on up to 1000 test users\n")
print("CF baseline (no reranker): 0.0484")
print()

old_ndcg = train_and_eval(use_cf_only=False, label="OLD: CF+content candidates")
new_ndcg = train_and_eval(use_cf_only=True,  label="NEW: CF-only candidates   ")

print(f"\nImprovement: {new_ndcg - old_ndcg:+.4f}")
print("Fix is working ✓" if new_ndcg > old_ndcg else "No improvement — investigate further")
