"""
Smoke test: simulate a session user who rates 5 Marvel/Star Wars movies highly,
then check what gets recommended. Run after retrain to validate real-world quality.

Run: USE_TF=0 python scripts/smoke_test.py
"""
import pickle, sys, numpy as np, pandas as pd
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

# --- Load models ---
print("Loading models...")
with open(ROOT / "artifacts/models/cf_model.pkl", "rb") as f:
    cf = pickle.load(f)
with open(ROOT / "artifacts/models/content_model.pkl", "rb") as f:
    cb = pickle.load(f)
with open(ROOT / "artifacts/models/reranker.pkl", "rb") as f:
    rd = pickle.load(f)
with open(ROOT / "artifacts/models/popularity_model.pkl", "rb") as f:
    pop = pickle.load(f)

ranker = rd["ranker"]
movies  = pd.read_csv(ROOT / "data/processed/movies.csv")
import ast
movies["genres"] = movies["genres"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
movie_lookup = {int(r.movie_id): r for r in movies.itertuples()}

import pandas as pd, json, numpy as np
import json
with open(ROOT / "artifacts/precomputed/user_train_count.json") as f:
    user_train_count = {int(k): v for k, v in json.load(f).items()}
item_stats = pd.read_parquet(ROOT / "artifacts/precomputed/item_stats.parquet").set_index("movie_id")
genre_lookup = {
    int(r.movie_id): set(r.genres) if isinstance(r.genres, list) else set()
    for r in movies.itertuples()
}

# Patch _user_seen (stripped by slim_models)
train = pd.read_parquet(ROOT / "data/processed/train.parquet")
user_seen = train.groupby("user_id")["movie_id"].apply(set).to_dict()
cf._user_seen  = user_seen
cb._user_seen  = user_seen
pop._user_seen = user_seen

CANDIDATE_K = 50

# --- Session ratings: 5 Marvel/Star Wars movies rated 5 stars ---
SESSION_RATINGS = [
    (260,   5.0),   # Star Wars: Episode IV - A New Hope (1977)
    (1196,  5.0),   # Star Wars: Episode V - The Empire Strikes Back (1980)
    (89745, 5.0),   # Avengers (2012)
    (59315, 5.0),   # Iron Man (2008)
    (86332, 5.0),   # Thor (2011)
]

print("\n=== Rated 5 stars ===")
for mid, r in SESSION_RATINGS:
    info = movie_lookup.get(mid)
    print(f"  ★ {info.title} ({', '.join(info.genres[:2])})")

# --- Build session profile ---
seen = {mid for mid, _ in SESSION_RATINGS}
liked_indices = [cb._item_index[mid] for mid, r in SESSION_RATINGS
                 if r >= 4.0 and mid in cb._item_index]

session_vec = cb._item_vectors[liked_indices].mean(axis=0)
norm = np.linalg.norm(session_vec)
if norm > 0:
    session_vec = session_vec / norm

# Content scores
raw_scores = cb._item_vectors.dot(session_vec)
cb_scores = {cb._index_item[i]: float(raw_scores[i])
             for i in range(len(raw_scores))
             if cb._index_item.get(i) not in seen}

top_cb = sorted(cb_scores.items(), key=lambda x: -x[1])[:CANDIDATE_K]

# Pop fallback
pop_fallback = [(mid, score) for mid, score in pop._item_scores.items()
                if mid not in seen and mid not in dict(top_cb)][:CANDIDATE_K]

candidates = dict(top_cb + pop_fallback)

# Genre profile
all_genres = [g for mid, _ in SESSION_RATINGS for g in genre_lookup.get(mid, set())]
genre_profile = pd.Series(all_genres).value_counts(normalize=True).to_dict() if all_genres else {}

# Build features
def build_row(mid, cb_score):
    stats = item_stats.loc[mid] if mid in item_stats.index else None
    return {
        "cf_score": 0.0,
        "content_score": cb_score,
        "pop_score": float(stats["pop_score"]) if stats is not None else 0.0,
        "avg_rating": float(stats["avg_rating"]) if stats is not None else 0.0,
        "rating_count": float(np.log1p(stats["rating_count"])) if stats is not None else 0.0,
        "genre_overlap": sum(genre_profile.get(g, 0.0) for g in genre_lookup.get(mid, set())),
        "user_interaction_count": 0.0,
    }

rows   = [build_row(mid, candidates[mid]) for mid in candidates]
scores = ranker.predict(pd.DataFrame(rows))
ranked = sorted(zip(candidates.keys(), scores), key=lambda x: -x[1])[:15]

print("\n=== Top 15 Recommendations ===")
for i, (mid, score) in enumerate(ranked, 1):
    info = movie_lookup.get(mid)
    if info:
        genres = ', '.join(info.genres[:3]) if isinstance(info.genres, list) else ''
        print(f"  {i:2}. {info.title:<55} [{genres}]")
