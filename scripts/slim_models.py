"""
Post-process trained model pickles to strip runtime-only data that inflates file size.

What this does:
  - ContentBasedModel: precompute _user_profiles from _train, then delete _train
  - All models: strip _user_seen (will be loaded from user_seen.parquet by model_loader)
  - Precompute item_stats.parquet and user_train_count.json from train.parquet
  - Save user_seen.parquet once (shared across all models)

Before: content(672MB) + cf(117MB) + pop(63MB) + train(123MB) = 975MB
After:  content(~5MB)  + cf(~55MB) + pop(<1MB) + precomputed(<5MB) = ~66MB

Run from project root:
    python scripts/slim_models.py
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

MODELS   = ROOT / "artifacts" / "models"
COMPUTED = ROOT / "artifacts" / "precomputed"


def main():
    COMPUTED.mkdir(parents=True, exist_ok=True)

    # ── 1. Precompute from train.parquet ──────────────────────────────────────
    print("Loading train.parquet…")
    train = pd.read_parquet(ROOT / "data/processed/train.parquet")
    print(f"  {len(train):,} rows")

    print("Computing item_stats…")
    item_stats = (
        train.groupby("movie_id")["rating"]
        .agg(avg_rating="mean", rating_count="count")
        .assign(pop_score=lambda d: d["rating_count"] * d["avg_rating"])
        .reset_index()
    )
    item_stats.to_parquet(COMPUTED / "item_stats.parquet", index=False)
    print(f"  Saved item_stats.parquet ({len(item_stats):,} movies)")

    print("Computing user_train_count…")
    user_train_count = train.groupby("user_id")["movie_id"].count().to_dict()
    with open(COMPUTED / "user_train_count.json", "w") as f:
        json.dump({str(k): int(v) for k, v in user_train_count.items()}, f)
    print(f"  Saved user_train_count.json ({len(user_train_count):,} users)")

    print("Computing user_seen…")
    user_seen_df = train[["user_id", "movie_id"]].drop_duplicates()
    user_seen_df.to_parquet(COMPUTED / "user_seen.parquet", index=False)
    print(f"  Saved user_seen.parquet ({len(user_seen_df):,} rows)")

    del train  # free memory

    # ── 2. Slim content model ─────────────────────────────────────────────────
    print("\nSlimming content model…")
    with open(MODELS / "content_model.pkl", "rb") as f:
        cb = pickle.load(f)

    has_train = hasattr(cb, "_train")
    if has_train:
        print(f"  _train is {cb._train.memory_usage(deep=True).sum()//1024//1024}MB in memory")
        # Precompute user profiles if not already done
        if not hasattr(cb, "_user_profiles") or not cb._user_profiles:
            print("  Precomputing user profiles…")
            from src.models.content_based import POSITIVE_THRESHOLD
            import scipy.sparse as sp
            pos = cb._train[cb._train["rating"] >= POSITIVE_THRESHOLD]
            cb._user_profiles = {}
            for user_id, group in pos.groupby("user_id"):
                indices = [cb._item_index[m] for m in group["movie_id"] if m in cb._item_index]
                if not indices:
                    continue
                vecs = cb._item_vectors[indices]
                # Support both sparse (old genre model) and dense (new embedding model)
                if sp.issparse(vecs):
                    profile = np.asarray(vecs.mean(axis=0)).flatten()
                else:
                    profile = vecs.mean(axis=0)
                norm = np.linalg.norm(profile)
                cb._user_profiles[user_id] = profile / norm if norm > 0 else profile
            print(f"  Precomputed {len(cb._user_profiles):,} user profiles")
        del cb._train
        print("  Deleted _train")
    else:
        print("  No _train to remove (already slim)")

    if hasattr(cb, "_user_seen"):
        del cb._user_seen
        print("  Deleted _user_seen")

    with open(MODELS / "content_model.pkl", "wb") as f:
        pickle.dump(cb, f)
    size = (MODELS / "content_model.pkl").stat().st_size // 1024 // 1024
    print(f"  Saved content_model.pkl ({size}MB)")

    # ── 3. Slim CF model ──────────────────────────────────────────────────────
    print("\nSlimming CF model…")
    with open(MODELS / "cf_model.pkl", "rb") as f:
        cf = pickle.load(f)

    if hasattr(cf, "_user_seen"):
        del cf._user_seen
        print("  Deleted _user_seen")

    with open(MODELS / "cf_model.pkl", "wb") as f:
        pickle.dump(cf, f)
    size = (MODELS / "cf_model.pkl").stat().st_size // 1024 // 1024
    print(f"  Saved cf_model.pkl ({size}MB)")

    # ── 4. Slim popularity model ───────────────────────────────────────────────
    print("\nSlimming popularity model…")
    with open(MODELS / "popularity_model.pkl", "rb") as f:
        pop = pickle.load(f)

    if hasattr(pop, "_user_seen"):
        del pop._user_seen
        print("  Deleted _user_seen")

    with open(MODELS / "popularity_model.pkl", "wb") as f:
        pickle.dump(pop, f)
    size = (MODELS / "popularity_model.pkl").stat().st_size // 1024 // 1024
    print(f"  Saved popularity_model.pkl ({size}MB)")

    # ── 5. Summary ─────────────────────────────────────────────────────────────
    print("\n── Summary ─────────────────────────────────────────────────────────")
    total = 0
    for p in sorted((MODELS).glob("*.pkl")):
        s = p.stat().st_size // 1024 // 1024
        total += s
        print(f"  {p.name}: {s}MB")
    for p in sorted(COMPUTED.glob("*")):
        s = p.stat().st_size // 1024 // 1024
        total += s
        print(f"  precomputed/{p.name}: {s}MB")
    print(f"  Total: {total}MB")
    print("\nDone. Update model_loader.py to use precomputed/ files, then restart backend.")


if __name__ == "__main__":
    main()
