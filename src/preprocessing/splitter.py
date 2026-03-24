"""
Time-aware per-user train / val / test split.

For each user, sort interactions by timestamp and take:
  - first 80%  -> train
  - next  10%  -> val
  - last  10%  -> test

Users with fewer than MIN_INTERACTIONS are placed entirely in train (cold-start
users — the content-based model will handle them at inference).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PROCESSED_DIR = Path(__file__).parents[2] / "data" / "processed"
MIN_INTERACTIONS = 10
TRAIN_FRAC = 0.8
VAL_FRAC = 0.1  # test gets the rest


def split(ratings: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (train, val, test) DataFrames."""
    ratings = ratings.sort_values(["user_id", "timestamp"])

    train_rows, val_rows, test_rows = [], [], []
    cold_users: list[int] = []

    for user_id, group in ratings.groupby("user_id"):
        n = len(group)
        if n < MIN_INTERACTIONS:
            train_rows.append(group)
            cold_users.append(int(user_id))
            continue

        n_train = max(1, int(np.floor(n * TRAIN_FRAC)))
        n_val = max(1, int(np.floor(n * VAL_FRAC)))

        train_rows.append(group.iloc[:n_train])
        val_rows.append(group.iloc[n_train : n_train + n_val])
        test_rows.append(group.iloc[n_train + n_val :])

    train = pd.concat(train_rows).reset_index(drop=True)
    val = pd.concat(val_rows).reset_index(drop=True) if val_rows else pd.DataFrame()
    test = pd.concat(test_rows).reset_index(drop=True) if test_rows else pd.DataFrame()

    print(
        f"Split: train={len(train):,}  val={len(val):,}  test={len(test):,}  "
        f"cold_users={len(cold_users):,}"
    )
    return train, val, test, cold_users


def save_splits(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    cold_users: list[int],
    out_dir: Path = PROCESSED_DIR,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    train.to_parquet(out_dir / "train.parquet", index=False)
    val.to_parquet(out_dir / "val.parquet", index=False)
    test.to_parquet(out_dir / "test.parquet", index=False)
    (out_dir / "cold_start_users.json").write_text(json.dumps(cold_users))
    print(f"Saved splits to {out_dir}")
