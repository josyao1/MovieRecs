# Master Project Plan: Hybrid Recommendation & Ranking Platform

## What this project is

A full ML experimentation pipeline over a movie recommendation domain.
The research questions are:

1. How much does collaborative filtering improve over a strong popularity
   baseline, and for which users?
2. When does content-based modeling outperform behavioral CF?
3. Can a learned hybrid reranker measurably improve ranking quality (NDCG)
   by combining signals from both?
4. What are the failure modes and tradeoffs, and how do they manifest in a
   production serving system?

Movies are the domain. The ML trajectory is the point.

---

## System Architecture

```
Raw interaction data (MovieLens 1M)
        |
        v
[data/raw/]  ── scripts/download_data.py
        |
        v
[src/preprocessing/]  ── clean, split (time-aware per-user)
        |
        v
[data/processed/]  ── train.parquet / val.parquet / test.parquet
                      ratings.csv / movies.csv
        |
        +──────────────────────────────────────+
        |                                      |
        v                                      v
[src/models/popularity.py]         [src/features/feature_engineer.py]
[src/models/collaborative_filter.py]           |
[src/models/content_based.py]                  |
        |                                      |
        +──────────────────────────────────────+
        |
        v
[src/models/hybrid_reranker.py]  ── LightGBM ranker over engineered features
        |
        v
[src/evaluation/metrics.py]  ── Precision/Recall/NDCG @K
        |
        v
[artifacts/]
  models/          ── saved .pkl / .joblib files
  embeddings/      ── user_embeddings.npy, item_embeddings.npy
  metrics/         ── comparison.json (committed to git)
        |
        v
[backend/main.py]  ── FastAPI: loads artifacts, serves ranked recs
        |
        v
[frontend/]  ── React: onboarding, recs, search, insights pages
```

---

## Phase Breakdown

### Phase 0 — Scaffold (COMPLETE)
Files: .gitignore, JOURNEY.md, PLAN.md, all directory structure,
       src/__init__.py stubs, backend/requirements.txt stub

### Phase 1 — Data Ingestion & EDA
Files to create:
- scripts/download_data.py
- src/preprocessing/data_loader.py
- notebooks/01_eda.ipynb

Outputs:
- data/processed/ratings.csv
- data/processed/movies.csv

EDA checkpoints:
- [ ] Rating distribution (skew toward 3-4 stars)
- [ ] Interactions per user (long tail)
- [ ] Interactions per movie (popularity power law)
- [ ] Sparsity of user-item matrix
- [ ] Timestamp coverage

### Phase 2 — Split Pipeline
Files to create:
- src/preprocessing/splitter.py
- notebooks/02_preprocessing.ipynb

Outputs:
- data/processed/train.parquet
- data/processed/val.parquet
- data/processed/test.parquet
- data/processed/cold_start_users.json (users with < threshold interactions)

Validation:
- [ ] No user appears in test but not train
- [ ] Timestamps strictly increase from train → val → test per user
- [ ] Cold/warm/hot user segment counts documented

### Phase 3 — Popularity Baseline
Files to create:
- src/models/popularity.py
- src/evaluation/metrics.py
- notebooks/03_popularity_baseline.ipynb

Outputs:
- artifacts/metrics/popularity.json

API contract for all models:
```python
model.recommend(user_id: int, top_k: int = 10) -> List[Tuple[int, float]]
# returns [(movie_id, score), ...] sorted descending
```

### Phase 4 — Collaborative Filtering
Files to create:
- src/models/collaborative_filter.py  (implicit ALS via implicit library)
- notebooks/04_collaborative_filtering.ipynb

Outputs:
- artifacts/embeddings/user_embeddings.npy
- artifacts/embeddings/item_embeddings.npy
- artifacts/models/cf_model.pkl
- artifacts/metrics/cf.json

Analysis checkpoints:
- [ ] Metric comparison vs popularity baseline
- [ ] Per-segment breakdown (cold/warm/hot)
- [ ] Embedding visualization (UMAP or t-SNE, optional)
- [ ] Sample: what does the nearest-neighbor of a known movie look like?

### Phase 5 — Content-Based Model
Files to create:
- src/features/content_features.py
- src/models/content_based.py
- notebooks/05_content_based.ipynb

Outputs:
- artifacts/embeddings/item_content_vectors.npy
- artifacts/models/content_model.pkl
- artifacts/metrics/content.json

Analysis checkpoints:
- [ ] Cold-start performance vs CF
- [ ] Genre coverage of recommendations
- [ ] Cases where content-based beats CF

### Phase 6 — Hybrid Reranker
Files to create:
- src/features/feature_engineer.py  (builds candidate × feature table)
- src/models/hybrid_reranker.py     (LightGBM ranker wrapper)
- notebooks/06_hybrid_pipeline.ipynb
- scripts/train_all.py              (end-to-end training entrypoint)

Outputs:
- artifacts/models/reranker.pkl
- artifacts/metrics/hybrid.json

Feature set:
| Feature | Source |
|---|---|
| cf_score | CF model dot product |
| content_score | cosine similarity |
| popularity_rank | global item rank |
| avg_rating | item metadata |
| rating_count | item metadata |
| genre_overlap | user profile × item genres |
| days_since_release | item metadata |
| user_interaction_count | train interactions |
| item_interaction_count | train interactions |

### Phase 7 — Evaluation & Error Analysis
Files to create:
- notebooks/07_evaluation.ipynb
- reports/evaluation/model_comparison.md

Outputs:
- artifacts/metrics/comparison.json  (all 4 models, all metrics, all segments)

Analysis sections:
1. Side-by-side metric table
2. Cold-start gap: CF vs content-based for users with <10 interactions
3. Popularity bias: are top-10 recs dominated by globally popular items?
4. Diversity metric: intra-list genre diversity
5. Failure cases: 5 users where all models perform poorly
6. Feature importance from LightGBM reranker

### Phase 8 — FastAPI Backend
Files to create:
- backend/main.py
- backend/routes/recommendations.py
- backend/routes/search.py
- backend/routes/insights.py
- backend/model_loader.py
- backend/requirements.txt
- src/serving/inference.py

Endpoints:
| Method | Path | Description |
|---|---|---|
| POST | /onboard | Accept rated/selected movies; create session user profile |
| GET | /recommendations/{user_id} | Ranked recs with explanation tags |
| GET | /item/{item_id} | Movie metadata |
| GET | /search | Query + personalized rerank |
| GET | /metrics | Precomputed model comparison metrics |
| GET | /insights | Narrative findings + model analysis data |

Explanation tags (for /recommendations):
- "Similar to [movie] you rated highly"
- "Popular among users with similar taste"
- "Matches your preference for [genre]"
- "Hidden gem in your favorite genre"

### Phase 9 — React Frontend
Directory: frontend/ (Create React App or Vite)

Pages:
1. /onboard — Select/rate 5-10 movies; genre filter chips; submit → user_id
2. /recommendations — Card grid with poster, title, year, genres, explanation
                      tags; sort by model (dropdown)
3. /search — Search bar; results list with relevance + personalization scores
4. /insights — Full analysis page (see below)

Insights page sections:
- Hero: "This is not a movie app. It's an ML experiment."
- Model comparison table (interactive, sortable)
- Metric bar charts (Recharts)
- Cold-start analysis chart
- Feature importance from reranker
- Failure case gallery
- Key findings callouts
- Future work section

---

## Dependencies

### Python (backend + ML)
```
# ML
implicit          # ALS collaborative filtering
lightgbm          # gradient boosted reranker
scikit-learn      # TF-IDF, preprocessing, metrics
numpy
pandas
scipy

# Backend
fastapi
uvicorn[standard]
pydantic

# Data
requests          # for download script

# Notebooks
jupyter
matplotlib
seaborn
umap-learn        # optional: embedding visualization
```

### JavaScript (frontend)
```
react
react-router-dom
recharts           # charts for insights page
axios              # API calls
tailwindcss        # styling
```

---

## Evaluation Metric Definitions

**Precision@K**: Of the K recommended items, fraction that appear in test set
  P@K = |recommended ∩ relevant| / K

**Recall@K**: Of all relevant items in test, fraction recovered in top K
  R@K = |recommended ∩ relevant| / |relevant|

**NDCG@K**: Normalized Discounted Cumulative Gain — rewards placing relevant
  items higher in the ranking
  DCG@K = sum(rel_i / log2(i+1) for i in 1..K)
  NDCG@K = DCG@K / IDCG@K

All metrics computed over users with at least 1 test interaction.
Averaged across users (macro average).

---

## Model Comparison Hypothesis

Before training, predicted ordering:
1. Hybrid reranker (should win on NDCG — better ordering)
2. Collaborative filtering (wins for warm/hot users)
3. Content-based (wins for cold users)
4. Popularity baseline (strong precision, weak personalization)

The interesting result to look for:
- Does CF actually beat popularity for sparse users, or does it overfit?
- Does the hybrid reranker improve NDCG even if precision is similar to CF?
- Is the cold-start gap large enough to justify maintaining content-based?
