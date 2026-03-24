# ML Journey: Hybrid Recommendation & Ranking Platform

This file is a living log of what was built, what was learned, and the ML concepts
encountered at each step. It exists to make the trajectory legible — not just the
end product, but the thinking and experimentation behind it.

The point of this project is NOT "I built a movie recommender."
The point is: I designed and ran an ML experimentation pipeline, compared modeling
strategies rigorously, understood why each one works or fails, and built a
production-style serving system around the results.
Movies are the domain. ML in production is the subject.

---

## ML Concepts Reference

Concepts encountered across this project, explained in context.

### Sparsity
The user-item interaction matrix has 6,040 users × 3,706 movies = ~22M possible
pairs. Only ~1M are observed (4.5%). The other 95.5% are unknown — we don't know
if the user would like those movies; they just haven't rated them.

This is the fundamental challenge of recommendation. Every model in this project
is essentially trying to fill in that sparse matrix intelligently.

### Selection Bias in Ratings
Users don't rate movies randomly — they rate movies they chose to watch, and they
chose to watch movies they expected to enjoy. So the observed ratings skew high
(mean 3.58 in this dataset). This means ratings are NOT a representative sample
of true preferences. A 1-star rating still means the user engaged. This is why
we treat ratings as implicit confidence signals rather than exact target values.

### The Ranking Problem
This is NOT a rating prediction task. We don't need to predict exact star ratings.
We need to produce a ranked list where the most relevant items appear at the top.
The difference matters: a model could have terrible MSE on rating prediction but
still produce an excellent top-10 ranking, and vice versa.

### Precision@K, Recall@K, NDCG@K
Standard ranking metrics, all computed at cutoff K=10:

- Precision@K: of the K items we recommend, what fraction were actually relevant?
  Measures: are our recommendations good?

- Recall@K: of all items the user would have liked (in the test set), what fraction
  did we surface in our top K?
  Measures: how much relevant content did we find?

- NDCG@K (Normalized Discounted Cumulative Gain): like precision, but rewards
  placing relevant items *higher* in the ranking. Recommending a relevant item at
  position 1 is worth more than at position 10. This is the most important metric
  for a ranked system.

The tension between precision and recall: a model can boost recall by recommending
more broadly (but precision drops). A model can boost precision by only recommending
sure things (but recall drops). NDCG captures whether we're putting the right things
first.

### Time-Aware Splitting
Recommendation is a future-prediction task: given past behavior, predict future
preferences. If we split randomly, future ratings leak into the training set, which
inflates metrics artificially. A time-aware split ensures the model only ever sees
the past when predicting the future.

### Popularity Bias
Models trained on interaction data inherit the biases of that data. Popular items
have more training signal (more ratings, stronger gradients), so models tend to
over-recommend them. This hurts personalization: a system that just recommends
whatever's popular is not useful to users with niche tastes.

### Matrix Factorization (Collaborative Filtering)
The key idea: decompose the sparse user-item matrix into two dense, low-rank matrices:
  Users (N × d) × Items (d × M) ≈ Ratings (N × M)

Each user gets a d-dimensional latent vector. Each item gets a d-dimensional latent
vector. The relevance score for a user-item pair is their dot product. Training
adjusts these vectors so dot products are high for observed positive interactions.

The d dimensions are latent — you don't define them. The model discovers structure
from patterns. One dimension might loosely encode "preference for 1980s action films",
another "preference for slow character studies". This is learned entirely from
who-rated-what.

### Implicit vs Explicit Feedback
Explicit: user directly rates an item (stars, thumbs).
Implicit: user behavior inferred as signal (clicks, watches, ratings treated as
  confidence weights rather than exact targets).

We use implicit ALS (Alternating Least Squares), which treats rating value as a
confidence weight: a 5-star rating is high-confidence positive; a 1-star is
low-confidence positive (user engaged but didn't love it). This avoids treating
low ratings as "negatives" when we don't have true negatives.

### Content-Based Filtering
Instead of learning from who-liked-what, represent items by their attributes
(genre, description, tags) and build a user profile by averaging the attribute
vectors of their liked items. Recommend items whose attributes are closest to
the user's profile (cosine similarity).

Strength: works without any interaction data — useful for new users or new items.
Weakness: limited by the quality and expressiveness of item features.

### Hybrid Reranking (Two-Stage Pipeline)
Stage 1 — Candidate Generation: fast, recall-oriented retrieval. Generate a pool
of ~100 candidate items per user from CF + content-based. Priority is to not miss
relevant items.

Stage 2 — Reranking: a learned model takes each (user, candidate) pair, computes
rich features from multiple signals (CF score, content score, popularity, recency,
genre overlap, etc.), and produces a final relevance score. Priority is precision
and correct ordering.

This separation is used in every major production recommendation system because
it balances speed (cheap retrieval over millions of items) with quality (expensive
scoring over a small candidate set).

### LightGBM (Gradient Boosted Trees for Ranking)
LightGBM with objective='lambdarank' trains a gradient boosted decision tree to
optimize NDCG directly. It takes the engineered feature table and learns which
combinations of features best predict relevance. Feature importance from GBDT
tells us which signals matter most — this is directly interpretable.

---

## Phase Completion Log

---

### Phase 0 — Project Scaffold
**Status:** COMPLETE | **Date:** 2026-03-24

Set up the full project structure: data pipeline, model source modules, backend,
frontend, artifact storage. Committed PLAN.md, JOURNEY.md, and a .gitignore that
keeps raw data and model artifacts out of git while committing metrics JSON
(so the insights page works without retraining).

Key decisions:
- MovieLens 1M: dense enough for meaningful CF, small enough for local training
- LightGBM reranker: handles mixed tabular features, interpretable via feature importance
- Time-aware split: simulates real future-prediction, avoids data leakage
- Four models: popularity → CF → content-based → hybrid reranker (increasing complexity)
- Metrics: Precision@10, Recall@10, NDCG@10

---

### Phase 1 — Data Ingestion & EDA
**Status:** COMPLETE | **Date:** 2026-03-24

**ML concept encountered:** Sparsity, selection bias, power-law item distributions

Outputs: data/processed/ratings.csv, movies.csv, notebooks/01_eda.ipynb,
         reports/evaluation/{rating_distribution, interactions_per_user,
         item_popularity, genre_distribution}.png

Findings:
- 1,000,209 ratings | 6,040 users | 3,706 movies
- Sparsity: 95.53% — each user rated ~0.04% of the catalog
- Rating skew high (mean 3.58, median 4.0) → confirmed selection bias
  → decision: use implicit confidence weighting, not rating regression
- All users have ≥20 ratings (pre-filtered dataset) → cold-start less severe
  than real systems → note this limitation on insights page
- Item popularity is a power law: top 1% of movies = 8.7% of all ratings
  → popularity bias will be a real risk in trained models
- User segments: cold 28.9% | warm 44.8% | hot 26.3%
- 34 months of data → time-aware split is meaningful

---

### Phase 2 — Train/Val/Test Split
**Status:** COMPLETE | **Date:** 2026-03-24

**ML concept encountered:** Time-aware splitting, data leakage prevention

Outputs: data/processed/{train,val,test}.parquet

Split: per-user chronological 80/10/10
Train: 797,758 | Val: 97,383 | Test: 105,068

Findings:
- Zero cold users (all to train-only) because MovieLens 1M pre-filters to ≥20
- Timestamps strictly increase train → val → test per user (verified)
- If split randomly instead: future ratings would leak into training, inflating
  all downstream metrics — this is why random splits are wrong for recs

---

### Phase 3 — Popularity Baseline
**Status:** COMPLETE | **Date:** 2026-03-24

**ML concept encountered:** Baselines, the value of simple benchmarks, Bayesian
averaging (count × avg_rating vs raw avg_rating)

Results (weighted = count × avg_rating):
  P@10=0.0409 | R@10=0.0446 | NDCG@10=0.0520  [n=5,849 users]

These are the benchmark numbers every subsequent model must beat.

Findings:
- avg_rating alone is a terrible ranker: rewards obscure 5-star films with 3 ratings
  (high variance from small samples). Multiplying by count stabilizes it —
  this is essentially Bayesian shrinkage toward the global mean.
- Top recs are exactly what you'd expect: Star Wars, The Matrix, Godfather.
  Crowd-pleasers with broad appeal. Zero personalization.
- The baseline is stronger than it looks. Many naive personalization systems
  can't beat it — which is a real problem in production.

---

### Phase 4 — Collaborative Filtering (Matrix Factorization)
**Status:** COMPLETE | **Date:** 2026-03-24

**ML concept encountered:** Matrix factorization, latent embeddings, ALS, implicit
feedback, popularity bias in learned models, recall vs precision tradeoff

Model: implicit ALS | 64 latent factors | 20 iterations | confidence α=40

Overall results:
  P@10=0.0288 | R@10=0.0520 | NDCG@10=0.0433
Popularity baseline:
  P@10=0.0409 | R@10=0.0446 | NDCG@10=0.0520

CF beats popularity on recall but loses on precision and NDCG.

Segment breakdown:
  cold  (<50 ratings):  P=0.026 | R=0.089 | NDCG=0.059
  warm  (50-199):       P=0.026 | R=0.037 | NDCG=0.032  ← weakest
  hot   (≥200 ratings): P=0.040 | R=0.021 | NDCG=0.040

Findings:
- Higher R@10 than popularity (0.052 vs 0.045): CF surfaces more relevant items
  overall, but ranks them lower → lower NDCG. The items are there, just not on top.
- Warm users are the hardest segment: not enough history to specialize, but too
  much history for a cold-start approach. Classic mid-range problem.
- Popularity bias: 32.2% of CF recs are from top-100 movies (random = 2.7%).
  ALS inherits training data bias — popular items have stronger, better-trained
  embeddings simply because they appeared more during training.
- Sample recs ARE more personalized (niche classics vs blockbusters) — CF is
  finding real signal. It just doesn't always win on ordering quality.
- Core insight: CF improves recall but sacrifices precision. It casts a wider net
  but ranks items less confidently than popularity does for sure-thing crowd pleasers.
  The reranker's job is to fix this ordering.

---

### Phase 5 — Content-Based Model
**Status:** COMPLETE | **Date:** 2026-03-24

**ML concept encountered:** Feature representation, cosine similarity, filter bubble
effect, expressiveness of feature spaces, limitations of one-hot encoding

Model: genre one-hot (18 features) | cosine similarity to mean user profile

Overall results:
  P@10=0.0085 | R@10=0.0149 | NDCG@10=0.0128
(worst of all models — by a large margin)

Segment breakdown:
  cold  (n=2071): P=0.007 | R=0.021 | NDCG=0.014
  warm  (n=2563): P=0.009 | R=0.014 | NDCG=0.013
  hot   (n=1215): P=0.010 | R=0.007 | NDCG=0.011

Findings:
- Genre-only features are too coarse. 18 binary dimensions cannot distinguish
  two users who both like "Action + Drama" but with completely different tastes
  within those genres. The feature space has too little resolution.
- Cold-start hypothesis FAILED: content-based did NOT help cold users
  (cold R@10=0.021 vs CF cold R@10=0.089). Even sparse collaborative signal
  beats genre matching when the feature representation is this weak.
- Genre distribution in recs mirrors the catalog (Drama/Comedy/Action dominate)
  → the model is essentially recommending "popular genre films", not personalized picks
- ML-1M has no tags file → forced genre-only. Richer features would change this:
  sentence embeddings on movie descriptions, director/actor metadata, user-generated
  tags from ML-20M would make content-based competitive.
- KEY CONCEPT — filter bubble: content-based systems have a known weakness of
  over-recommending within the user's existing preference bands. If you always
  liked sci-fi, you only ever see sci-fi. This reduces novelty and diversity.
- KEY INSIGHT — weak models in ensembles: a bad standalone model can still
  contribute useful signal as ONE feature in a hybrid reranker. Genre overlap
  might not drive a recommendation alone, but combined with CF score and popularity
  it may help the reranker make better decisions.

---

### Phase 6 — Hybrid Reranker
**Status:** PENDING

ML concept to encounter: two-stage retrieval + reranking, feature engineering for
ranking, gradient boosted trees (LambdaRank), feature importance as interpretability

What to do:
- Candidate generation: union of top-50 from CF + top-50 from content-based
- Build feature table: (cf_score, content_score, pop_score, avg_rating,
  rating_count, genre_overlap, user_interaction_count) per candidate pair
- Train LightGBM LambdaRank on val set labels
- Evaluate on test set; compare feature importances

Questions to answer:
- Does combining signals improve NDCG over CF alone?
- Which features does LightGBM weight most heavily?
- Does the reranker fix the recall/precision tradeoff we saw in CF?

---

### Phase 7 — Evaluation & Error Analysis
**Status:** PENDING

ML concept to encounter: model comparison, failure analysis, diversity vs accuracy
tradeoff, popularity bias measurement

What to do:
- Full comparison table: all 4 models × all metrics × all user segments
- Failure cases: users where all models fail
- Popularity bias: what % of recs are top-100 items per model?
- Diversity metric: intra-list genre diversity per model

---

### Phase 8 — FastAPI Backend
**Status:** PENDING

ML concept to encounter: model serving, artifact management, inference pipeline
design, candidate generation at request time

---

### Phase 9 — React Frontend
**Status:** PENDING

Pages: Onboarding | Recommendations | Search | Insights (model analysis)

---

## Running Model Comparison

| Model | P@10 | R@10 | NDCG@10 | Notes |
|---|---|---|---|---|
| Popularity baseline | 0.0409 | 0.0446 | 0.0520 | Non-personalized; strong floor |
| Collaborative filtering | 0.0288 | 0.0520 | 0.0433 | Better recall, worse ordering |
| Content-based | 0.0085 | 0.0149 | 0.0128 | Genre-only too coarse |
| Hybrid reranker | TBD | TBD | TBD | |

---

## Key Architectural Decisions

| Decision | Choice | Why |
|---|---|---|
| Dataset | MovieLens 1M | Dense enough for CF; small enough for local training |
| CF approach | Implicit ALS | Ratings as confidence weights, not regression targets |
| Content features | Genre one-hot | Only features available in ML-1M; extensible to embeddings |
| Reranker | LightGBM LambdaRank | Optimizes NDCG directly; interpretable feature importance |
| Split strategy | Per-user chronological | Simulates real future-prediction, prevents leakage |
| Eval metrics | P/R/NDCG @10 | Standard IR metrics; NDCG is ranking-aware |
| Backend | FastAPI | Python-native; easy model loading; async |
| Frontend | React + Recharts | Component model fits multi-page app |

---

## Open Questions / Future Work

- Richer content features: sentence embeddings (e.g. all-MiniLM) on movie descriptions
- Session-based modeling: transformer on interaction sequences (BERT4Rec style)
- Neural reranker: two-tower deep model instead of GBDT
- Online learning: update user embeddings from new interactions without full retrain
- Contextual signals: time of day, device type, mood tags
- A/B testing framework to compare models on live traffic
