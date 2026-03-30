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

Every concept encountered in this project, explained from first principles.
Written to be read linearly — each section builds on the previous ones.

---

### 1. The Recommendation Problem

At its core, recommendation is a prediction problem: given what a user has done
in the past, predict what they will want in the future.

The naive framing is "predict ratings" — train a model to guess that user A would
give movie X a 4.2/5. But this is the wrong framing. Nobody cares about the exact
predicted star rating. What matters is: which 10 movies should we show this user
right now, ordered so the best ones come first?

This reframes the problem as a **ranking** problem, not a regression problem.
That distinction shapes every modeling and evaluation decision in this project.

---

### 2. Sparsity — Why Recommendation is Hard

Imagine a spreadsheet: rows are users, columns are movies, cells are ratings.
In theory this is 6,040 × 3,706 = 22.4 million cells. In practice, only ~1 million
cells have values. The other 95.5% are empty.

That 95.5% is called **sparsity**. But here's the thing — empty doesn't mean the
user dislikes the movie. It means they haven't seen it, or they saw it but didn't
rate it. The emptiness is informative ambiguity, not a signal of dislike.

This creates the central difficulty: you are trying to learn patterns from a matrix
that is almost entirely missing. Every model in this project is a different strategy
for extrapolating intelligently from those ~1 million observed cells to make
predictions about the ~21 million unobserved ones.

Why is sparsity worse for some users?
- A user with 500 ratings gives the model a lot to work with. It can learn their
  preferences across many genres, eras, directors.
- A user with 5 ratings gives the model almost nothing. Their preferences are
  barely defined. This is the **cold-start problem** (covered below).

---

### 3. Selection Bias — Why Ratings Are a Distorted Signal

In an ideal world, ratings would be a random sample of a user's true preferences
across all movies. They're not.

People rate movies they chose to watch. They chose to watch movies they expected
to like. So the observed distribution of ratings skews positive (mean 3.58 here,
where 3 = "neutral"). This is called **selection bias** — the sample you observe
is not representative of the underlying population.

Concrete implication: if you train a model to predict ratings and evaluate it on
MSE (mean squared error), you are optimizing for a biased signal. A model that
learns "users generally give 3.5-4 stars" will have low MSE on the observed data
but will tell you nothing about which movies a user would actually prefer.

How we handle it: we treat ratings not as exact targets to predict, but as
**confidence weights**. A 5-star rating = strong evidence the user liked this
kind of content. A 1-star rating = the user engaged with the movie (that's signal)
even if they disliked it. We never treat unrated movies as negative — we just
don't know.

---

### 4. Implicit vs Explicit Feedback

**Explicit feedback**: the user deliberately tells you their preference.
  Examples: 5-star rating, thumbs up/down, "not interested" button.

**Implicit feedback**: user behavior you interpret as a preference signal.
  Examples: watch history, clicks, saves, replays, skips.

The distinction matters because most real-world systems are dominated by implicit
feedback. In Netflix's case, they have billions of plays but relatively few
explicit ratings. Ratings are expensive (users have to take action); behavior is
free (it just happens).

For MovieLens, we have explicit ratings but we treat them as implicit. Why?
Because of selection bias above — the rating value is noisy and biased. Instead
we use:
  confidence = 1 + α × rating

A 5-star rating → high-confidence observation. A 1-star → lower-confidence but
still a positive observation (the user engaged). α=40 is a standard scaling
parameter. This converts a 5-star rating into confidence=201 and a 1-star into
confidence=41 — same sign, very different weight.

The ALS model uses these confidence weights during training, not the raw ratings.

---

### 5. Data Splitting — Why Random Splits Are Wrong for Recommendation

In standard ML (say, image classification), splitting data randomly into
train/val/test is fine. Each image is independent. Training on 80% and testing
on 20% is representative.

Recommendation is different. **User interactions are ordered in time.** Rating
Inception in 2002 and rating Interstellar in 2010 are not interchangeable events.
The earlier rating is part of the user's history that should *inform* predictions
about later preferences.

If you split randomly:
- Some of the user's 2010 ratings land in the training set.
- You train the model on data that includes future behavior.
- At test time, you ask "would this user like X?" but the model already "saw" their
  future interactions during training.
- This is **data leakage**. Metrics will be optimistically inflated.

The fix is a **time-aware split**: for each user independently, sort interactions
by timestamp, then take the first 80% for training, next 10% for validation, last
10% for test. This way, every test interaction genuinely came after all training
interactions for that user — exactly the real-world scenario.

---

### 6. Evaluation Metrics — Measuring Ranking Quality

All three metrics are computed at cutoff K=10 (we evaluate the top 10 recommendations).

**Precision@K**
Of the K items we recommended, what fraction were actually relevant?
  P@10 = (# of relevant items in top 10) / 10

Relevant = movie the user rated ≥ 4 stars in the test set.

Example: if 2 of our 10 recommendations are in the user's test set → P@10 = 0.2
Interpretation: "20% of what we showed you was actually good."

**Recall@K**
Of all relevant items in the test set, what fraction did we surface?
  R@10 = (# of relevant items in top 10) / (total relevant items)

Example: user has 5 relevant test movies, we found 2 of them → R@10 = 0.4
Interpretation: "We found 40% of what would have been good for you."

**The precision-recall tradeoff**
These two metrics pull in opposite directions. Imagine you're allowed to recommend
100 movies instead of 10. Your recall goes up (more chances to hit relevant items),
but your precision drops (you're including more guesses). There's a fundamental
tradeoff — the right balance depends on what you're optimizing for.

**NDCG@K (Normalized Discounted Cumulative Gain)**
Precision and recall treat the top 10 as a bag — position within the list doesn't
matter. NDCG fixes this by rewarding you for placing relevant items *higher*.

The intuition: if you found a relevant movie, was it your #1 recommendation or
your #10? Showing it first is much more valuable. Users rarely scroll to the
bottom of a recommendation list.

Mathematically:
  DCG@K = sum over positions of: (1 if relevant else 0) / log2(position + 1)

Position 1 → divided by log2(2) = 1.0 (no discount)
Position 2 → divided by log2(3) = 0.63 (some discount)
Position 10 → divided by log2(11) = 0.29 (heavy discount)

NDCG divides by the ideal DCG (all relevant items at the very top), so it's
normalized to [0, 1].

NDCG is the most important metric here because real users stop scrolling early.
A relevant item at position 10 might as well not exist.

---

### 7. The Popularity Baseline — Why Simple Benchmarks Matter

Before training any ML model, always ask: "what's the dumbest possible thing I
could do, and how well does it work?"

For recommendation, the dumbest thing is: recommend globally popular items to
everyone, ignoring who they are. Our popularity model scores each movie as:
  score = interaction_count × average_rating

(using both together avoids rewarding obscure 5-star films with only 3 ratings)

This baseline achieved P@10=0.041, NDCG@10=0.052. That's a high bar.

Why is this important? Because if your personalized ML model can't beat this,
you haven't learned anything useful beyond what's already obvious from the data.
Many production recommender systems have failed exactly this check — they looked
impressive until someone compared them against "just show popular stuff."

**Bayesian averaging**: the reason we multiply count × avg_rating rather than
using avg_rating alone relates to statistical confidence. A movie with 3 five-star
ratings has a high average but we can't trust it — small sample. A movie with
3,000 ratings at avg 4.1 is more reliably good. Multiplying by count implicitly
shrinks low-count item scores toward zero, which is a form of regularization.

---

### 8. Collaborative Filtering and Matrix Factorization

**Collaborative filtering** means "use the behavior of similar users to inform
recommendations." If user A and user B have similar rating histories, and user B
loved a movie that user A hasn't seen, recommend it to user A.

Early CF systems found similar users by computing cosine similarity over raw rating
vectors. This doesn't scale (6,040 × 6,040 similarity computations, and rating
vectors are 3,706-dimensional and extremely sparse).

**Matrix factorization** solves this with a much more elegant approach. Instead
of comparing raw rating vectors, learn a compressed representation.

The math: you have a rating matrix R of shape (n_users × n_items). Factorize it:
  R ≈ U × V^T
  U is shape (n_users × d)     — one row per user, d-dimensional embedding
  V is shape (n_items × d)     — one row per item, d-dimensional embedding

where d << n_items (we used d=64, vs n_items=3,706). Each user becomes a
64-number vector. Each movie becomes a 64-number vector.

The predicted score for user u and movie i is: U[u] · V[i] (dot product).

Training adjusts U and V so that dot products are high for observed positive
interactions. The model never directly "learns" what the 64 dimensions mean —
they emerge from the data. But they often correspond to real structure:
- One dimension might encode "preference for blockbuster action films"
- Another might encode "appreciation for slow, character-driven European cinema"
- Another might capture "era preference: 1970s vs 2000s"

The model discovers these axes of taste automatically from who-rated-what.

**Why "low rank"?** The assumption is that user preferences can be explained by
a small number of underlying factors (d=64), not 3,706 independent factors for
every movie. This is a strong but empirically useful assumption — most human taste
can be characterized by dozens of preference axes, not thousands.

---

### 9. ALS — How Matrix Factorization Actually Trains

ALS stands for Alternating Least Squares. It's one algorithm for solving the
matrix factorization optimization problem.

The challenge: you want to find U and V such that U × V^T matches the observed
ratings. But you can't optimize both U and V simultaneously in closed form.

ALS's solution: alternate between them.
  Step 1: Hold V fixed. Optimize U — with V fixed, each user's embedding can
          be solved exactly (it's a least squares problem with a closed-form
          solution per user).
  Step 2: Hold U fixed. Optimize V — with U fixed, each item's embedding can
          be solved the same way.
  Repeat for N iterations (we used 20).

Each iteration improves both sets of embeddings. It converges to a good solution
without needing gradient descent or learning rate tuning.

The confidence-weighted version (implicit ALS) modifies the least squares to
weight each observed interaction by its confidence score. Interactions with higher
ratings contribute more to updating the embeddings.

---

### 10. Content-Based Filtering

Content-based filtering ignores other users entirely. It only looks at:
1. What are the attributes of each item (genre, tags, description)?
2. What attributes did this user tend to like?
3. Recommend items whose attributes match the user's profile.

**Building item vectors**
For each movie, we create a numerical vector representing its content.
With only genre information (18 genres), each movie becomes an 18-dimensional
binary vector: [1, 0, 1, 0, 0, 1, ...] meaning "has Drama, not Comedy, has
Thriller, ..."

This is called **one-hot encoding** applied to multi-label categories.

**Building the user profile**
Take all movies the user rated highly (≥ 4 stars). Average their item vectors.
The result is a 18-dimensional vector that represents the user's content
preferences — e.g., [0.7, 0.1, 0.6, ...] meaning "70% of liked movies had
Drama, 10% had Comedy, 60% had Thriller..."

**Scoring candidates**
For each unseen movie, compute the **cosine similarity** between the movie's
vector and the user's profile vector.

Cosine similarity measures the angle between two vectors, not their magnitude.
It ranges from 0 (completely orthogonal, no similarity) to 1 (identical direction).
It's ideal for sparse feature vectors because it normalizes for how many features
a movie has.

**Why it failed here**
18 genre dimensions are too coarse. Two users who both like "Action + Sci-Fi"
will get nearly identical recommendations even if one likes cerebral 2001: A Space
Odyssey and the other likes explosive Michael Bay films. The feature space doesn't
have enough resolution to capture the difference.

With richer features (sentence embeddings from movie descriptions, director info,
user-generated tags), content-based becomes much more powerful.

**The filter bubble problem**
A well-functioning content model can trap users in their own taste profile. If you
always recommend "more of the same," users never discover adjacent genres or
surprising films. Real systems deliberately inject diversity to avoid this.

---

### 11. Embeddings

An embedding is a dense, low-dimensional vector representation of a discrete entity
(a user, a movie, a word, an item).

Instead of representing "The Matrix" as movie #2571 (a meaningless integer), or
as a 3,706-dimensional one-hot vector (mostly zeros), an embedding represents it
as a 64-dimensional dense vector where every dimension has a meaningful learned value.

The power of embeddings: similar items end up with similar vectors. After CF
training, The Matrix and Blade Runner end up with similar item embeddings because
they were liked by similar users. You didn't explicitly tell the model these movies
are similar — it inferred it from the interaction patterns.

Embeddings are now a foundational concept across all of modern ML:
- Word2Vec: word embeddings where similar words have similar vectors
- BERT: contextual word embeddings
- CF matrix factorization: user and item embeddings
- Neural networks: every hidden layer is learning embeddings of its inputs

In this project: CF produces 64-dimensional embeddings for all 6,040 users and
3,706 movies. These embeddings encode taste and content in a shared space.

---

### 12. Feature Engineering for Ranking

When you have multiple signals (CF score, content score, popularity, etc.), one
approach is to try to combine them by hand: "score = 0.6 × CF + 0.3 × popularity
+ 0.1 × content." But choosing those weights is a guess.

A better approach: turn each signal into a **feature** and train a model to learn
the optimal combination from data.

In this project, for every (user, candidate movie) pair, we compute:
- cf_score: the dot product of user and movie embeddings from CF
- content_score: cosine similarity between user content profile and movie vector
- pop_score: item's global popularity (count × avg_rating)
- avg_rating: item's average rating
- rating_count: log of number of ratings (log because scale matters less than order)
- genre_overlap: how much the movie's genres match the user's genre history
- user_interaction_count: log of how many movies the user has rated in training

Each row in this feature table is one candidate pair. The label is 1 if the user
rated that movie ≥ 4 stars in the validation set, 0 otherwise.

This is **learning to rank** — train a model on these features to predict relevance.

**Feature interaction**: the key insight here was that genre_overlap outperformed
content_score even though both come from the same content model. Why? Because
genre_overlap is a *targeted interaction feature* — it directly captures the match
between this user's specific preferences and this movie's attributes. content_score
is a generic global similarity. Designing features that capture the interaction
between user and item is often more valuable than the raw model output.

---

### 13. Gradient Boosted Trees and LightGBM

LightGBM is a **gradient boosted decision tree** framework. Understanding it
requires understanding both decision trees and boosting.

**Decision tree**: a flowchart of if/else rules learned from data.
  "If cf_score > 0.8 AND genre_overlap > 0.3, predict relevant."
  Each split is chosen to maximize the purity of the resulting groups.

**Ensemble**: instead of one tree, train many. Each tree corrects the errors of
the previous ones. This is **gradient boosting** — each new tree is fit on the
*residual errors* (gradient) of the current ensemble.

After 300 trees, the model has learned thousands of rules about when different
feature combinations predict relevance. It handles non-linear relationships and
feature interactions automatically — no need to manually compute "cf_score ×
popularity" as a feature, because the tree splits will implicitly capture that.

**Why GBDT for ranking?**
- Handles mixed feature types (some scores, some log counts) without normalization
- Naturally captures interactions between features
- Feature importance is interpretable (how much did each feature reduce prediction
  error across all trees)
- Fast to train and inference
- Generally strong on tabular data with engineered features

**LambdaRank objective**
Standard tree objectives minimize MSE (regression) or cross-entropy (classification).
LambdaRank uses a custom objective that directly optimizes NDCG — the metric we
actually care about. Instead of treating each candidate as independent, it considers
the relative ordering of candidates for the same user and applies larger gradients
to swaps that would most improve NDCG. This alignment between training objective
and evaluation metric is important.

---

### 14. The Two-Stage Retrieval + Reranking Pipeline

This architecture is used in virtually every large-scale production recommendation
system (YouTube, Netflix, Spotify, Twitter, etc.). Understanding why requires
thinking about the tradeoffs.

**The problem**: you have millions of items. You could score each one for each user
with a rich model, but that's too slow. You could use a fast cheap model for all
items, but the quality is lower.

**The solution**: split the problem into two stages.

Stage 1 — Candidate generation (retrieval):
Goal: recall. Don't miss relevant items. Speed is critical.
Method: fast approximate nearest neighbor search in embedding space, or simple
        score thresholds from CF + content models.
Scale: from millions of items → ~100-500 candidates per user.
Each candidate takes microseconds to score.

Stage 2 — Reranking:
Goal: precision and correct ordering. Get the ranking right.
Method: a more expensive model (GBDT, neural network) with rich features per
        candidate pair.
Scale: only ~100-500 candidates, so it can afford to be slower.
Each candidate takes milliseconds to score — acceptable when there are only 100.

Why you can't just use the reranker for everything: if you applied the full
feature engineering pipeline to all 3,706 movies × 6,040 users, that's 22M
feature table rows to compute and score per inference. Impractical at scale.

The two-stage pipeline separates "fast but coarse" from "slow but precise."
This project uses CF + content-based as Stage 1 (generates 50 candidates each)
and LightGBM as Stage 2 (reranks the union of ~100 candidates).

---

### 15. Popularity Bias and the Accuracy-Diversity Tradeoff

**Popularity bias** occurs when a model systematically over-recommends popular
items at the expense of niche or long-tail items.

Why it happens mechanically:
- Popular items appear more frequently in training data
- More training examples → stronger, more well-trained embeddings
- Better embeddings → higher CF scores → ranked higher → recommended more often
- This feedback loop is circular: popular items get recommended → more interactions
  → even more popular in training data for the next version

In this project:
- Popularity baseline: 99.9% bias (trivially — it only recommends the top 100)
- CF: 32.2% bias (inherits popularity signal from training data)
- Hybrid: 49.7% bias (reranker learned that avg_rating and pop_score predict
  relevance, which is true, but amplifies popularity)

**The tradeoff**
On one axis: accuracy (NDCG, precision). On the other: diversity and novelty.
A model optimized purely for accuracy will recommend popular items because they
have broad appeal — safe bets. But a user who wants to discover niche films is
poorly served.

In production this is managed by adding explicit diversity constraints ("no more
than 2 movies from the same genre"), re-ranking to inject long-tail items, or
adding a "discovery mode" separate from "personalized mode."

Our hybrid won on NDCG but increased popularity bias compared to CF. Better at
predicting what users will rate highly in aggregate, but less likely to surface
the hidden gem they'd love. This is an empirical confirmation of the tradeoff.

---

### 16. The Cold-Start Problem

The cold-start problem occurs when a model has no prior data about a user or item.

**User cold-start**: a brand new user has no interaction history. CF has nothing
to learn from. Content-based can still work (ask the user their preferences upfront
during onboarding). Popularity baseline also works (recommend broadly popular items
until you have more signal).

**Item cold-start**: a brand new movie has no ratings. CF has no interaction data
to embed it. Content-based works (you know the genre, director, etc. from the
day of release).

In this project: MovieLens 1M was pre-filtered to users with ≥20 ratings, so
user cold-start is artificially mild. In a real production system, a large fraction
of users would have 1-5 interactions, making cold-start the dominant challenge.

Our onboarding page (Phase 9) simulates cold-start handling: ask the user to rate
5-10 movies before showing recommendations. This gives the content-based model
enough signal to build a genre profile, and gives CF enough signal to produce
initial embeddings.

---

### 17. Overfitting, Generalization, and Why Validation Sets Matter

**Overfitting**: a model that learns the training data too well — including its
noise and quirks — and fails to generalize to new data.

In recommendation: a model could memorize "user 42 rated movie 318 a 5 in January"
and use that as a direct lookup rather than learning a generalizable pattern. On
training data it looks perfect. On new data it's useless.

**Regularization**: techniques that penalize overfitting by constraining the model.
In ALS, the regularization parameter (0.01 in this project) penalizes large
embedding values. This prevents any single user or item from getting an extreme
embedding that overfits to a small number of interactions.

**The validation set's role**: during the hybrid reranker training, we used the
validation set (val.parquet) to generate training labels for LightGBM — "did this
user actually interact with this candidate in the next time period?" The test set
is kept completely separate until the final evaluation. This ensures we can't
accidentally optimize for the test set metrics.

---

### 18. Macro vs Micro Averaging

When you compute an average metric across thousands of users, the averaging
method matters.

**Macro average**: compute the metric for each user independently, then average
across users. Every user contributes equally regardless of how many test items
they have.

**Micro average**: pool all predictions and labels together, then compute one
global metric. Users with more test items contribute more.

We use macro averaging throughout. This means a user with 3 test items and a user
with 30 test items count equally. This is more honest for user-centric evaluation:
we want the system to work well for every user, not just for heavy raters.

The 49.1% failure rate we observed is a macro metric: 49.1% of individual users
got zero NDCG, regardless of whether they had 2 or 20 test items.

---

### 19. Feature Importance and Model Interpretability

One advantage of gradient boosted trees over neural networks is interpretability.

After training LightGBM, we can ask: "for each feature, how much did it contribute
to improving NDCG across all 300 trees?" The answer is the feature importance score.

Our feature importance ranking:
  genre_overlap (1900) > cf_score (1842) > user_interaction_count (1818) >
  avg_rating (1174) > pop_score (904) > rating_count (799) > content_score (563)

What this tells us:
- genre_overlap being #1 means the direct match between user taste and item genre
  is the single most predictive signal. This is the content model's contribution —
  not through its raw score, but through a well-engineered interaction feature.
- cf_score being #2 confirms that collaborative behavioral signal is valuable.
- user_interaction_count being #3 means the reranker learned to trust CF scores
  more for active users (who have better-trained embeddings). This is the model
  learning implicit uncertainty quantification.
- content_score being last confirms what Phase 5 showed: the raw cosine similarity
  output of the content model is a weak signal when features are just genre.

This kind of interpretability is valuable in production: it tells you where to
invest in feature engineering next (richer content features, session signals, etc.).

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
**Status:** COMPLETE | **Date:** 2026-03-26

**ML concepts encountered:** Two-stage retrieval + reranking, feature engineering
for ranking, LightGBM LambdaRank (optimizes NDCG directly), feature importance
as model interpretability, accuracy vs diversity tradeoff

Model: LightGBM LambdaRank | 300 estimators | 7 features | 2,500 training users
Training data: 242,214 candidate pairs | 1.81% positive rate

Overall results:
  P@10=0.0375 | R@10=0.0621 | NDCG@10=0.0542

Segment breakdown (vs CF):
  cold  (n=2071): P=0.028 | R=0.097 | NDCG=0.063  [CF: R=0.089, NDCG=0.059] +7%
  warm  (n=2563): P=0.037 | R=0.049 | NDCG=0.046  [CF: R=0.037, NDCG=0.032] +44%
  hot   (n=1215): P=0.054 | R=0.030 | NDCG=0.057  [CF: P=0.040, NDCG=0.040] +43%

Feature importance ranking:
  1. genre_overlap          (1900) ← highest
  2. cf_score               (1842)
  3. user_interaction_count (1818)
  4. avg_rating             (1174)
  5. pop_score               (904)
  6. rating_count            (799)
  7. content_score           (563) ← lowest

Key learnings:
- Hybrid wins R@10 (0.062) and NDCG@10 (0.054) — best of all models
- Popularity still wins P@10 (0.041 vs 0.038) — it plays safe with sure-thing
  crowd-pleasers; hybrid takes personalized bets and wins more often overall
- Warm/hot users benefit most from reranking (+44% NDCG) — more training signal
  means richer feature vectors for LightGBM to work with
- THE PARADOX: content_score (raw cosine similarity) is the LEAST important
  feature, but genre_overlap (a targeted interaction feature from the same model)
  is the MOST important. Lesson: well-designed interaction features beat raw model
  scores. Specificity of the feature matters more than the sophistication of the
  source model.
- ACCURACY vs DIVERSITY TRADEOFF: hybrid has 49.7% popularity bias in recs
  (vs CF 32.2%). The reranker learned avg_rating and pop_score are strong
  predictors — correct, but it amplifies the popularity bias beyond CF's level.
  Better metrics, but less diverse recommendations.
- user_interaction_count (#3 importance) = the model is learning "how active is
  this user" as a proxy for CF embedding quality. More active users → more reliable
  CF scores → reranker can trust them more. This is implicit calibration.

---

### Phase 7 — Evaluation & Error Analysis
**Status:** COMPLETE | **Date:** 2026-03-29

**ML concepts encountered:** Model comparison methodology, failure analysis,
accuracy vs diversity tradeoff, popularity bias measurement, metric interpretation

Output: artifacts/metrics/comparison.json (powers insights page)

Full comparison:
  Popularity  P=0.041 | R=0.045 | NDCG=0.052 | bias=99.9% | diversity=0.337
  CF          P=0.029 | R=0.052 | NDCG=0.043 | bias=32.2% | diversity=0.387
  Content     P=0.009 | R=0.015 | NDCG=0.013 | bias=9.5%  | diversity=0.193
  Hybrid      P=0.038 | R=0.062 | NDCG=0.054 | bias=49.7% | diversity=0.387

Key learnings:
- Popularity bias=99.9%: the baseline recommends from the SAME ~100 movies for
  every user. Its decent precision comes from broad appeal, not personalization.
  Any system claiming to personalize must be compared against this honestly.
- Content-based diversity paradox: lowest bias (9.5%) but worst accuracy. Proves
  that "less popular" does not mean "more personalized" — it's just randomly
  unpopular. Diversity without relevance is noise.
- 49.1% failure rate: in nearly HALF of users, no model places any relevant item
  in top-10. The macro-averaged NDCG of 0.054 is pulled up by users where models
  work. For ~50% of users, the system returns nothing useful.
  Causes: small test sets (3-6 items), strict ≥4.0 threshold, temporal drift
  (preferences at test time may have shifted from training signal).
- Hybrid: best R@10 + NDCG, matches CF diversity — the right model to serve.
  But its 49.7% popularity bias is higher than CF (32.2%), meaning the reranker's
  accuracy gains come partly from learning to favor popular items. Classic
  accuracy vs diversity tradeoff confirmed empirically.
- Feature importance (from reranker): genre_overlap > cf_score >
  user_interaction_count > avg_rating > pop_score > rating_count > content_score

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
| Hybrid reranker | 0.0375 | 0.0621 | 0.0542 | Best R@10 + NDCG; highest popularity bias |

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
