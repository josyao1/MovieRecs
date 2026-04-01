# How the MovieRecs Models Work

A plain-English guide to every model in this system — what it does, why it exists, and what the jargon means. Written for someone who knows what a movie recommendation is but isn't deep in ML.

---

## Table of Contents
1. [The Big Picture](#the-big-picture)
2. [Vocabulary You'll See Everywhere](#vocabulary-youll-see-everywhere)
3. [Model 1 — Popularity Baseline](#model-1--popularity-baseline)
4. [Model 2 — Collaborative Filtering (CF)](#model-2--collaborative-filtering-cf)
5. [Model 3 — Content-Based](#model-3--content-based)
6. [Model 4 — Hybrid Reranker](#model-4--hybrid-reranker)
7. [How They All Fit Together at Inference Time](#how-they-all-fit-together-at-inference-time)
8. [Why This Architecture](#why-this-architecture)

---

## The Big Picture

We have four models, used in layers:

```
User request
    │
    ▼
CF Model ──────────────┐
                        ├──► top-50 candidates each ──► Hybrid Reranker ──► top-10 final
Content-Based Model ───┘
    │
    ▼
(if no ratings exist for user)
Popularity Model ──────────────────────────────────────────────────────────► top-10 final
```

Two fast models generate a big pool of candidates. One smart model reranks that pool into the final list. If the user is brand new and has no ratings, the popularity model steps in.

---

## Vocabulary You'll See Everywhere

**Implicit feedback vs explicit feedback**
- *Explicit*: a user gave a star rating. You know exactly what they thought.
- *Implicit*: a user watched something, bought something, clicked something. You infer interest from behavior.
- This system uses *explicit* ratings (MovieLens star ratings) but treats them *implicitly* — meaning we don't try to predict the exact rating, we just use high ratings as a signal that the user liked something.

**Training data**
The dataset we learn from. Here: 25 million ratings from MovieLens, split into training and validation sets.

**Validation set**
A held-out chunk of ratings the models never see during training. Used to measure how good predictions are before deploying.

**Cold-start problem**
What to do when a new user has no history. CF breaks completely (it has nothing to go on). Content-based and popularity models pick up the slack.

**Embedding / vector**
A list of numbers that represents something — a movie, a user, a sentence. The key insight: things that are *similar* should have *similar* vectors. You can measure similarity by how close the vectors are in space.

**Dot product**
A way to measure how similar two vectors are. Multiply each pair of numbers and sum them up. Higher = more similar (when vectors are normalized).

**Cosine similarity**
A slightly fancier similarity measure. For two L2-normalized vectors it's identical to the dot product. "Normalized" just means we rescale each vector so its length is exactly 1, which makes dot products a fair comparison across items.

**L2 normalization**
Dividing a vector by its own length so it becomes a unit vector (length = 1). We do this to item embeddings so similarity scores are all on the same scale.

**Sparse matrix**
A giant grid (users × movies) that is *mostly empty* because most users have only rated a tiny fraction of all movies. We store only the non-zero entries to save memory.

**CSR matrix (Compressed Sparse Row)**
A memory-efficient format for storing a sparse matrix. Each row (user) stores only the columns (movies) where there's a value.

**Hyperparameter**
A setting you choose before training that controls *how* the model learns — like how many factors to use in CF, or how many trees to grow in LightGBM. Different from *parameters*, which are the numbers the model learns from data.

**Overfitting**
When a model memorizes the training data instead of learning general patterns. It performs great on training data but badly on new data.

**Regularization**
A penalty added during training to prevent overfitting. Forces the model to stay "simple" unless the data strongly supports complexity.

**NDCG (Normalized Discounted Cumulative Gain)**
The metric we actually care about. Measures whether the *best* recommendations appear at the *top* of the list. Getting the best movie at rank 1 is worth more than at rank 5. Ranges 0–1, higher is better.

---

## Model 1 — Popularity Baseline

**File:** `src/models/popularity.py`

### What it does
Recommends movies that are globally popular — lots of ratings, high average rating — minus anything the user has already seen.

### How it scores movies
```
score = interaction_count × average_rating
```
A movie rated 4.5 stars by 10,000 people scores much higher than one rated 4.5 stars by 50 people. This is called a "weighted" score — it balances quality against volume.

### Why it exists
Two reasons:
1. **Cold-start fallback.** A brand-new user has no history. CF needs history to work. Content-based needs liked movies to build a taste profile. Popularity needs nothing — it's always available.
2. **Benchmark.** Every other model has to beat this. If your fancy ML model can't outperform "just recommend popular movies," something is wrong.

### What it can't do
It gives everyone the same recommendations (personalized only by filtering out seen movies). It knows nothing about your taste.

---

## Model 2 — Collaborative Filtering (CF)

**File:** `src/models/collaborative_filter.py`
**Library:** [`implicit`](https://github.com/benfred/implicit)
**Algorithm:** Alternating Least Squares (ALS)

### Core idea
*"Find users who liked similar movies to you, then recommend what they liked that you haven't seen."*

It never looks at what movies are *about*. It only looks at *patterns of ratings*.

### The matrix factorization intuition

Imagine a giant grid: every row is a user, every column is a movie, every cell is a rating. Most cells are empty (you haven't rated most movies). CF's job is to fill in the blanks — to predict which empty cells would be high ratings.

**Matrix factorization** says: instead of storing this giant grid directly, compress it. Represent each user as a short list of numbers (a *user vector* or *embedding*) and each movie as a short list of numbers (*item vector*). The prediction for "how much will user A like movie X" is just the dot product of their two vectors.

```
predicted_rating(user, movie) = user_vector · movie_vector
```

The model learns these vectors such that dot products match the observed ratings as closely as possible.

### ALS — how it actually learns

**Alternating Least Squares** is the training algorithm. It works in two steps, repeated in alternation:

1. **Fix item vectors, solve for user vectors.** Given fixed movie embeddings, find the user embedding that best explains each user's ratings. This is a standard linear algebra problem (least squares).
2. **Fix user vectors, solve for item vectors.** Given fixed user embeddings, find the movie embedding that best explains each movie's ratings.

Repeat 20 times (iterations=20). Each pass refines both sides until they converge.

### Implicit feedback and confidence weights

We only use ratings ≥ 3.5 as "positive interactions" (the user liked it). Lower ratings are discarded — we treat them as neutral, not negative.

But we don't just record 1/0 (liked/not liked). We use a **confidence weight**:
```
confidence = 1 + 40 × rating
```
A 5-star rating gets confidence 201. A 3.5-star rating gets confidence 141. The model trusts high-confidence signals more when fitting the embeddings. The `40` (alpha) is a standard choice from the ALS paper.

### Factors (dimensions)

`factors=64` means each user and movie is represented by a 64-number vector. More factors = more expressive but slower and more prone to overfitting. 64 is a reasonable middle ground.

### At inference time

1. Look up the user's 64-number vector.
2. Multiply it against the matrix of all movie vectors (matrix multiplication — fast).
3. Sort by score, filter out seen movies, return top candidates.

### What it's good at
Capturing latent (hidden) patterns across users. If thousands of people who loved The Godfather also loved Goodfellas, CF picks that up without needing to know anything about mob movies.

### What it can't do
- New users with no ratings (cold-start)
- New movies with no ratings (item cold-start)
- Understanding *why* movies are similar

---

## Model 3 — Content-Based

**File:** `src/models/content_based.py`
**Model:** `all-MiniLM-L6-v2` (sentence transformer)

### Core idea
*"Represent every movie as a point in semantic space. Find movies near the ones you liked."*

Unlike CF, this model reads the actual text describing each movie.

### Building movie vectors

For each movie we build a single text string:
```
"Title. Genre1 Genre2. Plot description from TMDB."
```
Example:
```
"The Dark Knight. Action Crime Drama. Batman raises the stakes in his war on crime..."
```

This string is fed into **all-MiniLM-L6-v2**, a sentence transformer model (a compact BERT-style neural network). It outputs a 384-number vector that captures the *meaning* of the text. Movies about similar things end up with similar vectors.

### What is a sentence transformer?

A sentence transformer is a neural network trained on enormous amounts of text to understand language. It maps any piece of text to a fixed-size vector such that *semantically similar text produces similar vectors*. "A story about revenge" and "a tale of vengeance" would get very close vectors, even though none of the words match.

`all-MiniLM-L6-v2` is a small, fast version (6 layers, outputs 384 dimensions). It's accurate enough and fast enough to run on CPU for 62,000 movies in reasonable time.

### Building user taste profiles

A user's taste profile is the **mean** (average) of the item vectors for every movie they rated ≥ 3.5:

```
user_profile = mean(item_vector_1, item_vector_2, ... item_vector_n)
```

Then we L2-normalize it (scale it to length 1). This profile is a single 384-number vector sitting somewhere in the same semantic space as all the movies.

### Recommending

Dot-product the user's profile against every movie vector. Higher dot product = the movie's semantic meaning is closer to the user's taste centroid. Filter out seen movies, return the top candidates.

### Genre profiles

A separate `_user_genre_profiles` dict stores what fraction of a user's liked movies fell into each genre (e.g. 40% Action, 30% Drama, 15% Comedy). This gets passed to the reranker as a feature — it's a cheap but powerful signal.

### What it's good at
- Semantic understanding: distinguishes two action movies that are tonally very different
- Works for new users with a few ratings (cold-start is much milder than CF)
- Works for new movies as long as we have a description

### What it can't do
- Discovery across genre/style boundaries (CF is better at "people who liked X also liked Y even though they're different genres")
- Only as good as the descriptions — if TMDB has no overview, we just use title + genres

---

## Model 4 — Hybrid Reranker

**File:** `src/models/hybrid_reranker.py`
**Library:** LightGBM
**Algorithm:** LambdaRank (gradient-boosted decision trees)

### Core idea
*"Generate 50 candidates from CF and 50 from content-based (100 total, deduplicated), then train a smarter model to sort them."*

This is the **two-stage retrieval + reranking** architecture used in production at Netflix, Spotify, YouTube, etc. Stage 1 (retrieval) is fast and approximate. Stage 2 (reranking) is slower but more accurate.

### Why not just take the top-10 from CF or content-based directly?

Each model has blind spots. CF is good at cross-genre discovery but can't read plot descriptions. Content-based understands semantics but doesn't capture the crowd wisdom in ratings patterns. A separate model that has *both signals* as inputs can outperform either alone.

### What LightGBM is

LightGBM is a **gradient-boosted decision tree** library. Decision trees split data on feature thresholds ("is cf_score > 0.7?"). Gradient boosting builds many trees in sequence, each one correcting the mistakes of the previous ones. LightGBM is a fast, memory-efficient implementation of this.

### LambdaRank — optimizing the right thing

Most classification/regression models optimize metrics like accuracy or mean squared error. But we don't want to *predict ratings*, we want to get the *best movies at the top of the list*. That's a ranking problem, and the metric is NDCG.

**LambdaRank** is an algorithm that directly optimizes NDCG during training. Instead of asking "did I predict the rating correctly?" it asks "did I put the good movies higher than the bad ones?" This is the right objective for a recommendation system.

### The features

For each (user, candidate movie) pair, the reranker gets 7 features:

| Feature | What it is |
|---|---|
| `cf_score` | The raw score from the CF model |
| `content_score` | The raw cosine similarity from the content-based model |
| `pop_score` | The movie's global popularity score (count × avg rating) |
| `avg_rating` | The movie's average rating across all users |
| `rating_count` | log(1 + number of ratings) — log-scaled because this is skewed |
| `genre_overlap` | How much the movie's genres match the user's genre history |
| `user_interaction_count` | How many ratings this user has given (power user vs casual) |

### How it's trained

1. Take 2,000 users from the training set.
2. For each user, run CF and content-based to get up to 100 candidate movies.
3. Label each candidate 1 (user rated it ≥ 3.5 in the *validation set*) or 0 (not seen or low rating).
4. Build features for all (user, candidate) pairs.
5. Train LightGBM LambdaRank on this data — learn which features and combinations predict whether a candidate is actually good for a given user.

### Feature importance

After training you can inspect which features mattered most:
```python
reranker.feature_importance
```
Typically `cf_score` and `content_score` dominate, but `genre_overlap` and `rating_count` add meaningful signal.

### What it's good at
Everything — it combines all signals. This is the model that actually serves recommendations in production.

### What it can't do
- Still cold-starts to popularity model if CF and content-based return nothing
- Only as smart as the features you give it — if both CF and content-based miss a great movie, the reranker never sees it

---

## How They All Fit Together at Inference Time

When a user requests recommendations:

```
1. CF.recommend(user_id, top_k=50)        → up to 50 (movie_id, score) pairs
2. ContentBased.recommend(user_id, top_k=50) → up to 50 (movie_id, score) pairs
3. Union of both → deduplicated candidate pool (up to 100 movies)

4. For each candidate, build 7 features
5. LightGBM predicts a relevance score for each
6. Sort by score, return top 10

Edge case: if step 3 is empty (new user, no ratings) → fall back to Popularity.recommend()
```

---

## Why This Architecture

| Design choice | Reason |
|---|---|
| ALS instead of neural CF | Fast to train and serve, no GPU needed, proven on implicit feedback |
| Sentence transformers instead of TF-IDF | Captures semantics ("heist film" ≈ "robbery thriller"), not just keyword overlap |
| `all-MiniLM-L6-v2` specifically | 384 dims, fast on CPU, high quality for its size — good balance for 62K movies |
| LightGBM over neural ranker | Interpretable feature importance, fast training, no GPU needed, hard to overfit with tabular features |
| LambdaRank objective | Directly optimizes what we care about (NDCG) instead of a proxy metric |
| Two-stage retrieval + reranking | Industry-standard pattern: retrieval is O(n) fast, reranking is O(candidates) and can afford to be smarter |
| `--skip-cf` flag | CF takes the longest to train. Once trained it doesn't need to change unless the rating data changes significantly |
