# RecLab — Hybrid Movie Recommendation System

A full-stack movie recommendation platform built on MovieLens 25M (25M ratings, 62K movies). Combines collaborative filtering, content-based filtering, and a LightGBM reranker into a hybrid pipeline with a React frontend and FastAPI backend.

**Live site:** https://movie-recs-jet.vercel.app

## How it works

- **Onboard** — rate a handful of movies to bootstrap your taste profile
- **For You** — hybrid recommendations ranked by a learned reranker; toggle between Hybrid, Collaborative Filter, and Popularity Baseline views
- **Search** — browse and filter 62K movies by genre, decade, or title
- **Insights** — compare model performance (NDCG, Precision, Recall) and explore feature importance

## Stack

| Layer | Tech |
|---|---|
| Frontend | React + Vite, deployed on Vercel |
| Backend | FastAPI + uvicorn, deployed on Hugging Face Spaces (Docker) |
| Models | ALS collaborative filter, TF-IDF content model, LightGBM LambdaRank reranker |
| Data | MovieLens 25M |
