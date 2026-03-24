"""
FastAPI backend — stub.

Loads trained model artifacts and serves recommendation endpoints.
Full implementation in Phase 8.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Hybrid Rec Platform", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


# Routes added in Phase 8:
# from backend.routes import recommendations, search, insights
# app.include_router(recommendations.router)
# app.include_router(search.router)
# app.include_router(insights.router)
