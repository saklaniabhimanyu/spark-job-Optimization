"""
FastAPI service exposing the trained Spark-config recommendation model.

Run locally:
    uvicorn api.main:app --reload --port 8000

Then open http://localhost:8000 for the demo UI, or POST to /recommend.
Requires a trained model at models/best_model.joblib — run
`python main.py` or the pipeline notebook first (see README).
"""

import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.schemas import (
    HealthResponse, ModelInfoResponse, RecommendationResponse, WorkloadProfile,
)
from src.config import CANDIDATE_CONFIGS, FEATURES, MODELS_DIR
from src.model_training import load_model
from src.recommendation_engine import recommend_configs

app = FastAPI(
    title="Spark Job Optimizer API",
    description="Predicts Spark job execution time and recommends the "
                 "fastest configuration for a given workload profile.",
    version="0.1.0",
)

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

_model = None
_model_error = None


@app.on_event("startup")
def _load_model_on_startup():
    global _model, _model_error
    try:
        _model = load_model("best_model.joblib")
    except FileNotFoundError:
        _model_error = (
            "No trained model found at models/best_model.joblib. "
            "Run `python main.py` or the pipeline notebook first."
        )


@app.get("/", include_in_schema=False)
def serve_index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", model_loaded=_model is not None)


@app.get("/model/info", response_model=ModelInfoResponse)
def model_info():
    if _model is None:
        raise HTTPException(status_code=503, detail=_model_error)
    return ModelInfoResponse(
        model_type=type(_model).__name__,
        features=FEATURES,
        n_candidate_configs=len(CANDIDATE_CONFIGS),
    )


@app.post("/recommend", response_model=RecommendationResponse)
def recommend(profile: WorkloadProfile):
    if _model is None:
        raise HTTPException(status_code=503, detail=_model_error)

    workload = {
        "num_records": profile.num_records, "skewness": profile.skewness,
        "joins": profile.joins, "group_cardinality": profile.group_cardinality,
        "num_aggregations": profile.num_aggregations, "has_orderby": profile.has_orderby,
        "filter_selectivity": profile.filter_selectivity,
    }
    df_top = recommend_configs(_model, workload, top_n=profile.top_n)

    return RecommendationResponse(
        workload=profile,
        recommendations=df_top.to_dict(orient="records"),
    )
