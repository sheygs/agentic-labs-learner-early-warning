"""Rank day-14 snapshots for a mentor's outreach queue."""

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import joblib
import sklearn
from fastapi import FastAPI, Request
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator
from threadpoolctl import threadpool_limits

from data import FEATURES, HORIZON_DAYS, OBSERVATION_DAYS, engagement_features

LearnerID = Annotated[
    str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
]
DailyCount = Annotated[int, Field(strict=True, ge=0, le=1_000_000)]


class LearnerActivity(BaseModel):
    model_config = ConfigDict(extra="forbid")
    learner_id: LearnerID
    daily_clicks: Annotated[
        list[DailyCount],
        Field(
            min_length=OBSERVATION_DAYS,
            max_length=OBSERVATION_DAYS,
            description="Ordered per-day click totals for course days 0–13. Use zero for days without clicks.",
        ),
    ]


class CohortRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    learners: Annotated[list[LearnerActivity], Field(min_length=1, max_length=5000)]

    @model_validator(mode="after")
    def unique_learner_ids(self):
        if len({learner.learner_id for learner in self.learners}) != len(self.learners):
            raise ValueError("Learner IDs must be unique within a request.")
        return self


class RankedLearner(BaseModel):
    learner_id: str
    rank: int
    risk_score: float


class CohortPrediction(BaseModel):
    model_name: str
    model_version: str
    horizon_days: int
    ranked_learners: list[RankedLearner]


@asynccontextmanager
async def lifespan(app):
    artifact_path = Path(os.environ.get("MODEL_PATH", "artifacts/model.joblib"))
    artifact = joblib.load(artifact_path)
    metadata = artifact["metadata"]
    if metadata["scikit_learn_version"] != sklearn.__version__:
        raise ValueError(
            "Install the pinned dependencies or retrain the model with the installed scikit-learn."
        )
    if (
        metadata["feature_names"] != FEATURES
        or metadata["observation_days"] != OBSERVATION_DAYS
    ):
        raise ValueError(
            "The model artifact uses a different engagement feature schema."
        )
    if metadata["prediction_horizon_days"] != HORIZON_DAYS:
        raise ValueError("The model artifact uses a different prediction horizon.")
    app.state.model = artifact["model"]
    app.state.metadata = metadata
    # Keep a single thread limit for the process while synchronous request workers score.
    with threadpool_limits(limits=2):
        yield


app = FastAPI(
    title="Learner Early Warning",
    description="Prioritise mentor check-ins using the first 14 days of recorded learning activity.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health(request: Request):
    metadata = request.app.state.metadata
    return {
        "status": "ok",
        "model_name": metadata["model_name"],
        "model_version": metadata["model_version"],
    }


@app.post("/predict", response_model=CohortPrediction)
def predict(payload: CohortRequest, request: Request):
    features = engagement_features(
        [learner.daily_clicks for learner in payload.learners]
    )
    scores = request.app.state.model.predict_proba(features)[:, 1]
    scored = sorted(
        zip((learner.learner_id for learner in payload.learners), scores, strict=True),
        key=lambda item: (-item[1], item[0]),
    )
    metadata = request.app.state.metadata
    return CohortPrediction(
        model_name=metadata["model_name"],
        model_version=metadata["model_version"],
        horizon_days=HORIZON_DAYS,
        ranked_learners=[
            RankedLearner(learner_id=learner_id, rank=rank, risk_score=float(score))
            for rank, (learner_id, score) in enumerate(scored, start=1)
        ],
    )
