from typing import Annotated

from fastapi import APIRouter, HTTPException, Query

from app.core import lifecycle
from app.core.config import settings
from app.models.schemas import (
    HealthResponse,
    ModelStatus,
    Prediction,
    TrainResponse,
    ZoneRisk,
)

router = APIRouter()
Limit = Annotated[int, Query(ge=1, le=200)]


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        service=settings.service_name,
        status="ok",
        model_ready=lifecycle.model_service.status().trained,
    )


@router.get("/model/status", response_model=ModelStatus)
async def model_status() -> ModelStatus:
    return lifecycle.model_service.status()


@router.post("/model/train", response_model=TrainResponse)
async def train_model() -> TrainResponse:
    return await lifecycle.model_service.train()


@router.post("/predictions/generate", response_model=list[Prediction])
async def generate_predictions() -> list[Prediction]:
    return await lifecycle.prediction_service.generate_all()


@router.get("/predictions/latest", response_model=list[Prediction])
async def latest_predictions() -> list[Prediction]:
    return await lifecycle.prediction_service.latest()


@router.get("/predictions", response_model=list[Prediction])
async def predictions_by_zone(zone: str, limit: Limit = 50) -> list[Prediction]:
    return await lifecycle.prediction_service.list_by_zone(zone, limit)


@router.get("/zones/{zone}/risk", response_model=ZoneRisk)
async def zone_risk(zone: str) -> ZoneRisk:
    if zone not in settings.zones:
        raise HTTPException(status_code=404, detail="Zone not found")
    return await lifecycle.prediction_service.zone_risk(zone)
