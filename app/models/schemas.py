from typing import Literal

from pydantic import BaseModel, Field


CrowdLevel = Literal["low", "medium", "high", "critical"]
RiskLevel = Literal["normal", "warning", "critical"]


class HealthResponse(BaseModel):
    service: str
    status: str
    model_ready: bool


class ModelStatus(BaseModel):
    trained: bool
    model_path: str
    trained_at: str | None = None
    training_samples: int = 0


class TrainResponse(ModelStatus):
    message: str


class PredictionInputSummary(BaseModel):
    temperature: float
    humidity: int
    air_quality_index: int
    traffic_level: int
    people_count: int
    vehicle_count: int
    motion_level: str
    crowd_level: CrowdLevel


class Prediction(BaseModel):
    zone: str
    prediction_timestamp: str
    prediction_for: str
    predicted_temperature: float
    predicted_crowd_level: CrowdLevel
    risk_score: int = Field(ge=0, le=100)
    risk_level: RiskLevel
    recommendation: str
    input_summary: PredictionInputSummary


class ZoneRisk(BaseModel):
    zone: str
    risk_score: int = Field(ge=0, le=100)
    risk_level: RiskLevel
    recommendation: str
    prediction_timestamp: str
    prediction_for: str
