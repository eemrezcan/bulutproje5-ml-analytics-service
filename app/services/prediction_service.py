from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.clients.iot_client import IoTClient
from app.clients.video_client import VideoClient
from app.core.config import settings
from app.ml.features import build_feature_row, latest_by_zone
from app.ml.risk_engine import build_recommendation, calculate_risk_score, risk_level
from app.models.schemas import Prediction, PredictionInputSummary, ZoneRisk
from app.repositories.prediction_repository import PredictionRepository
from app.services.model_service import ModelService


class PredictionService:
    def __init__(
        self,
        repository: PredictionRepository,
        model_service: ModelService,
        iot_client: IoTClient,
        video_client: VideoClient,
    ) -> None:
        self._repository = repository
        self._model_service = model_service
        self._iot_client = iot_client
        self._video_client = video_client

    async def generate_all(self) -> list[Prediction]:
        iot_latest = await self._safe_iot_latest()
        video_latest = await self._safe_video_latest()
        iot_by_zone = latest_by_zone(iot_latest)
        video_by_zone = latest_by_zone(video_latest)
        now = datetime.now(timezone.utc)

        predictions = [
            self._build_prediction(
                zone=zone,
                iot_reading=iot_by_zone.get(zone),
                video_event=video_by_zone.get(zone),
                now=now,
            )
            for zone in settings.zones
        ]
        return await self._repository.put_many(predictions)

    async def latest(self) -> list[Prediction]:
        latest = await self._repository.latest(settings.zones)
        if latest:
            return latest
        return await self.generate_all()

    async def list_by_zone(self, zone: str, limit: int = 50) -> list[Prediction]:
        return await self._repository.list_by_zone(zone, limit)

    async def zone_risk(self, zone: str) -> ZoneRisk:
        predictions = await self._repository.list_by_zone(zone, 1)
        if not predictions:
            await self.generate_all()
            predictions = await self._repository.list_by_zone(zone, 1)
        prediction = predictions[0]
        return ZoneRisk(
            zone=prediction.zone,
            risk_score=prediction.risk_score,
            risk_level=prediction.risk_level,
            recommendation=prediction.recommendation,
            prediction_timestamp=prediction.prediction_timestamp,
            prediction_for=prediction.prediction_for,
        )

    async def _safe_iot_latest(self) -> list[dict]:
        try:
            return await self._iot_client.latest_readings()
        except Exception:
            return []

    async def _safe_video_latest(self) -> list[dict]:
        try:
            return await self._video_client.latest_analysis()
        except Exception:
            return []

    def _build_prediction(
        self,
        *,
        zone: str,
        iot_reading: dict | None,
        video_event: dict | None,
        now: datetime,
    ) -> Prediction:
        row = build_feature_row(zone, iot_reading, video_event, now)
        predicted_temperature, predicted_crowd_level = self._model_service.predictor.predict(
            row
        )
        score = calculate_risk_score(
            predicted_temperature=predicted_temperature,
            predicted_crowd_level=predicted_crowd_level,
            air_quality_index=row["air_quality_index"],
            traffic_level=row["traffic_level"],
            people_count=row["people_count"],
            vehicle_count=row["vehicle_count"],
        )
        return Prediction(
            zone=zone,
            prediction_timestamp=now.isoformat().replace("+00:00", "Z"),
            prediction_for=(now + timedelta(minutes=settings.prediction_horizon_minutes))
            .isoformat()
            .replace("+00:00", "Z"),
            predicted_temperature=predicted_temperature,
            predicted_crowd_level=predicted_crowd_level,
            risk_score=score,
            risk_level=risk_level(score),
            recommendation=build_recommendation(zone, score, predicted_crowd_level),
            input_summary=PredictionInputSummary(
                temperature=row["temperature"],
                humidity=row["humidity"],
                air_quality_index=row["air_quality_index"],
                traffic_level=row["traffic_level"],
                people_count=row["people_count"],
                vehicle_count=row["vehicle_count"],
                motion_level=row["motion_level"],
                crowd_level=row["crowd_level"],
            ),
        )
