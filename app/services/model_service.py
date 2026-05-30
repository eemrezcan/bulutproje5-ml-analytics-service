import asyncio

from app.core.config import settings
from app.ml.predictor import Predictor
from app.ml.trainer import load_bundle, train_and_save
from app.models.schemas import ModelStatus, TrainResponse


class ModelService:
    def __init__(self) -> None:
        self._predictor: Predictor | None = None

    @property
    def predictor(self) -> Predictor:
        if self._predictor is None:
            raise RuntimeError("Model is not ready")
        return self._predictor

    async def load_or_train(self) -> None:
        bundle = await asyncio.to_thread(load_bundle)
        if bundle is None:
            bundle = await asyncio.to_thread(train_and_save)
        self._predictor = Predictor(bundle)

    async def train(self) -> TrainResponse:
        bundle = await asyncio.to_thread(train_and_save)
        self._predictor = Predictor(bundle)
        status = self.status()
        return TrainResponse(**status.model_dump(), message="Synthetic model trained")

    def status(self) -> ModelStatus:
        metadata = self._predictor.metadata if self._predictor else {}
        return ModelStatus(
            trained=self._predictor is not None,
            model_path=str(settings.model_path),
            trained_at=metadata.get("trained_at"),
            training_samples=int(metadata.get("training_samples", 0)),
        )
