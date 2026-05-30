from app.clients.iot_client import IoTClient
from app.clients.video_client import VideoClient
from app.repositories.dynamodb import create_tables, wait_until_ready
from app.repositories.prediction_repository import PredictionRepository
from app.services.model_service import ModelService
from app.services.prediction_service import PredictionService

model_service = ModelService()
prediction_repository = PredictionRepository()
iot_client = IoTClient()
video_client = VideoClient()
prediction_service = PredictionService(
    prediction_repository,
    model_service,
    iot_client,
    video_client,
)


async def startup() -> None:
    await wait_until_ready()
    await create_tables()
    await model_service.load_or_train()
    await prediction_service.generate_all()


async def shutdown() -> None:
    return None
