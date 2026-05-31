import os
from pathlib import Path


class Settings:
    service_name: str = os.getenv("SERVICE_NAME", "ml-analytics-service")
    aws_region: str = os.getenv("AWS_REGION", "eu-central-1")
    dynamodb_endpoint_url: str | None = os.getenv("DYNAMODB_ENDPOINT_URL")
    predictions_table: str = os.getenv("ML_PREDICTIONS_TABLE", "ml_predictions")
    iot_service_url: str = os.getenv("IOT_SERVICE_URL", "http://localhost:8001")
    video_service_url: str = os.getenv("VIDEO_SERVICE_URL", "http://localhost:8002")
    request_timeout_seconds: float = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "5"))
    dynamodb_ready_timeout_seconds: int = int(
        os.getenv("DYNAMODB_READY_TIMEOUT_SECONDS", "60")
    )
    prediction_horizon_minutes: int = int(os.getenv("PREDICTION_HORIZON_MINUTES", "60"))
    app_dir: Path = Path(__file__).resolve().parents[1]
    model_dir: Path = Path(os.getenv("MODEL_DIR", str(app_dir / "artifacts")))
    model_path: Path = model_dir / "ml_models.joblib"
    zones: list[str] = [
        zone.strip()
        for zone in os.getenv(
            "ML_ZONES", "Meydan,Otogar,Kampus,Hastane,Sanayi"
        ).split(",")
        if zone.strip()
    ]
    cors_origins: list[str] = [
        origin.strip()
        for origin in os.getenv(
            "CORS_ORIGINS",
            "http://localhost:5173,http://127.0.0.1:5173",
        ).split(",")
        if origin.strip()
    ]


settings = Settings()
