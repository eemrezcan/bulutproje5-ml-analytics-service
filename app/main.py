import os

from fastapi import FastAPI

app = FastAPI(title="ML Analytics Service", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "service": os.getenv("SERVICE_NAME", "ml-analytics-service"),
        "status": "ok",
    }

