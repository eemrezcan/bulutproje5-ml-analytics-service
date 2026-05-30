import httpx

from app.core.config import settings


class IoTClient:
    def __init__(self) -> None:
        self._base_url = settings.iot_service_url.rstrip("/")

    async def latest_readings(self) -> list[dict]:
        async with httpx.AsyncClient(timeout=settings.request_timeout_seconds) as client:
            response = await client.get(f"{self._base_url}/readings/latest")
            response.raise_for_status()
            return response.json()

    async def zone_readings(self, zone: str, limit: int = 50) -> list[dict]:
        async with httpx.AsyncClient(timeout=settings.request_timeout_seconds) as client:
            response = await client.get(
                f"{self._base_url}/zones/{zone}/readings",
                params={"limit": limit},
            )
            response.raise_for_status()
            return response.json()
