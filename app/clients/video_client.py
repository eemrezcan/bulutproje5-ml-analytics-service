import httpx

from app.core.config import settings


class VideoClient:
    def __init__(self) -> None:
        self._base_url = settings.video_service_url.rstrip("/")

    async def latest_analysis(self) -> list[dict]:
        async with httpx.AsyncClient(timeout=settings.request_timeout_seconds) as client:
            response = await client.get(f"{self._base_url}/analysis/latest")
            response.raise_for_status()
            return response.json()

    async def zone_analysis(self, zone: str, limit: int = 50) -> list[dict]:
        async with httpx.AsyncClient(timeout=settings.request_timeout_seconds) as client:
            response = await client.get(
                f"{self._base_url}/zones/{zone}/analysis",
                params={"limit": limit},
            )
            response.raise_for_status()
            return response.json()
