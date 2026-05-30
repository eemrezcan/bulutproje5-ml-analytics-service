from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd

from app.core.config import settings


CROWD_LEVELS = ["low", "medium", "high", "critical"]


def parse_timestamp(value: str | None) -> datetime:
    if not value:
        return datetime.now(timezone.utc)
    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def normalize_crowd_level(value: str | None, people_count: int = 0) -> str:
    normalized = (value or "").lower()
    if normalized in CROWD_LEVELS:
        return normalized
    if people_count >= 100:
        return "critical"
    if people_count >= 65:
        return "high"
    if people_count >= 30:
        return "medium"
    return "low"


def latest_by_zone(items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for item in items:
        zone = item.get("zone")
        timestamp = parse_timestamp(item.get("timestamp"))
        if not zone:
            continue
        current = result.get(zone)
        if current is None or timestamp > parse_timestamp(current.get("timestamp")):
            result[zone] = item
    return result


def synthetic_iot(zone: str, when: datetime) -> dict[str, Any]:
    zone_index = settings.zones.index(zone) if zone in settings.zones else 0
    hour_wave = abs(12 - when.hour) / 12
    return {
        "zone": zone,
        "temperature": round(24 + zone_index * 1.4 + (1 - hour_wave) * 9, 1),
        "humidity": int(48 + zone_index * 3 + hour_wave * 12),
        "air_quality_index": int(62 + zone_index * 13 + (when.hour % 5) * 4),
        "traffic_level": int(35 + zone_index * 8 + max(0, 18 - abs(17 - when.hour)) * 2),
        "timestamp": when.isoformat(),
    }


def synthetic_video(zone: str, when: datetime) -> dict[str, Any]:
    zone_index = settings.zones.index(zone) if zone in settings.zones else 0
    activity = max(0, 18 - abs(17 - when.hour))
    people_count = int(18 + zone_index * 10 + activity * 3)
    vehicle_count = int(8 + zone_index * 5 + activity * 1.5)
    return {
        "zone": zone,
        "people_count": people_count,
        "vehicle_count": vehicle_count,
        "motion_level": "high" if activity > 12 else "medium" if activity > 6 else "low",
        "crowd_level": normalize_crowd_level(None, people_count),
        "timestamp": when.isoformat(),
    }


def build_feature_row(
    zone: str,
    iot_reading: dict[str, Any] | None,
    video_event: dict[str, Any] | None,
    now: datetime | None = None,
) -> dict[str, Any]:
    when = now or datetime.now(timezone.utc)
    iot = iot_reading or synthetic_iot(zone, when)
    video = video_event or synthetic_video(zone, when)
    reading_time = parse_timestamp(iot.get("timestamp") or video.get("timestamp"))
    people_count = int(video.get("people_count", 0))

    return {
        "zone": zone,
        "temperature": float(iot.get("temperature", 25.0)),
        "humidity": int(iot.get("humidity", 50)),
        "air_quality_index": int(iot.get("air_quality_index", 70)),
        "traffic_level": int(iot.get("traffic_level", 40)),
        "people_count": people_count,
        "vehicle_count": int(video.get("vehicle_count", 0)),
        "motion_level": str(video.get("motion_level", "low")).lower(),
        "crowd_level": normalize_crowd_level(video.get("crowd_level"), people_count),
        "hour": reading_time.hour,
        "day_of_week": reading_time.weekday(),
    }


def feature_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    columns = [
        "zone",
        "temperature",
        "humidity",
        "air_quality_index",
        "traffic_level",
        "people_count",
        "vehicle_count",
        "motion_level",
        "crowd_level",
        "hour",
        "day_of_week",
    ]
    return pd.DataFrame(rows, columns=columns)
