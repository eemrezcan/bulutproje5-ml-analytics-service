from app.models.schemas import RiskLevel


CROWD_WEIGHTS = {"low": 5, "medium": 18, "high": 32, "critical": 45}


def risk_level(score: int) -> RiskLevel:
    if score >= 80:
        return "critical"
    if score >= 55:
        return "warning"
    return "normal"


def calculate_risk_score(
    *,
    predicted_temperature: float,
    predicted_crowd_level: str,
    air_quality_index: int,
    traffic_level: int,
    people_count: int,
    vehicle_count: int,
) -> int:
    score = 8
    score += CROWD_WEIGHTS.get(predicted_crowd_level.lower(), 10)
    score += min(20, max(0, int((predicted_temperature - 24) * 2)))
    score += min(18, max(0, int((air_quality_index - 60) / 4)))
    score += min(16, max(0, int(traffic_level / 6)))
    score += min(10, int(people_count / 12))
    score += min(8, int(vehicle_count / 10))
    return max(0, min(100, score))


def build_recommendation(zone: str, score: int, predicted_crowd_level: str) -> str:
    level = risk_level(score)
    if level == "critical":
        return (
            f"{zone} bolgesinde kritik risk bekleniyor. Kalabalik, trafik ve saha "
            "ekipleri hemen hazir konuma alinmali."
        )
    if level == "warning":
        if predicted_crowd_level in {"high", "critical"}:
            return (
                f"{zone} bolgesinde kalabalik ve sicaklik artisi bekleniyor. "
                "Trafik ve saha ekipleri hazir tutulmali."
            )
        return (
            f"{zone} bolgesinde orta seviye risk bekleniyor. Sensor ve kamera "
            "akislari yakin takip edilmeli."
        )
    return f"{zone} bolgesinde risk normal seviyede. Rutin izleme yeterli gorunuyor."
