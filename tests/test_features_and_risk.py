from datetime import datetime, timezone

from app.ml.features import build_feature_row
from app.ml.risk_engine import calculate_risk_score, risk_level
from app.models.schemas import Prediction


def test_risk_level_thresholds() -> None:
    assert risk_level(20) == "normal"
    assert risk_level(55) == "warning"
    assert risk_level(80) == "critical"


def test_risk_score_increases_for_hot_crowded_area() -> None:
    normal = calculate_risk_score(
        predicted_temperature=25,
        predicted_crowd_level="low",
        air_quality_index=55,
        traffic_level=20,
        people_count=12,
        vehicle_count=4,
    )
    critical = calculate_risk_score(
        predicted_temperature=38,
        predicted_crowd_level="critical",
        air_quality_index=150,
        traffic_level=95,
        people_count=130,
        vehicle_count=70,
    )
    assert normal < critical
    assert critical == 100


def test_feature_row_normalizes_service_payloads() -> None:
    row = build_feature_row(
        "Meydan",
        {
            "temperature": 31.2,
            "humidity": 44,
            "air_quality_index": 118,
            "traffic_level": 71,
            "timestamp": "2026-05-28T12:00:00Z",
        },
        {
            "people_count": 46,
            "vehicle_count": 22,
            "motion_level": "MEDIUM",
            "crowd_level": "MEDIUM",
            "timestamp": "2026-05-28T12:00:00Z",
        },
        now=datetime(2026, 5, 28, 12, 0, tzinfo=timezone.utc),
    )
    assert row["hour"] == 12
    assert row["day_of_week"] == 3
    assert row["motion_level"] == "medium"
    assert row["crowd_level"] == "medium"


def test_prediction_schema_accepts_expected_payload() -> None:
    prediction = Prediction(
        zone="Meydan",
        prediction_timestamp="2026-05-28T12:00:00Z",
        prediction_for="2026-05-28T12:30:00Z",
        predicted_temperature=34.2,
        predicted_crowd_level="high",
        risk_score=78,
        risk_level="warning",
        recommendation="Meydan bolgesinde kalabalik ve sicaklik artisi bekleniyor.",
        input_summary={
            "temperature": 32.1,
            "humidity": 42,
            "air_quality_index": 118,
            "traffic_level": 71,
            "people_count": 46,
            "vehicle_count": 22,
            "motion_level": "medium",
            "crowd_level": "medium",
        },
    )
    assert prediction.zone == "Meydan"
    assert prediction.input_summary.people_count == 46
