from __future__ import annotations

from datetime import datetime, timedelta, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.core.config import settings
from app.ml.features import build_feature_row, feature_frame, normalize_crowd_level


NUMERIC_FEATURES = [
    "temperature",
    "humidity",
    "air_quality_index",
    "traffic_level",
    "people_count",
    "vehicle_count",
    "hour",
    "day_of_week",
]
CATEGORICAL_FEATURES = ["zone", "motion_level", "crowd_level"]


def _preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def generate_training_data(samples_per_zone: int = 180) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    rows: list[dict] = []
    temperatures: list[float] = []
    crowd_levels: list[str] = []
    base_time = datetime.now(timezone.utc) - timedelta(days=21)
    rng = np.random.default_rng(42)

    for zone_index, zone in enumerate(settings.zones):
        for offset in range(samples_per_zone):
            when = base_time + timedelta(hours=offset)
            daily_wave = np.sin((when.hour - 6) / 24 * 2 * np.pi)
            commute = max(0, 9 - abs(17 - when.hour)) + max(0, 5 - abs(8 - when.hour))
            temp = 24 + zone_index * 1.3 + daily_wave * 7 + rng.normal(0, 1.2)
            humidity = int(np.clip(55 - daily_wave * 12 + rng.normal(0, 5), 25, 85))
            aqi = int(np.clip(65 + zone_index * 13 + commute * 3 + rng.normal(0, 8), 20, 180))
            traffic = int(np.clip(30 + zone_index * 8 + commute * 6 + rng.normal(0, 7), 0, 100))
            people = int(np.clip(18 + zone_index * 9 + commute * 5 + rng.normal(0, 10), 0, 140))
            vehicles = int(np.clip(8 + zone_index * 5 + commute * 3 + rng.normal(0, 6), 0, 90))
            motion = "high" if commute > 8 else "medium" if commute > 3 else "low"
            current_crowd = normalize_crowd_level(None, people)

            row = build_feature_row(
                zone,
                {
                    "zone": zone,
                    "temperature": round(temp, 1),
                    "humidity": humidity,
                    "air_quality_index": aqi,
                    "traffic_level": traffic,
                    "timestamp": when.isoformat(),
                },
                {
                    "zone": zone,
                    "people_count": people,
                    "vehicle_count": vehicles,
                    "motion_level": motion,
                    "crowd_level": current_crowd,
                    "timestamp": when.isoformat(),
                },
                now=when,
            )
            rows.append(row)

            next_temp = temp + np.sin((when.hour + 1) / 24 * 2 * np.pi) * 1.2 + rng.normal(0, 0.6)
            next_people = people + commute * 2 + rng.normal(0, 8)
            temperatures.append(round(float(next_temp), 2))
            crowd_levels.append(normalize_crowd_level(None, int(next_people)))

    return feature_frame(rows), pd.Series(temperatures), pd.Series(crowd_levels)


def train_and_save() -> dict:
    features, temperature_target, crowd_target = generate_training_data()
    temperature_model = Pipeline(
        steps=[
            ("preprocessor", _preprocessor()),
            ("regressor", RandomForestRegressor(n_estimators=80, random_state=42)),
        ]
    )
    crowd_model = Pipeline(
        steps=[
            ("preprocessor", _preprocessor()),
            ("classifier", RandomForestClassifier(n_estimators=80, random_state=42)),
        ]
    )
    temperature_model.fit(features, temperature_target)
    crowd_model.fit(features, crowd_target)

    settings.model_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "training_samples": len(features),
        "zones": settings.zones,
    }
    bundle = {
        "temperature_model": temperature_model,
        "crowd_model": crowd_model,
        "metadata": metadata,
    }
    joblib.dump(bundle, settings.model_path)
    return bundle


def load_bundle() -> dict | None:
    if not settings.model_path.exists():
        return None
    return joblib.load(settings.model_path)
