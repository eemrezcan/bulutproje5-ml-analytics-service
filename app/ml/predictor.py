from __future__ import annotations

from app.ml.features import feature_frame, normalize_crowd_level


class Predictor:
    def __init__(self, bundle: dict) -> None:
        self._temperature_model = bundle["temperature_model"]
        self._crowd_model = bundle["crowd_model"]
        self.metadata = bundle.get("metadata", {})

    def predict(self, row: dict) -> tuple[float, str]:
        frame = feature_frame([row])
        predicted_temperature = float(self._temperature_model.predict(frame)[0])
        predicted_crowd = str(self._crowd_model.predict(frame)[0]).lower()
        return round(predicted_temperature, 1), normalize_crowd_level(predicted_crowd)
