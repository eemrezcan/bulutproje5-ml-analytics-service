import asyncio
from typing import Any

from boto3.dynamodb.conditions import Key

from app.core.config import settings
from app.models.schemas import Prediction
from app.repositories.dynamodb import from_dynamodb, table, to_dynamodb


class PredictionRepository:
    def __init__(self) -> None:
        self._table = table(settings.predictions_table)

    async def put(self, prediction: Prediction) -> Prediction:
        await asyncio.to_thread(
            self._table.put_item, Item=to_dynamodb(prediction.model_dump())
        )
        return prediction

    async def put_many(self, predictions: list[Prediction]) -> list[Prediction]:
        await asyncio.gather(*(self.put(prediction) for prediction in predictions))
        return predictions

    async def list_by_zone(self, zone: str, limit: int = 50) -> list[Prediction]:
        return await asyncio.to_thread(self._list_by_zone_sync, zone, limit)

    def _list_by_zone_sync(self, zone: str, limit: int) -> list[Prediction]:
        response = self._table.query(
            KeyConditionExpression=Key("zone").eq(zone),
            ScanIndexForward=False,
            Limit=limit,
        )
        return self._from_items(response.get("Items", []))

    async def latest(self, zones: list[str]) -> list[Prediction]:
        results = await asyncio.gather(*(self.list_by_zone(zone, 1) for zone in zones))
        latest = [items[0] for items in results if items]
        return sorted(latest, key=lambda item: item.prediction_timestamp, reverse=True)

    def _from_items(self, items: list[dict[str, Any]]) -> list[Prediction]:
        return [Prediction(**from_dynamodb(item)) for item in items]
