import asyncio
import time
from decimal import Decimal
from typing import Any

import boto3
from botocore.exceptions import ClientError, EndpointConnectionError

from app.core.config import settings


def _resource():
    kwargs: dict[str, Any] = {"region_name": settings.aws_region}
    if settings.dynamodb_endpoint_url:
        kwargs["endpoint_url"] = settings.dynamodb_endpoint_url
    return boto3.resource("dynamodb", **kwargs)


dynamodb = _resource()


def table(name: str):
    return dynamodb.Table(name)


def to_dynamodb(value: Any) -> Any:
    if isinstance(value, float):
        return Decimal(str(value))
    if isinstance(value, dict):
        return {key: to_dynamodb(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_dynamodb(item) for item in value]
    return value


def from_dynamodb(value: Any) -> Any:
    if isinstance(value, Decimal):
        if value % 1 == 0:
            return int(value)
        return float(value)
    if isinstance(value, dict):
        return {key: from_dynamodb(item) for key, item in value.items()}
    if isinstance(value, list):
        return [from_dynamodb(item) for item in value]
    return value


async def wait_until_ready() -> None:
    deadline = time.monotonic() + settings.dynamodb_ready_timeout_seconds
    last_error: Exception | None = None

    while time.monotonic() < deadline:
        try:
            await asyncio.to_thread(list, dynamodb.tables.all())
            return
        except (EndpointConnectionError, ClientError) as exc:
            last_error = exc
            await asyncio.sleep(1)

    raise RuntimeError("DynamoDB is not ready") from last_error


def _existing_table_names() -> set[str]:
    return {item.name for item in dynamodb.tables.all()}


async def create_tables() -> None:
    await asyncio.to_thread(_create_tables_sync)


def _create_tables_sync() -> None:
    if settings.predictions_table in _existing_table_names():
        return

    dynamodb.create_table(
        TableName=settings.predictions_table,
        KeySchema=[
            {"AttributeName": "zone", "KeyType": "HASH"},
            {"AttributeName": "prediction_timestamp", "KeyType": "RANGE"},
        ],
        AttributeDefinitions=[
            {"AttributeName": "zone", "AttributeType": "S"},
            {"AttributeName": "prediction_timestamp", "AttributeType": "S"},
        ],
        BillingMode="PAY_PER_REQUEST",
    ).wait_until_exists()
