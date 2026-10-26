from datetime import datetime
from types import SimpleNamespace

from bson import ObjectId
from fastapi.testclient import TestClient

from app import main
from app.config import Settings, get_settings
from app.security import create_access_token


def utc_datetime(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


class FakeCollection:
    def __init__(self, records):
        self.records = records

    def create_index(self, *args, **kwargs):
        return None

    def count_documents(self, query):
        return sum(matches(record, query) for record in self.records)

    def aggregate(self, pipeline):
        records = self.records
        for stage in pipeline:
            if "$match" in stage:
                records = [
                    record for record in records
                    if matches(record, stage["$match"])
                ]
            elif "$group" in stage:
                records = group_records(records, stage["$group"])
            elif "$sort" in stage:
                field, direction = next(iter(stage["$sort"].items()))
                records = sorted(
                    records,
                    key=lambda record: record[field],
                    reverse=direction < 0,
                )
        return records


def matches(record, query):
    for field, condition in query.items():
        if field == "$and":
            if not all(matches(record, item) for item in condition):
                return False
            continue
        if field == "$or":
            if not any(matches(record, item) for item in condition):
                return False
            continue

        value = record.get(field)
        if isinstance(condition, dict):
            if "$gte" in condition and (
                value is None or value < condition["$gte"]
            ):
                return False
            if "$lte" in condition and (
                value is None or value > condition["$lte"]
            ):
                return False
            if "$exists" in condition and (
                (field in record) != condition["$exists"]
            ):
                return False
        elif value != condition:
            return False
    return True


def resolve_group_key(record, expression):
    if isinstance(expression, str) and expression.startswith("$"):
        return record.get(expression[1:])
    if "$toLower" in expression:
        nested = expression["$toLower"]
        if "$ifNull" in nested:
            field, fallback = nested["$ifNull"]
            value = record.get(field[1:], fallback)
        else:
            value = nested
        return value.lower()
    if "$dateTrunc" in expression:
        config = expression["$dateTrunc"]
        timestamp = record[config["date"][1:]]
        if config["unit"] == "hour":
            bucket = timestamp.replace(minute=0, second=0, microsecond=0)
        else:
            bucket = timestamp.replace(
                hour=0,
                minute=0,
                second=0,
                microsecond=0,
            )
        return bucket.replace(tzinfo=None)
    raise AssertionError(f"Unsupported group expression: {expression}")


def group_records(records, group_spec):
    groups = {}
    for record in records:
        key = resolve_group_key(record, group_spec["_id"])
        groups.setdefault(key, []).append(record)

    output = []
    for key, members in groups.items():
        row = {"_id": key}
        for field, expression in group_spec.items():
            if field == "_id":
                continue
            if expression == {"$sum": 1}:
                row[field] = len(members)
            elif "$avg" in expression:
                values = [
                    member.get(expression["$avg"][1:])
                    for member in members
                ]
                values = [value for value in values if value is not None]
                row[field] = (
                    sum(values) / len(values) if values else None
                )
            else:
                raise AssertionError(f"Unsupported group accumulator: {expression}")
        output.append(row)
    return output


class FakeDatabase:
    def __init__(self):
        self.datasets = FakeCollection([
            {
                "id": "dataset-real",
                "created_at": utc_datetime("2026-10-25T10:00:00Z"),
            },
        ])
        self.predictions = FakeCollection([
            {
                "dataset_id": "dataset-real",
                "row_index": 0,
                "observed_at": utc_datetime("2026-10-25T12:05:00Z"),
                "prediction_type": "known_attack",
                "risk_score": 76,
                "severity": "high",
                "prediction_label": "port_scan",
                "model_version": "baseline-1",
                "feature_schema_version": "1.0.0",
                "prediction_timestamp": utc_datetime("2026-10-25T12:05:01Z"),
                "created_at": utc_datetime("2026-10-25T12:05:01Z"),
                "updated_at": utc_datetime("2026-10-25T12:05:01Z"),
            },
        ])
        self.standalone_predictions = FakeCollection([
            {
                "prediction_batch_id": "batch-a",
                "row_index": 0,
                "observed_at": utc_datetime("2026-10-25T12:15:00Z"),
                "prediction_type": "benign",
                "risk_score": 8,
                "severity": "info",
                "prediction_label": "benign",
                "model_version": "baseline-1",
                "feature_schema_version": "1.0.0",
                "prediction_timestamp": utc_datetime("2026-10-25T12:15:01Z"),
                "created_at": utc_datetime("2026-10-25T12:15:01Z"),
                "updated_at": utc_datetime("2026-10-25T12:15:01Z"),
            },
            {
                "prediction_batch_id": "batch-b",
                "row_index": 0,
                "observed_at": utc_datetime("2026-10-25T13:05:00Z"),
                "prediction_type": "uncertain",
                "risk_score": None,
                "severity": "info",
                "prediction_label": "unknown",
                "model_version": "baseline-1",
                "feature_schema_version": "1.0.0",
                "prediction_timestamp": utc_datetime("2026-10-25T13:05:01Z"),
                "created_at": utc_datetime("2026-10-25T13:05:01Z"),
                "updated_at": utc_datetime("2026-10-25T13:05:01Z"),
            },
        ])
        self.alerts = FakeCollection([
            {
                "created_at": utc_datetime("2026-10-25T12:06:00Z"),
                "severity": "high",
                "status": "open",
            },
            {
                "created_at": utc_datetime("2026-10-25T13:06:00Z"),
                "severity": "critical",
                "status": "closed",
            },
        ])
        self.users = SimpleNamespace(
            find_one=lambda query, projection=None: {
                "_id": ObjectId(query["_id"]),
                "is_active": True,
            }
        )
        self.revoked_tokens = FakeCollection([])
        self.revoked_tokens.find_one = lambda query: None


class FakeMongoClient:
    def __init__(self):
        self.admin = SimpleNamespace(command=lambda command: None)
        self.database = FakeDatabase()

    def __getitem__(self, name):
        return self.database


def test_dashboard_summary_aggregates_persisted_shaped_records(monkeypatch):
    settings = Settings(jwt_secret_key="s" * 32)
    monkeypatch.setattr(main, "ensure_indexes", lambda db: None)
    app = main.create_app(settings, mongo_client=FakeMongoClient())
    app.dependency_overrides[get_settings] = lambda: settings
    token, _ = create_access_token(str(ObjectId()), "analyst", settings)

    with TestClient(app) as client:
        response = client.get(
            "/api/v1/dashboard/summary",
            params={
                "from": "2026-10-25T00:00:00Z",
                "to": "2026-10-26T00:00:00Z",
            },
            headers={"Authorization": f"Bearer {token}"},
        )

    assert response.status_code == 200
    summary = response.json()
    assert summary["totals"] == {
        "datasets": 1,
        "observations": 3,
        "alerts": 2,
        "open_alerts": 1,
    }
    assert summary["severity_counts"] == {
        "critical": 1,
        "high": 1,
        "medium": 0,
        "low": 0,
        "info": 0,
    }
    assert summary["prediction_type_counts"] == {
        "known_attack": 1,
        "benign": 1,
        "anomaly": 0,
        "uncertain": 1,
    }
    assert summary["risk_series"] == [
        {
            "bucket_start": "2026-10-25T12:00:00Z",
            "count": 2,
            "mean_risk_score": 42.0,
        },
    ]


def test_dataset_filter_excludes_standalone_predictions(monkeypatch):
    settings = Settings(jwt_secret_key="s" * 32)
    monkeypatch.setattr(main, "ensure_indexes", lambda db: None)
    app = main.create_app(settings, mongo_client=FakeMongoClient())
    app.dependency_overrides[get_settings] = lambda: settings
    token, _ = create_access_token(str(ObjectId()), "analyst", settings)

    with TestClient(app) as client:
        response = client.get(
            "/api/v1/dashboard/summary",
            params={
                "from": "2026-10-25T00:00:00Z",
                "to": "2026-10-26T00:00:00Z",
                "dataset_id": "dataset-real",
            },
            headers={"Authorization": f"Bearer {token}"},
        )

    assert response.status_code == 200
    summary = response.json()
    assert summary["totals"]["datasets"] == 1
    assert summary["totals"]["observations"] == 1
    assert summary["prediction_type_counts"] == {
        "known_attack": 1,
        "benign": 0,
        "anomaly": 0,
        "uncertain": 0,
    }
    assert summary["risk_series"] == [
        {
            "bucket_start": "2026-10-25T12:00:00Z",
            "count": 1,
            "mean_risk_score": 76.0,
        },
    ]
