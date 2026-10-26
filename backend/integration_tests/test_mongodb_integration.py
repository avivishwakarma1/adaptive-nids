from datetime import datetime, timedelta, timezone

import pytest
from bson import ObjectId
from nids_ml import PredictionResult
from nids_ml import severity_for_score
from pymongo.collection import Collection
from pymongo.errors import PyMongoError


def observation(observed_at):
    return {
        "duration": 0.5,
        "protocol": "tcp",
        "source_port": 12345,
        "destination_port": 443,
        "flow_packets": 10,
        "flow_bytes": 500,
        "packets_per_second": 2,
        "bytes_per_second": 100,
        "connection_state": "established",
        "direction": "outbound",
        "observed_at": observed_at,
    }


def prediction(
    row_index,
    prediction_type="known_attack",
    risk_score=76,
    severity=None,
    timestamp="2026-10-25T12:00:00Z",
):
    labels = {
        "known_attack": "port_scan",
        "benign": "benign",
        "anomaly": "unknown",
        "uncertain": "unknown",
    }
    return PredictionResult(
        row_index=row_index,
        prediction_label=labels[prediction_type],
        prediction_type=prediction_type,
        confidence=0.91,
        anomaly_score=0.73,
        risk_score=risk_score,
        severity=severity_for_score(risk_score) if severity is None else severity,
        reason_codes=[],
        model_version="integration-test-model",
        feature_schema_version="1.0.0",
        prediction_timestamp=timestamp,
    )


def post_predictions(integration_client, observations, predictions):
    integration_client["predictor"].results = predictions
    integration_client["verify"]()

    return integration_client["client"].post(
        "/api/v1/predictions",
        json={"observations": observations},
        headers={
            "Authorization": f"Bearer {integration_client['token']}",
        },
    )


def iso(value):
    return value.isoformat().replace("+00:00", "Z")


def test_real_prediction_batch_persists_rows_and_preserves_response(
    integration_client,
):
    observations = [
        observation("2020-01-01T12:00:00Z"),
        observation("2020-01-01T12:30:00+00:00"),
    ]
    results = [
        prediction(
            1,
            "benign",
            8,
            "info",
            "2020-01-01T12:30:01Z",
        ),
        prediction(0, timestamp="2020-01-01T12:00:01Z"),
    ]

    response = post_predictions(integration_client, observations, results)

    assert response.status_code == 200, response.text
    assert response.json() == [result.model_dump() for result in results]
    assert all(
        set(row) == set(PredictionResult.model_fields)
        for row in response.json()
    )

    stored = list(
        integration_client["database"].standalone_predictions.find({
            "observed_at": {
                "$gte": datetime(2020, 1, 1, tzinfo=timezone.utc),
                "$lt": datetime(2020, 1, 2, tzinfo=timezone.utc),
            }
        }).sort("row_index", 1)
    )
    assert [row["row_index"] for row in stored] == [0, 1]
    assert [row["prediction_type"] for row in stored] == [
        "known_attack",
        "benign",
    ]
    assert [row["observed_at"] for row in stored] == [
        datetime(2020, 1, 1, 12, 0, tzinfo=timezone.utc),
        datetime(2020, 1, 1, 12, 30, tzinfo=timezone.utc),
    ]
    assert len({row["prediction_batch_id"] for row in stored}) == 1


def test_alerts_follow_the_known_attack_threshold_rule(integration_client):
    cases = [
        ("known_attack", 60, "high"),
        ("known_attack", 80, "critical"),
        ("known_attack", 59, "medium"),
        ("known_attack", 40, "medium"),
        ("benign", 100, "critical"),
    ]
    observations = [
        observation(f"2021-01-01T12:{index:02d}:00Z")
        for index in range(len(cases))
    ]
    results = [
        prediction(
            index,
            prediction_type,
            risk_score,
            severity,
            f"2021-01-01T12:{index:02d}:01Z",
        )
        for index, (prediction_type, risk_score, severity)
        in enumerate(cases)
    ]

    response = post_predictions(integration_client, observations, results)

    assert response.status_code == 200, response.text
    database = integration_client["database"]
    prediction_rows = list(database.standalone_predictions.find({
        "observed_at": {
            "$gte": datetime(2021, 1, 1, tzinfo=timezone.utc),
            "$lt": datetime(2021, 1, 2, tzinfo=timezone.utc),
        }
    }))
    batch_ids = {row["prediction_batch_id"] for row in prediction_rows}
    assert len(batch_ids) == 1
    alerts = list(
        database.alerts.find({
            "prediction_batch_id": {"$in": list(batch_ids)},
        }).sort("row_index", 1)
    )
    assert [alert["row_index"] for alert in alerts] == [0, 1]
    assert [alert["severity"] for alert in alerts] == ["high", "critical"]
    assert {alert["status"] for alert in alerts} == {"open"}
    assert {alert["prediction_type"] for alert in alerts} == {"known_attack"}


def test_dashboard_aggregates_legacy_and_standalone_records(integration_client):
    now = datetime.now(timezone.utc)
    bucket_start = now.replace(
        minute=0,
        second=0,
        microsecond=0,
    ) - timedelta(hours=1)
    observed_at = bucket_start.replace(minute=10)
    observations = [
        observation(iso(observed_at)),
        observation(iso(observed_at.replace(minute=20))),
    ]
    results = [
        prediction(0, risk_score=80, timestamp=iso(observed_at)),
        prediction(
            1,
            "benign",
            20,
            "low",
            iso(observed_at.replace(minute=20)),
        ),
    ]
    response = post_predictions(integration_client, observations, results)
    assert response.status_code == 200, response.text

    database = integration_client["database"]
    integration_client["verify"]()
    database.datasets.insert_one({
        "id": "it-legacy-dataset",
        "created_at": observed_at,
    })
    database.predictions.insert_one({
        "dataset_id": "it-legacy-dataset",
        "row_index": 0,
        "observed_at": observed_at.replace(minute=30),
        "prediction_type": "anomaly",
        "risk_score": 50,
        "severity": "medium",
    })

    summary_response = integration_client["client"].get(
        "/api/v1/dashboard/summary",
        params={
            "from": iso(bucket_start),
            "to": iso(now + timedelta(minutes=1)),
        },
        headers={
            "Authorization": f"Bearer {integration_client['token']}",
        },
    )

    assert summary_response.status_code == 200, summary_response.text
    summary = summary_response.json()
    assert summary["totals"] == {
        "datasets": 1,
        "observations": 3,
        "alerts": 1,
        "open_alerts": 1,
    }
    assert summary["prediction_type_counts"] == {
        "known_attack": 1,
        "benign": 1,
        "anomaly": 1,
        "uncertain": 0,
    }
    assert summary["severity_counts"]["critical"] == 1
    assert summary["severity_counts"]["high"] == 0
    assert summary["risk_series"] == [{
        "bucket_start": iso(bucket_start),
        "count": 3,
        "mean_risk_score": 50.0,
    }]


def test_replica_set_transaction_rolls_back_partial_batch(
    integration_client,
    monkeypatch,
):
    if integration_client["mode"] != "replica_set":
        pytest.skip("Transaction rollback requires the isolated replica set.")

    original_insert_many = Collection.insert_many
    attempted_batch_ids = []
    prediction_batch_ids = []

    def fail_alert_insert(collection, documents, *args, **kwargs):
        if (
            collection.name == "standalone_predictions"
            and kwargs.get("session") is not None
        ):
            result = original_insert_many(
                collection,
                documents,
                *args,
                **kwargs,
            )
            prediction_batch_ids.extend(
                document["prediction_batch_id"] for document in documents
            )
            return result
        if collection.name == "alerts" and kwargs.get("session") is not None:
            assert prediction_batch_ids
            attempted_batch_ids.extend(
                document["prediction_batch_id"] for document in documents
            )
            raise PyMongoError("injected alert write failure")
        return original_insert_many(collection, documents, *args, **kwargs)

    monkeypatch.setattr(Collection, "insert_many", fail_alert_insert)
    response = post_predictions(
        integration_client,
        [observation("2022-01-01T12:00:00Z")],
        [prediction(0, timestamp="2022-01-01T12:00:01Z")],
    )

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "PERSISTENCE_UNAVAILABLE"
    assert len(prediction_batch_ids) == 1
    assert len(attempted_batch_ids) == 1
    assert attempted_batch_ids == prediction_batch_ids
    batch_id = attempted_batch_ids[0]
    database = integration_client["database"]
    assert database.standalone_predictions.count_documents({
        "prediction_batch_id": batch_id,
    }) == 0
    assert database.alerts.count_documents({
        "prediction_batch_id": batch_id,
    }) == 0


def test_standalone_fallback_cleans_partial_batch(
    integration_client,
    monkeypatch,
):
    if integration_client["mode"] != "standalone":
        pytest.skip("Compensating cleanup requires the isolated standalone.")

    original_insert_many = Collection.insert_many
    attempted_batch_ids = []
    rejected_transaction_batch_ids = []
    fallback_prediction_batch_ids = []

    def fail_alert_insert(collection, documents, *args, **kwargs):
        if (
            collection.name == "standalone_predictions"
            and kwargs.get("session") is not None
        ):
            try:
                return original_insert_many(
                    collection,
                    documents,
                    *args,
                    **kwargs,
                )
            except PyMongoError as exc:
                if (
                    getattr(exc, "code", None) == 20
                    and "transaction" in str(exc).lower()
                ):
                    rejected_transaction_batch_ids.extend(
                        document["prediction_batch_id"]
                        for document in documents
                    )
                raise
        if (
            collection.name == "standalone_predictions"
            and kwargs.get("session") is None
        ):
            result = original_insert_many(
                collection,
                documents,
                *args,
                **kwargs,
            )
            fallback_prediction_batch_ids.extend(
                document["prediction_batch_id"] for document in documents
            )
            return result
        if collection.name == "alerts" and kwargs.get("session") is None:
            assert fallback_prediction_batch_ids
            attempted_batch_ids.extend(
                document["prediction_batch_id"] for document in documents
            )
            raise PyMongoError("injected alert write failure")
        return original_insert_many(collection, documents, *args, **kwargs)

    monkeypatch.setattr(Collection, "insert_many", fail_alert_insert)
    response = post_predictions(
        integration_client,
        [observation("2023-01-01T12:00:00Z")],
        [prediction(0, timestamp="2023-01-01T12:00:01Z")],
    )

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "PERSISTENCE_UNAVAILABLE"
    assert len(rejected_transaction_batch_ids) == 1
    assert len(fallback_prediction_batch_ids) == 1
    assert len(attempted_batch_ids) == 1
    assert rejected_transaction_batch_ids == fallback_prediction_batch_ids
    assert attempted_batch_ids == fallback_prediction_batch_ids
    batch_id = attempted_batch_ids[0]
    database = integration_client["database"]
    assert database.standalone_predictions.count_documents({
        "prediction_batch_id": batch_id,
    }) == 0
    assert database.alerts.count_documents({
        "prediction_batch_id": batch_id,
    }) == 0
