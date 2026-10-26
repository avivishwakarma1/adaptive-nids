
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from uuid import uuid4
import logging

from fastapi import (
    FastAPI,
    Request,
    Depends,
    HTTPException,
    Query,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from bson import ObjectId
from pymongo.errors import ConfigurationError, OperationFailure, PyMongoError
from starlette.exceptions import HTTPException as StarletteHTTPException

from .config import Settings, get_settings
from .db import create_mongo_client, ensure_indexes, ping_database
from .errors import (
    error_payload,
    http_exception_handler,
    validation_exception_handler,
    unhandled_exception_handler,
)
from .ml_adapter import load_active_predictor
from nids_ml import (
    InputValidationError,
    ModelUnavailableError,
    PredictionResult,
    SchemaVersionError,
)
from .schemas import (
    LoginRequest,
    LoginResponse,
    PredictionRequest,
    ErrorEnvelope,
    DashboardSummaryResponse,
)
from .security import verify_password, create_access_token, current_user

logger = logging.getLogger("adaptive_nids")


class PredictionBodyTooLarge(Exception):
    pass


class InvalidPredictionOutputError(Exception):
    pass


def utc_now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_utc_datetime(value: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError("timestamp must be a string")

    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timestamp must include a UTC offset")
    return parsed.astimezone(timezone.utc)


def build_prediction_documents(observations, results):
    if not isinstance(results, list) or len(results) != len(observations):
        raise InvalidPredictionOutputError

    results_by_row = {}
    for result in results:
        if not isinstance(result, PredictionResult):
            raise InvalidPredictionOutputError
        row_index = result.row_index
        if (
            type(row_index) is not int
            or row_index < 0
            or row_index >= len(observations)
            or row_index in results_by_row
        ):
            raise InvalidPredictionOutputError
        results_by_row[row_index] = result

    if set(results_by_row) != set(range(len(observations))):
        raise InvalidPredictionOutputError

    batch_id = uuid4().hex
    created_at = datetime.now(timezone.utc)
    prediction_documents = []
    alert_documents = []

    for row_index, observation in enumerate(observations):
        result = results_by_row[row_index]
        try:
            observed_at = parse_utc_datetime(observation["observed_at"])
            prediction_timestamp = parse_utc_datetime(
                result.prediction_timestamp
            )
        except (KeyError, TypeError, ValueError):
            raise InvalidPredictionOutputError from None

        prediction_id = ObjectId()
        prediction_documents.append({
            "_id": prediction_id,
            "prediction_batch_id": batch_id,
            "row_index": row_index,
            "observed_at": observed_at,
            "prediction_label": result.prediction_label,
            "prediction_type": result.prediction_type,
            "confidence": result.confidence,
            "anomaly_score": result.anomaly_score,
            "risk_score": result.risk_score,
            "severity": result.severity,
            "reason_codes": result.reason_codes,
            "model_version": result.model_version,
            "feature_schema_version": result.feature_schema_version,
            "prediction_timestamp": prediction_timestamp,
            "created_at": created_at,
            "updated_at": created_at,
        })

        if (
            result.prediction_type == "known_attack"
            and result.risk_score >= 60
            and result.severity in {"high", "critical"}
        ):
            alert_documents.append({
                "_id": ObjectId(),
                "prediction_batch_id": batch_id,
                "prediction_id": prediction_id,
                "row_index": row_index,
                "prediction_label": result.prediction_label,
                "prediction_type": result.prediction_type,
                "risk_score": result.risk_score,
                "created_at": created_at,
                "severity": result.severity,
                "status": "open",
            })

    return batch_id, prediction_documents, alert_documents


def _insert_prediction_batch(db, prediction_documents, alert_documents, session=None):
    options = {"ordered": True}
    if session is not None:
        options["session"] = session
    if prediction_documents:
        db.standalone_predictions.insert_many(
            prediction_documents,
            **options,
        )
    if alert_documents:
        db.alerts.insert_many(alert_documents, **options)


def _persist_without_transaction(
    db,
    batch_id,
    prediction_documents,
    alert_documents,
):
    try:
        _insert_prediction_batch(
            db,
            prediction_documents,
            alert_documents,
        )
    except PyMongoError:
        try:
            db.standalone_predictions.delete_many({
                "prediction_batch_id": batch_id
            })
            db.alerts.delete_many({"prediction_batch_id": batch_id})
            if (
                db.standalone_predictions.count_documents({
                    "prediction_batch_id": batch_id
                })
                or db.alerts.count_documents({"prediction_batch_id": batch_id})
            ):
                logger.error(
                    "Compensating cleanup left records for prediction batch %s.",
                    batch_id,
                )
        except PyMongoError:
            logger.error(
                "Compensating cleanup failed for prediction batch %s; "
                "partial records may remain.",
                batch_id,
                exc_info=True,
            )
        raise


def persist_prediction_batch(
    db,
    batch_id,
    prediction_documents,
    alert_documents,
):
    try:
        with db.client.start_session() as session:
            with session.start_transaction():
                _insert_prediction_batch(
                    db,
                    prediction_documents,
                    alert_documents,
                    session,
                )
        return
    except ConfigurationError:
        logger.info(
            "MongoDB sessions are unavailable; using compensating writes "
            "for prediction batch %s.",
            batch_id,
        )
    except OperationFailure as exc:
        if exc.code != 20 or "transaction" not in str(exc).lower():
            raise
        logger.info(
            "MongoDB transactions are unsupported; using compensating writes "
            "for prediction batch %s.",
            batch_id,
        )

    _persist_without_transaction(
        db,
        batch_id,
        prediction_documents,
        alert_documents,
    )


class PredictionRequestSizeLimitMiddleware:
    def __init__(self, app, *, path: str, max_bytes: int):
        self.app = app
        self.path = path
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send):
        if (
            scope["type"] != "http"
            or scope["method"] != "POST"
            or scope["path"] != self.path
        ):
            await self.app(scope, receive, send)
            return

        state = scope.setdefault("state", {})
        request_id = state.get("request_id", "unknown")
        headers = dict(scope.get("headers", []))
        content_length = headers.get(b"content-length")
        if content_length:
            try:
                if int(content_length) > self.max_bytes:
                    await self._too_large(scope, receive, send, request_id)
                    return
            except ValueError:
                pass

        body_size = 0

        async def limited_receive():
            nonlocal body_size
            message = await receive()
            if message["type"] == "http.request":
                body_size += len(message.get("body", b""))
                if body_size > self.max_bytes:
                    raise PredictionBodyTooLarge
            return message

        try:
            await self.app(scope, limited_receive, send)
        except PredictionBodyTooLarge:
            await self._too_large(scope, receive, send, request_id)

    async def _too_large(self, scope, receive, send, request_id: str):
        response = JSONResponse(
            status_code=413,
            content=error_payload_for_request_id(
                "PAYLOAD_TOO_LARGE",
                "Prediction request exceeds the configured size limit.",
                request_id,
            ),
        )
        await response(scope, receive, send)


def error_payload_for_request_id(code: str, message: str, request_id: str):
    return {
        "error": {
            "code": code,
            "message": message,
            "details": None,
            "request_id": request_id,
        }
    }


def create_app(settings: Settings | None = None, mongo_client=None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        client = (
            mongo_client
            if mongo_client is not None
            else create_mongo_client(settings.mongodb_uri)
        )

        app.state.mongo_client = client
        app.state.db = (
            client[settings.mongodb_database]
            if client is not None
            else None
        )
        app.state.predictor = None

        if client is not None:
            try:
                client.admin.command("ping")
                ensure_indexes(app.state.db)

                app.state.db.revoked_tokens.create_index(
                    "jti",
                    unique=True,
                )
                app.state.db.revoked_tokens.create_index(
                    "expires_at",
                    expireAfterSeconds=0,
                )
            except PyMongoError:
                logger.warning(
                    "Database unavailable during startup; "
                    "service will remain degraded."
                )

        if app.state.db is not None:
            try:
                app.state.predictor = load_active_predictor(
                    app.state.db,
                    settings.model_artifact_dir,
                    settings.active_model_version,
                )
            except (PyMongoError, ModelUnavailableError) as exc:
                app.state.predictor = None
                logger.warning("Active model unavailable during startup: %s", exc)

        yield

        if client is not None and client is not mongo_client:
            client.close()

    app = FastAPI(
        title="Adaptive AI-Driven NIDS API",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
    )
    app.add_middleware(
        PredictionRequestSizeLimitMiddleware,
        path=settings.nids_api_prefix + "/predictions",
        max_bytes=settings.max_upload_mb * 1024 * 1024,
    )

    app.add_exception_handler(
        StarletteHTTPException, http_exception_handler
    )
    app.add_exception_handler(
        RequestValidationError, validation_exception_handler
    )
    app.add_exception_handler(Exception, unhandled_exception_handler)

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        supplied = request.headers.get("X-Request-ID", "")
        request.state.request_id = (
            supplied[:100]
            if supplied and supplied.isprintable()
            else "req_" + uuid4().hex
        )

        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        return response

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "service": "adaptive-ai-nids-api",
            "api_version": "1.0.0",
            "timestamp": utc_now(),
        }

    @app.get("/ready")
    def ready(request: Request):
        db_ok = ping_database(
            getattr(request.app.state, "mongo_client", None)
        )
        model_loaded = getattr(request.app.state, "predictor", None) is not None
        model_status = "ok" if model_loaded else (
            "degraded" if settings.allow_degraded_model else "unavailable"
        )

        if not db_ok or (not model_loaded and not settings.allow_degraded_model):
            raise HTTPException(
                status_code=503,
                detail="Required dependencies are unavailable.",
            )

        return {
            "status": "ready",
            "dependencies": {
                "database": "ok",
                "model": model_status,
            },
            "timestamp": utc_now(),
        }

    @app.post(
        settings.nids_api_prefix + "/predictions",
        response_model=list[PredictionResult],
        responses={
            401: {"model": ErrorEnvelope, "description": "Authentication required or invalid."},
            413: {"model": ErrorEnvelope, "description": "Prediction request exceeds the configured body-size limit."},
            422: {"model": ErrorEnvelope, "description": "Request or ML feature validation failed."},
            503: {"model": ErrorEnvelope, "description": "Prediction or authentication service is unavailable."},
        },
    )
    def create_predictions(
        payload: PredictionRequest,
        request: Request,
        claims=Depends(current_user),
    ):
        predictor = getattr(request.app.state, "predictor", None)
        if predictor is None:
            raise HTTPException(
                status_code=503,
                detail="Prediction service is temporarily unavailable.",
            )

        try:
            results = predictor.predict(payload.observations)
        except (InputValidationError, SchemaVersionError) as exc:
            details = []
            for issue in getattr(exc, "issues", []):
                field_parts = ["observations"]
                if issue.row_index is not None:
                    field_parts.append(str(issue.row_index))
                if issue.field:
                    field_parts.append(issue.field)
                details.append({
                    "field": ".".join(field_parts),
                    "issue": issue.message,
                })
            if not details:
                details.append({
                    "field": "observations",
                    "issue": "Input does not satisfy the ML feature contract.",
                })
            return JSONResponse(
                status_code=422,
                content=error_payload(
                    request,
                    "VALIDATION_ERROR",
                    "Prediction input failed validation.",
                    details,
                ),
            )
        except ModelUnavailableError:
            logger.exception("Prediction model unavailable during inference")
            raise HTTPException(
                status_code=503,
                detail="Prediction service is temporarily unavailable.",
            )
        except Exception:
            logger.exception("Prediction inference failed")
            raise HTTPException(
                status_code=503,
                detail="Prediction service is temporarily unavailable.",
            )

        try:
            batch_id, prediction_documents, alert_documents = (
                build_prediction_documents(payload.observations, results)
            )
        except InvalidPredictionOutputError:
            logger.error(
                "Prediction inference returned results that do not map "
                "one-to-one to submitted observations."
            )
            return JSONResponse(
                status_code=503,
                content=error_payload(
                    request,
                    "INVALID_PREDICTION_OUTPUT",
                    "Prediction service is temporarily unavailable.",
                ),
            )

        db = getattr(request.app.state, "db", None)
        if db is None:
            return JSONResponse(
                status_code=503,
                content=error_payload(
                    request,
                    "PERSISTENCE_UNAVAILABLE",
                    "Prediction results could not be saved.",
                ),
            )

        try:
            persist_prediction_batch(
                db,
                batch_id,
                prediction_documents,
                alert_documents,
            )
        except PyMongoError:
            logger.error(
                "Prediction persistence failed for batch %s.",
                batch_id,
                exc_info=True,
            )
            return JSONResponse(
                status_code=503,
                content=error_payload(
                    request,
                    "PERSISTENCE_UNAVAILABLE",
                    "Prediction results could not be saved.",
                ),
            )

        return results

    @app.post(
        settings.nids_api_prefix + "/auth/login",
        response_model=LoginResponse,
    )
    def login(payload: LoginRequest, request: Request):
        db = getattr(request.app.state, "db", None)

        if db is None:
            raise HTTPException(
                status_code=503,
                detail="Authentication service is temporarily unavailable.",
            )

        try:
            user = db.users.find_one({
                "email_normalized": str(payload.email).strip().lower()
            })
        except PyMongoError:
            raise HTTPException(
                status_code=503,
                detail="Authentication service is temporarily unavailable.",
            )

        if (
            not user
            or not user.get("is_active", False)
            or not verify_password(
                user.get("password_hash", ""),
                payload.password,
            )
        ):
            raise HTTPException(
                status_code=401,
                detail="Invalid email or password.",
            )

        try:
            token, ttl = create_access_token(
                str(user["_id"]),
                user["role"],
                settings,
            )
        except RuntimeError:
            raise HTTPException(
                status_code=503,
                detail="Authentication is not configured.",
            )

        try:
            db.users.update_one(
                {"_id": user["_id"]},
                {"$set": {"last_login_at": datetime.now(timezone.utc)}},
            )
        except PyMongoError:
            raise HTTPException(
                status_code=503,
                detail="Authentication service is temporarily unavailable.",
            )

        created_at = user["created_at"]
        if hasattr(created_at, "isoformat"):
            created_at = created_at.isoformat().replace("+00:00", "Z")
        else:
            created_at = str(created_at)

        return {
            "access_token": token,
            "expires_in": ttl,
            "user": {
                "id": str(user["_id"]),
                "email": user["email"],
                "display_name": user["display_name"],
                "role": user["role"],
                "is_active": user["is_active"],
                "created_at": created_at,
            },
        }

    @app.post(
        settings.nids_api_prefix + "/auth/logout",
        status_code=204,
    )
    def logout(request: Request, claims=Depends(current_user)):
        db = getattr(request.app.state, "db", None)

        if db is None:
            raise HTTPException(
                status_code=503,
                detail="Authentication service is temporarily unavailable.",
            )

        try:
            expires_at = datetime.fromtimestamp(
                claims["exp"],
                tz=timezone.utc,
            )

            db.revoked_tokens.update_one(
                {"jti": claims["jti"]},
                {
                    "$setOnInsert": {
                        "jti": claims["jti"],
                        "expires_at": expires_at,
                        "revoked_at": datetime.now(timezone.utc),
                    }
                },
                upsert=True,
            )
        except PyMongoError:
            raise HTTPException(
                status_code=503,
                detail="Unable to complete logout.",
            )

        return None

    @app.get(settings.nids_api_prefix + "/auth/me")
    def me(
        request: Request,
        claims=Depends(current_user),
    ):
        db = getattr(request.app.state, "db", None)

        if db is None:
            raise HTTPException(
                status_code=503,
                detail="User service is temporarily unavailable.",
            )

        try:
            from bson import ObjectId

            user = db.users.find_one({
                "_id": ObjectId(claims["sub"])
            })
        except (ValueError, TypeError):
            user = None
        except PyMongoError:
            raise HTTPException(
                status_code=503,
                detail="User service is temporarily unavailable.",
            )

        if not user or not user.get("is_active"):
            raise HTTPException(
                status_code=401,
                detail="Invalid or inactive credentials.",
            )

        created_at = user["created_at"]
        if hasattr(created_at, "isoformat"):
            created_at = created_at.isoformat().replace("+00:00", "Z")
        else:
            created_at = str(created_at)

        return {
            "id": str(user["_id"]),
            "email": user["email"],
            "display_name": user["display_name"],
            "role": user["role"],
            "is_active": user["is_active"],
            "created_at": created_at,
        }

    @app.get(
        settings.nids_api_prefix + "/dashboard/summary",
        response_model=DashboardSummaryResponse,
    )
    def dashboard_summary(
        request: Request,
        from_: datetime | None = Query(default=None, alias="from"),
        to: datetime | None = Query(default=None),
        dataset_id: str | None = Query(
            default=None, min_length=1, max_length=128
        ),
        claims=Depends(current_user),
    ):
        db = getattr(request.app.state, "db", None)

        if db is None:
            raise HTTPException(
                status_code=503,
                detail="Dashboard data is temporarily unavailable.",
            )

        now = datetime.now(timezone.utc)

        start = (
            from_
            if from_ is not None
            else (
                to - timedelta(hours=24)
                if to is not None
                else now - timedelta(hours=24)
            )
        )
        end = to or now

        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)

        start = start.astimezone(timezone.utc)
        end = end.astimezone(timezone.utc)

        if start > end:
            raise HTTPException(
                status_code=422,
                detail="'from' must be earlier than or equal to 'to'.",
            )

        def iso_utc(value: datetime) -> str:
            if value.tzinfo is None or value.utcoffset() is None:
                value = value.replace(tzinfo=timezone.utc)
            return value.astimezone(timezone.utc).isoformat().replace(
                "+00:00", "Z"
            )

        severity_keys = ["critical", "high", "medium", "low", "info"]
        prediction_keys = [
            "known_attack",
            "benign",
            "anomaly",
            "uncertain",
        ]

        def zero_counts(keys):
            return {key: 0 for key in keys}

        dataset_filter = {}
        dataset_record_filter = {}

        if dataset_id:
            dataset_filter["dataset_id"] = dataset_id
            dataset_record_filter = {"id": dataset_id}

            from bson import ObjectId

            if ObjectId.is_valid(dataset_id):
                dataset_record_filter = {
                    "$or": [
                        {"_id": ObjectId(dataset_id)},
                        {"id": dataset_id},
                    ]
                }

        dataset_date_filter = {
            "created_at": {"$gte": start, "$lte": end}
        }

        if dataset_id:
            dataset_date_filter = {
                "$and": [
                    dataset_date_filter,
                    dataset_record_filter,
                ]
            }

        prediction_filter = {
            **dataset_filter,
            "observed_at": {"$gte": start, "$lte": end},
        }
        alert_filter = {
            **dataset_filter,
            "created_at": {"$gte": start, "$lte": end},
        }

        try:
            totals = {
                "datasets": int(
                    db.datasets.count_documents(dataset_date_filter)
                ),
                "observations": int(
                    db.predictions.count_documents(prediction_filter)
                    + (
                        db.standalone_predictions.count_documents({
                            "observed_at": {
                                "$gte": start,
                                "$lte": end,
                            }
                        })
                        if not dataset_id
                        else 0
                    )
                ),
                "alerts": int(
                    db.alerts.count_documents(alert_filter)
                ),
                "open_alerts": int(
                    db.alerts.count_documents({
                        **alert_filter,
                        "status": "open",
                    })
                ),
            }

            severity_counts = zero_counts(severity_keys)

            for row in db.alerts.aggregate([
                {"$match": alert_filter},
                {
                    "$group": {
                        "_id": {
                            "$toLower": {
                                "$ifNull": ["$severity", ""]
                            }
                        },
                        "count": {"$sum": 1},
                    }
                },
            ]):
                key = row["_id"]
                if key in severity_counts:
                    severity_counts[key] = int(row["count"])

            prediction_type_counts = zero_counts(prediction_keys)

            prediction_sources = [(db.predictions, prediction_filter)]
            if not dataset_id:
                prediction_sources.append((
                    db.standalone_predictions,
                    {"observed_at": {"$gte": start, "$lte": end}},
                ))

            for collection, collection_filter in prediction_sources:
                for row in collection.aggregate([
                    {"$match": collection_filter},
                    {
                        "$group": {
                            "_id": "$prediction_type",
                            "count": {"$sum": 1},
                        }
                    },
                ]):
                    key = row["_id"]
                    if key in prediction_type_counts:
                        prediction_type_counts[key] += int(row["count"])

            bucket_unit = (
                "hour"
                if end - start <= timedelta(hours=48)
                else "day"
            )

            grouped_risk = {}
            for collection, collection_filter in prediction_sources:
                for row in collection.aggregate([
                    {
                        "$match": {
                            **collection_filter,
                            "risk_score": {"$gte": 0, "$lte": 100},
                        }
                    },
                    {
                        "$group": {
                            "_id": {
                                "$dateTrunc": {
                                    "date": "$observed_at",
                                    "unit": bucket_unit,
                                    "timezone": "UTC",
                                }
                            },
                            "count": {"$sum": 1},
                            "mean_risk_score": {"$avg": "$risk_score"},
                        }
                    },
                    {"$sort": {"_id": 1}},
                ]):
                    group = grouped_risk.setdefault(
                        row["_id"],
                        {"count": 0, "weighted_risk": 0.0},
                    )
                    row_count = int(row["count"])
                    group["count"] += row_count
                    if row.get("mean_risk_score") is not None:
                        group["weighted_risk"] += (
                            float(row["mean_risk_score"]) * row_count
                        )

            risk_series = [
                {
                    "_id": bucket_start,
                    "count": values["count"],
                    "mean_risk_score": (
                        values["weighted_risk"] / values["count"]
                        if values["count"]
                        else None
                    ),
                }
                for bucket_start, values in sorted(grouped_risk.items())
            ]

        except PyMongoError:
            logger.exception("Dashboard summary database query failed")
            raise HTTPException(
                status_code=503,
                detail="Dashboard data is temporarily unavailable.",
            )

        return {
            "range": {
                "from": iso_utc(start),
                "to": iso_utc(end),
            },
            "totals": totals,
            "severity_counts": severity_counts,
            "prediction_type_counts": prediction_type_counts,
            "risk_series": [
                {
                    "bucket_start": iso_utc(row["_id"]),
                    "count": int(row["count"]),
                    "mean_risk_score": (
                        round(float(row["mean_risk_score"]), 2)
                        if row.get("mean_risk_score") is not None
                        else None
                    ),
                }
                for row in risk_series
            ],
            "generated_at": utc_now(),
        }

    return app


app = create_app()
