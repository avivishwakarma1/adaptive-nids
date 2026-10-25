
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LoginRequest(StrictModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=128)


class UserPublic(BaseModel):
    id: str
    email: EmailStr
    display_name: str
    role: Literal["admin", "analyst", "viewer"]
    is_active: bool
    created_at: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"
    expires_in: int
    user: UserPublic


class PredictionRequest(StrictModel):
    observations: list[dict[str, Any]] = Field(
        min_length=1,
        max_length=1000,
        description=(
            "Canonical NIDS flow records validated by nids_ml feature schema 1.0.0. "
            "Required fields: duration, protocol, source_port, destination_port, "
            "flow_packets, flow_bytes, packets_per_second, bytes_per_second, "
            "connection_state, direction, and observed_at. source_id is optional. "
            "Unknown fields and invalid values are rejected by the ML validator."
        ),
    )


class ErrorDetail(BaseModel):
    field: str
    issue: str


class ErrorBody(BaseModel):
    code: str
    message: str
    details: list[ErrorDetail] | None
    request_id: str


class ErrorEnvelope(BaseModel):
    error: ErrorBody


# Dashboard response models

class DashboardRange(BaseModel):
    from_: str = Field(alias="from")
    to: str


class DashboardTotals(BaseModel):
    datasets: int = Field(ge=0)
    observations: int = Field(ge=0)
    alerts: int = Field(ge=0)
    open_alerts: int = Field(ge=0)


class DashboardSeverityCounts(BaseModel):
    critical: int = Field(ge=0)
    high: int = Field(ge=0)
    medium: int = Field(ge=0)
    low: int = Field(ge=0)
    info: int = Field(ge=0)


class DashboardPredictionTypeCounts(BaseModel):
    known_attack: int = Field(ge=0)
    benign: int = Field(ge=0)
    anomaly: int = Field(ge=0)
    uncertain: int = Field(ge=0)


class DashboardRiskPoint(BaseModel):
    bucket_start: str
    count: int = Field(ge=0)
    mean_risk_score: float | None = Field(default=None, ge=0, le=100)


class DashboardSummaryResponse(BaseModel):
    range: DashboardRange
    totals: DashboardTotals
    severity_counts: DashboardSeverityCounts
    prediction_type_counts: DashboardPredictionTypeCounts
    risk_series: list[DashboardRiskPoint]
    generated_at: str
