from datetime import timezone
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from bson import ObjectId
from pymongo.errors import PyMongoError

from app.config import Settings
from app.security import (
    hash_password,
    verify_password,
    create_access_token,
    decode_access_token,
    current_user,
    require_roles,
)


def test_password_hash_round_trip():
    hashed = hash_password("not-a-real-password")
    assert hashed != "not-a-real-password"
    assert verify_password(hashed, "not-a-real-password")
    assert not verify_password(hashed, "wrong")

def test_jwt_claims_and_expiration():
    settings = Settings(jwt_secret_key="s"*32, jwt_access_token_minutes=1)
    token, ttl = create_access_token("user123", "analyst", settings)
    claims = decode_access_token(token, settings)
    assert claims["sub"] == "user123"
    assert claims["role"] == "analyst"
    assert ttl == 60

def test_invalid_jwt_rejected():
    settings = Settings(jwt_secret_key="s"*32)
    try:
        decode_access_token("not.a.token", settings)
    except Exception as exc:
        assert getattr(exc, "status_code", None) == 401
    else:
        assert False, "Invalid token should be rejected"


def test_require_roles_returns_claims_for_allowed_role():
    claims = {"sub": "actor-id", "role": "analyst"}
    dependency = require_roles("admin", "analyst")
    request = SimpleNamespace()

    assert dependency(request=request, claims=claims) is claims


def test_require_roles_audits_denied_role_and_returns_forbidden():
    audit_events = Mock()
    request = SimpleNamespace(
        method="POST",
        url=SimpleNamespace(path="/api/v1/predictions"),
        app=SimpleNamespace(
            state=SimpleNamespace(
                db=SimpleNamespace(audit_events=audit_events)
            )
        ),
    )
    claims = {"sub": "actor-id", "role": "viewer"}
    dependency = require_roles("admin", "analyst")

    with pytest.raises(HTTPException) as exc_info:
        dependency(request=request, claims=claims)

    assert exc_info.value.status_code == 403
    assert exc_info.value.detail == "Insufficient permissions."
    event = audit_events.insert_one.call_args.args[0]
    assert event["created_at"].tzinfo == timezone.utc
    assert event["actor_subject_id"] == "actor-id"
    assert event["actor_role"] == "viewer"
    assert event["action"] == "authorization.denied"
    assert event["request_method"] == "POST"
    assert event["request_path"] == "/api/v1/predictions"
    assert event["denial_reason"] == "role_not_allowed"


@pytest.mark.parametrize(
    "roles",
    [
        (),
        ("owner",),
        ("analyst", "owner"),
        (None,),
    ],
)
def test_require_roles_rejects_invalid_configuration(roles):
    with pytest.raises(ValueError):
        require_roles(*roles)


def test_require_roles_still_denies_when_audit_insert_fails(caplog):
    audit_events = Mock()
    audit_events.insert_one.side_effect = PyMongoError("database unavailable")
    request = SimpleNamespace(
        method="POST",
        url=SimpleNamespace(path="/api/v1/predictions"),
        app=SimpleNamespace(
            state=SimpleNamespace(
                db=SimpleNamespace(audit_events=audit_events)
            )
        ),
    )
    dependency = require_roles("analyst")

    with pytest.raises(HTTPException) as exc_info:
        dependency(
            request=request,
            claims={"sub": "actor-id", "role": "viewer"},
        )

    assert exc_info.value.status_code == 403
    assert "Failed to persist authorization denial audit event" in caplog.text
    assert "database unavailable" not in caplog.text


def test_require_roles_still_denies_when_audit_insert_raises_runtime_error(
    caplog,
):
    audit_events = Mock()
    audit_events.insert_one.side_effect = RuntimeError(
        "sensitive storage diagnostic"
    )
    request = SimpleNamespace(
        method="POST",
        url=SimpleNamespace(path="/api/v1/predictions"),
        app=SimpleNamespace(
            state=SimpleNamespace(
                db=SimpleNamespace(audit_events=audit_events)
            )
        ),
    )
    dependency = require_roles("analyst")

    with pytest.raises(HTTPException) as exc_info:
        dependency(
            request=request,
            claims={"sub": "actor-id", "role": "viewer"},
        )

    assert exc_info.value.status_code == 403
    assert exc_info.value.detail == "Insufficient permissions."
    assert "Failed to persist authorization denial audit event" in caplog.text
    assert "sensitive storage diagnostic" not in caplog.text


@pytest.mark.parametrize(
    "database",
    [
        None,
        SimpleNamespace(),
        SimpleNamespace(audit_events=None),
    ],
)
def test_require_roles_still_denies_when_audit_collection_is_missing(
    database,
    caplog,
):
    request = SimpleNamespace(
        method="POST",
        url=SimpleNamespace(path="/api/v1/predictions"),
        app=SimpleNamespace(state=SimpleNamespace(db=database)),
    )
    dependency = require_roles("analyst")

    with pytest.raises(HTTPException) as exc_info:
        dependency(
            request=request,
            claims={"sub": "actor-id", "role": "viewer"},
        )

    assert exc_info.value.status_code == 403
    assert exc_info.value.detail == "Insufficient permissions."
    assert "Failed to persist authorization denial audit event" in caplog.text


def authenticated_request(
    *,
    token_role="analyst",
    user_active=True,
    user_role="analyst",
    revoked=False,
    user_lookup_error=None,
):
    settings = Settings(jwt_secret_key="s" * 32)
    user_id = ObjectId()
    token, _ = create_access_token(str(user_id), token_role, settings)
    revoked_tokens = SimpleNamespace(
        find_one=lambda query: {"jti": query["jti"]} if revoked else None
    )

    def find_user(query, projection):
        assert query == {"_id": user_id}
        assert projection == {"is_active": 1, "role": 1}
        if user_lookup_error is not None:
            raise user_lookup_error
        if not user_active:
            return None
        user = {"is_active": True}
        if user_role is not None:
            user["role"] = user_role
        return user

    database = SimpleNamespace(
        revoked_tokens=revoked_tokens,
        users=SimpleNamespace(find_one=find_user),
    )
    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(db=database))
    )
    credentials = HTTPAuthorizationCredentials(
        scheme="Bearer",
        credentials=token,
    )
    return request, credentials, settings


def test_current_user_accepts_active_user_with_matching_database_role():
    request, credentials, settings = authenticated_request(
        token_role="analyst",
        user_role="analyst",
    )

    claims = current_user(request, credentials, settings)

    assert claims["role"] == "analyst"


@pytest.mark.parametrize(
    ("token_role", "user_role"),
    [
        ("analyst", "viewer"),
        ("viewer", "admin"),
    ],
)
def test_current_user_rejects_changed_database_role(token_role, user_role):
    request, credentials, settings = authenticated_request(
        token_role=token_role,
        user_role=user_role,
    )

    with pytest.raises(HTTPException) as exc_info:
        current_user(request, credentials, settings)

    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == "User role has changed. Please log in again."


@pytest.mark.parametrize("user_role", [None, "operator", []])
def test_current_user_rejects_missing_or_invalid_database_role(user_role):
    request, credentials, settings = authenticated_request(
        user_role=user_role,
    )

    with pytest.raises(HTTPException) as exc_info:
        current_user(request, credentials, settings)

    assert exc_info.value.status_code == 401


def test_current_user_still_rejects_inactive_user():
    request, credentials, settings = authenticated_request(
        user_active=False,
    )

    with pytest.raises(HTTPException) as exc_info:
        current_user(request, credentials, settings)

    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == "Invalid or inactive credentials."


def test_current_user_still_rejects_revoked_token():
    request, credentials, settings = authenticated_request(revoked=True)

    with pytest.raises(HTTPException) as exc_info:
        current_user(request, credentials, settings)

    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == "Token has been revoked. Please log in again."


def test_current_user_returns_service_unavailable_on_user_lookup_failure():
    request, credentials, settings = authenticated_request(
        user_lookup_error=PyMongoError("database unavailable"),
    )

    with pytest.raises(HTTPException) as exc_info:
        current_user(request, credentials, settings)

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == "Unable to verify user status."
