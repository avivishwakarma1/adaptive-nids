import logging
from datetime import datetime, timedelta, timezone
import jwt
from argon2 import PasswordHasher
from argon2.exceptions import VerificationError, VerifyMismatchError, InvalidHashError
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from .config import Settings, get_settings
from uuid import uuid4
from fastapi import Depends, HTTPException, Request
from pymongo.errors import PyMongoError

logger = logging.getLogger("adaptive_nids.security")
_hasher = PasswordHasher()
_bearer = HTTPBearer(auto_error=False)
ROLES = {"admin", "analyst", "viewer"}

def hash_password(password: str) -> str:
    return _hasher.hash(password)

def verify_password(password_hash: str, password: str) -> bool:
    try:
        return _hasher.verify(password_hash, password)
    except (VerificationError, VerifyMismatchError, InvalidHashError):
        return False


def create_access_token(
    user_id: str,
    role: str,
    settings: Settings,
    now=None,
) -> tuple[str, int]:
    if not settings.jwt_secret_key:
        raise RuntimeError("JWT secret is not configured")

    if role not in ROLES:
        raise ValueError("Invalid role")

    now = now or datetime.now(timezone.utc)
    ttl = settings.jwt_access_token_minutes * 60

    payload = {
        "sub": user_id,
        "role": role,
        "jti": uuid4().hex,
        "iat": now,
        "exp": now + timedelta(seconds=ttl),
        "iss": settings.jwt_issuer,
        "aud": settings.jwt_audience,
    }

    token = jwt.encode(
        payload,
        settings.jwt_secret_key,
        algorithm="HS256",
    )
    return token, ttl


def decode_access_token(token: str, settings: Settings) -> dict:
    if not settings.jwt_secret_key:
        raise HTTPException(
            status_code=503,
            detail="Authentication is not configured.",
        )

    try:
        claims = jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=["HS256"],
            issuer=settings.jwt_issuer,
            audience=settings.jwt_audience,
            options={
                "require": [
                    "sub", "role", "jti",
                    "iat", "exp", "iss", "aud",
                ]
            },
        )

        if claims["role"] not in ROLES:
            raise jwt.InvalidTokenError("invalid role")

        return claims

    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired credentials.",
        )


def current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    settings: Settings = Depends(get_settings),
):
    if credentials is None:
        raise HTTPException(
            status_code=401,
            detail="Authentication required.",
        )

    claims = decode_access_token(credentials.credentials, settings)

    db = getattr(request.app.state, "db", None)

    if db is None:
        raise HTTPException(
            status_code=503,
            detail="Authentication service is temporarily unavailable.",
        )

    try:
        revoked = db.revoked_tokens.find_one({"jti": claims["jti"]})
    except PyMongoError:
        logger.exception(
            "MongoDB error while checking authentication token revocation"
        )
        raise HTTPException(
            status_code=503,
            detail="Unable to verify token status.",
        )

    if revoked:
        raise HTTPException(
            status_code=401,
            detail="Token has been revoked. Please log in again.",
        )

    try:
        from bson import ObjectId

        user = db.users.find_one(
            {"_id": ObjectId(claims["sub"])},
            {"is_active": 1, "role": 1},
        )
    except (ValueError, TypeError):
        user = None
    except PyMongoError:
        logger.exception(
            "MongoDB error while checking authenticated user status"
        )
        raise HTTPException(
            status_code=503,
            detail="Unable to verify user status.",
        )

    if not user or not user.get("is_active", False):
        raise HTTPException(
            status_code=401,
            detail="Invalid or inactive credentials.",
        )

    user_role = user.get("role")
    if not isinstance(user_role, str) or user_role not in ROLES:
        raise HTTPException(
            status_code=401,
            detail="User role has changed. Please log in again.",
        )
    if user_role != claims["role"]:
        raise HTTPException(
            status_code=401,
            detail="User role has changed. Please log in again.",
        )

    return claims


def require_roles(*allowed_roles: str):
    if not allowed_roles or any(
        not isinstance(role, str) or role not in ROLES
        for role in allowed_roles
    ):
        raise ValueError("At least one valid role must be allowed.")

    def role_dependency(
        request: Request,
        claims: dict = Depends(current_user),
    ) -> dict:
        if claims["role"] not in allowed_roles:
            event = {
                "created_at": datetime.now(timezone.utc),
                "actor_subject_id": claims["sub"],
                "actor_role": claims["role"],
                "action": "authorization.denied",
                "request_method": request.method,
                "request_path": request.url.path,
                "denial_reason": "role_not_allowed",
            }
            database = getattr(request.app.state, "db", None)
            audit_events = (
                getattr(database, "audit_events", None)
                if database is not None
                else None
            )
            insert_audit_event = (
                getattr(audit_events, "insert_one", None)
                if audit_events is not None
                else None
            )

            if not callable(insert_audit_event):
                logger.error(
                    "Failed to persist authorization denial audit event"
                )
            else:
                try:
                    insert_audit_event(event)
                except Exception:
                    logger.error(
                        "Failed to persist authorization denial audit event"
                    )

            raise HTTPException(
                status_code=403,
                detail="Insufficient permissions.",
            )
        return claims

    return role_dependency
