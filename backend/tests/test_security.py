from datetime import datetime, timezone, timedelta
import jwt
from app.config import Settings
from app.security import hash_password, verify_password, create_access_token, decode_access_token

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
