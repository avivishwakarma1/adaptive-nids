from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

def error_payload(request: Request, code: str, message: str, details=None):
    return {"error": {"code": code, "message": message, "details": details, "request_id": getattr(request.state, "request_id", "unknown")}}

async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    mapping = {401:"UNAUTHENTICATED",403:"FORBIDDEN",404:"NOT_FOUND",409:"CONFLICT",413:"PAYLOAD_TOO_LARGE",415:"UNSUPPORTED_MEDIA_TYPE",429:"RATE_LIMITED",503:"MODEL_UNAVAILABLE"}
    code = mapping.get(exc.status_code, "INTERNAL_ERROR")
    message = exc.detail if isinstance(exc.detail, str) and exc.status_code < 500 else "The request could not be completed."
    return JSONResponse(status_code=exc.status_code, content=error_payload(request, code, message))

async def validation_exception_handler(request: Request, exc: RequestValidationError):
    details = [{"field":".".join(str(p) for p in e.get("loc", ())),"issue":e.get("msg","invalid value")} for e in exc.errors()]
    return JSONResponse(status_code=422, content=error_payload(request, "VALIDATION_ERROR", "Request validation failed.", details))

async def unhandled_exception_handler(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content=error_payload(request, "INTERNAL_ERROR", "An unexpected server error occurred."))
