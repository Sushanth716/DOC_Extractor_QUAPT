"""
Canonical error handling and API-010 exception mapping (Module M5).

Adheres to:
- PRD §7.2 API-010: Standardized error code envelope
- PRD §7.2 API-011 & SEC-012: No sensitive info or PII in error messages
- Preserves full compatibility with DocumentExtractorError hierarchy.
"""

from typing import Any, Dict, Optional
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from doc_extractor.exceptions import (
    ConfigurationError,
    ConflictError as IngestionConflictError,
    DocumentExtractorError,
    DocumentProcessingError,
    LLMExecutionError,
    MalwareDetectedError,
    ScannerUnavailableError,
    SecurityCheckFailedError,
    StorageError,
    UnsupportedFileError,
)

# Import approved M4 schema lifecycle exceptions for canonical HTTP mapping
try:
    from schema_lifecycle.exceptions import (
        SchemaNotApprovedError,
        SchemaNotFoundError,
        SchemaValidationError,
    )
except ImportError:  # pragma: no cover
    SchemaNotApprovedError = None
    SchemaNotFoundError = None
    SchemaValidationError = None


class ApiHttpError(DocumentExtractorError):
    """Base class for HTTP-aware API exceptions mapped to API-010 envelopes."""

    status_code: int = 500

    def __init__(
        self,
        message: str = "An unexpected error occurred.",
        code: str = "INTERNAL_ERROR",
        status_code: Optional[int] = None,
        request_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message=message, code=code, request_id=request_id, details=details)
        if status_code is not None:
            self.status_code = status_code


class AuthenticationRequiredError(ApiHttpError):
    """Raised when authentication credentials are missing or invalid (HTTP 401)."""

    status_code = 401

    def __init__(
        self,
        message: str = "Authentication credentials were not provided or are invalid.",
        code: str = "AUTHENTICATION_REQUIRED",
        request_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message=message, code=code, status_code=401, request_id=request_id, details=details)


class ForbiddenError(ApiHttpError):
    """Raised when caller lacks required authorization scopes or roles (HTTP 403)."""

    status_code = 403

    def __init__(
        self,
        message: str = "Principal lacks required permissions for this action.",
        code: str = "FORBIDDEN",
        request_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message=message, code=code, status_code=403, request_id=request_id, details=details)


class TenantAccessDeniedError(ForbiddenError):
    """Raised when caller attempts to access another tenant's resource (HTTP 403)."""

    def __init__(
        self,
        message: str = "Access to the requested tenant resource is forbidden.",
        code: str = "TENANT_ACCESS_DENIED",
        request_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message=message, code=code, request_id=request_id, details=details)


class NotFoundError(ApiHttpError):
    """Raised when the requested resource does not exist (HTTP 404)."""

    status_code = 404

    def __init__(
        self,
        message: str = "The requested resource was not found.",
        code: str = "DOCUMENT_NOT_FOUND",
        status_code: Optional[int] = None,
        request_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message=message, code=code, status_code=status_code or 404, request_id=request_id, details=details)


class TenantResourceNotFoundError(NotFoundError):
    """
    Raised when accessing a document or resource belonging to another tenant,
    concealing resource existence per PRD §19 (HTTP 404 DOCUMENT_NOT_FOUND).
    """

    def __init__(
        self,
        message: str = "Document not found.",
        code: str = "DOCUMENT_NOT_FOUND",
        request_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message=message, code=code, request_id=request_id, details=details)


class FileTooLargeError(ApiHttpError):
    """Raised when an uploaded document or payload exceeds configured size/page limits per PRD §19 (HTTP 413)."""

    status_code = 413

    def __init__(
        self,
        message: str = "The file is larger than the limit.",
        code: str = "FILE_TOO_LARGE",
        request_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message=message, code=code, status_code=413, request_id=request_id, details=details)


class ConflictError(ApiHttpError):
    """Raised when a state conflict or idempotency fingerprint mismatch occurs (HTTP 409)."""

    status_code = 409

    def __init__(
        self,
        message: str = "The requested operation conflicts with the current resource state.",
        code: str = "CONFLICT",
        request_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message=message, code=code, status_code=409, request_id=request_id, details=details)


class RateLimitExceededError(ApiHttpError):
    """Raised when client exceeds designated request rate limits per PRD §7.2 API-012 & §19 (HTTP 429)."""

    status_code = 429

    def __init__(
        self,
        message: str = "Too many requests. Try again shortly.",
        code: str = "RATE_LIMITED",
        retry_after_seconds: Optional[int] = None,
        request_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.retry_after_seconds = max(0, int(retry_after_seconds)) if retry_after_seconds is not None else None
        super().__init__(message=message, code=code, status_code=429, request_id=request_id, details=details)


class ApiValidationError(ApiHttpError):
    """Raised when client input fails parameter or schema validation (HTTP 422)."""

    status_code = 422

    def __init__(
        self,
        message: str = "Request parameter or payload validation failed.",
        code: str = "VALIDATION_ERROR",
        request_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message=message, code=code, status_code=422, request_id=request_id, details=details)


class InternalServerError(ApiHttpError):
    """Raised for unrecoverable server errors (HTTP 500)."""

    status_code = 500

    def __init__(
        self,
        message: str = "An unexpected internal error occurred.",
        code: str = "INTERNAL_ERROR",
        request_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message=message, code=code, status_code=500, request_id=request_id, details=details)


# Status code mapping for existing DocumentExtractorError hierarchy and PRD §19
_STATUS_CODE_MAP: Dict[Any, int] = {
    AuthenticationRequiredError: 401,
    ForbiddenError: 403,
    TenantAccessDeniedError: 403,
    NotFoundError: 404,
    TenantResourceNotFoundError: 404,
    ConflictError: 409,
    IngestionConflictError: 409,
    UnsupportedFileError: 400,          # PRD §19: 400 UNSUPPORTED_FILE
    SecurityCheckFailedError: 400,
    MalwareDetectedError: 400,          # PRD §19: 400 MALWARE_DETECTED
    FileTooLargeError: 413,             # PRD §19: 413 FILE_TOO_LARGE
    ScannerUnavailableError: 503,
    RateLimitExceededError: 429,        # PRD §19: 429 RATE_LIMITED
    ApiValidationError: 422,
    ConfigurationError: 500,
    StorageError: 500,
    LLMExecutionError: 502,
    DocumentProcessingError: 500,
}

if SchemaNotApprovedError is not None:
    _STATUS_CODE_MAP[SchemaNotApprovedError] = 409  # PRD §19: 409 SCHEMA_NOT_APPROVED
if SchemaNotFoundError is not None:
    _STATUS_CODE_MAP[SchemaNotFoundError] = 404
if SchemaValidationError is not None:
    _STATUS_CODE_MAP[SchemaValidationError] = 422


def get_status_code_for_exception(exc: Exception) -> int:
    """Derive HTTP status code from exception class hierarchy."""
    if hasattr(exc, "status_code") and isinstance(exc.status_code, int):
        return exc.status_code
    for exc_cls, code in _STATUS_CODE_MAP.items():
        if isinstance(exc, exc_cls):
            return code
    return 500


def build_error_response(
    code: str,
    message: str,
    status_code: int,
    request_id: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
) -> JSONResponse:
    """Builds a canonical API-010 JSONResponse with X-Request-ID and optional custom response headers."""
    resp_headers = dict(headers or {})
    if request_id:
        resp_headers["X-Request-ID"] = request_id

    payload = {
        "error": {
            "code": code,
            "message": message,
            "request_id": request_id,
            "details": details or {},
        }
    }
    return JSONResponse(status_code=status_code, content=payload, headers=resp_headers)


def extract_request_id(request: Request, exc: Optional[Exception] = None) -> str:
    """
    Helper to extract the authoritative request ID established by RequestIdMiddleware.
    request.state.request_id is strictly authoritative; exc.request_id is never permitted
    to override the middleware's assigned request identifier.
    Falls back to ContextVar or generates a server-controlled ID if unexpectedly absent.
    """
    if hasattr(request, "state") and getattr(request.state, "request_id", None):
        return request.state.request_id
    from doc_extractor.api.middleware import get_request_id_from_context
    ctx_id = get_request_id_from_context()
    if ctx_id:
        return ctx_id
    from uuid import uuid4
    fallback_id = f"req_{uuid4().hex}"
    if hasattr(request, "state"):
        request.state.request_id = fallback_id
    return fallback_id


async def document_extractor_error_handler(request: Request, exc: DocumentExtractorError) -> JSONResponse:
    """Handles all DocumentExtractorError subclasses and formats them per API-010."""
    status_code = get_status_code_for_exception(exc)
    req_id = extract_request_id(request, exc)
    extra_headers: Dict[str, str] = {}
    if hasattr(exc, "retry_after_seconds") and exc.retry_after_seconds is not None:
        extra_headers["Retry-After"] = str(max(0, int(exc.retry_after_seconds)))
    return build_error_response(
        code=exc.code,
        message=exc.message,
        status_code=status_code,
        request_id=req_id,
        details=exc.details,
        headers=extra_headers,
    )


async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """Handles Starlette/FastAPI HTTPExceptions conforming to API-010."""
    status_code = exc.status_code
    req_id = extract_request_id(request)

    code_lookup = {
        400: "BAD_REQUEST",
        401: "AUTHENTICATION_REQUIRED",
        403: "FORBIDDEN",
        404: "NOT_FOUND",
        405: "METHOD_NOT_ALLOWED",
        409: "CONFLICT",
        413: "FILE_TOO_LARGE",
        415: "UNSUPPORTED_MEDIA_TYPE",
        422: "VALIDATION_ERROR",
        429: "RATE_LIMITED",
        500: "INTERNAL_ERROR",
        503: "SERVICE_UNAVAILABLE",
    }
    code = code_lookup.get(status_code, f"HTTP_{status_code}")
    message = str(exc.detail) if exc.detail else "HTTP request failed."
    extra_headers: Dict[str, str] = {}
    if exc.headers and "Retry-After" in exc.headers:
        extra_headers["Retry-After"] = str(exc.headers["Retry-After"])
    return build_error_response(
        code=code,
        message=message,
        status_code=status_code,
        request_id=req_id,
        details={},
    )


async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Handles FastAPI Pydantic request validation errors without leaking sensitive data."""
    req_id = extract_request_id(request)
    sanitized_errors = []
    for err in exc.errors():
        sanitized_errors.append({
            "location": [str(x) for x in err.get("loc", [])],
            "message": err.get("msg", "Invalid value"),
            "type": err.get("type", "value_error"),
        })

    return build_error_response(
        code="VALIDATION_ERROR",
        message="Request validation failed. Check parameter and body specifications.",
        status_code=422,
        request_id=req_id,
        details={"validation_errors": sanitized_errors},
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all handler for unexpected internal server errors (500)."""
    req_id = extract_request_id(request)
    # Message must never expose internal stack traces or database info (API-011)
    return build_error_response(
        code="INTERNAL_ERROR",
        message="An unexpected internal server error occurred.",
        status_code=500,
        request_id=req_id,
        details={},
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Registers canonical API-010 exception handlers on the FastAPI application."""
    app.add_exception_handler(DocumentExtractorError, document_extractor_error_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_error_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)
