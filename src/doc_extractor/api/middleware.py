"""
Middleware components for the Document Extraction Engine REST API (Module M5).

Adheres to:
- PRD §7.2 API-010: Request ID correlation and tracing.
- Generation of unique request identifiers.
- ContextVar and request.state propagation across the entire request lifecycle.
- Inclusion of X-Request-ID in all HTTP response headers.
- Safe validation of client-supplied identifiers to prevent header injection or untrusted tracking IDs.
"""

import contextvars
import re
from typing import Callable, Optional
from uuid import uuid4
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

# Context variable to make request_id accessible anywhere in async execution context
request_id_ctx_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "request_id", default=None
)

# Strict regex for client-supplied request IDs: alphanumeric, underscores, hyphens only, 8-64 chars
_SAFE_REQUEST_ID_REGEX = re.compile(r"^[a-zA-Z0-9_\-]{8,64}$")


def get_request_id_from_context() -> Optional[str]:
    """Retrieve the current request ID from the async context variable."""
    return request_id_ctx_var.get()


class RequestIdMiddleware(BaseHTTPMiddleware):
    """
    Middleware managing request identifier generation, validation, propagation,
    and response header assignment.
    """

    def __init__(
        self,
        app: Callable,
        header_name: str = "X-Request-ID",
        allow_client_request_id: bool = True,
    ) -> None:
        super().__init__(app)
        self.header_name = header_name
        self.allow_client_request_id = allow_client_request_id

    def _resolve_request_id(self, incoming_header: Optional[str]) -> str:
        """
        Validates client-supplied header or generates a fresh authoritative ID.
        Untrusted or malformed headers are discarded to prevent injection.
        """
        if self.allow_client_request_id and incoming_header:
            candidate = incoming_header.strip()
            if _SAFE_REQUEST_ID_REGEX.match(candidate):
                return candidate

        # Fallback to server-generated authoritative UUID
        return f"req_{uuid4().hex}"

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        raw_header = request.headers.get(self.header_name)
        request_id = self._resolve_request_id(raw_header)

        # Store in request.state for Starlette/FastAPI route & error handlers
        request.state.request_id = request_id

        # Store in ContextVar for logging / auditing across coroutine boundaries
        token = request_id_ctx_var.set(request_id)

        try:
            response = await call_next(request)
            # Ensure response contains documented correlation header
            response.headers[self.header_name] = request_id
            return response
        finally:
            request_id_ctx_var.reset(token)
