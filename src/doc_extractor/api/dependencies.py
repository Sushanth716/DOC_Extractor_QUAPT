"""
FastAPI dependency injection providers and authorization guards (Module M5).

Adheres to:
- PRD §10.1 SEC-001, SEC-003, SEC-005: Strict tenant boundary isolation.
- Tenant context originates EXCLUSIVELY from the authenticated principal.
- Client headers, query parameters, or body fields asserting tenant identity are rejected.
- Provider-independent injection of repositories, queue brokers, and audit loggers.
"""

from typing import Callable, Optional, Sequence
from fastapi import Depends, Header, Request

from doc_extractor.api.config import ApiSettings, get_api_settings
from doc_extractor.api.exceptions import (
    AuthenticationRequiredError,
    ForbiddenError,
    InternalServerError,
)
from doc_extractor.api.ports.audit import AuditLogger, SafeStructuredAuditLogger
from doc_extractor.api.ports.authentication import (
    ApiPrincipal,
    AuthenticationProvider,
    TenantContext,
)
from doc_extractor.api.ports.queue import QueueConsumer, QueueProducer
from doc_extractor.api.ports.repositories import (
    ApiKeyRepository,
    AuditRepository,
    DocumentMetadataRepository,
    ReviewTaskRepository,
    WebhookDeliveryRepository,
    WebhookSubscriptionRepository,
)


def get_settings(request: Request) -> ApiSettings:
    """Retrieve application configuration from app state or singleton."""
    if hasattr(request.app.state, "settings") and request.app.state.settings is not None:
        return request.app.state.settings
    return get_api_settings()


def get_request_id(request: Request) -> str:
    """Retrieve correlated request ID established by RequestIdMiddleware."""
    req_id = getattr(request.state, "request_id", None)
    if not req_id:
        from doc_extractor.api.middleware import get_request_id_from_context
        req_id = get_request_id_from_context()
    return req_id or "unknown"


def get_auth_provider(request: Request) -> AuthenticationProvider:
    """Retrieve the registered AuthenticationProvider from application state."""
    provider = getattr(request.app.state, "auth_provider", None)
    if provider is not None:
        return provider

    # If api_key_repo is configured, auto-construct ApiKeyAuthenticationProvider (TASK-5.7-B)
    api_key_repo = getattr(request.app.state, "api_key_repo", None)
    if api_key_repo is not None:
        from doc_extractor.api.ports.audit import SafeStructuredAuditLogger
        from doc_extractor.api.providers.api_key_auth import ApiKeyAuthenticationProvider

        audit_logger = getattr(request.app.state, "audit_logger", None) or SafeStructuredAuditLogger()
        provider = ApiKeyAuthenticationProvider(repository=api_key_repo, audit_logger=audit_logger)
        request.app.state.auth_provider = provider
        return provider

    # Fail closed: No provider configured and no repository to construct one
    raise InternalServerError(
        message="Authentication provider is not configured on the application.",
        code="AUTH_PROVIDER_UNCONFIGURED",
    )


async def get_current_principal(
    request: Request,
    auth_header: Optional[str] = Header(None, alias="Authorization"),
    auth_provider: AuthenticationProvider = Depends(get_auth_provider),
) -> ApiPrincipal:
    """
    Authenticate the incoming request principal.

    Fail-closed: Returns an authoritative ApiPrincipal or raises AuthenticationRequiredError.
    Never falls back to an anonymous principal or default tenant.
    """
    if not auth_header:
        raise AuthenticationRequiredError(
            message="Authorization header is required.",
            code="AUTHENTICATION_REQUIRED",
        )

    principal = await auth_provider.authenticate(auth_header, request=request)
    if principal is None or not principal.is_active:
        raise AuthenticationRequiredError(
            message="Invalid or revoked authentication credentials.",
            code="INVALID_CREDENTIALS",
        )
    return principal


def get_tenant_context(
    principal: ApiPrincipal = Depends(get_current_principal),
) -> TenantContext:
    """
    Derives TenantContext strictly and exclusively from the authenticated principal.

    SEC-005 Guardrail: Under no circumstances is tenant identity parsed from
    query params, headers (e.g. X-Tenant-ID), or request bodies.
    """
    return principal.tenant_context


def require_scopes(*required_scopes: str) -> Callable:
    """Factory creating dependency to enforce required scopes on an endpoint."""

    async def _scope_guard(principal: ApiPrincipal = Depends(get_current_principal)) -> ApiPrincipal:
        for scope in required_scopes:
            if not principal.has_scope(scope):
                raise ForbiddenError(
                    message=f"Access denied. Missing required scope: {scope}",
                    code="FORBIDDEN_SCOPE",
                    details={"required_scope": scope},
                )
        return principal

    return _scope_guard


def require_permissions(*required_permissions: str) -> Callable:
    """Factory creating dependency to enforce required permissions on an endpoint."""

    async def _permission_guard(principal: ApiPrincipal = Depends(get_current_principal)) -> ApiPrincipal:
        for perm in required_permissions:
            if not principal.has_permission(perm):
                raise ForbiddenError(
                    message=f"Access denied. Missing required permission: {perm}",
                    code="FORBIDDEN_PERMISSION",
                    details={"required_permission": perm},
                )
        return principal

    return _permission_guard


# ============================================================================
# Hexagonal Port Dependencies (Dependency Injection Stubs)
# ============================================================================

def get_document_metadata_repository(request: Request) -> DocumentMetadataRepository:
    """Dependency injection provider for DocumentMetadataRepository."""
    repo = getattr(request.app.state, "document_metadata_repo", None)
    if repo is None:
        raise NotImplementedError("DocumentMetadataRepository adapter not configured.")
    return repo


def get_review_task_repository(request: Request) -> ReviewTaskRepository:
    """Dependency injection provider for ReviewTaskRepository."""
    repo = getattr(request.app.state, "review_task_repo", None)
    if repo is None:
        raise NotImplementedError("ReviewTaskRepository adapter not configured.")
    return repo


def get_api_key_repository(request: Request) -> ApiKeyRepository:
    """Dependency injection provider for ApiKeyRepository."""
    repo = getattr(request.app.state, "api_key_repo", None)
    if repo is None:
        raise NotImplementedError("ApiKeyRepository adapter not configured.")
    return repo


def get_webhook_subscription_repository(request: Request) -> WebhookSubscriptionRepository:
    """Dependency injection provider for WebhookSubscriptionRepository."""
    repo = getattr(request.app.state, "webhook_subscription_repo", None)
    if repo is None:
        raise NotImplementedError("WebhookSubscriptionRepository adapter not configured.")
    return repo


def get_webhook_delivery_repository(request: Request) -> WebhookDeliveryRepository:
    """Dependency injection provider for WebhookDeliveryRepository."""
    repo = getattr(request.app.state, "webhook_delivery_repo", None)
    if repo is None:
        raise NotImplementedError("WebhookDeliveryRepository adapter not configured.")
    return repo


def get_audit_repository(request: Request) -> AuditRepository:
    """Dependency injection provider for AuditRepository."""
    repo = getattr(request.app.state, "audit_repo", None)
    if repo is None:
        raise NotImplementedError("AuditRepository adapter not configured.")
    return repo


def get_queue_producer(request: Request) -> QueueProducer:
    """Dependency injection provider for durable QueueProducer."""
    producer = getattr(request.app.state, "queue_producer", None)
    if producer is None:
        raise NotImplementedError("QueueProducer adapter not configured.")
    return producer


def get_queue_consumer(request: Request) -> QueueConsumer:
    """Dependency injection provider for durable QueueConsumer."""
    consumer = getattr(request.app.state, "queue_consumer", None)
    if consumer is None:
        raise NotImplementedError("QueueConsumer adapter not configured.")
    return consumer


def get_audit_logger(request: Request) -> AuditLogger:
    """Dependency injection provider for structured AuditLogger."""
    logger = getattr(request.app.state, "audit_logger", None)
    if logger is None:
        return SafeStructuredAuditLogger()
    return logger


def get_api_key_service(
    request: Request,
    api_key_repo: ApiKeyRepository = Depends(get_api_key_repository),
    audit_logger: AuditLogger = Depends(get_audit_logger),
) -> Any:
    """Dependency injection provider for ApiKeyService."""
    service = getattr(request.app.state, "api_key_service", None)
    if service is not None:
        return service
    from doc_extractor.api.services.api_key_service import ApiKeyService

    return ApiKeyService(repository=api_key_repo, audit_logger=audit_logger)
