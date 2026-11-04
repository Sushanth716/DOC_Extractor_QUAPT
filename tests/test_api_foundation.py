"""
Automated test suite for M5 Phase 0: Core REST API Foundation.

Covers:
1. Application initialization and lifecycle.
2. Health endpoint integration (API-009).
3. Request ID generation, validation, and propagation (API-010).
4. Canonical API-010 error envelope format.
5. Request validation error handling.
6. Authentication dependency behavior (fail-closed, scopes, permissions).
7. Tenant-context isolation (SEC-005: tenant ID exclusively from principal).
8. Repository protocol contracts.
9. Durable queue protocol contracts (ARC-003).
10. OpenAPI specification generation.
11. Structured audit logger with SEC-012 PII redaction.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence
import pytest
from fastapi import APIRouter, Depends, FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field

from doc_extractor.api.app import create_app
from doc_extractor.api.config import ApiSettings, load_api_settings
from doc_extractor.api.contracts.errors import ErrorEnvelope
from doc_extractor.api.contracts.records import (
    ApiKeyRecord,
    AuditEventRecord,
    DocumentProcessingMessage,
    DocumentRecord,
    QueueEnvelope,
    ReviewTaskRecord,
    WebhookDeliveryRecord,
    WebhookSubscriptionRecord,
)
from doc_extractor.api.dependencies import (
    get_api_settings,
    get_current_principal,
    get_request_id,
    get_tenant_context,
    require_permissions,
    require_scopes,
)
from doc_extractor.api.exceptions import (
    ApiValidationError,
    AuthenticationRequiredError,
    ConflictError,
    FileTooLargeError,
    ForbiddenError,
    InternalServerError,
    NotFoundError,
    RateLimitExceededError,
    TenantAccessDeniedError,
    TenantResourceNotFoundError,
)
from schema_lifecycle.exceptions import SchemaNotApprovedError
from doc_extractor.api.middleware import get_request_id_from_context
from doc_extractor.api.ports.audit import (
    AuditLogger,
    SafeStructuredAuditLogger,
    sanitize_audit_metadata,
)
from doc_extractor.api.ports.authentication import (
    ApiPrincipal,
    AuthenticationProvider,
    PrincipalType,
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
from doc_extractor.exceptions import (
    ConfigurationError,
    MalwareDetectedError,
    ScannerUnavailableError,
    SecurityCheckFailedError,
    UnsupportedFileError,
)


# ============================================================================
# Dummy Test Router & Test Providers
# ============================================================================

sample_router = APIRouter(prefix="/test", tags=["Test"])


class SampleRequestBody(BaseModel):
    item_name: str = Field(..., min_length=3)
    quantity: int = Field(..., gt=0)


@sample_router.get("/echo-request-id")
async def echo_request_id(req_id: str = Depends(get_request_id)):
    return {"request_id": req_id, "context_req_id": get_request_id_from_context()}


@sample_router.post("/validate-body")
async def validate_body(body: SampleRequestBody):
    return {"received": body.model_dump()}


@sample_router.get("/trigger-error/{error_type}")
async def trigger_error(error_type: str):
    if error_type == "not_found":
        raise NotFoundError("The requested document does not exist.")
    elif error_type == "forbidden":
        raise ForbiddenError("You lack permission to perform this operation.")
    elif error_type == "tenant_denied":
        raise TenantAccessDeniedError("Access to tenant 't-restricted' is forbidden.")
    elif error_type == "tenant_resource_not_found":
        raise TenantResourceNotFoundError("Document not found.")
    elif error_type == "auth_required":
        raise AuthenticationRequiredError("Token is expired or invalid.")
    elif error_type == "conflict":
        raise ConflictError("Idempotency fingerprint conflict.")
    elif error_type == "rate_limit":
        raise RateLimitExceededError("Rate limit exceeded for endpoint.", retry_after_seconds=60)
    elif error_type == "validation":
        raise ApiValidationError("Custom payload validation error.")
    elif error_type == "unsupported_file":
        raise UnsupportedFileError("Corrupt header bytes detected.")
    elif error_type == "security_check":
        raise SecurityCheckFailedError("File exceeded size expansion limit.")
    elif error_type == "malware":
        raise MalwareDetectedError("EICAR test signature found.")
    elif error_type == "file_too_large":
        raise FileTooLargeError("File exceeds the size limit.")
    elif error_type == "schema_not_approved":
        raise SchemaNotApprovedError("This schema isn't approved for production.", schema_id="invoice_schema", version=1)
    elif error_type == "scanner_unavailable":
        raise ScannerUnavailableError("Scanner socket timed out.")
    elif error_type == "internal":
        raise InternalServerError("Something went wrong.")
    elif error_type == "unhandled":
        raise RuntimeError("Unexpected raw python runtime exception!")
    return {"status": "ok"}


@sample_router.get("/spoofed-request-id-error")
async def spoofed_request_id_error():
    raise NotFoundError("Resource missing", request_id="spoofed_client_id_999")


@sample_router.get("/guarded-scope")
async def guarded_scope(principal: ApiPrincipal = Depends(require_scopes("documents:read"))):
    return {"principal_id": principal.principal_id, "scope": "documents:read"}


@sample_router.get("/guarded-permission")
async def guarded_permission(principal: ApiPrincipal = Depends(require_permissions("schema:approve"))):
    return {"principal_id": principal.principal_id, "permission": "schema:approve"}


@sample_router.get("/tenant-boundary")
async def tenant_boundary(
    tenant: TenantContext = Depends(get_tenant_context),
    principal: ApiPrincipal = Depends(get_current_principal),
):
    return {
        "tenant_id": tenant.tenant_id,
        "is_active": tenant.is_active,
        "principal_id": principal.principal_id,
    }


class MockAuthProvider:
    """Mock authentication provider for dependency testing."""

    def __init__(self, principals: Optional[Dict[str, ApiPrincipal]] = None):
        self.principals = principals or {}

    async def authenticate(self, credentials: Any, **kwargs: Any) -> Optional[ApiPrincipal]:
        token = str(credentials).replace("Bearer ", "").strip()
        return self.principals.get(token)


@pytest.fixture
def auth_provider() -> MockAuthProvider:
    principals = {
        "valid-token-alice": ApiPrincipal(
            principal_id="user_alice",
            tenant_id="tenant_alpha",
            principal_type=PrincipalType.USER,
            roles={"reviewer"},
            scopes={"documents:read", "documents:write"},
            permissions={"documents:read"},
        ),
        "valid-token-approver": ApiPrincipal(
            principal_id="user_bob",
            tenant_id="tenant_alpha",
            principal_type=PrincipalType.USER,
            roles={"schema_approver"},
            scopes={"schema:approve"},
            permissions={"schema:approve"},
        ),
        "admin-token": ApiPrincipal(
            principal_id="user_admin",
            tenant_id="tenant_alpha",
            principal_type=PrincipalType.USER,
            roles={"admin"},
            scopes={"admin:config"},
        ),
        "wildcard-token": ApiPrincipal(
            principal_id="user_wildcard",
            tenant_id="tenant_alpha",
            principal_type=PrincipalType.USER,
            scopes={"*"},
        ),
        "revoked-token": ApiPrincipal(
            principal_id="user_revoked",
            tenant_id="tenant_alpha",
            is_active=False,
        ),
    }
    return MockAuthProvider(principals)


@pytest.fixture
def test_app(auth_provider: MockAuthProvider) -> FastAPI:
    settings = ApiSettings(
        app_env="development",
        app_name="test-doc-extractor",
        debug=True,
    )
    app = create_app(settings)
    app.state.auth_provider = auth_provider
    app.include_router(sample_router)
    return app


@pytest.fixture
def client(test_app: FastAPI) -> TestClient:
    return TestClient(test_app, raise_server_exceptions=False)


# ============================================================================
# 1. Application Initialization & Centralized Configuration
# ============================================================================

@pytest.mark.unit
def test_app_initialization_defaults():
    """Verify application factory initializes correctly with default configuration."""
    app = create_app()
    assert app.title == "Document Extraction Engine REST API"
    assert app.version == "1.0.0"
    assert app.docs_url == "/docs"
    assert app.state.settings is not None
    assert app.state.settings.allow_external_egress is False


@pytest.mark.unit
def test_app_custom_settings():
    """Verify application factory respects custom settings overrides."""
    custom = ApiSettings(
        app_env="development",
        api_title="Custom Extractor API",
        api_version="2.1.0",
        docs_url="/api-docs",
    )
    app = create_app(custom)
    assert app.title == "Custom Extractor API"
    assert app.version == "2.1.0"
    assert app.docs_url == "/api-docs"


@pytest.mark.security
def test_production_debug_guard():
    """Ensure debug mode in production or staging environments raises ConfigurationError."""
    with pytest.raises(ConfigurationError) as exc_info:
        load_api_settings({"app_env": "production", "debug": True})
    assert "Debug mode must be disabled in production" in str(exc_info.value)


@pytest.mark.security
def test_production_egress_guard():
    """Ensure external egress is strictly rejected per ARC-001/SEC-011."""
    with pytest.raises(ConfigurationError) as exc_info:
        load_api_settings({"allow_external_egress": True})
    assert "External egress (allow_external_egress=True) violates ARC-001/SEC-011" in str(exc_info.value)


@pytest.mark.unit
def test_app_lifespan():
    """Verify application startup and shutdown lifespan execution."""
    app = create_app()
    with TestClient(app) as test_client:
        response = test_client.get("/health")
        assert response.status_code == 200


# ============================================================================
# 2. Health Endpoint Integration (API-009)
# ============================================================================

@pytest.mark.unit
def test_health_endpoint_root(client: TestClient):
    """Verify GET /health returns PRD-compliant attributes and truthful status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["egress_isolated"] is True
    assert data["target_render_dpi"] == 200
    assert data["api_version"] == "1.0.0"
    assert data["malware_scanner_type"] == "mock"
    assert "malware_scanner_ready" in data


@pytest.mark.unit
def test_health_endpoint_versioned(client: TestClient):
    """Verify GET /api/v1/health returns identical healthy response per PRD §7.1."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["api_version"] == "1.0.0"


class MockOperationalScanner:
    """Mock scanner that reports operational readiness."""
    def is_operational(self) -> bool:
        return True


@pytest.mark.unit
def test_health_endpoint_with_active_scanner(test_app: FastAPI):
    """Verify health endpoint accurately reflects an operational malware scanner."""
    test_app.state.malware_scanner = MockOperationalScanner()
    c = TestClient(test_app, raise_server_exceptions=False)
    response = c.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["malware_scanner_ready"] is True
    assert data["malware_scanner_engine"] == "MockOperationalScanner"


# ============================================================================
# 3. Request ID Generation, Validation, and Propagation (API-010)
# ============================================================================

@pytest.mark.unit
def test_request_id_auto_generated(client: TestClient):
    """Verify server generates a unique request ID and sets X-Request-ID header."""
    response = client.get("/test/echo-request-id")
    assert response.status_code == 200
    req_header = response.headers.get("X-Request-ID")
    assert req_header is not None
    assert req_header.startswith("req_")
    body = response.json()
    assert body["request_id"] == req_header
    assert body["context_req_id"] == req_header


@pytest.mark.unit
def test_request_id_valid_client_header_accepted(client: TestClient):
    """Verify safe client-supplied request ID is accepted and echoed."""
    custom_id = "trace-client-test-12345"
    response = client.get("/test/echo-request-id", headers={"X-Request-ID": custom_id})
    assert response.status_code == 200
    assert response.headers.get("X-Request-ID") == custom_id
    assert response.json()["request_id"] == custom_id


@pytest.mark.security
def test_request_id_untrusted_header_discarded(client: TestClient):
    """
    Prevent untrusted client-supplied identifiers from being blindly trusted.
    Malicious or non-conforming IDs (SQLi, XSS, control chars, spaces) are discarded.
    """
    malicious_ids = [
        "<script>alert('xss')</script>",
        "../../etc/passwd",
        "bad\r\nid",
        "short",  # < 8 chars
        "x" * 128,  # > 64 chars
        "id with spaces inside",
        "id; DROP TABLE users;--",
    ]
    for bad_id in malicious_ids:
        response = client.get("/test/echo-request-id", headers={"X-Request-ID": bad_id})
        assert response.status_code == 200
        assigned_id = response.headers.get("X-Request-ID")
        assert assigned_id != bad_id
        assert assigned_id.startswith("req_")


# ============================================================================
# 4. Canonical API-010 Nested Error Envelope
# ============================================================================

@pytest.mark.unit
@pytest.mark.parametrize(
    "error_type,expected_status,expected_code",
    [
        ("not_found", 404, "DOCUMENT_NOT_FOUND"),
        ("forbidden", 403, "FORBIDDEN"),
        ("tenant_denied", 403, "TENANT_ACCESS_DENIED"),
        ("tenant_resource_not_found", 404, "DOCUMENT_NOT_FOUND"),
        ("auth_required", 401, "AUTHENTICATION_REQUIRED"),
        ("conflict", 409, "CONFLICT"),
        ("rate_limit", 429, "RATE_LIMITED"),
        ("validation", 422, "VALIDATION_ERROR"),
        ("unsupported_file", 400, "UNSUPPORTED_FILE"),
        ("security_check", 400, "SECURITY_CHECK_FAILED"),
        ("malware", 400, "MALWARE_DETECTED"),
        ("file_too_large", 413, "FILE_TOO_LARGE"),
        ("schema_not_approved", 409, "SCHEMA_NOT_APPROVED"),
        ("scanner_unavailable", 503, "SCANNER_UNAVAILABLE"),
        ("internal", 500, "INTERNAL_ERROR"),
    ],
)
def test_api_010_error_envelope_mapping(client: TestClient, error_type, expected_status, expected_code):
    """Verify all domain errors produce standard API-010 envelope and status codes."""
    response = client.get(f"/test/trigger-error/{error_type}")
    assert response.status_code == expected_status
    data = response.json()

    # Enforce API-010 nested structure
    assert "error" in data
    err = data["error"]
    assert err["code"] == expected_code
    assert isinstance(err["message"], str)
    assert len(err["message"]) > 0
    assert err["request_id"] == response.headers.get("X-Request-ID")
    assert isinstance(err["details"], dict)

    # Validate against Pydantic contract
    parsed_envelope = ErrorEnvelope.model_validate(data)
    assert parsed_envelope.error.code == expected_code


@pytest.mark.unit
def test_rate_limit_retry_after_header(client: TestClient):
    """Verify HTTP 429 RATE_LIMITED response carries non-negative integer Retry-After header."""
    response = client.get("/test/trigger-error/rate_limit")
    assert response.status_code == 429
    assert response.json()["error"]["code"] == "RATE_LIMITED"
    assert "Retry-After" in response.headers
    retry_val = response.headers["Retry-After"]
    assert retry_val.isdigit()
    assert int(retry_val) >= 0


@pytest.mark.security
def test_authoritative_request_id_not_overridden_by_exception(client: TestClient):
    """Verify authoritative middleware request.state.request_id is never overridden by exc.request_id."""
    response = client.get("/test/spoofed-request-id-error")
    assert response.status_code == 404
    data = response.json()
    server_req_id = response.headers.get("X-Request-ID")
    assert server_req_id is not None
    assert server_req_id.startswith("req_")
    # Response JSON request_id must match header and MUST NOT be the spoofed one
    assert data["error"]["request_id"] == server_req_id
    assert data["error"]["request_id"] != "spoofed_client_id_999"


@pytest.mark.unit
def test_unhandled_exception_masks_internals(client: TestClient):
    """Ensure unhandled 500 exceptions do not leak stack traces or internal details (API-011)."""
    response = client.get("/test/trigger-error/unhandled")
    assert response.status_code == 500
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == "INTERNAL_ERROR"
    assert "Unexpected raw python" not in data["error"]["message"]
    assert data["error"]["details"] == {}


@pytest.mark.unit
def test_starlette_not_found_mapped_to_api_010(client: TestClient):
    """Ensure non-existent endpoints return 404 with API-010 format."""
    response = client.get("/non-existent-endpoint-xyz")
    assert response.status_code == 404
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == "NOT_FOUND"
    assert data["error"]["request_id"] == response.headers.get("X-Request-ID")


# ============================================================================
# 5. Validation Error Handling
# ============================================================================

@pytest.mark.unit
def test_validation_error_envelope(client: TestClient):
    """Verify Pydantic request body validation failures format per API-010 without PII leakage."""
    response = client.post(
        "/test/validate-body",
        json={"item_name": "x", "quantity": -5},
    )
    assert response.status_code == 422
    data = response.json()
    assert "error" in data
    err = data["error"]
    assert err["code"] == "VALIDATION_ERROR"
    assert err["request_id"] == response.headers.get("X-Request-ID")
    assert "validation_errors" in err["details"]
    errors = err["details"]["validation_errors"]
    assert len(errors) >= 1
    for item in errors:
        assert "location" in item
        assert "message" in item


# ============================================================================
# 6. Authentication Dependency Behavior (Fail-Closed)
# ============================================================================

@pytest.mark.security
def test_auth_unconfigured_provider_fails_closed():
    """Verify that an app with no auth provider configured fails closed with 500 AUTH_PROVIDER_UNCONFIGURED."""
    raw_app = create_app()
    raw_app.include_router(sample_router)
    c = TestClient(raw_app, raise_server_exceptions=False)
    response = c.get("/test/guarded-scope", headers={"Authorization": "Bearer some-token"})
    assert response.status_code == 500
    data = response.json()
    assert data["error"]["code"] == "AUTH_PROVIDER_UNCONFIGURED"


@pytest.mark.security
def test_record_models_reject_extra_fields():
    """Ensure data contracts forbid unexpected extra fields to prevent injection."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        DocumentRecord(
            document_id="doc_1",
            tenant_id="tenant_1",
            filename="doc.pdf",
            mime_type="application/pdf",
            file_size_bytes=100,
            sha256_hash="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            # pyrefly: ignore[unexpected-keyword]
            unexpected_injection_field="malicious",
        )

    with pytest.raises(ValidationError):
        ReviewTaskRecord(
            task_id="task_1",
            document_id="doc_1",
            tenant_id="tenant_1",
            # pyrefly: ignore[unexpected-keyword]
            extra_field="disallowed",
        )


@pytest.mark.unit
def test_dependency_stubs_raise_not_implemented():
    """Verify hexagonal port dependency stubs raise NotImplementedError when unconfigured."""
    from doc_extractor.api.dependencies import (
        get_api_key_repository,
        get_audit_repository,
        get_document_metadata_repository,
        get_queue_consumer,
        get_queue_producer,
        get_review_task_repository,
        get_webhook_delivery_repository,
        get_webhook_subscription_repository,
    )

    class DummyRequest:
        app = type("App", (), {"state": type("State", (), {
            "document_metadata_repo": None,
            "review_task_repo": None,
            "api_key_repo": None,
            "webhook_subscription_repo": None,
            "webhook_delivery_repo": None,
            "audit_repo": None,
            "queue_producer": None,
            "queue_consumer": None,
        })()})()

    req = DummyRequest()
    with pytest.raises(NotImplementedError):
        get_document_metadata_repository(req)
    with pytest.raises(NotImplementedError):
        get_review_task_repository(req)
    with pytest.raises(NotImplementedError):
        get_api_key_repository(req)
    with pytest.raises(NotImplementedError):
        get_webhook_subscription_repository(req)
    with pytest.raises(NotImplementedError):
        get_webhook_delivery_repository(req)
    with pytest.raises(NotImplementedError):
        get_audit_repository(req)
    with pytest.raises(NotImplementedError):
        get_queue_producer(req)
    with pytest.raises(NotImplementedError):
        get_queue_consumer(req)


@pytest.mark.security
def test_auth_missing_header_fails_closed(client: TestClient):
    """Verify missing Authorization header produces 401 AUTHENTICATION_REQUIRED."""
    response = client.get("/test/guarded-scope")
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "AUTHENTICATION_REQUIRED"


@pytest.mark.security
def test_auth_invalid_credentials_fails_closed(client: TestClient):
    """Verify unknown or invalid token produces 401 INVALID_CREDENTIALS."""
    response = client.get("/test/guarded-scope", headers={"Authorization": "Bearer bad-token"})
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "INVALID_CREDENTIALS"


@pytest.mark.security
def test_auth_revoked_principal_fails_closed(client: TestClient):
    """Verify inactive principal produces 401 INVALID_CREDENTIALS."""
    response = client.get("/test/guarded-scope", headers={"Authorization": "Bearer revoked-token"})
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "INVALID_CREDENTIALS"


@pytest.mark.unit
def test_auth_scope_evaluation_success(client: TestClient):
    """Verify caller holding required scope accesses protected endpoint."""
    response = client.get("/test/guarded-scope", headers={"Authorization": "Bearer valid-token-alice"})
    assert response.status_code == 200
    assert response.json()["scope"] == "documents:read"


@pytest.mark.security
def test_auth_scope_evaluation_denial(client: TestClient):
    """Verify caller lacking required scope is denied with 403 FORBIDDEN_SCOPE."""
    response = client.get("/test/guarded-scope", headers={"Authorization": "Bearer valid-token-approver"})
    assert response.status_code == 403
    data = response.json()
    assert data["error"]["code"] == "FORBIDDEN_SCOPE"
    assert data["error"]["details"]["required_scope"] == "documents:read"


@pytest.mark.security
def test_admin_does_not_bypass_scope_evaluation(client: TestClient):
    """Verify admin role does not automatically bypass API scope requirements (SEC-003)."""
    # admin-token has role='admin' but only scope='admin:config', not 'documents:read'
    response = client.get("/test/guarded-scope", headers={"Authorization": "Bearer admin-token"})
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "FORBIDDEN_SCOPE"


@pytest.mark.security
def test_admin_does_not_receive_schema_approve():
    """Verify admin role does NOT receive SCHEMA_APPROVE permission per M4 least-privilege (SEC-003)."""
    admin_principal = ApiPrincipal(
        principal_id="user_admin",
        tenant_id="tenant_alpha",
        roles={"admin"},
        scopes={"admin:all"},
    )
    # Admin has schema:read, schema:rollback via M4 policy
    assert admin_principal.has_permission("schema:read") is True
    assert admin_principal.has_permission("schema:rollback") is True
    # Admin MUST NOT have schema:approve (four-eyes principle)
    assert admin_principal.has_permission("schema:approve") is False


@pytest.mark.security
def test_wildcard_scope_does_not_bypass_scope_or_permissions(client: TestClient):
    """Verify wildcard scope '*' does not grant arbitrary scopes or domain permissions."""
    # wildcard-token has scopes={'*'}
    response = client.get("/test/guarded-scope", headers={"Authorization": "Bearer wildcard-token"})
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "FORBIDDEN_SCOPE"

    wildcard_principal = ApiPrincipal(
        principal_id="user_wildcard",
        tenant_id="tenant_alpha",
        scopes={"*"},
    )
    assert wildcard_principal.has_scope("documents:read") is False
    assert wildcard_principal.has_permission("schema:approve") is False
    assert wildcard_principal.has_permission("documents:delete") is False


@pytest.mark.unit
def test_auth_permission_evaluation(client: TestClient):
    """Verify require_permissions dependency enforces explicit permission."""
    response = client.get("/test/guarded-permission", headers={"Authorization": "Bearer valid-token-approver"})
    assert response.status_code == 200
    assert response.json()["permission"] == "schema:approve"


@pytest.mark.security
def test_m4_conversion_preserves_scope_permission_separation():
    """Verify to_m4_principal and from_m4_principal do not cross-contaminate scopes and permissions."""
    from schema_lifecycle.roles import AuthenticatedPrincipal as M4Principal, SchemaPermission

    api_p = ApiPrincipal(
        principal_id="user_test",
        tenant_id="tenant_x",
        roles={"reviewer"},
        scopes={"documents:read", "documents:write"},
        permissions={"schema:read"},
    )
    m4_p = api_p.to_m4_principal()
    # M4 explicit permissions must contain only 'schema:read', NOT the API scopes!
    assert SchemaPermission.SCHEMA_READ in m4_p.explicit_permissions
    assert len(m4_p.explicit_permissions) == 1

    # Converting back must not copy M4 permissions into API scopes
    roundtrip = ApiPrincipal.from_m4_principal(m4_p)
    assert roundtrip.scopes == set()
    assert "schema:read" in roundtrip.permissions


# ============================================================================
# 7. Tenant-Context Isolation (SEC-005)
# ============================================================================

@pytest.mark.security
def test_tenant_context_originates_exclusively_from_principal(client: TestClient):
    """
    SEC-005 Guardrail: Tenant context comes strictly from authenticated principal.
    Spoofing attempts via headers or query parameters must have ZERO effect.
    """
    response = client.get(
        "/test/tenant-boundary?tenant_id=spoofed_tenant",
        headers={
            "Authorization": "Bearer valid-token-alice",
            "X-Tenant-ID": "malicious_spoofed_tenant_header",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["tenant_id"] == "tenant_alpha"
    assert data["tenant_id"] != "spoofed_tenant"
    assert data["tenant_id"] != "malicious_spoofed_tenant_header"


@pytest.mark.unit
def test_principal_m4_interoperability():
    """Verify ApiPrincipal seamlessly converts to and from M4 AuthenticatedPrincipal."""
    from schema_lifecycle.roles import AuthenticatedPrincipal as M4Principal, Role as M4Role

    api_p = ApiPrincipal(
        principal_id="user_test",
        tenant_id="tenant_x",
        roles={"reviewer"},
        scopes={"documents:read"},
        permissions={"documents:read"},
    )
    m4_p = api_p.to_m4_principal()
    assert isinstance(m4_p, M4Principal)
    assert m4_p.principal_id == "user_test"
    assert m4_p.tenant_id == "tenant_x"
    assert M4Role.REVIEWER in m4_p.roles

    # Convert back
    roundtrip_api_p = ApiPrincipal.from_m4_principal(m4_p)
    assert roundtrip_api_p.principal_id == "user_test"
    assert roundtrip_api_p.tenant_id == "tenant_x"
    assert "reviewer" in roundtrip_api_p.roles


# ============================================================================
# 8. Repository Protocol Contracts
# ============================================================================

class MockDocumentMetadataRepo:
    """Mock fulfilling DocumentMetadataRepository protocol."""

    def __init__(self):
        self._store: Dict[str, DocumentRecord] = {}

    async def get_by_id(self, tenant_id: str, document_id: str) -> Optional[DocumentRecord]:
        rec = self._store.get(document_id)
        return rec if rec and rec.tenant_id == tenant_id else None

    async def create(self, tenant_id: str, record: DocumentRecord) -> DocumentRecord:
        if record.tenant_id != tenant_id:
            raise ValueError("Tenant mismatch")
        self._store[record.document_id] = record
        return record

    async def update_status(
        self,
        tenant_id: str,
        document_id: str,
        status: str,
        error_details: Optional[Dict[str, Any]] = None,
        storage_path: Optional[str] = None,
        page_count: Optional[int] = None,
    ) -> Optional[DocumentRecord]:
        rec = await self.get_by_id(tenant_id, document_id)
        if not rec:
            return None
        updated = rec.model_copy(update={
            "status": status,
            "error_details": error_details,
            "updated_at": datetime.now(timezone.utc),
        })
        self._store[document_id] = updated
        return updated

    async def list_by_tenant(
        self,
        tenant_id: str,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[DocumentRecord]:
        return [r for r in self._store.values() if r.tenant_id == tenant_id]

    async def delete(self, tenant_id: str, document_id: str) -> bool:
        if document_id in self._store and self._store[document_id].tenant_id == tenant_id:
            del self._store[document_id]
            return True
        return False


class MockReviewTaskRepo:
    """Mock fulfilling ReviewTaskRepository protocol."""
    async def get_by_id(self, tenant_id: str, task_id: str) -> Optional[ReviewTaskRecord]: return None
    async def get_by_document_id(self, tenant_id: str, document_id: str) -> Optional[ReviewTaskRecord]: return None
    async def create(self, tenant_id: str, task: ReviewTaskRecord) -> ReviewTaskRecord: return task
    async def update_status(self, tenant_id: str, task_id: str, status: str, reviewer_id: Optional[str] = None, notes: Optional[str] = None) -> Optional[ReviewTaskRecord]: return None
    async def list_tasks(self, tenant_id: str, status: Optional[str] = None, reviewer_id: Optional[str] = None, limit: int = 50, offset: int = 0) -> List[ReviewTaskRecord]: return []


class MockApiKeyRepo:
    """Mock fulfilling ApiKeyRepository protocol."""
    async def get_by_key_hash(self, key_hash: str) -> Optional[ApiKeyRecord]: return None
    async def get_by_id(self, tenant_id: str, key_id: str) -> Optional[ApiKeyRecord]: return None
    async def create(self, tenant_id: str, key_record: ApiKeyRecord) -> ApiKeyRecord: return key_record
    async def revoke(self, tenant_id: str, key_id: str) -> bool: return True
    async def list_keys(self, tenant_id: str, limit: int = 50, offset: int = 0) -> List[ApiKeyRecord]: return []


class MockWebhookSubRepo:
    """Mock fulfilling WebhookSubscriptionRepository protocol."""
    async def get_by_id(self, tenant_id: str, subscription_id: str) -> Optional[WebhookSubscriptionRecord]: return None
    async def list_by_tenant(self, tenant_id: str, event_type: Optional[str] = None) -> List[WebhookSubscriptionRecord]: return []
    async def create(self, tenant_id: str, subscription: WebhookSubscriptionRecord) -> WebhookSubscriptionRecord: return subscription
    async def update(self, tenant_id: str, subscription_id: str, update_data: Dict[str, Any]) -> Optional[WebhookSubscriptionRecord]: return None
    async def delete(self, tenant_id: str, subscription_id: str) -> bool: return True


class MockWebhookDeliveryRepo:
    """Mock fulfilling WebhookDeliveryRepository protocol."""
    async def record_delivery(self, tenant_id: str, delivery: WebhookDeliveryRecord) -> WebhookDeliveryRecord: return delivery
    async def get_by_id(self, tenant_id: str, delivery_id: str) -> Optional[WebhookDeliveryRecord]: return None
    async def list_by_subscription(self, tenant_id: str, subscription_id: str, limit: int = 50, offset: int = 0) -> List[WebhookDeliveryRecord]: return []
    async def update_delivery_status(self, tenant_id: str, delivery_id: str, status: str, response_status_code: Optional[int] = None, error_message: Optional[str] = None, next_retry_at: Optional[datetime] = None) -> Optional[WebhookDeliveryRecord]: return None


class MockAuditRepo:
    """Mock fulfilling AuditRepository protocol."""
    async def append_event(self, event: AuditEventRecord) -> None: pass
    async def query_events(self, tenant_id: str, resource_id: Optional[str] = None, action: Optional[str] = None, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None, limit: int = 100, offset: int = 0) -> List[AuditEventRecord]: return []


@pytest.mark.unit
def test_repository_protocol_contracts():
    """Verify mock implementations adhere to all defined repository protocols."""
    assert isinstance(MockDocumentMetadataRepo(), DocumentMetadataRepository)
    assert isinstance(MockReviewTaskRepo(), ReviewTaskRepository)
    assert isinstance(MockApiKeyRepo(), ApiKeyRepository)
    assert isinstance(MockWebhookSubRepo(), WebhookSubscriptionRepository)
    assert isinstance(MockWebhookDeliveryRepo(), WebhookDeliveryRepository)
    assert isinstance(MockAuditRepo(), AuditRepository)


@pytest.mark.unit
def test_document_metadata_repository_operations():
    """Verify DocumentRecord model validation and basic repository operations."""
    async def _run():
        repo = MockDocumentMetadataRepo()
        doc = DocumentRecord(
            document_id="doc_123456",
            tenant_id="tenant_alpha",
            filename="invoice.pdf",
            mime_type="application/pdf",
            file_size_bytes=1024,
            sha256_hash="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        )
        created = await repo.create("tenant_alpha", doc)
        assert created.document_id == "doc_123456"

        # Multi-tenant isolation: tenant_beta cannot see tenant_alpha's document
        found_alpha = await repo.get_by_id("tenant_alpha", "doc_123456")
        assert found_alpha is not None
        found_beta = await repo.get_by_id("tenant_beta", "doc_123456")
        assert found_beta is None

    import asyncio
    asyncio.run(_run())


# ============================================================================
# 9. Durable Queue Protocol Contracts (ARC-003)
# ============================================================================

class MockQueueProducer:
    """Mock producer fulfilling QueueProducer protocol."""

    def __init__(self):
        self.published: List[DocumentProcessingMessage] = []

    async def publish(self, message: DocumentProcessingMessage, delay_seconds: int = 0) -> str:
        self.published.append(message)
        return f"msg_ack_{message.message_id}"

    async def publish_batch(self, messages: Sequence[DocumentProcessingMessage]) -> List[str]:
        ids = []
        for m in messages:
            ids.append(await self.publish(m))
        return ids


class MockQueueConsumer:
    """Mock consumer fulfilling QueueConsumer protocol."""

    def __init__(self, envelopes: List[QueueEnvelope]):
        self.envelopes = list(envelopes)
        self.acked: List[str] = []
        self.nacked: List[str] = []
        self.extended: List[str] = []

    async def receive_messages(self, max_messages: int = 10, visibility_timeout_seconds: int = 60) -> List[QueueEnvelope]:
        batch = self.envelopes[:max_messages]
        self.envelopes = self.envelopes[max_messages:]
        return batch

    async def ack(self, receipt_handle: str) -> None:
        self.acked.append(receipt_handle)

    async def nack(self, receipt_handle: str, requeue: bool = True, backoff_seconds: int = 0) -> None:
        self.nacked.append(receipt_handle)

    async def extend_lease(self, receipt_handle: str, extension_seconds: int = 60) -> None:
        self.extended.append(receipt_handle)


@pytest.mark.unit
def test_queue_protocol_contracts():
    """Verify mock queue classes adhere to QueueProducer and QueueConsumer protocols."""
    assert isinstance(MockQueueProducer(), QueueProducer)
    assert isinstance(MockQueueConsumer([]), QueueConsumer)


@pytest.mark.unit
def test_durable_queue_lifecycle():
    """Verify publishing, receiving, acknowledging, and extending message leases."""
    async def _run():
        msg = DocumentProcessingMessage(
            message_id="msg_001",
            document_id="doc_001",
            tenant_id="tenant_alpha",
            event_id="evt_001",
            deduplication_id="idemp_key_001",
            attempt_count=1,
            max_retries=3,
            payload={"schema_id": "invoice_v1"},
        )
        producer = MockQueueProducer()
        ack_id = await producer.publish(msg)
        assert ack_id == "msg_ack_msg_001"
        assert len(producer.published) == 1

        envelope = QueueEnvelope(
            receipt_handle="handle_lease_999",
            message=msg,
            visible_until=datetime.now(timezone.utc),
        )
        consumer = MockQueueConsumer([envelope])
        received = await consumer.receive_messages(max_messages=1)
        assert len(received) == 1
        assert received[0].message.document_id == "doc_001"

        # Extend lease (heartbeat)
        await consumer.extend_lease("handle_lease_999", 30)
        assert "handle_lease_999" in consumer.extended

        # Acknowledge
        await consumer.ack("handle_lease_999")
        assert "handle_lease_999" in consumer.acked

    import asyncio
    asyncio.run(_run())


# ============================================================================
# 10. OpenAPI Specification Generation
# ============================================================================

@pytest.mark.unit
def test_openapi_generation():
    """Verify OpenAPI 3.x schema generation conforms to standards."""
    app = create_app()
    openapi_spec = app.openapi()
    assert openapi_spec["openapi"].startswith("3.")
    assert openapi_spec["info"]["title"] == "Document Extraction Engine REST API"
    assert openapi_spec["info"]["version"] == "1.0.0"

    # Check endpoints are documented
    paths = openapi_spec["paths"]
    assert "/health" in paths
    assert "/api/v1/health" in paths
    assert "get" in paths["/health"]


# ============================================================================
# 11. Structured Audit Logger & Allowlist Metadata
# ============================================================================

@pytest.mark.security
def test_audit_metadata_allowlisting_and_redaction():
    """Verify SEC-012 allowlisting retains approved operational fields and discards PII/raw data."""
    raw_metadata = {
        # Disallowed / PII / Credential fields (must be discarded)
        "user_email": "analyst@example.com",
        "api_key": "secret-token-xyz-123",
        "password": "super-secret-password",
        "aadhaar": "1234-5678-9012",
        "raw_content": "Extracted text content from bank statement",
        "document_bytes": b"binary data \x00\x01",
        "nested": {"token": "bearer-token-123", "safe_counter": 42},
        "item_list": ["secret-1", "secret-2"],
        # Words that previously caused substring false positives (must be discarded as non-allowlisted)
        "company": "Acme Corp",
        "participant": "John Doe",
        # Approved allowlisted operational metadata fields (must be preserved)
        "status": "queued",
        "byte_count": 1024,
        "page_count": 5,
        "duration_ms": 42.5,
        "file_type": "application/pdf",
        "outcome_code": "SUCCESS",
    }
    sanitized = sanitize_audit_metadata(raw_metadata)

    # Disallowed fields must be completely discarded
    assert "user_email" not in sanitized
    assert "api_key" not in sanitized
    assert "password" not in sanitized
    assert "aadhaar" not in sanitized
    assert "raw_content" not in sanitized
    assert "document_bytes" not in sanitized
    assert "nested" not in sanitized
    assert "item_list" not in sanitized
    assert "company" not in sanitized
    assert "participant" not in sanitized

    # Allowlisted operational fields must be preserved with exact values and types
    assert sanitized["status"] == "queued"
    assert sanitized["byte_count"] == 1024
    assert sanitized["page_count"] == 5
    assert sanitized["duration_ms"] == 42.5
    assert sanitized["file_type"] == "application/pdf"
    assert sanitized["outcome_code"] == "SUCCESS"


@pytest.mark.security
def test_audit_metadata_rejects_unsupported_objects():
    """Verify arbitrary objects, file handles, and bytes cannot enter audit metadata."""
    class CustomObject:
        pass

    raw_metadata = {
        "status": "failed",
        "bad_obj": CustomObject(),
        "bad_bytes": b"some_bytes",
    }
    sanitized = sanitize_audit_metadata(raw_metadata)
    assert sanitized == {"status": "failed"}


@pytest.mark.security
def test_audit_metadata_field_specific_validation():
    """Verify field-specific validation for all 11 allowlisted operational metadata fields."""
    # 1. Free text passed as status
    assert "status" not in sanitize_audit_metadata({"status": "Document contains Alice's data"})
    assert "status" not in sanitize_audit_metadata({"status": "arbitrary status"})
    assert sanitize_audit_metadata({"status": "auto_approved"}) == {"status": "auto_approved"}
    assert sanitize_audit_metadata({"status": "FAILED"}) == {"status": "failed"}

    # 2. Arbitrary customer information passed as error_code
    assert "error_code" not in sanitize_audit_metadata({"error_code": "User SSN 123-45-6789 was not found"})
    assert "error_code" not in sanitize_audit_metadata({"error_code": "customer John Doe error"})
    assert sanitize_audit_metadata({"error_code": "RATE_LIMITED"}) == {"error_code": "RATE_LIMITED"}
    assert sanitize_audit_metadata({"error_code": "MALWARE_DETECTED"}) == {"error_code": "MALWARE_DETECTED"}

    # 3. Invalid schema identifiers
    assert "schema_id" not in sanitize_audit_metadata({"schema_id": "schema with spaces and <script>"})
    assert "schema_id" not in sanitize_audit_metadata({"schema_id": "bad;DROP TABLE;"})
    assert sanitize_audit_metadata({"schema_id": "casa_account_opening"}) == {"schema_id": "casa_account_opening"}
    assert sanitize_audit_metadata({"schema_id": "invoice-schema-v1.0"}) == {"schema_id": "invoice-schema-v1.0"}

    # 4. Invalid rule identifiers
    assert "rule_id" not in sanitize_audit_metadata({"rule_id": "rule id with spaces"})
    assert "rule_id" not in sanitize_audit_metadata({"rule_id": "rule#invalid!@$"})
    assert sanitize_audit_metadata({"rule_id": "R1"}) == {"rule_id": "R1"}
    assert sanitize_audit_metadata({"rule_id": "rule_totals_match"}) == {"rule_id": "rule_totals_match"}

    # 5. Unsupported MIME types
    assert "file_type" not in sanitize_audit_metadata({"file_type": "application/x-malicious-executable"})
    assert "file_type" not in sanitize_audit_metadata({"file_type": "text/html"})
    assert "file_type" not in sanitize_audit_metadata({"file_type": "audio/mp3"})
    assert sanitize_audit_metadata({"file_type": "application/pdf"}) == {"file_type": "application/pdf"}
    assert sanitize_audit_metadata({"file_type": "image/jpeg"}) == {"file_type": "image/jpeg"}
    assert sanitize_audit_metadata({"file_type": "pdf"}) == {"file_type": "pdf"}

    # 6. Schema versions (M4 compatible: positive canonical integer or SemVer with major >= 1)
    # 6a. Zero and negative inputs rejected
    assert "schema_version" not in sanitize_audit_metadata({"schema_version": 0})
    assert "schema_version" not in sanitize_audit_metadata({"schema_version": "0"})
    assert "schema_version" not in sanitize_audit_metadata({"schema_version": -1})
    assert "schema_version" not in sanitize_audit_metadata({"schema_version": "v0"})
    assert "schema_version" not in sanitize_audit_metadata({"schema_version": True})
    assert "schema_version" not in sanitize_audit_metadata({"schema_version": False})

    # 6b. Zero-major SemVer strings rejected (M4 canonical version must be >= 1)
    assert "schema_version" not in sanitize_audit_metadata({"schema_version": "0.0.0"})
    assert "schema_version" not in sanitize_audit_metadata({"schema_version": "0.1.0"})
    assert "schema_version" not in sanitize_audit_metadata({"schema_version": "0.0.1"})
    assert "schema_version" not in sanitize_audit_metadata({"schema_version": "0.9.0"})

    # 6c. Malformed SemVer and invalid free text rejected
    assert "schema_version" not in sanitize_audit_metadata({"schema_version": "1.0"})
    assert "schema_version" not in sanitize_audit_metadata({"schema_version": "invalid"})
    assert "schema_version" not in sanitize_audit_metadata({"schema_version": "01.0.0"})

    # 6d. Positive canonical integer and positive-major SemVer accepted and normalized
    assert sanitize_audit_metadata({"schema_version": 1}) == {"schema_version": "1"}
    assert sanitize_audit_metadata({"schema_version": "1"}) == {"schema_version": "1"}
    assert sanitize_audit_metadata({"schema_version": "1.0.0"}) == {"schema_version": "1.0.0"}
    assert sanitize_audit_metadata({"schema_version": "2.3.4"}) == {"schema_version": "2.3.4"}
    assert sanitize_audit_metadata({"schema_version": "1.0.0-alpha.1"}) == {"schema_version": "1.0.0-alpha.1"}
    assert sanitize_audit_metadata({"schema_version": "1.0.0+build.5"}) == {"schema_version": "1.0.0+build.5"}

    # 7. Negative counters
    assert "byte_count" not in sanitize_audit_metadata({"byte_count": -100})
    assert "page_count" not in sanitize_audit_metadata({"page_count": -1})
    assert "attempt_count" not in sanitize_audit_metadata({"attempt_count": -5})
    assert "byte_count" not in sanitize_audit_metadata({"byte_count": True})
    assert "page_count" not in sanitize_audit_metadata({"page_count": 5.5})
    assert sanitize_audit_metadata({"byte_count": 2048}) == {"byte_count": 2048}
    assert sanitize_audit_metadata({"page_count": 0}) == {"page_count": 0}
    assert sanitize_audit_metadata({"attempt_count": 1}) == {"attempt_count": 1}

    # 8. Non-finite numeric values
    assert "duration_ms" not in sanitize_audit_metadata({"duration_ms": float("nan")})
    assert "duration_ms" not in sanitize_audit_metadata({"duration_ms": float("inf")})
    assert "duration_ms" not in sanitize_audit_metadata({"duration_ms": float("-inf")})
    assert "duration_ms" not in sanitize_audit_metadata({"duration_ms": -1.0})
    assert "duration_ms" not in sanitize_audit_metadata({"duration_ms": True})
    assert sanitize_audit_metadata({"duration_ms": 150.25}) == {"duration_ms": 150.25}
    assert sanitize_audit_metadata({"duration_ms": 0.0}) == {"duration_ms": 0.0}

    # 9. Valid existing identifiers and operational values combined
    valid_payload = {
        "status": "processing",
        "duration_ms": 85.5,
        "byte_count": 4096,
        "page_count": 2,
        "file_type": "application/pdf",
        "outcome_code": "SUCCESS",
        "error_code": "NONE",
        "rule_id": "R_TOTAL_CHECK",
        "schema_id": "casa_account_opening",
        "schema_version": 2,
        "attempt_count": 1,
    }
    sanitized = sanitize_audit_metadata(valid_payload)
    assert len(sanitized) == 11
    assert sanitized["status"] == "processing"
    assert sanitized["duration_ms"] == 85.5
    assert sanitized["byte_count"] == 4096
    assert sanitized["page_count"] == 2
    assert sanitized["file_type"] == "application/pdf"
    assert sanitized["outcome_code"] == "SUCCESS"
    assert sanitized["error_code"] == "NONE"
    assert sanitized["rule_id"] == "R_TOTAL_CHECK"
    assert sanitized["schema_id"] == "casa_account_opening"
    assert sanitized["schema_version"] == "2"
    assert sanitized["attempt_count"] == 1


@pytest.mark.unit
def test_audit_logger_emits_structured_record():
    """Verify SafeStructuredAuditLogger formats and emits structured events via logging."""
    import logging

    records = []

    class CapturingHandler(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = CapturingHandler()
    logger = logging.getLogger("test.doc_extractor.audit")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    audit_logger = SafeStructuredAuditLogger(logger)

    async def _run():
        await audit_logger.log_action(
            actor_id="user_admin",
            tenant_id="tenant_alpha",
            action="document.upload",
            resource_id="doc_123",
            resource_type="document",
            request_id="req_test_001",
            outcome="SUCCESS",
            metadata={"status": "queued", "byte_count": 2048, "secret_key": "leak"},
        )

    import asyncio
    asyncio.run(_run())

    assert len(records) >= 1
    rec = records[0]
    assert "AUDIT_EVENT: document.upload on document:doc_123 by user_admin [SUCCESS]" in rec.getMessage()
    extra_data = getattr(rec, "extra_data", {})
    assert extra_data["outcome"] == "SUCCESS"
    assert extra_data["metadata"]["status"] == "queued"
    assert extra_data["metadata"]["byte_count"] == 2048
    assert "secret_key" not in extra_data["metadata"]


@pytest.mark.security
def test_audit_logger_serialization_resilience():
    """Ensure audit logger handles unexpected serialization errors gracefully without crashing."""
    import logging

    records = []

    class CapturingHandler(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = CapturingHandler()
    logger = logging.getLogger("test.doc_extractor.resilience")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    audit_logger = SafeStructuredAuditLogger(logger)

    # Subclass with an un-serializable model_dump to simulate catastrophic serialization failure
    class BrokenEvent:
        event_id = "evt_broken"
        action = "test.action"
        request_id = "req_123"
        tenant_id = "t_1"
        metadata = {}
        def model_dump(self, mode="json"):
            raise TypeError("Catastrophic JSON serialization failure")

    async def _run():
        await audit_logger.log_event(BrokenEvent())

    import asyncio
    asyncio.run(_run())

    # Must log AUDIT_FAILURE without raising an unhandled exception
    assert any("AUDIT_FAILURE" in r.getMessage() for r in records)


# ============================================================================
# 12. CORS & Network Hardening
# ============================================================================

@pytest.mark.security
def test_cors_rejects_wildcard_with_credentials_in_settings():
    """Ensure ApiSettings rejects wildcard origin ('*') combined with allow_credentials=True."""
    with pytest.raises(ConfigurationError) as exc_info:
        load_api_settings({"cors_origins": ["*"], "cors_allow_credentials": True})
    assert "Wildcard CORS origin ('*') cannot be combined with credentials" in str(exc_info.value)


@pytest.mark.security
def test_cors_rejects_wildcard_in_production():
    """Ensure ApiSettings strictly forbids wildcard origin ('*') in production environments."""
    with pytest.raises(ConfigurationError) as exc_info:
        load_api_settings({
            "app_env": "production",
            "debug": False,
            "cors_origins": ["*"],
            "cors_allow_credentials": False,
        })
    assert "Wildcard CORS origin ('*') is strictly prohibited in production" in str(exc_info.value)


@pytest.mark.security
def test_cors_exposes_request_id_header():
    """Verify CORSMiddleware exposes X-Request-ID to browser clients via Access-Control-Expose-Headers."""
    settings = ApiSettings(
        app_env="development",
        cors_origins=["https://console.company.internal"],
        cors_allow_credentials=True,
    )
    app = create_app(settings)
    c = TestClient(app, raise_server_exceptions=False)
    response = c.get(
        "/health",
        headers={"Origin": "https://console.company.internal"},
    )
    assert response.status_code == 200
    exposed = response.headers.get("Access-Control-Expose-Headers", "")
    assert "X-Request-ID" in exposed


# ============================================================================
# 13. Dependency Declarations
# ============================================================================

@pytest.mark.unit
def test_runtime_dependencies_exclude_httpx():
    """Verify httpx is not declared as a production runtime dependency."""
    import tomllib
    with open("pyproject.toml", "rb") as f:
        data = tomllib.load(f)
    deps = data["project"]["dependencies"]
    assert not any(d.startswith("httpx") for d in deps), "httpx must be in dev dependencies only"
    dev_deps = data["project"]["optional-dependencies"]["dev"]
    assert any(d.startswith("httpx") for d in dev_deps), "httpx must be in dev dependencies"
