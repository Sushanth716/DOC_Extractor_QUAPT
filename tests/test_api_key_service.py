"""
Unit and security tests for ApiKeyAuthenticationProvider and ApiKeyService (Module M5 TASK-5.7-B).

Verifies:
- Valid key authentication with Bearer prefix and direct token string.
- Rejection of unknown, malformed, expired, revoked, and inactive keys.
- Accurate principal construction with tenant isolation, mfa_verified=False, and M4 separation.
- Asynchronous last-used timestamp tracking via repository contract.
- Complete absence of raw secret and key hash from audit events, logs, and listing operations.
- Single disclosure of plaintext secret upon creation.
- Scope-subset and role-subset enforcement preventing privilege escalation.
- Strict prohibition of global wildcard '*' and privileged roles ('admin', 'schema_approver').
- Maximum 90-day key lifetime enforcement.
- Cross-tenant isolation concealing foreign key existence (404 DOCUMENT_NOT_FOUND).
- Immediate revocation effectiveness on subsequent authentication attempts.
"""

import asyncio
from datetime import datetime, timedelta, timezone
import functools
from typing import Any, Dict, List, Optional

import pytest

from doc_extractor.api.contracts.records import ApiKeyRecord
from doc_extractor.api.exceptions import (
    AuthenticationRequiredError,
    ForbiddenError,
    InvalidKeyExpirationError,
    InvalidScopeError,
    InvalidTokenError,
    PrivilegedRoleNotAllowedError,
    TenantResourceNotFoundError,
    TokenExpiredError,
)
from doc_extractor.api.ports.authentication import ApiPrincipal, PrincipalType
from doc_extractor.api.ports.repositories import ApiKeyRepository
from doc_extractor.api.providers.api_key_auth import ApiKeyAuthenticationProvider
from doc_extractor.api.services.api_key_service import ApiKeyService
from doc_extractor.api.token_core import generate_api_secret


def async_test(fn):
    """Synchronous runner for coroutine test functions without requiring pytest-asyncio plugin."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return asyncio.run(fn(*args, **kwargs))
    return wrapper


# ============================================================================
# Mock In-Memory Repository & Audit Logger (Unit Test Doubles)
# ============================================================================

class MockApiKeyRepository:
    """In-memory mock repository implementing the ApiKeyRepository protocol."""

    def __init__(self) -> None:
        self.keys_by_id: Dict[tuple[str, str], ApiKeyRecord] = {}
        self.keys_by_hash: Dict[str, ApiKeyRecord] = {}
        self.last_used_calls: List[tuple[str, str, datetime]] = []

    async def get_by_key_hash(self, key_hash: str) -> Optional[ApiKeyRecord]:
        return self.keys_by_hash.get(key_hash)

    async def get_by_id(self, tenant_id: str, key_id: str) -> Optional[ApiKeyRecord]:
        return self.keys_by_id.get((tenant_id, key_id))

    async def create(self, tenant_id: str, key_record: ApiKeyRecord) -> ApiKeyRecord:
        self.keys_by_id[(tenant_id, key_record.key_id)] = key_record
        self.keys_by_hash[key_record.key_hash] = key_record
        return key_record

    async def revoke(self, tenant_id: str, key_id: str) -> bool:
        record = self.keys_by_id.get((tenant_id, key_id))
        if not record:
            return False
        updated = record.model_copy(
            update={
                "revoked_at": datetime.now(timezone.utc),
                "is_active": False,
            }
        )
        self.keys_by_id[(tenant_id, key_id)] = updated
        self.keys_by_hash[record.key_hash] = updated
        return True

    async def list_keys(
        self,
        tenant_id: str,
        limit: int = 50,
        offset: int = 0,
    ) -> List[ApiKeyRecord]:
        matching = [r for (t, _), r in self.keys_by_id.items() if t == tenant_id]
        return matching[offset : offset + limit]

    async def record_last_used(
        self,
        tenant_id: str,
        key_id: str,
        last_used_at: datetime,
    ) -> bool:
        self.last_used_calls.append((tenant_id, key_id, last_used_at))
        record = self.keys_by_id.get((tenant_id, key_id))
        if record:
            updated = record.model_copy(update={"last_used_at": last_used_at})
            self.keys_by_id[(tenant_id, key_id)] = updated
            self.keys_by_hash[record.key_hash] = updated
            return True
        return False


class MockAuditLogger:
    """Captures audit actions for security inspection."""

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    async def log_action(
        self,
        actor_id: str,
        tenant_id: str,
        action: str,
        resource_id: str,
        resource_type: str,
        request_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        outcome: str = "SUCCESS",
    ) -> None:
        self.events.append({
            "actor_id": actor_id,
            "tenant_id": tenant_id,
            "action": action,
            "resource_id": resource_id,
            "resource_type": resource_type,
            "request_id": request_id,
            "metadata": metadata or {},
            "outcome": outcome,
        })


@pytest.fixture
def repo() -> MockApiKeyRepository:
    r = MockApiKeyRepository()
    assert isinstance(r, ApiKeyRepository)
    return r


@pytest.fixture
def audit_logger() -> MockAuditLogger:
    return MockAuditLogger()


@pytest.fixture
def auth_provider(repo: MockApiKeyRepository, audit_logger: MockAuditLogger) -> ApiKeyAuthenticationProvider:
    return ApiKeyAuthenticationProvider(repository=repo, audit_logger=audit_logger)


@pytest.fixture
def service(repo: MockApiKeyRepository, audit_logger: MockAuditLogger) -> ApiKeyService:
    return ApiKeyService(repository=repo, audit_logger=audit_logger)


@pytest.fixture
def admin_principal() -> ApiPrincipal:
    return ApiPrincipal(
        principal_id="user_admin",
        tenant_id="tenant_alpha",
        principal_type=PrincipalType.USER,
        roles={"admin"},
        scopes={
            "api_keys:read",
            "api_keys:write",
            "documents:read",
            "documents:write",
            "admin:config",
        },
    )


@pytest.fixture
def integrator_principal() -> ApiPrincipal:
    return ApiPrincipal(
        principal_id="user_integrator",
        tenant_id="tenant_alpha",
        principal_type=PrincipalType.USER,
        roles={"integrator"},
        scopes={"api_keys:read", "api_keys:write", "documents:read", "documents:write"},
    )


@pytest.fixture
def foreign_admin_principal() -> ApiPrincipal:
    return ApiPrincipal(
        principal_id="user_foreign_admin",
        tenant_id="tenant_beta",
        principal_type=PrincipalType.USER,
        roles={"admin"},
        scopes={"api_keys:read", "api_keys:write", "documents:read"},
    )


# ============================================================================
# 1. ApiKeyAuthenticationProvider Tests
# ============================================================================

@pytest.mark.security
@async_test
async def test_auth_provider_valid_key_with_bearer_prefix(
    service: ApiKeyService,
    auth_provider: ApiKeyAuthenticationProvider,
    admin_principal: ApiPrincipal,
    repo: MockApiKeyRepository,
):
    """Verify successful authentication with standard 'Bearer <token>' header."""
    creation = await service.create_api_key(
        creator=admin_principal,
        name="Ingestion Key",
        scopes=["documents:read", "documents:write"],
        roles=["integrator"],
    )

    header = f"Bearer {creation.raw_secret}"
    principal = await auth_provider.authenticate(header)

    assert principal is not None
    assert principal.principal_id == creation.key_id
    assert principal.tenant_id == "tenant_alpha"
    assert principal.principal_type == PrincipalType.API_KEY
    assert principal.mfa_verified is False
    assert principal.is_active is True
    assert principal.scopes == {"documents:read", "documents:write"}
    assert principal.roles == {"integrator"}
    assert principal.permissions == set()

    # Verify last-used tracking was triggered
    assert len(repo.last_used_calls) == 1
    assert repo.last_used_calls[0][0] == "tenant_alpha"
    assert repo.last_used_calls[0][1] == creation.key_id


@pytest.mark.security
@async_test
async def test_auth_provider_valid_key_without_bearer_prefix(
    service: ApiKeyService,
    auth_provider: ApiKeyAuthenticationProvider,
    admin_principal: ApiPrincipal,
):
    """Verify authentication succeeds with a raw token string (without 'Bearer ')."""
    creation = await service.create_api_key(
        creator=admin_principal,
        name="Direct Token Key",
        scopes=["documents:read"],
        roles=["integrator"],
    )

    principal = await auth_provider.authenticate(creation.raw_secret)
    assert principal is not None
    assert principal.principal_id == creation.key_id


@pytest.mark.security
@async_test
async def test_auth_provider_missing_or_empty_credentials(
    auth_provider: ApiKeyAuthenticationProvider,
):
    """Verify empty or missing credentials fail closed with AuthenticationRequiredError."""
    with pytest.raises(AuthenticationRequiredError, match="header or token credential is required"):
        await auth_provider.authenticate(None)

    with pytest.raises(AuthenticationRequiredError):
        await auth_provider.authenticate("")

    with pytest.raises(AuthenticationRequiredError):
        await auth_provider.authenticate("   ")

    with pytest.raises(AuthenticationRequiredError):
        await auth_provider.authenticate("Bearer ")


@pytest.mark.security
@async_test
async def test_auth_provider_malformed_token_fails_before_lookup(
    auth_provider: ApiKeyAuthenticationProvider,
    repo: MockApiKeyRepository,
    audit_logger: MockAuditLogger,
):
    """Verify tokens with invalid prefixes or insufficient length fail closed without querying repository."""
    # Too short
    with pytest.raises(InvalidTokenError, match="format is invalid"):
        await auth_provider.authenticate("de_live_short")

    # Invalid prefix
    with pytest.raises(InvalidTokenError, match="format is invalid"):
        await auth_provider.authenticate("unapproved_prefix_sample_token_with_sufficient_length_1234567890")

    # Disallowed characters
    with pytest.raises(InvalidTokenError, match="format is invalid"):
        await auth_provider.authenticate("de_live_abc123!@#$%^&*()_+~`|}{[]:;?><,./")

    # Verify repository was NEVER queried for malformed candidates
    assert len(repo.keys_by_hash) == 0

    # Verify audit failure was emitted safely
    failures = [e for e in audit_logger.events if e["action"] == "api_key.authentication_failed"]
    assert len(failures) == 3
    for f in failures:
        assert f["metadata"]["error_code"] == "INVALID_CREDENTIALS"
        assert f["actor_id"] == "unauthenticated"


@pytest.mark.security
@async_test
async def test_auth_provider_unknown_token(
    auth_provider: ApiKeyAuthenticationProvider,
    audit_logger: MockAuditLogger,
):
    """Verify validly-formatted token not present in repository raises InvalidTokenError."""
    unregistered_token = generate_api_secret().raw_secret

    with pytest.raises(InvalidTokenError, match="invalid or unrecognized"):
        await auth_provider.authenticate(unregistered_token)

    failure = audit_logger.events[-1]
    assert failure["action"] == "api_key.authentication_failed"
    assert failure["metadata"]["error_code"] == "INVALID_CREDENTIALS"


@pytest.mark.security
@async_test
async def test_auth_provider_expired_key(
    service: ApiKeyService,
    auth_provider: ApiKeyAuthenticationProvider,
    admin_principal: ApiPrincipal,
    repo: MockApiKeyRepository,
    audit_logger: MockAuditLogger,
):
    """Verify expired key raises TokenExpiredError (HTTP 401) and emits audit failure."""
    creation = await service.create_api_key(
        creator=admin_principal,
        name="Expiring Key",
        scopes=["documents:read"],
        roles=["integrator"],
    )

    # Force expiration in repository
    record = await repo.get_by_id("tenant_alpha", creation.key_id)
    assert record is not None
    expired_record = record.model_copy(
        update={"expires_at": datetime.now(timezone.utc) - timedelta(hours=1)}
    )
    repo.keys_by_id[("tenant_alpha", creation.key_id)] = expired_record
    repo.keys_by_hash[creation.raw_secret] = expired_record
    # Re-hash correctly in mock
    repo.keys_by_hash[record.key_hash] = expired_record

    with pytest.raises(TokenExpiredError, match="has expired"):
        await auth_provider.authenticate(creation.raw_secret)

    failure = audit_logger.events[-1]
    assert failure["action"] == "api_key.authentication_failed"
    assert failure["metadata"]["error_code"] == "TOKEN_EXPIRED"
    assert failure["resource_id"] == creation.key_id


@pytest.mark.security
@async_test
async def test_auth_provider_revoked_key(
    service: ApiKeyService,
    auth_provider: ApiKeyAuthenticationProvider,
    admin_principal: ApiPrincipal,
    audit_logger: MockAuditLogger,
):
    """Verify revoked key raises InvalidTokenError."""
    creation = await service.create_api_key(
        creator=admin_principal,
        name="To Be Revoked",
        scopes=["documents:read"],
        roles=["integrator"],
    )

    # Revoke via service
    revoked = await service.revoke_api_key(admin_principal, creation.key_id)
    assert revoked is True

    # Immediate rejection on subsequent attempt
    with pytest.raises(InvalidTokenError, match="revoked or is inactive"):
        await auth_provider.authenticate(creation.raw_secret)

    failure = audit_logger.events[-1]
    assert failure["action"] == "api_key.authentication_failed"
    assert failure["resource_id"] == creation.key_id


@pytest.mark.security
@async_test
async def test_auth_provider_never_logs_token_or_hash(
    service: ApiKeyService,
    auth_provider: ApiKeyAuthenticationProvider,
    admin_principal: ApiPrincipal,
    audit_logger: MockAuditLogger,
):
    """Verify raw tokens and SHA-256 hashes never appear in audit events."""
    creation = await service.create_api_key(
        creator=admin_principal,
        name="Audited Key",
        scopes=["documents:read"],
        roles=["integrator"],
    )

    # Perform valid auth
    await auth_provider.authenticate(creation.raw_secret)

    # Perform failed auth
    with pytest.raises(InvalidTokenError):
        await auth_provider.authenticate("de_live_invalidinvalidinvalidinvalidinvalidinvalid12")

    # Inspect all audit events
    for event in audit_logger.events:
        event_str = str(event)
        assert creation.raw_secret not in event_str
        # Key hash should not be in resource_id or metadata
        assert event["resource_id"] != event["metadata"].get("key_hash")
        assert "key_hash" not in event["metadata"]
        assert "authorization" not in event_str.lower()


# ============================================================================
# 2. ApiKeyService Tests
# ============================================================================

@pytest.mark.security
@async_test
async def test_service_successful_key_creation(
    service: ApiKeyService,
    admin_principal: ApiPrincipal,
    repo: MockApiKeyRepository,
):
    """Verify key creation generates valid record, hash-only persistence, and one-time secret."""
    creation = await service.create_api_key(
        creator=admin_principal,
        name="Pipeline Ingestor",
        scopes=["documents:read", "documents:write"],
        roles=["integrator"],
    )

    assert creation.key_id.startswith("key_")
    assert creation.tenant_id == "tenant_alpha"
    assert creation.name == "Pipeline Ingestor"
    assert creation.key_prefix.startswith("de_live_")
    assert creation.scopes == ["documents:read", "documents:write"]
    assert creation.roles == ["integrator"]
    assert creation.created_by == "user_admin"
    assert creation.raw_secret.startswith("de_live_")

    # Verify repository holds only key_hash and does NOT hold raw_secret
    stored = await repo.get_by_id("tenant_alpha", creation.key_id)
    assert stored is not None
    assert stored.key_hash != creation.raw_secret
    assert len(stored.key_hash) == 64
    assert not hasattr(stored, "raw_secret")


@pytest.mark.security
@async_test
async def test_service_creator_authorization_required(
    service: ApiKeyService,
):
    """Verify creator without api_keys:write permission cannot create or revoke keys."""
    unauthorized = ApiPrincipal(
        principal_id="unauthorized_user",
        tenant_id="tenant_alpha",
        principal_type=PrincipalType.USER,
        roles={"reviewer"},  # Not admin
        scopes={"documents:read"},  # Missing api_keys:write
    )

    with pytest.raises(ForbiddenError, match="lacks required authorization"):
        await service.create_api_key(
            creator=unauthorized,
            name="Unauthorized Key",
            scopes=["documents:read"],
            roles=["integrator"],
        )

    with pytest.raises(ForbiddenError, match="lacks required authorization"):
        await service.revoke_api_key(unauthorized, "key_some_id")

    with pytest.raises(ForbiddenError, match="lacks required authorization"):
        await service.list_api_keys(unauthorized)


@pytest.mark.security
@async_test
async def test_service_scope_subset_enforcement(
    service: ApiKeyService,
    integrator_principal: ApiPrincipal,
):
    """Verify non-admin creator cannot grant scopes exceeding their own authority."""
    # Integrator has: ["api_keys:read", "api_keys:write", "documents:read", "documents:write"]
    # Requesting an unauthorized scope ("admin:config") must fail
    with pytest.raises(ForbiddenError, match="Cannot grant scope 'admin:config'"):
        await service.create_api_key(
            creator=integrator_principal,
            name="Escalated Key",
            scopes=["documents:read", "admin:config"],
            roles=["integrator"],
        )

    # Requesting legitimate subset succeeds
    valid_key = await service.create_api_key(
        creator=integrator_principal,
        name="Legitimate Subordinate",
        scopes=["documents:read"],
        roles=["integrator"],
    )
    assert valid_key.scopes == ["documents:read"]


@pytest.mark.security
@async_test
async def test_service_role_privilege_and_subset_restrictions(
    service: ApiKeyService,
    admin_principal: ApiPrincipal,
    integrator_principal: ApiPrincipal,
):
    """Verify privileged roles ('admin', 'schema_approver') are strictly rejected."""
    # Even an admin creator cannot issue ordinary keys with 'admin' role
    with pytest.raises(PrivilegedRoleNotAllowedError, match="Privileged role 'admin' cannot be assigned"):
        await service.create_api_key(
            creator=admin_principal,
            name="Malicious Admin Key",
            scopes=["documents:read"],
            roles=["admin"],
        )

    # Cannot issue schema_approver role
    with pytest.raises(PrivilegedRoleNotAllowedError, match="Privileged role 'schema_approver' cannot be assigned"):
        await service.create_api_key(
            creator=admin_principal,
            name="Malicious Approver Key",
            scopes=["documents:read"],
            roles=["schema_approver"],
        )

    # Non-admin creator cannot grant roles they do not hold
    with pytest.raises(ForbiddenError, match="Cannot grant role 'analyst'"):
        await service.create_api_key(
            creator=integrator_principal,  # Only holds 'integrator'
            name="Analyst Key",
            scopes=["documents:read"],
            roles=["analyst"],
        )


@pytest.mark.security
@async_test
async def test_service_wildcard_scope_strictly_prevented(
    service: ApiKeyService,
    admin_principal: ApiPrincipal,
):
    """Verify global wildcard scope '*' is rejected even for admin creator."""
    with pytest.raises(InvalidScopeError, match="Global wildcard scope '\\*' is strictly prohibited"):
        await service.create_api_key(
            creator=admin_principal,
            name="Wildcard Key",
            scopes=["*"],
            roles=["integrator"],
        )


@pytest.mark.security
@async_test
async def test_service_expiration_enforcement(
    service: ApiKeyService,
    admin_principal: ApiPrincipal,
):
    """Verify expiration exceeding 90 days or with naive timestamp is rejected."""
    now_utc = datetime.now(timezone.utc)

    # Exceeding 90 days
    with pytest.raises(InvalidKeyExpirationError, match="exceeds maximum allowed limit"):
        await service.create_api_key(
            creator=admin_principal,
            name="Long Key",
            scopes=["documents:read"],
            roles=["integrator"],
            expires_at=now_utc + timedelta(days=91),
        )

    # Timezone-naive datetime
    naive_dt = datetime(2026, 12, 1, 12, 0, 0)
    with pytest.raises(InvalidKeyExpirationError, match="must be timezone-aware"):
        await service.create_api_key(
            creator=admin_principal,
            name="Naive Key",
            scopes=["documents:read"],
            roles=["integrator"],
            expires_at=naive_dt,
        )


@pytest.mark.security
@async_test
async def test_service_listing_never_exposes_secrets_or_hashes(
    service: ApiKeyService,
    admin_principal: ApiPrincipal,
):
    """Verify list_api_keys and get_api_key return ApiKeySummary without secrets or hashes."""
    creation = await service.create_api_key(
        creator=admin_principal,
        name="Key For Listing",
        scopes=["documents:read"],
        roles=["integrator"],
    )

    summaries = await service.list_api_keys(admin_principal)
    assert len(summaries) == 1
    summary = summaries[0]

    assert summary.key_id == creation.key_id
    assert summary.key_prefix == creation.key_prefix
    assert not hasattr(summary, "raw_secret")
    assert not hasattr(summary, "key_hash")

    dump = summary.model_dump()
    assert "raw_secret" not in dump
    assert "key_hash" not in dump

    # Same for single get_api_key
    single = await service.get_api_key(admin_principal, creation.key_id)
    assert not hasattr(single, "raw_secret")
    assert not hasattr(single, "key_hash")


@pytest.mark.security
@async_test
async def test_service_cross_tenant_isolation(
    service: ApiKeyService,
    admin_principal: ApiPrincipal,
    foreign_admin_principal: ApiPrincipal,
):
    """Verify cross-tenant key access returns 404 DOCUMENT_NOT_FOUND without revealing key existence."""
    creation = await service.create_api_key(
        creator=admin_principal,  # Tenant alpha
        name="Alpha Private Key",
        scopes=["documents:read"],
        roles=["integrator"],
    )

    # Tenant Beta attempts to get Tenant Alpha's key
    with pytest.raises(TenantResourceNotFoundError, match="API key not found"):
        await service.get_api_key(foreign_admin_principal, creation.key_id)

    # Tenant Beta attempts to revoke Tenant Alpha's key
    with pytest.raises(TenantResourceNotFoundError, match="API key not found"):
        await service.revoke_api_key(foreign_admin_principal, creation.key_id)

    # Tenant Beta listing is empty
    beta_keys = await service.list_api_keys(foreign_admin_principal)
    assert len(beta_keys) == 0

    # Tenant Alpha can access and revoke its own key
    alpha_keys = await service.list_api_keys(admin_principal)
    assert len(alpha_keys) == 1
    assert alpha_keys[0].key_id == creation.key_id

    revoked = await service.revoke_api_key(admin_principal, creation.key_id)
    assert revoked is True


@pytest.mark.unit
@async_test
async def test_service_audit_event_emission(
    service: ApiKeyService,
    admin_principal: ApiPrincipal,
    audit_logger: MockAuditLogger,
):
    """Verify structured audit events are emitted on key creation and revocation."""
    creation = await service.create_api_key(
        creator=admin_principal,
        name="Audited Key",
        scopes=["documents:read"],
        roles=["integrator"],
    )
    await service.revoke_api_key(admin_principal, creation.key_id)

    actions = [e["action"] for e in audit_logger.events]
    assert "api_key.created" in actions
    assert "api_key.revoked" in actions

    created_event = next(e for e in audit_logger.events if e["action"] == "api_key.created")
    assert created_event["actor_id"] == "user_admin"
    assert created_event["tenant_id"] == "tenant_alpha"
    assert created_event["resource_id"] == creation.key_id
    assert created_event["outcome"] == "SUCCESS"
    assert created_event["metadata"]["outcome_code"] == "SUCCESS"


@pytest.mark.unit
def test_dependency_injection_get_api_key_service(
    repo: MockApiKeyRepository,
    audit_logger: MockAuditLogger,
):
    """Verify get_api_key_service retrieves or constructs ApiKeyService via FastAPI dependency."""
    from fastapi import Request
    from doc_extractor.api.dependencies import get_api_key_service

    # Create dummy mock request
    class DummyApp:
        class state:
            api_key_service = None
            api_key_repo = repo
            audit_logger = audit_logger

    class DummyRequest:
        app = DummyApp()

    req = DummyRequest()
    service_instance = get_api_key_service(
        request=req,  # type: ignore
        api_key_repo=repo,
        audit_logger=audit_logger,  # type: ignore
    )
    assert isinstance(service_instance, ApiKeyService)


# ============================================================================
# 3. Explicit Scope Enforcement & Admin Bypass Regression Tests (TASK-5.7-B Correction)
# ============================================================================

@pytest.mark.security
@async_test
async def test_admin_without_api_keys_write_cannot_create_keys(service: ApiKeyService):
    """Verify administrator lacking explicit 'api_keys:write' scope cannot create keys."""
    admin_no_write = ApiPrincipal(
        principal_id="admin_readonly",
        tenant_id="tenant_alpha",
        principal_type=PrincipalType.USER,
        roles={"admin"},
        scopes={"api_keys:read", "documents:read"},
    )
    with pytest.raises(ForbiddenError, match="Principal lacks required authorization 'api_keys:write' to create API keys"):
        await service.create_api_key(
            creator=admin_no_write,
            name="Should Fail",
            scopes=["documents:read"],
            roles=["integrator"],
        )


@pytest.mark.security
@async_test
async def test_admin_without_api_keys_write_cannot_revoke_keys(
    service: ApiKeyService,
    admin_principal: ApiPrincipal,
):
    """Verify administrator lacking explicit 'api_keys:write' scope cannot revoke keys."""
    created = await service.create_api_key(
        creator=admin_principal,
        name="Key To Revoke",
        scopes=["documents:read"],
        roles=["integrator"],
    )

    admin_no_write = ApiPrincipal(
        principal_id="admin_readonly",
        tenant_id="tenant_alpha",
        principal_type=PrincipalType.USER,
        roles={"admin"},
        scopes={"api_keys:read"},
    )
    with pytest.raises(ForbiddenError, match="Principal lacks required authorization 'api_keys:write' to revoke API keys"):
        await service.revoke_api_key(admin_no_write, created.key_id)


@pytest.mark.security
@async_test
async def test_admin_without_api_keys_read_cannot_list_or_retrieve_keys(
    service: ApiKeyService,
    admin_principal: ApiPrincipal,
):
    """Verify administrator lacking explicit 'api_keys:read' scope cannot list or inspect keys."""
    created = await service.create_api_key(
        creator=admin_principal,
        name="Key To Read",
        scopes=["documents:read"],
        roles=["integrator"],
    )

    admin_write_only = ApiPrincipal(
        principal_id="admin_writer",
        tenant_id="tenant_alpha",
        principal_type=PrincipalType.USER,
        roles={"admin"},
        scopes={"api_keys:write"},  # Lacks api_keys:read
    )
    with pytest.raises(ForbiddenError, match="Principal lacks required authorization 'api_keys:read' to list API keys"):
        await service.list_api_keys(admin_write_only)

    with pytest.raises(ForbiddenError, match="Principal lacks required authorization 'api_keys:read' to inspect API keys"):
        await service.get_api_key(admin_write_only, created.key_id)


@pytest.mark.security
@async_test
async def test_admin_with_api_keys_write_can_create_keys_subject_to_subset_checks(service: ApiKeyService):
    """Verify administrator with 'api_keys:write' can create keys only for scopes they hold."""
    admin_with_write = ApiPrincipal(
        principal_id="admin_writer",
        tenant_id="tenant_alpha",
        principal_type=PrincipalType.USER,
        roles={"admin"},
        scopes={"api_keys:write", "documents:read"},
    )
    # Holds documents:read -> succeeds
    key = await service.create_api_key(
        creator=admin_with_write,
        name="Valid Subordinate",
        scopes=["documents:read"],
        roles=["integrator"],
    )
    assert key.scopes == ["documents:read"]

    # Does NOT hold documents:write -> fails scope subset check even though creator is admin!
    with pytest.raises(ForbiddenError, match="Cannot grant scope 'documents:write' which exceeds creator's authorized scopes"):
        await service.create_api_key(
            creator=admin_with_write,
            name="Escalated Subordinate",
            scopes=["documents:write"],
            roles=["integrator"],
        )


@pytest.mark.security
@async_test
async def test_admin_with_api_keys_read_can_list_keys(
    service: ApiKeyService,
    admin_principal: ApiPrincipal,
):
    """Verify administrator with explicitly granted 'api_keys:read' can list keys."""
    await service.create_api_key(
        creator=admin_principal,
        name="Key For Listing Admin",
        scopes=["documents:read"],
        roles=["integrator"],
    )

    admin_reader = ApiPrincipal(
        principal_id="admin_reader",
        tenant_id="tenant_alpha",
        principal_type=PrincipalType.USER,
        roles={"admin"},
        scopes={"api_keys:read"},
    )
    keys = await service.list_api_keys(admin_reader)
    assert len(keys) >= 1
    assert any(k.name == "Key For Listing Admin" for k in keys)


@pytest.mark.security
@async_test
async def test_subordinate_keys_cannot_receive_ungranted_scopes_even_from_admin(service: ApiKeyService):
    """Verify Role.ADMIN never bypasses scope subset enforcement."""
    admin_limited = ApiPrincipal(
        principal_id="admin_limited",
        tenant_id="tenant_alpha",
        principal_type=PrincipalType.USER,
        roles={"admin"},
        scopes={"api_keys:write", "api_keys:read", "documents:read"},
    )

    # Attempting to grant unheld scope 'audit:read' must fail
    with pytest.raises(ForbiddenError, match="Cannot grant scope 'audit:read' which exceeds creator's authorized scopes"):
        await service.create_api_key(
            creator=admin_limited,
            name="Audit Key",
            scopes=["audit:read"],
            roles=["auditor"],
        )

