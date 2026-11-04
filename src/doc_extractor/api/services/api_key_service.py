"""
API Key Management Service (Module M5 TASK-5.7-B).

Adheres to:
- PRD §7.3 API-019: List, create (secret shown once), and revoke keys.
- PRD §10.1 SEC-002: CSPRNG generation, SHA-256 storage, 90-day rotation, per-key isolation.
- PRD §10.1 SEC-003: Scope subset check, privilege escalation prevention.
- PRD §10.1 SEC-006: Multi-tenant boundary isolation.
- PRD §10.1 SEC-012: Zero secret/hash logging in audit logs and API responses.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

from doc_extractor.api.contracts.records import ApiKeyCreationResult, ApiKeySummary
from doc_extractor.api.exceptions import (
    ApiValidationError,
    ForbiddenError,
    TenantResourceNotFoundError,
)
from doc_extractor.api.ports.audit import AuditLogger
from doc_extractor.api.ports.authentication import ApiPrincipal
from doc_extractor.api.ports.repositories import ApiKeyRepository
from doc_extractor.api.token_core import (
    DEFAULT_TOKEN_PREFIX,
    create_api_key_record,
    generate_api_secret,
    validate_api_key_roles,
    validate_api_key_scopes,
    validate_key_expiration,
)

logger = logging.getLogger("doc_extractor.services.api_key")


class ApiKeyService:
    """
    Backend-independent management service for tenant API keys.

    Enforces:
    1. Tenant isolation: All queries derive tenant_id strictly from the authenticated caller.
    2. Privilege escalation prevention: Creator cannot grant scopes or roles beyond their own grants.
    3. Non-privileged roles: Ordinary keys cannot receive 'admin' or 'schema_approver'.
    4. Wildcard prohibition: Global wildcard scope '*' is rejected.
    5. Maximum 90-day lifetime enforcement.
    6. Audit trail emission for creation and revocation.
    7. Single disclosure: Plaintext secret is returned only in ApiKeyCreationResult.
    """

    def __init__(
        self,
        repository: ApiKeyRepository,
        audit_logger: Optional[AuditLogger] = None,
    ) -> None:
        self._repository = repository
        self._audit_logger = audit_logger

    def _verify_creator_authz(
        self,
        creator: ApiPrincipal,
        required_scope: str,
        action_name: str,
    ) -> None:
        """
        Verifies creator holds explicit required API scope under default-deny.
        Role.ADMIN never substitutes for an explicitly granted API scope.
        """
        if not creator.has_scope(required_scope):
            raise ForbiddenError(
                message=f"Principal lacks required authorization '{required_scope}' to {action_name}.",
                code="FORBIDDEN",
                details={"required_scope": required_scope},
            )

    def _enforce_scope_and_role_subsets(
        self,
        creator: ApiPrincipal,
        requested_scopes: Sequence[str],
        requested_roles: Sequence[str],
    ) -> tuple[list[str], list[str]]:
        """
        Validates scopes and roles against security invariants and creator's authority.
        Prevents subordinate keys from gaining greater privilege than creator.
        All creators (including administrators) must possess the scopes they grant.
        """
        # 1. Base validation (blocks '*', checks formats)
        valid_scopes = validate_api_key_scopes(requested_scopes)

        # 2. Base role validation (blocks 'admin', 'schema_approver')
        valid_roles = validate_api_key_roles(requested_roles, allow_privileged_roles=False)

        # 3. Scope subset: NO creator (including admin) can grant scopes they do not hold
        for s in valid_scopes:
            if not creator.has_scope(s):
                raise ForbiddenError(
                    message=f"Cannot grant scope '{s}' which exceeds creator's authorized scopes.",
                    code="FORBIDDEN_SCOPE",
                    details={"unauthorized_scope": s},
                )

        # 4. Role subset: Non-admin creators cannot grant roles they do not hold
        if "admin" not in creator.roles:
            for r in valid_roles:
                if r not in creator.roles:
                    raise ForbiddenError(
                        message=f"Cannot grant role '{r}' which exceeds creator's authorized roles.",
                        code="FORBIDDEN",
                        details={"unauthorized_role": r},
                    )

        return sorted(valid_scopes), sorted(valid_roles)

    async def create_api_key(
        self,
        creator: ApiPrincipal,
        name: str,
        scopes: Sequence[str],
        roles: Sequence[str],
        expires_at: Optional[datetime] = None,
        prefix: str = DEFAULT_TOKEN_PREFIX,
    ) -> ApiKeyCreationResult:
        """
        Create a new scoped API key for the creator's tenant.

        Security Guarantees:
        - Plaintext secret token is returned in ApiKeyCreationResult exactly once.
        - Only cryptographic SHA-256 hash is persisted.
        - Creator cannot escalate privileges beyond their own grants.
        - Maximum lifetime cannot exceed 90 days.
        """
        # 1. Authorize creator
        self._verify_creator_authz(creator, required_scope="api_keys:write", action_name="create API keys")

        if not name or not name.strip():
            raise ApiValidationError(
                message="API key name must be a non-empty string.",
                code="VALIDATION_ERROR",
                details={"field": "name"},
            )

        # 2. Validate and enforce scope/role subsets
        clean_scopes, clean_roles = self._enforce_scope_and_role_subsets(creator, scopes, roles)

        # 3. Validate expiration (max 90 days, UTC)
        now_utc = datetime.now(timezone.utc)
        valid_exp = validate_key_expiration(expires_at, created_at=now_utc)

        # 4. Generate 256-bit CSPRNG token
        token = generate_api_secret(nbytes=32, prefix=prefix)

        # 5. Build and persist ApiKeyRecord (strictly hash-only)
        record = create_api_key_record(
            token=token,
            tenant_id=creator.tenant_id,
            name=name.strip(),
            scopes=clean_scopes,
            roles=clean_roles,
            expires_at=valid_exp,
            created_by=creator.principal_id,
            created_at=now_utc,
            allow_privileged_roles=False,
        )

        persisted = await self._repository.create(creator.tenant_id, record)

        # 6. Audit event (SEC-012: no secret, no hash)
        if self._audit_logger:
            try:
                await self._audit_logger.log_action(
                    actor_id=creator.principal_id,
                    tenant_id=creator.tenant_id,
                    action="api_key.created",
                    resource_id=persisted.key_id,
                    resource_type="api_key",
                    metadata={"outcome_code": "SUCCESS"},
                    outcome="SUCCESS",
                )
            except Exception as exc:  # pragma: no cover
                logger.warning("Audit logging failed for api_key.created: %s", exc)

        # 7. Return creation result disclosing secret ONCE
        return ApiKeyCreationResult(
            key_id=persisted.key_id,
            tenant_id=persisted.tenant_id,
            name=persisted.name,
            key_prefix=persisted.key_prefix,
            scopes=list(persisted.scopes),
            roles=list(persisted.roles),
            created_at=persisted.created_at,
            expires_at=persisted.expires_at,
            created_by=persisted.created_by,
            raw_secret=token.raw_secret,
        )

    async def list_api_keys(
        self,
        creator: ApiPrincipal,
        limit: int = 50,
        offset: int = 0,
    ) -> List[ApiKeySummary]:
        """
        List all API keys belonging to the creator's tenant.

        Security Guarantees:
        - Multi-tenant isolation: strictly filters by creator.tenant_id.
        - Zero secret disclosure: outputs ApiKeySummary (omits raw secrets and hashes).
        """
        # Authorize creator: explicit 'api_keys:read' scope required (SEC-002, SEC-003)
        self._verify_creator_authz(creator, required_scope="api_keys:read", action_name="list API keys")

        records = await self._repository.list_keys(
            tenant_id=creator.tenant_id,
            limit=limit,
            offset=offset,
        )
        return [ApiKeySummary.from_record(r) for r in records]

    async def get_api_key(
        self,
        creator: ApiPrincipal,
        key_id: str,
    ) -> ApiKeySummary:
        """
        Retrieve a single API key metadata record within creator's tenant.

        Conceals existence of keys belonging to other tenants by raising TenantResourceNotFoundError.
        """
        # Authorize creator: explicit 'api_keys:read' scope required (SEC-002, SEC-003)
        self._verify_creator_authz(creator, required_scope="api_keys:read", action_name="inspect API keys")

        record = await self._repository.get_by_id(creator.tenant_id, key_id)
        if record is None:
            raise TenantResourceNotFoundError(
                message="API key not found.",
                code="DOCUMENT_NOT_FOUND",
            )

        return ApiKeySummary.from_record(record)

    async def revoke_api_key(
        self,
        creator: ApiPrincipal,
        key_id: str,
    ) -> bool:
        """
        Revoke an active API key within creator's tenant immediately.

        Conceals cross-tenant existence: Returns 404 DOCUMENT_NOT_FOUND if key not in tenant.
        """
        self._verify_creator_authz(creator, required_scope="api_keys:write", action_name="revoke API keys")

        record = await self._repository.get_by_id(creator.tenant_id, key_id)
        if record is None:
            raise TenantResourceNotFoundError(
                message="API key not found.",
                code="DOCUMENT_NOT_FOUND",
            )

        success = await self._repository.revoke(creator.tenant_id, key_id)

        if self._audit_logger:
            try:
                await self._audit_logger.log_action(
                    actor_id=creator.principal_id,
                    tenant_id=creator.tenant_id,
                    action="api_key.revoked",
                    resource_id=key_id,
                    resource_type="api_key",
                    metadata={"outcome_code": "SUCCESS"},
                    outcome="SUCCESS",
                )
            except Exception as exc:  # pragma: no cover
                logger.warning("Audit logging failed for api_key.revoked: %s", exc)

        return success

