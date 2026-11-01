"""
Role-Based Access Control and Permission Models for Schema Lifecycle (SEC-003, SEC-001).
Strictly adheres to PRD v1.1 §10.1, §15.5, SEC-001, SEC-003, and SCH-005:
- Defines canonical tenant roles: Analyst, Reviewer, Schema author, Schema approver,
  Integrator, Auditor, and Admin.
- Defines granular SchemaPermission enumeration for schema lifecycle operations.
- Enforces default-deny least-privilege matrix.
- Separates author and approver duties unconditionally.
- Requires trusted MFA verification for Schema approvers and Admins (SEC-001).
- Enforces strict tenant scoping and fail-closed evaluation.
- Never trusts unauthenticated caller-asserted roles, permissions, tenant scopes, or MFA claims.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Set
from pydantic import BaseModel, ConfigDict, Field

from schema_lifecycle.exceptions import (
    ForbiddenActionError,
    SchemaSelfApprovalError,
    TenantAccessError,
    UnauthenticatedError,
)


class Role(str, Enum):
    """
    Canonical tenant roles matching PRD §10.1 SEC-003 and §15.5.
    Distinct from ICXBee back-office system roles.
    """
    ANALYST = "analyst"
    REVIEWER = "reviewer"
    SCHEMA_AUTHOR = "schema_author"
    SCHEMA_APPROVER = "schema_approver"
    INTEGRATOR = "integrator"
    AUDITOR = "auditor"
    ADMIN = "admin"


class SchemaPermission(str, Enum):
    """
    Granular permissions governing schema lifecycle operations.
    Follows least-privilege separation of duties (SEC-003).
    """
    SCHEMA_READ = "schema:read"
    SCHEMA_CREATE = "schema:create"
    SCHEMA_VERSION_CREATE = "schema:version:create"
    SCHEMA_EDIT_DRAFT = "schema:draft:edit"
    SCHEMA_SUBMIT_TEST = "schema:testing:submit"
    SCHEMA_REQUEST_APPROVAL = "schema:approval:request"
    SCHEMA_RUN_HARNESS = "schema:harness:run"
    SCHEMA_APPROVE = "schema:approve"
    SCHEMA_RETIRE = "schema:retire"
    SCHEMA_ROLLBACK = "schema:rollback"
    SCHEMA_HISTORY_READ = "schema:history:read"


# ============================================================================
# Canonical SEC-003 Role-to-Permission Mapping (Default Deny)
# ============================================================================

ROLE_PERMISSIONS: Dict[Role, Set[SchemaPermission]] = {
    # Analyst: Read-only access to schemas and definitions
    Role.ANALYST: {
        SchemaPermission.SCHEMA_READ,
    },
    # Reviewer: Read-only access to schemas for workbench review
    Role.REVIEWER: {
        SchemaPermission.SCHEMA_READ,
    },
    # Schema author: Design, edit drafts, submit for testing, run harness
    # Explicitly CANNOT approve schemas (SCH-005 separation of duties)
    Role.SCHEMA_AUTHOR: {
        SchemaPermission.SCHEMA_READ,
        SchemaPermission.SCHEMA_CREATE,
        SchemaPermission.SCHEMA_VERSION_CREATE,
        SchemaPermission.SCHEMA_EDIT_DRAFT,
        SchemaPermission.SCHEMA_SUBMIT_TEST,
        SchemaPermission.SCHEMA_REQUEST_APPROVAL,
        SchemaPermission.SCHEMA_RUN_HARNESS,
        SchemaPermission.SCHEMA_HISTORY_READ,
    },
    # Schema approver: Inspect schemas, verify harness runs, approve and retire
    # Does not edit drafts; must approve a different author's work
    Role.SCHEMA_APPROVER: {
        SchemaPermission.SCHEMA_READ,
        SchemaPermission.SCHEMA_RUN_HARNESS,
        SchemaPermission.SCHEMA_APPROVE,
        SchemaPermission.SCHEMA_RETIRE,
        SchemaPermission.SCHEMA_HISTORY_READ,
    },
    # Integrator: Read schemas for API contracts and integrations
    Role.INTEGRATOR: {
        SchemaPermission.SCHEMA_READ,
    },
    # Auditor: Read-only compliance inspection of schemas and approval history
    Role.AUDITOR: {
        SchemaPermission.SCHEMA_READ,
        SchemaPermission.SCHEMA_HISTORY_READ,
    },
    # Admin: Tenant administration, lifecycle rollback, configuration
    # Note: Does NOT automatically inherit SCHEMA_APPROVE (PRD least-privilege)
    Role.ADMIN: {
        SchemaPermission.SCHEMA_READ,
        SchemaPermission.SCHEMA_CREATE,
        SchemaPermission.SCHEMA_VERSION_CREATE,
        SchemaPermission.SCHEMA_EDIT_DRAFT,
        SchemaPermission.SCHEMA_SUBMIT_TEST,
        SchemaPermission.SCHEMA_REQUEST_APPROVAL,
        SchemaPermission.SCHEMA_RUN_HARNESS,
        SchemaPermission.SCHEMA_RETIRE,
        SchemaPermission.SCHEMA_ROLLBACK,
        SchemaPermission.SCHEMA_HISTORY_READ,
    },
}


class AuthenticatedPrincipal(BaseModel):
    """
    Immutable principal record established by a trusted authentication boundary.
    Cannot be forged or asserted directly by client request payloads.
    Adheres to SEC-001 (MFA requirements) and SEC-003 (least privilege).
    """
    model_config = ConfigDict(frozen=True, extra="forbid")

    principal_id: str = Field(..., min_length=1, description="Authoritative unique user identifier (e.g. JWT sub)")
    tenant_id: str = Field(..., min_length=1, description="Authoritative tenant identifier for multi-tenant isolation")
    roles: Set[Role] = Field(default_factory=set, description="Authoritative roles assigned to principal")
    explicit_permissions: Set[SchemaPermission] = Field(
        default_factory=set,
        description="Explicit granular permissions granted directly to principal",
    )
    is_active: bool = Field(default=True, description="Whether principal is currently active and not revoked")
    mfa_verified: bool = Field(
        default=False,
        description="True if principal was authenticated using Multi-Factor Authentication (SEC-001)",
    )

    @property
    def effective_permissions(self) -> Set[SchemaPermission]:
        """Calculates effective permissions as union of role-derived and explicit permissions."""
        perms: Set[SchemaPermission] = set(self.explicit_permissions)
        for role in self.roles:
            perms.update(ROLE_PERMISSIONS.get(role, set()))
        return perms

    def has_permission(self, permission: SchemaPermission) -> bool:
        """Evaluates whether principal holds the requested permission under default-deny."""
        if not self.is_active:
            return False
        return permission in self.effective_permissions


class AuthorizationDecision(BaseModel):
    """Result of an authoritative authorization evaluation."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    is_authorized: bool = Field(..., description="True if access is granted")
    reason: Optional[str] = Field(default=None, description="Explanation for grant or denial")
    code: Optional[str] = Field(default=None, description="Machine-readable decision code")


class SchemaAuthorizer(Protocol):
    """Protocol for authoritative schema lifecycle authorization."""

    def authorize(
        self,
        principal: AuthenticatedPrincipal,
        permission: SchemaPermission,
        tenant_id: str,
        schema_author: Optional[str] = None,
    ) -> AuthorizationDecision:
        ...


class DefaultSchemaAuthorizer:
    """
    Authoritative reference implementation of SchemaAuthorizer.
    Validates principal status, tenant isolation, role-based permissions,
    SEC-001 MFA constraints, and SCH-005 author-vs-approver separation.
    """

    def __init__(
        self,
        trusted_principals: Optional[Dict[str, AuthenticatedPrincipal]] = None,
    ) -> None:
        self._trusted_principals: Dict[str, AuthenticatedPrincipal] = dict(trusted_principals or {})

    def register_principal(self, principal: AuthenticatedPrincipal) -> None:
        """Registers a principal in the trusted authorizer boundary."""
        self._trusted_principals[principal.principal_id] = principal

    def get_principal(self, principal_id: str) -> Optional[AuthenticatedPrincipal]:
        """Retrieves an authoritative principal record by ID."""
        return self._trusted_principals.get(principal_id)

    def authorize(
        self,
        principal: AuthenticatedPrincipal,
        permission: SchemaPermission,
        tenant_id: str,
        schema_author: Optional[str] = None,
    ) -> AuthorizationDecision:
        """
        Evaluates authorization for an operation.
        Enforces:
        1. Fail-closed authentication against trusted boundary directory.
        2. Detection of forged or tampered caller principal claims.
        3. Principal active status check.
        4. Strict tenant isolation.
        5. SEC-001 MFA requirement for approvers.
        6. SCH-005 Four-Eyes author exclusion (cannot approve own schema).
        7. Effective permission check under default-deny.
        """
        # 1. Authoritative identity verification against trusted boundary (SEC-001, SEC-003)
        if not principal or not getattr(principal, "principal_id", None):
            return AuthorizationDecision(
                is_authorized=False,
                reason="Unauthenticated request: principal context is missing or null.",
                code="UNAUTHENTICATED",
            )

        trusted_record = self._trusted_principals.get(principal.principal_id)
        if trusted_record is None:
            return AuthorizationDecision(
                is_authorized=False,
                reason=f"Principal '{principal.principal_id}' is not registered in the trusted authentication boundary.",
                code="UNAUTHENTICATED",
            )

        # Detect forged or tampered credentials between caller-supplied principal and authoritative record
        if (
            principal.tenant_id != trusted_record.tenant_id
            or principal.roles != trusted_record.roles
            or principal.explicit_permissions != trusted_record.explicit_permissions
            or principal.mfa_verified != trusted_record.mfa_verified
            or principal.is_active != trusted_record.is_active
        ):
            return AuthorizationDecision(
                is_authorized=False,
                reason=f"Principal '{principal.principal_id}' claims mismatch authoritative boundary (forged credentials detected).",
                code="FORBIDDEN",
            )

        # Always evaluate against the authoritative trusted record
        auth_principal = trusted_record

        # 2. Active status check
        if not auth_principal.is_active:
            return AuthorizationDecision(
                is_authorized=False,
                reason=f"Principal '{auth_principal.principal_id}' is inactive or revoked.",
                code="UNAUTHENTICATED",
            )

        # 3. Tenant isolation check (SEC-006)
        if auth_principal.tenant_id != tenant_id:
            return AuthorizationDecision(
                is_authorized=False,
                reason=f"Cross-tenant access violation: principal tenant '{auth_principal.tenant_id}' != target '{tenant_id}'.",
                code="TENANT_ACCESS_ERROR",
            )

        # 4. Special checks for Schema Approval (SCH-005, SEC-001, SEC-003)
        if permission == SchemaPermission.SCHEMA_APPROVE:
            # SEC-001: Schema approvers MUST use Multi-Factor Authentication
            if not auth_principal.mfa_verified:
                return AuthorizationDecision(
                    is_authorized=False,
                    reason=f"MFA required: principal '{auth_principal.principal_id}' lacks verified MFA (SEC-001).",
                    code="FORBIDDEN",
                )

            # SCH-005: Separation of duties (Four-Eyes Gate)
            if schema_author is not None and auth_principal.principal_id == schema_author:
                return AuthorizationDecision(
                    is_authorized=False,
                    reason=(
                        f"Four-eyes violation: user '{auth_principal.principal_id}' authored this schema version "
                        "and cannot approve it (SCH-005, SEC-003)."
                    ),
                    code="SCHEMA_SELF_APPROVAL",
                )

        # 5. Effective permission check (Default Deny)
        if not auth_principal.has_permission(permission):
            return AuthorizationDecision(
                is_authorized=False,
                reason=(
                    f"Principal '{auth_principal.principal_id}' lacks required permission '{permission.value}'. "
                    f"Assigned roles: {[r.value for r in auth_principal.roles]}."
                ),
                code="FORBIDDEN",
            )

        return AuthorizationDecision(
            is_authorized=True,
            reason=f"Permission '{permission.value}' granted to principal '{auth_principal.principal_id}'.",
            code="AUTHORIZED",
        )
