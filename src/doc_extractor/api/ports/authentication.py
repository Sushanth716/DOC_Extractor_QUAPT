"""
Provider-independent authentication and tenant context interface (Module M5).

Adheres to:
- PRD §10.1 SEC-001, SEC-003, SEC-005: Strict tenant boundary isolation.
- Tenant identity comes EXCLUSIVELY from the authenticated principal.
- Untrusted caller-asserted tenant IDs via headers, query params, or body are prohibited.
- Provider-independent contract decouples HTTP framework from auth mechanisms (API key, mTLS, JWT).
- Interoperable with M4 schema_lifecycle.roles.AuthenticatedPrincipal.
- Strict default-deny / fail-closed semantics: no insecure anonymous fallbacks.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional, Protocol, Set, runtime_checkable
from pydantic import BaseModel, ConfigDict, Field

# Try importing M4 roles for full type interoperability
try:
    from schema_lifecycle.roles import (
        AuthenticatedPrincipal as M4Principal,
        Role as M4Role,
        ROLE_PERMISSIONS as M4_ROLE_PERMISSIONS,
        SchemaPermission as M4Permission,
    )
except ImportError:  # pragma: no cover
    M4Principal = None
    M4Role = None
    M4_ROLE_PERMISSIONS = None
    M4Permission = None


class PrincipalType(str, Enum):
    """Categorization of authenticated API principals."""
    API_KEY = "api_key"
    USER = "user"
    SERVICE = "service"
    SYSTEM = "system"


class TenantContext(BaseModel):
    """
    Immutable representation of an authenticated tenant boundary.
    Guaranteed to originate exclusively from a verified principal.
    """
    model_config = ConfigDict(frozen=True, extra="forbid")

    tenant_id: str = Field(..., min_length=1, description="Authoritative tenant identifier")
    name: Optional[str] = Field(None, description="Optional tenant display name")
    is_active: bool = Field(default=True, description="Whether tenant is currently in good standing")


class ApiPrincipal(BaseModel):
    """
    Authoritative authenticated principal representing the caller.

    Holds identity, tenant binding, canonical roles, and granular API scopes.
    """
    model_config = ConfigDict(frozen=True, extra="forbid")

    principal_id: str = Field(..., min_length=1, description="Unique subject identifier (key ID, user ID)")
    tenant_id: str = Field(..., min_length=1, description="Mandatory tenant identifier binding")
    principal_type: PrincipalType = Field(default=PrincipalType.API_KEY, description="Caller credential type")
    roles: Set[str] = Field(default_factory=set, description="Assigned role names")
    scopes: Set[str] = Field(default_factory=set, description="Granted API scopes (e.g. documents:read)")
    permissions: Set[str] = Field(default_factory=set, description="Granular permissions")
    is_active: bool = Field(default=True, description="Active status; inactive principals fail closed")
    mfa_verified: bool = Field(default=False, description="Whether credential verified MFA (SEC-001)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Redacted principal metadata")

    @property
    def tenant_context(self) -> TenantContext:
        """Derives authoritative TenantContext solely from verified principal."""
        return TenantContext(tenant_id=self.tenant_id, is_active=self.is_active)

    def has_scope(self, scope: str) -> bool:
        """
        Evaluates whether principal holds the requested API scope under default-deny.
        Evaluates only explicitly granted API scopes.
        Does not grant Role.ADMIN or wildcard '*' automatic scope bypass.
        """
        if not self.is_active:
            return False
        if scope in self.scopes:
            return True
        if ":" in scope:
            domain_wildcard = f"{scope.split(':', 1)[0]}:*"
            if domain_wildcard in self.scopes:
                return True
        return False

    def has_permission(self, permission: str) -> bool:
        """
        Evaluates whether principal holds specific domain permission under default-deny.
        Evaluates explicit permissions and the approved M4 role-permission policy.
        API scopes and domain permissions are strictly independent; does not fall back to has_scope().
        """
        if not self.is_active:
            return False
        if permission in self.permissions:
            return True
        # Evaluate against canonical M4 role-to-permission mapping
        if M4_ROLE_PERMISSIONS is not None and M4Role is not None:
            for role_str in self.roles:
                try:
                    m4_role = M4Role(role_str) if not isinstance(role_str, M4Role) else role_str
                    allowed_perms = M4_ROLE_PERMISSIONS.get(m4_role, set())
                    for p in allowed_perms:
                        val = p.value if hasattr(p, "value") else str(p)
                        if val == permission or str(p) == permission:
                            return True
                except (ValueError, KeyError):
                    continue
        return False

    def to_m4_principal(self) -> Any:
        """
        Bridges ApiPrincipal to M4 schema_lifecycle AuthenticatedPrincipal.
        Preserves compatibility with existing M4 authorization rules.
        Never merges API scopes into M4 explicit permissions.
        """
        if M4Principal is None:
            raise RuntimeError("M4 schema_lifecycle.roles module is not available in environment.")

        m4_roles: Set[Any] = set()
        if M4Role is not None:
            for r in self.roles:
                try:
                    m4_roles.add(M4Role(r))
                except ValueError:
                    pass

        m4_perms: Set[Any] = set()
        if M4Permission is not None:
            for p in self.permissions:
                try:
                    m4_perms.add(M4Permission(p))
                except ValueError:
                    pass

        return M4Principal(
            principal_id=self.principal_id,
            tenant_id=self.tenant_id,
            roles=m4_roles,
            explicit_permissions=m4_perms,
            is_active=self.is_active,
            mfa_verified=self.mfa_verified,
        )

    @classmethod
    def from_m4_principal(
        cls,
        principal: Any,
        principal_type: PrincipalType = PrincipalType.USER,
    ) -> "ApiPrincipal":
        """
        Constructs an ApiPrincipal from an M4 AuthenticatedPrincipal.
        Never copies M4 permissions into API scopes.
        """
        roles = {r.value if hasattr(r, "value") else str(r) for r in getattr(principal, "roles", set())}
        perms = {p.value if hasattr(p, "value") else str(p) for p in getattr(principal, "explicit_permissions", set())}
        return cls(
            principal_id=principal.principal_id,
            tenant_id=principal.tenant_id,
            principal_type=principal_type,
            roles=roles,
            scopes=set(),
            permissions=perms,
            is_active=principal.is_active,
            mfa_verified=principal.mfa_verified,
        )


@runtime_checkable
class AuthenticationProvider(Protocol):
    """
    Protocol for provider-independent authentication.
    Concrete providers (API key, OAuth2/OIDC, mTLS) implement this interface.
    """

    async def authenticate(self, credentials: Any, **kwargs: Any) -> Optional[ApiPrincipal]:
        """
        Authenticate credentials and return the authoritative ApiPrincipal.
        Must return None or raise an exception on invalid/missing credentials.
        Never fall back to an anonymous/default principal.
        """
        ...
