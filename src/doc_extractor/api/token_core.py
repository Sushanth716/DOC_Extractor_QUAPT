"""
Secure API Token Core (Module M5 TASK-5.7-A).

Adheres strictly to:
- PRD §10.1 SEC-002: CSPRNG generation (256-bit entropy), SHA-256 hash stored only,
  keys rotate at most every 90 days, revocation support, fail-closed validation.
- PRD §10.1 SEC-003: Least-privilege access, no ordinary API key grants admin/approver.
- PRD §10.1 SEC-001: Admins and schema approvers require MFA; API keys always mfa_verified=False.
- PRD §10.1 SEC-006: Tenant isolation.
- PRD §10.1 SEC-012 & API-011: Raw secret tokens never stored, logged, or returned in telemetry.
"""

from __future__ import annotations

import hashlib
import re
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Set

from doc_extractor.api.contracts.records import ApiKeyRecord, utc_now
from doc_extractor.api.exceptions import (
    InvalidKeyExpirationError,
    InvalidScopeError,
    InvalidTokenError,
    PrivilegedRoleNotAllowedError,
    TokenExpiredError,
)
from doc_extractor.api.ports.authentication import ApiPrincipal, PrincipalType

# Minimum entropy requirement: 32 random bytes = 256 bits of CSPRNG entropy (SEC-002)
TOKEN_ENTROPY_MIN_BYTES: int = 32

# Approved key prefixes for environment / service identification
DEFAULT_TOKEN_PREFIX: str = "de_live_"
ALLOWED_TOKEN_PREFIXES: FrozenSet[str] = frozenset({"de_live_", "de_test_"})

# Key lifecycle limits (PRD SEC-002: keys rotate at most every 90 days)
MAX_KEY_LIFETIME_DAYS: int = 90
DEFAULT_KEY_LIFETIME_DAYS: int = 90

# Scope and role security invariants
FORBIDDEN_KEY_SCOPES: FrozenSet[str] = frozenset({"*"})

# Privileged roles that must NEVER be issued to ordinary programmatic API keys (SEC-001, SEC-003)
# Schema approvers and admins require MFA, which API keys can never satisfy.
PRIVILEGED_ROLES: FrozenSet[str] = frozenset({"admin", "schema_approver"})

# Permitted tenant roles for API key issuance
ALLOWED_KEY_ROLES: FrozenSet[str] = frozenset({
    "analyst",
    "reviewer",
    "schema_author",
    "integrator",
    "auditor",
})

# Scope naming format: domain:action (e.g. documents:read, review:claim, documents:*)
_SCOPE_REGEX = re.compile(r"^[a-z0-9_\-]+:[a-z0-9_\-*]+$")


@dataclass(frozen=True)
class GeneratedToken:
    """
    Ephemeral container for freshly generated API key credentials.

    Contains the plaintext secret token strictly for immediate one-time return to caller.
    The raw secret is masked in string representations to prevent accidental log exposure.
    """
    raw_secret: str
    key_prefix: str
    key_hash: str
    key_id: str

    def __repr__(self) -> str:
        return f"<GeneratedToken key_id={self.key_id!r} prefix={self.key_prefix!r} raw_secret='[REDACTED]'>"

    def __str__(self) -> str:
        return f"<GeneratedToken key_id={self.key_id!r} prefix={self.key_prefix!r}>"


def hash_api_token(raw_token: str) -> str:
    """
    Compute deterministic SHA-256 hex digest of the complete raw API token (SEC-002).

    Args:
        raw_token: Complete plaintext token string.

    Returns:
        64-character lowercase hexadecimal SHA-256 hash.

    Raises:
        InvalidTokenError: If token is empty or not a string.
    """
    if not raw_token or not isinstance(raw_token, str):
        raise InvalidTokenError(message="API token must be a non-empty string.")

    token_bytes = raw_token.strip().encode("utf-8")
    if not token_bytes:
        raise InvalidTokenError(message="API token must not be whitespace only.")

    return hashlib.sha256(token_bytes).hexdigest()


def extract_token_prefix(token: str, max_chars: int = 12) -> str:
    """
    Safely extract non-sensitive visible prefix from an API token for identification (SEC-012).

    Enforces prefix length constraint between 4 and 16 characters (matching ApiKeyRecord).

    Args:
        token: Raw token string.
        max_chars: Target length of prefix (clamped to [4, 16]).

    Returns:
        Visible prefix substring.

    Raises:
        InvalidTokenError: If token is too short to extract a safe prefix.
    """
    if not token or not isinstance(token, str):
        raise InvalidTokenError(message="Cannot extract prefix from empty or non-string token.")

    cleaned = token.strip()
    if len(cleaned) < 4:
        raise InvalidTokenError(
            message=f"Token length ({len(cleaned)}) is too short to extract safe prefix (minimum 4)."
        )

    clamped_len = max(4, min(max_chars, 16))
    return cleaned[:clamped_len]


def generate_api_secret(
    nbytes: int = TOKEN_ENTROPY_MIN_BYTES,
    prefix: str = DEFAULT_TOKEN_PREFIX,
    key_id: Optional[str] = None,
) -> GeneratedToken:
    """
    Generate a cryptographically secure API secret token (SEC-002).

    Uses Python's `secrets` module (CSPRNG) with at least 32 random bytes (256 bits) of entropy.
    Derives deterministic SHA-256 hash and visible identification prefix.

    Args:
        nbytes: Number of random entropy bytes (must be >= 32).
        prefix: Token identification prefix (must be in ALLOWED_TOKEN_PREFIXES).
        key_id: Optional explicit key identifier; generates random UUID hex if omitted.

    Returns:
        GeneratedToken container with raw secret and derived hash/prefix.

    Raises:
        ValueError: If entropy < 32 bytes or prefix is invalid.
    """
    if nbytes < TOKEN_ENTROPY_MIN_BYTES:
        raise ValueError(
            f"Token entropy must be at least {TOKEN_ENTROPY_MIN_BYTES} bytes (256 bits). Got {nbytes}."
        )

    if prefix not in ALLOWED_TOKEN_PREFIXES:
        raise ValueError(
            f"Token prefix '{prefix}' is not allowed. Must be one of: {sorted(ALLOWED_TOKEN_PREFIXES)}"
        )

    # 32 bytes base64url encoded yields 43 URL-safe characters
    random_entropy = secrets.token_urlsafe(nbytes)
    raw_secret = f"{prefix}{random_entropy}"

    key_prefix = extract_token_prefix(raw_secret, max_chars=12)
    key_hash = hash_api_token(raw_secret)
    assigned_key_id = key_id or f"key_{secrets.token_hex(8)}"

    return GeneratedToken(
        raw_secret=raw_secret,
        key_prefix=key_prefix,
        key_hash=key_hash,
        key_id=assigned_key_id,
    )


def validate_key_expiration(
    expires_at: Optional[datetime],
    created_at: Optional[datetime] = None,
    max_lifetime_days: int = MAX_KEY_LIFETIME_DAYS,
) -> datetime:
    """
    Validate requested API key expiration against SEC-002 policy.

    Enforces:
    - Timezone-aware UTC timestamps for all datetimes.
    - Default expiration to created_at + max_lifetime_days if omitted.
    - Expiration strictly in the future relative to created_at.
    - Lifetime <= max_lifetime_days (90 days).

    Args:
        expires_at: Caller-requested expiration timestamp, or None for default.
        created_at: Key creation timestamp (defaults to current UTC time).
        max_lifetime_days: Maximum permitted lifetime in days (default 90).

    Returns:
        Validated timezone-aware UTC expiration datetime.

    Raises:
        InvalidKeyExpirationError: If timestamps are naive, past, or exceed maximum lifetime.
    """
    now_utc = datetime.now(timezone.utc)
    base_created = created_at or now_utc

    if base_created.tzinfo is None:
        raise InvalidKeyExpirationError(
            message="created_at timestamp must be timezone-aware (UTC).",
            details={"field": "created_at"},
        )

    # Normalize base_created to UTC
    base_created_utc = base_created.astimezone(timezone.utc)
    max_allowed_utc = base_created_utc + timedelta(days=max_lifetime_days)

    if expires_at is None:
        return max_allowed_utc

    if expires_at.tzinfo is None:
        raise InvalidKeyExpirationError(
            message="expires_at timestamp must be timezone-aware (UTC).",
            details={"field": "expires_at"},
        )

    expires_at_utc = expires_at.astimezone(timezone.utc)

    # Must be in the future relative to creation
    if expires_at_utc <= base_created_utc:
        raise InvalidKeyExpirationError(
            message="API key expiration must be in the future relative to creation time.",
            details={
                "expires_at": expires_at_utc.isoformat(),
                "created_at": base_created_utc.isoformat(),
            },
        )

    # Must not exceed maximum allowed lifetime (SEC-002: 90 days)
    if expires_at_utc > max_allowed_utc:
        raise InvalidKeyExpirationError(
            message=f"API key lifetime exceeds maximum allowed limit of {max_lifetime_days} days.",
            details={
                "expires_at": expires_at_utc.isoformat(),
                "max_allowed": max_allowed_utc.isoformat(),
                "max_lifetime_days": max_lifetime_days,
            },
        )

    return expires_at_utc


def validate_api_key_scopes(
    scopes: Sequence[str],
) -> Set[str]:
    """
    Validate requested API scopes for API key issuance under default-deny.

    Security Invariant:
    - Never permit global wildcard scope '*' (PRD SEC-002).
    - Scopes must follow domain:action or domain:* pattern.

    Args:
        scopes: Sequence of scope strings.

    Returns:
        Set of normalized, validated scope strings.

    Raises:
        InvalidScopeError: If wildcard '*' or malformed scope format is detected.
    """
    validated: Set[str] = set()

    for raw in scopes:
        scope = raw.strip()
        if not scope:
            continue

        if scope in FORBIDDEN_KEY_SCOPES:
            raise InvalidScopeError(
                message="Global wildcard scope '*' is strictly prohibited for API keys (PRD SEC-002).",
                details={"forbidden_scope": scope},
            )

        if not _SCOPE_REGEX.match(scope):
            raise InvalidScopeError(
                message=f"Invalid API scope format '{scope}'. Scopes must conform to 'domain:action' (e.g. 'documents:read').",
                details={"invalid_scope": scope},
            )

        validated.add(scope)

    return validated


def validate_api_key_roles(
    roles: Sequence[str],
    allow_privileged_roles: bool = False,
) -> Set[str]:
    """
    Validate requested roles for API key issuance (SEC-003, SEC-001).

    Security Invariant:
    - Ordinary API keys must never be issued with 'admin' or 'schema_approver' roles.
    - API keys cannot satisfy MFA; schema approval requires verified MFA.

    Args:
        roles: Sequence of role strings.
        allow_privileged_roles: Set True strictly for bootstrap administrative keys.

    Returns:
        Set of normalized, validated role strings.

    Raises:
        PrivilegedRoleNotAllowedError: If privileged role requested without explicit allow.
        ApiValidationError: If unknown role is supplied.
    """
    validated: Set[str] = set()

    for raw in roles:
        role = raw.strip().lower()
        if not role:
            continue

        if role in PRIVILEGED_ROLES and not allow_privileged_roles:
            raise PrivilegedRoleNotAllowedError(
                message=f"Privileged role '{role}' cannot be assigned to API keys (SEC-001, SEC-003).",
                details={"privileged_role": role, "allowed_roles": sorted(ALLOWED_KEY_ROLES)},
            )

        if role not in ALLOWED_KEY_ROLES and role not in PRIVILEGED_ROLES:
            raise PrivilegedRoleNotAllowedError(
                message=f"Unrecognized role '{role}'. Allowed API key roles: {sorted(ALLOWED_KEY_ROLES)}",
                details={"invalid_role": role},
            )

        validated.add(role)

    return validated


def create_api_key_principal(
    key_record: ApiKeyRecord,
    current_time: Optional[datetime] = None,
) -> ApiPrincipal:
    """
    Construct an authoritative ApiPrincipal from an ApiKeyRecord.

    Security Guarantees:
    - principal_type is strictly PrincipalType.API_KEY.
    - mfa_verified is ALWAYS False (API keys are machine credentials; SEC-001).
    - API scopes are populated into principal.scopes.
    - API scopes are NEVER copied into principal.permissions (strict M4 separation).
    - Revoked or expired keys produce an inactive principal (is_active=False).

    Args:
        key_record: Stored API key record.
        current_time: Reference time for expiration evaluation (defaults to UTC now).

    Returns:
        Authoritative ApiPrincipal.
    """
    now_utc = (current_time or datetime.now(timezone.utc)).astimezone(timezone.utc)

    # Default-deny status check
    is_active = key_record.is_active and (key_record.revoked_at is None)

    if key_record.expires_at is not None:
        exp_utc = key_record.expires_at.astimezone(timezone.utc)
        if exp_utc <= now_utc:
            is_active = False

    return ApiPrincipal(
        principal_id=key_record.key_id,
        tenant_id=key_record.tenant_id,
        principal_type=PrincipalType.API_KEY,
        roles=set(key_record.roles),
        scopes=set(key_record.scopes),
        permissions=set(),  # Strict separation: Scopes NEVER merge into domain permissions
        is_active=is_active,
        mfa_verified=False,  # Security Invariant: API keys never have MFA
        metadata={
            "key_prefix": key_record.key_prefix,
            "key_name": key_record.name,
        },
    )


def create_api_key_record(
    token: GeneratedToken,
    tenant_id: str,
    name: str,
    scopes: Sequence[str],
    roles: Sequence[str],
    expires_at: Optional[datetime] = None,
    created_by: Optional[str] = None,
    created_at: Optional[datetime] = None,
    allow_privileged_roles: bool = False,
) -> ApiKeyRecord:
    """
    Construct a validated ApiKeyRecord from a GeneratedToken.

    Security Guarantee:
    - Raw token secret is NEVER present on the constructed ApiKeyRecord.
    - Only deterministic SHA-256 hash and safe visible prefix are stored.

    Args:
        token: Freshly generated token container.
        tenant_id: Mandatory bound tenant identifier.
        name: Human-readable key label.
        scopes: Requested API scopes (validated against wildcard rejection).
        roles: Requested tenant roles (validated against privileged role rejection).
        expires_at: Requested expiration timestamp (enforces max 90-day lifetime).
        created_by: Optional creator principal ID.
        created_at: Optional creation timestamp (defaults to UTC now).
        allow_privileged_roles: Whether privileged roles ('admin') are allowed (bootstrap only).

    Returns:
        Validated ApiKeyRecord ready for repository persistence.
    """
    now_utc = (created_at or datetime.now(timezone.utc)).astimezone(timezone.utc)
    valid_exp = validate_key_expiration(expires_at, created_at=now_utc)
    valid_scopes = validate_api_key_scopes(scopes)
    valid_roles = validate_api_key_roles(roles, allow_privileged_roles=allow_privileged_roles)

    return ApiKeyRecord(
        key_id=token.key_id,
        tenant_id=tenant_id,
        name=name.strip(),
        key_hash=token.key_hash,
        key_prefix=token.key_prefix,
        scopes=sorted(valid_scopes),
        roles=sorted(valid_roles),
        created_at=now_utc,
        expires_at=valid_exp,
        revoked_at=None,
        is_active=True,
        created_by=created_by,
        last_used_at=None,
    )

