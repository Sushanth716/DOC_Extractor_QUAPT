"""
Unit and security tests for Secure API Token Core (Module M5 TASK-5.7-A).

Verifies:
- 256-bit token entropy using CSPRNG.
- Token encoding and safe prefix extraction.
- Deterministic SHA-256 hash correctness and collision resistance.
- Complete absence of raw secret token from persistent records, string representations, and logs.
- Expiration boundaries, naive datetime rejection, and UTC timezone normalization.
- Strict enforcement of maximum 90-day key lifetime (PRD SEC-002).
- Global wildcard scope '*' rejection and domain wildcard validation.
- Privileged role ('admin', 'schema_approver') rejection on ordinary API key issuance.
- Strict invariant that API-key principals always have mfa_verified=False (SEC-001).
- Strict separation between API route scopes and M4 schema domain permissions.
- Repository protocol contract with typed last_used_at tracking.
"""

import collections
import hashlib
import math
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import pytest

from doc_extractor.api.contracts.records import ApiKeyRecord, utc_now
from doc_extractor.api.exceptions import (
    InvalidKeyExpirationError,
    InvalidScopeError,
    InvalidTokenError,
    PrivilegedRoleNotAllowedError,
)
from doc_extractor.api.ports.authentication import ApiPrincipal, PrincipalType
from doc_extractor.api.ports.repositories import ApiKeyRepository
from doc_extractor.api.token_core import (
    DEFAULT_KEY_LIFETIME_DAYS,
    DEFAULT_TOKEN_PREFIX,
    FORBIDDEN_KEY_SCOPES,
    MAX_KEY_LIFETIME_DAYS,
    PRIVILEGED_ROLES,
    TOKEN_ENTROPY_MIN_BYTES,
    GeneratedToken,
    create_api_key_principal,
    create_api_key_record,
    extract_token_prefix,
    generate_api_secret,
    hash_api_token,
    validate_api_key_roles,
    validate_api_key_scopes,
    validate_key_expiration,
)

# Optional M4 imports for permission separation verification
try:
    from schema_lifecycle.roles import (
        AuthenticatedPrincipal as M4Principal,
        Role as M4Role,
        SchemaPermission as M4Permission,
    )
except ImportError:
    M4Principal = None
    M4Role = None
    M4Permission = None


# ============================================================================
# 1. 256-bit Token Entropy and Generation
# ============================================================================

@pytest.mark.security
def test_token_entropy_minimum_bytes():
    """Verify that generate_api_secret enforces at least 32 bytes (256 bits) of entropy."""
    # 32 bytes is the minimum allowed
    token = generate_api_secret(nbytes=32)
    assert token.raw_secret.startswith(DEFAULT_TOKEN_PREFIX)
    # Prefix (8 chars) + 32 bytes base64url (43 chars) = 51 chars
    assert len(token.raw_secret) >= 51

    # Requesting fewer than 32 bytes must be strictly rejected
    with pytest.raises(ValueError, match="at least 32 bytes"):
        generate_api_secret(nbytes=31)

    with pytest.raises(ValueError, match="at least 32 bytes"):
        generate_api_secret(nbytes=16)

    with pytest.raises(ValueError, match="at least 32 bytes"):
        generate_api_secret(nbytes=0)


@pytest.mark.security
def test_token_entropy_distribution():
    """Generate 200 tokens and verify high character entropy and absence of collisions."""
    tokens = [generate_api_secret(nbytes=32).raw_secret for _ in range(200)]
    assert len(set(tokens)) == 200, "Collision detected among CSPRNG generated tokens!"

    # Calculate Shannon entropy across the sample of random payloads
    payloads = [t[len(DEFAULT_TOKEN_PREFIX):] for t in tokens]
    char_counts = collections.Counter("".join(payloads))
    total_chars = sum(char_counts.values())
    entropy = -sum((cnt / total_chars) * math.log2(cnt / total_chars) for cnt in char_counts.values())

    # Base64url alphabet has 64 characters -> max theoretical entropy is log2(64) = 6.0
    # Good CSPRNG over 200 samples should comfortably exceed 5.5 bits/char
    assert entropy > 5.5, f"Shannon entropy too low: {entropy:.2f} bits/char"


# ============================================================================
# 2. Token Encoding and Prefix Extraction
# ============================================================================

@pytest.mark.security
def test_token_encoding_and_prefix_structure():
    """Verify token encoding formats and safe prefix extraction."""
    token = generate_api_secret(prefix="de_live_")
    assert token.raw_secret.startswith("de_live_")
    assert token.key_prefix.startswith("de_live_")
    assert 4 <= len(token.key_prefix) <= 16

    test_token = generate_api_secret(prefix="de_test_")
    assert test_token.raw_secret.startswith("de_test_")
    assert test_token.key_prefix.startswith("de_test_")

    # Invalid prefix is rejected
    with pytest.raises(ValueError, match="not allowed"):
        generate_api_secret(prefix="unapproved_")


@pytest.mark.security
def test_safe_prefix_extraction():
    """Verify safe prefix extraction edge cases."""
    assert extract_token_prefix("de_live_abc123xyz", max_chars=12) == "de_live_abc1"
    assert extract_token_prefix("de_live_abc123xyz", max_chars=8) == "de_live_"
    assert extract_token_prefix("de_live_abc123xyz", max_chars=16) == "de_live_abc123xy"
    assert extract_token_prefix("abcd") == "abcd"

    # Too short or empty tokens
    with pytest.raises(InvalidTokenError):
        extract_token_prefix("abc")
    with pytest.raises(InvalidTokenError):
        extract_token_prefix("")
    with pytest.raises(InvalidTokenError):
        # pyrefly: ignore[arg-type]
        extract_token_prefix(None)  # type: ignore


# ============================================================================
# 3. Deterministic SHA-256 Hashing & Collision Resistance
# ============================================================================

@pytest.mark.security
def test_sha256_hash_correctness():
    """Verify deterministic SHA-256 hashing against standard hashlib implementation."""
    raw = "de_live_test_secret_token_1234567890_abcdef"
    expected = hashlib.sha256(raw.encode("utf-8")).hexdigest()

    computed = hash_api_token(raw)
    assert computed == expected
    assert len(computed) == 64
    assert computed == computed.lower()

    # Empty token fails closed
    with pytest.raises(InvalidTokenError):
        hash_api_token("")
    with pytest.raises(InvalidTokenError):
        hash_api_token("   ")
    with pytest.raises(InvalidTokenError):
        # pyrefly: ignore[arg-type]
        hash_api_token(None)  # type: ignore


@pytest.mark.security
def test_different_tokens_produce_different_hashes():
    """Verify collision resistance and avalanche effect."""
    t1 = "de_live_secret_token_alpha_1"
    t2 = "de_live_secret_token_alpha_2"

    h1 = hash_api_token(t1)
    h2 = hash_api_token(t2)
    assert h1 != h2

    # Avalanche test: 1-character difference changes majority of hash bits
    bits1 = bin(int(h1, 16))[2:].zfill(256)
    bits2 = bin(int(h2, 16))[2:].zfill(256)
    differing_bits = sum(b1 != b2 for b1, b2 in zip(bits1, bits2))
    assert differing_bits > 100, f"Avalanche effect insufficient: {differing_bits} / 256 bits changed"


# ============================================================================
# 4. Raw Secret Absent from Persisted Records and Logs
# ============================================================================

@pytest.mark.security
def test_raw_token_absent_from_persisted_records():
    """Verify raw token secret is never stored in ApiKeyRecord, its dict, json, or repr."""
    token = generate_api_secret()
    record = create_api_key_record(
        token=token,
        tenant_id="tenant_alpha",
        name="Production Ingestion Key",
        scopes=["documents:read", "documents:write"],
        roles=["integrator"],
    )

    dump_dict = record.model_dump()
    dump_json = record.model_dump_json()
    record_repr = repr(record)
    record_str = str(record)

    # Raw secret must NEVER be present anywhere in the record representation
    assert token.raw_secret not in dump_dict.values()
    assert token.raw_secret not in dump_json
    assert token.raw_secret not in record_repr
    assert token.raw_secret not in record_str

    # Only key_hash and key_prefix exist
    assert record.key_hash == token.key_hash
    assert record.key_prefix == token.key_prefix
    assert not hasattr(record, "raw_secret")
    assert not hasattr(record, "secret_token")


@pytest.mark.security
def test_generated_token_redacts_secret_in_repr():
    """Verify GeneratedToken hides raw secret in __repr__ and __str__ to prevent accidental logging."""
    token = generate_api_secret()
    token_repr = repr(token)
    token_str = str(token)

    assert "[REDACTED]" in token_repr
    assert token.raw_secret not in token_repr
    assert token.raw_secret not in token_str


# ============================================================================
# 5. Expiration Boundaries & Timezone Handling
# ============================================================================

@pytest.mark.security
def test_expiration_validation_default_90_days():
    """Verify default expiration is strictly 90 days from creation in UTC."""
    now_utc = datetime.now(timezone.utc)
    exp = validate_key_expiration(expires_at=None, created_at=now_utc)

    assert exp.tzinfo == timezone.utc
    expected = now_utc + timedelta(days=90)
    assert abs((exp - expected).total_seconds()) < 1.0


@pytest.mark.security
def test_expiration_rejects_naive_datetime():
    """Verify timezone-naive datetimes fail closed (rejecting ambiguous timestamps)."""
    naive_dt = datetime(2026, 12, 1, 12, 0, 0)  # No tzinfo

    with pytest.raises(InvalidKeyExpirationError, match="must be timezone-aware"):
        validate_key_expiration(expires_at=naive_dt)

    with pytest.raises(InvalidKeyExpirationError, match="must be timezone-aware"):
        validate_key_expiration(expires_at=None, created_at=naive_dt)


@pytest.mark.security
def test_expiration_normalizes_non_utc_timezones():
    """Verify non-UTC timezone-aware timestamps are properly converted to UTC."""
    # Asia/Kolkata is UTC+5:30
    ist_tz = timezone(timedelta(hours=5, minutes=30))
    created_ist = datetime(2026, 11, 4, 12, 0, 0, tzinfo=ist_tz)
    expires_ist = created_ist + timedelta(days=30)

    validated = validate_key_expiration(expires_at=expires_ist, created_at=created_ist)
    assert validated.tzinfo == timezone.utc
    assert abs((validated - created_ist.astimezone(timezone.utc) - timedelta(days=30)).total_seconds()) < 1.0


@pytest.mark.security
def test_expiration_rejects_past_or_current_timestamp():
    """Verify expiration timestamp cannot be in the past or equal to creation."""
    now_utc = datetime.now(timezone.utc)

    # Past
    past_exp = now_utc - timedelta(hours=1)
    with pytest.raises(InvalidKeyExpirationError, match="in the future"):
        validate_key_expiration(expires_at=past_exp, created_at=now_utc)

    # Exactly now / creation
    with pytest.raises(InvalidKeyExpirationError, match="in the future"):
        validate_key_expiration(expires_at=now_utc, created_at=now_utc)


# ============================================================================
# 6. Maximum 90-Day Key Lifetime (PRD SEC-002)
# ============================================================================

@pytest.mark.security
def test_maximum_90_day_lifetime_enforcement():
    """Verify strict rejection of keys requested with lifetimes exceeding 90 days."""
    now_utc = datetime.now(timezone.utc)

    # Exactly 90 days is valid
    exp_90d = now_utc + timedelta(days=90)
    assert validate_key_expiration(expires_at=exp_90d, created_at=now_utc) == exp_90d

    # 89 days is valid
    exp_89d = now_utc + timedelta(days=89)
    assert validate_key_expiration(expires_at=exp_89d, created_at=now_utc) == exp_89d

    # 90 days + 1 second is rejected
    exp_too_long_1s = now_utc + timedelta(days=90, seconds=1)
    with pytest.raises(InvalidKeyExpirationError, match="exceeds maximum allowed limit"):
        validate_key_expiration(expires_at=exp_too_long_1s, created_at=now_utc)

    # 91 days is rejected
    exp_91d = now_utc + timedelta(days=91)
    with pytest.raises(InvalidKeyExpirationError, match="exceeds maximum allowed limit"):
        validate_key_expiration(expires_at=exp_91d, created_at=now_utc)

    # 1 year is rejected
    exp_1y = now_utc + timedelta(days=365)
    with pytest.raises(InvalidKeyExpirationError, match="exceeds maximum allowed limit"):
        validate_key_expiration(expires_at=exp_1y, created_at=now_utc)


# ============================================================================
# 7. Wildcard Scope Rejection & Scope Validation
# ============================================================================

@pytest.mark.security
def test_global_wildcard_scope_strictly_prohibited():
    """Verify that global wildcard scope '*' is strictly rejected under all circumstances."""
    with pytest.raises(InvalidScopeError, match="Global wildcard scope '\\*' is strictly prohibited"):
        validate_api_key_scopes(["*"])

    with pytest.raises(InvalidScopeError, match="Global wildcard scope '\\*' is strictly prohibited"):
        validate_api_key_scopes(["documents:read", "*"])

    with pytest.raises(InvalidScopeError, match="Global wildcard scope '\\*' is strictly prohibited"):
        validate_api_key_scopes(["  *  "])


@pytest.mark.security
def test_valid_and_domain_wildcard_scopes():
    """Verify that structured scopes and domain-specific wildcards are accepted."""
    valid_scopes = ["documents:read", "documents:write", "review:claim", "documents:*"]
    validated = validate_api_key_scopes(valid_scopes)
    assert validated == set(valid_scopes)

    # Invalid scope formatting is rejected
    with pytest.raises(InvalidScopeError, match="Invalid API scope format"):
        validate_api_key_scopes(["invalid_scope_without_colon"])

    with pytest.raises(InvalidScopeError, match="Invalid API scope format"):
        validate_api_key_scopes(["DOCUMENTS:READ"])  # Uppercase disallowed


# ============================================================================
# 8. Privileged Role Rejection for Ordinary Keys (SEC-001, SEC-003)
# ============================================================================

@pytest.mark.security
def test_privileged_roles_rejected_for_ordinary_api_keys():
    """Verify that ordinary API keys cannot be granted admin or schema_approver roles."""
    with pytest.raises(PrivilegedRoleNotAllowedError, match="Privileged role 'admin' cannot be assigned"):
        validate_api_key_roles(["admin"], allow_privileged_roles=False)

    with pytest.raises(PrivilegedRoleNotAllowedError, match="Privileged role 'schema_approver' cannot be assigned"):
        validate_api_key_roles(["schema_approver"], allow_privileged_roles=False)

    with pytest.raises(PrivilegedRoleNotAllowedError):
        validate_api_key_roles(["analyst", "admin"], allow_privileged_roles=False)

    # Allowed tenant roles are accepted
    allowed = ["analyst", "reviewer", "schema_author", "integrator", "auditor"]
    validated = validate_api_key_roles(allowed, allow_privileged_roles=False)
    assert validated == set(allowed)


@pytest.mark.security
def test_unrecognized_roles_rejected():
    """Verify unapproved or arbitrary roles fail closed."""
    with pytest.raises(PrivilegedRoleNotAllowedError, match="Unrecognized role"):
        validate_api_key_roles(["super_user"])

    with pytest.raises(PrivilegedRoleNotAllowedError, match="Unrecognized role"):
        validate_api_key_roles(["root"])


@pytest.mark.security
def test_bootstrap_privileged_role_still_rejects_wildcard_scope():
    """Verify that even when bootstrap admin role is allowed, wildcard scope '*' remains forbidden."""
    roles = validate_api_key_roles(["admin"], allow_privileged_roles=True)
    assert "admin" in roles

    # Wildcard scope still strictly forbidden!
    with pytest.raises(InvalidScopeError, match="Global wildcard scope '\\*' is strictly prohibited"):
        validate_api_key_scopes(["*"])


# ============================================================================
# 9. MFA Invariant: API-Key Principals ALWAYS Have mfa_verified=False
# ============================================================================

@pytest.mark.security
def test_api_key_principal_mfa_strictly_false():
    """Verify that create_api_key_principal ALWAYS assigns mfa_verified=False."""
    token = generate_api_secret()
    record = create_api_key_record(
        token=token,
        tenant_id="tenant_alpha",
        name="Test Key",
        scopes=["documents:read"],
        roles=["integrator"],
    )

    principal = create_api_key_principal(record)
    assert principal.principal_type == PrincipalType.API_KEY
    assert principal.mfa_verified is False
    assert principal.is_active is True

    # Even if someone modifies the record or metadata, principal mfa_verified remains False
    tampered_record = record.model_copy(update={"is_active": True})
    tampered_principal = create_api_key_principal(tampered_record)
    assert tampered_principal.mfa_verified is False


@pytest.mark.security
def test_api_key_principal_cannot_satisfy_m4_mfa_requirement():
    """Verify that an API-key principal converted to M4 principal cannot satisfy SEC-001 MFA requirement."""
    token = generate_api_secret()
    record = create_api_key_record(
        token=token,
        tenant_id="tenant_alpha",
        name="Test Key",
        scopes=["documents:read"],
        roles=["integrator"],
    )

    principal = create_api_key_principal(record)
    assert principal.mfa_verified is False

    m4_principal = principal.to_m4_principal()
    assert m4_principal.mfa_verified is False

    # In M4, schema approvers and admins require MFA. An API key cannot approve schemas:
    if M4Permission is not None and M4Role is not None:
        from schema_lifecycle.roles import DefaultSchemaAuthorizer
        authorizer = DefaultSchemaAuthorizer(
            trusted_principals={m4_principal.principal_id: m4_principal}
        )
        decision = authorizer.authorize(
            principal=m4_principal,
            permission=M4Permission.SCHEMA_APPROVE,
            tenant_id="tenant_alpha",
        )
        assert decision.is_authorized is False
        assert "MFA required" in decision.reason or "Missing required permission" in decision.reason


# ============================================================================
# 10. Separation Between API Scopes and M4 Domain Permissions
# ============================================================================

@pytest.mark.security
def test_strict_scope_and_permission_separation():
    """Verify that API scopes never leak into M4 permissions and vice versa."""
    token = generate_api_secret()
    record = create_api_key_record(
        token=token,
        tenant_id="tenant_alpha",
        name="Integration Key",
        scopes=["documents:read", "documents:write"],
        roles=["integrator"],
    )

    principal = create_api_key_principal(record)

    # Scopes are present in scopes set
    assert principal.has_scope("documents:read") is True
    assert principal.has_scope("documents:write") is True

    # Permissions set is strictly empty (never copies scopes into permissions)
    assert principal.permissions == set()
    assert principal.has_permission("documents:read") is False
    assert principal.has_permission("documents:write") is False

    # Domain permissions are evaluated strictly against M4 role mapping
    # Role.INTEGRATOR grants SchemaPermission.SCHEMA_READ
    assert principal.has_permission("schema:read") is True
    # Role.INTEGRATOR does NOT grant schema:approve or schema:edit
    assert principal.has_permission("schema:approve") is False
    assert principal.has_permission("schema:draft:edit") is False

    # M4 conversion preserves empty explicit permissions
    m4_principal = principal.to_m4_principal()
    assert len(m4_principal.explicit_permissions) == 0


# ============================================================================
# 11. Principal Expiration and Revocation Evaluation
# ============================================================================

@pytest.mark.security
def test_principal_evaluates_expired_key_as_inactive():
    """Verify that an expired key record produces an inactive principal."""
    now_utc = datetime.now(timezone.utc)
    token = generate_api_secret()
    record = ApiKeyRecord(
        key_id=token.key_id,
        tenant_id="tenant_alpha",
        name="Expired Key",
        key_hash=token.key_hash,
        key_prefix=token.key_prefix,
        scopes=["documents:read"],
        roles=["integrator"],
        created_at=now_utc - timedelta(days=10),
        expires_at=now_utc - timedelta(days=1),  # Expired yesterday
        revoked_at=None,
        is_active=True,
    )

    principal = create_api_key_principal(record, current_time=now_utc)
    assert principal.is_active is False
    assert principal.has_scope("documents:read") is False


@pytest.mark.security
def test_principal_evaluates_revoked_key_as_inactive():
    """Verify that a revoked key record produces an inactive principal."""
    now_utc = datetime.now(timezone.utc)
    token = generate_api_secret()
    record = ApiKeyRecord(
        key_id=token.key_id,
        tenant_id="tenant_alpha",
        name="Revoked Key",
        key_hash=token.key_hash,
        key_prefix=token.key_prefix,
        scopes=["documents:read"],
        roles=["integrator"],
        created_at=now_utc - timedelta(days=5),
        expires_at=now_utc + timedelta(days=85),
        revoked_at=now_utc - timedelta(hours=2),  # Revoked 2 hours ago
        is_active=True,
    )

    principal = create_api_key_principal(record, current_time=now_utc)
    assert principal.is_active is False
    assert principal.has_scope("documents:read") is False


# ============================================================================
# 12. Repository Protocol Contract & last_used_at Tracking
# ============================================================================

@pytest.mark.unit
def test_api_key_record_last_used_at_persistence():
    """Verify ApiKeyRecord accepts and validates optional last_used_at timestamp."""
    token = generate_api_secret()
    record = create_api_key_record(
        token=token,
        tenant_id="tenant_alpha",
        name="Active Key",
        scopes=["documents:read"],
        roles=["integrator"],
    )
    assert record.last_used_at is None

    # Update with timezone-aware timestamp
    now = datetime.now(timezone.utc)
    updated = record.model_copy(update={"last_used_at": now})
    assert updated.last_used_at == now
    assert updated.key_hash == token.key_hash


@pytest.mark.unit
def test_repository_record_last_used_protocol():
    """Verify ApiKeyRepository protocol supports record_last_used."""
    class InMemoryApiKeyRepo:
        def __init__(self):
            self.keys = {}

        async def get_by_key_hash(self, key_hash: str) -> Optional[ApiKeyRecord]:
            for k in self.keys.values():
                if k.key_hash == key_hash:
                    return k
            return None

        async def get_by_id(self, tenant_id: str, key_id: str) -> Optional[ApiKeyRecord]:
            record = self.keys.get(key_id)
            if record and record.tenant_id == tenant_id:
                return record
            return None

        async def create(self, tenant_id: str, key_record: ApiKeyRecord) -> ApiKeyRecord:
            self.keys[key_record.key_id] = key_record
            return key_record

        async def revoke(self, tenant_id: str, key_id: str) -> bool:
            record = await self.get_by_id(tenant_id, key_id)
            if not record:
                return False
            self.keys[key_id] = record.model_copy(
                update={"revoked_at": datetime.now(timezone.utc), "is_active": False}
            )
            return True

        async def list_keys(self, tenant_id: str, limit: int = 50, offset: int = 0) -> List[ApiKeyRecord]:
            return [k for k in self.keys.values() if k.tenant_id == tenant_id]

        async def record_last_used(self, tenant_id: str, key_id: str, last_used_at: datetime) -> bool:
            record = await self.get_by_id(tenant_id, key_id)
            if not record:
                return False
            self.keys[key_id] = record.model_copy(update={"last_used_at": last_used_at})
            return True

    repo = InMemoryApiKeyRepo()
    assert isinstance(repo, ApiKeyRepository)

    async def _test():
        token = generate_api_secret()
        record = create_api_key_record(
            token=token,
            tenant_id="tenant_alpha",
            name="In-Memory Test Key",
            scopes=["documents:read"],
            roles=["integrator"],
        )
        created = await repo.create("tenant_alpha", record)
        assert created.last_used_at is None

        # Record usage
        used_time = datetime.now(timezone.utc)
        success = await repo.record_last_used("tenant_alpha", created.key_id, used_time)
        assert success is True

        fetched = await repo.get_by_id("tenant_alpha", created.key_id)
        assert fetched is not None
        assert fetched.last_used_at == used_time

        # Cross-tenant usage update fails
        cross_tenant_update = await repo.record_last_used("tenant_beta", created.key_id, used_time)
        assert cross_tenant_update is False

    import asyncio
    asyncio.run(_test())
