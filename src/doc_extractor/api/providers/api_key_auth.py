"""
API Key Authentication Provider (Module M5 TASK-5.7-B).

Adheres to:
- PRD §10.1 SEC-002: Deterministic SHA-256 hash lookup, expiration, and revocation verification.
- PRD §10.1 SEC-001: Strict tenant boundary isolation, mfa_verified=False.
- PRD §10.1 SEC-012 & API-011: Raw secret tokens and hashes are never logged or leaked.
- Fail-closed design: Returns None or raises AuthenticationRequiredError subclasses.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any, Optional

from doc_extractor.api.contracts.records import ApiKeyRecord
from doc_extractor.api.exceptions import (
    AuthenticationRequiredError,
    InvalidTokenError,
    TokenExpiredError,
)
from doc_extractor.api.ports.audit import AuditLogger
from doc_extractor.api.ports.authentication import ApiPrincipal, AuthenticationProvider
from doc_extractor.api.ports.repositories import ApiKeyRepository
from doc_extractor.api.token_core import (
    ALLOWED_TOKEN_PREFIXES,
    create_api_key_principal,
    hash_api_token,
)

logger = logging.getLogger("doc_extractor.auth.api_key")

# Strict regex for token payload: safe alphanumeric, hyphen, underscore
_TOKEN_CHARS_REGEX = re.compile(r"^[a-zA-Z0-9_\-]+$")

# Minimum token length: prefix (>= 8 chars) + 32-byte base64url entropy (43 chars) = 51 chars
_MIN_TOKEN_LENGTH: int = 51


class ApiKeyAuthenticationProvider(AuthenticationProvider):
    """
    Concrete AuthenticationProvider validating scoped API keys against ApiKeyRepository.

    Performs:
    1. Pre-lookup format validation (length, prefix, character set).
    2. Deterministic SHA-256 hashing.
    3. Scoped repository lookup (get_by_key_hash).
    4. Active, revocation, and expiration state evaluation.
    5. Authoritative ApiPrincipal construction (with mfa_verified=False and scope separation).
    6. Asynchronous last-used timestamp tracking.
    """

    def __init__(
        self,
        repository: ApiKeyRepository,
        audit_logger: Optional[AuditLogger] = None,
    ) -> None:
        self._repository = repository
        self._audit_logger = audit_logger

    def _extract_token_from_credentials(self, credentials: Any) -> Optional[str]:
        """Extract and clean raw token from Authorization header or direct string."""
        if not credentials or not isinstance(credentials, str):
            return None
        cleaned = credentials.strip()
        if cleaned.lower().startswith("bearer "):
            token = cleaned[7:].strip()
        else:
            token = cleaned
        return token if token else None

    def _validate_token_format(self, token: str) -> bool:
        """
        Validate token structure before hitting the repository.
        Prevents injection, invalid format queries, and reduces database load.
        """
        if len(token) < _MIN_TOKEN_LENGTH:
            return False

        has_valid_prefix = any(token.startswith(p) for p in ALLOWED_TOKEN_PREFIXES)
        if not has_valid_prefix:
            return False

        if not _TOKEN_CHARS_REGEX.match(token):
            return False

        return True

    async def _safe_log_audit_failure(
        self,
        actor_id: str = "unauthenticated",
        tenant_id: str = "unknown",
        resource_id: str = "unknown",
        error_code: str = "INVALID_CREDENTIALS",
    ) -> None:
        """Log a safe, sanitized audit failure event without raw tokens or hashes."""
        if not self._audit_logger:
            return
        try:
            await self._audit_logger.log_action(
                actor_id=actor_id,
                tenant_id=tenant_id,
                action="api_key.authentication_failed",
                resource_id=resource_id,
                resource_type="api_key",
                outcome="FAILED",
                metadata={"error_code": error_code},
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to emit audit event for auth failure: %s", exc)

    async def authenticate(
        self,
        credentials: Any,
        **kwargs: Any,
    ) -> Optional[ApiPrincipal]:
        """
        Authenticate an API client using their API key credential.

        Fail-closed: Returns an authoritative ApiPrincipal or raises AuthenticationRequiredError.
        Never falls back to an anonymous principal or default tenant.

        Args:
            credentials: Raw Authorization header string or token string.

        Returns:
            Authoritative ApiPrincipal if valid.

        Raises:
            AuthenticationRequiredError: If credentials are missing.
            InvalidTokenError: If credentials are malformed, unknown, revoked, or inactive.
            TokenExpiredError: If key has expired.
        """
        raw_token = self._extract_token_from_credentials(credentials)
        if not raw_token:
            raise AuthenticationRequiredError(
                message="Authorization header or token credential is required.",
                code="AUTHENTICATION_REQUIRED",
            )

        # 1. Format validation before repository lookup
        if not self._validate_token_format(raw_token):
            await self._safe_log_audit_failure(error_code="INVALID_CREDENTIALS")
            raise InvalidTokenError(
                message="API key format is invalid.",
                code="INVALID_CREDENTIALS",
            )

        # 2. Deterministic SHA-256 hash calculation (SEC-002)
        key_hash = hash_api_token(raw_token)

        # 3. Repository lookup
        record = await self._repository.get_by_key_hash(key_hash)
        if record is None:
            await self._safe_log_audit_failure(error_code="INVALID_CREDENTIALS")
            raise InvalidTokenError(
                message="API key is invalid or unrecognized.",
                code="INVALID_CREDENTIALS",
            )

        # 4. Revocation and active status check
        if not record.is_active or record.revoked_at is not None:
            await self._safe_log_audit_failure(
                actor_id=record.key_id,
                tenant_id=record.tenant_id,
                resource_id=record.key_id,
                error_code="INVALID_CREDENTIALS",
            )
            raise InvalidTokenError(
                message="API key has been revoked or is inactive.",
                code="INVALID_CREDENTIALS",
            )

        # 5. Expiration check (SEC-002: Expired tokens return 401)
        now_utc = datetime.now(timezone.utc)
        if record.expires_at is not None:
            expires_utc = record.expires_at.astimezone(timezone.utc)
            if expires_utc <= now_utc:
                await self._safe_log_audit_failure(
                    actor_id=record.key_id,
                    tenant_id=record.tenant_id,
                    resource_id=record.key_id,
                    error_code="TOKEN_EXPIRED",
                )
                raise TokenExpiredError(
                    message="API key has expired.",
                    code="TOKEN_EXPIRED",
                )

        # 6. Build authoritative ApiPrincipal
        principal = create_api_key_principal(record, current_time=now_utc)

        # 7. Record last-used timestamp
        try:
            await self._repository.record_last_used(
                tenant_id=record.tenant_id,
                key_id=record.key_id,
                last_used_at=now_utc,
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to record API key last_used_at for key '%s': %s", record.key_id, exc)

        return principal

