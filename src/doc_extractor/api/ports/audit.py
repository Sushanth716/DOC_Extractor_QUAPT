"""
Structured audit logging interface (Module M5).

Adheres to:
- PRD §10.1 SEC-012: Audit logs SHALL NOT contain extracted document content,
  raw text, PII, or PHI.
- PRD §21: Events carry identifiers, counts, and codes only.
- Strict allowlist metadata policy: rejects arbitrary objects, raw bytes, nested
  structures, and unapproved metadata fields.
- Captures: actor_id, tenant_id, action, resource_id, resource_type, timestamp,
  request_id, outcome, and allowlisted operational metadata.
- Cryptographic hash-chains and write-once immutable storage belong to M7.
"""

import logging
import math
import re
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Protocol, runtime_checkable
from uuid import uuid4

from doc_extractor.api.contracts.records import AuditEventRecord

# Approved document lifecycle states (SCH-012, IngestionState, PRD §20)
APPROVED_DOCUMENT_STATUSES = frozenset({
    "queued",
    "validating",
    "preprocessing",
    "ready_for_extraction",
    "processing",
    "auto_approved",
    "needs_review",
    "reviewed",
    "rejected",
    "rejected_unsupported",
    "rejected_malware",
    "failed",
})

# Approved MIME types and canonical short extension values (Settings.allowed_mime_types, FR-ING-002)
APPROVED_FILE_TYPES = frozenset({
    "application/pdf",
    "image/jpeg",
    "image/png",
    "image/tiff",
    "pdf",
    "jpeg",
    "jpg",
    "png",
    "tiff",
    "tif",
})

# Machine-readable token format for outcome_code and error_code (e.g., "SUCCESS", "RATE_LIMITED", "MALWARE_DETECTED")
# Strictly disallows spaces, lowercase free text, customer information, and punctuation
_CODE_TOKEN_REGEX = re.compile(r"^[A-Z0-9_]{2,64}$")

# Identifier format for schema_id and rule_id matching repository contracts (e.g., "casa_account_opening", "R1")
_IDENTIFIER_REGEX = re.compile(r"^[a-zA-Z0-9_\-\.]{1,64}$")

# Integer schema version string format (e.g., "1", "2")
_INTEGER_VERSION_REGEX = re.compile(r"^[1-9]\d*$")

# Semantic version format conforming to SemVer 2.0.0 with positive major version >= 1 (e.g., "1.0.0", "2.3.4")
# M4 canonical schema versions strictly require major version >= 1 (SCH-006, PRD §6.2)
_SEMVER_REGEX = re.compile(
    r"^([1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    r"(?:-((?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?"
    r"(?:\+([0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$"
)

# Documented allowlist of approved operational metadata fields and their allowed primitive types (PRD §21 / SEC-012)
ALLOWED_OPERATIONAL_METADATA_FIELDS = {
    "status": str,
    "duration_ms": (int, float),
    "byte_count": int,
    "page_count": int,
    "file_type": str,
    "outcome_code": str,
    "error_code": str,
    "rule_id": str,
    "schema_id": str,
    "schema_version": (int, str),
    "attempt_count": int,
}

# Maximum character limit on permitted string metadata values to prevent smuggling payload text
MAX_METADATA_STRING_LENGTH = 128


def _validate_field(key: str, val: Any) -> Optional[Any]:
    """
    Applies strict field-specific validation to an allowlisted operational metadata field.
    Returns the validated, normalized value or None if invalid.
    """
    # 1. status: Approved document lifecycle state
    if key == "status":
        if isinstance(val, str):
            candidate = val.strip().lower()
            if candidate in APPROVED_DOCUMENT_STATUSES:
                return candidate
        return None

    # 2. file_type: Approved MIME type or canonical extension
    if key == "file_type":
        if isinstance(val, str):
            candidate = val.strip().lower()
            if candidate in APPROVED_FILE_TYPES:
                return candidate
        return None

    # 3. outcome_code: Uppercase machine-readable outcome token
    if key == "outcome_code":
        if isinstance(val, str):
            candidate = val.strip().upper()
            if _CODE_TOKEN_REGEX.match(candidate):
                return candidate
        return None

    # 4. error_code: Uppercase machine-readable error token
    if key == "error_code":
        if isinstance(val, str):
            candidate = val.strip().upper()
            if _CODE_TOKEN_REGEX.match(candidate):
                return candidate
        return None

    # 5. rule_id: Repository rule identifier
    if key == "rule_id":
        if isinstance(val, str):
            candidate = val.strip()
            if _IDENTIFIER_REGEX.match(candidate):
                return candidate
        return None

    # 6. schema_id: Repository schema identifier
    if key == "schema_id":
        if isinstance(val, str):
            candidate = val.strip()
            if _IDENTIFIER_REGEX.match(candidate):
                return candidate
        return None

    # 7. schema_version: Positive canonical integer >= 1 or valid SemVer string with major >= 1
    if key == "schema_version":
        if isinstance(val, bool):
            return None
        if isinstance(val, int) and val >= 1:
            return str(val)
        if isinstance(val, str):
            candidate = val.strip()
            if not candidate:
                return None
            if _INTEGER_VERSION_REGEX.match(candidate):
                return candidate
            if _SEMVER_REGEX.match(candidate):
                return candidate
        return None

    # 8. duration_ms: Finite, non-negative numeric (int or float)
    if key == "duration_ms":
        if isinstance(val, bool):
            return None
        if isinstance(val, (int, float)):
            if math.isfinite(val) and val >= 0.0:
                return float(val)
        return None

    # 9. byte_count: Non-negative integer counter
    if key == "byte_count":
        if isinstance(val, bool) or not isinstance(val, int):
            return None
        if val >= 0:
            return val
        return None

    # 10. page_count: Non-negative integer counter
    if key == "page_count":
        if isinstance(val, bool) or not isinstance(val, int):
            return None
        if val >= 0:
            return val
        return None

    # 11. attempt_count: Non-negative integer counter
    if key == "attempt_count":
        if isinstance(val, bool) or not isinstance(val, int):
            return None
        if val >= 0:
            return val
        return None

    return None


def sanitize_audit_metadata(metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Sanitizes operational audit metadata against a strict allowlist with field-specific
    validation rules (PRD §21 & SEC-012).

    Policy:
    1. Only explicitly allowlisted operational metadata fields are retained.
    2. Unknown fields, arbitrary keys, tokens, credentials, and free-form PII are discarded.
    3. Nested dictionaries, lists, tuples, sets, bytes, file handles, and arbitrary objects
       are strictly rejected/discarded.
    4. Each allowlisted field enforces strict, field-specific constraints:
       - status: Approved document lifecycle states (SCH-012, IngestionState).
       - file_type: Approved MIME types and canonical extensions.
       - outcome_code / error_code: Validated uppercase machine-readable code tokens.
       - rule_id / schema_id: Validated repository identifier format.
       - schema_version: Positive canonical integer >= 1 (normalized to string) or valid SemVer string with major >= 1.
       - duration_ms: Finite, non-negative float.
       - byte_count / page_count / attempt_count: Finite, non-negative integers (booleans rejected).
    5. Arbitrary user-provided free text with spaces or unstructured sentences is rejected.
    6. Invalid or unrecognized values are dropped from the sanitized result.
    """
    if not metadata or not isinstance(metadata, dict):
        return {}

    sanitized: Dict[str, Any] = {}
    for key, val in metadata.items():
        if not isinstance(key, str):
            continue
        norm_key = key.strip()
        if norm_key not in ALLOWED_OPERATIONAL_METADATA_FIELDS:
            # Policy: Discard all unapproved / unknown fields
            continue

        # Explicitly reject nested objects, iterables, and raw binary/byte data
        if isinstance(val, (bytes, bytearray, memoryview, dict, list, tuple, set)):
            continue

        validated_val = _validate_field(norm_key, val)
        if validated_val is not None:
            sanitized[norm_key] = validated_val

    return sanitized


@runtime_checkable
class AuditLogger(Protocol):
    """Protocol for recording operational and security audit events."""

    async def log_event(self, event: AuditEventRecord) -> None:
        """Record a pre-constructed audit event."""
        ...

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
        """Convenience method to construct, sanitize, and record an audit event."""
        ...


class SafeStructuredAuditLogger:
    """
    Reference implementation of AuditLogger that formats structured events
    and logs them via Python's logging subsystem with strict allowlist redaction.
    """

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self._logger = logger or logging.getLogger("doc_extractor.audit")

    async def log_event(self, event: AuditEventRecord) -> None:
        """
        Record a pre-constructed audit event with fail-safe error handling.
        Policy: Audit serialization failures log an error event instead of crashing
        the business transaction, ensuring traceability without operational outage.
        """
        try:
            sanitized_meta = sanitize_audit_metadata(event.metadata)
            event_dict = event.model_dump(mode="json")
            event_dict["metadata"] = sanitized_meta

            self._logger.info(
                "AUDIT_EVENT: %s on %s:%s by %s [%s]",
                event.action,
                event.resource_type,
                event.resource_id,
                event.actor_id,
                event.outcome,
                extra={
                    "request_id": event.request_id,
                    "tenant_id": event.tenant_id,
                    "extra_data": event_dict,
                },
            )
        except Exception as exc:
            self._logger.error(
                "AUDIT_FAILURE: Failed to serialize audit event %s (%s): %s",
                getattr(event, "event_id", "unknown"),
                getattr(event, "action", "unknown"),
                str(exc),
                extra={
                    "request_id": getattr(event, "request_id", None),
                    "tenant_id": getattr(event, "tenant_id", "unknown"),
                },
            )

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
        event = AuditEventRecord(
            event_id=f"audit_{uuid4().hex[:16]}",
            actor_id=actor_id,
            tenant_id=tenant_id,
            action=action,
            resource_id=resource_id,
            resource_type=resource_type,
            timestamp=datetime.now(timezone.utc),
            request_id=request_id,
            outcome=outcome,
            metadata=sanitize_audit_metadata(metadata),
        )
        await self.log_event(event)
