"""
Exception hierarchy for Schema Metamodel, Validation, and Serialization.
Conforms to PRD §7.2 API-010 error envelopes and API-011 (no PII in errors).
"""

from typing import Any, Dict, List, Optional

try:
    from doc_extractor.exceptions import DocumentExtractorError
except ImportError:
    class DocumentExtractorError(Exception):  # pragma: no cover
        """Fallback base exception when doc_extractor is not in pythonpath."""
        def __init__(
            self,
            message: str,
            code: str = "INTERNAL_ERROR",
            request_id: Optional[str] = None,
            details: Optional[Dict[str, Any]] = None,
        ) -> None:
            super().__init__(message)
            self.message = message
            self.code = code
            self.request_id = request_id
            self.details = details or {}

        def to_envelope(self) -> Dict[str, Any]:
            return {
                "error": {
                    "code": self.code,
                    "message": self.message,
                    "request_id": self.request_id,
                    "details": self.details,
                }
            }


class SchemaLifecycleError(DocumentExtractorError):
    """Base exception for all schema lifecycle and serializer operations."""

    def __init__(
        self,
        message: str,
        code: str = "SCHEMA_LIFECYCLE_ERROR",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message=message, code=code, details=details)


class SchemaValidationError(SchemaLifecycleError):
    """Raised when a schema violates metamodel constraints (SCH-001, SCH-002)."""

    def __init__(
        self,
        message: str,
        errors: Optional[List[Dict[str, Any]]] = None,
        code: str = "SCHEMA_VALIDATION_ERROR",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        merged = dict(details or {})
        if errors:
            merged["validation_errors"] = errors
        super().__init__(message=message, code=code, details=merged)
        self.errors = errors or []


class SchemaSerializationError(SchemaLifecycleError):
    """Raised when YAML or JSON parsing or dumping fails (SCH-003)."""

    def __init__(
        self,
        message: str,
        format_type: str = "yaml",
        code: str = "SCHEMA_SERIALIZATION_ERROR",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        merged = dict(details or {})
        merged["format"] = format_type
        super().__init__(message=message, code=code, details=merged)
        self.format_type = format_type


class SchemaVersionError(SchemaLifecycleError):
    """Raised when schema version conversion or parsing fails."""

    def __init__(
        self,
        message: str,
        version: Any,
        code: str = "SCHEMA_VERSION_ERROR",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        merged = dict(details or {})
        merged["version"] = str(version)
        super().__init__(message=message, code=code, details=merged)
        self.version = version


class InvalidStateTransitionError(SchemaLifecycleError):
    """Raised when an invalid lifecycle state transition is attempted (SCH-004)."""

    def __init__(
        self,
        message: str,
        from_status: Any,
        to_status: Any,
        schema_id: Optional[str] = None,
        version: Optional[int] = None,
        code: str = "INVALID_STATE_TRANSITION",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        merged = dict(details or {})
        merged["from_status"] = str(getattr(from_status, "value", from_status))
        merged["to_status"] = str(getattr(to_status, "value", to_status))
        if schema_id:
            merged["schema_id"] = schema_id
        if version is not None:
            merged["version"] = version
        super().__init__(message=message, code=code, details=merged)
        self.from_status = from_status
        self.to_status = to_status


class SchemaNotApprovedError(SchemaLifecycleError):
    """
    Raised when attempting to bind or resolve a non-approved schema in production (SCH-004, API-010).
    Standard PRD code: SCHEMA_NOT_APPROVED.
    """

    def __init__(
        self,
        message: str = "This schema isn't approved for production.",
        schema_id: Optional[str] = None,
        version: Optional[int] = None,
        code: str = "SCHEMA_NOT_APPROVED",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        merged = dict(details or {})
        if schema_id:
            merged["schema_id"] = schema_id
        if version is not None:
            merged["version"] = version
        super().__init__(message=message, code=code, details=merged)
        self.schema_id = schema_id
        self.version = version


class SchemaNotFoundError(SchemaLifecycleError):
    """Raised when a schema or specific schema version is not found."""

    def __init__(
        self,
        message: str,
        schema_id: str,
        version: Optional[int] = None,
        tenant_id: Optional[str] = None,
        code: str = "SCHEMA_NOT_FOUND",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        merged = dict(details or {})
        merged["schema_id"] = schema_id
        if version is not None:
            merged["version"] = version
        if tenant_id is not None:
            merged["tenant_id"] = tenant_id
        super().__init__(message=message, code=code, details=merged)
        self.schema_id = schema_id
        self.version = version
        self.tenant_id = tenant_id


class SchemaPromotionNotAllowedError(InvalidStateTransitionError):
    """Raised when direct promotion to approved is attempted without the TASK-4.2 approval service."""

    def __init__(
        self,
        message: str = "Direct promotion to approved is reserved for the TASK-4.2 approval service.",
        schema_id: Optional[str] = None,
        version: Optional[int] = None,
        from_status: Any = "any",
        code: str = "SCHEMA_PROMOTION_NOT_ALLOWED",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message=message,
            from_status=from_status,
            to_status="approved",
            schema_id=schema_id,
            version=version,
            code=code,
            details=details,
        )


class SchemaRollbackError(SchemaLifecycleError):
    """Raised when schema rollback fails or targets an invalid version (SCH-006)."""

    def __init__(
        self,
        message: str,
        schema_id: str,
        target_version: int,
        code: str = "SCHEMA_ROLLBACK_ERROR",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        merged = dict(details or {})
        merged["schema_id"] = schema_id
        merged["target_version"] = target_version
        super().__init__(message=message, code=code, details=merged)
        self.schema_id = schema_id
        self.target_version = target_version


class TenantAccessError(SchemaLifecycleError):
    """Raised when accessing schemas across tenant boundaries or without tenant context."""

    def __init__(
        self,
        message: str = "Tenant access violation or missing tenant identifier.",
        tenant_id: Optional[str] = None,
        code: str = "TENANT_ACCESS_ERROR",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        merged = dict(details or {})
        if tenant_id:
            merged["tenant_id"] = tenant_id
        super().__init__(message=message, code=code, details=merged)
        self.tenant_id = tenant_id


class UnauthenticatedError(SchemaLifecycleError):
    """Raised when authentication is missing, invalid, or principal is inactive (SEC-001)."""

    def __init__(
        self,
        message: str = "Authentication required or principal is inactive.",
        code: str = "UNAUTHENTICATED",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message=message, code=code, details=details)


class ForbiddenActionError(SchemaLifecycleError):
    """Raised when a principal lacks required permissions or roles (SEC-003)."""

    def __init__(
        self,
        message: str = "Principal lacks required role or permission for this operation.",
        code: str = "FORBIDDEN",
        principal_id: Optional[str] = None,
        permission: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        merged = dict(details or {})
        if principal_id:
            merged["principal_id"] = principal_id
        if permission:
            merged["permission"] = permission
        super().__init__(message=message, code=code, details=merged)
        self.principal_id = principal_id
        self.permission = permission


class SchemaSelfApprovalError(ForbiddenActionError):
    """
    Raised when a schema author attempts to approve their own schema version (SCH-005, SEC-003).
    Standard PRD proposed code: SCHEMA_SELF_APPROVAL.
    """

    def __init__(
        self,
        message: str = "A second person must approve this schema. Authors cannot approve their own schema version.",
        schema_id: Optional[str] = None,
        version: Optional[int] = None,
        author: Optional[str] = None,
        code: str = "SCHEMA_SELF_APPROVAL",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        merged = dict(details or {})
        if schema_id:
            merged["schema_id"] = schema_id
        if version is not None:
            merged["version"] = version
        if author:
            merged["author"] = author
        super().__init__(
            message=message,
            code=code,
            principal_id=author,
            permission="schema:approve",
            details=merged,
        )
        self.schema_id = schema_id
        self.version = version
        self.author = author


class BenchmarkVerificationError(SchemaLifecycleError):
    """
    Raised when test harness results fail verification or do not qualify for approval (SCH-005, SCH-007).
    Standard PRD code: HARNESS_RESULT_INVALID.
    """

    def __init__(
        self,
        message: str = "Test harness result failed validation or does not qualify for schema approval.",
        harness_run_id: Optional[str] = None,
        schema_id: Optional[str] = None,
        version: Optional[int] = None,
        code: str = "HARNESS_RESULT_INVALID",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        merged = dict(details or {})
        if harness_run_id:
            merged["harness_run_id"] = harness_run_id
        if schema_id:
            merged["schema_id"] = schema_id
        if version is not None:
            merged["version"] = version
        super().__init__(message=message, code=code, details=merged)
        self.harness_run_id = harness_run_id
        self.schema_id = schema_id
        self.version = version
