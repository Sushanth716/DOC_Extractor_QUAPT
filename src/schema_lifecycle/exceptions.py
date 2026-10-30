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

