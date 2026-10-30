"""
Shared error definitions and hierarchy for Module A and engine integrations.

Adheres to:
- PRD §7.2 API-010: Standardized error code envelope
- PRD §7.2 API-011: Error messages SHALL NOT contain extracted document content or PII
- PRD §9 GR-001: Strict rejection with standard machine-readable codes
"""

from typing import Any, Dict, Optional


class DocumentExtractorError(Exception):
    """
    Base exception for all Document Extraction Engine errors.
    
    Ensures safe error formatting:
    1. Holds machine-readable standard error codes.
    2. Enforces that messages do not expose document content or PII.
    3. Formats to the PRD standard error envelope.
    """

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
        """
        Return the standardized error envelope required by PRD API-010:
        {
            "error": {
                "code": "...",
                "message": "...",
                "request_id": "...",
                "details": { ... }
            }
        }
        """
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "request_id": self.request_id,
                "details": self.details,
            }
        }


class ConfigurationError(DocumentExtractorError):
    """Raised when environment settings or module configurations are invalid."""

    def __init__(
        self,
        message: str,
        request_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message=message,
            code="CONFIGURATION_FAILED",
            request_id=request_id,
            details=details,
        )


class UnsupportedFileError(DocumentExtractorError):
    """
    Raised when an uploaded file is not supported or fails magic-byte verification.
    PRD standard error code: UNSUPPORTED_FILE (API-010, FR-ING-002, GR-001).
    """

    def __init__(
        self,
        message: str = "The uploaded file format is not supported or content is corrupt.",
        request_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message=message,
            code="UNSUPPORTED_FILE",
            request_id=request_id,
            details=details,
        )


class SecurityCheckFailedError(DocumentExtractorError):
    """
    Raised when malware scan, decompression-bomb check, or security boundary fails.
    Corresponds to PRD FR-ING-003, SEC-009, GR-001.
    """

    def __init__(
        self,
        message: str = "File rejected due to security policy or malware check.",
        code: str = "SECURITY_CHECK_FAILED",
        request_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message=message,
            code=code,
            request_id=request_id,
            details=details,
        )


class MalwareDetectedError(DocumentExtractorError):
    """
    Raised when malware scanning detects an infected or malicious file (e.g. EICAR).
    Corresponds to PRD FR-ING-003, SEC-009, GR-001.
    """

    def __init__(
        self,
        message: str = "File rejected: malware signature detected.",
        request_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message=message,
            code="MALWARE_DETECTED",
            request_id=request_id,
            details=details,
        )


class ScannerUnavailableError(DocumentExtractorError):
    """
    Raised when a required malware scanner is unreachable or fails to complete.
    Enforces fail-closed security boundary.
    """

    def __init__(
        self,
        message: str = "Malware scan failed or scanner is unavailable.",
        request_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message=message,
            code="SCANNER_UNAVAILABLE",
            request_id=request_id,
            details=details,
        )


class ConflictError(DocumentExtractorError):
    """
    Raised when an idempotency key is reused with a different request fingerprint or content.
    PRD standard error code: IDEMPOTENCY_KEY_CONFLICT.
    """

    def __init__(
        self,
        message: str = "Idempotency key reused with different request parameters or content.",
        request_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message=message,
            code="IDEMPOTENCY_KEY_CONFLICT",
            request_id=request_id,
            details=details,
        )


class DocumentProcessingError(DocumentExtractorError):
    """Raised when an internal stage (rendering, de-skewing, transformation) fails."""

    def __init__(
        self,
        message: str,
        code: str = "PROCESSING_FAILED",
        request_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message=message,
            code=code,
            request_id=request_id,
            details=details,
        )


class StorageError(DocumentExtractorError):
    """Raised when local filesystem or encrypted storage operations fail."""

    def __init__(
        self,
        message: str,
        code: str = "STORAGE_FAILED",
        request_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message=message,
            code=code,
            request_id=request_id,
            details=details,
        )


class LLMExecutionError(DocumentExtractorError):
    """
    Raised when LLM grammar-constrained extraction fails and fallback is disabled (SCH-007).
    Signals provider network failure, invalid model response, or schema validation failure.
    """

    def __init__(
        self,
        message: str = "LLM extraction execution failed under strict no-fallback policy.",
        model_name: Optional[str] = None,
        request_id: Optional[str] = None,
        code: str = "LLM_EXECUTION_FAILED",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        merged = dict(details or {})
        if model_name:
            merged["model_name"] = model_name
        super().__init__(
            message=message,
            code=code,
            request_id=request_id,
            details=merged,
        )
        self.model_name = model_name
