"""
Canonical error envelope contracts compliant with PRD §7.2 API-010.

Format:
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable message",
    "request_id": "req_identifier",
    "details": {}
  }
}
"""

from typing import Any, Dict, Optional
from pydantic import BaseModel, ConfigDict, Field


class ErrorDetail(BaseModel):
    """Inner error payload required by PRD API-010."""
    model_config = ConfigDict(extra="forbid")

    code: str = Field(..., description="Standard machine-readable error code")
    message: str = Field(..., description="Safe human-readable error description")
    request_id: Optional[str] = Field(None, description="Request identifier for distributed tracing")
    details: Dict[str, Any] = Field(default_factory=dict, description="Structured error context details")


class ErrorEnvelope(BaseModel):
    """PRD API-010 top-level canonical error envelope."""
    model_config = ConfigDict(extra="forbid")

    error: ErrorDetail = Field(..., description="Canonical error container")
