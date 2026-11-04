"""
Typed data contracts for repositories, queues, and domain entities (Module M5).

All records are strongly typed, immutable or strictly validated via Pydantic,
and enforce tenant scoping where appropriate.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


def utc_now() -> datetime:
    """Helper returning timezone-aware current UTC time."""
    return datetime.now(timezone.utc)


class DocumentRecord(BaseModel):
    """Metadata record for an ingested/processed document."""
    model_config = ConfigDict(extra="forbid")

    document_id: str = Field(..., min_length=1, description="Unique document identifier")
    tenant_id: str = Field(..., min_length=1, description="Owning tenant identifier")
    status: str = Field(default="queued", description="Processing lifecycle status (SCH-012)")
    filename: str = Field(..., min_length=1, description="Original filename without path")
    mime_type: str = Field(..., description="Verified MIME type")
    file_size_bytes: int = Field(..., ge=0, description="Size of file in bytes")
    page_count: int = Field(default=0, ge=0, description="Total number of pages rendered")
    sha256_hash: str = Field(..., min_length=64, max_length=64, description="Cryptographic SHA-256 hash")
    created_at: datetime = Field(default_factory=utc_now, description="UTC ingestion timestamp")
    updated_at: datetime = Field(default_factory=utc_now, description="UTC last-update timestamp")
    storage_path: Optional[str] = Field(None, description="Tenant-isolated relative filesystem path")
    schema_id: Optional[str] = Field(None, description="Associated extraction schema ID")
    schema_version: Optional[str] = Field(None, description="Associated schema version")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Redacted, non-PII metadata")
    error_details: Optional[Dict[str, Any]] = Field(None, description="Error diagnostics if failed")


class ReviewTaskRecord(BaseModel):
    """Human-in-the-loop review task record for the extraction workbench."""
    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(..., min_length=1, description="Unique review task identifier")
    document_id: str = Field(..., min_length=1, description="Associated document ID")
    tenant_id: str = Field(..., min_length=1, description="Owning tenant identifier")
    status: str = Field(default="pending", description="Task status (pending, in_review, approved, rejected)")
    assigned_reviewer_id: Optional[str] = Field(None, description="Principal ID of assigned reviewer")
    priority: int = Field(default=0, description="Task processing priority")
    created_at: datetime = Field(default_factory=utc_now, description="Task creation timestamp")
    updated_at: datetime = Field(default_factory=utc_now, description="Last update timestamp")
    completed_at: Optional[datetime] = Field(None, description="Task completion timestamp")
    reviewer_notes: Optional[str] = Field(None, description="Reviewer notes or reason for rejection")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Redacted task context metadata")


class ApiKeyRecord(BaseModel):
    """API key credential and authorization scope record."""
    model_config = ConfigDict(extra="forbid")

    key_id: str = Field(..., min_length=1, description="Public API key identifier (prefix)")
    tenant_id: str = Field(..., min_length=1, description="Bound tenant identifier")
    name: str = Field(..., min_length=1, description="Descriptive label for the API key")
    key_hash: str = Field(..., min_length=32, description="Salted cryptographic hash of secret token")
    key_prefix: str = Field(..., min_length=4, max_length=16, description="Visible key prefix for identification")
    scopes: List[str] = Field(default_factory=list, description="Granted API scopes/permissions")
    roles: List[str] = Field(default_factory=list, description="Canonical roles assigned to the API key")
    created_at: datetime = Field(default_factory=utc_now, description="Key generation timestamp")
    expires_at: Optional[datetime] = Field(None, description="Optional key expiration timestamp")
    revoked_at: Optional[datetime] = Field(None, description="Timestamp when key was revoked, if any")
    is_active: bool = Field(default=True, description="Whether key is currently active")
    created_by: Optional[str] = Field(None, description="Principal ID that created this key")


class WebhookSubscriptionRecord(BaseModel):
    """Tenant webhook event subscription registration."""
    model_config = ConfigDict(extra="forbid")

    subscription_id: str = Field(..., min_length=1, description="Unique webhook subscription identifier")
    tenant_id: str = Field(..., min_length=1, description="Owning tenant identifier")
    target_url: str = Field(..., min_length=1, description="Destination HTTPS webhook endpoint")
    event_types: List[str] = Field(..., min_length=1, description="List of subscribed event patterns")
    secret_hash: str = Field(..., description="Hash of webhook signing secret")
    is_active: bool = Field(default=True, description="Whether subscription is enabled")
    created_at: datetime = Field(default_factory=utc_now, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=utc_now, description="Last update timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Custom delivery metadata")


class WebhookDeliveryRecord(BaseModel):
    """Audit record of a single webhook dispatch attempt."""
    model_config = ConfigDict(extra="forbid")

    delivery_id: str = Field(..., min_length=1, description="Unique dispatch attempt identifier")
    subscription_id: str = Field(..., min_length=1, description="Target subscription identifier")
    tenant_id: str = Field(..., min_length=1, description="Owning tenant identifier")
    event_id: str = Field(..., min_length=1, description="Triggering event identifier")
    event_type: str = Field(..., min_length=1, description="Type of event dispatched")
    attempt_number: int = Field(default=1, ge=1, description="Dispatch attempt sequence number")
    status: str = Field(default="pending", description="Delivery status (pending, delivered, failed, dead_letter)")
    response_status_code: Optional[int] = Field(None, description="HTTP response code returned by target")
    error_message: Optional[str] = Field(None, description="Error diagnostics if dispatch failed")
    created_at: datetime = Field(default_factory=utc_now, description="Dispatch trigger timestamp")
    completed_at: Optional[datetime] = Field(None, description="Dispatch resolution timestamp")
    next_retry_at: Optional[datetime] = Field(None, description="Scheduled retry timestamp if transient failure")


class AuditEventRecord(BaseModel):
    """Structured audit trail event record (SEC-012, no PII/PHI)."""
    model_config = ConfigDict(extra="forbid")

    event_id: str = Field(..., min_length=1, description="Unique audit event identifier")
    actor_id: str = Field(..., min_length=1, description="Principal or system actor identifier")
    tenant_id: str = Field(..., min_length=1, description="Scoping tenant identifier")
    action: str = Field(..., min_length=1, description="Canonical action type (e.g., document.upload)")
    resource_id: str = Field(..., min_length=1, description="Identifier of affected resource")
    resource_type: str = Field(..., min_length=1, description="Type of affected resource (e.g., document, schema)")
    timestamp: datetime = Field(default_factory=utc_now, description="UTC event occurrence timestamp")
    request_id: Optional[str] = Field(None, description="Correlated request ID")
    outcome: str = Field(default="SUCCESS", description="Event outcome status (e.g., SUCCESS, DENIED, FAILED)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Redacted, non-PII audit context")


class DocumentProcessingMessage(BaseModel):
    """
    Durable typed message payload for asynchronous document processing (ARC-003).
    Correlates document_id, tenant_id, and event_id with idempotency and retry semantics.
    """
    model_config = ConfigDict(extra="forbid")

    message_id: str = Field(..., min_length=1, description="Unique queue message identifier")
    document_id: str = Field(..., min_length=1, description="Target document identifier")
    tenant_id: str = Field(..., min_length=1, description="Owning tenant identifier")
    event_id: str = Field(..., min_length=1, description="Originating event identifier")
    deduplication_id: Optional[str] = Field(None, description="Idempotency deduplication key")
    attempt_count: int = Field(default=1, ge=1, description="Current processing attempt number")
    max_retries: int = Field(default=3, ge=0, description="Maximum allowed processing retries before dead letter")
    created_at: datetime = Field(default_factory=utc_now, description="Message enqueue timestamp")
    payload: Dict[str, Any] = Field(default_factory=dict, description="Execution parameters (schema, storage, options)")
    correlation_id: Optional[str] = Field(None, description="Correlation identifier for distributed tracing")


class QueueEnvelope(BaseModel):
    """Wrapper envelope returned by queue consumers holding lease and receipt handle."""
    model_config = ConfigDict(extra="forbid")

    receipt_handle: str = Field(..., min_length=1, description="Opaque lease/receipt handle for ack/nack")
    message: DocumentProcessingMessage = Field(..., description="Strongly typed processing message")
    approximate_receive_count: int = Field(default=1, ge=1, description="Number of times message was delivered")
    visible_until: datetime = Field(..., description="Timestamp until which message visibility is leased")
