"""
Repository protocol interfaces for the Document Extraction Engine (Module M5).

Adheres to:
- Hexagonal architecture / ports and adapters.
- Explicit method signatures and typed contracts.
- Mandatory tenant scoping on all data operations to enforce multi-tenant isolation.
- No concrete database adapters implemented in this phase.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from doc_extractor.api.contracts.records import (
    ApiKeyRecord,
    AuditEventRecord,
    DocumentRecord,
    ReviewTaskRecord,
    WebhookDeliveryRecord,
    WebhookSubscriptionRecord,
)


@runtime_checkable
class DocumentMetadataRepository(Protocol):
    """Repository port for persisting and querying document metadata."""

    async def get_by_id(self, tenant_id: str, document_id: str) -> Optional[DocumentRecord]:
        """Retrieve a single document metadata record scoped strictly to the tenant."""
        ...

    async def create(self, tenant_id: str, record: DocumentRecord) -> DocumentRecord:
        """Persist a new document metadata record."""
        ...

    async def update_status(
        self,
        tenant_id: str,
        document_id: str,
        status: str,
        error_details: Optional[Dict[str, Any]] = None,
        storage_path: Optional[str] = None,
        page_count: Optional[int] = None,
    ) -> Optional[DocumentRecord]:
        """Update processing status and error diagnostics for a document."""
        ...

    async def list_by_tenant(
        self,
        tenant_id: str,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[DocumentRecord]:
        """List documents belonging exclusively to the given tenant with optional status filter."""
        ...

    async def delete(self, tenant_id: str, document_id: str) -> bool:
        """Delete a document record if owned by tenant."""
        ...


@runtime_checkable
class ReviewTaskRepository(Protocol):
    """Repository port for human-in-the-loop review workbench tasks."""

    async def get_by_id(self, tenant_id: str, task_id: str) -> Optional[ReviewTaskRecord]:
        """Retrieve a review task by its ID scoped to the tenant."""
        ...

    async def get_by_document_id(self, tenant_id: str, document_id: str) -> Optional[ReviewTaskRecord]:
        """Retrieve the review task associated with a specific document."""
        ...

    async def create(self, tenant_id: str, task: ReviewTaskRecord) -> ReviewTaskRecord:
        """Create a new human review task."""
        ...

    async def update_status(
        self,
        tenant_id: str,
        task_id: str,
        status: str,
        reviewer_id: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> Optional[ReviewTaskRecord]:
        """Update review task status, assigned reviewer, and review outcome notes."""
        ...

    async def list_tasks(
        self,
        tenant_id: str,
        status: Optional[str] = None,
        reviewer_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[ReviewTaskRecord]:
        """List review tasks for the tenant with pagination and optional filtering."""
        ...


@runtime_checkable
class ApiKeyRepository(Protocol):
    """Repository port for API key credential storage and lookup (TASK-5.7)."""

    async def get_by_key_hash(self, key_hash: str) -> Optional[ApiKeyRecord]:
        """Look up an active API key by its cryptographic hash for authentication."""
        ...

    async def get_by_id(self, tenant_id: str, key_id: str) -> Optional[ApiKeyRecord]:
        """Retrieve key metadata by ID scoped to the tenant."""
        ...

    async def create(self, tenant_id: str, key_record: ApiKeyRecord) -> ApiKeyRecord:
        """Register a newly generated API key for a tenant."""
        ...

    async def revoke(self, tenant_id: str, key_id: str) -> bool:
        """Revoke an active API key immediately."""
        ...

    async def list_keys(
        self,
        tenant_id: str,
        limit: int = 50,
        offset: int = 0,
    ) -> List[ApiKeyRecord]:
        """List all API keys belonging to the tenant."""
        ...


@runtime_checkable
class WebhookSubscriptionRepository(Protocol):
    """Repository port for outbound webhook subscriptions (TASK-5.6)."""

    async def get_by_id(self, tenant_id: str, subscription_id: str) -> Optional[WebhookSubscriptionRecord]:
        """Retrieve a webhook subscription by ID scoped to the tenant."""
        ...

    async def list_by_tenant(
        self,
        tenant_id: str,
        event_type: Optional[str] = None,
    ) -> List[WebhookSubscriptionRecord]:
        """List active webhook subscriptions for a tenant, optionally filtered by event type."""
        ...

    async def create(
        self,
        tenant_id: str,
        subscription: WebhookSubscriptionRecord,
    ) -> WebhookSubscriptionRecord:
        """Register a new webhook subscription for a tenant."""
        ...

    async def update(
        self,
        tenant_id: str,
        subscription_id: str,
        update_data: Dict[str, Any],
    ) -> Optional[WebhookSubscriptionRecord]:
        """Update subscription properties (URL, event types, active status)."""
        ...

    async def delete(self, tenant_id: str, subscription_id: str) -> bool:
        """Delete a webhook subscription."""
        ...


@runtime_checkable
class WebhookDeliveryRepository(Protocol):
    """Repository port for tracking webhook delivery attempts and audit logs."""

    async def record_delivery(
        self,
        tenant_id: str,
        delivery: WebhookDeliveryRecord,
    ) -> WebhookDeliveryRecord:
        """Record an outbound delivery attempt."""
        ...

    async def get_by_id(
        self,
        tenant_id: str,
        delivery_id: str,
    ) -> Optional[WebhookDeliveryRecord]:
        """Retrieve a delivery attempt record by ID."""
        ...

    async def list_by_subscription(
        self,
        tenant_id: str,
        subscription_id: str,
        limit: int = 50,
        offset: int = 0,
    ) -> List[WebhookDeliveryRecord]:
        """List delivery history for a webhook subscription."""
        ...

    async def update_delivery_status(
        self,
        tenant_id: str,
        delivery_id: str,
        status: str,
        response_status_code: Optional[int] = None,
        error_message: Optional[str] = None,
        next_retry_at: Optional[datetime] = None,
    ) -> Optional[WebhookDeliveryRecord]:
        """Update status and outcome of an in-flight delivery attempt."""
        ...


@runtime_checkable
class AuditRepository(Protocol):
    """Repository port for structured audit event persistence (SEC-012)."""

    async def append_event(self, event: AuditEventRecord) -> None:
        """Append a structured, non-PII audit event."""
        ...

    async def query_events(
        self,
        tenant_id: str,
        resource_id: Optional[str] = None,
        action: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[AuditEventRecord]:
        """Query historical audit events for compliance inspection."""
        ...
