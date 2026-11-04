"""
API Ports package (Hexagonal Architecture).
"""

from doc_extractor.api.ports.audit import AuditLogger, SafeStructuredAuditLogger, sanitize_audit_metadata
from doc_extractor.api.ports.authentication import (
    ApiPrincipal,
    AuthenticationProvider,
    PrincipalType,
    TenantContext,
)
from doc_extractor.api.ports.queue import QueueConsumer, QueueProducer
from doc_extractor.api.ports.repositories import (
    ApiKeyRepository,
    AuditRepository,
    DocumentMetadataRepository,
    ReviewTaskRepository,
    WebhookDeliveryRepository,
    WebhookSubscriptionRepository,
)

__all__ = [
    "ApiKeyRepository",
    "ApiPrincipal",
    "AuditLogger",
    "AuditRepository",
    "AuthenticationProvider",
    "DocumentMetadataRepository",
    "PrincipalType",
    "QueueConsumer",
    "QueueProducer",
    "ReviewTaskRepository",
    "SafeStructuredAuditLogger",
    "TenantContext",
    "WebhookDeliveryRepository",
    "WebhookSubscriptionRepository",
    "sanitize_audit_metadata",
]
