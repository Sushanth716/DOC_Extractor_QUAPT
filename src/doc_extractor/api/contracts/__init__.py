"""
API Contracts package.
"""

from doc_extractor.api.contracts.errors import ErrorDetail, ErrorEnvelope
from doc_extractor.api.contracts.records import (
    ApiKeyCreationResult,
    ApiKeyRecord,
    ApiKeySummary,
    AuditEventRecord,
    DocumentProcessingMessage,
    DocumentRecord,
    QueueEnvelope,
    ReviewTaskRecord,
    WebhookDeliveryRecord,
    WebhookSubscriptionRecord,
    utc_now,
)

__all__ = [
    "ErrorDetail",
    "ErrorEnvelope",
    "ApiKeyCreationResult",
    "ApiKeyRecord",
    "ApiKeySummary",
    "AuditEventRecord",
    "DocumentProcessingMessage",
    "DocumentRecord",
    "QueueEnvelope",
    "ReviewTaskRecord",
    "WebhookDeliveryRecord",
    "WebhookSubscriptionRecord",
    "utc_now",
]
