"""
Durable message queue protocol interfaces (Module M5).

Adheres to:
- PRD ARC-003: Durable, asynchronous processing decoupled from HTTP request cycles.
- Provider-independent contracts for producers and consumers.
- Idempotent message handling via correlation and deduplication keys.
- Negative acknowledgement with retry and backoff semantics.
- Message lease visibility and heartbeat extensions.
- STRICT PROHIBITION: FastAPI BackgroundTasks, process-local queues, and in-memory
  production queues are not acceptable substitutes for durable processing.
"""

from typing import List, Protocol, Sequence, runtime_checkable
from doc_extractor.api.contracts.records import DocumentProcessingMessage, QueueEnvelope


@runtime_checkable
class QueueProducer(Protocol):
    """
    Contract for publishing document extraction and pipeline processing tasks
    to a durable broker.
    """

    async def publish(
        self,
        message: DocumentProcessingMessage,
        delay_seconds: int = 0,
    ) -> str:
        """
        Publish a typed document-processing message to the durable queue.

        Args:
            message: Strongly typed message containing document_id, tenant_id,
                     event_id, idempotency key, and execution payload.
            delay_seconds: Optional execution delay before message becomes visible.

        Returns:
            Durable broker message tracking identifier.
        """
        ...

    async def publish_batch(
        self,
        messages: Sequence[DocumentProcessingMessage],
    ) -> List[str]:
        """
        Publish a batch of messages atomically or sequentially depending on broker.

        Returns:
            List of published message tracking identifiers in corresponding order.
        """
        ...


@runtime_checkable
class QueueConsumer(Protocol):
    """
    Contract for worker processes consuming document processing messages
    from a durable broker.
    """

    async def receive_messages(
        self,
        max_messages: int = 10,
        visibility_timeout_seconds: int = 60,
    ) -> List[QueueEnvelope]:
        """
        Fetch available messages from the durable queue and acquire lease.

        Args:
            max_messages: Maximum batch size to retrieve (1-100).
            visibility_timeout_seconds: Initial lease duration in seconds during which
                                        no other consumer will receive these messages.

        Returns:
            List of QueueEnvelope objects containing receipt handles and messages.
        """
        ...

    async def ack(self, receipt_handle: str) -> None:
        """
        Positively acknowledge successful processing, permanently deleting the message
        from the queue.

        Args:
            receipt_handle: The lease handle received from receive_messages.
        """
        ...

    async def nack(
        self,
        receipt_handle: str,
        requeue: bool = True,
        backoff_seconds: int = 0,
    ) -> None:
        """
        Negatively acknowledge a message due to processing error or transient failure.

        Args:
            receipt_handle: The lease handle received from receive_messages.
            requeue: Whether to return the message to the queue for retry. If False,
                     message is routed to Dead Letter Queue (DLQ).
            backoff_seconds: Delay before the message becomes visible for redelivery.
        """
        ...

    async def extend_lease(
        self,
        receipt_handle: str,
        extension_seconds: int = 60,
    ) -> None:
        """
        Heartbeat to extend the lease of an in-flight long-running processing task,
        preventing premature visibility timeout and duplicate processing.

        Args:
            receipt_handle: The lease handle received from receive_messages.
            extension_seconds: Number of seconds to extend the lease by.
        """
        ...
