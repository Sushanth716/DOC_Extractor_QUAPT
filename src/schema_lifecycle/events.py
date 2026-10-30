"""
Audit and Lifecycle Events for Schema Evolution (SCH-006, PRD §7.1, §10.1).
Records immutable audit trail of all schema lifecycle state transitions and rollbacks.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional
import uuid

from pydantic import BaseModel, ConfigDict, Field

from schema_lifecycle.enums import SchemaStatus


class LifecycleEventType(str, Enum):
    """
    Schema lifecycle event categories:
    - CREATED: Initial or incremental schema version created
    - STATUS_CHANGED: Status transition (e.g. draft -> testing)
    - APPROVED: Schema promoted to approved (matches PRD §7.1 schema.approved)
    - RETIRED: Schema retired/decommissioned
    - ROLLBACK: Rollback event restoring a prior approved version (SCH-006)
    """
    CREATED = "schema.created"
    STATUS_CHANGED = "schema.status_changed"
    APPROVED = "schema.approved"
    RETIRED = "schema.retired"
    ROLLBACK = "schema.rollback"


class LifecycleEvent(BaseModel):
    """
    Immutable audit event recording a lifecycle transition, version creation, or rollback.
    Adheres to PRD §7.1 and SCH-006 append-only ledger requirements.
    """
    model_config = ConfigDict(frozen=True)

    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str = Field(..., description="Tenant identifier for multi-tenant isolation")
    schema_id: str = Field(..., description="Schema identifier")
    version: int = Field(..., description="Canonical integer schema version")
    event_type: LifecycleEventType = Field(..., description="Type of lifecycle event")
    actor: str = Field(..., description="Author or operator initiating the event")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC event creation timestamp",
    )
    from_status: Optional[SchemaStatus] = Field(
        default=None,
        description="Previous lifecycle status",
    )
    to_status: Optional[SchemaStatus] = Field(
        default=None,
        description="New lifecycle status after transition",
    )
    details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Contextual metadata (e.g. diff summary, target_version, reason)",
    )
