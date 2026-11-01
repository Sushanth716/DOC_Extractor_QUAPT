"""
Schema Repository Abstraction and In-Memory Implementation (SCH-004, SCH-006).
Provides tenant-isolated, immutable storage for schema snapshots and append-only lifecycle events.
Enforces deep immutability through defensive copies at all repository boundaries.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import copy
from datetime import datetime, timezone
import threading
from typing import Any, Dict, List, Optional, Tuple
import uuid

from pydantic import BaseModel, ConfigDict, Field, model_validator

from schema_lifecycle.enums import SchemaStatus
from schema_lifecycle.events import LifecycleEvent
from schema_lifecycle.exceptions import (
    BenchmarkVerificationError,
    InvalidStateTransitionError,
    SchemaNotFoundError,
    SchemaVersionError,
    TenantAccessError,
)
from schema_lifecycle.metamodel import SchemaDefinition


class SchemaApprovalRecord(BaseModel):
    """
    Immutable audit record representing an approved schema version (SCH-005, SEC-003).
    Captures author-approver separation, exact schema version, and passing harness run reference.
    Contains no extracted document PII.
    """
    model_config = ConfigDict(frozen=True, extra="forbid")

    approval_id: str = Field(default_factory=lambda: f"appr_{uuid.uuid4().hex[:12]}")
    tenant_id: str = Field(..., description="Tenant identifier for multi-tenant isolation")
    schema_id: str = Field(..., description="Schema identifier")
    version: int = Field(..., ge=1, description="Exact canonical integer schema version")
    author_id: str = Field(..., description="Identity of schema author")
    approver_id: str = Field(..., description="Identity of second-user approver holding Schema approver role")
    approval_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC approval timestamp",
    )
    harness_run_id: str = Field(..., description="Identifier of passing test harness run")
    benchmark_policy_version: str = Field(default="1.0.0", description="Evaluation policy version applied during benchmark")
    metric_summary: Dict[str, Any] = Field(
        default_factory=dict,
        description="PII-free summary of M1, M6, and calibration metrics",
    )
    reason: Optional[str] = Field(default=None, description="Optional sign-off reason or notes")
    lifecycle_event_id: str = Field(..., description="Reference to append-only LifecycleEvent ID")



class SchemaSnapshot(BaseModel):
    """
    Immutable snapshot of a schema definition version (SCH-006).
    Preserves schema_id, tenant_id, canonical version, author, timestamp, and diff.
    Enforces deep immutability against mutations of nested definitions and diff mappings.
    """
    model_config = ConfigDict(frozen=True)

    schema_id: str = Field(..., description="Unique schema identifier")
    tenant_id: str = Field(..., description="Tenant identifier for multi-tenant isolation")
    version: int = Field(..., description="Canonical integer schema version (>= 1)")
    status: SchemaStatus = Field(default=SchemaStatus.DRAFT, description="Current lifecycle status")
    definition: SchemaDefinition = Field(..., description="Canonical schema definition metamodel")
    author: str = Field(..., description="User or service identity that authored this version")
    created_at: datetime = Field(..., description="UTC creation timestamp")
    diff: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Deterministic structural diff against previous version",
    )
    approved_by: Optional[str] = Field(
        default=None,
        description="Approver identifier (reserved for TASK-4.2)",
    )
    harness_run_id: Optional[str] = Field(
        default=None,
        description="Linked benchmark/test harness run identifier",
    )
    parent_version: Optional[int] = Field(
        default=None,
        description="Preceding integer version from which this was derived",
    )

    @model_validator(mode="before")
    @classmethod
    def deep_copy_nested_structures(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # Defensive deep copy to guarantee input arguments cannot mutate snapshot internals
            if "definition" in data and not isinstance(data["definition"], dict):
                data["definition"] = copy.deepcopy(data["definition"])
            if "diff" in data and data["diff"] is not None:
                data["diff"] = copy.deepcopy(data["diff"])
        return data


class SchemaRepository(ABC):
    """
    Abstract interface for managing immutable schema snapshots and lifecycle audit events.
    Enforces strict tenant scoping across all storage and query methods.
    """

    @abstractmethod
    def save_snapshot(self, snapshot: SchemaSnapshot) -> None:
        """
        Persists an immutable schema snapshot.
        Fails if a snapshot with the same (tenant_id, schema_id, version) already exists.
        """
        pass

    @abstractmethod
    def get_snapshot(
        self,
        schema_id: str,
        version: int,
        tenant_id: str,
    ) -> Optional[SchemaSnapshot]:
        """Retrieves a specific schema version snapshot for a tenant."""
        pass

    @abstractmethod
    def get_latest_version(
        self,
        schema_id: str,
        tenant_id: str,
    ) -> Optional[SchemaSnapshot]:
        """Retrieves the highest integer version snapshot for a schema in a tenant."""
        pass

    @abstractmethod
    def list_versions(
        self,
        schema_id: str,
        tenant_id: str,
    ) -> List[SchemaSnapshot]:
        """Lists all version snapshots for a schema in a tenant, sorted ascending by version."""
        pass

    @abstractmethod
    def get_active_version(
        self,
        schema_id: str,
        tenant_id: str,
    ) -> Optional[SchemaSnapshot]:
        """Retrieves the currently active approved schema version snapshot for production binding."""
        pass

    @abstractmethod
    def set_active_version(
        self,
        schema_id: str,
        version: int,
        tenant_id: str,
    ) -> None:
        """Sets the active version pointer for a schema in a tenant."""
        pass

    @abstractmethod
    def update_status(
        self,
        schema_id: str,
        version: int,
        tenant_id: str,
        new_status: SchemaStatus,
    ) -> SchemaSnapshot:
        """
        Updates the lifecycle status of a specific schema version snapshot.
        Preserves the underlying definition and records history without overwriting definitions.
        """
        pass

    @abstractmethod
    def record_event(self, event: LifecycleEvent) -> None:
        """Appends a lifecycle event to the tenant-scoped audit ledger."""
        pass

    @abstractmethod
    def list_events(
        self,
        schema_id: str,
        tenant_id: str,
        version: Optional[int] = None,
    ) -> List[LifecycleEvent]:
        """Lists all audit events for a schema in a tenant, optionally filtered by version."""
        pass

    @abstractmethod
    def promote_and_activate(
        self,
        tenant_id: str,
        schema_id: str,
        version: int,
        approved_by: str,
        harness_run_id: str,
        event: LifecycleEvent,
        approval_record: Optional[SchemaApprovalRecord] = None,
    ) -> SchemaSnapshot:
        """
        Atomically promotes a schema version from TESTING to APPROVED, updates approved_by
        and harness_run_id on the snapshot, sets active production pointer, records lifecycle event,
        and persists approval record in a single atomic operation.
        """
        pass

    @abstractmethod
    def get_approval_record(
        self,
        schema_id: str,
        version: int,
        tenant_id: str,
    ) -> Optional[SchemaApprovalRecord]:
        """Retrieves the approval record for a specific approved schema version."""
        pass

    @abstractmethod
    def list_approval_records(
        self,
        schema_id: str,
        tenant_id: str,
    ) -> List[SchemaApprovalRecord]:
        """Lists all approval records for a schema in a tenant."""
        pass

    @abstractmethod
    def is_harness_run_consumed(
        self,
        harness_run_id: str,
        tenant_id: str,
    ) -> bool:
        """Checks if a harness run has already been consumed to approve a schema."""
        pass


class InMemorySchemaRepository(SchemaRepository):
    """
    In-memory reference implementation of SchemaRepository.
    Enforces strict tenant isolation, deep immutability via defensive copies,
    and append-only event auditing.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        # Key: (tenant_id, schema_id, version) -> SchemaSnapshot
        self._snapshots: Dict[Tuple[str, str, int], SchemaSnapshot] = {}
        # Key: (tenant_id, schema_id) -> active_version int
        self._active_pointers: Dict[Tuple[str, str], int] = {}
        # Append-only list of lifecycle events
        self._events: List[LifecycleEvent] = []
        # Stored approval records
        self._approval_records: List[SchemaApprovalRecord] = []
        # Consumed harness runs: Key (tenant_id, harness_run_id) -> (schema_id, version)
        self._consumed_harness_runs: Dict[Tuple[str, str], Tuple[str, int]] = {}


    def _validate_tenant(self, tenant_id: Optional[str]) -> str:
        if not tenant_id or not isinstance(tenant_id, str) or not tenant_id.strip():
            raise TenantAccessError("tenant_id must be a non-empty string for tenant isolation.")
        return tenant_id.strip()

    def save_snapshot(self, snapshot: SchemaSnapshot) -> None:
        tenant_id = self._validate_tenant(snapshot.tenant_id)
        key = (tenant_id, snapshot.schema_id, snapshot.version)

        with self._lock:
            if key in self._snapshots:
                raise SchemaVersionError(
                    f"Schema '{snapshot.schema_id}' version {snapshot.version} already exists "
                    f"for tenant '{tenant_id}'. Schema snapshots are immutable.",
                    version=snapshot.version,
                )
            # Store a defensive deep copy so future mutations of the passed object cannot affect storage
            self._snapshots[key] = copy.deepcopy(snapshot)

    def get_snapshot(
        self,
        schema_id: str,
        version: int,
        tenant_id: str,
    ) -> Optional[SchemaSnapshot]:
        tid = self._validate_tenant(tenant_id)
        key = (tid, schema_id, version)
        with self._lock:
            snap = self._snapshots.get(key)
            return copy.deepcopy(snap) if snap is not None else None

    def get_latest_version(
        self,
        schema_id: str,
        tenant_id: str,
    ) -> Optional[SchemaSnapshot]:
        tid = self._validate_tenant(tenant_id)
        with self._lock:
            versions = [
                s for (t, sid, _), s in self._snapshots.items()
                if t == tid and sid == schema_id
            ]
            if not versions:
                return None
            return copy.deepcopy(max(versions, key=lambda s: s.version))

    def list_versions(
        self,
        schema_id: str,
        tenant_id: str,
    ) -> List[SchemaSnapshot]:
        tid = self._validate_tenant(tenant_id)
        with self._lock:
            versions = [
                s for (t, sid, _), s in self._snapshots.items()
                if t == tid and sid == schema_id
            ]
            return [copy.deepcopy(s) for s in sorted(versions, key=lambda s: s.version)]

    def get_active_version(
        self,
        schema_id: str,
        tenant_id: str,
    ) -> Optional[SchemaSnapshot]:
        tid = self._validate_tenant(tenant_id)
        with self._lock:
            active_ver = self._active_pointers.get((tid, schema_id))
            if active_ver is None:
                return None
            snap = self._snapshots.get((tid, schema_id, active_ver))
            return copy.deepcopy(snap) if snap is not None else None

    def set_active_version(
        self,
        schema_id: str,
        version: int,
        tenant_id: str,
    ) -> None:
        tid = self._validate_tenant(tenant_id)
        with self._lock:
            snapshot = self._snapshots.get((tid, schema_id, version))
            if snapshot is None:
                raise SchemaNotFoundError(
                    f"Cannot set active version: Schema '{schema_id}' version {version} not found.",
                    schema_id=schema_id,
                    version=version,
                    tenant_id=tid,
                )
            self._active_pointers[(tid, schema_id)] = version

    def update_status(
        self,
        schema_id: str,
        version: int,
        tenant_id: str,
        new_status: SchemaStatus,
    ) -> SchemaSnapshot:
        tid = self._validate_tenant(tenant_id)
        key = (tid, schema_id, version)
        with self._lock:
            existing = self._snapshots.get(key)
            if existing is None:
                raise SchemaNotFoundError(
                    f"Schema '{schema_id}' version {version} not found for tenant '{tid}'.",
                    schema_id=schema_id,
                    version=version,
                    tenant_id=tid,
                )

            # Preserves immutable definition while reflecting current lifecycle status
            updated = existing.model_copy(update={"status": new_status})
            self._snapshots[key] = copy.deepcopy(updated)
            return copy.deepcopy(updated)

    def record_event(self, event: LifecycleEvent) -> None:
        self._validate_tenant(event.tenant_id)
        with self._lock:
            self._events.append(copy.deepcopy(event))

    def list_events(
        self,
        schema_id: str,
        tenant_id: str,
        version: Optional[int] = None,
    ) -> List[LifecycleEvent]:
        tid = self._validate_tenant(tenant_id)
        with self._lock:
            matches = [
                e for e in self._events
                if e.tenant_id == tid and e.schema_id == schema_id
            ]
            if version is not None:
                matches = [e for e in matches if e.version == version]
            return [copy.deepcopy(e) for e in matches]

    def promote_and_activate(
        self,
        tenant_id: str,
        schema_id: str,
        version: int,
        approved_by: str,
        harness_run_id: str,
        event: LifecycleEvent,
        approval_record: Optional[SchemaApprovalRecord] = None,
    ) -> SchemaSnapshot:
        """
        Atomically promotes a schema version from TESTING to APPROVED, updates approved_by
        and harness_run_id on the snapshot, sets active production pointer, records lifecycle event,
        and persists approval record in a single atomic operation with rollback safety.
        """
        tid = self._validate_tenant(tenant_id)
        key = (tid, schema_id, version)

        with self._lock:
            existing = self._snapshots.get(key)
            if existing is None:
                raise SchemaNotFoundError(
                    f"Schema '{schema_id}' version {version} not found for tenant '{tid}'.",
                    schema_id=schema_id,
                    version=version,
                    tenant_id=tid,
                )

            if existing.status != SchemaStatus.TESTING:
                raise InvalidStateTransitionError(
                    message=(
                        f"Schema must be in 'testing' status before it can be approved, "
                        f"got current status '{existing.status.value}'."
                    ),
                    from_status=existing.status,
                    to_status=SchemaStatus.APPROVED,
                    schema_id=schema_id,
                    version=version,
                )

            # Check if harness_run_id was already consumed for another approval
            if harness_run_id:
                consumed_key = (tid, harness_run_id)
                if consumed_key in self._consumed_harness_runs:
                    prev_sid, prev_ver = self._consumed_harness_runs[consumed_key]
                    if (prev_sid, prev_ver) != (schema_id, version):
                        raise BenchmarkVerificationError(
                            message=f"Harness run '{harness_run_id}' has already been consumed to approve schema '{prev_sid}' v{prev_ver}.",
                            harness_run_id=harness_run_id,
                            schema_id=schema_id,
                            version=version,
                        )

            # Snapshot rollback state in memory before mutating
            orig_snapshot = self._snapshots[key]
            orig_pointer = self._active_pointers.get((tid, schema_id))
            orig_events_len = len(self._events)
            orig_records_len = len(self._approval_records)
            orig_consumed = dict(self._consumed_harness_runs)

            try:
                # Update snapshot with status, approved_by, and harness_run_id
                updated = existing.model_copy(
                    update={
                        "status": SchemaStatus.APPROVED,
                        "approved_by": approved_by,
                        "harness_run_id": harness_run_id,
                    }
                )
                self._snapshots[key] = copy.deepcopy(updated)
                self._active_pointers[(tid, schema_id)] = version
                self._events.append(copy.deepcopy(event))
                if approval_record is not None:
                    self._approval_records.append(copy.deepcopy(approval_record))
                if harness_run_id:
                    self._consumed_harness_runs[(tid, harness_run_id)] = (schema_id, version)

                return copy.deepcopy(updated)
            except Exception:
                # Rollback all state changes if any failure occurs
                self._snapshots[key] = orig_snapshot
                if orig_pointer is not None:
                    self._active_pointers[(tid, schema_id)] = orig_pointer
                elif (tid, schema_id) in self._active_pointers:
                    del self._active_pointers[(tid, schema_id)]
                del self._events[orig_events_len:]
                del self._approval_records[orig_records_len:]
                self._consumed_harness_runs = orig_consumed
                raise

    def get_approval_record(
        self,
        schema_id: str,
        version: int,
        tenant_id: str,
    ) -> Optional[SchemaApprovalRecord]:
        tid = self._validate_tenant(tenant_id)
        with self._lock:
            for rec in reversed(self._approval_records):
                if rec.tenant_id == tid and rec.schema_id == schema_id and rec.version == version:
                    return copy.deepcopy(rec)
            return None

    def list_approval_records(
        self,
        schema_id: str,
        tenant_id: str,
    ) -> List[SchemaApprovalRecord]:
        tid = self._validate_tenant(tenant_id)
        with self._lock:
            return [
                copy.deepcopy(rec)
                for rec in self._approval_records
                if rec.tenant_id == tid and rec.schema_id == schema_id
            ]

    def is_harness_run_consumed(
        self,
        harness_run_id: str,
        tenant_id: str,
    ) -> bool:
        tid = self._validate_tenant(tenant_id)
        with self._lock:
            return (tid, harness_run_id) in self._consumed_harness_runs
