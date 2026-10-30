"""
Schema Repository Abstraction and In-Memory Implementation (SCH-004, SCH-006).
Provides tenant-isolated, immutable storage for schema snapshots and append-only lifecycle events.
Enforces deep immutability through defensive copies at all repository boundaries.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import copy
from datetime import datetime
import threading
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, model_validator

from schema_lifecycle.enums import SchemaStatus
from schema_lifecycle.events import LifecycleEvent
from schema_lifecycle.exceptions import (
    SchemaNotFoundError,
    SchemaVersionError,
    TenantAccessError,
)
from schema_lifecycle.metamodel import SchemaDefinition


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
