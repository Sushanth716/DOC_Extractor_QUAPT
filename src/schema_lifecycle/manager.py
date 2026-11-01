"""
Schema Lifecycle Manager (SCH-004, SCH-006).
Orchestrates immutable versioning, state machine validation, rollback, and production binding.
Strictly adheres to PRD §6.1, §6.2, §7.1, SCH-004, and SCH-006.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
import uuid

from schema_lifecycle.diff import compute_schema_diff
from schema_lifecycle.enums import SchemaStatus
from schema_lifecycle.events import LifecycleEvent, LifecycleEventType
from schema_lifecycle.exceptions import (
    ForbiddenActionError,
    InvalidStateTransitionError,
    SchemaNotApprovedError,
    SchemaNotFoundError,
    SchemaPromotionNotAllowedError,
    SchemaRollbackError,
    SchemaSelfApprovalError,
    SchemaVersionError,
    TenantAccessError,
)
from schema_lifecycle.adapters import SchemaVersionAdapter
from schema_lifecycle.metamodel import SchemaDefinition
from schema_lifecycle.repository import (
    InMemorySchemaRepository,
    SchemaApprovalRecord,
    SchemaRepository,
    SchemaSnapshot,
)
from schema_lifecycle.state_machine import SchemaStateMachine


class SchemaLifecycleManager:
    """
    Core service coordinating schema lifecycle state transitions, immutable snapshot
    creation, non-destructive rollbacks, and production schema resolution.
    """

    def __init__(
        self,
        repository: Optional[SchemaRepository] = None,
        approval_service: Optional[Any] = None,
        strict_approval_gate: bool = False,
    ) -> None:
        self.repository = repository or InMemorySchemaRepository()
        self._approval_service: Optional[Any] = approval_service
        self._approval_token: Optional[str] = None
        self._strict_approval_gate: bool = strict_approval_gate or (approval_service is not None)

    def set_approval_service(self, service: Any, token: Optional[str] = None) -> None:
        """
        Binds the trusted SchemaApprovalService capability to lock down direct promotion bypass (SCH-005).
        Once bound, direct promotions via internal_promote_to_approved() without the trusted token are blocked.
        """
        self._approval_service = service
        self._approval_token = token or uuid.uuid4().hex
        self._strict_approval_gate = True

    def create_schema(
        self,
        tenant_id: str,
        definition: SchemaDefinition,
        author: str,
    ) -> SchemaSnapshot:
        """
        Creates the initial version (v1) snapshot of a schema for a given tenant.

        SCH-004: Newly created schemas start in 'draft' status by default.
        SCH-006: Preserves schema_id, tenant_id, canonical version, author, timestamp, diff.
        """
        if not author or not isinstance(author, str) or not author.strip():
            raise ValueError("Author identity must be a non-empty string.")

        # Eliminate approval bypass (SCH-004, SCH-005): Creating schema directly as approved is rejected
        if definition.status == SchemaStatus.APPROVED:
            raise SchemaPromotionNotAllowedError(
                message="Cannot create schema directly in 'approved' status. Schemas must start in 'draft' or 'testing' (SCH-004, SCH-005).",
                schema_id=definition.schema_id,
                version=definition.version,
            )

        # Check if schema already exists for this tenant
        existing = self.repository.get_latest_version(definition.schema_id, tenant_id)
        if existing is not None:
            raise SchemaVersionError(
                f"Schema '{definition.schema_id}' already exists for tenant '{tenant_id}'. "
                "Use create_version() to create an incremental version.",
                version=definition.version,
            )

        now = datetime.now(timezone.utc)
        diff = compute_schema_diff(None, definition)

        snapshot = SchemaSnapshot(
            schema_id=definition.schema_id,
            tenant_id=tenant_id,
            version=definition.version,
            status=definition.status,
            definition=definition,
            author=author.strip(),
            created_at=now,
            diff=diff,
            parent_version=None,
        )

        self.repository.save_snapshot(snapshot)

        self.repository.record_event(
            LifecycleEvent(
                tenant_id=tenant_id,
                schema_id=definition.schema_id,
                version=definition.version,
                event_type=LifecycleEventType.CREATED,
                actor=author.strip(),
                timestamp=now,
                from_status=None,
                to_status=definition.status,
                details={"diff_summary": diff["summary"]},
            )
        )

        return snapshot

    def create_version(
        self,
        tenant_id: str,
        schema_id: str,
        definition: SchemaDefinition,
        author: str,
        comment: Optional[str] = None,
    ) -> SchemaSnapshot:
        """
        Creates an immutable new integer version of an existing schema (SCH-006).
        Calculates a deterministic structural diff against the latest version.
        New versions start in 'draft' status per SCH-004.
        """
        if not author or not isinstance(author, str) or not author.strip():
            raise ValueError("Author identity must be a non-empty string.")

        latest = self.repository.get_latest_version(schema_id, tenant_id)
        if latest is None:
            raise SchemaNotFoundError(
                f"Cannot create new version: Schema '{schema_id}' does not exist for tenant '{tenant_id}'.",
                schema_id=schema_id,
                tenant_id=tenant_id,
            )

        if definition.schema_id != schema_id:
            raise ValueError(
                f"Definition schema_id '{definition.schema_id}' does not match target schema_id '{schema_id}'."
            )

        next_version = latest.version + 1
        now = datetime.now(timezone.utc)

        # Compute deterministic diff against latest definition
        diff = compute_schema_diff(latest.definition, definition)

        # Build versioned definition starting in DRAFT
        new_def_dict = definition.to_dict()
        new_def_dict["version"] = next_version
        new_def_dict["status"] = SchemaStatus.DRAFT.value
        new_def_dict["tenant_id"] = tenant_id
        versioned_def = SchemaDefinition.from_dict(new_def_dict)

        snapshot = SchemaSnapshot(
            schema_id=schema_id,
            tenant_id=tenant_id,
            version=next_version,
            status=SchemaStatus.DRAFT,
            definition=versioned_def,
            author=author.strip(),
            created_at=now,
            diff=diff,
            parent_version=latest.version,
        )

        self.repository.save_snapshot(snapshot)

        self.repository.record_event(
            LifecycleEvent(
                tenant_id=tenant_id,
                schema_id=schema_id,
                version=next_version,
                event_type=LifecycleEventType.CREATED,
                actor=author.strip(),
                timestamp=now,
                from_status=None,
                to_status=SchemaStatus.DRAFT,
                details={
                    "parent_version": latest.version,
                    "diff_summary": diff["summary"],
                    "comment": comment,
                },
            )
        )

        return snapshot

    def transition_status(
        self,
        tenant_id: str,
        schema_id: str,
        version: int,
        target_status: SchemaStatus,
        actor: str,
        reason: Optional[str] = None,
    ) -> SchemaSnapshot:
        """
        Validates and transitions a schema version to a new lifecycle status (SCH-004).
        Ordinary lifecycle operations CANNOT promote schemas to 'approved'.
        Direct promotion to 'approved' raises SchemaPromotionNotAllowedError.
        """
        if not actor or not isinstance(actor, str) or not actor.strip():
            raise ValueError("Actor identity must be a non-empty string.")

        snapshot = self.repository.get_snapshot(schema_id, version, tenant_id)
        if snapshot is None:
            raise SchemaNotFoundError(
                f"Schema '{schema_id}' version {version} not found for tenant '{tenant_id}'.",
                schema_id=schema_id,
                version=version,
                tenant_id=tenant_id,
            )

        # Validate through state machine (unconditionally blocks promotion to APPROVED)
        SchemaStateMachine.validate_transition(
            current_status=snapshot.status,
            target_status=target_status,
            schema_id=schema_id,
            version=version,
        )

        now = datetime.now(timezone.utc)
        updated = self.repository.update_status(schema_id, version, tenant_id, target_status)

        event_type = (
            LifecycleEventType.RETIRED
            if target_status == SchemaStatus.RETIRED
            else LifecycleEventType.STATUS_CHANGED
        )

        details: Dict[str, Any] = {}
        if reason:
            details["reason"] = reason

        self.repository.record_event(
            LifecycleEvent(
                tenant_id=tenant_id,
                schema_id=schema_id,
                version=version,
                event_type=event_type,
                actor=actor.strip(),
                timestamp=now,
                from_status=snapshot.status,
                to_status=target_status,
                details=details,
            )
        )

        return updated

    def internal_promote_to_approved(
        self,
        tenant_id: str,
        schema_id: str,
        version: int,
        approver: str,
        harness_run_id: Optional[str] = None,
        reason: Optional[str] = None,
        *,
        trusted_token: Optional[str] = None,
        event: Optional[LifecycleEvent] = None,
        approval_record: Optional[SchemaApprovalRecord] = None,
    ) -> SchemaSnapshot:
        """
        Controlled internal integration point.
        Promotes a schema version from 'testing' to 'approved' after four-eyes review
        and test-harness verification.
        Enforces four-eyes author exclusion (SCH-005, SEC-003).
        Atomically updates status, approved_by, harness_run_id, and active version pointer.
        Blocks direct promotion bypass when an approval gate is configured (Check 1).
        """
        # Block direct unauthorized bypass when approval gate is active
        if self._strict_approval_gate or self._approval_service is not None:
            if not trusted_token or trusted_token != self._approval_token:
                raise ForbiddenActionError(
                    message=(
                        "Direct unauthorized promotion via internal_promote_to_approved() is blocked. "
                        "Schema promotion must be authorized through SchemaApprovalService with "
                        "trusted principal authorization, MFA, and verified benchmark results (SCH-005, SEC-003)."
                    ),
                    principal_id=approver,
                    code="FORBIDDEN",
                )

        if not approver or not isinstance(approver, str) or not approver.strip():
            raise ValueError("Approver identity must be a non-empty string.")

        snapshot = self.repository.get_snapshot(schema_id, version, tenant_id)
        if snapshot is None:
            raise SchemaNotFoundError(
                f"Schema '{schema_id}' version {version} not found for tenant '{tenant_id}'.",
                schema_id=schema_id,
                version=version,
                tenant_id=tenant_id,
            )

        # Enforces that version must be in TESTING before approval
        SchemaStateMachine.validate_approval_transition(
            current_status=snapshot.status,
            schema_id=schema_id,
            version=version,
        )

        # SCH-005 / SEC-003 Four-Eyes author exclusion
        approver_id = approver.strip()
        if approver_id == snapshot.author:
            raise SchemaSelfApprovalError(
                message=(
                    f"User '{approver_id}' authored schema '{schema_id}' version {version} "
                    "and cannot approve it. A second user with the approver role must approve (SCH-005, SEC-003)."
                ),
                schema_id=schema_id,
                version=version,
                author=snapshot.author,
            )

        now = datetime.now(timezone.utc)
        resolved_harness_id = harness_run_id or f"internal_{uuid.uuid4().hex[:8]}"

        if event is None:
            details: Dict[str, Any] = {
                "author": snapshot.author,
                "approved_by": approver_id,
                "approver": approver_id,
                "harness_run_id": resolved_harness_id,
            }
            if reason:
                details["reason"] = reason

            event = LifecycleEvent(
                tenant_id=tenant_id,
                schema_id=schema_id,
                version=version,
                event_type=LifecycleEventType.APPROVED,
                actor=approver_id,
                timestamp=now,
                from_status=snapshot.status,
                to_status=SchemaStatus.APPROVED,
                details=details,
            )

        if approval_record is None:
            approval_record = SchemaApprovalRecord(
                tenant_id=tenant_id,
                schema_id=schema_id,
                version=version,
                author_id=snapshot.author,
                approver_id=approver_id,
                approval_timestamp=now,
                harness_run_id=resolved_harness_id,
                benchmark_policy_version="1.0.0",
                metric_summary={"source": "internal_promote_to_approved"},
                reason=reason,
                lifecycle_event_id=event.event_id,
            )

        # Atomic promotion and activation through repository
        updated = self.repository.promote_and_activate(
            tenant_id=tenant_id,
            schema_id=schema_id,
            version=version,
            approved_by=approver_id,
            harness_run_id=resolved_harness_id,
            event=event,
            approval_record=approval_record,
        )

        return updated

    def rollback(
        self,
        tenant_id: str,
        schema_id: str,
        target_version: int,
        actor: str,
        reason: Optional[str] = None,
    ) -> SchemaSnapshot:
        """
        Non-destructive rollback to a previously approved schema version (SCH-006).
        Re-activates the target version for production without deleting or mutating history.
        Reconciles terminal RETIRED status by validating rollback re-activation through the state machine.
        """
        if not actor or not isinstance(actor, str) or not actor.strip():
            raise ValueError("Actor identity must be a non-empty string.")

        target = self.repository.get_snapshot(schema_id, target_version, tenant_id)
        if target is None:
            raise SchemaNotFoundError(
                f"Rollback target schema '{schema_id}' version {target_version} not found for tenant '{tenant_id}'.",
                schema_id=schema_id,
                version=target_version,
                tenant_id=tenant_id,
            )

        # Verify that target_version was EVER approved (SCH-006)
        events = self.repository.list_events(schema_id, tenant_id, version=target_version)
        was_approved = target.status == SchemaStatus.APPROVED or any(
            e.to_status == SchemaStatus.APPROVED or e.event_type == LifecycleEventType.APPROVED
            for e in events
        )

        if not was_approved:
            raise SchemaRollbackError(
                f"Cannot rollback to version {target_version}: schema version was never approved in its lifecycle.",
                schema_id=schema_id,
                target_version=target_version,
            )

        # Explicitly validate rollback transition through state machine (reconciling terminal RETIRED status)
        SchemaStateMachine.validate_rollback_transition(
            current_status=target.status,
            schema_id=schema_id,
            version=target_version,
        )

        current_active = self.repository.get_active_version(schema_id, tenant_id)
        prev_active_version = current_active.version if current_active else None

        # Re-activate target version as approved if it was retired
        if target.status != SchemaStatus.APPROVED:
            target = self.repository.update_status(schema_id, target_version, tenant_id, SchemaStatus.APPROVED)

        # Update active pointer to target version
        self.repository.set_active_version(schema_id, target_version, tenant_id)

        now = datetime.now(timezone.utc)
        self.repository.record_event(
            LifecycleEvent(
                tenant_id=tenant_id,
                schema_id=schema_id,
                version=target_version,
                event_type=LifecycleEventType.ROLLBACK,
                actor=actor.strip(),
                timestamp=now,
                from_status=target.status,
                to_status=SchemaStatus.APPROVED,
                details={
                    "target_version": target_version,
                    "previous_active_version": prev_active_version,
                    "reason": reason or f"Rollback from v{prev_active_version} to v{target_version}",
                },
            )
        )

        return target

    def resolve_active_schema(
        self,
        tenant_id: str,
        schema_id: str,
    ) -> SchemaSnapshot:
        """
        Resolves the currently active approved schema version for production use (SCH-004).

        Raises:
            SchemaNotApprovedError: If no approved schema is active for this tenant/schema_id.
        """
        active = self.repository.get_active_version(schema_id, tenant_id)
        if active is None:
            raise SchemaNotApprovedError(
                message=f"No approved schema active for schema_id='{schema_id}' in tenant '{tenant_id}'.",
                schema_id=schema_id,
            )

        if active.status != SchemaStatus.APPROVED:
            raise SchemaNotApprovedError(
                message=(
                    f"Active schema '{schema_id}' version {active.version} is in status '{active.status.value}', "
                    "not approved for production."
                ),
                schema_id=schema_id,
                version=active.version,
            )

        return active

    def bind_production_schema(
        self,
        tenant_id: str,
        schema_id: str,
        version: Optional[int] = None,
    ) -> SchemaSnapshot:
        """
        Binds a schema for production extraction (SCH-004).
        If version is specified, verifies it is in APPROVED status.
        If version is None, resolves the currently active approved version.

        Raises:
            SchemaNotApprovedError: If the specified or resolved schema is not approved.
        """
        if version is None:
            return self.resolve_active_schema(tenant_id, schema_id)

        snapshot = self.repository.get_snapshot(schema_id, version, tenant_id)
        if snapshot is None:
            raise SchemaNotFoundError(
                f"Schema '{schema_id}' version {version} not found for tenant '{tenant_id}'.",
                schema_id=schema_id,
                version=version,
                tenant_id=tenant_id,
            )

        if snapshot.status != SchemaStatus.APPROVED:
            raise SchemaNotApprovedError(
                message=(
                    f"Attempted to bind schema '{schema_id}' version {version} with status "
                    f"'{snapshot.status.value}' in production. Only 'approved' schemas can be bound (SCH-004)."
                ),
                schema_id=schema_id,
                version=version,
            )

        return snapshot

    def bind_test_schema(
        self,
        tenant_id: str,
        schema_id: str,
        version: Union[int, str],
    ) -> SchemaSnapshot:
        """
        Binds an exact schema version strictly for evaluation harness execution (SCH-007, US-DOCX-010).

        Requirements:
        1. Resolve using exact tenant_id, schema_id, and canonical integer version.
        2. Permit harness binding ONLY when status is SchemaStatus.TESTING.
        3. Reject DRAFT, RETIRED, APPROVED, missing and cross-tenant schemas with appropriate existing exceptions:
           - Missing / cross-tenant -> SchemaNotFoundError
           - Blank tenant -> TenantAccessError
           - Non-TESTING status -> InvalidStateTransitionError
        4. Return the exact immutable SchemaSnapshot.
        5. Never modify the active production version pointer.
        6. Never change schema lifecycle status during binding.
        7. Keep bind_production_schema() and resolve_active_schema() behavior unchanged.
        """
        if not tenant_id or not isinstance(tenant_id, str) or not tenant_id.strip():
            raise TenantAccessError("Tenant identifier must be a non-empty string for test schema binding.")

        if not schema_id or not isinstance(schema_id, str) or not schema_id.strip():
            raise ValueError("schema_id must be a non-empty string.")

        canonical_version = SchemaVersionAdapter.to_canonical(version)

        snapshot = self.repository.get_snapshot(schema_id.strip(), canonical_version, tenant_id.strip())
        if snapshot is None:
            raise SchemaNotFoundError(
                f"Schema '{schema_id}' version {canonical_version} not found for tenant '{tenant_id}'.",
                schema_id=schema_id,
                version=canonical_version,
                tenant_id=tenant_id,
            )

        if snapshot.status != SchemaStatus.TESTING:
            raise InvalidStateTransitionError(
                message=(
                    f"Attempted to bind schema '{schema_id}' version {canonical_version} with status "
                    f"'{snapshot.status.value}' for test harness execution. Only schemas in 'testing' "
                    "status may be bound for evaluation (SCH-007, US-DOCX-010)."
                ),
                from_status=snapshot.status,
                to_status=SchemaStatus.TESTING,
                schema_id=schema_id,
                version=canonical_version,
                code="INVALID_STATE_TRANSITION",
                details={
                    "status": snapshot.status.value,
                    "allowed_status": SchemaStatus.TESTING.value,
                },
            )

        return snapshot

    def get_version_history(
        self,
        tenant_id: str,
        schema_id: str,
    ) -> List[SchemaSnapshot]:
        """Lists complete version history for a schema in a tenant (SCH-006)."""
        return self.repository.list_versions(schema_id, tenant_id)

    def get_audit_events(
        self,
        tenant_id: str,
        schema_id: str,
        version: Optional[int] = None,
    ) -> List[LifecycleEvent]:
        """Lists append-only lifecycle and rollback audit events for a schema (SCH-006)."""
        return self.repository.list_events(schema_id, tenant_id, version=version)
