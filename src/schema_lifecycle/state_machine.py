"""
Schema Lifecycle State Machine (SCH-004, SCH-006).
Enforces valid and invalid lifecycle transitions among draft, testing, approved, and retired.
Strictly adheres to PRD §6.1, §6.2, SCH-004, and SCH-005.

Ambiguity Documentation:
1. Approval Boundary & Direct Promotion:
   PRD SCH-004 specifies states 'draft', 'testing', 'approved', 'retired'.
   PRD SCH-005 and TASK-4.1 constraints dictate that ordinary lifecycle operations
   CANNOT promote schemas to 'approved'. Direct promotion via standard lifecycle methods
   unconditionally raises SchemaPromotionNotAllowedError.
   A dedicated internal integration point (validate_approval_transition) is reserved
   exclusively for the TASK-4.2 approval service (enforcing the prerequisite that the
   schema must be in 'testing' status).

2. Reconciling Terminal 'retired' Status with Rollback:
   PRD SCH-004 defines 'retired' as a lifecycle state, while SCH-006 dictates that rollback
   re-activates a prior version without deleting history.
   The PRD is ambiguous on whether a superseded approved version transitions to 'retired'
   or remains in 'approved' status while inactive. If an approved version was retired,
   treating 'retired' as strictly immutable without re-activation would prevent rollback.
   The state machine explicitly reconciles this ambiguity via validate_rollback_transition():
   during an authorized rollback of a previously approved version, re-activating from
   RETIRED -> APPROVED is validated and permitted, while arbitrary transitions out of RETIRED
   via standard lifecycle operations remain strictly forbidden.

3. Abandoning Draft or Testing Versions:
   PRD does not explicitly specify whether drafts can transition directly to 'retired'.
   The state machine permits DRAFT -> RETIRED and TESTING -> RETIRED as documented proposed
   transitions to allow authors to discard obsolete or flawed drafts cleanly without leaving them open.
"""

from __future__ import annotations

from typing import Dict, Optional, Set

from schema_lifecycle.enums import SchemaStatus
from schema_lifecycle.exceptions import (
    InvalidStateTransitionError,
    SchemaPromotionNotAllowedError,
)


class SchemaStateMachine:
    """
    Validates lifecycle state transitions for schema definitions.
    """

    # Permitted standard transition paths
    _VALID_TRANSITIONS: Dict[SchemaStatus, Set[SchemaStatus]] = {
        SchemaStatus.DRAFT: {
            SchemaStatus.TESTING,
            SchemaStatus.RETIRED,
        },
        SchemaStatus.TESTING: {
            SchemaStatus.DRAFT,
            SchemaStatus.APPROVED,
            SchemaStatus.RETIRED,
        },
        SchemaStatus.APPROVED: {
            SchemaStatus.RETIRED,
        },
        SchemaStatus.RETIRED: set(),  # Terminal state for standard transitions
    }

    @classmethod
    def get_allowed_transitions(cls, current_status: SchemaStatus) -> Set[SchemaStatus]:
        """Returns the set of permitted next statuses from current_status."""
        return set(cls._VALID_TRANSITIONS.get(current_status, set()))

    @classmethod
    def is_valid_transition(
        cls,
        current_status: SchemaStatus,
        target_status: SchemaStatus,
    ) -> bool:
        """Checks if a transition between two statuses is structurally valid."""
        allowed = cls._VALID_TRANSITIONS.get(current_status, set())
        return target_status in allowed

    @classmethod
    def validate_transition(
        cls,
        current_status: SchemaStatus,
        target_status: SchemaStatus,
        schema_id: Optional[str] = None,
        version: Optional[int] = None,
    ) -> None:
        """
        Validates transition from current_status to target_status.
        Ordinary lifecycle callers cannot promote schemas to APPROVED.

        Raises:
            InvalidStateTransitionError: If the transition is invalid or to self.
            SchemaPromotionNotAllowedError: If transition to APPROVED is attempted.
        """
        if current_status == target_status:
            raise InvalidStateTransitionError(
                message=f"Schema is already in status '{current_status.value}'. Self-transition is invalid.",
                from_status=current_status,
                to_status=target_status,
                schema_id=schema_id,
                version=version,
            )

        # Standard lifecycle operations can NEVER directly promote to APPROVED
        if target_status == SchemaStatus.APPROVED:
            raise SchemaPromotionNotAllowedError(
                message=(
                    "Direct promotion to 'approved' is not allowed through standard lifecycle transition. "
                    "Promotion is reserved for the TASK-4.2 approval service requiring test harness pass and four-eyes review."
                ),
                schema_id=schema_id,
                version=version,
                from_status=current_status,
            )

        allowed = cls._VALID_TRANSITIONS.get(current_status, set())
        if target_status not in allowed:
            raise InvalidStateTransitionError(
                message=(
                    f"Invalid lifecycle transition from '{current_status.value}' to '{target_status.value}'. "
                    f"Allowed transitions from '{current_status.value}': "
                    f"{[s.value for s in sorted(allowed, key=lambda s: s.value)] or 'none (terminal state)'}."
                ),
                from_status=current_status,
                to_status=target_status,
                schema_id=schema_id,
                version=version,
            )

    @classmethod
    def validate_approval_transition(
        cls,
        current_status: SchemaStatus,
        schema_id: Optional[str] = None,
        version: Optional[int] = None,
    ) -> None:
        """
        Controlled internal integration point reserved for TASK-4.2 approval service.
        Promotion to APPROVED is only valid from TESTING status.

        Raises:
            InvalidStateTransitionError: If current_status is not TESTING.
        """
        if current_status != SchemaStatus.TESTING:
            raise InvalidStateTransitionError(
                message=(
                    f"Schema must be in 'testing' status before it can be approved, "
                    f"got current status '{current_status.value}'."
                ),
                from_status=current_status,
                to_status=SchemaStatus.APPROVED,
                schema_id=schema_id,
                version=version,
            )

    @classmethod
    def validate_rollback_transition(
        cls,
        current_status: SchemaStatus,
        schema_id: Optional[str] = None,
        version: Optional[int] = None,
    ) -> None:
        """
        Validates transition during rollback to an approved version.
        Reconciles terminal RETIRED status: re-activating an approved version that was
        subsequently retired is permitted during authorized rollback.

        Raises:
            InvalidStateTransitionError: If the rollback target is in DRAFT or TESTING.
        """
        if current_status in (SchemaStatus.DRAFT, SchemaStatus.TESTING):
            raise InvalidStateTransitionError(
                message=(
                    f"Cannot rollback to a schema version in '{current_status.value}' status. "
                    "Only previously approved versions can be restored."
                ),
                from_status=current_status,
                to_status=SchemaStatus.APPROVED,
                schema_id=schema_id,
                version=version,
            )
