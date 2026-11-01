"""
Four-Eyes Approval Gate Service and Trusted Benchmark Verification (SCH-005, SEC-003).
Strictly adheres to PRD v1.1 §10.1, §13.1..13.3, §15.5, SCH-005, and SEC-003:
- Four-Eyes Approval: author cannot approve own schema version.
- Requires passing test-harness evaluation retrieved from an authoritative BenchmarkReportStore.
- Verifies exact tenant, schema ID, canonical integer version, and provenance integrity.
- Rejects mock, heuristic, synthetic-only, and failed-sample benchmark reports.
- Distinguishes schema approval eligibility from G1/GR-005 release qualification via explicit ApprovalPolicy.
- Enforces SEC-003 role-based authorization, least privilege, and SEC-001 MFA verification.
- Executes atomic, rollback-safe repository promotion and single-use harness run consumption.
"""

from __future__ import annotations

import copy
from datetime import datetime, timezone
import threading
from typing import Any, Dict, List, Optional, Protocol, Tuple, TYPE_CHECKING
import uuid

from pydantic import BaseModel, ConfigDict, Field

from doc_extractor.contracts_c import ExecutionMode
from schema_lifecycle.enums import SchemaStatus
from schema_lifecycle.events import LifecycleEvent, LifecycleEventType
from schema_lifecycle.exceptions import (
    BenchmarkVerificationError,
    ForbiddenActionError,
    InvalidStateTransitionError,
    SchemaNotFoundError,
    SchemaSelfApprovalError,
    TenantAccessError,
    UnauthenticatedError,
)
from schema_lifecycle.manager import SchemaLifecycleManager
from schema_lifecycle.repository import (
    SchemaApprovalRecord,
    SchemaRepository,
    SchemaSnapshot,
)
from schema_lifecycle.roles import (
    AuthenticatedPrincipal,
    Role,
    SchemaAuthorizer,
    SchemaPermission,
)
from schema_lifecycle.state_machine import SchemaStateMachine

if TYPE_CHECKING:
    from schema_lifecycle.testing.contracts import DatasetSplit
    from schema_lifecycle.testing.runner import BenchmarkReport



# ============================================================================
# Approval Policy (Distinguishing Schema Approval vs. Release Qualification)
# ============================================================================

class ApprovalPolicy(BaseModel):
    """
    Configurable policy specifying benchmark requirements for schema version approval (SCH-005).
    Separates schema-level approval eligibility from product-level G1 release qualification
    and GR-005 1,000-run longevity qualification.
    """
    model_config = ConfigDict(frozen=True, extra="forbid")

    policy_name: str = "default_schema_approval_policy"
    require_locked_test: bool = False  # Schema approval vs release qualification
    require_gate_eligible: bool = False  # Gate G1 is separate from schema approval
    require_gr005_conformance: bool = False  # GR-005 1,000-run is longevity/release qualification
    min_f1_all_fields: float = 0.92  # PRD M1 threshold
    min_f1_key_fields: float = 0.95  # PRD M1 threshold
    max_calibration_gap_points: float = 3.0  # PRD M6 threshold
    max_report_age_seconds: Optional[int] = None  # Optional expiration window


# ============================================================================
# Authoritative Benchmark Report Store Protocol & In-Memory Implementation
# ============================================================================

class BenchmarkReportStore(Protocol):
    """Protocol for retrieving and tracking consumption of authoritative BenchmarkReports."""

    def get_report(self, harness_run_id: str) -> Optional[BenchmarkReport]:
        """Retrieves an authoritative benchmark report by its run ID."""
        ...

    def is_consumed(self, harness_run_id: str) -> bool:
        """Checks if a harness run ID has already been consumed to approve a schema."""
        ...

    def mark_consumed(
        self,
        harness_run_id: str,
        tenant_id: str,
        schema_id: str,
        version: int,
    ) -> None:
        """Marks a harness run ID as consumed for a specific schema approval."""
        ...

    def unmark_consumed(self, harness_run_id: str) -> None:
        """Unmarks a consumed harness run ID when promotion transaction aborts."""
        ...

    def list_reports_for_schema(
        self,
        tenant_id: str,
        schema_id: str,
        version: int,
    ) -> List[BenchmarkReport]:
        """Lists all reports associated with a specific tenant, schema ID, and version."""
        ...

    def get_latest_passing_report(
        self,
        tenant_id: str,
        schema_id: str,
        version: int,
    ) -> Optional[BenchmarkReport]:
        """Retrieves the latest passing benchmark report for a specific schema version (US-DOCX-010)."""
        ...


class InMemoryBenchmarkReportStore:
    """
    Thread-safe in-memory store for benchmark reports and single-use consumption tracking.
    Suitable for development, testing, and isolated execution.
    """

    def __init__(self, reports: Optional[Dict[str, BenchmarkReport]] = None) -> None:
        self._lock = threading.RLock()
        self._reports: Dict[str, BenchmarkReport] = {}
        self._consumed: Dict[str, Dict[str, Any]] = {}
        if reports:
            for rep in reports.values():
                self.save_report(rep)

    def save_report(self, report: BenchmarkReport) -> None:
        """Stores an authoritative benchmark report."""
        with self._lock:
            self._reports[report.benchmark_id] = copy.deepcopy(report)

    def get_report(self, harness_run_id: str) -> Optional[BenchmarkReport]:
        with self._lock:
            rep = self._reports.get(harness_run_id)
            return copy.deepcopy(rep) if rep is not None else None

    def is_consumed(self, harness_run_id: str) -> bool:
        with self._lock:
            return harness_run_id in self._consumed

    def mark_consumed(
        self,
        harness_run_id: str,
        tenant_id: str,
        schema_id: str,
        version: int,
    ) -> None:
        with self._lock:
            if harness_run_id in self._consumed:
                prev = self._consumed[harness_run_id]
                raise BenchmarkVerificationError(
                    f"Harness run ID '{harness_run_id}' has already been consumed to approve schema '{prev['schema_id']}' v{prev['version']}.",
                    harness_run_id=harness_run_id,
                    schema_id=schema_id,
                    version=version,
                )
            self._consumed[harness_run_id] = {
                "tenant_id": tenant_id,
                "schema_id": schema_id,
                "version": version,
                "consumed_at": datetime.now(timezone.utc).isoformat(),
            }

    def unmark_consumed(self, harness_run_id: str) -> None:
        """Safely unmarks a consumed harness run ID to allow retry after promotion failure."""
        with self._lock:
            self._consumed.pop(harness_run_id, None)

    def list_reports_for_schema(
        self,
        tenant_id: str,
        schema_id: str,
        version: int,
    ) -> List[BenchmarkReport]:
        """Lists all reports associated with a specific tenant, schema ID, and version."""
        with self._lock:
            matches = [
                copy.deepcopy(r)
                for r in self._reports.values()
                if r.tenant_id == tenant_id
                and r.schema_id == schema_id
                and r.schema_version == version
            ]
            return sorted(matches, key=lambda r: str(getattr(r, "run_timestamp", "")))

    def get_latest_passing_report(
        self,
        tenant_id: str,
        schema_id: str,
        version: int,
    ) -> Optional[BenchmarkReport]:
        """Retrieves the latest passing benchmark report for a specific schema version (US-DOCX-010)."""
        with self._lock:
            passing = [
                copy.deepcopy(r)
                for r in self._reports.values()
                if r.tenant_id == tenant_id
                and r.schema_id == schema_id
                and r.schema_version == version
                and getattr(getattr(r, "metrics_decision", None), "overall_pass", False)
                and not self.is_consumed(r.benchmark_id)
            ]
            if not passing:
                return None
            return max(passing, key=lambda r: str(getattr(r, "run_timestamp", "")))


# ============================================================================
# Benchmark Result Verifier
# ============================================================================

class BenchmarkResultVerifier:
    """
    Authoritative verifier for benchmark evaluation results (SCH-005, SCH-007).
    Validates BenchmarkReport retrieved from a trusted BenchmarkReportStore.
    Does NOT duplicate evaluation or recalculate metrics.
    """

    def __init__(
        self,
        report_store: BenchmarkReportStore,
        policy: Optional[ApprovalPolicy] = None,
    ) -> None:
        self.report_store = report_store
        self.policy = policy or ApprovalPolicy()

    def verify_for_approval(
        self,
        harness_run_id: str,
        tenant_id: str,
        schema_id: str,
        version: int,
    ) -> BenchmarkReport:
        """
        Verifies that a benchmark report qualifies the schema version for approval.
        Validates context binding, metrics pass, provenance integrity, and policy compliance.
        """
        # 1. Retrieve authoritative report from store
        report = self.report_store.get_report(harness_run_id)
        if report is None:
            raise BenchmarkVerificationError(
                f"Harness result '{harness_run_id}' not found in trusted report store.",
                harness_run_id=harness_run_id,
                schema_id=schema_id,
                version=version,
            )

        # 2. Anti-replay check
        if self.report_store.is_consumed(harness_run_id):
            raise BenchmarkVerificationError(
                f"Harness result '{harness_run_id}' has already been consumed to approve another schema version.",
                harness_run_id=harness_run_id,
                schema_id=schema_id,
                version=version,
            )

        # 3. Exact tenant matching
        if report.tenant_id != tenant_id:
            raise BenchmarkVerificationError(
                f"Report tenant '{report.tenant_id}' does not match schema tenant '{tenant_id}'.",
                harness_run_id=harness_run_id,
                schema_id=schema_id,
                version=version,
            )

        # 4. Exact schema ID matching
        if report.schema_id != schema_id:
            raise BenchmarkVerificationError(
                f"Report schema ID '{report.schema_id}' does not match target schema ID '{schema_id}'.",
                harness_run_id=harness_run_id,
                schema_id=schema_id,
                version=version,
            )

        # 5. Exact canonical schema version matching
        if report.schema_version != version:
            raise BenchmarkVerificationError(
                f"Report schema version {report.schema_version} does not match target version {version}.",
                harness_run_id=harness_run_id,
                schema_id=schema_id,
                version=version,
            )

        # 5b. US-DOCX-010: Latest passing harness result enforcement
        # If a newer benchmark run exists for the same schema version, an older run cannot be used.
        if hasattr(self.report_store, "list_reports_for_schema"):
            schema_reports = self.report_store.list_reports_for_schema(tenant_id, schema_id, version)
            newer_runs = [
                r for r in schema_reports
                if r.benchmark_id != harness_run_id
                and str(getattr(r, "run_timestamp", "")) > str(getattr(report, "run_timestamp", ""))
            ]
            if newer_runs:
                latest_newer = max(newer_runs, key=lambda r: str(getattr(r, "run_timestamp", "")))
                is_newer_passing = getattr(getattr(latest_newer, "metrics_decision", None), "overall_pass", False)
                if is_newer_passing:
                    raise BenchmarkVerificationError(
                        f"Outdated harness result '{harness_run_id}' rejected. "
                        f"A newer passing benchmark run '{latest_newer.benchmark_id}' (timestamp {latest_newer.run_timestamp}) "
                        f"exists for schema '{schema_id}' v{version}. "
                        "US-DOCX-010 requires attaching the latest passing harness result.",
                        harness_run_id=harness_run_id,
                        schema_id=schema_id,
                        version=version,
                    )
                else:
                    raise BenchmarkVerificationError(
                        f"Superseded harness result '{harness_run_id}' rejected. "
                        f"A subsequent benchmark run '{latest_newer.benchmark_id}' (timestamp {latest_newer.run_timestamp}) "
                        f"for schema '{schema_id}' v{version} failed. Cannot approve schema with an older run.",
                        harness_run_id=harness_run_id,
                        schema_id=schema_id,
                        version=version,
                    )

        # 6. Overall metrics decision pass
        if not report.metrics_decision.overall_pass:
            raise BenchmarkVerificationError(
                f"Benchmark evaluation metrics decision did not pass: {report.metrics_decision.violations}",
                harness_run_id=harness_run_id,
                schema_id=schema_id,
                version=version,
            )

        # 7. Required M1 and M6 metric criteria
        if not report.metrics_decision.meets_m1_all_fields:
            raise BenchmarkVerificationError(
                "M1 metric requirement failed: all-fields F1 below threshold.",
                harness_run_id=harness_run_id,
                schema_id=schema_id,
                version=version,
            )
        if not report.metrics_decision.meets_m1_key_fields:
            raise BenchmarkVerificationError(
                "M1 metric requirement failed: key-fields F1 below threshold.",
                harness_run_id=harness_run_id,
                schema_id=schema_id,
                version=version,
            )
        if not report.metrics_decision.meets_m6_calibration:
            raise BenchmarkVerificationError(
                "M6 metric requirement failed: calibration gap exceeds threshold.",
                harness_run_id=harness_run_id,
                schema_id=schema_id,
                version=version,
            )

        # 8. Insufficient data status check
        if report.metrics_decision.has_insufficient_data or report.aggregate_metrics.has_insufficient_data:
            raise BenchmarkVerificationError(
                f"Benchmark evaluation has insufficient data: {report.metrics_decision.insufficient_data_reasons}",
                harness_run_id=harness_run_id,
                schema_id=schema_id,
                version=version,
            )

        # 9. Genuine LLM inference provenance (reject mock and heuristic)
        if report.is_mock_run or ExecutionMode.MOCK.value in report.execution_modes:
            raise BenchmarkVerificationError(
                "Mock execution rejected: mock provenance cannot qualify a schema for approval.",
                harness_run_id=harness_run_id,
                schema_id=schema_id,
                version=version,
            )
        if (
            ExecutionMode.HEURISTIC.value in report.execution_modes
            or ExecutionMode.HEURISTIC_FALLBACK.value in report.execution_modes
        ):
            raise BenchmarkVerificationError(
                "Heuristic execution rejected: untrusted provenance cannot qualify a schema for approval.",
                harness_run_id=harness_run_id,
                schema_id=schema_id,
                version=version,
            )

        # 10. Synthetic-only and failed samples rejection
        if report.real_samples <= 0:
            raise BenchmarkVerificationError(
                "Synthetic-only benchmark report cannot qualify a schema for approval (real_samples=0).",
                harness_run_id=harness_run_id,
                schema_id=schema_id,
                version=version,
            )
        if report.failed_samples > 0:
            raise BenchmarkVerificationError(
                f"Benchmark contains {report.failed_samples} failed sample(s). All evaluated samples must succeed.",
                harness_run_id=harness_run_id,
                schema_id=schema_id,
                version=version,
            )

        # 11. ApprovalPolicy criteria
        report_split_str = str(getattr(report.dataset_split, "value", report.dataset_split))
        if self.policy.require_locked_test and report_split_str != "locked_test":
            raise BenchmarkVerificationError(
                f"Approval policy requires locked_test dataset split, got '{report_split_str}'.",
                harness_run_id=harness_run_id,
                schema_id=schema_id,
                version=version,
            )
        if self.policy.require_gate_eligible and not report.gate_eligible:
            raise BenchmarkVerificationError(
                f"Approval policy requires gate_eligible=True: {report.gate_eligibility_reasons}",
                harness_run_id=harness_run_id,
                schema_id=schema_id,
                version=version,
            )
        if self.policy.require_gr005_conformance and report.gr005_accounting.status != "VERIFIED":
            raise BenchmarkVerificationError(
                f"Approval policy requires GR-005 VERIFIED status, got '{report.gr005_accounting.status}'.",
                harness_run_id=harness_run_id,
                schema_id=schema_id,
                version=version,
            )

        # 12. Optional report expiration check
        if self.policy.max_report_age_seconds is not None:
            try:
                run_dt = datetime.fromisoformat(report.run_timestamp)
                now = datetime.now(timezone.utc)
                age_seconds = (now - run_dt).total_seconds()
                if age_seconds > self.policy.max_report_age_seconds:
                    raise BenchmarkVerificationError(
                        f"Benchmark report expired: age {age_seconds:.0f}s exceeds maximum allowed {self.policy.max_report_age_seconds}s.",
                        harness_run_id=harness_run_id,
                        schema_id=schema_id,
                        version=version,
                    )
            except (ValueError, TypeError) as exc:
                raise BenchmarkVerificationError(
                    f"Malformed report timestamp '{report.run_timestamp}': {exc}",
                    harness_run_id=harness_run_id,
                    schema_id=schema_id,
                    version=version,
                ) from exc

        return report


# ============================================================================
# Four-Eyes Schema Approval Service
# ============================================================================

class SchemaApprovalService:
    """
    Four-Eyes Approval Gate Service (SCH-005, SEC-003).
    Coordinates:
    - Principal authentication & tenant isolation.
    - SEC-003 role & permission verification.
    - SCH-005 author exclusion (separation of duties).
    - Lifecycle state machine precondition (must be in TESTING).
    - Authoritative benchmark result verification.
    - Single-use anti-replay tracking.
    - Atomic repository promotion and active version activation.
    """

    def __init__(
        self,
        manager: SchemaLifecycleManager,
        authorizer: SchemaAuthorizer,
        verifier: BenchmarkResultVerifier,
        report_store: BenchmarkReportStore,
    ) -> None:
        self.manager = manager
        self.repository: SchemaRepository = manager.repository
        self.authorizer = authorizer
        self.verifier = verifier
        self.report_store = report_store
        self._approval_token = f"tok_{uuid.uuid4().hex}"
        # Lock down manager to prevent direct promotion bypass (Check 1)
        if hasattr(self.manager, "set_approval_service"):
            self.manager.set_approval_service(self, self._approval_token)

    def approve_schema(
        self,
        principal: AuthenticatedPrincipal,
        tenant_id: str,
        schema_id: str,
        version: int,
        harness_run_id: str,
        reason: Optional[str] = None,
    ) -> Tuple[SchemaSnapshot, SchemaApprovalRecord]:
        """
        Executes four-eyes schema approval promoting a schema version from TESTING to APPROVED.

        Raises:
            UnauthenticatedError: If principal is unauthenticated or inactive.
            TenantAccessError / ForbiddenActionError: If tenant mismatch or lacking permission.
            SchemaNotFoundError: If schema version does not exist.
            InvalidStateTransitionError: If schema version is not in TESTING status.
            SchemaSelfApprovalError: If principal authored the schema version (SCH-005).
            BenchmarkVerificationError: If harness report is missing, invalid, or failed.
        """
        # 1. Principal authentication and active status check
        if not principal or not principal.is_active:
            raise UnauthenticatedError(
                f"Principal '{getattr(principal, 'principal_id', 'unknown')}' is not authenticated or inactive."
            )

        # 2. Tenant isolation check (SEC-006)
        if principal.tenant_id != tenant_id:
            raise ForbiddenActionError(
                f"Cross-tenant access violation: principal tenant '{principal.tenant_id}' does not match target '{tenant_id}'.",
                principal_id=principal.principal_id,
                permission=SchemaPermission.SCHEMA_APPROVE.value,
            )

        # 3. Retrieve current schema snapshot
        snapshot = self.repository.get_snapshot(schema_id, version, tenant_id)
        if snapshot is None:
            raise SchemaNotFoundError(
                f"Schema '{schema_id}' version {version} not found for tenant '{tenant_id}'.",
                schema_id=schema_id,
                version=version,
                tenant_id=tenant_id,
            )

        # 4. Enforce state machine precondition: must be in TESTING status (SCH-004)
        if snapshot.status != SchemaStatus.TESTING:
            raise InvalidStateTransitionError(
                message=(
                    f"Schema must be in 'testing' status before it can be approved, "
                    f"got current status '{snapshot.status.value}'."
                ),
                from_status=snapshot.status,
                to_status=SchemaStatus.APPROVED,
                schema_id=schema_id,
                version=version,
            )

        # 5. Separation of Duties (Four-Eyes Gate, SCH-005, SEC-003):
        # The schema author CANNOT approve their own version.
        if principal.principal_id == snapshot.author:
            raise SchemaSelfApprovalError(
                message=(
                    f"User '{principal.principal_id}' authored schema '{schema_id}' version {version} "
                    "and cannot approve it. A second user with the Schema approver role must approve (SCH-005, SEC-003)."
                ),
                schema_id=schema_id,
                version=version,
                author=snapshot.author,
            )

        # 6. Authorize approver principal via SchemaAuthorizer
        auth_decision = self.authorizer.authorize(
            principal=principal,
            permission=SchemaPermission.SCHEMA_APPROVE,
            tenant_id=tenant_id,
            schema_author=snapshot.author,
        )
        if not auth_decision.is_authorized:
            if auth_decision.code == "SCHEMA_SELF_APPROVAL":
                raise SchemaSelfApprovalError(
                    message=auth_decision.reason or "Self-approval is forbidden.",
                    schema_id=schema_id,
                    version=version,
                    author=snapshot.author,
                )
            if auth_decision.code == "UNAUTHENTICATED":
                raise UnauthenticatedError(auth_decision.reason or "Authentication required.")
            raise ForbiddenActionError(
                message=f"Schema approval forbidden: {auth_decision.reason}",
                principal_id=principal.principal_id,
                permission=SchemaPermission.SCHEMA_APPROVE.value,
            )

        # 7. Authoritatively verify test-harness benchmark report
        report = self.verifier.verify_for_approval(
            harness_run_id=harness_run_id,
            tenant_id=tenant_id,
            schema_id=schema_id,
            version=version,
        )

        # 8. Assemble PII-safe metric summary for audit retention
        metric_summary = {
            "macro_f1_all_fields": report.aggregate_metrics.macro_f1_all_fields,
            "macro_f1_key_fields": report.aggregate_metrics.macro_f1_key_fields,
            "dataset_calibration_gap_points": report.aggregate_metrics.dataset_calibration_gap_points,
            "real_samples": report.real_samples,
            "total_samples": report.total_samples,
            "dataset_split": report.dataset_split.value,
            "policy_version": report.policy_version,
            "gate_eligible": report.gate_eligible,
        }

        # 9. Create LifecycleEvent and SchemaApprovalRecord
        approval_id = f"appr_{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc)

        event = LifecycleEvent(
            tenant_id=tenant_id,
            schema_id=schema_id,
            version=version,
            event_type=LifecycleEventType.APPROVED,
            actor=principal.principal_id,
            timestamp=now,
            from_status=snapshot.status,
            to_status=SchemaStatus.APPROVED,
            details={
                "author": snapshot.author,
                "approver": principal.principal_id,
                "harness_run_id": harness_run_id,
                "approval_id": approval_id,
                "reason": reason,
            },
        )

        approval_record = SchemaApprovalRecord(
            approval_id=approval_id,
            tenant_id=tenant_id,
            schema_id=schema_id,
            version=version,
            author_id=snapshot.author,
            approver_id=principal.principal_id,
            approval_timestamp=now,
            harness_run_id=harness_run_id,
            benchmark_policy_version=report.policy_version,
            metric_summary=metric_summary,
            reason=reason,
            lifecycle_event_id=event.event_id,
        )

        # 10. Atomic repository promotion and active version activation via manager
        promoted_snapshot = self.manager.internal_promote_to_approved(
            tenant_id=tenant_id,
            schema_id=schema_id,
            version=version,
            approver=principal.principal_id,
            harness_run_id=harness_run_id,
            reason=reason,
            trusted_token=self._approval_token,
            event=event,
            approval_record=approval_record,
        )

        # 11. Mark harness run ID as consumed in report store
        try:
            self.report_store.mark_consumed(
                harness_run_id=harness_run_id,
                tenant_id=tenant_id,
                schema_id=schema_id,
                version=version,
            )
        except Exception:
            raise

        return promoted_snapshot, approval_record
