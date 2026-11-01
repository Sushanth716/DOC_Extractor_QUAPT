"""
Acceptance Tests for TASK-4.2 Four-Eyes Approval Gate and Role-Based Access Control.
Strictly adheres to PRD v1.1 §10.1, §13.1..13.3, §15.5, SCH-005, and SEC-003:
- Four-Eyes Approval: author cannot approve own schema version (SCHEMA_SELF_APPROVAL).
- SEC-003 Role-Based Access Control: least-privilege matrix across all 7 roles.
- SEC-001 MFA Verification: approvers must have verified MFA.
- Trusted Benchmark Verification: authoritatively verified passing benchmark from report store.
- Rejection of mock, heuristic, synthetic-only, and failed-sample reports.
- Separation of schema approval from product release qualification (ApprovalPolicy).
- Immutability of schema versions and approval records.
- Atomic promotion, active version pointer update, and rollback safety under concurrency.
"""

import copy
from datetime import datetime, timezone
import threading
import pytest

from doc_extractor.contracts_c import ExecutionMode
from schema_lifecycle.approval import (
    ApprovalPolicy,
    BenchmarkReportStore,
    BenchmarkResultVerifier,
    InMemoryBenchmarkReportStore,
    SchemaApprovalRecord,
    SchemaApprovalService,
)
from schema_lifecycle.enums import FieldType, SchemaStatus
from schema_lifecycle.events import LifecycleEvent, LifecycleEventType
from schema_lifecycle.exceptions import (
    BenchmarkVerificationError,
    ForbiddenActionError,
    InvalidStateTransitionError,
    SchemaNotFoundError,
    SchemaPromotionNotAllowedError,
    SchemaSelfApprovalError,
    TenantAccessError,
    UnauthenticatedError,
)
from schema_lifecycle.manager import SchemaLifecycleManager
from schema_lifecycle.metamodel import FieldDefinition, SchemaDefinition
from schema_lifecycle.repository import InMemorySchemaRepository
from schema_lifecycle.roles import (
    AuthenticatedPrincipal,
    DefaultSchemaAuthorizer,
    Role,
    ROLE_PERMISSIONS,
    SchemaAuthorizer,
    SchemaPermission,
)
from schema_lifecycle.testing.contracts import (
    AggregateMetrics,
    DatasetSplit,
    FieldMetrics,
)
from schema_lifecycle.testing.policy import EvaluationDecision
from schema_lifecycle.testing.runner import (
    BenchmarkReport,
    GR005Accounting,
)


# ============================================================================
# Test Fixtures & Helpers
# ============================================================================

def make_test_schema(
    schema_id: str = "DOC-01",
    version: int = 1,
    status: SchemaStatus = SchemaStatus.DRAFT,
    tenant_id: str = "tenant_alpha",
) -> SchemaDefinition:
    """Helper to build a valid SchemaDefinition fixture."""
    return SchemaDefinition(
        schema_id=schema_id,
        version=version,
        status=status,
        tenant_id=tenant_id,
        domain="BFSI",
        document_type="casa_account_opening",
        description="CASA account opening schema",
        fields=[
            FieldDefinition(name="applicant_name", type=FieldType.STRING, required=True),
            FieldDefinition(name="pan_number", type=FieldType.STRING, format="^[A-Z]{5}[0-9]{4}[A-Z]$"),
            FieldDefinition(name="initial_deposit", type=FieldType.DECIMAL),
        ],
    )


def make_passing_benchmark_report(
    benchmark_id: str = "bench_run_001",
    tenant_id: str = "tenant_alpha",
    schema_id: str = "DOC-01",
    schema_version: int = 1,
    dataset_split: DatasetSplit = DatasetSplit.LOCKED_TEST,
    is_mock_run: bool = False,
    execution_modes: list = None,
    real_samples: int = 100,
    synthetic_samples: int = 0,
    failed_samples: int = 0,
    overall_pass: bool = True,
    gate_eligible: bool = True,
    gr005_status: str = "VERIFIED",
    meets_m1_all: bool = True,
    meets_m1_key: bool = True,
    meets_m6_cal: bool = True,
    has_insufficient_data: bool = False,
    run_timestamp: str = None,
) -> BenchmarkReport:
    """Constructs a deterministic passing BenchmarkReport fixture with configurable overrides."""
    run_ts = run_timestamp or datetime.now(timezone.utc).isoformat()
    modes = execution_modes if execution_modes is not None else [ExecutionMode.GENUINE_LLM.value]

    agg_metrics = AggregateMetrics(
        primary_aggregation="macro",
        macro_f1_all_fields=0.96 if meets_m1_all else 0.85,
        macro_f1_key_fields=0.98 if meets_m1_key else 0.88,
        macro_precision_all=0.97,
        macro_recall_all=0.95,
        micro_f1_all_fields=0.96,
        micro_f1_key_fields=0.98,
        micro_precision_all=0.97,
        micro_recall_all=0.95,
        mean_calibration_gap_points=1.5 if meets_m6_cal else 5.0,
        dataset_calibration_gap_points=1.5 if meets_m6_cal else 5.0,
        calibration_sample_count=290,
        total_samples=real_samples + synthetic_samples,
        total_field_types=3,
        key_field_count=1,
        total_predictions=300,
        total_tp=290,
        total_fp=5,
        total_fn=5,
        total_tn=0,
        escape_count=0,
        has_insufficient_data=has_insufficient_data,
        insufficient_data_reasons=["MOCK_INSUFFICIENT"] if has_insufficient_data else [],
    )

    metrics_dec = EvaluationDecision(
        policy_version="1.0.0",
        dataset_split=dataset_split,
        primary_aggregation="macro",
        meets_m1_all_fields=meets_m1_all,
        meets_m1_key_fields=meets_m1_key,
        meets_m6_calibration=meets_m6_cal,
        meets_m11_escapes=True if dataset_split == DatasetSplit.LOCKED_TEST else None,
        meets_gr003_false_fill=True,
        has_insufficient_data=has_insufficient_data,
        insufficient_data_reasons=["MOCK_INSUFFICIENT"] if has_insufficient_data else [],
        overall_pass=overall_pass and meets_m1_all and meets_m1_key and meets_m6_cal and not has_insufficient_data,
        violations=[] if overall_pass else ["METRIC_FAILURE"],
        summary="Passing decision" if overall_pass else "Violations occurred",
    )

    gr005 = GR005Accounting(
        status=gr005_status,
        total_attempted_runs=1000,
        successful_validations=1000 if gr005_status == "VERIFIED" else 100,
        unique_qualifying_runs=1000 if gr005_status == "VERIFIED" else 100,
        validation_exceptions=0 if gr005_status != "FAILED" else 3,
        conformance_run_ids=[f"run_{i}" for i in range(10)],
        required_runs=1000,
        is_conformance_verified=gr005_status == "VERIFIED",
        limitation_note="Verified" if gr005_status == "VERIFIED" else "Incomplete",
    )

    return BenchmarkReport(
        benchmark_id=benchmark_id,
        run_timestamp=run_ts,
        dataset_id=f"ds_{schema_id}",
        tenant_id=tenant_id,
        schema_id=schema_id,
        schema_version=schema_version,
        document_type="casa_account_opening",
        dataset_split=dataset_split,
        policy_version="1.0.0",
        primary_aggregation="macro",
        total_samples=real_samples + synthetic_samples,
        real_samples=real_samples,
        synthetic_samples=synthetic_samples,
        failed_samples=failed_samples,
        is_mock_run=is_mock_run,
        execution_modes=modes,
        providers=["litellm"],
        models=["claude-3-5-sonnet"],
        prompt_versions=["1.0.0"],
        field_metrics={},
        aggregate_metrics=agg_metrics,
        metrics_decision=metrics_dec,
        synthetic_field_metrics=None,
        synthetic_aggregate_metrics=None,
        gate_eligible=gate_eligible,
        gate_eligibility_reasons=["GATE_QUALIFIED"] if gate_eligible else ["NOT_QUALIFIED"],
        gr005_accounting=gr005,
        sample_observations=[],
        overall_pass=overall_pass,
        summary="Passing benchmark report summary",
    )


@pytest.fixture
def repo() -> InMemorySchemaRepository:
    return InMemorySchemaRepository()


@pytest.fixture
def manager(repo: InMemorySchemaRepository) -> SchemaLifecycleManager:
    return SchemaLifecycleManager(repository=repo)


@pytest.fixture
def authorizer() -> DefaultSchemaAuthorizer:
    auth = DefaultSchemaAuthorizer()
    bob = AuthenticatedPrincipal(
        principal_id="bob",
        tenant_id="tenant_alpha",
        roles={Role.SCHEMA_APPROVER},
        mfa_verified=True,
    )
    auth.register_principal(bob)
    return auth


@pytest.fixture
def report_store() -> InMemoryBenchmarkReportStore:
    return InMemoryBenchmarkReportStore()


@pytest.fixture
def approval_service(
    manager: SchemaLifecycleManager,
    authorizer: DefaultSchemaAuthorizer,
    report_store: InMemoryBenchmarkReportStore,
) -> SchemaApprovalService:
    verifier = BenchmarkResultVerifier(report_store=report_store, policy=ApprovalPolicy())
    return SchemaApprovalService(
        manager=manager,
        authorizer=authorizer,
        verifier=verifier,
        report_store=report_store,
    )


# ============================================================================
# 1. Four-Eyes and Permissions (SCH-005, SEC-001, SEC-003)
# ============================================================================

def test_author_cannot_approve_own_version(manager, approval_service, authorizer, report_store):
    """SCH-005 Check 1: Author cannot approve their own schema version."""
    # Alice authors schema v1
    schema_def = make_test_schema(status=SchemaStatus.DRAFT)
    manager.create_schema("tenant_alpha", schema_def, author="alice")
    manager.transition_status("tenant_alpha", "DOC-01", 1, SchemaStatus.TESTING, actor="alice")

    # Seed passing benchmark report
    report = make_passing_benchmark_report(benchmark_id="run_101", tenant_id="tenant_alpha", schema_id="DOC-01", schema_version=1)
    report_store.save_report(report)

    # Alice has approver role + MFA, but she is the author
    alice = AuthenticatedPrincipal(
        principal_id="alice",
        tenant_id="tenant_alpha",
        roles={Role.SCHEMA_AUTHOR, Role.SCHEMA_APPROVER},
        mfa_verified=True,
    )
    authorizer.register_principal(alice)

    # Alice attempts self-approval -> Refused with SCHEMA_SELF_APPROVAL
    with pytest.raises(SchemaSelfApprovalError) as exc_info:
        approval_service.approve_schema(
            principal=alice,
            tenant_id="tenant_alpha",
            schema_id="DOC-01",
            version=1,
            harness_run_id="run_101",
        )
    assert exc_info.value.code == "SCHEMA_SELF_APPROVAL"
    assert "authored schema" in str(exc_info.value).lower()
    assert exc_info.value.author == "alice"


def test_different_user_with_approver_role_can_approve(manager, approval_service, authorizer, report_store):
    """SCH-005 Check 2: A different user holding Schema approver role can approve."""
    # Alice authors schema v1
    schema_def = make_test_schema(status=SchemaStatus.DRAFT)
    manager.create_schema("tenant_alpha", schema_def, author="alice")
    manager.transition_status("tenant_alpha", "DOC-01", 1, SchemaStatus.TESTING, actor="alice")

    # Passing report in store
    report = make_passing_benchmark_report(benchmark_id="run_102", tenant_id="tenant_alpha", schema_id="DOC-01", schema_version=1)
    report_store.save_report(report)

    # Bob is a distinct approver with MFA
    bob = AuthenticatedPrincipal(
        principal_id="bob",
        tenant_id="tenant_alpha",
        roles={Role.SCHEMA_APPROVER},
        mfa_verified=True,
    )
    authorizer.register_principal(bob)

    snap, record = approval_service.approve_schema(
        principal=bob,
        tenant_id="tenant_alpha",
        schema_id="DOC-01",
        version=1,
        harness_run_id="run_102",
        reason="Passing four-eyes review",
    )

    assert snap.status == SchemaStatus.APPROVED
    assert snap.approved_by == "bob"
    assert snap.harness_run_id == "run_102"
    assert record.author_id == "alice"
    assert record.approver_id == "bob"
    assert manager.resolve_active_schema("tenant_alpha", "DOC-01").version == 1


def test_author_with_admin_role_cannot_self_approve(manager, approval_service, authorizer, report_store):
    """SCH-005 Check 3: Author holding Admin role cannot self-approve (four-eyes is absolute)."""
    schema_def = make_test_schema(status=SchemaStatus.DRAFT)
    manager.create_schema("tenant_alpha", schema_def, author="admin_alice")
    manager.transition_status("tenant_alpha", "DOC-01", 1, SchemaStatus.TESTING, actor="admin_alice")

    report = make_passing_benchmark_report(benchmark_id="run_103", tenant_id="tenant_alpha", schema_id="DOC-01", schema_version=1)
    report_store.save_report(report)

    admin_alice = AuthenticatedPrincipal(
        principal_id="admin_alice",
        tenant_id="tenant_alpha",
        roles={Role.ADMIN, Role.SCHEMA_APPROVER},
        mfa_verified=True,
    )
    authorizer.register_principal(admin_alice)

    with pytest.raises(SchemaSelfApprovalError) as exc_info:
        approval_service.approve_schema(
            principal=admin_alice,
            tenant_id="tenant_alpha",
            schema_id="DOC-01",
            version=1,
            harness_run_id="run_103",
        )
    assert exc_info.value.code == "SCHEMA_SELF_APPROVAL"


def test_analyst_cannot_approve(manager, approval_service, authorizer, report_store):
    """SEC-003 Check 4: Analyst cannot approve schemas."""
    manager.create_schema("tenant_alpha", make_test_schema(status=SchemaStatus.TESTING), author="alice")
    report = make_passing_benchmark_report(benchmark_id="run_104", tenant_id="tenant_alpha", schema_id="DOC-01", schema_version=1)
    report_store.save_report(report)

    analyst = AuthenticatedPrincipal(principal_id="dan", tenant_id="tenant_alpha", roles={Role.ANALYST}, mfa_verified=True)
    authorizer.register_principal(analyst)

    with pytest.raises(ForbiddenActionError) as exc_info:
        approval_service.approve_schema(analyst, "tenant_alpha", "DOC-01", 1, "run_104")
    assert exc_info.value.code == "FORBIDDEN"


def test_reviewer_cannot_approve(manager, approval_service, authorizer, report_store):
    """SEC-003 Check 5: Reviewer cannot approve schemas."""
    manager.create_schema("tenant_alpha", make_test_schema(status=SchemaStatus.TESTING), author="alice")
    report = make_passing_benchmark_report(benchmark_id="run_105", tenant_id="tenant_alpha", schema_id="DOC-01", schema_version=1)
    report_store.save_report(report)

    reviewer = AuthenticatedPrincipal(principal_id="rev", tenant_id="tenant_alpha", roles={Role.REVIEWER}, mfa_verified=True)
    authorizer.register_principal(reviewer)

    with pytest.raises(ForbiddenActionError) as exc_info:
        approval_service.approve_schema(reviewer, "tenant_alpha", "DOC-01", 1, "run_105")
    assert exc_info.value.code == "FORBIDDEN"


def test_integrator_cannot_approve(manager, approval_service, authorizer, report_store):
    """SEC-003 Check 6: Integrator cannot approve schemas."""
    manager.create_schema("tenant_alpha", make_test_schema(status=SchemaStatus.TESTING), author="alice")
    report = make_passing_benchmark_report(benchmark_id="run_106", tenant_id="tenant_alpha", schema_id="DOC-01", schema_version=1)
    report_store.save_report(report)

    integrator = AuthenticatedPrincipal(principal_id="integ", tenant_id="tenant_alpha", roles={Role.INTEGRATOR}, mfa_verified=True)
    authorizer.register_principal(integrator)

    with pytest.raises(ForbiddenActionError) as exc_info:
        approval_service.approve_schema(integrator, "tenant_alpha", "DOC-01", 1, "run_106")
    assert exc_info.value.code == "FORBIDDEN"


def test_auditor_cannot_approve(manager, approval_service, authorizer, report_store):
    """SEC-003 Check 7: Auditor cannot approve schemas."""
    manager.create_schema("tenant_alpha", make_test_schema(status=SchemaStatus.TESTING), author="alice")
    report = make_passing_benchmark_report(benchmark_id="run_107", tenant_id="tenant_alpha", schema_id="DOC-01", schema_version=1)
    report_store.save_report(report)

    auditor = AuthenticatedPrincipal(principal_id="audit", tenant_id="tenant_alpha", roles={Role.AUDITOR}, mfa_verified=True)
    authorizer.register_principal(auditor)

    with pytest.raises(ForbiddenActionError) as exc_info:
        approval_service.approve_schema(auditor, "tenant_alpha", "DOC-01", 1, "run_107")
    assert exc_info.value.code == "FORBIDDEN"


def test_admin_without_explicit_approval_permission_cannot_approve(manager, approval_service, authorizer, report_store):
    """SEC-003 Check 8: Admin without explicit approval role/permission cannot approve."""
    manager.create_schema("tenant_alpha", make_test_schema(status=SchemaStatus.TESTING), author="alice")
    report = make_passing_benchmark_report(benchmark_id="run_108", tenant_id="tenant_alpha", schema_id="DOC-01", schema_version=1)
    report_store.save_report(report)

    admin = AuthenticatedPrincipal(principal_id="admin_user", tenant_id="tenant_alpha", roles={Role.ADMIN}, mfa_verified=True)
    authorizer.register_principal(admin)

    with pytest.raises(ForbiddenActionError) as exc_info:
        approval_service.approve_schema(admin, "tenant_alpha", "DOC-01", 1, "run_108")
    assert exc_info.value.code == "FORBIDDEN"


def test_unauthenticated_principal_is_rejected(manager, approval_service, authorizer, report_store):
    """SEC-001 Check 9: Unauthenticated or inactive principal is rejected."""
    manager.create_schema("tenant_alpha", make_test_schema(status=SchemaStatus.TESTING), author="alice")
    report = make_passing_benchmark_report(benchmark_id="run_109", tenant_id="tenant_alpha", schema_id="DOC-01", schema_version=1)
    report_store.save_report(report)

    inactive_approver = AuthenticatedPrincipal(
        principal_id="bob",
        tenant_id="tenant_alpha",
        roles={Role.SCHEMA_APPROVER},
        is_active=False,
        mfa_verified=True,
    )
    with pytest.raises(UnauthenticatedError) as exc_info:
        approval_service.approve_schema(inactive_approver, "tenant_alpha", "DOC-01", 1, "run_109")
    assert exc_info.value.code == "UNAUTHENTICATED"


def test_approver_without_trusted_mfa_is_rejected(manager, approval_service, authorizer, report_store):
    """SEC-001 Check 10: Approver without verified MFA is rejected."""
    manager.create_schema("tenant_alpha", make_test_schema(status=SchemaStatus.TESTING), author="alice")
    report = make_passing_benchmark_report(benchmark_id="run_110", tenant_id="tenant_alpha", schema_id="DOC-01", schema_version=1)
    report_store.save_report(report)

    no_mfa_approver = AuthenticatedPrincipal(
        principal_id="bob",
        tenant_id="tenant_alpha",
        roles={Role.SCHEMA_APPROVER},
        mfa_verified=False,  # Lacks MFA
    )
    authorizer.register_principal(no_mfa_approver)

    with pytest.raises(ForbiddenActionError) as exc_info:
        approval_service.approve_schema(no_mfa_approver, "tenant_alpha", "DOC-01", 1, "run_110")
    assert exc_info.value.code == "FORBIDDEN"
    assert "mfa" in str(exc_info.value).lower()


# ============================================================================
# 2. Benchmark Verification (SCH-005, SCH-007)
# ============================================================================

def test_missing_harness_result_rejected(manager, approval_service, authorizer, report_store):
    """SCH-005 Check 11: Missing harness result is rejected."""
    manager.create_schema("tenant_alpha", make_test_schema(status=SchemaStatus.TESTING), author="alice")
    bob = AuthenticatedPrincipal(principal_id="bob", tenant_id="tenant_alpha", roles={Role.SCHEMA_APPROVER}, mfa_verified=True)

    with pytest.raises(BenchmarkVerificationError) as exc_info:
        approval_service.approve_schema(bob, "tenant_alpha", "DOC-01", 1, "non_existent_run_999")
    assert exc_info.value.code == "HARNESS_RESULT_INVALID"
    assert "not found" in str(exc_info.value).lower()


def test_arbitrary_or_untrusted_report_rejected(manager, authorizer, report_store):
    """SCH-005 Check 12: Arbitrary or untrusted report object not in trusted store is rejected."""
    verifier = BenchmarkResultVerifier(report_store=report_store)
    # Verifier only resolves from report_store; caller cannot pass unverified in-memory report directly
    with pytest.raises(BenchmarkVerificationError) as exc_info:
        verifier.verify_for_approval("unregistered_run_id", "tenant_alpha", "DOC-01", 1)
    assert exc_info.value.code == "HARNESS_RESULT_INVALID"


def test_tenant_mismatch_rejected(manager, approval_service, authorizer, report_store):
    """SCH-005 Check 13: Benchmark run on another tenant is rejected."""
    manager.create_schema("tenant_alpha", make_test_schema(status=SchemaStatus.TESTING), author="alice")
    # Report belongs to tenant_beta
    report = make_passing_benchmark_report(benchmark_id="run_113", tenant_id="tenant_beta", schema_id="DOC-01", schema_version=1)
    report_store.save_report(report)

    bob = AuthenticatedPrincipal(principal_id="bob", tenant_id="tenant_alpha", roles={Role.SCHEMA_APPROVER}, mfa_verified=True)

    with pytest.raises(BenchmarkVerificationError) as exc_info:
        approval_service.approve_schema(bob, "tenant_alpha", "DOC-01", 1, "run_113")
    assert "tenant" in str(exc_info.value).lower()


def test_schema_id_mismatch_rejected(manager, approval_service, authorizer, report_store):
    """SCH-005 Check 14: Benchmark run on a different schema ID is rejected."""
    manager.create_schema("tenant_alpha", make_test_schema(schema_id="DOC-01", status=SchemaStatus.TESTING), author="alice")
    # Report is for DOC-02
    report = make_passing_benchmark_report(benchmark_id="run_114", tenant_id="tenant_alpha", schema_id="DOC-02", schema_version=1)
    report_store.save_report(report)

    bob = AuthenticatedPrincipal(principal_id="bob", tenant_id="tenant_alpha", roles={Role.SCHEMA_APPROVER}, mfa_verified=True)

    with pytest.raises(BenchmarkVerificationError) as exc_info:
        approval_service.approve_schema(bob, "tenant_alpha", "DOC-01", 1, "run_114")
    assert "schema id" in str(exc_info.value).lower()


def test_schema_version_mismatch_rejected(manager, approval_service, authorizer, report_store):
    """SCH-005 Check 15: Benchmark run on another schema version is rejected."""
    manager.create_schema("tenant_alpha", make_test_schema(status=SchemaStatus.TESTING), author="alice")
    # Report is for version 2, while schema is version 1
    report = make_passing_benchmark_report(benchmark_id="run_115", tenant_id="tenant_alpha", schema_id="DOC-01", schema_version=2)
    report_store.save_report(report)

    bob = AuthenticatedPrincipal(principal_id="bob", tenant_id="tenant_alpha", roles={Role.SCHEMA_APPROVER}, mfa_verified=True)

    with pytest.raises(BenchmarkVerificationError) as exc_info:
        approval_service.approve_schema(bob, "tenant_alpha", "DOC-01", 1, "run_115")
    assert "version" in str(exc_info.value).lower()


def test_failed_metrics_rejected(manager, approval_service, authorizer, report_store):
    """SCH-005 Check 16: Report with failed evaluation metrics is rejected."""
    manager.create_schema("tenant_alpha", make_test_schema(status=SchemaStatus.TESTING), author="alice")
    # Report has failing M1 all-fields metric
    report = make_passing_benchmark_report(
        benchmark_id="run_116",
        tenant_id="tenant_alpha",
        schema_id="DOC-01",
        schema_version=1,
        meets_m1_all=False,
        overall_pass=False,
    )
    report_store.save_report(report)

    bob = AuthenticatedPrincipal(principal_id="bob", tenant_id="tenant_alpha", roles={Role.SCHEMA_APPROVER}, mfa_verified=True)

    with pytest.raises(BenchmarkVerificationError) as exc_info:
        approval_service.approve_schema(bob, "tenant_alpha", "DOC-01", 1, "run_116")
    assert "metric" in str(exc_info.value).lower()


def test_mock_or_heuristic_provenance_rejected(manager, approval_service, authorizer, report_store):
    """SCH-005 Check 17: Mock or heuristic provenance execution is rejected."""
    manager.create_schema("tenant_alpha", make_test_schema(status=SchemaStatus.TESTING), author="alice")
    # Mock run report
    report = make_passing_benchmark_report(
        benchmark_id="run_117",
        tenant_id="tenant_alpha",
        schema_id="DOC-01",
        schema_version=1,
        is_mock_run=True,
        execution_modes=[ExecutionMode.MOCK.value],
    )
    report_store.save_report(report)

    bob = AuthenticatedPrincipal(principal_id="bob", tenant_id="tenant_alpha", roles={Role.SCHEMA_APPROVER}, mfa_verified=True)

    with pytest.raises(BenchmarkVerificationError) as exc_info:
        approval_service.approve_schema(bob, "tenant_alpha", "DOC-01", 1, "run_117")
    assert "mock" in str(exc_info.value).lower()


def test_synthetic_only_report_rejected(manager, approval_service, authorizer, report_store):
    """SCH-005 Check 18: Synthetic-only report is rejected."""
    manager.create_schema("tenant_alpha", make_test_schema(status=SchemaStatus.TESTING), author="alice")
    report = make_passing_benchmark_report(
        benchmark_id="run_118",
        tenant_id="tenant_alpha",
        schema_id="DOC-01",
        schema_version=1,
        real_samples=0,
        synthetic_samples=100,
    )
    report_store.save_report(report)

    bob = AuthenticatedPrincipal(principal_id="bob", tenant_id="tenant_alpha", roles={Role.SCHEMA_APPROVER}, mfa_verified=True)

    with pytest.raises(BenchmarkVerificationError) as exc_info:
        approval_service.approve_schema(bob, "tenant_alpha", "DOC-01", 1, "run_118")
    assert "synthetic-only" in str(exc_info.value).lower()


def test_failed_sample_report_rejected(manager, approval_service, authorizer, report_store):
    """SCH-005 Check 19: Report with failed extraction samples is rejected."""
    manager.create_schema("tenant_alpha", make_test_schema(status=SchemaStatus.TESTING), author="alice")
    report = make_passing_benchmark_report(
        benchmark_id="run_119",
        tenant_id="tenant_alpha",
        schema_id="DOC-01",
        schema_version=1,
        failed_samples=2,
    )
    report_store.save_report(report)

    bob = AuthenticatedPrincipal(principal_id="bob", tenant_id="tenant_alpha", roles={Role.SCHEMA_APPROVER}, mfa_verified=True)

    with pytest.raises(BenchmarkVerificationError) as exc_info:
        approval_service.approve_schema(bob, "tenant_alpha", "DOC-01", 1, "run_119")
    assert "failed sample" in str(exc_info.value).lower()


def test_incomplete_or_failed_approval_policy_rejected(manager, authorizer, report_store):
    """SCH-005 Check 20: Incomplete or failed approval policy criteria are rejected."""
    # Policy requiring locked_test split
    strict_policy = ApprovalPolicy(require_locked_test=True, require_gate_eligible=True)
    verifier = BenchmarkResultVerifier(report_store=report_store, policy=strict_policy)
    service = SchemaApprovalService(manager=manager, authorizer=authorizer, verifier=verifier, report_store=report_store)

    manager.create_schema("tenant_alpha", make_test_schema(status=SchemaStatus.TESTING), author="alice")
    # Report run on development split
    dev_report = make_passing_benchmark_report(
        benchmark_id="run_120",
        tenant_id="tenant_alpha",
        schema_id="DOC-01",
        schema_version=1,
        dataset_split=DatasetSplit.DEVELOPMENT,
    )
    report_store.save_report(dev_report)

    bob = AuthenticatedPrincipal(principal_id="bob", tenant_id="tenant_alpha", roles={Role.SCHEMA_APPROVER}, mfa_verified=True)

    with pytest.raises(BenchmarkVerificationError) as exc_info:
        service.approve_schema(bob, "tenant_alpha", "DOC-01", 1, "run_120")
    assert "locked_test" in str(exc_info.value).lower()


def test_consumed_harness_run_cannot_approve_another_version(manager, approval_service, authorizer, report_store):
    """SCH-005 Check 21: A consumed harness run cannot be reused to approve another version."""
    manager.create_schema("tenant_alpha", make_test_schema(status=SchemaStatus.TESTING), author="alice")
    report = make_passing_benchmark_report(benchmark_id="run_121", tenant_id="tenant_alpha", schema_id="DOC-01", schema_version=1)
    report_store.save_report(report)

    bob = AuthenticatedPrincipal(principal_id="bob", tenant_id="tenant_alpha", roles={Role.SCHEMA_APPROVER}, mfa_verified=True)

    # 1. First approval consumes run_121
    approval_service.approve_schema(bob, "tenant_alpha", "DOC-01", 1, "run_121")
    assert report_store.is_consumed("run_121") is True

    # 2. Create version 2 in TESTING
    v2_def = make_test_schema(version=1)
    manager.create_version("tenant_alpha", "DOC-01", v2_def, author="alice")
    manager.transition_status("tenant_alpha", "DOC-01", 2, SchemaStatus.TESTING, actor="alice")

    # 3. Attempting to reuse run_121 for v2 fails anti-replay
    with pytest.raises(BenchmarkVerificationError) as exc_info:
        approval_service.approve_schema(bob, "tenant_alpha", "DOC-01", 2, "run_121")
    assert "already been consumed" in str(exc_info.value).lower()


def test_development_tuning_vs_locked_test_follows_policy(manager, authorizer, report_store):
    """SCH-005 Check 22: Development vs locked_test split enforcement follows explicit ApprovalPolicy."""
    # Permissive policy for local unit testing (require_locked_test=False)
    permissive_policy = ApprovalPolicy(require_locked_test=False)
    verifier = BenchmarkResultVerifier(report_store=report_store, policy=permissive_policy)
    service = SchemaApprovalService(manager=manager, authorizer=authorizer, verifier=verifier, report_store=report_store)

    manager.create_schema("tenant_alpha", make_test_schema(status=SchemaStatus.TESTING), author="alice")
    tuning_report = make_passing_benchmark_report(
        benchmark_id="run_122",
        tenant_id="tenant_alpha",
        schema_id="DOC-01",
        schema_version=1,
        dataset_split=DatasetSplit.TUNING,
    )
    report_store.save_report(tuning_report)

    bob = AuthenticatedPrincipal(principal_id="bob", tenant_id="tenant_alpha", roles={Role.SCHEMA_APPROVER}, mfa_verified=True)

    # Permissive policy permits tuning split
    snap, _ = service.approve_schema(bob, "tenant_alpha", "DOC-01", 1, "run_122")
    assert snap.status == SchemaStatus.APPROVED


# ============================================================================
# 3. Lifecycle Enforcement (SCH-004, SCH-005)
# ============================================================================

def test_draft_schema_cannot_be_approved(manager, approval_service, authorizer, report_store):
    """SCH-004 Check 23: Schema in DRAFT status cannot be approved."""
    manager.create_schema("tenant_alpha", make_test_schema(status=SchemaStatus.DRAFT), author="alice")
    report = make_passing_benchmark_report(benchmark_id="run_123", tenant_id="tenant_alpha", schema_id="DOC-01", schema_version=1)
    report_store.save_report(report)

    bob = AuthenticatedPrincipal(principal_id="bob", tenant_id="tenant_alpha", roles={Role.SCHEMA_APPROVER}, mfa_verified=True)

    with pytest.raises(InvalidStateTransitionError) as exc_info:
        approval_service.approve_schema(bob, "tenant_alpha", "DOC-01", 1, "run_123")
    assert "testing" in str(exc_info.value).lower()


def test_retired_schema_cannot_be_approved(manager, approval_service, authorizer, report_store):
    """SCH-004 Check 24: Schema in RETIRED status cannot be approved."""
    manager.create_schema("tenant_alpha", make_test_schema(status=SchemaStatus.DRAFT), author="alice")
    manager.transition_status("tenant_alpha", "DOC-01", 1, SchemaStatus.RETIRED, actor="alice")
    report = make_passing_benchmark_report(benchmark_id="run_124", tenant_id="tenant_alpha", schema_id="DOC-01", schema_version=1)
    report_store.save_report(report)

    bob = AuthenticatedPrincipal(principal_id="bob", tenant_id="tenant_alpha", roles={Role.SCHEMA_APPROVER}, mfa_verified=True)

    with pytest.raises(InvalidStateTransitionError) as exc_info:
        approval_service.approve_schema(bob, "tenant_alpha", "DOC-01", 1, "run_124")
    assert "testing" in str(exc_info.value).lower()


def test_direct_transition_to_approved_remains_blocked(manager):
    """SCH-004 Check 25: Direct transition to APPROVED remains blocked via transition_status."""
    manager.create_schema("tenant_alpha", make_test_schema(status=SchemaStatus.TESTING), author="alice")

    with pytest.raises(SchemaPromotionNotAllowedError) as exc_info:
        manager.transition_status("tenant_alpha", "DOC-01", 1, SchemaStatus.APPROVED, actor="bob")
    assert exc_info.value.code == "SCHEMA_PROMOTION_NOT_ALLOWED"


def test_creating_schema_directly_as_approved_rejected(manager):
    """SCH-004 Check 26: Creating a schema directly as APPROVED is rejected."""
    casa_def = make_test_schema(schema_id="DOC-01", version=1, status=SchemaStatus.APPROVED)
    with pytest.raises(SchemaPromotionNotAllowedError) as exc_info:
        manager.create_schema("tenant_alpha", casa_def, author="admin")
    assert exc_info.value.code == "SCHEMA_PROMOTION_NOT_ALLOWED"
    assert "cannot create schema directly in 'approved' status" in str(exc_info.value).lower()


# ============================================================================
# 4. Immutability, Atomicity & Audit (SCH-005, SCH-006, SEC-008)
# ============================================================================

def test_snapshot_persists_approved_by_and_harness_run_id(manager, approval_service, authorizer, report_store):
    """SCH-005 Check 27: Snapshot retains approved_by and harness_run_id."""
    manager.create_schema("tenant_alpha", make_test_schema(status=SchemaStatus.TESTING), author="alice")
    report = make_passing_benchmark_report(benchmark_id="run_127", tenant_id="tenant_alpha", schema_id="DOC-01", schema_version=1)
    report_store.save_report(report)

    bob = AuthenticatedPrincipal(principal_id="bob", tenant_id="tenant_alpha", roles={Role.SCHEMA_APPROVER}, mfa_verified=True)
    snap, _ = approval_service.approve_schema(bob, "tenant_alpha", "DOC-01", 1, "run_127")

    # Verify snapshot returned and snapshot in repository both persist metadata
    assert snap.approved_by == "bob"
    assert snap.harness_run_id == "run_127"

    repo_snap = manager.repository.get_snapshot("DOC-01", 1, "tenant_alpha")
    assert repo_snap.approved_by == "bob"
    assert repo_snap.harness_run_id == "run_127"


def test_approval_record_contains_complete_metadata(manager, approval_service, authorizer, report_store):
    """SCH-005 Check 28: Approval record contains complete metadata without PII."""
    manager.create_schema("tenant_alpha", make_test_schema(status=SchemaStatus.TESTING), author="alice")
    report = make_passing_benchmark_report(benchmark_id="run_128", tenant_id="tenant_alpha", schema_id="DOC-01", schema_version=1)
    report_store.save_report(report)

    bob = AuthenticatedPrincipal(principal_id="bob", tenant_id="tenant_alpha", roles={Role.SCHEMA_APPROVER}, mfa_verified=True)
    _, record = approval_service.approve_schema(bob, "tenant_alpha", "DOC-01", 1, "run_128", reason="Compliance sign-off")

    assert record.tenant_id == "tenant_alpha"
    assert record.schema_id == "DOC-01"
    assert record.version == 1
    assert record.author_id == "alice"
    assert record.approver_id == "bob"
    assert record.harness_run_id == "run_128"
    assert record.benchmark_policy_version == "1.0.0"
    assert record.reason == "Compliance sign-off"
    assert "macro_f1_all_fields" in record.metric_summary

    # Stored record retrievable from repository
    stored_rec = manager.repository.get_approval_record("DOC-01", 1, "tenant_alpha")
    assert stored_rec is not None
    assert stored_rec.approval_id == record.approval_id


def test_definition_remains_immutable_after_approval(manager, approval_service, authorizer, report_store):
    """SCH-006 Check 29: Definition remains deeply immutable after approval."""
    manager.create_schema("tenant_alpha", make_test_schema(status=SchemaStatus.TESTING), author="alice")
    report = make_passing_benchmark_report(benchmark_id="run_129", tenant_id="tenant_alpha", schema_id="DOC-01", schema_version=1)
    report_store.save_report(report)

    bob = AuthenticatedPrincipal(principal_id="bob", tenant_id="tenant_alpha", roles={Role.SCHEMA_APPROVER}, mfa_verified=True)
    snap, _ = approval_service.approve_schema(bob, "tenant_alpha", "DOC-01", 1, "run_129")

    # Mutation attempt on frozen snapshot raises error
    with pytest.raises(Exception):
        snap.status = SchemaStatus.DRAFT  # frozen=True


def test_rollback_preserves_approval_history(manager, approval_service, authorizer, report_store):
    """SCH-006 Check 30: Rollback preserves approval history."""
    # 1. Author and approve v1
    manager.create_schema("tenant_alpha", make_test_schema(status=SchemaStatus.TESTING), author="alice")
    r1 = make_passing_benchmark_report(benchmark_id="run_v1", tenant_id="tenant_alpha", schema_id="DOC-01", schema_version=1)
    report_store.save_report(r1)
    bob = AuthenticatedPrincipal(principal_id="bob", tenant_id="tenant_alpha", roles={Role.SCHEMA_APPROVER}, mfa_verified=True)
    approval_service.approve_schema(bob, "tenant_alpha", "DOC-01", 1, "run_v1")

    # 2. Author and approve v2
    v2_def = make_test_schema(version=1)
    manager.create_version("tenant_alpha", "DOC-01", v2_def, author="charlie")
    manager.transition_status("tenant_alpha", "DOC-01", 2, SchemaStatus.TESTING, actor="charlie")
    r2 = make_passing_benchmark_report(benchmark_id="run_v2", tenant_id="tenant_alpha", schema_id="DOC-01", schema_version=2)
    report_store.save_report(r2)
    approval_service.approve_schema(bob, "tenant_alpha", "DOC-01", 2, "run_v2")

    assert manager.resolve_active_schema("tenant_alpha", "DOC-01").version == 2

    # 3. Roll back to v1
    manager.rollback("tenant_alpha", "DOC-01", 1, actor="admin_user")
    assert manager.resolve_active_schema("tenant_alpha", "DOC-01").version == 1

    # Both approval records are preserved
    records = manager.repository.list_approval_records("DOC-01", "tenant_alpha")
    assert len(records) == 2
    assert [r.version for r in records] == [1, 2]


def test_concurrent_approvals_cannot_corrupt_pointer(manager, approval_service, authorizer, report_store):
    """SCH-005 Check 31: Concurrent approvals cannot corrupt the active version pointer."""
    manager.create_schema("tenant_alpha", make_test_schema(status=SchemaStatus.TESTING), author="alice")
    report = make_passing_benchmark_report(benchmark_id="run_131", tenant_id="tenant_alpha", schema_id="DOC-01", schema_version=1)
    report_store.save_report(report)

    bob = AuthenticatedPrincipal(principal_id="bob", tenant_id="tenant_alpha", roles={Role.SCHEMA_APPROVER}, mfa_verified=True)

    results = []
    errors = []

    def try_approve():
        try:
            snap, rec = approval_service.approve_schema(bob, "tenant_alpha", "DOC-01", 1, "run_131")
            results.append(snap)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=try_approve) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Exactly one approval succeeded, the others were rejected (due to consumed harness run or non-TESTING status)
    assert len(results) == 1
    assert len(errors) == 4
    assert manager.resolve_active_schema("tenant_alpha", "DOC-01").version == 1


def test_repository_failure_leaves_no_partial_promotion(manager, repo):
    """SCH-005 Check 32: Repository failure leaves no partial promotion (rollback safety)."""
    manager.create_schema("tenant_alpha", make_test_schema(status=SchemaStatus.TESTING), author="alice")
    event = LifecycleEvent(
        tenant_id="tenant_alpha",
        schema_id="DOC-01",
        version=1,
        event_type=LifecycleEventType.APPROVED,
        actor="bob",
        from_status=SchemaStatus.TESTING,
        to_status=SchemaStatus.APPROVED,
    )

    # Custom list that raises during append to simulate failure during promotion
    class FailingList(list):
        def append(self, item):
            raise RuntimeError("Simulated event audit storage failure")
    orig_events = repo._events
    repo._events = FailingList(orig_events)

    with pytest.raises(RuntimeError) as exc_info:
        repo.promote_and_activate(
            tenant_id="tenant_alpha",
            schema_id="DOC-01",
            version=1,
            approved_by="bob",
            harness_run_id="run_132",
            event=event,
        )
    assert "Simulated event audit" in str(exc_info.value)

    # Repository state must remain in TESTING with NO active pointer
    snap = repo.get_snapshot("DOC-01", 1, "tenant_alpha")
    assert snap.status == SchemaStatus.TESTING
    assert snap.approved_by is None
    assert repo.get_active_version("DOC-01", "tenant_alpha") is None


def test_approval_event_and_snapshot_state_consistent(manager, approval_service, authorizer, report_store):
    """SCH-005 Check 33: Approval event and snapshot state remain strictly consistent."""
    manager.create_schema("tenant_alpha", make_test_schema(status=SchemaStatus.TESTING), author="alice")
    report = make_passing_benchmark_report(benchmark_id="run_133", tenant_id="tenant_alpha", schema_id="DOC-01", schema_version=1)
    report_store.save_report(report)

    bob = AuthenticatedPrincipal(principal_id="bob", tenant_id="tenant_alpha", roles={Role.SCHEMA_APPROVER}, mfa_verified=True)
    snap, record = approval_service.approve_schema(bob, "tenant_alpha", "DOC-01", 1, "run_133")

    events = manager.repository.list_events("DOC-01", "tenant_alpha", version=1)
    approval_events = [e for e in events if e.event_type == LifecycleEventType.APPROVED]
    assert len(approval_events) == 1
    app_ev = approval_events[0]

    assert app_ev.actor == "bob"
    assert app_ev.to_status == SchemaStatus.APPROVED
    assert app_ev.details["author"] == "alice"
    assert app_ev.details["approver"] == "bob"
    assert app_ev.details["harness_run_id"] == "run_133"


def test_replayed_harness_result_cannot_be_consumed_twice(manager, approval_service, authorizer, report_store):
    """SCH-005 Check 34: Replayed harness result cannot be consumed twice in report store."""
    manager.create_schema("tenant_alpha", make_test_schema(status=SchemaStatus.TESTING), author="alice")
    report = make_passing_benchmark_report(benchmark_id="run_134", tenant_id="tenant_alpha", schema_id="DOC-01", schema_version=1)
    report_store.save_report(report)

    bob = AuthenticatedPrincipal(principal_id="bob", tenant_id="tenant_alpha", roles={Role.SCHEMA_APPROVER}, mfa_verified=True)
    approval_service.approve_schema(bob, "tenant_alpha", "DOC-01", 1, "run_134")

    # Mark consumed directly again raises BenchmarkVerificationError
    with pytest.raises(BenchmarkVerificationError) as exc_info:
        report_store.mark_consumed("run_134", "tenant_alpha", "DOC-01", 1)
    assert "already been consumed" in str(exc_info.value).lower()


# ============================================================================
# 5. Comprehensive Permission Matrix Tests (SEC-003)
# ============================================================================

ALL_ROLES = [
    Role.ANALYST,
    Role.REVIEWER,
    Role.SCHEMA_AUTHOR,
    Role.SCHEMA_APPROVER,
    Role.INTEGRATOR,
    Role.AUDITOR,
    Role.ADMIN,
]

ALL_OPERATIONS = [
    SchemaPermission.SCHEMA_READ,
    SchemaPermission.SCHEMA_CREATE,
    SchemaPermission.SCHEMA_VERSION_CREATE,
    SchemaPermission.SCHEMA_EDIT_DRAFT,
    SchemaPermission.SCHEMA_SUBMIT_TEST,
    SchemaPermission.SCHEMA_REQUEST_APPROVAL,
    SchemaPermission.SCHEMA_RUN_HARNESS,
    SchemaPermission.SCHEMA_APPROVE,
    SchemaPermission.SCHEMA_RETIRE,
    SchemaPermission.SCHEMA_ROLLBACK,
    SchemaPermission.SCHEMA_HISTORY_READ,
]


@pytest.mark.parametrize("role", ALL_ROLES)
@pytest.mark.parametrize("permission", ALL_OPERATIONS)
def test_sec_003_role_permission_matrix(role: Role, permission: SchemaPermission, authorizer: DefaultSchemaAuthorizer):
    """
    Parametrized test verifying every role x operation permission outcome matches
    the SEC-003 least-privilege matrix exactly.
    """
    principal = AuthenticatedPrincipal(
        principal_id=f"user_{role.value}",
        tenant_id="tenant_alpha",
        roles={role},
        mfa_verified=True,
    )
    authorizer.register_principal(principal)

    decision = authorizer.authorize(
        principal=principal,
        permission=permission,
        tenant_id="tenant_alpha",
        schema_author="different_author",
    )

    expected_allowed = permission in ROLE_PERMISSIONS[role]
    assert decision.is_authorized == expected_allowed, (
        f"Role {role.value} with permission {permission.value}: expected {expected_allowed}, got {decision.is_authorized}"
    )


# ============================================================================
# TASK-4.2 Final Security Integrity Audit Regression Suite
# ============================================================================

def test_check1_direct_manager_promotion_bypass_is_blocked(manager, approval_service, report_store):
    """
    Check 1 Regression: A caller cannot bypass SchemaApprovalService by invoking
    manager.internal_promote_to_approved() directly with a different approver username
    and arbitrary harness run ID.
    """
    schema_def = make_test_schema(status=SchemaStatus.DRAFT)
    manager.create_schema("tenant_alpha", schema_def, author="alice")
    manager.transition_status("tenant_alpha", "DOC-01", 1, SchemaStatus.TESTING, actor="alice")

    # Direct unauthorized promotion without trusted capability token MUST fail closed
    with pytest.raises(ForbiddenActionError) as exc_info:
        manager.internal_promote_to_approved(
            tenant_id="tenant_alpha",
            schema_id="DOC-01",
            version=1,
            approver="charlie",
            harness_run_id="arbitrary_unverified_harness_run_id",
        )
    assert exc_info.value.code == "FORBIDDEN"
    assert "Direct unauthorized promotion" in str(exc_info.value)

    # Verify repository snapshot remains strictly in TESTING status and active version is unset
    snap = manager.repository.get_snapshot("DOC-01", 1, "tenant_alpha")
    assert snap.status == SchemaStatus.TESTING
    assert snap.approved_by is None
    assert snap.harness_run_id is None
    assert manager.repository.get_active_version("DOC-01", "tenant_alpha") is None


def test_check2_unregistered_principal_fails_closed(manager, approval_service, authorizer, report_store):
    """
    Check 2 Regression: An unregistered caller attempting to construct an AuthenticatedPrincipal
    with forged approver role and MFA fails closed with UNAUTHENTICATED.
    """
    manager.create_schema("tenant_alpha", make_test_schema(status=SchemaStatus.TESTING), author="alice")
    report = make_passing_benchmark_report(benchmark_id="run_chk2_1", tenant_id="tenant_alpha", schema_id="DOC-01", schema_version=1)
    report_store.save_report(report)

    # Forged principal not registered in authorizer boundary
    forged_principal = AuthenticatedPrincipal(
        principal_id="unregistered_attacker",
        tenant_id="tenant_alpha",
        roles={Role.SCHEMA_APPROVER},
        mfa_verified=True,
    )

    with pytest.raises(UnauthenticatedError) as exc_info:
        approval_service.approve_schema(forged_principal, "tenant_alpha", "DOC-01", 1, "run_chk2_1")
    assert exc_info.value.code == "UNAUTHENTICATED"
    assert "not registered in the trusted authentication boundary" in str(exc_info.value)


def test_check2_forged_mfa_claim_detected_and_rejected(manager, approval_service, authorizer, report_store):
    """
    Check 2 Regression: A registered principal registered with mfa_verified=False cannot
    forge mfa_verified=True in request context.
    """
    manager.create_schema("tenant_alpha", make_test_schema(status=SchemaStatus.TESTING), author="alice")
    report = make_passing_benchmark_report(benchmark_id="run_chk2_2", tenant_id="tenant_alpha", schema_id="DOC-01", schema_version=1)
    report_store.save_report(report)

    # Register dave without MFA in trusted authorizer
    dave_authoritative = AuthenticatedPrincipal(
        principal_id="dave",
        tenant_id="tenant_alpha",
        roles={Role.SCHEMA_APPROVER},
        mfa_verified=False,
    )
    authorizer.register_principal(dave_authoritative)

    # Caller attempts to submit forged principal with mfa_verified=True
    forged_dave = dave_authoritative.model_copy(update={"mfa_verified": True})

    with pytest.raises(ForbiddenActionError) as exc_info:
        approval_service.approve_schema(forged_dave, "tenant_alpha", "DOC-01", 1, "run_chk2_2")
    assert exc_info.value.code == "FORBIDDEN"
    assert "forged credentials detected" in str(exc_info.value)


def test_check3_latest_passing_harness_result_enforced_us_docx_010(manager, approval_service, authorizer, report_store):
    """
    Check 3 Regression (US-DOCX-010): When an older passing report and a newer passing report
    exist for the same schema version, attempting to approve using the older report is rejected.
    """
    manager.create_schema("tenant_alpha", make_test_schema(status=SchemaStatus.TESTING), author="alice")

    # Run 1: older passing report (timestamp 2026-10-30T10:00:00Z)
    report_older = make_passing_benchmark_report(
        benchmark_id="run_older_passing",
        tenant_id="tenant_alpha",
        schema_id="DOC-01",
        schema_version=1,
    )
    object.__setattr__(report_older, "run_timestamp", "2026-10-30T10:00:00+00:00")
    report_store.save_report(report_older)

    # Run 2: newer passing report (timestamp 2026-10-30T11:00:00Z)
    report_newer = make_passing_benchmark_report(
        benchmark_id="run_newer_passing",
        tenant_id="tenant_alpha",
        schema_id="DOC-01",
        schema_version=1,
    )
    object.__setattr__(report_newer, "run_timestamp", "2026-10-30T11:00:00+00:00")
    report_store.save_report(report_newer)

    bob = authorizer.get_principal("bob")

    # Attempting to approve using the older report must be rejected per US-DOCX-010
    with pytest.raises(BenchmarkVerificationError) as exc_info:
        approval_service.approve_schema(bob, "tenant_alpha", "DOC-01", 1, "run_older_passing")
    assert exc_info.value.code == "HARNESS_RESULT_INVALID"
    assert "Outdated harness result" in str(exc_info.value)
    assert "US-DOCX-010" in str(exc_info.value)

    # Approving using the latest passing report succeeds
    snap, record = approval_service.approve_schema(bob, "tenant_alpha", "DOC-01", 1, "run_newer_passing")
    assert snap.status == SchemaStatus.APPROVED
    assert snap.harness_run_id == "run_newer_passing"
    assert record.harness_run_id == "run_newer_passing"


def test_check3_superseded_by_failed_run_rejected(manager, approval_service, authorizer, report_store):
    """
    Check 3 Regression: If a newer benchmark run was executed and failed, an older passing
    report cannot be attached to approve the schema.
    """
    manager.create_schema("tenant_alpha", make_test_schema(status=SchemaStatus.TESTING), author="alice")

    # Older passing report
    report_older = make_passing_benchmark_report(
        benchmark_id="run_pass_old",
        tenant_id="tenant_alpha",
        schema_id="DOC-01",
        schema_version=1,
    )
    object.__setattr__(report_older, "run_timestamp", "2026-10-30T09:00:00+00:00")
    report_store.save_report(report_older)

    # Newer failed report
    report_newer_failed = make_passing_benchmark_report(
        benchmark_id="run_fail_new",
        tenant_id="tenant_alpha",
        schema_id="DOC-01",
        schema_version=1,
        overall_pass=False,
        meets_m1_all=False,
    )
    object.__setattr__(report_newer_failed, "run_timestamp", "2026-10-30T12:00:00+00:00")
    report_store.save_report(report_newer_failed)

    bob = authorizer.get_principal("bob")

    # Must be rejected because the subsequent benchmark run on this version failed
    with pytest.raises(BenchmarkVerificationError) as exc_info:
        approval_service.approve_schema(bob, "tenant_alpha", "DOC-01", 1, "run_pass_old")
    assert exc_info.value.code == "HARNESS_RESULT_INVALID"
    assert "Superseded harness result" in str(exc_info.value)


def test_check4_failed_promotion_allows_legitimate_retry(manager, approval_service, authorizer, report_store):
    """
    Check 4 Regression: If repository fails during promotion, no state is partially approved,
    the harness run ID is not marked consumed, and a subsequent legitimate retry succeeds.
    """
    manager.create_schema("tenant_alpha", make_test_schema(status=SchemaStatus.TESTING), author="alice")
    report = make_passing_benchmark_report(benchmark_id="run_retry_1", tenant_id="tenant_alpha", schema_id="DOC-01", schema_version=1)
    report_store.save_report(report)

    bob = authorizer.get_principal("bob")

    # Simulate transient repository failure during promote_and_activate
    real_promote = manager.repository.promote_and_activate
    call_count = 0

    def failing_promote(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("Simulated transient database connection abort")
        return real_promote(*args, **kwargs)

    manager.repository.promote_and_activate = failing_promote

    # First attempt fails with transient error
    with pytest.raises(RuntimeError) as exc_info:
        approval_service.approve_schema(bob, "tenant_alpha", "DOC-01", 1, "run_retry_1")
    assert "Simulated transient" in str(exc_info.value)

    # Verify harness run ID is NOT consumed in report store
    assert not report_store.is_consumed("run_retry_1")
    # Verify harness run ID is NOT consumed in repository
    assert not manager.repository.is_harness_run_consumed("run_retry_1", "tenant_alpha")
    # Verify schema is still in TESTING and no active version pointer exists
    snap = manager.repository.get_snapshot("DOC-01", 1, "tenant_alpha")
    assert snap.status == SchemaStatus.TESTING
    assert manager.repository.get_active_version("DOC-01", "tenant_alpha") is None

    # Legitimate retry must succeed without replay rejection
    snap_ok, record_ok = approval_service.approve_schema(bob, "tenant_alpha", "DOC-01", 1, "run_retry_1")
    assert snap_ok.status == SchemaStatus.APPROVED
    assert report_store.is_consumed("run_retry_1")
    assert manager.repository.is_harness_run_consumed("run_retry_1", "tenant_alpha")
    assert manager.repository.get_active_version("DOC-01", "tenant_alpha").version == 1


def test_check4_concurrent_approvals_cannot_reuse_harness_run(manager, approval_service, authorizer, report_store):
    """
    Check 4 Regression: Concurrent approval attempts cannot reuse the same harness result.
    """
    import concurrent.futures

    manager.create_schema("tenant_alpha", make_test_schema(status=SchemaStatus.TESTING), author="alice")
    report = make_passing_benchmark_report(benchmark_id="run_concurrent_chk4", tenant_id="tenant_alpha", schema_id="DOC-01", schema_version=1)
    report_store.save_report(report)

    bob = authorizer.get_principal("bob")

    results = []
    errors = []

    def attempt_approval():
        try:
            res = approval_service.approve_schema(bob, "tenant_alpha", "DOC-01", 1, "run_concurrent_chk4")
            results.append(res)
        except Exception as exc:
            errors.append(exc)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(attempt_approval) for _ in range(4)]
        concurrent.futures.wait(futures)

    # Exactly one promotion succeeds
    assert len(results) == 1
    # Exactly 3 attempts fail
    assert len(errors) == 3
    # Schema is approved exactly once
    snap = manager.repository.get_snapshot("DOC-01", 1, "tenant_alpha")
    assert snap.status == SchemaStatus.APPROVED
