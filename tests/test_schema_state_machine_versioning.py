"""
Acceptance Tests for Schema State Machine & Immutable Versioning (TASK-4.1).
Covers PRD Requirements:
- SCH-004: Lifecycle state machine (draft, testing, approved, retired),
  production schema binding restrictions (SCHEMA_NOT_APPROVED).
- SCH-006: Immutable integer version snapshots, author and timestamp preservation,
  deterministic structural diffs, complete version history, non-destructive rollback,
  append-only audit events, and tenant isolation.
"""

from datetime import datetime, timezone
import pytest

from doc_extractor.classification.base import (
    ApprovedSchemaRef,
    DocumentClassificationService,
    SchemaRegistry,
    SchemaResolutionError,
)
from doc_extractor.classification.rules import RuleBasedClassifierAdapter
from doc_extractor.contracts_b import MVPDocumentType
from doc_extractor.pipeline_b import ModuleBPipeline
from schema_lifecycle.adapters import (
    LifecycleSchemaRegistryAdapter,
    SchemaVersionAdapter,
)
from schema_lifecycle.diff import compute_schema_diff
from schema_lifecycle.enums import FieldType, PIIClass, RuleOnFail, SchemaStatus
from schema_lifecycle.events import LifecycleEventType
from schema_lifecycle.exceptions import (
    InvalidStateTransitionError,
    SchemaNotApprovedError,
    SchemaNotFoundError,
    SchemaPromotionNotAllowedError,
    SchemaRollbackError,
    SchemaVersionError,
    TenantAccessError,
)
from schema_lifecycle.manager import SchemaLifecycleManager
from schema_lifecycle.metamodel import (
    FieldDefinition,
    RuleDefinition,
    SchemaDefinition,
)
from schema_lifecycle.repository import (
    InMemorySchemaRepository,
    SchemaSnapshot,
)
from schema_lifecycle.state_machine import SchemaStateMachine


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
        rules=[
            RuleDefinition(
                id="R1",
                expression="initial_deposit >= 1000",
                on_fail=RuleOnFail.REVIEW,
                description="Minimum deposit check",
            )
        ],
    )


# ============================================================================
# 1. SCH-004: Valid and Invalid State Transitions
# ============================================================================

def test_sch_004_valid_state_transitions():
    """SCH-004: Valid transitions among draft, testing, approved, and retired."""
    # draft -> testing
    assert SchemaStateMachine.is_valid_transition(SchemaStatus.DRAFT, SchemaStatus.TESTING) is True
    SchemaStateMachine.validate_transition(SchemaStatus.DRAFT, SchemaStatus.TESTING)

    # draft -> retired (abandoning draft)
    assert SchemaStateMachine.is_valid_transition(SchemaStatus.DRAFT, SchemaStatus.RETIRED) is True
    SchemaStateMachine.validate_transition(SchemaStatus.DRAFT, SchemaStatus.RETIRED)

    # testing -> draft (revisions needed after testing)
    assert SchemaStateMachine.is_valid_transition(SchemaStatus.TESTING, SchemaStatus.DRAFT) is True
    SchemaStateMachine.validate_transition(SchemaStatus.TESTING, SchemaStatus.DRAFT)

    # testing -> retired (discarding during test evaluation)
    assert SchemaStateMachine.is_valid_transition(SchemaStatus.TESTING, SchemaStatus.RETIRED) is True
    SchemaStateMachine.validate_transition(SchemaStatus.TESTING, SchemaStatus.RETIRED)

    # approved -> retired (superseding/decommissioning approved schema)
    assert SchemaStateMachine.is_valid_transition(SchemaStatus.APPROVED, SchemaStatus.RETIRED) is True
    SchemaStateMachine.validate_transition(SchemaStatus.APPROVED, SchemaStatus.RETIRED)

    # testing -> approved (valid transition for internal approval service integration point)
    assert SchemaStateMachine.is_valid_transition(SchemaStatus.TESTING, SchemaStatus.APPROVED) is True
    SchemaStateMachine.validate_approval_transition(SchemaStatus.TESTING)


def test_sch_004_invalid_state_transitions():
    """SCH-004: Invalid state transitions are rejected with InvalidStateTransitionError."""
    # draft -> approved directly is forbidden (must pass testing and approval service)
    with pytest.raises(InvalidStateTransitionError) as exc:
        SchemaStateMachine.validate_transition(SchemaStatus.DRAFT, SchemaStatus.APPROVED)
    assert exc.value.code == "SCHEMA_PROMOTION_NOT_ALLOWED" or exc.value.code == "INVALID_STATE_TRANSITION"

    # approved -> draft is forbidden (approved schemas are immutable, modify via new version)
    with pytest.raises(InvalidStateTransitionError) as exc:
        SchemaStateMachine.validate_transition(SchemaStatus.APPROVED, SchemaStatus.DRAFT)
    assert exc.value.code == "INVALID_STATE_TRANSITION"

    # approved -> testing is forbidden
    with pytest.raises(InvalidStateTransitionError) as exc:
        SchemaStateMachine.validate_transition(SchemaStatus.APPROVED, SchemaStatus.TESTING)
    assert exc.value.code == "INVALID_STATE_TRANSITION"

    # retired is terminal; transitions out of retired are rejected
    for target in [SchemaStatus.DRAFT, SchemaStatus.TESTING, SchemaStatus.APPROVED]:
        with pytest.raises(InvalidStateTransitionError) as exc:
            SchemaStateMachine.validate_transition(SchemaStatus.RETIRED, target)

    # Self-transitions are invalid
    for status in SchemaStatus:
        with pytest.raises(InvalidStateTransitionError) as exc:
            SchemaStateMachine.validate_transition(status, status)
        assert "already in status" in str(exc.value)


def test_sch_004_direct_promotion_blocked_without_approval_service():
    """
    Critical Constraint: Ordinary lifecycle operations cannot promote schemas to approved.
    Caller-controlled authorization parameter is removed.
    TASK-4.2 approval service integration point is internal and requires TESTING status.
    """
    manager = SchemaLifecycleManager()
    schema_def = make_test_schema(status=SchemaStatus.DRAFT)
    manager.create_schema("tenant_alpha", schema_def, author="author_alice")

    # Transition to testing succeeds
    manager.transition_status("tenant_alpha", "DOC-01", 1, SchemaStatus.TESTING, actor="author_alice")

    # Direct promotion to approved via standard call MUST be blocked unconditionally
    with pytest.raises(SchemaPromotionNotAllowedError) as exc:
        manager.transition_status(
            tenant_id="tenant_alpha",
            schema_id="DOC-01",
            version=1,
            target_status=SchemaStatus.APPROVED,
            actor="author_alice",
        )
    assert exc.value.code == "SCHEMA_PROMOTION_NOT_ALLOWED"
    assert "reserved for the TASK-4.2 approval service" in str(exc.value)

    # Verify ordinary caller cannot pass is_approval_service parameter (parameter removed)
    with pytest.raises(TypeError):
        manager.transition_status(
            tenant_id="tenant_alpha",
            schema_id="DOC-01",
            version=1,
            target_status=SchemaStatus.APPROVED,
            actor="author_alice",
            is_approval_service=True,  # type: ignore
        )

    # Dedicated internal integration point (TASK-4.2 simulation) promotes successfully when in TESTING
    approved_snap = manager.internal_promote_to_approved(
        tenant_id="tenant_alpha",
        schema_id="DOC-01",
        version=1,
        approver="approver_bob",
        harness_run_id="run_1234",
    )
    assert approved_snap.status == SchemaStatus.APPROVED

    # Dedicated internal integration point rejects approval if schema is NOT in TESTING (e.g. in DRAFT)
    draft_schema = make_test_schema(schema_id="DOC-02", version=1, status=SchemaStatus.DRAFT)
    manager.create_schema("tenant_alpha", draft_schema, author="author_carol")
    with pytest.raises(InvalidStateTransitionError) as exc_draft:
        manager.internal_promote_to_approved(
            tenant_id="tenant_alpha",
            schema_id="DOC-02",
            version=1,
            approver="approver_bob",
        )
    assert "Schema must be in 'testing' status" in str(exc_draft.value)


# ============================================================================
# 2. SCH-004: Rejection of Non-Approved Production Bindings
# ============================================================================

def test_sch_004_rejection_of_draft_schema_in_production():
    """
    SCH-004 Acceptance Criteria:
    Attempt to bind a 'draft' schema in production returns SCHEMA_NOT_APPROVED.
    """
    manager = SchemaLifecycleManager()
    schema_def = make_test_schema(status=SchemaStatus.DRAFT)
    manager.create_schema("tenant_alpha", schema_def, author="author_alice")

    # 1. Explicit version binding of draft version
    with pytest.raises(SchemaNotApprovedError) as exc:
        manager.bind_production_schema("tenant_alpha", "DOC-01", version=1)
    assert exc.value.code == "SCHEMA_NOT_APPROVED"

    # 2. Active schema resolution when only draft exists
    with pytest.raises(SchemaNotApprovedError) as exc:
        manager.resolve_active_schema("tenant_alpha", "DOC-01")
    assert exc.value.code == "SCHEMA_NOT_APPROVED"


def test_sch_004_rejection_of_testing_and_retired_in_production():
    """SCH-004: Testing and retired versions are rejected in production binding."""
    manager = SchemaLifecycleManager()
    schema_def = make_test_schema(status=SchemaStatus.DRAFT)
    manager.create_schema("tenant_alpha", schema_def, author="author_alice")

    # Transition to testing
    manager.transition_status("tenant_alpha", "DOC-01", 1, SchemaStatus.TESTING, actor="author_alice")
    with pytest.raises(SchemaNotApprovedError) as exc:
        manager.bind_production_schema("tenant_alpha", "DOC-01", version=1)
    assert exc.value.code == "SCHEMA_NOT_APPROVED"

    # Transition to retired
    manager.transition_status("tenant_alpha", "DOC-01", 1, SchemaStatus.RETIRED, actor="author_alice")
    with pytest.raises(SchemaNotApprovedError) as exc:
        manager.bind_production_schema("tenant_alpha", "DOC-01", version=1)
    assert exc.value.code == "SCHEMA_NOT_APPROVED"


# ============================================================================
# 3. SCH-006: Creation of Immutable New Versions
# ============================================================================

def test_sch_006_immutable_version_snapshots():
    """
    SCH-006: Every schema change creates an immutable version with author,
    timestamp, and deterministic diff.
    """
    manager = SchemaLifecycleManager()
    v1_def = make_test_schema(version=1)
    snap1 = manager.create_schema("tenant_alpha", v1_def, author="alice")
    assert snap1.version == 1
    assert snap1.status == SchemaStatus.DRAFT
    assert snap1.author == "alice"

    # Immutability check on SchemaSnapshot model
    with pytest.raises(Exception):
        snap1.version = 2  # Frozen model assignment must fail

    # Modify definition to create version 2
    v2_def = make_test_schema(version=1)
    v2_def.fields.append(FieldDefinition(name="nominee_name", type=FieldType.STRING))
    snap2 = manager.create_version("tenant_alpha", "DOC-01", v2_def, author="bob", comment="Added nominee")

    assert snap2.version == 2
    assert snap2.status == SchemaStatus.DRAFT
    assert snap2.author == "bob"
    assert snap2.parent_version == 1

    # Attempting to save a conflicting version fails
    with pytest.raises(SchemaVersionError) as exc:
        manager.repository.save_snapshot(snap1)
    assert "already exists" in str(exc.value)

    # v1 remains completely unchanged
    v1_stored = manager.repository.get_snapshot("DOC-01", 1, "tenant_alpha")
    assert v1_stored is not None
    assert len(v1_stored.definition.fields) == 3
    assert len(snap2.definition.fields) == 4


# ============================================================================
# 4. SCH-006: Author and Timestamp Preservation
# ============================================================================

def test_sch_006_author_and_timestamp_preservation():
    """SCH-006: Snapshot preserves exact author identity and creation timestamp."""
    manager = SchemaLifecycleManager()
    t_start = datetime.now(timezone.utc)

    snap1 = manager.create_schema("tenant_alpha", make_test_schema(), author="alice@example.com")
    assert snap1.author == "alice@example.com"
    assert snap1.created_at >= t_start
    assert snap1.created_at.tzinfo == timezone.utc

    # Reject empty or whitespace-only author
    with pytest.raises(ValueError):
        manager.create_schema("tenant_alpha", make_test_schema(schema_id="DOC-02"), author="")
    with pytest.raises(ValueError):
        manager.create_schema("tenant_alpha", make_test_schema(schema_id="DOC-02"), author="   ")


# ============================================================================
# 5. SCH-006: Deterministic Structural Diffs
# ============================================================================

def test_sch_006_deterministic_structural_diffs():
    """
    SCH-006 & SCH-003: Deterministic structural diff captures added, removed,
    and modified fields and rules in a strictly reproducible shape.
    """
    old_schema = make_test_schema(version=1)

    # Create new definition with changes:
    # 1. Add field "nominee_name"
    # 2. Remove field "initial_deposit"
    # 3. Modify field "applicant_name" (change required to False, add confidence threshold)
    # 4. Add rule "R2"
    new_schema = SchemaDefinition(
        schema_id="DOC-01",
        version=2,
        status=SchemaStatus.DRAFT,
        domain="BFSI",
        document_type="casa_account_opening",
        fields=[
            FieldDefinition(name="applicant_name", type=FieldType.STRING, required=False, confidence_threshold=0.9),
            FieldDefinition(name="pan_number", type=FieldType.STRING, format="^[A-Z]{5}[0-9]{4}[A-Z]$"),
            FieldDefinition(name="nominee_name", type=FieldType.STRING),
        ],
        rules=[
            RuleDefinition(id="R1", expression="initial_deposit >= 1000"),
            RuleDefinition(id="R2", expression="applicant_name != null", on_fail=RuleOnFail.REJECT),
        ],
    )

    diff1 = compute_schema_diff(old_schema, new_schema)
    diff2 = compute_schema_diff(old_schema, new_schema)

    # Strict reproducibility (deterministic dictionary and ordering)
    assert diff1 == diff2

    # Structural verification
    assert diff1["has_changes"] is True
    assert diff1["summary"]["added_fields"] == 1
    assert diff1["fields"]["added"][0]["name"] == "nominee_name"

    assert diff1["summary"]["removed_fields"] == 1
    assert diff1["fields"]["removed"][0]["name"] == "initial_deposit"

    assert diff1["summary"]["modified_fields"] == 1
    assert diff1["fields"]["modified"][0]["name"] == "applicant_name"
    assert diff1["fields"]["modified"][0]["changes"]["required"] == {"old": True, "new": False}
    assert diff1["fields"]["modified"][0]["changes"]["confidence_threshold"] == {"old": None, "new": 0.9}

    assert diff1["summary"]["added_rules"] == 1
    assert diff1["rules"]["added"][0]["id"] == "R2"

    # Identical schemas yield empty diff (SCH-003 roundtrip diff is empty)
    diff_identical = compute_schema_diff(old_schema, old_schema)
    assert diff_identical["has_changes"] is False
    assert diff_identical["summary"]["total_changes"] == 0
    assert len(diff_identical["fields"]["added"]) == 0
    assert len(diff_identical["fields"]["removed"]) == 0
    assert len(diff_identical["fields"]["modified"]) == 0


# ============================================================================
# 6. SCH-006: Complete Version History
# ============================================================================

def test_sch_006_complete_version_history():
    """SCH-006: Maintain complete version history across sequential edits."""
    manager = SchemaLifecycleManager()

    # Create v1
    v1_def = make_test_schema(version=1)
    manager.create_schema("tenant_alpha", v1_def, author="alice")

    # Create v2
    v2_def = make_test_schema(version=1)
    v2_def.fields.append(FieldDefinition(name="field_2", type=FieldType.STRING))
    manager.create_version("tenant_alpha", "DOC-01", v2_def, author="bob")

    # Create v3
    v3_def = make_test_schema(version=1)
    v3_def.fields.append(FieldDefinition(name="field_3", type=FieldType.STRING))
    manager.create_version("tenant_alpha", "DOC-01", v3_def, author="charlie")

    history = manager.get_version_history("tenant_alpha", "DOC-01")
    assert len(history) == 3
    assert [s.version for s in history] == [1, 2, 3]
    assert [s.author for s in history] == ["alice", "bob", "charlie"]


# ============================================================================
# 7. SCH-006: Non-Destructive Rollback and Active Version Resolution
# ============================================================================

def test_sch_006_rollback_and_active_version_resolution():
    """
    SCH-006 Acceptance Criteria:
    Rollback shall re-activate a prior approved version without deleting history.
    After rollback, new documents bind the older version; the audit log shows the rollback.
    """
    manager = SchemaLifecycleManager()

    # 1. Author and approve Version 1
    v1_def = make_test_schema(version=1)
    manager.create_schema("tenant_alpha", v1_def, author="alice")
    manager.transition_status("tenant_alpha", "DOC-01", 1, SchemaStatus.TESTING, actor="alice")
    manager.internal_promote_to_approved("tenant_alpha", "DOC-01", 1, approver="approver1")
    assert manager.resolve_active_schema("tenant_alpha", "DOC-01").version == 1

    # 2. Author and approve Version 2
    v2_def = make_test_schema(version=1)
    v2_def.fields.append(FieldDefinition(name="new_field", type=FieldType.STRING))
    manager.create_version("tenant_alpha", "DOC-01", v2_def, author="bob")
    manager.transition_status("tenant_alpha", "DOC-01", 2, SchemaStatus.TESTING, actor="bob")
    manager.internal_promote_to_approved("tenant_alpha", "DOC-01", 2, approver="approver2")
    assert manager.resolve_active_schema("tenant_alpha", "DOC-01").version == 2

    # 3. Rollback to Version 1
    rolled_back = manager.rollback(
        tenant_id="tenant_alpha",
        schema_id="DOC-01",
        target_version=1,
        actor="ops_lead",
        reason="Incident mitigation: Version 2 parser regressions",
    )
    assert rolled_back.version == 1
    assert rolled_back.status == SchemaStatus.APPROVED

    # 4. Verify that new production documents now bind Version 1
    active_now = manager.resolve_active_schema("tenant_alpha", "DOC-01")
    assert active_now.version == 1
    bound_snap = manager.bind_production_schema("tenant_alpha", "DOC-01")
    assert bound_snap.version == 1

    # 5. Non-destructive: Version history keeps both versions completely intact
    history = manager.get_version_history("tenant_alpha", "DOC-01")
    assert len(history) == 2
    assert history[0].version == 1
    assert history[1].version == 2
    assert history[1].definition.fields[-1].name == "new_field"  # Not deleted or modified

    # 6. Audit log records the rollback event
    events = manager.get_audit_events("tenant_alpha", "DOC-01")
    rollback_events = [e for e in events if e.event_type == LifecycleEventType.ROLLBACK]
    assert len(rollback_events) == 1
    r_event = rollback_events[0]
    assert r_event.version == 1
    assert r_event.actor == "ops_lead"
    assert r_event.details["previous_active_version"] == 2
    assert r_event.details["target_version"] == 1


def test_sch_006_rollback_rejection_for_never_approved_version():
    """SCH-006: Rollback to a version that was never approved is rejected."""
    manager = SchemaLifecycleManager()

    # v1 approved
    v1_def = make_test_schema(version=1)
    manager.create_schema("tenant_alpha", v1_def, author="alice")
    manager.transition_status("tenant_alpha", "DOC-01", 1, SchemaStatus.TESTING, actor="alice")
    manager.internal_promote_to_approved("tenant_alpha", "DOC-01", 1, approver="approver1")

    # v2 drafted but never approved
    v2_def = make_test_schema(version=1)
    manager.create_version("tenant_alpha", "DOC-01", v2_def, author="bob")

    with pytest.raises(SchemaRollbackError) as exc:
        manager.rollback("tenant_alpha", "DOC-01", target_version=2, actor="ops")
    assert exc.value.code == "SCHEMA_ROLLBACK_ERROR"
    assert "was never approved" in str(exc.value)

    # Rollback to non-existent version 99
    with pytest.raises(SchemaNotFoundError):
        manager.rollback("tenant_alpha", "DOC-01", target_version=99, actor="ops")


# ============================================================================
# 8. Tenant Isolation Tests
# ============================================================================

def test_tenant_isolation_strict_partitioning():
    """Acceptance test: Strict tenant isolation prevents cross-tenant schema contamination."""
    manager = SchemaLifecycleManager()

    # Tenant A schema
    schema_a = make_test_schema(schema_id="DOC-01", tenant_id="tenant_a")
    manager.create_schema("tenant_a", schema_a, author="alice_a")

    # Tenant B schema with same schema_id but different field
    schema_b = make_test_schema(schema_id="DOC-01", tenant_id="tenant_b")
    schema_b.fields.append(FieldDefinition(name="tenant_b_secret_field", type=FieldType.STRING))
    manager.create_schema("tenant_b", schema_b, author="bob_b")

    # Tenant A sees only Tenant A's fields
    snap_a = manager.repository.get_snapshot("DOC-01", 1, "tenant_a")
    assert snap_a is not None
    assert "tenant_b_secret_field" not in [f.name for f in snap_a.definition.fields]

    # Tenant B sees only Tenant B's fields
    snap_b = manager.repository.get_snapshot("DOC-01", 1, "tenant_b")
    assert snap_b is not None
    assert "tenant_b_secret_field" in [f.name for f in snap_b.definition.fields]

    # Tenant C (unrelated) querying DOC-01 finds nothing
    assert manager.repository.get_snapshot("DOC-01", 1, "tenant_c") is None
    with pytest.raises(SchemaNotFoundError):
        manager.create_version("tenant_c", "DOC-01", schema_a, author="mallory")

    # Empty tenant_id is strictly rejected
    with pytest.raises(TenantAccessError):
        manager.repository.get_snapshot("DOC-01", 1, "")


# ============================================================================
# 9. Existing SchemaRegistry Compatibility & Adapter Tests
# ============================================================================

def test_schema_registry_adapter_compatibility():
    """
    Scope 10 & Acceptance Criteria:
    Integrate with the existing SchemaRegistry through an adapter without
    modifying its public contract.
    """
    manager = SchemaLifecycleManager()

    # Seed approved CASA schema (DOC-01) for tenant_alpha
    casa_def = make_test_schema(schema_id="DOC-01", version=1, status=SchemaStatus.APPROVED)
    manager.create_schema("tenant_alpha", casa_def, author="admin")

    adapter = LifecycleSchemaRegistryAdapter(
        manager=manager,
        default_tenant="tenant_alpha",
    )

    # 1. Adapter satisfies SchemaRegistry interface
    assert isinstance(adapter, SchemaRegistry)

    # 2. Resolves approved schema reference
    ref = adapter.resolve_approved_schema(MVPDocumentType.DOC_01_CASA)
    assert isinstance(ref, ApprovedSchemaRef)
    assert ref.schema_id == "DOC-01"
    assert ref.schema_version == "1.0.0"
    assert ref.is_active is True

    # 3. is_schema_approved checks
    assert adapter.is_schema_approved("DOC-01", "1.0.0") is True
    assert adapter.is_schema_approved("DOC-01", "2.0.0") is False
    assert adapter.is_schema_approved("UNKNOWN", "1.0.0") is False

    # 4. Unknown doc type raises SchemaResolutionError per existing contract
    with pytest.raises(SchemaResolutionError) as exc:
        adapter.resolve_approved_schema(MVPDocumentType.UNKNOWN)
    assert exc.value.code == "UNKNOWN_DOC_TYPE_NO_SCHEMA"

    # 5. Non-approved doc type raises SchemaResolutionError with NO_APPROVED_SCHEMA
    with pytest.raises(SchemaResolutionError) as exc:
        adapter.resolve_approved_schema(MVPDocumentType.DOC_02_INSURANCE_APP)
    assert exc.value.code == "NO_APPROVED_SCHEMA"


def test_schema_registry_adapter_pipeline_b_integration():
    """
    Verifies that LifecycleSchemaRegistryAdapter integrates seamlessly into
    DocumentClassificationService and ModuleBPipeline without modifying any existing pipeline code.
    """
    import tempfile
    from doc_extractor.storage import LocalDiskStorage

    manager = SchemaLifecycleManager()

    # Seed approved CASA schema (DOC-01)
    casa_def = make_test_schema(schema_id="DOC-01", version=1, status=SchemaStatus.APPROVED)
    manager.create_schema("tenant_alpha", casa_def, author="admin")

    adapter = LifecycleSchemaRegistryAdapter(
        manager=manager,
        default_tenant="tenant_alpha",
    )

    # 1. Direct classification service integration
    classifier_engine = RuleBasedClassifierAdapter()
    classification_service = DocumentClassificationService(
        classifier=classifier_engine,
        schema_registry=adapter,
    )
    ref = classification_service.schema_registry.resolve_approved_schema(
        MVPDocumentType.DOC_01_CASA,
        tenant_id="tenant_alpha",
    )
    assert ref.schema_id == "DOC-01"
    assert ref.schema_version == "1.0.0"
    assert ref.is_active is True

    # 2. ModuleBPipeline initialization with adapter
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = LocalDiskStorage(base_dir=tmpdir, tenant_id="tenant_alpha")
        pipeline = ModuleBPipeline(
            storage=storage,
            schema_registry=adapter,
            tenant_id="tenant_alpha",
        )
        assert pipeline.schema_registry is adapter
        # Classification service inside pipeline resolves schema via adapter
        pipe_ref = pipeline.classification_service.schema_registry.resolve_approved_schema(
            MVPDocumentType.DOC_01_CASA,
            tenant_id="tenant_alpha",
        )
        assert pipe_ref.schema_id == "DOC-01"
        assert pipe_ref.schema_version == "1.0.0"


# ============================================================================
# 10. Additional Edge Cases and Robustness Tests
# ============================================================================

def test_edge_cases_state_machine_allowed_transitions():
    """Validates get_allowed_transitions for all statuses."""
    draft_allowed = SchemaStateMachine.get_allowed_transitions(SchemaStatus.DRAFT)
    assert SchemaStatus.TESTING in draft_allowed
    assert SchemaStatus.RETIRED in draft_allowed
    assert SchemaStatus.APPROVED not in draft_allowed

    retired_allowed = SchemaStateMachine.get_allowed_transitions(SchemaStatus.RETIRED)
    assert len(retired_allowed) == 0


def test_edge_cases_diff_metadata_and_validation():
    """Validates diff computation when metadata changes and error on None schema."""
    with pytest.raises(ValueError, match="new_schema cannot be None"):
        compute_schema_diff(None, None)  # type: ignore

    v1 = make_test_schema(version=1)
    v2_meta = SchemaDefinition(
        schema_id="DOC-01",
        version=2,
        status=SchemaStatus.DRAFT,
        domain="HEALTHCARE",  # Changed from BFSI
        document_type="discharge_summary",  # Changed
        description="Updated description",  # Changed
        fields=v1.fields,
        rules=v1.rules,
    )
    diff = compute_schema_diff(v1, v2_meta)
    assert diff["has_changes"] is True
    assert diff["summary"]["metadata_changed"] is True
    assert diff["metadata"]["domain"] == {"old": "BFSI", "new": "HEALTHCARE"}
    assert diff["metadata"]["document_type"] == {"old": "casa_account_opening", "new": "discharge_summary"}
    assert diff["metadata"]["description"] == {"old": "CASA account opening schema", "new": "Updated description"}


def test_edge_cases_manager_validation_and_errors():
    """Tests various boundary errors across SchemaLifecycleManager."""
    manager = SchemaLifecycleManager()

    # Create v1
    v1_def = make_test_schema(version=1)
    manager.create_schema("tenant_alpha", v1_def, author="author1")

    # Mismatched schema_id in definition
    v2_bad = make_test_schema(schema_id="DOC-99", version=1)
    with pytest.raises(ValueError, match="does not match target schema_id"):
        manager.create_version("tenant_alpha", "DOC-01", v2_bad, author="author2")

    # Blank actor in transition_status
    with pytest.raises(ValueError, match="Actor identity must be a non-empty string"):
        manager.transition_status("tenant_alpha", "DOC-01", 1, SchemaStatus.TESTING, actor="")

    # Blank actor in rollback
    with pytest.raises(ValueError, match="Actor identity must be a non-empty string"):
        manager.rollback("tenant_alpha", "DOC-01", 1, actor="")

    # Transition non-existent schema
    with pytest.raises(SchemaNotFoundError):
        manager.transition_status("tenant_alpha", "NON_EXISTENT", 1, SchemaStatus.TESTING, actor="author1")

    # Bind non-existent version
    with pytest.raises(SchemaNotFoundError):
        manager.bind_production_schema("tenant_alpha", "DOC-01", version=999)

    # Repository update_status on non-existent version
    with pytest.raises(SchemaNotFoundError):
        manager.repository.update_status("DOC-01", 999, "tenant_alpha", SchemaStatus.TESTING)

    # Repository set_active_version on non-existent version
    with pytest.raises(SchemaNotFoundError):
        manager.repository.set_active_version("DOC-01", 999, "tenant_alpha")


def test_edge_cases_rollback_to_previously_approved_then_retired_version():
    """
    Validates that rolling back to a previously approved version that was subsequently
    retired successfully restores it to APPROVED status and sets it active.
    """
    manager = SchemaLifecycleManager()

    # 1. v1 approved
    v1_def = make_test_schema(version=1)
    manager.create_schema("tenant_alpha", v1_def, author="alice")
    manager.transition_status("tenant_alpha", "DOC-01", 1, SchemaStatus.TESTING, actor="alice")
    manager.internal_promote_to_approved("tenant_alpha", "DOC-01", 1, approver="approver")

    # 2. v1 subsequently retired
    manager.transition_status("tenant_alpha", "DOC-01", 1, SchemaStatus.RETIRED, actor="ops")
    v1_snap = manager.repository.get_snapshot("DOC-01", 1, "tenant_alpha")
    assert v1_snap.status == SchemaStatus.RETIRED

    # 3. Rollback to v1
    restored = manager.rollback("tenant_alpha", "DOC-01", target_version=1, actor="ops_lead")
    assert restored.version == 1
    assert restored.status == SchemaStatus.APPROVED
    assert manager.resolve_active_schema("tenant_alpha", "DOC-01").version == 1


def test_edge_cases_adapter_graceful_failures():
    """Validates LifecycleSchemaRegistryAdapter returns False/handles invalid version strings."""
    manager = SchemaLifecycleManager()
    casa_def = make_test_schema(schema_id="DOC-01", version=1, status=SchemaStatus.APPROVED)
    manager.create_schema("tenant_alpha", casa_def, author="admin")

    adapter = LifecycleSchemaRegistryAdapter(manager=manager, default_tenant="tenant_alpha")

    # Invalid string version returns False gracefully
    assert adapter.is_schema_approved("DOC-01", "invalid_semver_xyz") is False
    assert adapter.is_schema_approved("DOC-01", "-1.0.0") is False

    # Document type with no mapping
    custom_adapter = LifecycleSchemaRegistryAdapter(
        manager=manager,
        default_tenant="tenant_alpha",
        document_type_mappings={},  # empty mapping
    )
    with pytest.raises(SchemaResolutionError) as exc:
        custom_adapter.resolve_approved_schema(MVPDocumentType.DOC_01_CASA)
    assert exc.value.code == "NO_APPROVED_SCHEMA"


# ============================================================================
# 11. Deep Immutability & Rollback Consistency Tests (Remediation)
# ============================================================================

def test_deep_immutability_retrieved_snapshot_mutation():
    """
    Remediation Audit Item 2: Deep immutability.
    Proves that mutating fields, rules, descriptions, or diff on a retrieved snapshot
    has zero effect on historical versions stored in the repository.
    """
    manager = SchemaLifecycleManager()
    v1_def = make_test_schema(version=1)
    manager.create_schema("tenant_alpha", v1_def, author="alice")

    # Retrieve snapshot v1 and attempt malicious in-place mutations
    snap = manager.repository.get_snapshot("DOC-01", 1, "tenant_alpha")
    assert snap is not None
    original_field_count = len(snap.definition.fields)
    original_req = snap.definition.fields[0].required
    original_desc = snap.definition.description

    # Mutate nested lists and objects
    snap.definition.fields.append(FieldDefinition(name="unauthorized_field", type=FieldType.STRING))
    snap.definition.fields[0].required = not original_req
    snap.definition.rules.append(RuleDefinition(id="MALICIOUS", expression="1 == 1"))
    snap.definition.description = "HACKED_DESCRIPTION"
    if snap.diff:
        snap.diff["summary"]["added_fields"] = 9999

    # Fetch a fresh snapshot from the repository
    fresh = manager.repository.get_snapshot("DOC-01", 1, "tenant_alpha")
    assert fresh is not None
    assert len(fresh.definition.fields) == original_field_count
    assert fresh.definition.fields[0].required == original_req
    assert [f.name for f in fresh.definition.fields] == [f.name for f in v1_def.fields]
    assert len(fresh.definition.rules) == len(v1_def.rules)
    assert fresh.definition.description == original_desc
    if fresh.diff:
        assert fresh.diff["summary"]["added_fields"] != 9999


def test_deep_immutability_input_definition_mutation():
    """
    Remediation Audit Item 2: Deep immutability.
    Proves that mutating the original SchemaDefinition object after passing it to
    create_schema() or create_version() does not corrupt the repository snapshot.
    """
    manager = SchemaLifecycleManager()
    v1_def = make_test_schema(version=1)
    manager.create_schema("tenant_alpha", v1_def, author="alice")

    # Mutate the input object that was passed to create_schema
    v1_def.fields.append(FieldDefinition(name="post_creation_field", type=FieldType.STRING))
    v1_def.description = "MUTATED_AFTER_CREATION"

    # Fetch from repository
    stored = manager.repository.get_snapshot("DOC-01", 1, "tenant_alpha")
    assert stored is not None
    assert "post_creation_field" not in [f.name for f in stored.definition.fields]
    assert stored.definition.description == "CASA account opening schema"


def test_rollback_reconciles_retired_status_and_rejects_draft():
    """
    Remediation Audit Item 3: Rollback consistency.
    Ensures validate_rollback_transition allows APPROVED and RETIRED, but rejects DRAFT and TESTING.
    """
    # Permitted in rollback (re-activation of previously approved schema)
    SchemaStateMachine.validate_rollback_transition(SchemaStatus.APPROVED)
    SchemaStateMachine.validate_rollback_transition(SchemaStatus.RETIRED)

    # Rejected in rollback
    with pytest.raises(InvalidStateTransitionError):
        SchemaStateMachine.validate_rollback_transition(SchemaStatus.DRAFT)
    with pytest.raises(InvalidStateTransitionError):
        SchemaStateMachine.validate_rollback_transition(SchemaStatus.TESTING)
