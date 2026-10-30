"""
Integration and Regression Tests for Dynamic Module C Adapter (Stage 2D).
Validates:
1. End-to-end extraction using a TESTING-bound schema.
2. Rejection of DRAFT, APPROVED, and RETIRED schemas.
3. Nested object field paths.
4. Repeating groups, nested arrays, and stable ordering.
5. Checkbox checked/unchecked evidence grounding and strict evidence requirement.
6. Required and optional null handling.
7. Correct absent versus illegible null reasons.
8. Dynamic chronology, arithmetic, and regex rule evaluation.
9. Strict no-fallback behavior.
10. Genuine LLM provenance preservation and rejection of untrusted provenance.
11. Tenant isolation and exact schema-version resolution.
12. Canonical envelope structure and schema-version preservation.
"""

from datetime import datetime, timezone
import pytest

from doc_extractor.contracts_b import (
    BoundingBox,
    CheckboxMark,
    ClassificationOutput,
    ConfidenceProvenance,
    LayoutBlock,
    LayoutElementType,
    ModuleBOutputPayload,
    MVPClassificationDomain,
    MVPDocumentType,
    PageLayoutOutput,
)
from doc_extractor.contracts_c import (
    CanonicalExtractionEnvelope,
    ExecutionMode,
    ExecutionProvenance,
)
from doc_extractor.dynamic_adapter import DynamicExtractionAdapter
from doc_extractor.exceptions import LLMExecutionError
from doc_extractor.extractor import SchemaGuidedExtractor
from schema_lifecycle.enums import FieldType, RuleOnFail, SchemaStatus
from schema_lifecycle.exceptions import (
    InvalidStateTransitionError,
    SchemaNotFoundError,
    TenantAccessError,
)
from schema_lifecycle.manager import SchemaLifecycleManager
from schema_lifecycle.metamodel import FieldDefinition, RuleDefinition, SchemaDefinition
from schema_lifecycle.repository import InMemorySchemaRepository


@pytest.fixture
def lifecycle_manager() -> SchemaLifecycleManager:
    repo = InMemorySchemaRepository()
    return SchemaLifecycleManager(repository=repo)


@pytest.fixture
def sample_payload() -> ModuleBOutputPayload:
    cb_opt_in = CheckboxMark(
        mark_id="cb_opt_in",
        state="[x]",
        ocr_confidence=0.99,
        bbox_preprocessed=BoundingBox(x0=10, y0=200, x1=30, y1=220),
        bbox_original=BoundingBox(x0=10, y0=200, x1=30, y1=220),
    )
    cb_paper = CheckboxMark(
        mark_id="cb_paper",
        state="[ ]",
        ocr_confidence=0.98,
        bbox_preprocessed=BoundingBox(x0=10, y0=230, x1=30, y1=250),
        bbox_original=BoundingBox(x0=10, y0=230, x1=30, y1=250),
    )
    b_name = LayoutBlock(
        block_id="b_name",
        element_type=LayoutElementType.KEY_VALUE_PAIR,
        text="Applicant Name: ANANYA SHARMA",
        ocr_confidence=0.98,
        bbox_preprocessed=BoundingBox(x0=10, y0=50, x1=200, y1=70),
        bbox_original=BoundingBox(x0=10, y0=50, x1=200, y1=70),
        reading_order_index=1,
    )
    b_city = LayoutBlock(
        block_id="b_city",
        element_type=LayoutElementType.KEY_VALUE_PAIR,
        text="City: BENGALURU",
        ocr_confidence=0.97,
        bbox_preprocessed=BoundingBox(x0=10, y0=80, x1=150, y1=100),
        bbox_original=BoundingBox(x0=10, y0=80, x1=150, y1=100),
        reading_order_index=2,
    )
    b_nom1 = LayoutBlock(
        block_id="b_nom1",
        element_type=LayoutElementType.PARAGRAPH,
        text="Nominee: ROHIT SHARMA (Brother) 60%",
        ocr_confidence=0.99,
        bbox_preprocessed=BoundingBox(x0=10, y0=110, x1=250, y1=130),
        bbox_original=BoundingBox(x0=10, y0=110, x1=250, y1=130),
        reading_order_index=3,
    )
    b_nom2 = LayoutBlock(
        block_id="b_nom2",
        element_type=LayoutElementType.PARAGRAPH,
        text="Nominee: KAVITA SHARMA (Mother) 40%",
        ocr_confidence=0.99,
        bbox_preprocessed=BoundingBox(x0=10, y0=140, x1=250, y1=160),
        bbox_original=BoundingBox(x0=10, y0=140, x1=250, y1=160),
        reading_order_index=4,
    )
    b_dates = LayoutBlock(
        block_id="b_dates",
        element_type=LayoutElementType.KEY_VALUE_PAIR,
        text="Admission Date: 2026-05-10\nDischarge Date: 2026-05-15",
        ocr_confidence=0.99,
        bbox_preprocessed=BoundingBox(x0=10, y0=170, x1=200, y1=190),
        bbox_original=BoundingBox(x0=10, y0=170, x1=200, y1=190),
        reading_order_index=5,
    )
    b_amounts = LayoutBlock(
        block_id="b_amounts",
        element_type=LayoutElementType.KEY_VALUE_PAIR,
        text="Room Rent: 5000\nPharmacy: 3000\nTotal Bill: 8000",
        ocr_confidence=0.99,
        bbox_preprocessed=BoundingBox(x0=10, y0=260, x1=200, y1=290),
        bbox_original=BoundingBox(x0=10, y0=260, x1=200, y1=290),
        reading_order_index=6,
    )
    b_pan = LayoutBlock(
        block_id="b_pan",
        element_type=LayoutElementType.KEY_VALUE_PAIR,
        text="PAN: ABCDE1234F",
        ocr_confidence=0.99,
        bbox_preprocessed=BoundingBox(x0=10, y0=300, x1=150, y1=320),
        bbox_original=BoundingBox(x0=10, y0=300, x1=150, y1=320),
        reading_order_index=7,
    )
    page = PageLayoutOutput(
        page_number=1,
        markdown_content="Applicant Name: ANANYA SHARMA\nCity: BENGALURU\nPAN: ABCDE1234F",
        blocks=[b_name, b_city, b_nom1, b_nom2, b_dates, b_amounts, b_pan],
        checkboxes=[cb_opt_in, cb_paper],
        mean_ocr_confidence=0.98,
    )
    return ModuleBOutputPayload(
        run_id="run_dyn_test_001",
        document_id="doc_dyn_001",
        tenant_id="tenant_alpha",
        status="processing",
        classification=ClassificationOutput(
            domain=MVPClassificationDomain.BFSI,
            document_type=MVPDocumentType.DOC_01_CASA,
            classification_confidence=0.98,
            confidence_provenance=ConfidenceProvenance.RULE_HEURISTIC,
            schema_id="kyc_dynamic_schema",
            schema_version="1",
        ),
        pages=[page],
        combined_markdown=page.markdown_content,
        parser_model_name="mock-layout-parser",
        classifier_model_name="mock-classifier",
        processing_time_ms=60.0,
        created_at=datetime.now(timezone.utc).isoformat(),
        completed_at=datetime.now(timezone.utc).isoformat(),
    )


def build_test_schema_def(schema_id: str = "kyc_dynamic_schema", version: int = 1) -> SchemaDefinition:
    return SchemaDefinition(
        schema_id=schema_id,
        version=version,
        tenant_id="tenant_alpha",
        domain="BFSI",
        status=SchemaStatus.DRAFT,
        fields=[
            FieldDefinition(name="applicant.full_name", type=FieldType.STRING, required=True),
            FieldDefinition(name="applicant.pan_number", type=FieldType.STRING, required=False, format="PAN"),
            FieldDefinition(name="applicant.address.city", type=FieldType.STRING, required=False),
            FieldDefinition(name="nominees[].name", type=FieldType.STRING, required=False),
            FieldDefinition(name="nominees[].share_percentage", type=FieldType.DECIMAL, required=False),
            FieldDefinition(name="services.opt_in_online", type=FieldType.BOOLEAN, required=False),
            FieldDefinition(name="services.paper_statements", type=FieldType.BOOLEAN, required=False),
            FieldDefinition(name="dates.admission_date", type=FieldType.DATE, required=False),
            FieldDefinition(name="dates.discharge_date", type=FieldType.DATE, required=False),
            FieldDefinition(name="billing.items[].amount", type=FieldType.DECIMAL, required=False),
            FieldDefinition(name="billing.total_amount", type=FieldType.DECIMAL, required=False),
            FieldDefinition(name="optional_missing_field", type=FieldType.STRING, required=False),
            FieldDefinition(name="required_missing_field", type=FieldType.STRING, required=True, nullable=True),
        ],
        rules=[
            RuleDefinition(
                id="CHRON_01",
                expression="dates.admission_date <= dates.discharge_date",
                target_fields=["dates.admission_date", "dates.discharge_date"],
                error_code="CHRONOLOGY_MISMATCH",
                on_fail=RuleOnFail.REVIEW,
            ),
            RuleDefinition(
                id="ARITH_01",
                expression="sum(billing.items[].amount) == billing.total_amount",
                target_fields=["billing.items[].amount", "billing.total_amount"],
                error_code="ARITHMETIC_MISMATCH",
                on_fail=RuleOnFail.REVIEW,
            ),
            RuleDefinition(
                id="PAN_01",
                expression="regex:^[A-Z]{5}[0-9]{4}[A-Z]$",
                target_fields=["applicant.pan_number"],
                error_code="INVALID_FORMAT",
                on_fail=RuleOnFail.REVIEW,
            ),
        ],
    )


# 1. End-to-end extraction using a TESTING-bound schema
def test_end_to_end_extraction_testing_schema(lifecycle_manager, sample_payload):
    schema_def = build_test_schema_def()
    lifecycle_manager.create_schema("tenant_alpha", schema_def, author="alice")
    lifecycle_manager.transition_status("tenant_alpha", "kyc_dynamic_schema", 1, SchemaStatus.TESTING, actor="alice")

    def mock_caller(payload, schema_cls):
        # Return dict matching compiled schema structure
        data = {
            "applicant": {
                "full_name": "ANANYA SHARMA",
                "pan_number": "ABCDE1234F",
                "address": {"city": "BENGALURU"},
            },
            "nominees": [
                {"name": "ROHIT SHARMA", "share_percentage": 60.0},
                {"name": "KAVITA SHARMA", "share_percentage": 40.0},
            ],
            "services": {
                "opt_in_online": True,
                "paper_statements": False,
            },
            "dates": {
                "admission_date": "2026-05-10",
                "discharge_date": "2026-05-15",
            },
            "billing": {
                "items": [{"amount": 5000.0}, {"amount": 3000.0}],
                "total_amount": 8000.0,
            },
            "optional_missing_field": None,
            "required_missing_field": "PROVIDED_REQUIRED_VALUE",
        }
        return data, "req_end_to_end_123"

    extractor = SchemaGuidedExtractor(
        model_name="Qwen2.5-7B-Instruct",
        use_llm=True,
        allow_fallback=False,
        llm_caller=mock_caller,
    )
    adapter = DynamicExtractionAdapter(manager=lifecycle_manager, extractor=extractor)

    envelope = adapter.extract_and_assemble(
        tenant_id="tenant_alpha",
        schema_id="kyc_dynamic_schema",
        version=1,
        payload=sample_payload,
    )

    assert isinstance(envelope, CanonicalExtractionEnvelope)
    assert envelope.document_metadata.schema_id == "kyc_dynamic_schema"
    assert envelope.document_metadata.schema_version == 1
    assert envelope.run.provenance.mode == ExecutionMode.GENUINE_LLM
    assert envelope.run.provenance.request_id == "req_end_to_end_123"
    assert "applicant.full_name" in envelope.fields
    assert envelope.fields["applicant.full_name"].value == "ANANYA SHARMA"
    assert envelope.fields["applicant.full_name"].evidence is not None


# 2. Rejection of DRAFT, APPROVED, and RETIRED schemas
def test_rejection_of_non_testing_schemas(lifecycle_manager, sample_payload):
    schema_def = build_test_schema_def()
    lifecycle_manager.create_schema("tenant_alpha", schema_def, author="alice")

    adapter = DynamicExtractionAdapter(manager=lifecycle_manager)

    # 1. DRAFT rejection
    with pytest.raises(InvalidStateTransitionError) as exc_draft:
        adapter.extract_and_assemble("tenant_alpha", "kyc_dynamic_schema", 1, sample_payload)
    assert "draft" in str(exc_draft.value).lower()

    # 2. Transition to TESTING -> then APPROVED
    lifecycle_manager.transition_status("tenant_alpha", "kyc_dynamic_schema", 1, SchemaStatus.TESTING, actor="alice")
    lifecycle_manager.internal_promote_to_approved("tenant_alpha", "kyc_dynamic_schema", 1, approver="bob")

    with pytest.raises(InvalidStateTransitionError) as exc_app:
        adapter.extract_and_assemble("tenant_alpha", "kyc_dynamic_schema", 1, sample_payload)
    assert "approved" in str(exc_app.value).lower()

    # 3. Transition to RETIRED
    lifecycle_manager.transition_status("tenant_alpha", "kyc_dynamic_schema", 1, SchemaStatus.RETIRED, actor="ops")
    with pytest.raises(InvalidStateTransitionError) as exc_ret:
        adapter.extract_and_assemble("tenant_alpha", "kyc_dynamic_schema", 1, sample_payload)
    assert "retired" in str(exc_ret.value).lower()

    # 4. Unknown schema rejection
    with pytest.raises(SchemaNotFoundError):
        adapter.extract_and_assemble("tenant_alpha", "non_existent_schema", 1, sample_payload)

    # 5. Cross-tenant rejection
    with pytest.raises(SchemaNotFoundError):
        adapter.extract_and_assemble("other_tenant", "kyc_dynamic_schema", 1, sample_payload)


# 3. Nested object field paths preservation
def test_nested_object_field_paths(lifecycle_manager, sample_payload):
    schema_def = build_test_schema_def()
    lifecycle_manager.create_schema("tenant_alpha", schema_def, author="alice")
    lifecycle_manager.transition_status("tenant_alpha", "kyc_dynamic_schema", 1, SchemaStatus.TESTING, actor="alice")

    def mock_caller(payload, schema_cls):
        return {
            "applicant": {
                "full_name": "ANANYA SHARMA",
                "address": {"city": "BENGALURU"},
            },
            "required_missing_field": "VALUE",
        }, "req_nested"

    extractor = SchemaGuidedExtractor(use_llm=True, allow_fallback=False, llm_caller=mock_caller)
    adapter = DynamicExtractionAdapter(manager=lifecycle_manager, extractor=extractor)

    envelope = adapter.extract_and_assemble("tenant_alpha", "kyc_dynamic_schema", 1, sample_payload)

    assert "applicant.full_name" in envelope.fields
    assert "applicant.address.city" in envelope.fields
    assert envelope.fields["applicant.address.city"].value == "BENGALURU"
    assert envelope.fields["applicant.address.city"].evidence.text == "City: BENGALURU"


# 4. Repeating groups, nested arrays, and stable ordering
def test_repeating_groups_nested_arrays(lifecycle_manager, sample_payload):
    schema_def = build_test_schema_def()
    lifecycle_manager.create_schema("tenant_alpha", schema_def, author="alice")
    lifecycle_manager.transition_status("tenant_alpha", "kyc_dynamic_schema", 1, SchemaStatus.TESTING, actor="alice")

    def mock_caller(payload, schema_cls):
        return {
            "nominees": [
                {"name": "ROHIT SHARMA", "share_percentage": 60.0},
                {"name": "KAVITA SHARMA", "share_percentage": 40.0},
            ],
            "billing": {
                "items": [{"amount": 5000.0}, {"amount": 3000.0}],
                "total_amount": 8000.0,
            },
            "required_missing_field": "VAL",
        }, "req_arrays"

    extractor = SchemaGuidedExtractor(use_llm=True, allow_fallback=False, llm_caller=mock_caller)
    adapter = DynamicExtractionAdapter(manager=lifecycle_manager, extractor=extractor)

    envelope = adapter.extract_and_assemble("tenant_alpha", "kyc_dynamic_schema", 1, sample_payload)

    assert "nominees.0.name" in envelope.fields
    assert envelope.fields["nominees.0.name"].value == "ROHIT SHARMA"
    assert "nominees.0.share_percentage" in envelope.fields
    assert envelope.fields["nominees.0.share_percentage"].value == 60.0

    assert "nominees.1.name" in envelope.fields
    assert envelope.fields["nominees.1.name"].value == "KAVITA SHARMA"
    assert "nominees.1.share_percentage" in envelope.fields
    assert envelope.fields["nominees.1.share_percentage"].value == 40.0

    # Stable array indices in nested objects
    assert "billing.items.0.amount" in envelope.fields
    assert envelope.fields["billing.items.0.amount"].value == 5000.0
    assert "billing.items.1.amount" in envelope.fields
    assert envelope.fields["billing.items.1.amount"].value == 3000.0


# 5. Checkbox checked/unchecked evidence grounding and strict evidence requirement
def test_checkbox_evidence_grounding_and_strict_requirement(lifecycle_manager, sample_payload):
    schema_def = build_test_schema_def()
    lifecycle_manager.create_schema("tenant_alpha", schema_def, author="alice")
    lifecycle_manager.transition_status("tenant_alpha", "kyc_dynamic_schema", 1, SchemaStatus.TESTING, actor="alice")

    def mock_caller(payload, schema_cls):
        return {
            "services": {
                "opt_in_online": True,      # [x] present on page
                "paper_statements": False,   # [ ] present on page
            },
            "required_missing_field": "VAL",
        }, "req_cb"

    extractor = SchemaGuidedExtractor(use_llm=True, allow_fallback=False, llm_caller=mock_caller)
    adapter = DynamicExtractionAdapter(manager=lifecycle_manager, extractor=extractor)

    envelope = adapter.extract_and_assemble("tenant_alpha", "kyc_dynamic_schema", 1, sample_payload)

    # Checked checkbox evidence
    opt_in_field = envelope.fields["services.opt_in_online"]
    assert opt_in_field.value is True
    assert opt_in_field.evidence is not None
    assert opt_in_field.evidence.text == "[x]"

    # Unchecked checkbox evidence
    paper_field = envelope.fields["services.paper_statements"]
    assert paper_field.value is False
    assert paper_field.evidence is not None
    assert paper_field.evidence.text == "[ ]"

    # Now test a boolean with NO grounded checkbox in layout
    def mock_caller_ungrounded_cb(payload, schema_cls):
        return {
            "services": {
                "opt_in_online": True,
                "paper_statements": True,  # Page only has '[ ]', not '[x]' for paper_statements
            },
            "required_missing_field": "VAL",
        }, "req_cb_ungrounded"

    extractor2 = SchemaGuidedExtractor(use_llm=True, allow_fallback=False, llm_caller=mock_caller_ungrounded_cb)
    adapter2 = DynamicExtractionAdapter(manager=lifecycle_manager, extractor=extractor2)
    envelope2 = adapter2.extract_and_assemble("tenant_alpha", "kyc_dynamic_schema", 1, sample_payload)

    # Strict boolean evidence enforcement: ungrounded boolean is forced to null with EVIDENCE_MISSING flag
    ungrounded_field = envelope2.fields["services.paper_statements"]
    assert ungrounded_field.value is None
    assert ungrounded_field.review_required is True
    assert ungrounded_field.validation == "failed"
    assert any(flag.code == "EVIDENCE_MISSING" for flag in envelope2.flags)


# 6. Required and optional null handling
def test_required_and_optional_null_handling(lifecycle_manager, sample_payload):
    schema_def = build_test_schema_def()
    lifecycle_manager.create_schema("tenant_alpha", schema_def, author="alice")
    lifecycle_manager.transition_status("tenant_alpha", "kyc_dynamic_schema", 1, SchemaStatus.TESTING, actor="alice")

    def mock_caller(payload, schema_cls):
        # required_missing_field is None
        return {
            "applicant": {"full_name": "ANANYA SHARMA"},
            "optional_missing_field": None,
            "required_missing_field": None,
        }, "req_nulls"

    extractor = SchemaGuidedExtractor(use_llm=True, allow_fallback=False, llm_caller=mock_caller)
    adapter = DynamicExtractionAdapter(manager=lifecycle_manager, extractor=extractor)

    envelope = adapter.extract_and_assemble("tenant_alpha", "kyc_dynamic_schema", 1, sample_payload)

    # Optional missing field
    opt_field = envelope.fields["optional_missing_field"]
    assert opt_field.value is None
    assert opt_field.null_reason == "absent"
    assert opt_field.review_required is False
    assert opt_field.validation == "not_applicable"

    # Required missing field
    req_field = envelope.fields["required_missing_field"]
    assert req_field.value is None
    assert req_field.null_reason == "absent"
    assert req_field.review_required is True
    assert req_field.validation == "failed"

    # Status must route to needs_review due to required field failure
    assert envelope.document_metadata.status == "needs_review"


# 7. Correct absent versus illegible null reasons
def test_absent_vs_illegible_null_reasons(lifecycle_manager, sample_payload):
    schema_def = build_test_schema_def()
    lifecycle_manager.create_schema("tenant_alpha", schema_def, author="alice")
    lifecycle_manager.transition_status("tenant_alpha", "kyc_dynamic_schema", 1, SchemaStatus.TESTING, actor="alice")

    # Add a degraded block to page 1 indicating illegible PAN
    sample_payload.pages[0].blocks.append(
        LayoutBlock(
            block_id="b_illegible_pan",
            element_type=LayoutElementType.KEY_VALUE_PAIR,
            text="PAN: [illegible smudged]",
            ocr_confidence=0.20,
            bbox_preprocessed=BoundingBox(x0=10, y0=400, x1=150, y1=420),
            bbox_original=BoundingBox(x0=10, y0=400, x1=150, y1=420),
            reading_order_index=10,
        )
    )

    def mock_caller(payload, schema_cls):
        return {
            "applicant": {"full_name": "ANANYA SHARMA", "pan_number": None},
            "optional_missing_field": None,
            "required_missing_field": "VAL",
        }, "req_illegible"

    extractor = SchemaGuidedExtractor(use_llm=True, allow_fallback=False, llm_caller=mock_caller)
    adapter = DynamicExtractionAdapter(manager=lifecycle_manager, extractor=extractor)

    envelope = adapter.extract_and_assemble("tenant_alpha", "kyc_dynamic_schema", 1, sample_payload)

    # PAN field has an illegible signal in layout block
    pan_field = envelope.fields["applicant.pan_number"]
    assert pan_field.value is None
    assert pan_field.null_reason == "illegible"
    assert pan_field.review_required is True
    assert pan_field.validation == "failed"

    # Purely absent field without text mention
    opt_field = envelope.fields["optional_missing_field"]
    assert opt_field.value is None
    assert opt_field.null_reason == "absent"
    assert opt_field.review_required is False


# 8. Dynamic chronology, arithmetic, and regex rule evaluation
def test_dynamic_rules_evaluation(lifecycle_manager, sample_payload):
    schema_def = build_test_schema_def()
    lifecycle_manager.create_schema("tenant_alpha", schema_def, author="alice")
    lifecycle_manager.transition_status("tenant_alpha", "kyc_dynamic_schema", 1, SchemaStatus.TESTING, actor="alice")

    def mock_caller_rule_failures(payload, schema_cls):
        return {
            "applicant": {
                "full_name": "ANANYA SHARMA",
                "pan_number": "INVALID_PAN_123",  # Violates PAN regex
            },
            "dates": {
                "admission_date": "2026-05-20",
                "discharge_date": "2026-05-10",  # Inverted: violates chronology
            },
            "billing": {
                "items": [{"amount": 5000.0}, {"amount": 3000.0}],
                "total_amount": 9999.0,  # Sum is 8000 != 9999, violates arithmetic
            },
            "required_missing_field": "VAL",
        }, "req_rules"

    extractor = SchemaGuidedExtractor(use_llm=True, allow_fallback=False, llm_caller=mock_caller_rule_failures)
    adapter = DynamicExtractionAdapter(manager=lifecycle_manager, extractor=extractor)

    envelope = adapter.extract_and_assemble("tenant_alpha", "kyc_dynamic_schema", 1, sample_payload)

    # 1. Chronology mismatch flag
    assert any(flag.code == "CHRONOLOGY_MISMATCH" for flag in envelope.flags)
    assert envelope.fields["dates.admission_date"].review_required is True
    assert envelope.fields["dates.discharge_date"].review_required is True

    # 2. Arithmetic mismatch flag
    assert any(flag.code == "ARITHMETIC_MISMATCH" for flag in envelope.flags)
    assert envelope.fields["billing.total_amount"].review_required is True

    # 3. PAN format validation failure
    assert envelope.fields["applicant.pan_number"].validation == "failed"
    assert envelope.fields["applicant.pan_number"].review_required is True

    # Status must be needs_review
    assert envelope.document_metadata.status == "needs_review"


# 9. Strict no-fallback behavior
def test_strict_no_fallback_behavior(lifecycle_manager, sample_payload):
    schema_def = build_test_schema_def()
    lifecycle_manager.create_schema("tenant_alpha", schema_def, author="alice")
    lifecycle_manager.transition_status("tenant_alpha", "kyc_dynamic_schema", 1, SchemaStatus.TESTING, actor="alice")

    def failing_caller(payload, schema_cls):
        raise ConnectionResetError("Provider network connection reset")

    extractor = SchemaGuidedExtractor(
        model_name="Qwen2.5-7B-Instruct",
        use_llm=True,
        allow_fallback=False,
        llm_caller=failing_caller,
    )
    adapter = DynamicExtractionAdapter(manager=lifecycle_manager, extractor=extractor)

    with pytest.raises(LLMExecutionError) as exc_info:
        adapter.extract_and_assemble("tenant_alpha", "kyc_dynamic_schema", 1, sample_payload)

    assert "Provider network connection reset" in str(exc_info.value)


# 10. Genuine LLM provenance preservation and rejection of untrusted provenance
def test_provenance_preservation_and_untrusted_rejection(lifecycle_manager, sample_payload):
    schema_def = build_test_schema_def()
    lifecycle_manager.create_schema("tenant_alpha", schema_def, author="alice")
    lifecycle_manager.transition_status("tenant_alpha", "kyc_dynamic_schema", 1, SchemaStatus.TESTING, actor="alice")

    # Mock mode rejected when allow_mock=False
    extractor_mock = SchemaGuidedExtractor(is_mock=True)
    adapter_strict = DynamicExtractionAdapter(
        manager=lifecycle_manager,
        extractor=extractor_mock,
        allow_mock=False,
    )

    with pytest.raises(LLMExecutionError) as exc_info:
        adapter_strict.extract_and_assemble("tenant_alpha", "kyc_dynamic_schema", 1, sample_payload)
    assert "untrusted provenance mode" in str(exc_info.value).lower()

    # Mock mode accepted when allow_mock=True
    adapter_mock_allowed = DynamicExtractionAdapter(
        manager=lifecycle_manager,
        extractor=extractor_mock,
        allow_mock=True,
    )
    envelope = adapter_mock_allowed.extract_and_assemble("tenant_alpha", "kyc_dynamic_schema", 1, sample_payload)
    assert envelope.run.provenance.mode == ExecutionMode.MOCK


# 11. Tenant isolation and exact schema-version resolution
def test_tenant_isolation_and_exact_version_resolution(lifecycle_manager, sample_payload):
    schema_v1 = build_test_schema_def(version=1)
    schema_v2 = build_test_schema_def(version=2)
    schema_v2.fields.append(FieldDefinition(name="extra_v2_field", type=FieldType.STRING, required=False))

    lifecycle_manager.create_schema("tenant_alpha", schema_v1, author="alice")
    lifecycle_manager.transition_status("tenant_alpha", "kyc_dynamic_schema", 1, SchemaStatus.TESTING, actor="alice")
    lifecycle_manager.create_version("tenant_alpha", "kyc_dynamic_schema", schema_v2, author="alice")
    lifecycle_manager.transition_status("tenant_alpha", "kyc_dynamic_schema", 2, SchemaStatus.TESTING, actor="alice")

    def mock_caller(payload, schema_cls):
        return {"applicant": {"full_name": "ANANYA"}, "required_missing_field": "VAL"}, "req_exact"

    extractor = SchemaGuidedExtractor(use_llm=True, allow_fallback=False, llm_caller=mock_caller)
    adapter = DynamicExtractionAdapter(manager=lifecycle_manager, extractor=extractor)

    # Resolve version 1
    env1 = adapter.extract_and_assemble("tenant_alpha", "kyc_dynamic_schema", 1, sample_payload)
    assert env1.document_metadata.schema_version == 1
    assert "extra_v2_field" not in env1.fields

    # Resolve version 2
    env2 = adapter.extract_and_assemble("tenant_alpha", "kyc_dynamic_schema", 2, sample_payload)
    assert env2.document_metadata.schema_version == 2
    assert "extra_v2_field" in env2.fields


# 12. Canonical envelope structure and schema-version preservation
def test_canonical_envelope_structure_preservation(lifecycle_manager, sample_payload):
    schema_def = build_test_schema_def()
    lifecycle_manager.create_schema("tenant_alpha", schema_def, author="alice")
    lifecycle_manager.transition_status("tenant_alpha", "kyc_dynamic_schema", 1, SchemaStatus.TESTING, actor="alice")

    def mock_caller(payload, schema_cls):
        return {"applicant": {"full_name": "ANANYA SHARMA"}, "required_missing_field": "OK"}, "req_env"

    extractor = SchemaGuidedExtractor(use_llm=True, allow_fallback=False, llm_caller=mock_caller)
    adapter = DynamicExtractionAdapter(manager=lifecycle_manager, extractor=extractor)

    envelope = adapter.extract_and_assemble("tenant_alpha", "kyc_dynamic_schema", 1, sample_payload)

    assert envelope.document_id == "doc_dyn_001"
    assert envelope.document_metadata.schema_id == "kyc_dynamic_schema"
    assert envelope.document_metadata.schema_version == 1
    assert envelope.document_metadata.document_type == "DOC-01"
    assert envelope.document_metadata.page_count == 1
    assert envelope.run.run_id == "run_dyn_test_001"
    assert envelope.run.prompt_version == "1.0.0"
    assert envelope.run.provenance.mode == ExecutionMode.GENUINE_LLM
