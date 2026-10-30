"""
Unit Tests for Schema Metamodel (SCH-001, SCH-002).
Validates:
- All PRD field types: string, integer, decimal, boolean, date, enum, object, array.
- Required vs optional status.
- Allowed values enforcement (especially for enum).
- Format and regex validation.
- Evidence requirements (default True).
- Confidence thresholds (range [0.0, 1.0]).
- PII classifications.
- Rejection of invalid types, missing types, duplicate names, out-of-range thresholds.
- Nested objects and repeating groups (nominees[]).
- Canonical integer schema versions and explicit adapters.
"""

import pytest
from pydantic import ValidationError

from schema_lifecycle.enums import FieldType, PIIClass, RuleOnFail, SchemaStatus
from schema_lifecycle.exceptions import (
    SchemaValidationError,
    SchemaVersionError,
)
from schema_lifecycle.metamodel import (
    FieldDefinition,
    RuleDefinition,
    SchemaDefinition,
)
from schema_lifecycle.adapters import SchemaVersionAdapter
from doc_extractor.contracts_b import MVPDocumentType


# ============================================================================
# SCH-001: Field Definition Metamodel Tests
# ============================================================================

def test_field_definition_valid_all_types():
    """Verify all 8 PRD field types can be validly instantiated (SCH-001)."""
    types = [
        (FieldType.STRING, None),
        (FieldType.INTEGER, None),
        (FieldType.DECIMAL, None),
        (FieldType.BOOLEAN, None),
        (FieldType.DATE, None),
        (FieldType.ENUM, ["SAVINGS", "CURRENT", "SALARY"]),
        (FieldType.OBJECT, None),
        (FieldType.ARRAY, None),
    ]

    for f_type, allowed in types:
        f = FieldDefinition(
            name=f"test_{f_type.value}",
            type=f_type,
            allowed_values=allowed,
        )
        assert f.type == f_type
        assert f.evidence_required is True  # Default True per SCH-001
        assert f.required is False          # Default False


def test_field_definition_missing_type_fails():
    """SCH-001: A schema missing 'type' for any field fails validation."""
    with pytest.raises(ValidationError) as exc_info:
        FieldDefinition(name="pan_number")  # type missing
    assert "type" in str(exc_info.value)


def test_field_definition_invalid_type_fails():
    """SCH-001: Unsupported field types must be rejected."""
    with pytest.raises(ValidationError) as exc_info:
        FieldDefinition(name="amount", type="unsupported_type")
    assert "Invalid field type" in str(exc_info.value) or "Supported types" in str(exc_info.value)


def test_field_definition_enum_requires_allowed_values():
    """SCH-001: Type enum without allowed_values fails validation."""
    with pytest.raises(ValidationError) as exc_info:
        FieldDefinition(name="account_type", type=FieldType.ENUM)
    assert "requires non-empty 'allowed_values'" in str(exc_info.value)

    with pytest.raises(ValidationError) as exc_info:
        FieldDefinition(name="account_type", type=FieldType.ENUM, allowed_values=[])
    assert "requires non-empty 'allowed_values'" in str(exc_info.value)


def test_field_definition_confidence_threshold_range():
    """SCH-001: Confidence threshold must be in range [0.0, 1.0]."""
    # Valid boundaries
    f0 = FieldDefinition(name="f0", type=FieldType.STRING, confidence_threshold=0.0)
    assert f0.confidence_threshold == 0.0
    f1 = FieldDefinition(name="f1", type=FieldType.STRING, confidence_threshold=1.0)
    assert f1.confidence_threshold == 1.0
    f_mid = FieldDefinition(name="f_mid", type=FieldType.STRING, confidence_threshold=0.85)
    assert f_mid.confidence_threshold == 0.85

    # Invalid negative
    with pytest.raises(ValidationError) as exc_info:
        FieldDefinition(name="f_neg", type=FieldType.STRING, confidence_threshold=-0.1)
    assert "between 0.0 and 1.0" in str(exc_info.value)

    # Invalid > 1.0
    with pytest.raises(ValidationError) as exc_info:
        FieldDefinition(name="f_high", type=FieldType.STRING, confidence_threshold=1.05)
    assert "between 0.0 and 1.0" in str(exc_info.value)


def test_field_definition_regex_format_validation():
    """SCH-001: Invalid regex pattern in format is rejected."""
    # Valid regex
    f = FieldDefinition(name="pan", type=FieldType.STRING, format="^[A-Z]{5}[0-9]{4}[A-Z]$")
    assert f.format == "^[A-Z]{5}[0-9]{4}[A-Z]$"

    # Invalid regex (unbalanced bracket)
    with pytest.raises(ValidationError) as exc_info:
        FieldDefinition(name="pan", type=FieldType.STRING, format="^[A-Z{5}$")
    assert "Invalid regex pattern" in str(exc_info.value)


def test_field_definition_pii_classification():
    """SCH-001, SEC-007: PII classifications are recognized and normalized."""
    f1 = FieldDefinition(name="pan", type=FieldType.STRING, pii_class=PIIClass.GOVERNMENT_ID)
    assert f1.pii_class == PIIClass.GOVERNMENT_ID

    # String input normalized to enum
    f2 = FieldDefinition(name="name", type=FieldType.STRING, pii_class="DIRECT_IDENTIFIER")
    assert f2.pii_class == PIIClass.DIRECT_IDENTIFIER

    # Custom PII class allowed as string
    f3 = FieldDefinition(name="custom_pii", type=FieldType.STRING, pii_class="custom_confidential")
    assert f3.pii_class == "custom_confidential"


def test_field_definition_empty_name_fails():
    """Field name cannot be empty or whitespace."""
    with pytest.raises(ValidationError):
        FieldDefinition(name="", type=FieldType.STRING)
    with pytest.raises(ValidationError):
        FieldDefinition(name="   ", type=FieldType.STRING)


# ============================================================================
# SCH-002: Nested Objects & Repeating Groups Tests
# ============================================================================

def test_nested_object_definition():
    """SCH-002: Supports nested objects with child fields."""
    applicant = FieldDefinition(
        name="applicant",
        type=FieldType.OBJECT,
        fields=[
            FieldDefinition(name="full_name", type=FieldType.STRING, required=True),
            FieldDefinition(name="pan_card_number", type=FieldType.STRING, format="^[A-Z]{5}[0-9]{4}[A-Z]$"),
        ],
    )
    assert applicant.is_object is True
    assert applicant.is_array is False
    assert len(applicant.fields) == 2


def test_nested_object_duplicate_child_name_rejected():
    """SCH-002: Duplicate child field names in an object are rejected."""
    with pytest.raises(ValidationError) as exc_info:
        FieldDefinition(
            name="applicant",
            type=FieldType.OBJECT,
            fields=[
                FieldDefinition(name="full_name", type=FieldType.STRING),
                FieldDefinition(name="full_name", type=FieldType.STRING),
            ],
        )
    assert "Duplicate child field name 'full_name'" in str(exc_info.value)


def test_repeating_groups_array_items():
    """SCH-002: Repeating groups with array items produce an array specification."""
    nominees = FieldDefinition(
        name="nominees",
        type=FieldType.ARRAY,
        description="List of nominees",
        items=FieldDefinition(
            name="nominee",
            type=FieldType.OBJECT,
            fields=[
                FieldDefinition(name="name", type=FieldType.STRING, required=True),
                FieldDefinition(name="relationship", type=FieldType.STRING),
                FieldDefinition(name="age", type=FieldType.INTEGER),
                FieldDefinition(name="share_percentage", type=FieldType.DECIMAL),
            ],
        ),
    )
    assert nominees.is_array is True
    assert nominees.is_repeating_group is True
    assert nominees.items.type == FieldType.OBJECT
    assert len(nominees.items.fields) == 4


def test_repeating_group_bracket_notation():
    """SCH-002 acceptance criteria: A schema with 'nominees[]' produces an array in the output."""
    nominee_arr = FieldDefinition(
        name="nominees[]",
        type=FieldType.ARRAY,
        description="Repeating nominees group",
    )
    assert nominee_arr.is_array is True
    assert nominee_arr.clean_name == "nominees"
    assert nominee_arr.name == "nominees[]"


# ============================================================================
# RuleDefinition Tests
# ============================================================================

def test_rule_definition_valid():
    """Validates RuleDefinition creation matching PRD §6.2 example."""
    rule = RuleDefinition(
        id="R1",
        expression="applicant.date_of_birth <= today - 18y",
        on_fail=RuleOnFail.REVIEW,
        target_fields=["applicant.date_of_birth"],
    )
    assert rule.id == "R1"
    assert rule.on_fail == RuleOnFail.REVIEW
    assert "18y" in rule.expression


def test_rule_definition_invalid_empty():
    """Empty rule id or expression fails validation."""
    with pytest.raises(ValidationError):
        RuleDefinition(id="", expression="x > 0")
    with pytest.raises(ValidationError):
        RuleDefinition(id="R1", expression="")


# ============================================================================
# SchemaDefinition Metamodel Integrity Tests
# ============================================================================

def test_schema_definition_valid():
    """Validates complete SchemaDefinition construction."""
    schema = SchemaDefinition(
        schema_id="casa_account_opening",
        version=3,
        status=SchemaStatus.APPROVED,
        domain="BFSI",
        fields=[
            FieldDefinition(name="applicant.full_name", type=FieldType.STRING, required=True),
            FieldDefinition(name="account_details.initial_deposit_amount", type=FieldType.DECIMAL),
        ],
        rules=[
            RuleDefinition(id="R1", expression="deposit >= 1000", on_fail="review"),
        ],
    )
    assert schema.schema_id == "casa_account_opening"
    assert schema.version == 3
    assert schema.status == SchemaStatus.APPROVED
    assert len(schema.fields) == 2
    assert len(schema.rules) == 1


def test_schema_definition_empty_fields_fails():
    """Schema must define at least one field."""
    with pytest.raises(ValidationError) as exc_info:
        SchemaDefinition(
            schema_id="empty_schema",
            version=1,
            fields=[],
        )
    assert "at least one field" in str(exc_info.value)


def test_schema_definition_duplicate_field_names_fails():
    """Duplicate top-level field names must be rejected."""
    with pytest.raises(ValidationError) as exc_info:
        SchemaDefinition(
            schema_id="dup_schema",
            version=1,
            fields=[
                FieldDefinition(name="field_a", type=FieldType.STRING),
                FieldDefinition(name="field_a", type=FieldType.INTEGER),
            ],
        )
    assert "Duplicate field name 'field_a'" in str(exc_info.value)


def test_schema_definition_duplicate_rule_ids_fails():
    """Duplicate rule IDs must be rejected."""
    with pytest.raises(ValidationError) as exc_info:
        SchemaDefinition(
            schema_id="dup_rule_schema",
            version=1,
            fields=[FieldDefinition(name="amount", type=FieldType.DECIMAL)],
            rules=[
                RuleDefinition(id="R1", expression="amount > 0"),
                RuleDefinition(id="R1", expression="amount < 1000000"),
            ],
        )
    assert "Duplicate rule id 'R1'" in str(exc_info.value)


def test_schema_canonical_integer_version_enforced():
    """Canonical integer version constraint: string versions rejected directly."""
    # Direct int version >= 1 succeeds
    s = SchemaDefinition(
        schema_id="s1",
        version=1,
        fields=[FieldDefinition(name="f", type=FieldType.STRING)],
    )
    assert s.version == 1

    # String version without adapter must be rejected
    with pytest.raises(ValidationError) as exc_info:
        SchemaDefinition(
            schema_id="s_str",
            version="1.0.0",  # String version
            fields=[FieldDefinition(name="f", type=FieldType.STRING)],
        )
    assert "Schema version must be a canonical integer" in str(exc_info.value)

    # Negative or zero integer rejected
    with pytest.raises(ValidationError):
        SchemaDefinition(
            schema_id="s_neg",
            version=0,
            fields=[FieldDefinition(name="f", type=FieldType.STRING)],
        )

    # Boolean rejected (since bool is subclass of int in Python)
    with pytest.raises(ValidationError):
        SchemaDefinition(
            schema_id="s_bool",
            version=True,
            fields=[FieldDefinition(name="f", type=FieldType.STRING)],
        )


# ============================================================================
# SchemaVersionAdapter Tests
# ============================================================================

def test_version_adapter_conversions():
    """Test SchemaVersionAdapter canonical integer and semver conversions."""
    assert SchemaVersionAdapter.to_canonical(1) == 1
    assert SchemaVersionAdapter.to_canonical("3") == 3
    assert SchemaVersionAdapter.to_canonical("1.0.0") == 1
    assert SchemaVersionAdapter.to_canonical("v2.4") == 2
    assert SchemaVersionAdapter.to_canonical("V4") == 4

    assert SchemaVersionAdapter.to_semver(1) == "1.0.0"
    assert SchemaVersionAdapter.to_semver(3) == "3.0.0"

    # Invalid version conversions
    with pytest.raises(SchemaVersionError):
        SchemaVersionAdapter.to_canonical("invalid")
    with pytest.raises(SchemaVersionError):
        SchemaVersionAdapter.to_canonical(0)
    with pytest.raises(SchemaVersionError):
        SchemaVersionAdapter.to_canonical(-5)


def test_version_adapter_legacy_dict():
    """Adapter constructs canonical SchemaDefinition from legacy string-version dictionary."""
    legacy_data = {
        "schema_id": "legacy_doc",
        "version": "1.0.0",  # Legacy SemVer string
        "status": "approved",
        "fields": [{"name": "title", "type": "string"}],
    }
    schema = SchemaVersionAdapter.from_legacy_dict(legacy_data)
    assert schema.version == 1
    assert schema.schema_id == "legacy_doc"


def test_version_adapter_approved_schema_ref_bridge():
    """Adapter bridges SchemaDefinition with doc_extractor ApprovedSchemaRef."""
    schema = SchemaDefinition(
        schema_id="DOC-01",
        version=2,
        status=SchemaStatus.APPROVED,
        fields=[FieldDefinition(name="account_number", type=FieldType.STRING)],
    )

    ref = SchemaVersionAdapter.to_approved_schema_ref(
        schema=schema,
        document_type=MVPDocumentType.DOC_01_CASA,
        is_active=True,
    )
    assert ref.schema_id == "DOC-01"
    assert ref.schema_version == "2.0.0"
    assert ref.document_type == MVPDocumentType.DOC_01_CASA
    assert ref.is_active is True

    # Round-trip back from ref
    schema_id, canonical_version = SchemaVersionAdapter.from_approved_schema_ref(ref)
    assert schema_id == "DOC-01"
    assert canonical_version == 2


# ============================================================================
# Field Tree and Query Helpers
# ============================================================================

def test_schema_get_field_and_tree():
    """Tests get_field and to_tree helpers for SCR-05 visual field tree."""
    schema = SchemaDefinition(
        schema_id="casa",
        version=1,
        fields=[
            FieldDefinition(name="applicant.full_name", type=FieldType.STRING),
            FieldDefinition(name="applicant.pan", type=FieldType.STRING),
            FieldDefinition(name="nominees[]", type=FieldType.ARRAY),
            FieldDefinition(name="nominees[].name", type=FieldType.STRING),
        ],
    )

    # get_field lookup
    assert schema.get_field("applicant.full_name") is not None
    assert schema.get_field("nominees[]") is not None
    assert schema.get_field("nominees") is not None  # clean_name lookup
    assert schema.get_field("non_existent") is None

    # to_tree produces hierarchical structure
    tree = schema.to_tree()
    assert "applicant" in tree
    assert tree["applicant"]["type"] == "object"
    assert "full_name" in tree["applicant"]["children"]
    assert "pan" in tree["applicant"]["children"]
    assert "nominees" in tree
    assert tree["nominees"]["type"] == "array"

