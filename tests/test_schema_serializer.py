"""
Comprehensive Unit Tests for Schema Serializer and Lossless Round-Trip (SCH-003).
Validates:
- Import/Export of YAML and JSON.
- Safe YAML parsing (injection attack defense).
- Rejection of malformed syntax, non-mapping roots, invalid types.
- Semantic round-trip equality without silently dropping properties.
- Full compatibility with PRD v1.1 §6.2 illustrative CASA schema.
- Full coverage of launch schemas (DOC-01, DOC-02 repeating group, DOC-03).
- Error envelope standard compliance (API-010, API-011).
"""

import pytest
import yaml

from schema_lifecycle.enums import FieldType, PIIClass, RuleOnFail, SchemaStatus
from schema_lifecycle.exceptions import (
    SchemaLifecycleError,
    SchemaSerializationError,
    SchemaValidationError,
)
from schema_lifecycle.metamodel import (
    FieldDefinition,
    RuleDefinition,
    SchemaDefinition,
)
from schema_lifecycle.serializer import (
    safe_dump_json,
    safe_dump_yaml,
    safe_load_json,
    safe_load_yaml,
    schema_from_dict,
    schema_from_json,
    schema_from_yaml,
    schema_to_dict,
    schema_to_json,
    schema_to_yaml,
)


# ============================================================================
# PRD §6.2 Illustrative Schema Round-Trip
# ============================================================================

PRD_6_2_YAML = """
schema_id: casa_account_opening
version: 3
status: approved
domain: BFSI
fields:
  - name: applicant.full_name
    type: string
    required: true
    description: Applicant name, merged from character boxes, uppercase as printed
    pii_class: direct_identifier
    confidence_threshold: 0.90
  - name: applicant.pan_card_number
    type: string
    required: false
    format: '^[A-Z]{5}[0-9]{4}[A-Z]$'
    pii_class: government_id
  - name: account_details.initial_deposit_amount
    type: decimal
    currency: INR
  - name: account_details.opt_in_services.cheque_book
    type: boolean
    source: checkbox
rules:
  - id: R1
    expression: applicant.date_of_birth <= today - 18y
    on_fail: review
"""


def test_prd_6_2_schema_import_and_round_trip():
    """SCH-003: Direct validation against PRD §6.2 example schema."""
    schema = SchemaDefinition.from_yaml(PRD_6_2_YAML)

    # Core attributes
    assert schema.schema_id == "casa_account_opening"
    assert schema.version == 3
    assert schema.status == SchemaStatus.APPROVED
    assert schema.domain == "BFSI"
    assert len(schema.fields) == 4
    assert len(schema.rules) == 1

    # Field properties
    f0 = schema.fields[0]
    assert f0.name == "applicant.full_name"
    assert f0.type == FieldType.STRING
    assert f0.required is True
    assert f0.pii_class == PIIClass.DIRECT_IDENTIFIER
    assert f0.confidence_threshold == 0.90

    f1 = schema.fields[1]
    assert f1.name == "applicant.pan_card_number"
    assert f1.required is False
    assert f1.format == "^[A-Z]{5}[0-9]{4}[A-Z]$"
    assert f1.pii_class == PIIClass.GOVERNMENT_ID

    # Extra non-standard schema properties preserved (currency, source)
    f2 = schema.fields[2]
    assert f2.type == FieldType.DECIMAL
    assert getattr(f2, "currency", None) == "INR"

    f3 = schema.fields[3]
    assert f3.type == FieldType.BOOLEAN
    assert getattr(f3, "source", None) == "checkbox"

    # Rule attributes
    r0 = schema.rules[0]
    assert r0.id == "R1"
    assert r0.expression == "applicant.date_of_birth <= today - 18y"
    assert r0.on_fail == RuleOnFail.REVIEW

    # Export to YAML and re-import (SCH-003 round-trip acceptance criteria)
    exported_yaml = schema.to_yaml()
    reimported = SchemaDefinition.from_yaml(exported_yaml)

    assert schema == reimported
    assert schema.to_dict() == reimported.to_dict()
    assert reimported.to_yaml() == exported_yaml

    # Export to JSON and re-import
    exported_json = schema.to_json()
    reimported_json = SchemaDefinition.from_json(exported_json)

    assert schema == reimported_json
    assert schema.to_dict() == reimported_json.to_dict()


# ============================================================================
# Launch Schemas Round-Trip Coverage
# ============================================================================

def test_doc_02_insurance_repeating_groups_round_trip():
    """SCH-002, SCH-003: DOC-02 Insurance schema with repeating nominees group."""
    doc_02_yaml = """
schema_id: insurance_application
version: 1
status: approved
domain: BFSI
fields:
  - name: proposer_name
    type: string
    required: true
    pii_class: direct_identifier
  - name: policy_term_years
    type: integer
    required: true
  - name: sum_insured
    type: decimal
    currency: INR
  - name: nominees[]
    type: array
    description: Repeating list of application nominees
  - name: nominees[].name
    type: string
    required: true
    pii_class: direct_identifier
  - name: nominees[].relationship
    type: string
  - name: nominees[].age
    type: integer
  - name: nominees[].share_percentage
    type: decimal
  - name: medical_declaration.has_diabetes
    type: boolean
    source: checkbox
rules:
  - id: R_NOMINEE_SUM
    expression: sum(nominees[].share_percentage) == 100
    on_fail: review
    error_code: ARITHMETIC_MISMATCH
"""
    schema = schema_from_yaml(doc_02_yaml)
    assert schema.schema_id == "insurance_application"
    assert len(schema.fields) == 9
    assert len(schema.rules) == 1

    # Round trip YAML
    exported_yaml = schema_to_yaml(schema)
    reimported = schema_from_yaml(exported_yaml)
    assert schema == reimported
    assert schema.to_dict() == reimported.to_dict()

    # Round trip JSON
    exported_json = schema_to_json(schema)
    reimported_json = schema_from_json(exported_json)
    assert schema == reimported_json


def test_doc_03_discharge_summary_round_trip():
    """SCH-001, SCH-003: DOC-03 Hospital discharge summary schema."""
    doc_03_yaml = """
schema_id: hospital_discharge_summary
version: 1
status: approved
domain: Healthcare
fields:
  - name: patient_name
    type: string
    required: true
    pii_class: direct_identifier
  - name: ip_number
    type: string
    required: true
  - name: admission_date
    type: date
    required: true
  - name: discharge_date
    type: date
    required: true
  - name: diagnosis
    type: string
    pii_class: health_information
  - name: medications[]
    type: array
    description: Prescribed discharge medications
  - name: medications[].medicine_name
    type: string
    required: true
  - name: medications[].dosage
    type: string
  - name: medications[].frequency
    type: string
  - name: medications[].duration
    type: string
rules:
  - id: R_CHRONOLOGY
    expression: discharge_date >= admission_date
    on_fail: review
    error_code: CHRONOLOGY_ERROR
"""
    schema = schema_from_yaml(doc_03_yaml)
    assert schema.domain == "Healthcare"
    assert len(schema.fields) == 10

    # Lossless round-trip verification
    exported_yaml = schema.to_yaml()
    reimported = SchemaDefinition.from_yaml(exported_yaml)
    assert schema == reimported
    assert schema.to_dict() == reimported.to_dict()


# ============================================================================
# Preservation of Extra Properties (No Silently Dropped Properties)
# ============================================================================

def test_preserves_arbitrary_metadata_round_trip():
    """Ensure custom metadata at schema, field, and rule levels is preserved (SCH-003)."""
    schema_dict = {
        "schema_id": "custom_schema",
        "version": 2,
        "status": "draft",
        "custom_schema_tag": "internal-audit-v1",
        "author": "compliance_officer_42",
        "fields": [
            {
                "name": "tax_id",
                "type": "string",
                "custom_field_meta": {"mask_pattern": "XXX-XX-####", "encryption": "AES256"},
                "ui_widget": "masked_input",
            }
        ],
        "rules": [
            {
                "id": "R_CUSTOM",
                "expression": "len(tax_id) == 10",
                "on_fail": "reject",
                "custom_rule_param": 10,
            }
        ],
    }

    schema = SchemaDefinition.from_dict(schema_dict)
    assert getattr(schema, "custom_schema_tag", None) == "internal-audit-v1"
    assert getattr(schema, "author", None) == "compliance_officer_42"
    assert getattr(schema.fields[0], "ui_widget", None) == "masked_input"
    assert getattr(schema.rules[0], "custom_rule_param", None) == 10

    # YAML round trip
    yaml_str = schema.to_yaml()
    reloaded = SchemaDefinition.from_yaml(yaml_str)
    assert reloaded == schema
    assert getattr(reloaded, "custom_schema_tag", None) == "internal-audit-v1"
    assert getattr(reloaded.fields[0], "ui_widget", None) == "masked_input"

    # JSON round trip
    json_str = schema.to_json()
    reloaded_json = SchemaDefinition.from_json(json_str)
    assert reloaded_json == schema


# ============================================================================
# Safe YAML Parsing & Security Checks
# ============================================================================

def test_safe_load_yaml_blocks_injection():
    """Security check: YAML constructor injection must be rejected safely."""
    malicious_yaml = """
schema_id: exploit
version: 1
fields:
  - name: payload
    type: !!python/object/apply:os.system ["echo compromised"]
"""
    with pytest.raises(SchemaSerializationError) as exc_info:
        safe_load_yaml(malicious_yaml)
    assert "Failed to parse YAML syntax" in str(exc_info.value) or "yaml" in exc_info.value.format_type


def test_safe_load_yaml_rejects_non_mapping():
    """YAML root must be a mapping (dictionary), not a list or scalar."""
    with pytest.raises(SchemaSerializationError) as exc_info:
        safe_load_yaml("- item 1\n- item 2")
    assert "must be a mapping (dictionary)" in str(exc_info.value)

    with pytest.raises(SchemaSerializationError):
        safe_load_yaml("just a bare string")

    with pytest.raises(SchemaSerializationError):
        safe_load_yaml("12345")


def test_safe_load_yaml_rejects_non_string():
    """Input to safe_load_yaml must be string."""
    with pytest.raises(SchemaSerializationError):
        safe_load_yaml(12345)  # type: ignore


def test_safe_load_json_rejects_malformed_and_non_object():
    """JSON safe loading rejects invalid syntax and non-dict roots."""
    with pytest.raises(SchemaSerializationError):
        safe_load_json("{malformed json")

    with pytest.raises(SchemaSerializationError):
        safe_load_json('["list", "not", "object"]')

    with pytest.raises(SchemaSerializationError):
        safe_load_json(None)  # type: ignore


# ============================================================================
# Schema Validation Failure Handling on Import
# ============================================================================

def test_import_yaml_missing_field_type_raises_schema_validation_error():
    """SCH-001: Missing field type on YAML import raises SchemaValidationError."""
    invalid_yaml = """
schema_id: invalid_schema
version: 1
status: draft
fields:
  - name: broken_field
    # type is missing
"""
    with pytest.raises(SchemaValidationError) as exc_info:
        SchemaDefinition.from_yaml(invalid_yaml)
    assert "Field 'type' is required" in str(exc_info.value) or "validation failed" in str(exc_info.value)


def test_import_yaml_invalid_field_type_raises_schema_validation_error():
    """SCH-001: Invalid field type on YAML import raises SchemaValidationError."""
    invalid_yaml = """
schema_id: invalid_schema
version: 1
status: draft
fields:
  - name: broken_field
    type: unsupported_unknown_type
"""
    with pytest.raises(SchemaValidationError) as exc_info:
        SchemaDefinition.from_yaml(invalid_yaml)
    assert "Invalid field type" in str(exc_info.value) or "validation failed" in str(exc_info.value)


def test_import_yaml_out_of_range_threshold_raises_schema_validation_error():
    """Confidence threshold out of range on YAML import raises SchemaValidationError."""
    invalid_yaml = """
schema_id: invalid_schema
version: 1
status: draft
fields:
  - name: threshold_field
    type: string
    confidence_threshold: 1.5
"""
    with pytest.raises(SchemaValidationError) as exc_info:
        SchemaDefinition.from_yaml(invalid_yaml)
    assert "between 0.0 and 1.0" in str(exc_info.value) or "validation failed" in str(exc_info.value)


def test_error_envelope_conformance():
    """API-010: Exceptions format properly into standard error envelope."""
    err = SchemaValidationError(
        message="Type missing for field applicant.pan",
        errors=[{"loc": ["fields", 0, "type"], "msg": "Field required"}],
    )
    envelope = err.to_envelope()
    assert "error" in envelope
    assert envelope["error"]["code"] == "SCHEMA_VALIDATION_ERROR"
    assert envelope["error"]["message"] == "Type missing for field applicant.pan"
    assert "validation_errors" in envelope["error"]["details"]

