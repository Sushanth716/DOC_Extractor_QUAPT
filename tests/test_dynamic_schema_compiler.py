"""
Unit Tests for DynamicSchemaCompiler (TASK-4.3 Stage 2A).
Covers PRD Requirements:
- SCH-001, SCH-002, SCH-003, SCH-011.
- All eight field types: string, integer, decimal, boolean, date, enum, object, array.
- Required vs optional property semantics separately from nullable values.
- Decimal precision preservation and datetime.date ISO handling.
- Enum restrictions and regex format constraints.
- Absence and null fidelity: non-conversion to default objects or empty lists.
- Deterministic, collision-resistant model and enum naming.
- Canonical JSON Schema Draft 2020-12 generation and validation.
- Isolation between schemas, versions, and nested models.
"""

from datetime import date
from decimal import Decimal
import json
import pytest
from pydantic import ValidationError
import jsonschema

from schema_lifecycle.compiler import (
    CompiledFieldMetadata,
    DynamicSchemaCompiler,
)
from schema_lifecycle.enums import FieldType, PIIClass, RuleOnFail, SchemaStatus
from schema_lifecycle.exceptions import SchemaValidationError
from schema_lifecycle.metamodel import (
    FieldDefinition,
    RuleDefinition,
    SchemaDefinition,
)


# ============================================================================
# 1. All Eight Field Types
# ============================================================================

def test_all_eight_field_types_compilation_and_validation():
    """Verify compilation and validation across all eight supported metamodel types."""
    schema = SchemaDefinition(
        schema_id="all_types_schema",
        version=1,
        status=SchemaStatus.TESTING,
        fields=[
            FieldDefinition(name="str_field", type=FieldType.STRING),
            FieldDefinition(name="int_field", type=FieldType.INTEGER),
            FieldDefinition(name="dec_field", type=FieldType.DECIMAL),
            FieldDefinition(name="bool_field", type=FieldType.BOOLEAN),
            FieldDefinition(name="date_field", type=FieldType.DATE),
            FieldDefinition(name="enum_field", type=FieldType.ENUM, allowed_values=["OPT_A", "OPT_B"]),
            FieldDefinition(
                name="obj_field",
                type=FieldType.OBJECT,
                fields=[
                    FieldDefinition(name="inner_str", type=FieldType.STRING),
                ],
            ),
            FieldDefinition(name="arr_field", type=FieldType.ARRAY, item_type=FieldType.STRING),
        ],
    )

    Model = DynamicSchemaCompiler.compile(schema)
    assert Model.__name__ == "AllTypesSchema_V1_Model"
    assert len(Model.__field_metadata__) == 9  # 8 fields + 1 child in obj_field

    # Valid instantiation with all types
    inst = Model.model_validate({
        "str_field": "test string",
        "int_field": 42,
        "dec_field": "1234.5678",
        "bool_field": True,
        "date_field": "2026-10-29",
        "enum_field": "OPT_A",
        "obj_field": {"inner_str": "nested"},
        "arr_field": ["alpha", "beta"],
    })

    assert inst.str_field == "test string"
    assert inst.int_field == 42
    assert isinstance(inst.dec_field, Decimal)
    assert inst.dec_field == Decimal("1234.5678")
    assert inst.bool_field is True
    assert isinstance(inst.date_field, date)
    assert inst.date_field == date(2026, 10, 29)
    assert inst.enum_field.value == "OPT_A"
    assert inst.obj_field.inner_str == "nested"
    assert inst.arr_field == ["alpha", "beta"]


# ============================================================================
# 2. Decimal Precision Preservation
# ============================================================================

def test_decimal_precision_preservation():
    """Verify that Decimal fields retain exact precision without floating-point drift."""
    schema = SchemaDefinition(
        schema_id="precision_schema",
        version=1,
        fields=[
            FieldDefinition(name="exact_amount", type=FieldType.DECIMAL, required=True),
        ],
    )
    Model = DynamicSchemaCompiler.compile(schema)

    # 18-decimal-place monetary string
    high_precision_str = "987654321012345678.123456789012345678"
    inst = Model.model_validate({"exact_amount": high_precision_str})

    assert isinstance(inst.exact_amount, Decimal)
    assert str(inst.exact_amount) == high_precision_str
    # Contrast with float conversion which loses lower digits
    assert float(high_precision_str) != inst.exact_amount

    # Also accept Decimal directly
    inst_dec = Model.model_validate({"exact_amount": Decimal(high_precision_str)})
    assert inst_dec.exact_amount == Decimal(high_precision_str)

    # Invalid decimal string fails
    with pytest.raises(ValidationError):
        Model.model_validate({"exact_amount": "not_a_valid_decimal"})


# ============================================================================
# 3. ISO Date Handling
# ============================================================================

def test_date_iso_handling_and_validation():
    """Verify that date fields require ISO 8601 formatting and convert to datetime.date."""
    schema = SchemaDefinition(
        schema_id="date_schema",
        version=1,
        fields=[
            FieldDefinition(name="event_date", type=FieldType.DATE),
        ],
    )
    Model = DynamicSchemaCompiler.compile(schema)

    # Valid ISO date
    inst = Model.model_validate({"event_date": "2026-10-31"})
    assert isinstance(inst.event_date, date)
    assert inst.event_date == date(2026, 10, 31)

    # Direct date object
    inst_obj = Model.model_validate({"event_date": date(2026, 10, 31)})
    assert inst_obj.event_date == date(2026, 10, 31)

    # Invalid non-date string fails
    with pytest.raises(ValidationError):
        Model.model_validate({"event_date": "invalid-date-string"})

    # Malformed ISO month/day fails
    with pytest.raises(ValidationError):
        Model.model_validate({"event_date": "2026-13-45"})


# ============================================================================
# 4. Enum Restrictions & Allowed Values
# ============================================================================

def test_enum_restrictions_and_allowed_values():
    """Verify that enum fields restrict input strictly to declared allowed values."""
    schema = SchemaDefinition(
        schema_id="enum_schema",
        version=1,
        fields=[
            FieldDefinition(
                name="account_type",
                type=FieldType.ENUM,
                allowed_values=["SAVINGS", "CURRENT", "SALARY"],
                required=True,
            ),
        ],
    )
    Model = DynamicSchemaCompiler.compile(schema)

    # Valid choices
    for choice in ["SAVINGS", "CURRENT", "SALARY"]:
        inst = Model.model_validate({"account_type": choice})
        assert inst.account_type.value == choice

    # Invalid choice rejected
    with pytest.raises(ValidationError):
        Model.model_validate({"account_type": "INVESTMENT"})

    with pytest.raises(ValidationError):
        Model.model_validate({"account_type": "savings"})  # case-sensitive

    # Enum class is attached and clean
    meta = Model.__field_metadata__["account_type"]
    assert meta.allowed_values == ["SAVINGS", "CURRENT", "SALARY"]


# ============================================================================
# 5. Regex & Format Constraints
# ============================================================================

def test_regex_format_constraint_validation():
    """Verify that regex format constraints on string fields enforce pattern matching."""
    schema = SchemaDefinition(
        schema_id="regex_schema",
        version=1,
        fields=[
            FieldDefinition(
                name="pan",
                type=FieldType.STRING,
                format="^[A-Z]{5}[0-9]{4}[A-Z]$",
                required=True,
            ),
        ],
    )
    Model = DynamicSchemaCompiler.compile(schema)

    # Valid PAN
    inst = Model.model_validate({"pan": "ABCDE1234F"})
    assert inst.pan == "ABCDE1234F"

    # Invalid PAN format fails
    with pytest.raises(ValidationError) as exc_info:
        Model.model_validate({"pan": "12345ABCDE"})
    assert "pattern" in str(exc_info.value).lower() or "match" in str(exc_info.value).lower()


# ============================================================================
# 6. Required vs Optional Property Semantics
# ============================================================================

def test_required_versus_optional_properties():
    """
    PRD SCH-001: Required properties must be provided in payload.
    Optional properties may be omitted.
    """
    schema = SchemaDefinition(
        schema_id="req_opt_schema",
        version=1,
        fields=[
            FieldDefinition(name="mandatory_field", type=FieldType.STRING, required=True),
            FieldDefinition(name="optional_field", type=FieldType.STRING, required=False),
        ],
    )
    Model = DynamicSchemaCompiler.compile(schema)

    # Both provided
    inst1 = Model.model_validate({"mandatory_field": "val1", "optional_field": "val2"})
    assert inst1.mandatory_field == "val1"
    assert inst1.optional_field == "val2"

    # Only mandatory provided (optional omitted) -> valid!
    inst2 = Model.model_validate({"mandatory_field": "val1"})
    assert inst2.mandatory_field == "val1"
    assert inst2.optional_field is None

    # Mandatory omitted -> fails validation!
    with pytest.raises(ValidationError) as exc_info:
        Model.model_validate({"optional_field": "val2"})
    assert "mandatory_field" in str(exc_info.value)


# ============================================================================
# 7. Explicit Null vs Omitted Properties & Nullable Semantics
# ============================================================================

def test_explicit_null_versus_omitted_properties():
    """
    Verify all 4 combinations:
    1. required=True, nullable=False (mandatory, cannot be null)
    2. required=True, nullable=True (mandatory key, but value can be null)
    3. required=False, nullable=True (optional key, can be null or omitted)
    4. required=False, nullable=False (optional key, can be omitted, but if passed cannot be null)
    """
    schema = SchemaDefinition(
        schema_id="null_matrix_schema",
        version=1,
        fields=[
            FieldDefinition(name="req_not_null", type=FieldType.STRING, required=True, nullable=False),
            FieldDefinition(name="req_null", type=FieldType.STRING, required=True, nullable=True),
            FieldDefinition(name="opt_null", type=FieldType.STRING, required=False, nullable=True),
            FieldDefinition(name="opt_not_null", type=FieldType.STRING, required=False, nullable=False),
        ],
    )
    Model = DynamicSchemaCompiler.compile(schema)

    # 1. req_not_null: omitted -> fail; null -> fail; value -> pass
    with pytest.raises(ValidationError):
        Model.model_validate({"req_null": "b"})  # omitted
    with pytest.raises(ValidationError):
        Model.model_validate({"req_not_null": None, "req_null": "b"})  # explicit null
    inst_req_not_null = Model.model_validate({"req_not_null": "a", "req_null": None})
    assert inst_req_not_null.req_not_null == "a"
    assert inst_req_not_null.req_null is None  # req_null is nullable!

    # 2. req_null: omitted -> fail; null -> pass; value -> pass
    with pytest.raises(ValidationError):
        Model.model_validate({"req_not_null": "a"})  # req_null omitted
    inst_req_null = Model.model_validate({"req_not_null": "a", "req_null": None})
    assert inst_req_null.req_null is None

    # 3. opt_null: omitted -> pass (None); null -> pass (None); value -> pass
    inst_opt_omitted = Model.model_validate({"req_not_null": "a", "req_null": "b"})
    assert inst_opt_omitted.opt_null is None
    inst_opt_null = Model.model_validate({"req_not_null": "a", "req_null": "b", "opt_null": None})
    assert inst_opt_null.opt_null is None
    inst_opt_val = Model.model_validate({"req_not_null": "a", "req_null": "b", "opt_null": "c"})
    assert inst_opt_val.opt_null == "c"

    # 4. opt_not_null: omitted -> pass (None); null -> fail; value -> pass
    inst_opt_not_null_omitted = Model.model_validate({"req_not_null": "a", "req_null": "b"})
    assert inst_opt_not_null_omitted.opt_not_null is None
    with pytest.raises(ValidationError) as exc:
        Model.model_validate({"req_not_null": "a", "req_null": "b", "opt_not_null": None})
    assert "cannot be null" in str(exc.value).lower()
    inst_opt_not_null_val = Model.model_validate({"req_not_null": "a", "req_null": "b", "opt_not_null": "d"})
    assert inst_opt_not_null_val.opt_not_null == "d"


# ============================================================================
# 8. Absence & Null Fidelity (Non-conversion to Defaults)
# ============================================================================

def test_absent_and_null_not_converted_to_empty_collections_or_default_objects():
    """
    CRITICAL REQUIREMENT 7:
    Do not silently convert absent or null values into empty arrays or default objects.
    Omitting an array or object must leave it as None, not [] or SubModel().
    """
    schema = SchemaDefinition(
        schema_id="fidelity_schema",
        version=1,
        fields=[
            FieldDefinition(
                name="applicant",
                type=FieldType.OBJECT,
                fields=[
                    FieldDefinition(name="full_name", type=FieldType.STRING),
                ],
            ),
            FieldDefinition(
                name="nominees",
                type=FieldType.ARRAY,
                items=FieldDefinition(
                    name="nominee",
                    type=FieldType.OBJECT,
                    fields=[FieldDefinition(name="name", type=FieldType.STRING)],
                ),
            ),
            FieldDefinition(name="tags", type=FieldType.ARRAY, item_type=FieldType.STRING),
        ],
    )
    Model = DynamicSchemaCompiler.compile(schema)

    # 1. Omitted payload ({})
    inst_empty = Model.model_validate({})
    assert inst_empty.applicant is None   # NOT Applicant()
    assert inst_empty.nominees is None    # NOT []
    assert inst_empty.tags is None        # NOT []

    # 2. Explicit null payload
    inst_nulls = Model.model_validate({"applicant": None, "nominees": None, "tags": None})
    assert inst_nulls.applicant is None
    assert inst_nulls.nominees is None
    assert inst_nulls.tags is None

    # 3. Explicit empty list provided in payload
    inst_empty_list = Model.model_validate({"nominees": [], "tags": []})
    assert inst_empty_list.nominees == []
    assert inst_empty_list.tags == []


# ============================================================================
# 9. Nested Objects Multilevel Compilation
# ============================================================================

def test_nested_objects_multilevel_compilation():
    """Verify compilation of 3+ levels of nested objects."""
    schema = SchemaDefinition(
        schema_id="nested_multilevel",
        version=2,
        fields=[
            FieldDefinition(
                name="level1",
                type=FieldType.OBJECT,
                fields=[
                    FieldDefinition(
                        name="level2",
                        type=FieldType.OBJECT,
                        fields=[
                            FieldDefinition(
                                name="level3",
                                type=FieldType.OBJECT,
                                fields=[
                                    FieldDefinition(name="leaf_val", type=FieldType.INTEGER),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )
    Model = DynamicSchemaCompiler.compile(schema)
    assert Model.__name__ == "NestedMultilevel_V2_Model"

    inst = Model.model_validate({
        "level1": {
            "level2": {
                "level3": {
                    "leaf_val": 999,
                },
            },
        },
    })
    assert inst.level1.level2.level3.leaf_val == 999


# ============================================================================
# 10. Scalar and Object Arrays & Repeating Groups Ordering
# ============================================================================

def test_scalar_arrays_and_repeating_groups_ordering():
    """Verify scalar arrays and repeating groups preserve exact element ordering."""
    schema = SchemaDefinition(
        schema_id="arrays_schema",
        version=1,
        fields=[
            FieldDefinition(name="scores", type=FieldType.ARRAY, item_type=FieldType.INTEGER),
            FieldDefinition(
                name="items",
                type=FieldType.ARRAY,
                items=FieldDefinition(
                    name="item",
                    type=FieldType.OBJECT,
                    fields=[
                        FieldDefinition(name="id", type=FieldType.INTEGER),
                        FieldDefinition(name="label", type=FieldType.STRING),
                    ],
                ),
            ),
        ],
    )
    Model = DynamicSchemaCompiler.compile(schema)

    scores_data = [10, 50, 30, 90, 20]
    items_data = [
        {"id": 1, "label": "first"},
        {"id": 2, "label": "second"},
        {"id": 3, "label": "third"},
    ]

    inst = Model.model_validate({"scores": scores_data, "items": items_data})

    # Exact preserved ordering
    assert inst.scores == scores_data
    assert [it.id for it in inst.items] == [1, 2, 3]
    assert [it.label for it in inst.items] == ["first", "second", "third"]


# ============================================================================
# 11. Nested Model Name Collision Resistance
# ============================================================================

def test_nested_model_name_collision_resistance():
    """
    Verify that identically named child objects under different branches
    generate distinct, collision-resistant model class names.
    """
    schema = SchemaDefinition(
        schema_id="collision_test",
        version=1,
        fields=[
            # applicant has an 'address' object
            FieldDefinition(
                name="applicant",
                type=FieldType.OBJECT,
                fields=[
                    FieldDefinition(
                        name="address",
                        type=FieldType.OBJECT,
                        fields=[FieldDefinition(name="city", type=FieldType.STRING)],
                    ),
                ],
            ),
            # nominee ALSO has an 'address' object
            FieldDefinition(
                name="nominee",
                type=FieldType.OBJECT,
                fields=[
                    FieldDefinition(
                        name="address",
                        type=FieldType.OBJECT,
                        fields=[FieldDefinition(name="pin", type=FieldType.STRING)],
                    ),
                ],
            ),
        ],
    )
    Model = DynamicSchemaCompiler.compile(schema)

    # Verify both child models exist with distinct names
    applicant_addr_cls = Model.model_fields["applicant"].annotation
    # Unwrap Optional if present
    if hasattr(applicant_addr_cls, "__args__"):
        applicant_cls = [a for a in applicant_addr_cls.__args__ if a is not type(None)][0]
    else:
        applicant_cls = applicant_addr_cls

    nominee_addr_cls = Model.model_fields["nominee"].annotation
    if hasattr(nominee_addr_cls, "__args__"):
        nominee_cls = [a for a in nominee_addr_cls.__args__ if a is not type(None)][0]
    else:
        nominee_cls = nominee_addr_cls

    app_addr = applicant_cls.model_fields["address"].annotation
    if hasattr(app_addr, "__args__"):
        app_addr_cls = [a for a in app_addr.__args__ if a is not type(None)][0]
    else:
        app_addr_cls = app_addr

    nom_addr = nominee_cls.model_fields["address"].annotation
    if hasattr(nom_addr, "__args__"):
        nom_addr_cls = [a for a in nom_addr.__args__ if a is not type(None)][0]
    else:
        nom_addr_cls = nom_addr

    assert app_addr_cls.__name__ == "CollisionTest_V1_Applicant_Address_Model"
    assert nom_addr_cls.__name__ == "CollisionTest_V1_Nominee_Address_Model"
    assert app_addr_cls is not nom_addr_cls


# ============================================================================
# 12. Deterministic Compilation & Identity/Version Isolation
# ============================================================================

def test_deterministic_compilation_and_isolation():
    """Verify deterministic compilation for identical definitions and isolation between versions."""
    fields = [FieldDefinition(name="field_x", type=FieldType.STRING)]

    s1_v1 = SchemaDefinition(schema_id="schema_alpha", version=1, fields=fields)
    s1_v1_dup = SchemaDefinition(schema_id="schema_alpha", version=1, fields=fields)
    s1_v2 = SchemaDefinition(schema_id="schema_alpha", version=2, fields=fields)
    s2_v1 = SchemaDefinition(schema_id="schema_beta", version=1, fields=fields)

    m1_a = DynamicSchemaCompiler.compile(s1_v1)
    m1_b = DynamicSchemaCompiler.compile(s1_v1_dup)
    m2 = DynamicSchemaCompiler.compile(s1_v2)
    m3 = DynamicSchemaCompiler.compile(s2_v1)

    # Identical definitions yield identical class names and JSON schemas
    assert m1_a.__name__ == "SchemaAlpha_V1_Model"
    assert m1_b.__name__ == "SchemaAlpha_V1_Model"
    schema_a = DynamicSchemaCompiler.compile_json_schema(m1_a)
    schema_b = DynamicSchemaCompiler.compile_json_schema(m1_b)
    assert schema_a == schema_b

    # Version isolation
    assert m2.__name__ == "SchemaAlpha_V2_Model"
    assert m2.__name__ != m1_a.__name__

    # Identity isolation
    assert m3.__name__ == "SchemaBeta_V1_Model"
    assert m3.__name__ != m1_a.__name__


# ============================================================================
# 13. Canonical JSON Schema Dialect & Structure Validation
# ============================================================================

def test_canonical_json_schema_dialect_and_structure_validation():
    """Verify canonical JSON Schema Draft 2020-12 declaration and validation."""
    schema = SchemaDefinition(
        schema_id="json_schema_test",
        version=1,
        fields=[
            FieldDefinition(name="username", type=FieldType.STRING, required=True),
            FieldDefinition(name="age", type=FieldType.INTEGER),
            FieldDefinition(name="balance", type=FieldType.DECIMAL),
            FieldDefinition(name="active", type=FieldType.BOOLEAN),
            FieldDefinition(name="registered_on", type=FieldType.DATE),
            FieldDefinition(name="role", type=FieldType.ENUM, allowed_values=["ADMIN", "USER"]),
        ],
    )
    json_schema = DynamicSchemaCompiler.compile_json_schema(schema)

    # 1. Dialect declaration
    assert json_schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert json_schema["title"] == "JsonSchemaTest_V1_Model"
    assert json_schema["type"] == "object"

    # 2. Required array contains exactly username
    assert json_schema["required"] == ["username"]

    # 3. Validated by official jsonschema Draft202012Validator
    jsonschema.Draft202012Validator.check_schema(json_schema)


# ============================================================================
# 14. Metadata Preservation: Paths, Evidence, Thresholds, PII, Rules
# ============================================================================

def test_metadata_preservation():
    """Verify metadata preservation on compiled models (paths, thresholds, evidence, PII, rules)."""
    rule1 = RuleDefinition(id="R1", expression="balance >= 0", on_fail=RuleOnFail.REVIEW, target_fields=["balance"])
    schema = SchemaDefinition(
        schema_id="meta_schema",
        version=1,
        fields=[
            FieldDefinition(
                name="applicant.pan",
                type=FieldType.STRING,
                format="^[A-Z]{5}[0-9]{4}[A-Z]$",
                required=True,
                evidence_required=True,
                confidence_threshold=0.90,
                pii_class=PIIClass.GOVERNMENT_ID,
            ),
            FieldDefinition(
                name="balance",
                type=FieldType.DECIMAL,
                confidence_threshold=0.80,
                evidence_required=True,
            ),
        ],
        rules=[rule1],
    )
    Model = DynamicSchemaCompiler.compile(schema)

    # Check model-level attachments
    assert Model.__schema_id__ == "meta_schema"
    assert Model.__schema_version__ == 1
    assert Model.__schema_definition__ is schema
    assert len(Model.__rules__) == 1

    # Check field metadata registry
    meta_pan = Model.__field_metadata__["applicant.pan"]
    assert meta_pan.field_path == "applicant.pan"
    assert meta_pan.field_type == FieldType.STRING
    assert meta_pan.required is True
    assert meta_pan.evidence_required is True
    assert meta_pan.confidence_threshold == 0.90
    assert meta_pan.pii_class == PIIClass.GOVERNMENT_ID
    assert meta_pan.format == "^[A-Z]{5}[0-9]{4}[A-Z]$"

    meta_bal = Model.__field_metadata__["balance"]
    assert meta_bal.confidence_threshold == 0.80
    assert len(meta_bal.rules) == 1
    assert meta_bal.rules[0].id == "R1"


# ============================================================================
# 15. Unflatten and Flatten Utilities
# ============================================================================

def test_unflatten_and_flatten_utilities():
    """Verify bidirectional unflattening and flattening between dot-paths and models."""
    schema = SchemaDefinition(
        schema_id="flat_test",
        version=1,
        fields=[
            FieldDefinition(name="applicant.full_name", type=FieldType.STRING),
            FieldDefinition(name="applicant.pan", type=FieldType.STRING),
            FieldDefinition(
                name="nominees",
                type=FieldType.ARRAY,
                items=FieldDefinition(
                    name="nominee",
                    type=FieldType.OBJECT,
                    fields=[
                        FieldDefinition(name="name", type=FieldType.STRING),
                        FieldDefinition(name="share", type=FieldType.DECIMAL),
                    ],
                ),
            ),
        ],
    )
    Model = DynamicSchemaCompiler.compile(schema)

    flat_input = {
        "applicant.full_name": "SARAH CONNOR",
        "applicant.pan": "ABCDE1234F",
        "nominees.0.name": "JOHN CONNOR",
        "nominees.0.share": "100.00",
    }

    # 1. Unflatten to nested dict
    nested = DynamicSchemaCompiler.unflatten(flat_input)
    assert nested == {
        "applicant": {"full_name": "SARAH CONNOR", "pan": "ABCDE1234F"},
        "nominees": [{"name": "JOHN CONNOR", "share": "100.00"}],
    }

    # 2. Validate model using nested dict
    inst = Model.model_validate(nested)
    assert inst.applicant.full_name == "SARAH CONNOR"
    assert inst.nominees[0].share == Decimal("100.00")

    # 3. Flatten model instance back to dot-notation
    re_flattened = DynamicSchemaCompiler.flatten(inst)
    assert re_flattened["applicant.full_name"] == "SARAH CONNOR"
    assert re_flattened["applicant.pan"] == "ABCDE1234F"
    assert re_flattened["nominees.0.name"] == "JOHN CONNOR"
    assert re_flattened["nominees.0.share"] == Decimal("100.00")


# ============================================================================
# 16. Error Handling & Validation
# ============================================================================

def test_invalid_schema_compilation_rejection():
    """Verify rejection of invalid compilation arguments."""
    with pytest.raises(SchemaValidationError, match="Expected SchemaDefinition"):
        DynamicSchemaCompiler.compile("not_a_schema")  # type: ignore[arg-type]

    with pytest.raises(SchemaValidationError, match="Expected SchemaDefinition or compiled BaseModel"):
        DynamicSchemaCompiler.compile_json_schema(12345)  # type: ignore[arg-type]

