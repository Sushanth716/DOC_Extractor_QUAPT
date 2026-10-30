"""
Bridge module exposing schema_lifecycle package under doc_extractor namespace.
"""

from schema_lifecycle import (
    FieldDefinition,
    FieldType,
    PIIClass,
    RuleDefinition,
    RuleOnFail,
    SchemaDefinition,
    SchemaLifecycleError,
    SchemaSerializationError,
    SchemaStatus,
    SchemaValidationError,
    SchemaVersionAdapter,
    SchemaVersionError,
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

__all__ = [
    "SchemaStatus",
    "FieldType",
    "PIIClass",
    "RuleOnFail",
    "FieldDefinition",
    "RuleDefinition",
    "SchemaDefinition",
    "SchemaLifecycleError",
    "SchemaValidationError",
    "SchemaSerializationError",
    "SchemaVersionError",
    "safe_load_yaml",
    "safe_dump_yaml",
    "safe_load_json",
    "safe_dump_json",
    "schema_to_dict",
    "schema_from_dict",
    "schema_to_yaml",
    "schema_from_yaml",
    "schema_to_json",
    "schema_from_json",
    "SchemaVersionAdapter",
]

