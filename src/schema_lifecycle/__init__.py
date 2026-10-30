"""
Schema Lifecycle Package (SCH-001, SCH-002, SCH-003).
Metamodel definitions, validation, explicit version adapters, and lossless YAML/JSON serializers.
Strictly adheres to PRD v1.1 §6.1, §6.2, §6.3.
"""

from schema_lifecycle.enums import (
    FieldType,
    PIIClass,
    RuleOnFail,
    SchemaStatus,
)
from schema_lifecycle.exceptions import (
    SchemaLifecycleError,
    SchemaSerializationError,
    SchemaValidationError,
    SchemaVersionError,
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
from schema_lifecycle.adapters import (
    SchemaVersionAdapter,
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

