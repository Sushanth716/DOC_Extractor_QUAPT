"""
Schema Lifecycle Package (SCH-001, SCH-002, SCH-003, SCH-004, SCH-006).
Metamodel definitions, validation, explicit version adapters, lossless YAML/JSON serializers,
state machine transition validation, immutable versioning, rollback, and repository abstractions.
Strictly adheres to PRD v1.1 §6.1, §6.2, §6.3, §7.1, SCH-004, and SCH-006.
"""

from schema_lifecycle.enums import (
    FieldType,
    PIIClass,
    RuleOnFail,
    SchemaStatus,
)
from schema_lifecycle.exceptions import (
    InvalidStateTransitionError,
    SchemaLifecycleError,
    SchemaNotApprovedError,
    SchemaNotFoundError,
    SchemaPromotionNotAllowedError,
    SchemaRollbackError,
    SchemaSerializationError,
    SchemaValidationError,
    SchemaVersionError,
    TenantAccessError,
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
from schema_lifecycle.diff import (
    compute_schema_diff,
)
from schema_lifecycle.events import (
    LifecycleEvent,
    LifecycleEventType,
)
from schema_lifecycle.state_machine import (
    SchemaStateMachine,
)
from schema_lifecycle.repository import (
    InMemorySchemaRepository,
    SchemaRepository,
    SchemaSnapshot,
)
from schema_lifecycle.manager import (
    SchemaLifecycleManager,
)
from schema_lifecycle.adapters import (
    LifecycleSchemaRegistryAdapter,
    SchemaVersionAdapter,
)

__all__ = [
    # Enums
    "SchemaStatus",
    "FieldType",
    "PIIClass",
    "RuleOnFail",
    "LifecycleEventType",
    # Metamodel
    "FieldDefinition",
    "RuleDefinition",
    "SchemaDefinition",
    # Snapshots & Repository
    "SchemaSnapshot",
    "SchemaRepository",
    "InMemorySchemaRepository",
    # State Machine & Manager
    "SchemaStateMachine",
    "SchemaLifecycleManager",
    # Events & Diff
    "LifecycleEvent",
    "compute_schema_diff",
    # Adapters
    "SchemaVersionAdapter",
    "LifecycleSchemaRegistryAdapter",
    # Serializer
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
    # Exceptions
    "SchemaLifecycleError",
    "SchemaValidationError",
    "SchemaSerializationError",
    "SchemaVersionError",
    "InvalidStateTransitionError",
    "SchemaNotApprovedError",
    "SchemaNotFoundError",
    "SchemaPromotionNotAllowedError",
    "SchemaRollbackError",
    "TenantAccessError",
]
