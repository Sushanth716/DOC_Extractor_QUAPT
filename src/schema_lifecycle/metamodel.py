"""
Schema Metamodel Definitions (SCH-001, SCH-002, SCH-003).
Strictly adheres to PRD v1.1 §6.1, §6.2, §6.3, and SCH-001..003.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Union
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from schema_lifecycle.enums import FieldType, PIIClass, RuleOnFail, SchemaStatus
from schema_lifecycle.exceptions import SchemaSerializationError, SchemaValidationError


class RuleDefinition(BaseModel):
    """
    Business or validation rule definition matching PRD §6.2 and FR-VAL-006.
    """
    model_config = ConfigDict(extra="allow", validate_assignment=True)

    id: str = Field(..., description="Unique rule identifier (e.g. 'R1')")
    expression: str = Field(..., description="Rule expression to evaluate")
    on_fail: Union[RuleOnFail, str] = Field(
        default=RuleOnFail.REVIEW,
        description="Action to take when rule evaluation fails (e.g. 'review', 'reject')",
    )
    description: Optional[str] = Field(default=None, description="Human-readable rule explanation")
    target_fields: List[str] = Field(
        default_factory=list,
        description="Fields affected or flagged when this rule fails",
    )
    error_code: Optional[str] = Field(
        default=None,
        description="Standard machine-readable error code when rule fails",
    )

    @field_validator("id")
    @classmethod
    def validate_id(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("Rule 'id' must be a non-empty string")
        return v.strip()

    @field_validator("expression")
    @classmethod
    def validate_expression(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("Rule 'expression' must be a non-empty string")
        return v.strip()

    @field_validator("on_fail", mode="before")
    @classmethod
    def normalize_on_fail(cls, v: Any) -> Any:
        if isinstance(v, str):
            v_clean = v.strip().lower()
            try:
                return RuleOnFail(v_clean)
            except ValueError:
                return v_clean
        return v


class FieldDefinition(BaseModel):
    """
    Field definition metamodel matching PRD §6.1 SCH-001 and §6.2 SCH-002.
    Defines per field:
    - name: field name (supports dot-paths and array notation)
    - type: string, integer, decimal, boolean, date, enum, object, array
    - description: description of extraction semantics
    - required: required or optional (default False)
    - allowed_values: permitted values (mandatory for enum)
    - format: regex pattern or format description
    - evidence_required: evidence required flag (default True per SCH-001)
    - confidence_threshold: per-field threshold in [0.0, 1.0]
    - pii_class: classification for data protection (SEC-007)
    - fields: nested child fields for object types (SCH-002)
    - items: child field(s) for array/repeating groups (SCH-002)
    - extra attributes (e.g. currency, source) preserved without loss
    """
    model_config = ConfigDict(extra="allow", validate_assignment=True)

    name: str = Field(..., description="Field name or dot/bracket path")
    type: FieldType = Field(..., description="Field data type (SCH-001)")
    description: Optional[str] = Field(default=None, description="Field extraction description")
    required: bool = Field(default=False, description="Whether field is required or optional")
    allowed_values: Optional[List[Any]] = Field(
        default=None,
        description="Allowed discrete values (required for enum type)",
    )
    format: Optional[str] = Field(
        default=None,
        description="Format specifier or regex pattern (e.g. '^[A-Z]{5}[0-9]{4}[A-Z]$')",
    )
    evidence_required: bool = Field(
        default=True,
        description="Evidence bounding box required flag (SCH-001 default true)",
    )
    confidence_threshold: Optional[float] = Field(
        default=None,
        description="Minimum confidence score threshold in range [0.0, 1.0]",
    )
    pii_class: Optional[Union[PIIClass, str]] = Field(
        default=None,
        description="PII classification category (SEC-007)",
    )
    fields: Optional[List[FieldDefinition]] = Field(
        default=None,
        description="Nested child fields for object types (SCH-002)",
    )
    items: Optional[Union[FieldDefinition, List[FieldDefinition]]] = Field(
        default=None,
        description="Child element definition(s) for repeating groups / arrays (SCH-002)",
    )
    item_type: Optional[FieldType] = Field(
        default=None,
        description="Primitive item type if array holds homogeneous scalar items",
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("Field 'name' must be a non-empty string")
        return v.strip()

    @field_validator("type", mode="before")
    @classmethod
    def validate_and_normalize_type(cls, v: Any) -> FieldType:
        if v is None:
            raise ValueError("Field 'type' is required (SCH-001)")
        if isinstance(v, FieldType):
            return v
        if isinstance(v, str):
            v_norm = v.strip().lower()
            try:
                return FieldType(v_norm)
            except ValueError:
                valid_types = [t.value for t in FieldType]
                raise ValueError(
                    f"Invalid field type '{v}'. Supported types are: {', '.join(valid_types)}"
                )
        raise ValueError(f"Field 'type' must be a valid FieldType or string, got {type(v).__name__}")

    @field_validator("confidence_threshold")
    @classmethod
    def validate_confidence_threshold(cls, v: Optional[float]) -> Optional[float]:
        if v is not None:
            if not isinstance(v, (int, float)):
                raise ValueError(f"Confidence threshold must be a numeric float, got {type(v).__name__}")
            if v < 0.0 or v > 1.0:
                raise ValueError(f"Confidence threshold must be between 0.0 and 1.0, got {v}")
        return v

    @field_validator("format")
    @classmethod
    def validate_format_regex(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and isinstance(v, str) and (v.startswith("^") or v.endswith("$")):
            try:
                re.compile(v)
            except re.error as exc:
                raise ValueError(f"Invalid regex pattern in format '{v}': {exc}")
        return v

    @field_validator("pii_class", mode="before")
    @classmethod
    def normalize_pii_class(cls, v: Any) -> Any:
        if isinstance(v, str):
            v_clean = v.strip().lower()
            try:
                return PIIClass(v_clean)
            except ValueError:
                return v_clean
        return v

    @model_validator(mode="after")
    def validate_type_specific_rules(self) -> FieldDefinition:
        # SCH-001: Enum validation
        if self.type == FieldType.ENUM:
            if self.allowed_values is None or len(self.allowed_values) == 0:
                raise ValueError(f"Field '{self.name}' of type 'enum' requires non-empty 'allowed_values'")

        # Duplicate child field names check for objects
        if self.fields:
            seen_names: Set[str] = set()
            for child in self.fields:
                if child.name in seen_names:
                    raise ValueError(f"Duplicate child field name '{child.name}' under '{self.name}'")
                seen_names.add(child.name)

        return self

    @property
    def is_array(self) -> bool:
        """Indicates whether this field represents an array or repeating group."""
        return self.type == FieldType.ARRAY or self.name.endswith("[]") or self.items is not None

    @property
    def is_object(self) -> bool:
        """Indicates whether this field represents a nested object."""
        return self.type == FieldType.OBJECT or bool(self.fields)

    @property
    def is_repeating_group(self) -> bool:
        """Indicates whether this field is a repeating group (SCH-002)."""
        return self.is_array and (self.items is not None or self.name.endswith("[]") or bool(self.fields))

    @property
    def clean_name(self) -> str:
        """Returns the field name without array bracket suffix."""
        if self.name.endswith("[]"):
            return self.name[:-2]
        return self.name


# Rebuild recursive model to resolve forward references
FieldDefinition.model_rebuild()


class SchemaDefinition(BaseModel):
    """
    Canonical Schema Definition Metamodel (SCH-001, SCH-002, SCH-003).
    Strictly adheres to PRD §6.1, §6.2 and §6.3.
    """
    model_config = ConfigDict(extra="allow", validate_assignment=True)

    schema_id: str = Field(..., description="Unique schema identifier (e.g. 'casa_account_opening')")
    version: int = Field(..., description="Canonical integer schema version (e.g. 1, 2, 3)")
    status: SchemaStatus = Field(default=SchemaStatus.DRAFT, description="Schema lifecycle status (SCH-004)")
    tenant_id: Optional[str] = Field(default=None, description="Tenant identifier for multi-tenant isolation")
    domain: Optional[str] = Field(default=None, description="Industry domain (e.g. 'BFSI', 'Healthcare')")
    document_type: Optional[str] = Field(default=None, description="Document type name or enum mapping")
    description: Optional[str] = Field(default=None, description="Human-readable schema description")
    fields: List[FieldDefinition] = Field(default_factory=list, description="Field definitions list (SCH-001, SCH-002)")
    rules: List[RuleDefinition] = Field(default_factory=list, description="Business and validation rules list")

    @field_validator("schema_id")
    @classmethod
    def validate_schema_id(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("schema_id must be a non-empty string")
        return v.strip()

    @field_validator("version", mode="before")
    @classmethod
    def validate_version(cls, v: Any) -> int:
        if isinstance(v, bool) or not isinstance(v, int):
            raise ValueError(
                f"Schema version must be a canonical integer, got {type(v).__name__}: {v!r}. "
                "Use SchemaVersionAdapter for existing string-version interfaces."
            )
        if v < 1:
            raise ValueError(f"Schema version must be a positive integer >= 1, got {v}")
        return v

    @field_validator("status", mode="before")
    @classmethod
    def normalize_status(cls, v: Any) -> Any:
        if isinstance(v, str):
            v_clean = v.strip().lower()
            try:
                return SchemaStatus(v_clean)
            except ValueError:
                valid_statuses = [s.value for s in SchemaStatus]
                raise ValueError(
                    f"Invalid schema status '{v}'. Allowed statuses are: {', '.join(valid_statuses)}"
                )
        return v

    @model_validator(mode="after")
    def validate_schema_integrity(self) -> SchemaDefinition:
        # Enforce that fields list is non-empty
        if not self.fields:
            raise ValueError("Schema must define at least one field in 'fields'")

        # Duplicate top-level field names check
        seen_names: Set[str] = set()
        for f in self.fields:
            if f.name in seen_names:
                raise ValueError(f"Duplicate field name '{f.name}' in schema '{self.schema_id}'")
            seen_names.add(f.name)

        # Duplicate rule IDs check
        seen_rules: Set[str] = set()
        for r in self.rules:
            if r.id in seen_rules:
                raise ValueError(f"Duplicate rule id '{r.id}' in schema '{self.schema_id}'")
            seen_rules.add(r.id)

        return self

    def to_dict(self, exclude_none: bool = True) -> Dict[str, Any]:
        """
        Dumps the schema definition to a dictionary preserving all properties.
        Enums are serialized to primitive strings; extra fields are preserved.
        """
        data = self.model_dump(mode="json", exclude_none=exclude_none)
        # If rules list is empty and exclude_none is True, omit empty rules list for cleaner serialization
        if exclude_none and "rules" in data and not data["rules"]:
            del data["rules"]
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SchemaDefinition:
        """
        Constructs a SchemaDefinition from a dictionary, raising SchemaValidationError on failure.
        """
        if not isinstance(data, dict):
            raise SchemaValidationError(f"Expected dictionary for schema definition, got {type(data).__name__}")
        try:
            return cls(**data)
        except ValidationError as exc:
            raise SchemaValidationError(
                message=f"Schema validation failed: {exc}",
                errors=exc.errors(),
            ) from exc
        except ValueError as exc:
            raise SchemaValidationError(message=str(exc)) from exc

    def to_yaml(self, sort_keys: bool = False) -> str:
        """Serializes this schema definition to YAML string (SCH-003)."""
        from schema_lifecycle.serializer import schema_to_yaml
        return schema_to_yaml(self, sort_keys=sort_keys)

    @classmethod
    def from_yaml(cls, yaml_str: str) -> SchemaDefinition:
        """Deserializes a schema definition from a YAML string (SCH-003)."""
        from schema_lifecycle.serializer import schema_from_yaml
        return schema_from_yaml(yaml_str)

    def to_json(self, indent: int = 2) -> str:
        """Serializes this schema definition to JSON string (SCH-003)."""
        from schema_lifecycle.serializer import schema_to_json
        return schema_to_json(self, indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> SchemaDefinition:
        """Deserializes a schema definition from a JSON string (SCH-003)."""
        from schema_lifecycle.serializer import schema_from_json
        return schema_from_json(json_str)

    def get_field(self, field_path: str) -> Optional[FieldDefinition]:
        """
        Retrieves a field definition by path, searching flat paths, array brackets,
        and nested hierarchies.
        """
        normalized_path = field_path.strip()

        # Direct match in top-level fields
        for f in self.fields:
            if f.name == normalized_path:
                return f
            if f.clean_name == normalized_path:
                return f

        # Recursive search for hierarchical fields
        def _search(fields: List[FieldDefinition], target: str) -> Optional[FieldDefinition]:
            for field in fields:
                if field.name == target or field.clean_name == target:
                    return field
                if field.fields:
                    res = _search(field.fields, target)
                    if res:
                        return res
                if field.items:
                    items_list = [field.items] if isinstance(field.items, FieldDefinition) else field.items
                    res = _search(items_list, target)
                    if res:
                        return res
            return None

        return _search(self.fields, normalized_path)

    def to_tree(self) -> Dict[str, Any]:
        """
        Produces a hierarchical tree representation of all fields for SCR-05 Schema Studio.
        Handles both hierarchical definitions and dot-notation paths.
        """
        tree: Dict[str, Any] = {}

        for f in self.fields:
            # Handle dot-notation paths (e.g. applicant.full_name or account_details.initial_deposit_amount)
            parts = f.name.split(".")
            curr = tree
            for i, part in enumerate(parts[:-1]):
                clean_part = part[:-2] if part.endswith("[]") else part
                is_arr = part.endswith("[]")
                if clean_part not in curr:
                    curr[clean_part] = {
                        "type": "array" if is_arr else "object",
                        "children": {},
                    }
                elif "children" not in curr[clean_part]:
                    curr[clean_part]["children"] = {}
                curr = curr[clean_part]["children"]

            leaf_name = parts[-1]
            clean_leaf = leaf_name[:-2] if leaf_name.endswith("[]") else leaf_name
            leaf_dict = f.model_dump(mode="json", exclude_none=True)
            if leaf_name.endswith("[]"):
                leaf_dict["type"] = "array"

            if clean_leaf in curr and "children" in curr[clean_leaf]:
                children = curr[clean_leaf]["children"]
                curr[clean_leaf] = leaf_dict
                curr[clean_leaf]["children"] = children
            else:
                curr[clean_leaf] = leaf_dict

        return tree

    def flatten_fields(self) -> List[FieldDefinition]:
        """
        Returns a flat list of all fields with resolved dot-paths and bracket notation.
        """
        flat: List[FieldDefinition] = []

        def _traverse(fields: List[FieldDefinition], prefix: str = ""):
            for field in fields:
                current_path = f"{prefix}.{field.name}" if prefix else field.name
                if field.fields:
                    _traverse(field.fields, current_path)
                elif field.items:
                    items_list = [field.items] if isinstance(field.items, FieldDefinition) else field.items
                    array_prefix = f"{current_path}[]"
                    _traverse(items_list, array_prefix)
                else:
                    flat.append(field)

        _traverse(self.fields)
        return flat if flat else list(self.fields)
