"""
Dynamic Schema Compiler (SCH-001, SCH-002, SCH-003, SCH-011).
Compiles canonical SchemaDefinition metamodel instances into:
1. Dynamically generated Pydantic v2 response models.
2. Corresponding canonical JSON Schemas (Draft 2020-12 / Draft-07).
Strictly adheres to PRD v1.1 §6.1, §6.2, §6.3, and SCH-001..003.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from enum import Enum
import keyword
import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    create_model,
    field_validator,
)
import jsonschema

from schema_lifecycle.enums import FieldType, PIIClass
from schema_lifecycle.exceptions import SchemaValidationError
from schema_lifecycle.metamodel import FieldDefinition, RuleDefinition, SchemaDefinition


class CompiledFieldMetadata(BaseModel):
    """
    Metadata preserved for a compiled schema field (SCH-001, SCH-002, SEC-007).
    Preserves field paths, evidence requirements, confidence thresholds, and PII classifications.
    """
    model_config = ConfigDict(frozen=True, extra="allow")

    field_path: str = Field(..., description="Canonical dot or bracket path of the field")
    name: str = Field(..., description="Local field attribute name")
    field_type: FieldType = Field(..., description="Canonical metamodel FieldType")
    python_type: Any = Field(..., description="Compiled Python runtime type")
    description: Optional[str] = Field(default=None, description="Field extraction description")
    required: bool = Field(default=False, description="Whether property is required in input payload")
    nullable: bool = Field(default=True, description="Whether null/None is permitted as a value")
    format: Optional[str] = Field(default=None, description="Format specifier or regex pattern")
    allowed_values: Optional[List[Any]] = Field(default=None, description="Permitted discrete values for enums")
    evidence_required: bool = Field(default=True, description="Evidence bounding box requirement flag")
    confidence_threshold: Optional[float] = Field(default=None, description="Per-field confidence threshold")
    pii_class: Optional[Union[PIIClass, str]] = Field(default=None, description="PII classification category")
    rules: List[RuleDefinition] = Field(default_factory=list, description="Validation rules targeting this field")


class BaseCompiledModel(BaseModel):
    """Base model for all dynamically compiled Pydantic models."""
    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
        validate_assignment=True,
    )


class _FieldNode:
    """Internal hierarchical node for building nested models."""
    def __init__(self, name: str, field_path: str):
        self.name: str = name
        self.field_path: str = field_path
        self.field_def: Optional[FieldDefinition] = None
        self.is_object: bool = False
        self.is_array: bool = False
        self.children: Dict[str, _FieldNode] = {}
        self.item_node: Optional[_FieldNode] = None
        self.item_scalar_type: Optional[FieldType] = None
        self.item_scalar_allowed_values: Optional[List[Any]] = None
        self.item_scalar_format: Optional[str] = None


class DynamicSchemaCompiler:
    """
    Compiles SchemaDefinition instances into runtime Pydantic v2 models and canonical JSON Schemas.
    Guarantees deterministic, collision-resistant naming and full metamodel fidelity.
    """

    CANONICAL_JSON_SCHEMA_DIALECT: str = "https://json-schema.org/draft/2020-12/schema"

    @classmethod
    def compile(cls, schema: SchemaDefinition) -> Type[BaseModel]:
        """
        Compiles a SchemaDefinition into a runtime Pydantic v2 BaseModel class.

        Guarantees:
        - Supports all 8 field types: string, integer, decimal, boolean, date, enum, object, array.
        - Supports Decimal for decimal fields and datetime.date for date fields.
        - Preserves required vs optional properties separately from nullable values.
        - Does not silently convert absent or null values into empty arrays or default objects.
        - Generates deterministic, collision-resistant model names using schema identity and version.
        - Attaches field metadata, original schema definition, and rules for downstream processing.
        """
        if not isinstance(schema, SchemaDefinition):
            raise SchemaValidationError(f"Expected SchemaDefinition instance, got {type(schema).__name__}")

        if not schema.fields:
            raise SchemaValidationError("Cannot compile schema definition with empty fields list")

        metadata_registry: Dict[str, CompiledFieldMetadata] = {}
        tree = cls._build_tree(schema)

        root_model = cls._compile_node(
            schema=schema,
            node=tree,
            metadata_registry=metadata_registry,
            is_root=True,
        )

        # Attach comprehensive metadata to root model class
        root_model.__schema_definition__ = schema  # type: ignore[attr-defined]
        root_model.__schema_id__ = schema.schema_id  # type: ignore[attr-defined]
        root_model.__schema_version__ = schema.version  # type: ignore[attr-defined]
        root_model.__field_metadata__ = metadata_registry  # type: ignore[attr-defined]
        root_model.__field_paths__ = list(metadata_registry.keys())  # type: ignore[attr-defined]
        root_model.__schema_rules__ = list(schema.rules)  # type: ignore[attr-defined]
        root_model.__rules__ = list(schema.rules)  # type: ignore[attr-defined]
        root_model.__json_schema_dialect__ = cls.CANONICAL_JSON_SCHEMA_DIALECT  # type: ignore[attr-defined]

        return root_model

    @classmethod
    def compile_json_schema(
        cls,
        schema: Union[SchemaDefinition, Type[BaseModel]],
        dialect: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generates canonical JSON Schema Draft 2020-12 for a SchemaDefinition or compiled model.
        Validates schema structure against the declared dialect using jsonschema validator.
        """
        target_dialect = dialect or cls.CANONICAL_JSON_SCHEMA_DIALECT

        if isinstance(schema, SchemaDefinition):
            model = cls.compile(schema)
        elif isinstance(schema, type) and issubclass(schema, BaseModel):
            model = schema
        else:
            raise SchemaValidationError(f"Expected SchemaDefinition or compiled BaseModel, got {type(schema).__name__}")

        raw_schema = model.model_json_schema()
        raw_schema["$schema"] = target_dialect

        # Dialect adjustment for non-nullable optional fields
        field_meta: Dict[str, CompiledFieldMetadata] = getattr(model, "__field_metadata__", {})
        cls._adjust_json_schema_nullability(raw_schema, field_meta)

        # Validate generated schema against canonical dialect
        try:
            jsonschema.Draft202012Validator.check_schema(raw_schema)
        except jsonschema.SchemaError as exc:
            raise SchemaValidationError(f"Generated JSON Schema failed dialect check: {exc}") from exc

        return raw_schema

    @classmethod
    def unflatten(cls, flat_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Converts flat dot-notation and bracket dictionary into a nested structure
        matching the compiled Pydantic models.
        """
        root: Dict[str, Any] = {}
        for key, value in flat_dict.items():
            norm_key = re.sub(r"\[(\d+)\]", r".\1", key)
            tokens = norm_key.split(".")
            curr: Any = root
            for i, token in enumerate(tokens[:-1]):
                next_token = tokens[i + 1]
                is_next_digit = next_token.isdigit()
                if token.isdigit():
                    idx = int(token)
                    while len(curr) <= idx:
                        curr.append([] if is_next_digit else {})
                    curr = curr[idx]
                else:
                    if token not in curr:
                        curr[token] = [] if is_next_digit else {}
                    curr = curr[token]
            last_token = tokens[-1]
            if last_token.isdigit():
                idx = int(last_token)
                while len(curr) <= idx:
                    curr.append(None)
                curr[idx] = value
            else:
                curr[last_token] = value
        return root

    @classmethod
    def flatten(cls, data_or_model: Union[BaseModel, Dict[str, Any]], prefix: str = "") -> Dict[str, Any]:
        """
        Flattens a compiled model instance or nested dictionary into canonical dot-paths.
        """
        out: Dict[str, Any] = {}
        data = data_or_model.model_dump(mode="python") if hasattr(data_or_model, "model_dump") else data_or_model

        if isinstance(data, dict):
            for k, v in data.items():
                new_prefix = f"{prefix}.{k}" if prefix else str(k)
                out.update(cls.flatten(v, new_prefix))
        elif isinstance(data, list):
            for i, item in enumerate(data):
                new_prefix = f"{prefix}.{i}"
                out.update(cls.flatten(item, new_prefix))
        else:
            out[prefix] = data
        return out

    # ========================================================================
    # Private Tree Building & Compilation Methods
    # ========================================================================

    @classmethod
    def _build_tree(cls, schema: SchemaDefinition) -> _FieldNode:
        """Constructs an intermediate hierarchical node tree from schema fields."""
        root = _FieldNode(name="root", field_path="")
        root.is_object = True

        for f in schema.fields:
            cls._insert_field(root, f, schema)

        return root

    @classmethod
    def _insert_field(cls, root: _FieldNode, f: FieldDefinition, schema: SchemaDefinition) -> None:
        """Inserts a FieldDefinition into the tree, handling dot-paths, brackets, objects, and arrays."""
        clean_name = f.name.strip()
        parts = clean_name.split(".")
        curr = root
        accum_path_parts: List[str] = []

        for i, part in enumerate(parts):
            is_leaf = (i == len(parts) - 1)
            is_bracket_arr = part.endswith("[]")
            token = part[:-2] if is_bracket_arr else part
            accum_path_parts.append(token)
            path_so_far = ".".join(accum_path_parts)

            if token not in curr.children:
                curr.children[token] = _FieldNode(name=token, field_path=path_so_far)
            node = curr.children[token]

            if is_bracket_arr:
                node.is_array = True
                if node.item_node is None:
                    node.item_node = _FieldNode(name=f"{token}_item", field_path=f"{path_so_far}[]")
                    node.item_node.is_object = True

                if not is_leaf:
                    curr = node.item_node
                else:
                    node.field_def = f
                    if f.type == FieldType.ARRAY and f.item_type:
                        node.item_scalar_type = f.item_type
                    elif f.items:
                        cls._populate_array_items(node, f.items)
            else:
                if not is_leaf:
                    node.is_object = True
                    curr = node
                else:
                    node.field_def = f
                    if f.type == FieldType.OBJECT or bool(f.fields):
                        node.is_object = True
                        if f.fields:
                            for child_field in f.fields:
                                cls._insert_hierarchical_child(node, child_field, path_so_far)
                    elif f.type == FieldType.ARRAY or f.is_array:
                        node.is_array = True
                        if f.item_type:
                            node.item_scalar_type = f.item_type
                        elif f.items:
                            cls._populate_array_items(node, f.items)

    @classmethod
    def _populate_array_items(
        cls,
        node: _FieldNode,
        items: Union[FieldDefinition, List[FieldDefinition]],
    ) -> None:
        """Populates item specification for an array node."""
        if isinstance(items, list):
            node.item_node = _FieldNode(name=f"{node.name}_item", field_path=f"{node.field_path}[]")
            node.item_node.is_object = True
            for child in items:
                cls._insert_hierarchical_child(node.item_node, child, f"{node.field_path}[]")
        elif isinstance(items, FieldDefinition):
            if items.type == FieldType.OBJECT or bool(items.fields):
                node.item_node = _FieldNode(name=f"{node.name}_item", field_path=f"{node.field_path}[]")
                node.item_node.is_object = True
                node.item_node.field_def = items
                if items.fields:
                    for child in items.fields:
                        cls._insert_hierarchical_child(node.item_node, child, f"{node.field_path}[]")
            else:
                node.item_scalar_type = items.type
                node.item_scalar_allowed_values = items.allowed_values
                node.item_scalar_format = items.format

    @classmethod
    def _insert_hierarchical_child(cls, parent: _FieldNode, child: FieldDefinition, parent_path: str) -> None:
        """Recursively inserts hierarchical child fields into parent node."""
        clean_name = child.clean_name
        child_path = f"{parent_path}.{clean_name}"
        child_node = _FieldNode(name=clean_name, field_path=child_path)
        child_node.field_def = child

        if child.type == FieldType.OBJECT or bool(child.fields):
            child_node.is_object = True
            if child.fields:
                for grandchild in child.fields:
                    cls._insert_hierarchical_child(child_node, grandchild, child_path)
        elif child.type == FieldType.ARRAY or child.is_array:
            child_node.is_array = True
            if child.item_type:
                child_node.item_scalar_type = child.item_type
            elif child.items:
                cls._populate_array_items(child_node, child.items)

        parent.children[clean_name] = child_node

    @classmethod
    def _compile_node(
        cls,
        schema: SchemaDefinition,
        node: _FieldNode,
        metadata_registry: Dict[str, CompiledFieldMetadata],
        is_root: bool = False,
        is_item: bool = False,
    ) -> Type[BaseModel]:
        """Recursively compiles a _FieldNode into a dynamic Pydantic BaseModel."""
        model_name = cls._make_model_name(
            schema_id=schema.schema_id,
            version=schema.version,
            field_path=None if is_root else node.field_path,
            is_item=is_item,
        )

        field_definitions: Dict[str, Any] = {}
        validators: Dict[str, Callable[..., Any]] = {}

        for child_name, child_node in node.children.items():
            py_field_name, field_alias = cls._get_safe_field_name(child_name)
            fdef = child_node.field_def

            # Determine requiredness and nullability
            required = fdef.required if fdef is not None else False
            nullable = cls._resolve_nullable(fdef, required)

            description = fdef.description if fdef else None
            format_pat = fdef.format if fdef else None

            # 1. Nested Object
            if child_node.is_object and not child_node.is_array:
                child_model = cls._compile_node(schema, child_node, metadata_registry)
                annotation, default_field = cls._build_pydantic_field_spec(
                    py_type=child_model,
                    required=required,
                    nullable=nullable,
                    description=description,
                    alias=field_alias,
                )
                field_definitions[py_field_name] = (annotation, default_field)
                cls._register_metadata(
                    metadata_registry=metadata_registry,
                    schema=schema,
                    node=child_node,
                    py_type=child_model,
                    field_type=FieldType.OBJECT,
                    required=required,
                    nullable=nullable,
                )

            # 2. Array / Repeating Group
            elif child_node.is_array:
                if child_node.item_node is not None and (child_node.item_node.children or child_node.item_node.is_object):
                    item_model = cls._compile_node(schema, child_node.item_node, metadata_registry, is_item=True)
                    list_type = List[item_model]  # type: ignore[valid-type]
                elif child_node.item_scalar_type:
                    scalar_py_type = cls._resolve_scalar_type(
                        schema=schema,
                        field_path=f"{child_node.field_path}[]",
                        field_type=child_node.item_scalar_type,
                        allowed_values=child_node.item_scalar_allowed_values,
                    )
                    list_type = List[scalar_py_type]  # type: ignore[valid-type]
                else:
                    list_type = List[Any]  # type: ignore[valid-type]

                annotation, default_field = cls._build_pydantic_field_spec(
                    py_type=list_type,
                    required=required,
                    nullable=nullable,
                    description=description,
                    alias=field_alias,
                )
                field_definitions[py_field_name] = (annotation, default_field)
                cls._register_metadata(
                    metadata_registry=metadata_registry,
                    schema=schema,
                    node=child_node,
                    py_type=list_type,
                    field_type=FieldType.ARRAY,
                    required=required,
                    nullable=nullable,
                )

            # 3. Scalar Field
            else:
                field_type = fdef.type if fdef is not None else FieldType.STRING
                scalar_py_type = cls._resolve_scalar_type(
                    schema=schema,
                    field_path=child_node.field_path,
                    field_type=field_type,
                    allowed_values=fdef.allowed_values if fdef else None,
                )

                pattern = None
                if field_type == FieldType.STRING and format_pat and (format_pat.startswith("^") or format_pat.endswith("$")):
                    pattern = format_pat

                annotation, default_field = cls._build_pydantic_field_spec(
                    py_type=scalar_py_type,
                    required=required,
                    nullable=nullable,
                    description=description,
                    pattern=pattern,
                    alias=field_alias,
                )
                field_definitions[py_field_name] = (annotation, default_field)
                cls._register_metadata(
                    metadata_registry=metadata_registry,
                    schema=schema,
                    node=child_node,
                    py_type=scalar_py_type,
                    field_type=field_type,
                    required=required,
                    nullable=nullable,
                )

            # Register non-null enforcement validator for optional non-nullable fields
            if not required and not nullable:
                val_func_name = f"_val_not_null_{py_field_name}"
                validators[val_func_name] = field_validator(py_field_name, mode="before")(cls._make_not_null_validator(child_name))

        # Dynamically create Pydantic model
        model = create_model(
            model_name,
            __base__=BaseCompiledModel,
            __validators__=validators if validators else None,
            **field_definitions,
        )

        return model

    @classmethod
    def _resolve_nullable(cls, field_def: Optional[FieldDefinition], required: bool) -> bool:
        """Determines if a field permits None/null values."""
        if field_def is None:
            return not required

        # Check explicit nullable attribute
        is_explicit = getattr(field_def, "nullable", None)
        if is_explicit is None and hasattr(field_def, "model_extra") and field_def.model_extra:
            is_explicit = field_def.model_extra.get("nullable")

        if is_explicit is not None:
            return bool(is_explicit)

        # Default extraction semantics: optional fields are nullable; required fields are non-null
        return not required

    @classmethod
    def _resolve_scalar_type(
        cls,
        schema: SchemaDefinition,
        field_path: str,
        field_type: FieldType,
        allowed_values: Optional[List[Any]] = None,
    ) -> Any:
        """Maps canonical FieldType to exact Python runtime type."""
        if field_type == FieldType.STRING:
            return str
        elif field_type == FieldType.INTEGER:
            return int
        elif field_type == FieldType.DECIMAL:
            return Decimal
        elif field_type == FieldType.BOOLEAN:
            return bool
        elif field_type == FieldType.DATE:
            return date
        elif field_type == FieldType.ENUM:
            if not allowed_values:
                raise SchemaValidationError(f"Field '{field_path}' of type 'enum' requires allowed_values")
            enum_name = cls._make_enum_name(schema.schema_id, schema.version, field_path)
            return cls._make_enum_class(enum_name, allowed_values)
        return str

    @classmethod
    def _build_pydantic_field_spec(
        cls,
        py_type: Any,
        required: bool,
        nullable: bool,
        description: Optional[str] = None,
        pattern: Optional[str] = None,
        alias: Optional[str] = None,
    ) -> Tuple[Any, Any]:
        """
        Builds (annotation, FieldInfo) respecting required vs optional and nullable semantics.
        Guarantees that absent / null fields default to None rather than empty objects or lists.
        """
        field_kwargs: Dict[str, Any] = {}
        if description:
            field_kwargs["description"] = description
        if pattern:
            field_kwargs["pattern"] = pattern
        if alias:
            field_kwargs["alias"] = alias

        if required:
            if nullable:
                # Required in payload, but value can be null
                annotation = Optional[py_type]
                field_spec = Field(..., **field_kwargs)
            else:
                # Required in payload and cannot be null
                annotation = py_type
                field_spec = Field(..., **field_kwargs)
        else:
            if nullable:
                # Optional in payload, defaults to None
                annotation = Optional[py_type]
                field_spec = Field(default=None, **field_kwargs)
            else:
                # Optional in payload; if provided, cannot be null
                annotation = Optional[py_type]
                field_spec = Field(default=None, **field_kwargs)

        return annotation, field_spec

    @classmethod
    def _make_not_null_validator(cls, field_name: str) -> Callable[[Any, Any], Any]:
        """Creates a field validator rejecting explicit null for non-nullable optional fields."""
        def validator(v: Any) -> Any:
            if v is None:
                raise ValueError(f"Field '{field_name}' is declared non-nullable and cannot be null when provided")
            return v
        return validator

    @classmethod
    def _register_metadata(
        cls,
        metadata_registry: Dict[str, CompiledFieldMetadata],
        schema: SchemaDefinition,
        node: _FieldNode,
        py_type: Any,
        field_type: FieldType,
        required: bool,
        nullable: bool,
    ) -> None:
        """Records compiled field metadata in the registry."""
        fdef = node.field_def
        matching_rules = [
            r for r in schema.rules
            if node.field_path in r.target_fields or any(t.startswith(node.field_path) for t in r.target_fields)
        ]
        meta = CompiledFieldMetadata(
            field_path=node.field_path,
            name=node.name,
            field_type=field_type,
            python_type=py_type,
            description=fdef.description if fdef else None,
            required=required,
            nullable=nullable,
            format=fdef.format if fdef else None,
            allowed_values=fdef.allowed_values if fdef else None,
            evidence_required=fdef.evidence_required if fdef else True,
            confidence_threshold=fdef.confidence_threshold if fdef else None,
            pii_class=fdef.pii_class if fdef else None,
            rules=matching_rules,
        )
        metadata_registry[node.field_path] = meta

    @classmethod
    def _make_enum_class(cls, enum_name: str, allowed_values: List[Any]) -> Type[Enum]:
        """Dynamically creates a typed Enum class preserving member values."""
        members: Dict[str, Any] = {}
        for i, val in enumerate(allowed_values):
            clean_member = re.sub(r"\W|^(?=\d)", "_", str(val)).upper()
            if not clean_member or clean_member in members:
                clean_member = f"VAL_{i}_{clean_member}"
            members[clean_member] = val

        if all(isinstance(v, int) and not isinstance(v, bool) for v in allowed_values):
            return Enum(enum_name, members, type=int)
        elif all(isinstance(v, str) for v in allowed_values):
            return Enum(enum_name, members, type=str)
        return Enum(enum_name, members)

    @classmethod
    def _make_model_name(
        cls,
        schema_id: str,
        version: int,
        field_path: Optional[str] = None,
        is_item: bool = False,
    ) -> str:
        """Generates deterministic, collision-resistant model class names."""
        prefix = f"{cls._to_pascal_case(schema_id)}_V{version}"
        if not field_path:
            return f"{prefix}_Model"

        cleaned_path = field_path.replace("[]", "")
        parts = [cls._to_pascal_case(p) for p in cleaned_path.split(".") if p]
        path_str = "_".join(parts)
        suffix = "_Item_Model" if is_item else "_Model"
        return f"{prefix}_{path_str}{suffix}"

    @classmethod
    def _make_enum_name(cls, schema_id: str, version: int, field_path: str) -> str:
        """Generates deterministic, collision-resistant enum class names."""
        prefix = f"{cls._to_pascal_case(schema_id)}_V{version}"
        cleaned_path = field_path.replace("[]", "")
        parts = [cls._to_pascal_case(p) for p in cleaned_path.split(".") if p]
        path_str = "_".join(parts)
        return f"{prefix}_{path_str}_Enum"

    @classmethod
    def _to_pascal_case(cls, text: str) -> str:
        """Converts arbitrary string into clean PascalCase identifier."""
        clean = re.sub(r"[^a-zA-Z0-9]+", " ", str(text)).strip()
        if not clean:
            return "Item"
        words = clean.split()
        return "".join(w.capitalize() for w in words)

    @classmethod
    def _get_safe_field_name(cls, name: str) -> Tuple[str, Optional[str]]:
        """Returns safe Python attribute name and optional alias if altered."""
        clean = name.replace("-", "_").strip()
        alias = name if name != clean else None

        if keyword.iskeyword(clean) or clean in ("fields", "schema", "model_validate", "model_dump"):
            py_name = f"{clean}_"
            return py_name, name
        return clean, alias

    @classmethod
    def _adjust_json_schema_nullability(
        cls,
        json_schema: Dict[str, Any],
        field_meta: Dict[str, CompiledFieldMetadata],
    ) -> None:
        """Recursively cleans up anyOf in JSON schema for non-nullable optional fields."""
        props = json_schema.get("properties", {})
        for prop_name, prop_spec in props.items():
            if not isinstance(prop_spec, dict):
                continue
            # Look up metadata
            meta = next((m for m in field_meta.values() if m.name == prop_name), None)
            if meta and not meta.nullable:
                if "anyOf" in prop_spec:
                    non_null_subtypes = [s for s in prop_spec["anyOf"] if s.get("type") != "null"]
                    if len(non_null_subtypes) == 1:
                        preserved = {k: v for k, v in prop_spec.items() if k not in ("anyOf", "default")}
                        merged = {**non_null_subtypes[0], **preserved}
                        props[prop_name] = merged
