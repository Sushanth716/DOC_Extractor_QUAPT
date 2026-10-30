"""
YAML and JSON Serialization and Safe Parsing for Schema Definitions (SCH-003).
Strictly adheres to PRD §6.1, §6.2, and SCH-003.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Dict
import yaml

from schema_lifecycle.exceptions import SchemaSerializationError, SchemaValidationError

if TYPE_CHECKING:
    from schema_lifecycle.metamodel import SchemaDefinition


def safe_load_yaml(content: str) -> Dict[str, Any]:
    """
    Safely loads a YAML string into a Python dictionary.
    Guarantees no arbitrary Python object instantiation (SafeLoader).
    """
    if not isinstance(content, str):
        raise SchemaSerializationError(
            f"Expected string content for YAML parsing, got {type(content).__name__}",
            format_type="yaml",
        )
    try:
        data = yaml.safe_load(content)
    except yaml.YAMLError as exc:
        raise SchemaSerializationError(
            f"Failed to parse YAML syntax: {exc}",
            format_type="yaml",
        ) from exc

    if not isinstance(data, dict):
        raise SchemaSerializationError(
            f"YAML document must be a mapping (dictionary) at the root, got {type(data).__name__}",
            format_type="yaml",
        )
    return data


def safe_dump_yaml(data: Dict[str, Any], sort_keys: bool = False) -> str:
    """
    Safely serializes a Python dictionary to a YAML string.
    Preserves property order when sort_keys=False.
    """
    try:
        return yaml.safe_dump(
            data,
            sort_keys=sort_keys,
            default_flow_style=False,
            allow_unicode=True,
        )
    except Exception as exc:
        raise SchemaSerializationError(
            f"Failed to dump dictionary to YAML: {exc}",
            format_type="yaml",
        ) from exc


def safe_load_json(content: str) -> Dict[str, Any]:
    """
    Safely loads a JSON string into a Python dictionary.
    """
    if not isinstance(content, str):
        raise SchemaSerializationError(
            f"Expected string content for JSON parsing, got {type(content).__name__}",
            format_type="json",
        )
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise SchemaSerializationError(
            f"Failed to parse JSON syntax: {exc}",
            format_type="json",
        ) from exc

    if not isinstance(data, dict):
        raise SchemaSerializationError(
            f"JSON document must be an object at the root, got {type(data).__name__}",
            format_type="json",
        )
    return data


def safe_dump_json(data: Dict[str, Any], indent: int = 2) -> str:
    """
    Safely serializes a Python dictionary to a JSON string.
    """
    try:
        return json.dumps(data, indent=indent, ensure_ascii=False)
    except Exception as exc:
        raise SchemaSerializationError(
            f"Failed to dump dictionary to JSON: {exc}",
            format_type="json",
        ) from exc


def schema_to_dict(schema: SchemaDefinition, exclude_none: bool = True) -> Dict[str, Any]:
    """
    Converts a SchemaDefinition instance to a dictionary, preserving all properties.
    """
    return schema.to_dict(exclude_none=exclude_none)


def schema_from_dict(data: Dict[str, Any]) -> SchemaDefinition:
    """
    Parses and validates a SchemaDefinition from a dictionary.
    """
    from schema_lifecycle.metamodel import SchemaDefinition
    return SchemaDefinition.from_dict(data)


def schema_to_yaml(schema: SchemaDefinition, sort_keys: bool = False) -> str:
    """
    Exports a SchemaDefinition instance to a YAML string (SCH-003).
    """
    data = schema_to_dict(schema)
    return safe_dump_yaml(data, sort_keys=sort_keys)


def schema_from_yaml(yaml_str: str) -> SchemaDefinition:
    """
    Imports and validates a SchemaDefinition from a YAML string (SCH-003).
    """
    data = safe_load_yaml(yaml_str)
    return schema_from_dict(data)


def schema_to_json(schema: SchemaDefinition, indent: int = 2) -> str:
    """
    Exports a SchemaDefinition instance to a JSON string (SCH-003).
    """
    data = schema_to_dict(schema)
    return safe_dump_json(data, indent=indent)


def schema_from_json(json_str: str) -> SchemaDefinition:
    """
    Imports and validates a SchemaDefinition from a JSON string (SCH-003).
    """
    data = safe_load_json(json_str)
    return schema_from_dict(data)

