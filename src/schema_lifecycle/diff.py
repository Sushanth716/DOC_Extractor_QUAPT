"""
Deterministic Structural Diff Engine for Schema Metamodel (SCH-006).
Computes deterministic, reproducible differences between schema definitions.
Strictly adheres to PRD §6.1, §6.2, and SCH-006.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from schema_lifecycle.metamodel import SchemaDefinition


def compute_schema_diff(
    old_schema: Optional[SchemaDefinition],
    new_schema: SchemaDefinition,
) -> Dict[str, Any]:
    """
    Computes a deterministic structural diff between two SchemaDefinitions.

    If old_schema is None, computes the initial baseline diff for a new schema.
    All dictionary keys and list elements are deterministically ordered.
    """
    if new_schema is None:
        raise ValueError("new_schema cannot be None")

    if old_schema is None:
        # Initial baseline version diff
        added_fields = [
            f.model_dump(mode="json", exclude_none=True)
            for f in sorted(new_schema.fields, key=lambda x: x.name)
        ]
        added_rules = [
            r.model_dump(mode="json", exclude_none=True)
            for r in sorted(new_schema.rules, key=lambda x: x.id)
        ]
        metadata_changes: Dict[str, Dict[str, Any]] = {}
        for key in ["domain", "document_type", "description"]:
            val = getattr(new_schema, key, None)
            if val is not None:
                metadata_changes[key] = {"old": None, "new": val}

        return {
            "is_initial": True,
            "has_changes": True,
            "fields": {
                "added": added_fields,
                "removed": [],
                "modified": [],
            },
            "rules": {
                "added": added_rules,
                "removed": [],
                "modified": [],
            },
            "metadata": metadata_changes,
            "summary": {
                "added_fields": len(added_fields),
                "removed_fields": 0,
                "modified_fields": 0,
                "added_rules": len(added_rules),
                "removed_rules": 0,
                "modified_rules": 0,
                "metadata_changed": bool(metadata_changes),
                "total_changes": len(added_fields) + len(added_rules) + len(metadata_changes),
            },
        }

    # 1. Compare Fields
    old_fields_map = {f.name: f for f in old_schema.fields}
    new_fields_map = {f.name: f for f in new_schema.fields}

    added_field_names = sorted(set(new_fields_map.keys()) - set(old_fields_map.keys()))
    removed_field_names = sorted(set(old_fields_map.keys()) - set(new_fields_map.keys()))
    common_field_names = sorted(set(old_fields_map.keys()) & set(new_fields_map.keys()))

    added_fields = [
        new_fields_map[name].model_dump(mode="json", exclude_none=True)
        for name in added_field_names
    ]
    removed_fields = [
        old_fields_map[name].model_dump(mode="json", exclude_none=True)
        for name in removed_field_names
    ]

    modified_fields: List[Dict[str, Any]] = []
    for name in common_field_names:
        old_dump = old_fields_map[name].model_dump(mode="json", exclude_none=True)
        new_dump = new_fields_map[name].model_dump(mode="json", exclude_none=True)

        if old_dump != new_dump:
            field_changes: Dict[str, Any] = {}
            all_keys = sorted(set(old_dump.keys()) | set(new_dump.keys()))
            for k in all_keys:
                old_val = old_dump.get(k)
                new_val = new_dump.get(k)
                if old_val != new_val:
                    field_changes[k] = {"old": old_val, "new": new_val}

            if field_changes:
                modified_fields.append({
                    "name": name,
                    "changes": field_changes,
                })

    # 2. Compare Rules
    old_rules_map = {r.id: r for r in old_schema.rules}
    new_rules_map = {r.id: r for r in new_schema.rules}

    added_rule_ids = sorted(set(new_rules_map.keys()) - set(old_rules_map.keys()))
    removed_rule_ids = sorted(set(old_rules_map.keys()) - set(new_rules_map.keys()))
    common_rule_ids = sorted(set(old_rules_map.keys()) & set(new_rules_map.keys()))

    added_rules = [
        new_rules_map[rid].model_dump(mode="json", exclude_none=True)
        for rid in added_rule_ids
    ]
    removed_rules = [
        old_rules_map[rid].model_dump(mode="json", exclude_none=True)
        for rid in removed_rule_ids
    ]

    modified_rules: List[Dict[str, Any]] = []
    for rid in common_rule_ids:
        old_dump = old_rules_map[rid].model_dump(mode="json", exclude_none=True)
        new_dump = new_rules_map[rid].model_dump(mode="json", exclude_none=True)

        if old_dump != new_dump:
            rule_changes: Dict[str, Any] = {}
            all_keys = sorted(set(old_dump.keys()) | set(new_dump.keys()))
            for k in all_keys:
                old_val = old_dump.get(k)
                new_val = new_dump.get(k)
                if old_val != new_val:
                    rule_changes[k] = {"old": old_val, "new": new_val}

            if rule_changes:
                modified_rules.append({
                    "id": rid,
                    "changes": rule_changes,
                })

    # 3. Compare Metadata
    metadata_changes = {}
    for key in sorted(["domain", "document_type", "description"]):
        old_val = getattr(old_schema, key, None)
        new_val = getattr(new_schema, key, None)
        if old_val != new_val:
            metadata_changes[key] = {"old": old_val, "new": new_val}

    summary = {
        "added_fields": len(added_fields),
        "removed_fields": len(removed_fields),
        "modified_fields": len(modified_fields),
        "added_rules": len(added_rules),
        "removed_rules": len(removed_rules),
        "modified_rules": len(modified_rules),
        "metadata_changed": bool(metadata_changes),
    }
    total_changes = (
        summary["added_fields"]
        + summary["removed_fields"]
        + summary["modified_fields"]
        + summary["added_rules"]
        + summary["removed_rules"]
        + summary["modified_rules"]
        + (1 if summary["metadata_changed"] else 0)
    )
    summary["total_changes"] = total_changes

    return {
        "is_initial": False,
        "has_changes": total_changes > 0,
        "fields": {
            "added": added_fields,
            "removed": removed_fields,
            "modified": modified_fields,
        },
        "rules": {
            "added": added_rules,
            "removed": removed_rules,
            "modified": modified_rules,
        },
        "metadata": metadata_changes,
        "summary": summary,
    }
