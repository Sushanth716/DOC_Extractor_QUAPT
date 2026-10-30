"""
Explicit Version Adapters for Schema Metamodel Integration.
Bridges canonical integer schema versions with legacy / external string-version interfaces.
Preserves existing SchemaRegistry and ApprovedSchemaRef contracts without modification.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

from schema_lifecycle.exceptions import SchemaVersionError

if TYPE_CHECKING:
    from schema_lifecycle.metamodel import SchemaDefinition


class SchemaVersionAdapter:
    """
    Explicit adapter bridging canonical integer schema versions (PRD §6.2)
    and external / legacy string-version interfaces (such as ApprovedSchemaRef).
    """

    @classmethod
    def to_canonical(cls, version: Union[int, str]) -> int:
        """
        Converts an integer or string version representation to the canonical positive integer.
        
        Examples:
            1 -> 1
            "3" -> 3
            "1.0.0" -> 1
            "v2.1" -> 2
        """
        if isinstance(version, int) and not isinstance(version, bool):
            if version < 1:
                raise SchemaVersionError(
                    f"Canonical schema version must be >= 1, got {version}",
                    version=version,
                )
            return version

        if isinstance(version, str):
            clean = version.strip()
            if not clean:
                raise SchemaVersionError("Version string cannot be empty", version=version)

            # Strip leading 'v' or 'V' if present
            if clean.startswith(("v", "V")):
                clean = clean[1:].strip()

            # Direct integer string
            if clean.isdigit():
                val = int(clean)
                if val < 1:
                    raise SchemaVersionError(f"Canonical schema version must be >= 1, got {val}", version=version)
                return val

            # Semver / dotted version string (e.g. "1.0.0" or "2.3")
            match = re.match(r"^(\d+)(?:\.\d+)*", clean)
            if match:
                val = int(match.group(1))
                if val < 1:
                    raise SchemaVersionError(f"Canonical schema version must be >= 1, got {val}", version=version)
                return val

        raise SchemaVersionError(
            f"Cannot parse schema version '{version}' to a canonical positive integer",
            version=version,
        )

    @classmethod
    def to_semver(cls, version: int) -> str:
        """
        Converts a canonical integer version to semantic version string (e.g. 1 -> "1.0.0").
        """
        canonical = cls.to_canonical(version)
        return f"{canonical}.0.0"

    @classmethod
    def adapt_dict(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adapts a raw dictionary that may contain a string-based version into one
        with a canonical integer version.
        """
        adapted = dict(data)
        if "version" in adapted:
            adapted["version"] = cls.to_canonical(adapted["version"])
        return adapted

    @classmethod
    def from_legacy_dict(cls, data: Dict[str, Any]) -> SchemaDefinition:
        """
        Constructs a canonical SchemaDefinition from a dictionary containing a legacy string version.
        """
        from schema_lifecycle.metamodel import SchemaDefinition
        adapted = cls.adapt_dict(data)
        return SchemaDefinition.from_dict(adapted)

    @classmethod
    def to_approved_schema_ref(
        cls,
        schema: SchemaDefinition,
        document_type: Any,
        is_active: bool = True,
    ) -> Any:
        """
        Constructs an ApprovedSchemaRef from a canonical SchemaDefinition,
        formatting the integer version to the string contract expected by doc_extractor.
        """
        from doc_extractor.classification.base import ApprovedSchemaRef
        return ApprovedSchemaRef(
            schema_id=schema.schema_id,
            schema_version=cls.to_semver(schema.version),
            document_type=document_type,
            is_active=is_active,
        )

    @classmethod
    def from_approved_schema_ref(cls, ref: Any) -> Tuple[str, int]:
        """
        Extracts (schema_id, canonical_version) from an ApprovedSchemaRef instance.
        """
        canonical_version = cls.to_canonical(ref.schema_version)
        return ref.schema_id, canonical_version

