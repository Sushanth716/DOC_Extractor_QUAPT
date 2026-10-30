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


class LifecycleSchemaRegistryAdapter:
    """
    Adapter integrating SchemaLifecycleManager into the existing SchemaRegistry contract.
    Strictly conforms to SchemaRegistry interface without modifying its public contract.
    Enforces SCH-004 requirement that only approved schemas are resolved for production.
    """

    DEFAULT_MVP_MAPPINGS: Dict[Any, str] = {}

    def __init__(
        self,
        manager: Any,
        default_tenant: str = "default",
        document_type_mappings: Optional[Dict[Any, str]] = None,
    ) -> None:
        from doc_extractor.classification.base import SchemaRegistry
        from doc_extractor.contracts_b import MVPDocumentType

        if not isinstance(self, SchemaRegistry):
            # Dynamic mixin to satisfy isinstance(self, SchemaRegistry)
            self.__class__ = type(
                "LifecycleSchemaRegistryAdapter",
                (LifecycleSchemaRegistryAdapter, SchemaRegistry),
                {},
            )

        self.manager = manager
        self.default_tenant = default_tenant

        default_map = {
            MVPDocumentType.DOC_01_CASA: "DOC-01",
            MVPDocumentType.DOC_02_INSURANCE_APP: "DOC-02",
            MVPDocumentType.DOC_03_DISCHARGE_SUMMARY: "DOC-03",
        }
        self.mappings = dict(document_type_mappings if document_type_mappings is not None else default_map)

    def resolve_approved_schema(
        self,
        document_type: Any,
        tenant_id: Optional[str] = None,
    ) -> Any:
        from doc_extractor.classification.base import SchemaResolutionError
        from doc_extractor.contracts_b import MVPDocumentType
        from schema_lifecycle.exceptions import SchemaNotApprovedError, SchemaNotFoundError

        if document_type == MVPDocumentType.UNKNOWN:
            raise SchemaResolutionError(
                message="Cannot resolve schema for UNKNOWN document type (FR-CLS-002).",
                document_type=document_type.value,
                code="UNKNOWN_DOC_TYPE_NO_SCHEMA",
            )

        schema_id = self.mappings.get(document_type)
        if not schema_id:
            raise SchemaResolutionError(
                message=f"No schema mapping configured for document type '{document_type.value}'.",
                document_type=document_type.value,
                code="NO_APPROVED_SCHEMA",
            )

        tid = tenant_id or self.default_tenant
        try:
            snapshot = self.manager.resolve_active_schema(tenant_id=tid, schema_id=schema_id)
        except (SchemaNotApprovedError, SchemaNotFoundError) as exc:
            raise SchemaResolutionError(
                message=f"No approved schema active for document type '{document_type.value}': {exc.message}",
                document_type=document_type.value,
                code="NO_APPROVED_SCHEMA",
            ) from exc

        return SchemaVersionAdapter.to_approved_schema_ref(
            schema=snapshot.definition,
            document_type=document_type,
            is_active=True,
        )

    def is_schema_approved(
        self,
        schema_id: str,
        schema_version: str,
        tenant_id: Optional[str] = None,
    ) -> bool:
        from schema_lifecycle.enums import SchemaStatus

        tid = tenant_id or self.default_tenant
        try:
            canonical_ver = SchemaVersionAdapter.to_canonical(schema_version)
        except Exception:
            return False

        snapshot = self.manager.repository.get_snapshot(schema_id, canonical_ver, tid)
        if snapshot is None:
            return False
        return snapshot.status == SchemaStatus.APPROVED
