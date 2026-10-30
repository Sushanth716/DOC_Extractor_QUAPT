"""
Dynamic Extraction Adapter (PRD SCH-007, API-007, US-DOCX-010).
Integrates:
1. SchemaLifecycleManager.bind_test_schema()
2. DynamicSchemaCompiler
3. SchemaGuidedExtractor (strict no-fallback, trusted provenance)
4. FieldAssemblyService (shared evidence grounding, confidence calibration, null handling)
5. RuleEvaluator (dynamic business and cross-field rules)
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Union

from doc_extractor.contracts_b import ModuleBOutputPayload
from doc_extractor.contracts_c import (
    CanonicalExtractionEnvelope,
    ExtractedField,
    FlagItem,
)
from doc_extractor.exceptions import LLMExecutionError
from doc_extractor.extractor import SchemaGuidedExtractor
from doc_extractor.field_assembly import (
    assemble_canonical_envelope,
    assemble_field_entry,
    find_null_reason_signal,
)
from doc_extractor.rule_evaluator import evaluate_dynamic_rules
from doc_extractor.validation_rules import (
    normalize_currency_amount,
    normalize_date_to_iso,
    validate_aadhaar,
    validate_ifsc,
    validate_pan,
)
from schema_lifecycle.compiler import CompiledFieldMetadata, DynamicSchemaCompiler
from schema_lifecycle.enums import FieldType
from schema_lifecycle.manager import SchemaLifecycleManager


class DynamicExtractionAdapter:
    """
    Adapter executing dynamic schema extraction and canonical envelope assembly.
    Strictly isolated from static document routes and production schema activation.
    """

    def __init__(
        self,
        manager: SchemaLifecycleManager,
        extractor: Optional[SchemaGuidedExtractor] = None,
        allow_mock: bool = False,
    ):
        self.manager = manager
        self.extractor = extractor or SchemaGuidedExtractor(use_llm=True, allow_fallback=False)
        self.allow_mock = allow_mock

    def extract_and_assemble(
        self,
        tenant_id: str,
        schema_id: str,
        version: Union[int, str],
        payload: ModuleBOutputPayload,
    ) -> CanonicalExtractionEnvelope:
        """
        Executes end-to-end dynamic extraction against an exact testing schema version.

        Sequence:
        1. Binds exact test schema (status == TESTING) via SchemaLifecycleManager.
        2. Compiles SchemaDefinition into runtime Pydantic model via DynamicSchemaCompiler.
        3. Executes SchemaGuidedExtractor with strict allow_fallback=False.
        4. Verifies genuine LLM provenance (or mock if explicitly allowed).
        5. Flattens extracted model and resolves fields to CompiledFieldMetadata.
        6. Evaluates dynamic cross-field rules via RuleEvaluator.
        7. Assembles ExtractedField envelopes with evidence grounding and confidence calibration.
        8. Applies document status gating and emits CanonicalExtractionEnvelope.
        """
        # 1. Resolve exact schema snapshot in TESTING status (SCH-007)
        snapshot = self.manager.bind_test_schema(
            tenant_id=tenant_id,
            schema_id=schema_id,
            version=version,
        )

        # 2. Compile schema definition to runtime model
        compiled_model_cls = DynamicSchemaCompiler.compile(snapshot.definition)
        meta_dict: Dict[str, CompiledFieldMetadata] = getattr(compiled_model_cls, "__field_metadata__", {})
        rules = getattr(compiled_model_cls, "__rules__", [])

        # 3. Execute extraction with strict no-fallback
        extraction_result = self.extractor.extract_with_provenance(
            payload=payload,
            schema_cls=compiled_model_cls,
        )
        extracted_model = extraction_result.payload
        provenance = extraction_result.provenance

        # 4. Verify execution provenance integrity (SCH-007)
        if not provenance.is_trusted_for_benchmark(allow_mock=self.allow_mock):
            raise LLMExecutionError(
                message=(
                    f"Extraction returned untrusted provenance mode '{provenance.mode.value}' "
                    "for benchmark evaluation (SCH-007)."
                ),
                model_name=provenance.model_name,
                details={
                    "mode": provenance.mode.value,
                    "fallback_occurred": provenance.fallback_occurred,
                    "fallback_reason": provenance.fallback_reason,
                },
            )

        # 5. Flatten extracted model into canonical dot-paths
        flat_dict = DynamicSchemaCompiler.flatten(extracted_model)

        fields_envelope: Dict[str, ExtractedField] = {}
        flags: List[FlagItem] = []

        def resolve_metadata(path: str) -> Optional[CompiledFieldMetadata]:
            if path in meta_dict:
                return meta_dict[path]
            # Replace array indices with [] notation
            bracket_path = re.sub(r"\.\d+(\.|$)", r"[]\1", path)
            if bracket_path in meta_dict:
                return meta_dict[bracket_path]
            bracket_path_alt = re.sub(r"\.\d+", "[]", path)
            if bracket_path_alt in meta_dict:
                return meta_dict[bracket_path_alt]
            return None

        # 6. Process all fields present in flat_dict
        processed_declared_paths = set()
        for field_path, raw_value in flat_dict.items():
            meta = resolve_metadata(field_path)
            if meta:
                processed_declared_paths.add(meta.field_path)

            val, err = self._validate_and_normalize_field(raw_value, meta)

            null_reason = None
            if val is None:
                null_reason = find_null_reason_signal(field_path, payload.pages)

            field_entry = assemble_field_entry(
                field_name=field_path,
                value=val,
                pages=payload.pages,
                flags=flags,
                error=err,
                required=meta.required if meta else False,
                confidence_threshold=meta.confidence_threshold if (meta and meta.confidence_threshold is not None) else 0.75,
                evidence_required=meta.evidence_required if meta else True,
                explicit_null_reason=null_reason,
                strict_boolean_evidence=True,
            )
            fields_envelope[field_path] = field_entry

        # Account for declared scalar/top-level fields omitted from extraction
        for declared_path, meta in meta_dict.items():
            if declared_path not in processed_declared_paths and not ("[]" in declared_path) and meta.field_type != FieldType.OBJECT:
                null_reason = find_null_reason_signal(declared_path, payload.pages)
                fields_envelope[declared_path] = assemble_field_entry(
                    field_name=declared_path,
                    value=None,
                    pages=payload.pages,
                    flags=flags,
                    required=meta.required,
                    confidence_threshold=meta.confidence_threshold if meta.confidence_threshold is not None else 0.75,
                    evidence_required=meta.evidence_required,
                    explicit_null_reason=null_reason,
                    strict_boolean_evidence=True,
                )

        # 7. Evaluate dynamic cross-field rules
        evaluate_dynamic_rules(
            rules=rules,
            flat_dict=flat_dict,
            fields_envelope=fields_envelope,
            flags=flags,
        )

        # 8. Assemble canonical output envelope with exact bound version
        return assemble_canonical_envelope(
            document_id=payload.document_id,
            document_type=payload.classification.document_type.value,
            schema_id=snapshot.schema_id,
            schema_version=snapshot.version,
            classification_confidence=payload.classification.classification_confidence,
            page_count=len(payload.pages),
            fields=fields_envelope,
            flags=flags,
            run_id=payload.run_id,
            parser_model_name=payload.parser_model_name,
            extractor_model_name=getattr(self.extractor, "model_name", "unknown"),
            prompt_version=getattr(self.extractor, "prompt_version", "1.0.0"),
            provenance=provenance,
            payload_status=payload.status,
        )

    def _validate_and_normalize_field(
        self,
        value: Any,
        meta: Optional[CompiledFieldMetadata],
    ) -> tuple[Any, Optional[str]]:
        """Applies format and type-specific normalization and validation rules."""
        if value is None or meta is None:
            return value, None

        val = value
        err = None

        # Date normalization
        if meta.field_type == FieldType.DATE:
            norm_date, date_err = normalize_date_to_iso(str(val))
            if norm_date:
                val = norm_date
            if date_err:
                err = date_err

        # Currency / Decimal normalization
        elif meta.field_type in (FieldType.DECIMAL, FieldType.INTEGER):
            if isinstance(val, str) and (re.search(r"[,\$₹£€]|INR|USD", val)):
                norm_amt, _, amt_err = normalize_currency_amount(val)
                if norm_amt is not None:
                    val = int(norm_amt) if meta.field_type == FieldType.INTEGER else norm_amt
                if amt_err:
                    err = amt_err

        # Standard or Regex Format validation
        fmt = meta.format
        if fmt:
            fmt_upper = fmt.upper()
            if "PAN" in fmt_upper or fmt == "^[A-Z]{5}[0-9]{4}[A-Z]$":
                val, err = validate_pan(str(val))
            elif "AADHAAR" in fmt_upper:
                val, err = validate_aadhaar(str(val))
            elif "IFSC" in fmt_upper or fmt == "^[A-Z]{4}0[A-Z0-9]{6}$":
                val, err = validate_ifsc(str(val))
            else:
                try:
                    if not re.search(fmt, str(val)):
                        err = "INVALID_FORMAT"
                except re.error:
                    pass

        return val, err
