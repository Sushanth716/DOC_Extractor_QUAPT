"""
Dataset and Metric Contracts for Schema Test Harness (SCH-007, US-DOCX-010).
Strictly adheres to PRD v1.1 §13.1, §13.3, §6.1, and §6.3:
- Supports dataset splits: development, tuning, locked_test (PRD §13.1).
- Distinct annotators and adjudicated ground truth for gate-grade evaluation.
- Document content kept outside metadata via controlled document references.
- Per-field and aggregate metric contracts with 10-bin calibration and false-fill rate.
"""

from __future__ import annotations

from enum import Enum
import re
from typing import Any, Dict, List, Literal, Optional, Set, Union
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


class DatasetSplit(str, Enum):
    """PRD §13.1 benchmark evaluation splits."""
    DEVELOPMENT = "development"
    TUNING = "tuning"
    LOCKED_TEST = "locked_test"


class MatchStrategy(str, Enum):
    """PRD §13.3 field matching strategies."""
    EXACT_NORMALIZED = "exact_normalized"    # Identifiers, dates, amounts, enums
    NUMERIC_TOLERANCE = "numeric_tolerance"  # Currency amounts, percentages, floating point
    TOKEN_F1 = "token_f1"                    # Free text (e.g. diagnosis, notes)
    BOOLEAN_EXACT = "boolean_exact"          # Checkboxes and boolean flags


class GroundTruthField(BaseModel):
    """
    Ground truth specification for a single schema field (PRD §13.1, §13.3).
    Supports expected values, null-if-absent (GR-003), explicit key-field designation (M1),
    and matching rules.
    """
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    field_path: str = Field(..., description="Canonical dot or bracket path of the field (e.g. 'applicant.full_name')")
    expected_value: Optional[Any] = Field(default=None, description="Expected normalized value; None if field is absent/null")
    field_type: Optional[str] = Field(
        default=None,
        description="Field data type: 'string', 'integer', 'decimal', 'date', 'enum', 'boolean' (PRD §6.1)",
    )
    normalization_profile: Optional[str] = Field(
        default=None,
        description="Explicit normalization profile: 'identifier', 'numeric', 'date', 'case_sensitive_identifier', etc.",
    )
    date_format: Optional[str] = Field(
        default=None,
        description="Explicit declared date format (e.g. '%Y-%m-%d', '%d/%m/%Y', '%d-%b-%Y')",
    )
    is_null: bool = Field(default=False, description="True if field is expected to be null/absent in the document (GR-003)")
    is_key_field: bool = Field(
        default=False,
        description="Explicit flag indicating if field is a key field subject to M1 (≥95%) and M11 escapes",
    )
    match_strategy: MatchStrategy = Field(default=MatchStrategy.EXACT_NORMALIZED, description="Matching rule to apply (PRD §13.3)")
    tolerance: Optional[float] = Field(default=0.01, ge=0.0, description="Tolerance for numeric comparisons")
    token_f1_threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Field-specific token-F1 acceptance threshold; if None, uses policy default",
    )
    description: Optional[str] = Field(default=None, description="Optional annotation or provenance note")

    @field_validator("field_path")
    @classmethod
    def validate_field_path(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("field_path must be a non-empty string")
        return v.strip()

    @model_validator(mode="before")
    @classmethod
    def validate_null_consistency_before(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if data.get("expected_value") is None:
                data["is_null"] = True
            elif data.get("is_null") is True:
                fpath = data.get("field_path", "unknown")
                raise ValueError(
                    f"Field '{fpath}' has is_null=True but expected_value is not None ({data.get('expected_value')!r})"
                )
        return data

    @model_validator(mode="after")
    def validate_null_consistency(self) -> "GroundTruthField":
        if self.expected_value is None and not self.is_null:
            object.__setattr__(self, "is_null", True)
        elif self.is_null and self.expected_value is not None:
            raise ValueError(
                f"Field '{self.field_path}' has is_null=True but expected_value is not None ({self.expected_value!r})"
            )
        return self


class GroundTruthDocument(BaseModel):
    """
    Double-keyed ground truth document annotation (PRD §13.1).
    Preserves document hash, annotator identities, and adjudication status.
    """
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    sample_id: str = Field(..., description="Unique sample identifier")
    document_hash_sha256: str = Field(..., description="SHA-256 hexadecimal digest of document artifact")
    annotator_ids: List[str] = Field(..., min_length=1, description="Distinct annotator identities (PRD §13.1)")
    adjudicated: bool = Field(default=False, description="True if annotations have been adjudicated between annotators")
    adjudicator_id: Optional[str] = Field(default=None, description="Identifier of the adjudicator if adjudicated")
    fields: Dict[str, GroundTruthField] = Field(default_factory=dict, description="Ground truth fields mapped by field_path")

    @field_validator("sample_id")
    @classmethod
    def validate_sample_id(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("sample_id must be a non-empty string")
        return v.strip()

    @field_validator("document_hash_sha256")
    @classmethod
    def validate_hash_format(cls, v: str) -> str:
        clean = v.strip().lower()
        if not re.match(r"^[0-9a-f]{64}$", clean):
            raise ValueError(f"document_hash_sha256 must be a 64-character hexadecimal SHA-256 string, got {v!r}")
        return clean

    @field_validator("annotator_ids")
    @classmethod
    def validate_annotator_ids(cls, v: List[str]) -> List[str]:
        clean_ids = [aid.strip() for aid in v if isinstance(aid, str) and aid.strip()]
        if len(clean_ids) != len(v):
            raise ValueError("annotator_ids cannot contain empty or non-string identifiers")
        if len(set(clean_ids)) != len(clean_ids):
            raise ValueError(f"annotator_ids must be distinct; duplicate IDs found: {v}")
        return clean_ids

    def validate_for_gate(self) -> None:
        """
        PRD §13.1: Gate-grade ground truth requires double-keying by at least two distinct
        annotators and formal adjudication.
        """
        if len(self.annotator_ids) < 2:
            raise ValueError(
                f"Gate-grade ground truth for sample '{self.sample_id}' requires at least 2 distinct annotators; "
                f"got {len(self.annotator_ids)}: {self.annotator_ids}"
            )
        if not self.adjudicated:
            raise ValueError(
                f"Gate-grade ground truth for sample '{self.sample_id}' must be marked as adjudicated (adjudicated=True)"
            )


class EvaluationSample(BaseModel):
    """
    Evaluation sample referencing a document artifact without embedding raw contents (PRD §13.1).
    Preserves tenant isolation, stratum classification, and references ground truth.
    """
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    sample_id: str = Field(..., description="Unique sample identifier")
    tenant_id: str = Field(..., description="Tenant identifier for multi-tenant isolation")
    document_type: str = Field(..., description="Document type name or code (e.g. DOC-01)")
    document_ref: str = Field(..., description="Controlled URI / storage reference to document artifact")
    document_hash_sha256: str = Field(..., description="SHA-256 hexadecimal digest of document artifact")
    stratum: Optional[str] = Field(
        default=None,
        description="Stratum category (clean_print, skewed_scan, character_grid, handwriting per PRD §13.1)",
    )
    ground_truth: GroundTruthDocument = Field(..., description="Ground truth annotations for this sample")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="PII-safe sample metadata")

    @field_validator("sample_id", "tenant_id", "document_type", "document_ref")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("Identifier fields must be non-empty strings")
        return v.strip()

    @field_validator("document_hash_sha256")
    @classmethod
    def validate_hash_format(cls, v: str) -> str:
        clean = v.strip().lower()
        if not re.match(r"^[0-9a-f]{64}$", clean):
            raise ValueError(f"document_hash_sha256 must be a 64-character hexadecimal SHA-256 string, got {v!r}")
        return clean

    @model_validator(mode="after")
    def validate_sample_alignment(self) -> "EvaluationSample":
        if self.sample_id != self.ground_truth.sample_id:
            raise ValueError(
                f"sample_id mismatch: sample has '{self.sample_id}', ground truth has '{self.ground_truth.sample_id}'"
            )
        if self.document_hash_sha256 != self.ground_truth.document_hash_sha256:
            raise ValueError(
                f"document_hash_sha256 mismatch: sample has '{self.document_hash_sha256}', "
                f"ground truth has '{self.ground_truth.document_hash_sha256}'"
            )
        return self


class EvaluationDataset(BaseModel):
    """
    Collection of evaluation samples bound to a schema, version, tenant, and split (PRD §13.1).
    Enforces gate-grade validation when dataset_split is locked_test.
    """
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    dataset_id: str = Field(..., description="Unique dataset identifier")
    tenant_id: str = Field(..., description="Tenant identifier for multi-tenant isolation")
    schema_id: str = Field(..., description="Bound schema identifier")
    schema_version: int = Field(..., ge=1, description="Canonical integer schema version (SCH-006)")
    document_type: str = Field(..., description="Document type name or code (e.g. DOC-01)")
    dataset_split: DatasetSplit = Field(..., description="Dataset split (development, tuning, locked_test)")
    samples: List[EvaluationSample] = Field(..., min_length=1, description="List of evaluation samples")
    description: Optional[str] = Field(default=None, description="Dataset description or origin")

    @field_validator("dataset_id", "tenant_id", "schema_id", "document_type")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("Dataset identifiers must be non-empty strings")
        return v.strip()

    @model_validator(mode="after")
    def validate_dataset_consistency(self) -> "EvaluationDataset":
        seen_sample_ids: Set[str] = set()
        for idx, sample in enumerate(self.samples):
            if sample.tenant_id != self.tenant_id:
                raise ValueError(
                    f"Sample at index {idx} ('{sample.sample_id}') has tenant_id '{sample.tenant_id}' "
                    f"which does not match dataset tenant_id '{self.tenant_id}'"
                )
            if sample.document_type != self.document_type:
                raise ValueError(
                    f"Sample at index {idx} ('{sample.sample_id}') has document_type '{sample.document_type}' "
                    f"which does not match dataset document_type '{self.document_type}'"
                )
            if sample.sample_id in seen_sample_ids:
                raise ValueError(f"Duplicate sample_id '{sample.sample_id}' in dataset '{self.dataset_id}'")
            seen_sample_ids.add(sample.sample_id)

            # Gate-grade enforcement for locked_test split (PRD §13.1)
            if self.dataset_split == DatasetSplit.LOCKED_TEST:
                sample.ground_truth.validate_for_gate()

        return self


# ============================================================================
# Prediction Contracts
# ============================================================================

class FieldPrediction(BaseModel):
    """Extracted field prediction produced by an extraction run."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    field_path: str = Field(..., description="Canonical dot or bracket path of the field")
    predicted_value: Optional[Any] = Field(default=None, description="Extracted value; None if predicted absent/null")
    confidence: float = Field(ge=0.0, le=1.0, description="Stated calibrated confidence score in [0.0, 1.0]")
    is_null: bool = Field(default=False, description="Flag indicating model predicted null/absent")

    @field_validator("field_path")
    @classmethod
    def validate_field_path(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("field_path must be a non-empty string")
        return v.strip()

    @model_validator(mode="before")
    @classmethod
    def validate_null_consistency_before(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if data.get("predicted_value") is None:
                data["is_null"] = True
            elif data.get("is_null") is True:
                fpath = data.get("field_path", "unknown")
                raise ValueError(
                    f"FieldPrediction for '{fpath}' has is_null=True but predicted_value is not None ({data.get('predicted_value')!r})"
                )
        return data

    @model_validator(mode="after")
    def validate_null_consistency(self) -> "FieldPrediction":
        if self.predicted_value is None and not self.is_null:
            object.__setattr__(self, "is_null", True)
        elif self.is_null and self.predicted_value is not None:
            raise ValueError(
                f"FieldPrediction for '{self.field_path}' has is_null=True but predicted_value is not None ({self.predicted_value!r})"
            )
        return self


class SamplePrediction(BaseModel):
    """Complete document-level predictions for a single sample."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    sample_id: str = Field(..., description="Sample identifier matching EvaluationSample.sample_id")
    document_status: str = Field(
        default="auto_approved",
        description="Terminal pipeline status: auto_approved, needs_review, failed (SCH-012, GR-027)",
    )
    predictions: Dict[str, FieldPrediction] = Field(default_factory=dict, description="Field predictions mapped by field_path")

    @field_validator("sample_id")
    @classmethod
    def validate_sample_id(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("sample_id must be a non-empty string")
        return v.strip()


# ============================================================================
# Metric Contracts
# ============================================================================

class ConfidenceBinMetric(BaseModel):
    """Per-bin confidence calibration measurement (PRD §13.3)."""
    model_config = ConfigDict(frozen=True)

    bin_index: int = Field(ge=0, le=9, description="0-based bin index [0..9]")
    bin_lower: float = Field(ge=0.0, le=1.0)
    bin_upper: float = Field(ge=0.0, le=1.0)
    sample_count: int = Field(ge=0)
    mean_confidence: float = Field(ge=0.0, le=1.0)
    accuracy: float = Field(ge=0.0, le=1.0)
    calibration_gap: float = Field(ge=0.0, le=1.0)


class FieldMetrics(BaseModel):
    """
    Per-field evaluation metrics (SCH-007, PRD §13.3).
    Reports precision, recall, F1, false-fill rate, confusion counts, calibration gap,
    and raw token-F1 score for free-text fields.
    """
    model_config = ConfigDict(frozen=True)

    field_path: str = Field(..., description="Canonical field path")
    is_key_field: bool = Field(..., description="Flag indicating if field is a key field (M1)")
    field_type: Optional[str] = Field(default=None, description="Field data type if specified")
    precision: float = Field(ge=0.0, le=1.0, description="Precision: TP / (TP + FP)")
    recall: float = Field(ge=0.0, le=1.0, description="Recall: TP / (TP + FN)")
    f1: float = Field(ge=0.0, le=1.0, description="Field F1: Harmonic mean of precision and recall (M1)")
    false_fill_rate: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="False-fill rate: FP on null ground truth / null support; None if null_support == 0 (undefined/insufficient null cases per PRD §13.3)",
    )
    true_positives: int = Field(ge=0, description="Correct non-null extractions")
    false_positives: int = Field(ge=0, description="Incorrect extractions + extractions on null ground truth")
    false_negatives: int = Field(ge=0, description="Omissions of non-null fields + incorrect extractions")
    true_negatives: int = Field(ge=0, description="Correct null predictions on null ground truth")
    support: int = Field(ge=0, description="Number of non-null ground-truth occurrences")
    null_support: int = Field(ge=0, description="Number of null ground-truth occurrences")
    mean_token_f1: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Raw average token-F1 score across evaluated instances for free-text fields",
    )
    has_sufficient_data: bool = Field(
        default=True,
        description="False if field has zero evaluated instances in ground truth (support==0 and null_support==0)",
    )
    calibration_gap_points: float = Field(
        ge=0.0,
        description="10-bin weighted calibration gap in percentage points (PRD M6)",
    )
    bin_metrics: List[ConfidenceBinMetric] = Field(default_factory=list, description="10-bin breakdown")


class AggregateMetrics(BaseModel):
    """
    Dataset-wide aggregate metrics (M1, M6, M11, SCH-007).
    Reports both macro-averaged and micro-averaged metrics with escape counts.
    Explicitly tracks evaluation data sufficiency and the primary aggregation method.
    """
    model_config = ConfigDict(frozen=True)

    # Aggregation method applied for M1 evaluation (provisional default: macro)
    primary_aggregation: Literal["macro", "micro"] = Field(
        default="macro",
        description="Aggregation method selected for M1: 'macro' (provisional default) or 'micro'",
    )

    # Macro metrics (unweighted mean across fields, provisional default for schema gating)
    macro_f1_all_fields: float = Field(ge=0.0, le=1.0, description="Unweighted mean F1 across all fields (M1)")
    macro_f1_key_fields: float = Field(ge=0.0, le=1.0, description="Unweighted mean F1 across key fields (M1)")
    macro_f1_non_key_fields: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    macro_precision_all: float = Field(ge=0.0, le=1.0)
    macro_recall_all: float = Field(ge=0.0, le=1.0)

    # Micro metrics (pooled instance-level counts)
    micro_f1_all_fields: float = Field(ge=0.0, le=1.0, description="Pooled micro F1 across all fields")
    micro_f1_key_fields: float = Field(ge=0.0, le=1.0, description="Pooled micro F1 across key fields")
    micro_precision_all: float = Field(ge=0.0, le=1.0)
    micro_recall_all: float = Field(ge=0.0, le=1.0)

    # Calibration & Error metrics
    mean_calibration_gap_points: float = Field(ge=0.0, description="Unweighted mean of per-field calibration gaps")
    dataset_calibration_gap_points: Optional[float] = Field(
        default=None,
        description="Dataset-wide 10-bin calibration gap in points (M6); None if calibration population is empty",
    )
    calibration_sample_count: int = Field(ge=0, description="Total predictions evaluated for calibration")
    total_null_support: int = Field(ge=0, default=0, description="Total null ground-truth occurrences across all fields")
    total_false_fills: int = Field(ge=0, default=0, description="Total false positive extractions against null ground truth")
    overall_false_fill_rate: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Pooled false-fill rate; None if total_null_support == 0 (undefined/insufficient null cases)",
    )

    # Data sufficiency tracking
    has_insufficient_data: bool = Field(
        default=False,
        description="True if evaluation data is insufficient (missing key fields, empty calibration population, etc.)",
    )
    insufficient_data_reasons: List[str] = Field(
        default_factory=list,
        description="Reasons why evaluation data is considered insufficient",
    )

    # Escape evaluation (strictly evaluated on locked_test, None for dev/tuning per PRD M11 / GR-027)
    escape_count: Optional[int] = Field(
        default=None,
        description="Auto-approved documents with incorrect key field; evaluated only on locked_test (M11)",
    )
    total_auto_approved: Optional[int] = Field(default=None, description="Total auto-approved documents in split")

    # Counts
    total_samples: int = Field(ge=0)
    total_field_types: int = Field(ge=0)
    key_field_count: int = Field(ge=0)
    total_predictions: int = Field(ge=0)
    total_tp: int = Field(ge=0)
    total_fp: int = Field(ge=0)
    total_fn: int = Field(ge=0)
    total_tn: int = Field(ge=0)
