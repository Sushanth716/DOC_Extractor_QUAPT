"""
Module C Contracts: Canonical Extraction Envelope and Field Output Schema.
Strictly adheres to PRD §6.3 (SCH-011, SCH-012, SCH-013) and §5.4..5.6 (FR-EXT-003, FR-EXT-004, FR-CNF-001..003).
"""

from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
from pydantic import BaseModel, ConfigDict, Field, field_validator


class ExecutionMode(str, Enum):
    """Execution mode provenance tracking (SCH-007)."""
    GENUINE_LLM = "genuine_llm"
    HEURISTIC = "heuristic"
    HEURISTIC_FALLBACK = "heuristic_fallback"
    MOCK = "mock"


class ExecutionProvenance(BaseModel):
    """
    PRD SCH-007 execution provenance for benchmark and audit verification.
    Created strictly by trusted application runtime code, never by the LLM response.
    Application runtime trust boundary (not cryptographic attestation).
    """
    mode: ExecutionMode = Field(..., description="Actual execution mode used to produce fields")
    model_name: Optional[str] = Field(default=None, description="Actual provider or model identifier")
    provider: Optional[str] = Field(default=None, description="Inference engine/provider (e.g. litellm, mock)")
    prompt_version: str = Field(default="1.0.0", description="Extraction prompt version")
    fallback_occurred: bool = Field(default=False, description="Whether heuristic fallback was triggered")
    fallback_reason: Optional[str] = Field(default=None, description="Exception or error triggering fallback")
    request_id: Optional[str] = Field(default=None, description="Provider completion or request ID when available")
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional execution diagnostics")

    def is_genuine_llm(self) -> bool:
        """True if and only if genuine LLM execution completed without heuristic fallback."""
        return self.mode == ExecutionMode.GENUINE_LLM and not self.fallback_occurred

    def is_trusted_for_benchmark(self, allow_mock: bool = False) -> bool:
        """
        Benchmark verification: rejects heuristic and heuristic fallback.
        Permits mock only if explicitly enabled for deterministic offline testing.
        """
        if self.fallback_occurred:
            return False
        if self.mode == ExecutionMode.GENUINE_LLM:
            return bool(self.model_name and self.model_name.strip())
        if allow_mock and self.mode == ExecutionMode.MOCK:
            return True
        return False


class ExtractionResult(BaseModel):
    """
    Execution wrapper encapsulating extracted payload model and verified execution provenance.
    Separates model output from trusted runtime execution provenance.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    payload: BaseModel
    provenance: ExecutionProvenance



class EvidenceSnippet(BaseModel):
    """
    PRD FR-EXT-004, GR-002, SCH-013:
    Evidence provenance bounding box and source text snippet.
    """
    page: int = Field(ge=1, description="1-based page number")
    bbox: Tuple[float, float, float, float] = Field(
        description="(x0, y0, x1, y1) in pixels of 200 DPI page image (SCH-013)"
    )
    text: str = Field(description="Raw text snippet providing evidence")

    @field_validator("bbox", mode="before")
    @classmethod
    def validate_bbox(cls, v: Any) -> Tuple[float, float, float, float]:
        if not isinstance(v, (list, tuple)) or len(v) != 4:
            raise ValueError(
                f"Bounding box must contain exactly 4 coordinates (x0, y0, x1, y1), got {len(v) if isinstance(v, (list, tuple)) else v}."
            )
        x0, y0, x1, y1 = v
        if float(x1) < float(x0):
            raise ValueError(f"Invalid bounding box: x1 ({x1}) must be >= x0 ({x0}).")
        if float(y1) < float(y0):
            raise ValueError(f"Invalid bounding box: y1 ({y1}) must be >= y0 ({y0}).")
        return (float(x0), float(y0), float(x1), float(y1))


# Alias for backward compatibility
EvidenceModel = EvidenceSnippet


class ExtractedField(BaseModel):
    """
    PRD §6.3 & FR-EXT-003, FR-EXT-004, FR-CNF-001, FR-CNF-002:
    Individual extracted field envelope.
    """
    value: Any = Field(default=None, description="Normalized field value or null if absent/illegible")
    confidence: float = Field(ge=0.0, le=1.0, description="Calibrated confidence score in [0.0, 1.0]")
    null_reason: Optional[Literal["absent", "illegible"]] = Field(
        default=None,
        description="Reason for null value: absent or illegible (FR-EXT-003)"
    )
    review_required: bool = Field(
        default=False,
        description="Flag indicating field requires human review (< 0.75 or validation failure)"
    )
    validation: Literal["passed", "failed", "not_applicable"] = Field(
        default="passed",
        description="Validation outcome: passed, failed, or not_applicable"
    )
    evidence: Optional[EvidenceSnippet] = Field(
        default=None,
        description="Source evidence bounding box and snippet (FR-EXT-004)"
    )


# Alias for backward compatibility
FieldOutput = ExtractedField


class FlagItem(BaseModel):
    """Business rule or anomaly flag (e.g. ARITHMETIC_MISMATCH, CHRONOLOGY_MISMATCH)."""
    code: str = Field(description="Machine-readable error/flag code")
    message: Optional[str] = Field(default=None, description="Human-readable explanation")
    fields: List[str] = Field(default_factory=list, description="Target field paths involved in the flag")


class DocumentMetadata(BaseModel):
    """PRD §6.3 document metadata envelope."""
    document_type: str = Field(description="Document type identifier (e.g. CASA_ACCOUNT_OPENING, DOC-01)")
    schema_id: str = Field(description="Bound schema identifier")
    schema_version: Union[str, int] = Field(description="Version of bound schema (PRD SCH-011)")
    classification_confidence: float = Field(ge=0.0, le=1.0)
    status: Literal["processing", "auto_approved", "needs_review", "reviewed", "rejected", "failed"] = Field(
        description="Terminal document status (SCH-012)"
    )
    page_count: int = Field(ge=0, description="Total number of pages processed")


class RunMetadata(BaseModel):
    """PRD §6.3 run metadata for deterministic auditability (FR-AUD-001)."""
    run_id: str
    model_versions: Dict[str, str] = Field(default_factory=dict)
    prompt_version: str = "1.0.0"
    provenance: Optional[ExecutionProvenance] = Field(
        default=None,
        description="Execution provenance tracking model vs heuristic mode (SCH-007)",
    )


class CanonicalExtractionEnvelope(BaseModel):
    """
    PRD §6.3 Canonical JSON Output Envelope.
    Fully represents extraction result with fields, evidence, flags, and audit metadata.
    """
    document_id: str
    document_metadata: DocumentMetadata
    fields: Dict[str, ExtractedField] = Field(default_factory=dict)
    flags: List[FlagItem] = Field(default_factory=list)
    run: RunMetadata


# Backward compatibility alias
CanonicalOutputEnvelope = CanonicalExtractionEnvelope
