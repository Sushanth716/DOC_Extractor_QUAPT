"""
Test Harness Runner & Benchmark Execution Engine (PRD SCH-007, US-DOCX-010).
Strictly adheres to PRD v1.1 §13.1..13.3, §14 Gate G1, M1, M6, M11, GR-003, GR-005:
- Connects EvaluationDataset, testing schema binding, dynamic Module C extraction adapter,
  trusted execution provenance, and versioned evaluation policies.
- Implements trusted principal-based authorization and cryptographic verification for locked_test split.
- Enforces strict SHA-256 hash verification against actual document artifact bytes or canonical payload.
- Excludes synthetic data from release-gate metrics (evaluates real data exclusively for gate decisions).
- Tracks genuine LLM vs mock vs synthetic data and separates metrics pass from gate eligibility.
- Tracks GR-005 schema conformance runs with explicit 1,000-run boundary reporting.
- Produces deterministic, auditable BenchmarkReport records.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import hmac
import json
import os
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Literal, Optional, Protocol, Set, Tuple, Union
import uuid

from pydantic import BaseModel, ConfigDict, Field, field_validator

from doc_extractor.contracts_b import ModuleBOutputPayload
from doc_extractor.contracts_c import (
    CanonicalExtractionEnvelope,
    ExecutionMode,
    ExecutionProvenance,
    ExtractedField,
)
from doc_extractor.dynamic_adapter import DynamicExtractionAdapter
from doc_extractor.exceptions import LLMExecutionError
from schema_lifecycle.enums import SchemaStatus
from schema_lifecycle.exceptions import SchemaLifecycleError
from schema_lifecycle.manager import SchemaLifecycleManager
from schema_lifecycle.testing.contracts import (
    AggregateMetrics,
    DatasetSplit,
    EvaluationDataset,
    EvaluationSample,
    FieldMetrics,
    FieldPrediction,
    SamplePrediction,
)
from schema_lifecycle.testing.metrics import evaluate_dataset
from schema_lifecycle.testing.policy import EvaluationDecision, EvaluationPolicy


# ============================================================================
# Exceptions
# ============================================================================

class LockedSetAccessError(SchemaLifecycleError):
    """Raised when access to the locked_test dataset split is unauthorized or denied."""

    def __init__(
        self,
        message: str,
        principal_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        dataset_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        merged = dict(details or {})
        if principal_id:
            merged["principal_id"] = principal_id
        if tenant_id:
            merged["tenant_id"] = tenant_id
        if dataset_id:
            merged["dataset_id"] = dataset_id
        super().__init__(message=message, code="LOCKED_SET_ACCESS_DENIED", details=merged)


class DocumentIntegrityError(SchemaLifecycleError):
    """Raised when payload resolution fails or document SHA-256 hash drifts from ground truth."""

    def __init__(
        self,
        message: str,
        sample_id: Optional[str] = None,
        expected_hash: Optional[str] = None,
        actual_hash: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        merged = dict(details or {})
        if sample_id:
            merged["sample_id"] = sample_id
        if expected_hash:
            merged["expected_hash"] = expected_hash
        if actual_hash:
            merged["actual_hash"] = actual_hash
        super().__init__(message=message, code="DOCUMENT_INTEGRITY_ERROR", details=merged)


# ============================================================================
# Authorization Contracts (PRD Check 3)
# ============================================================================

class TrustedPrincipal(BaseModel):
    """
    Principal record maintained within the trusted authorization boundary.
    Cannot be forged or asserted by caller request parameters.
    """
    model_config = ConfigDict(extra="forbid", frozen=True)

    principal_id: str = Field(..., description="Unique principal identifier from trusted auth source")
    roles: List[str] = Field(default_factory=list, description="Roles authoritatively assigned to principal")
    permissions: List[str] = Field(default_factory=list, description="Explicit authoritatively granted permissions")
    tenant_scopes: List[str] = Field(default_factory=list, description="Authorized tenant IDs or ['*']")
    dataset_scopes: List[str] = Field(default_factory=list, description="Authorized dataset IDs or ['*']")
    is_active: bool = Field(default=True, description="Whether principal is active and valid")


class AuthorizationContext(BaseModel):
    """
    Caller context submitted to the test harness.
    Caller provides their claimed principal identity and optional cryptographic token or event reason.
    Caller does NOT self-assert roles or permissions; authorization is resolved
    by the authorizer against the trusted boundary.
    """
    model_config = ConfigDict(extra="forbid", frozen=True)

    principal_id: str = Field(..., description="Unique principal identifier claiming access")
    auth_token: Optional[str] = Field(
        default=None,
        description="Cryptographic HMAC bearer token proving authentication from trusted auth boundary",
    )
    event_reason: Optional[str] = Field(
        default=None,
        description="Formal release gate event justifying locked evaluation (e.g. 'GATE_G1_VERIFICATION')",
    )


class AuthorizationDecision(BaseModel):
    """Result of evaluating AuthorizationContext against locked_test requirements."""
    model_config = ConfigDict(extra="forbid", frozen=True)

    is_authorized: bool = Field(..., description="True if access is granted")
    reason: Optional[str] = Field(default=None, description="Explanation for grant or denial")
    audit_id: Optional[str] = Field(default=None, description="Audit tracking identifier")


class LockedSetAuthorizer(Protocol):
    """Protocol for validating locked_test split evaluation authorization."""

    def authorize(
        self,
        context: AuthorizationContext,
        tenant_id: str,
        dataset_id: str,
        split: DatasetSplit,
    ) -> AuthorizationDecision:
        ...


class DefaultLockedSetAuthorizer:
    """
    Trusted policy-based locked_test authorizer.
    Validates principal against an internal trusted principal registry or cryptographic HMAC token.
    Enforces required permissions, allowed roles, tenant scopes, dataset scopes, and fail-closed security.
    """

    def __init__(
        self,
        trusted_principals: Optional[Dict[str, TrustedPrincipal]] = None,
        secret_key: Optional[str] = None,
        required_permission: str = "EXECUTE_LOCKED_TEST",
        allowed_roles: Optional[List[str]] = None,
    ) -> None:
        self._trusted_principals: Dict[str, TrustedPrincipal] = dict(trusted_principals or {})
        self.secret_key = secret_key
        self.required_permission = required_permission
        self.allowed_roles = allowed_roles or ["GATE_APPROVER", "RELEASE_MANAGER", "SYSTEM_ADMIN"]

    def register_principal(self, principal: TrustedPrincipal) -> None:
        """Register or update a trusted principal within the authorization boundary."""
        self._trusted_principals[principal.principal_id] = principal

    def mint_token(self, principal_id: str, tenant_id: str, dataset_id: str) -> str:
        """Helper for minting an HMAC token from trusted auth boundary."""
        if not self.secret_key:
            raise ValueError("Secret key required to mint cryptographic tokens.")
        msg = f"{principal_id}:{tenant_id}:{dataset_id}"
        sig = hmac.new(self.secret_key.encode("utf-8"), msg.encode("utf-8"), hashlib.sha256).hexdigest()
        return f"{msg}:{sig}"

    def authorize(
        self,
        context: AuthorizationContext,
        tenant_id: str,
        dataset_id: str,
        split: DatasetSplit,
    ) -> AuthorizationDecision:
        if split != DatasetSplit.LOCKED_TEST:
            return AuthorizationDecision(
                is_authorized=True,
                reason="Non-locked split does not require special locked-set authorization.",
            )

        if not context or not context.principal_id or not context.principal_id.strip():
            return AuthorizationDecision(
                is_authorized=False,
                reason="Missing or blank principal_id in AuthorizationContext.",
            )

        # 1. Cryptographic token verification if token is present
        if context.auth_token:
            if not self.secret_key:
                return AuthorizationDecision(
                    is_authorized=False,
                    reason="Cryptographic token provided but authorizer has no secret_key configured.",
                )
            parts = context.auth_token.rsplit(":", 1)
            if len(parts) != 2:
                return AuthorizationDecision(
                    is_authorized=False,
                    reason="Malformed cryptographic auth_token.",
                )
            msg, provided_sig = parts
            expected_sig = hmac.new(self.secret_key.encode("utf-8"), msg.encode("utf-8"), hashlib.sha256).hexdigest()
            if not hmac.compare_digest(provided_sig, expected_sig):
                return AuthorizationDecision(
                    is_authorized=False,
                    reason="Invalid cryptographic signature on auth_token.",
                )
            token_parts = msg.split(":")
            if len(token_parts) != 3:
                return AuthorizationDecision(
                    is_authorized=False,
                    reason="Malformed payload in cryptographic auth_token.",
                )
            tok_principal, tok_tenant, tok_dataset = token_parts
            if tok_principal != context.principal_id:
                return AuthorizationDecision(
                    is_authorized=False,
                    reason=f"Token principal '{tok_principal}' does not match context principal '{context.principal_id}'.",
                )
            if tok_tenant != "*" and tok_tenant != tenant_id:
                return AuthorizationDecision(
                    is_authorized=False,
                    reason=f"Token tenant scope '{tok_tenant}' does not match requested tenant '{tenant_id}'.",
                )
            if tok_dataset != "*" and tok_dataset != dataset_id:
                return AuthorizationDecision(
                    is_authorized=False,
                    reason=f"Token dataset scope '{tok_dataset}' does not match requested dataset '{dataset_id}'.",
                )

        # 2. Lookup principal in trusted registry
        principal = self._trusted_principals.get(context.principal_id)
        if principal is None:
            return AuthorizationDecision(
                is_authorized=False,
                reason=f"Principal '{context.principal_id}' is not recognized in the trusted authorization boundary.",
            )

        if not principal.is_active:
            return AuthorizationDecision(
                is_authorized=False,
                reason=f"Principal '{context.principal_id}' is inactive or revoked.",
            )

        # 3. Permission or Role check
        has_perm = (
            self.required_permission in principal.permissions
            or any(r in self.allowed_roles for r in principal.roles)
        )
        if not has_perm:
            return AuthorizationDecision(
                is_authorized=False,
                reason=(
                    f"Principal '{context.principal_id}' lacks required permission "
                    f"'{self.required_permission}' or allowed roles {self.allowed_roles}."
                ),
            )

        # 4. Tenant scope validation
        if not principal.tenant_scopes or not ("*" in principal.tenant_scopes or tenant_id in principal.tenant_scopes):
            return AuthorizationDecision(
                is_authorized=False,
                reason=f"Principal '{context.principal_id}' is not authorized for tenant '{tenant_id}'.",
            )

        # 5. Dataset scope validation
        if not principal.dataset_scopes or not ("*" in principal.dataset_scopes or dataset_id in principal.dataset_scopes):
            return AuthorizationDecision(
                is_authorized=False,
                reason=f"Principal '{context.principal_id}' is not authorized for dataset '{dataset_id}'.",
            )

        return AuthorizationDecision(
            is_authorized=True,
            reason=f"Principal '{context.principal_id}' granted locked-test execution.",
            audit_id=f"audit_locked_{uuid.uuid4().hex[:12]}",
        )


# ============================================================================
# Payload Resolver Protocol & Implementations (PRD Check 1)
# ============================================================================

class PayloadResolver(Protocol):
    """Protocol for resolving sample document references into ModuleBOutputPayload instances."""

    def resolve_payload(self, sample: EvaluationSample) -> ModuleBOutputPayload:
        ...


class InMemoryPayloadResolver:
    """
    Deterministic in-memory payload resolver for unit tests and synthetic doubles.
    Stores and retrieves payloads and validates SHA-256 against actual source artifact bytes
    or canonical payload representation.
    """

    def __init__(
        self,
        payloads: Optional[Dict[str, ModuleBOutputPayload]] = None,
        source_bytes: Optional[Dict[str, bytes]] = None,
    ) -> None:
        self.payloads: Dict[str, ModuleBOutputPayload] = dict(payloads or {})
        self.source_bytes: Dict[str, bytes] = dict(source_bytes or {})

    def register(
        self,
        key: str,
        payload: ModuleBOutputPayload,
        raw_bytes: Optional[bytes] = None,
    ) -> None:
        self.payloads[key] = payload
        if raw_bytes is not None:
            self.source_bytes[key] = raw_bytes

    def resolve_payload(self, sample: EvaluationSample) -> ModuleBOutputPayload:
        payload = (
            self.payloads.get(sample.sample_id)
            or self.payloads.get(sample.document_ref)
            or self.payloads.get(sample.document_hash_sha256)
        )
        if payload is None:
            raise DocumentIntegrityError(
                f"Missing in-memory payload for sample '{sample.sample_id}' (ref: '{sample.document_ref}').",
                sample_id=sample.sample_id,
            )

        # 1. Canonical payload hashing contract (PRD Check 1)
        if sample.metadata.get("hash_target") == "canonical_payload":
            canonical_bytes = payload.model_dump_json().encode("utf-8")
            actual_hash = hashlib.sha256(canonical_bytes).hexdigest()
            if actual_hash != sample.document_hash_sha256:
                raise DocumentIntegrityError(
                    f"Canonical payload SHA-256 hash mismatch for sample '{sample.sample_id}': "
                    f"expected {sample.document_hash_sha256}, got {actual_hash}.",
                    sample_id=sample.sample_id,
                    expected_hash=sample.document_hash_sha256,
                    actual_hash=actual_hash,
                )
            return payload

        # 2. Actual document artifact byte verification (PRD Check 1)
        raw_bytes = (
            self.source_bytes.get(sample.sample_id)
            or self.source_bytes.get(sample.document_ref)
            or self.source_bytes.get(sample.document_hash_sha256)
            or payload.extra_metadata.get("source_bytes")
        )
        if raw_bytes is None:
            raise DocumentIntegrityError(
                f"Missing source artifact bytes for sample '{sample.sample_id}' (ref: '{sample.document_ref}'). "
                f"Cannot verify SHA-256 document integrity from metadata alone.",
                sample_id=sample.sample_id,
                expected_hash=sample.document_hash_sha256,
            )

        if isinstance(raw_bytes, str):
            raw_bytes = raw_bytes.encode("utf-8")

        actual_hash = hashlib.sha256(raw_bytes).hexdigest()
        if actual_hash != sample.document_hash_sha256:
            raise DocumentIntegrityError(
                f"Document artifact SHA-256 hash mismatch for sample '{sample.sample_id}': "
                f"expected {sample.document_hash_sha256}, got actual byte hash {actual_hash}.",
                sample_id=sample.sample_id,
                expected_hash=sample.document_hash_sha256,
                actual_hash=actual_hash,
            )

        # 3. Reject metadata claiming expected hash when actual bytes differ or contradict
        embedded_hash = (
            payload.extra_metadata.get("document_hash_sha256")
            or payload.extra_metadata.get("file_hash_sha256")
        )
        if embedded_hash and embedded_hash != actual_hash:
            raise DocumentIntegrityError(
                f"Untrusted payload metadata: metadata claims hash '{embedded_hash}' "
                f"which contradicts actual artifact hash '{actual_hash}'.",
                sample_id=sample.sample_id,
                expected_hash=sample.document_hash_sha256,
                actual_hash=embedded_hash,
            )

        return payload


class LocalPayloadResolver:
    """
    Secure filesystem payload resolver constrained to a configured dataset root directory.
    Rejects path traversal, unapproved absolute paths, symlink escapes, missing files,
    and SHA-256 payload/artifact content hash mismatches against actual source bytes.
    """

    def __init__(self, base_dir: Union[str, Path]) -> None:
        self.base_dir = Path(base_dir).resolve()
        if not self.base_dir.is_dir():
            raise ValueError(f"LocalPayloadResolver base_dir must be an existing directory: {self.base_dir}")

    def resolve_payload(self, sample: EvaluationSample) -> ModuleBOutputPayload:
        doc_ref = sample.document_ref.strip()
        if doc_ref.startswith("file://"):
            doc_ref = doc_ref[7:]
        elif doc_ref.startswith("storage://"):
            doc_ref = doc_ref[10:]

        ref_path = Path(doc_ref)
        if ref_path.is_absolute():
            resolved = ref_path.resolve()
        else:
            resolved = (self.base_dir / ref_path).resolve()

        # Enforce boundary security (no path traversal, no symlink escape outside base_dir)
        try:
            resolved.relative_to(self.base_dir)
        except ValueError:
            raise DocumentIntegrityError(
                f"Path traversal or symlink escape detected: reference '{sample.document_ref}' "
                f"resolves to '{resolved}' which is outside base directory '{self.base_dir}'.",
                sample_id=sample.sample_id,
            )

        if not resolved.is_file():
            raise DocumentIntegrityError(
                f"Payload/artifact file not found for sample '{sample.sample_id}': '{resolved}'.",
                sample_id=sample.sample_id,
            )

        try:
            content_bytes = resolved.read_bytes()
        except Exception as exc:
            raise DocumentIntegrityError(
                f"Failed to read payload file '{resolved}' for sample '{sample.sample_id}': {exc}",
                sample_id=sample.sample_id,
            ) from exc

        # Calculate actual SHA-256 of file contents
        actual_file_hash = hashlib.sha256(content_bytes).hexdigest()

        try:
            payload = ModuleBOutputPayload.model_validate_json(content_bytes)
        except Exception as exc:
            raise DocumentIntegrityError(
                f"Failed to parse ModuleBOutputPayload from '{resolved}' for sample '{sample.sample_id}': {exc}",
                sample_id=sample.sample_id,
            ) from exc

        # Check explicit canonical payload contract vs actual artifact file/byte hashing
        if sample.metadata.get("hash_target") == "canonical_payload":
            canonical_bytes = payload.model_dump_json().encode("utf-8")
            canonical_hash = hashlib.sha256(canonical_bytes).hexdigest()
            if canonical_hash != sample.document_hash_sha256:
                raise DocumentIntegrityError(
                    f"Canonical payload SHA-256 hash mismatch for sample '{sample.sample_id}': "
                    f"expected {sample.document_hash_sha256}, got {canonical_hash}.",
                    sample_id=sample.sample_id,
                    expected_hash=sample.document_hash_sha256,
                    actual_hash=canonical_hash,
                )
        else:
            raw_source_bytes = payload.extra_metadata.get("source_bytes")
            if raw_source_bytes is not None:
                if isinstance(raw_source_bytes, str):
                    raw_source_bytes = raw_source_bytes.encode("utf-8")
                actual_artifact_hash = hashlib.sha256(raw_source_bytes).hexdigest()
                if actual_artifact_hash != sample.document_hash_sha256:
                    raise DocumentIntegrityError(
                        f"Actual artifact SHA-256 hash mismatch for sample '{sample.sample_id}': "
                        f"expected {sample.document_hash_sha256}, got actual artifact hash {actual_artifact_hash}.",
                        sample_id=sample.sample_id,
                        expected_hash=sample.document_hash_sha256,
                        actual_hash=actual_artifact_hash,
                    )
                embedded_hash = payload.extra_metadata.get("document_hash_sha256")
                if embedded_hash and embedded_hash != actual_artifact_hash:
                    raise DocumentIntegrityError(
                        f"Untrusted payload metadata: metadata claims hash '{embedded_hash}' "
                        f"which contradicts actual artifact hash '{actual_artifact_hash}'.",
                        sample_id=sample.sample_id,
                        expected_hash=sample.document_hash_sha256,
                        actual_hash=embedded_hash,
                    )
            else:
                # File on disk is treated as the source document artifact
                if actual_file_hash != sample.document_hash_sha256:
                    raise DocumentIntegrityError(
                        f"Actual file SHA-256 hash mismatch for sample '{sample.sample_id}': "
                        f"expected {sample.document_hash_sha256}, got actual file hash {actual_file_hash}.",
                        sample_id=sample.sample_id,
                        expected_hash=sample.document_hash_sha256,
                        actual_hash=actual_file_hash,
                    )

                embedded_hash = (
                    payload.extra_metadata.get("document_hash_sha256")
                    or payload.extra_metadata.get("file_hash_sha256")
                )
                if embedded_hash and embedded_hash != actual_file_hash:
                    raise DocumentIntegrityError(
                        f"Untrusted payload metadata: metadata claims hash '{embedded_hash}' "
                        f"which contradicts actual file hash '{actual_file_hash}'.",
                        sample_id=sample.sample_id,
                        expected_hash=sample.document_hash_sha256,
                        actual_hash=embedded_hash,
                    )

        return payload


# ============================================================================
# Prediction Mapping
# ============================================================================

def map_envelope_to_sample_prediction(
    sample_id: str,
    envelope: CanonicalExtractionEnvelope,
    expected_field_paths: Optional[Set[str]] = None,
) -> Tuple[SamplePrediction, List[str], List[str], List[str]]:
    """
    Pure mapper converting CanonicalExtractionEnvelope to SamplePrediction.
    Explicitly distinguishes and returns:
    - (SamplePrediction, extracted_field_paths, explicit_null_paths, omitted_field_paths)

    Guarantees that ground-truth fields omitted from the canonical output are not lost;
    they are mapped as predicted_value=None so compute_field_metrics accurately records an omission (FN).
    """
    predictions: Dict[str, FieldPrediction] = {}
    extracted_paths: List[str] = []
    explicit_null_paths: List[str] = []
    omitted_paths: List[str] = []

    # 1. Process fields present in envelope
    for field_path, field in envelope.fields.items():
        if field.value is None:
            explicit_null_paths.append(field_path)
            predictions[field_path] = FieldPrediction(
                field_path=field_path,
                predicted_value=None,
                confidence=field.confidence,
                is_null=True,
            )
        else:
            extracted_paths.append(field_path)
            predictions[field_path] = FieldPrediction(
                field_path=field_path,
                predicted_value=field.value,
                confidence=field.confidence,
                is_null=False,
            )

    # 2. Account for declared ground-truth fields omitted from the envelope output
    if expected_field_paths:
        for expected_path in expected_field_paths:
            if expected_path not in predictions:
                omitted_paths.append(expected_path)
                predictions[expected_path] = FieldPrediction(
                    field_path=expected_path,
                    predicted_value=None,
                    confidence=0.0,
                    is_null=True,
                )

    sample_pred = SamplePrediction(
        sample_id=sample_id,
        document_status=envelope.document_metadata.status,
        predictions=predictions,
    )

    return sample_pred, extracted_paths, explicit_null_paths, omitted_paths


# ============================================================================
# Report & Observation Models
# ============================================================================

GR005Status = Literal["INCOMPLETE", "VERIFIED", "FAILED"]


class GR005Accounting(BaseModel):
    """
    PRD GR-005 Schema Conformance Accounting:
    'Output must conform to the schema. 1,000 runs, 0 exceptions (M3).'
    Strictly tracks execution run counts, unique qualifying runs, exceptions, and conformance status.
    """
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["INCOMPLETE", "VERIFIED", "FAILED"] = Field(
        ...,
        description="PRD GR-005 conformance status: INCOMPLETE (<1000 runs), VERIFIED (>=1000 runs, 0 exceptions), or FAILED (>0 exceptions)",
    )
    total_attempted_runs: int = Field(default=0, description="Total schema-constrained extraction runs attempted")
    successful_validations: int = Field(default=0, description="Distinct qualifying runs with successful schema validation and zero exceptions")
    unique_qualifying_runs: int = Field(default=0, description="Count of distinct unique qualifying execution IDs")
    validation_exceptions: int = Field(default=0, description="Runs that raised schema validation exceptions")
    conformance_run_ids: List[str] = Field(default_factory=list, description="Run IDs of qualifying conformance runs")
    required_runs: int = Field(default=1000, description="PRD GR-005 benchmark target run count (1,000 runs)")
    is_conformance_verified: bool = Field(
        default=False,
        description="True if unique_qualifying_runs >= required_runs and validation_exceptions == 0",
    )
    limitation_note: Optional[str] = Field(
        default=None,
        description="Audit note documenting whether 1,000 repeated runs were actually completed in this session",
    )


class SampleObservation(BaseModel):
    """Detailed diagnostic observation per sample in the benchmark evaluation."""
    model_config = ConfigDict(extra="forbid", frozen=True)

    sample_id: str
    document_hash: str
    document_status: str
    is_synthetic: bool = False
    execution_time_ms: float = 0.0
    provenance: Optional[ExecutionProvenance] = None
    extracted_fields: List[str] = Field(default_factory=list)
    explicit_null_fields: List[str] = Field(default_factory=list)
    omitted_fields: List[str] = Field(default_factory=list)
    has_key_error: bool = False
    error: Optional[str] = None


class BenchmarkReport(BaseModel):
    """
    Structured, reproducible, and auditable benchmark report (PRD SCH-007, US-DOCX-010).
    Captures dataset scope, schema identity, per-sample observations, metrics,
    calibration, false-fill rates, M11 escapes, GR-005 accounting, and gate eligibility.
    """
    model_config = ConfigDict(extra="forbid", frozen=True)

    benchmark_id: str = Field(..., description="Unique UUID for this benchmark execution")
    run_timestamp: str = Field(..., description="ISO 8601 UTC timestamp of execution")
    dataset_id: str
    tenant_id: str
    schema_id: str
    schema_version: int
    document_type: str
    dataset_split: DatasetSplit
    policy_version: str
    primary_aggregation: Literal["macro", "micro"]
    total_samples: int
    real_samples: int
    synthetic_samples: int
    failed_samples: int
    is_mock_run: bool
    execution_modes: List[str]
    providers: List[str]
    models: List[str]
    prompt_versions: List[str]
    field_metrics: Dict[str, FieldMetrics]
    aggregate_metrics: AggregateMetrics
    metrics_decision: EvaluationDecision
    synthetic_field_metrics: Optional[Dict[str, FieldMetrics]] = Field(
        default=None,
        description="Diagnostic field metrics computed separately from synthetic samples (not part of release gate)",
    )
    synthetic_aggregate_metrics: Optional[AggregateMetrics] = Field(
        default=None,
        description="Diagnostic aggregate metrics computed separately from synthetic samples (not part of release gate)",
    )
    gate_eligible: bool = Field(
        ...,
        description="True if benchmark meets all criteria for formal release gate qualification",
    )
    gate_eligibility_reasons: List[str] = Field(
        default_factory=list,
        description="Reasons why run is or is not eligible for release-gate qualification",
    )
    gr005_accounting: GR005Accounting
    sample_observations: List[SampleObservation]
    overall_pass: bool = Field(..., description="True if metrics passed AND run is gate-eligible")
    summary: str

    def to_json(self, indent: int = 2) -> str:
        """Deterministic JSON serialization suitable for audit and reproducibility."""
        return self.model_dump_json(indent=indent)


# ============================================================================
# Test Harness Runner
# ============================================================================

class TestHarnessRunner:
    """
    Executable benchmark runner connecting EvaluationDataset, testing schema binding,
    DynamicExtractionAdapter, trusted execution provenance, and evaluation metrics.
    Strictly adheres to PRD §13.1..13.3, §14 Gate G1, M1, M6, M11, GR-003, GR-005.
    """
    __test__ = False

    def __init__(
        self,
        lifecycle_manager: SchemaLifecycleManager,
        adapter: DynamicExtractionAdapter,
        payload_resolver: PayloadResolver,
        policy: Optional[EvaluationPolicy] = None,
        authorizer: Optional[LockedSetAuthorizer] = None,
        allow_mock: bool = False,
        min_locked_samples: int = 100,
        gr005_target_runs: int = 1000,
    ) -> None:
        self.lifecycle_manager = lifecycle_manager
        self.adapter = adapter
        self.payload_resolver = payload_resolver
        self.policy = policy or EvaluationPolicy()
        self.authorizer = authorizer or DefaultLockedSetAuthorizer()
        self.allow_mock = allow_mock
        self.min_locked_samples = min_locked_samples
        self.gr005_target_runs = gr005_target_runs

    def run_benchmark(
        self,
        dataset: EvaluationDataset,
        auth_context: Optional[AuthorizationContext] = None,
    ) -> BenchmarkReport:
        """
        Executes end-to-end benchmark evaluation across an EvaluationDataset.
        Enforces locked-test authorization, hash integrity, strict no-fallback,
        evaluates real samples exclusively for release-gate decisions,
        and produces an auditable BenchmarkReport.
        """
        benchmark_id = f"bench_{uuid.uuid4().hex[:12]}"
        run_timestamp = datetime.now(timezone.utc).isoformat()
        start_time_all = time.perf_counter()

        # ---------------------------------------------------------------------
        # 1. Dataset Consistency & Split Validation
        # ---------------------------------------------------------------------
        if not dataset.samples:
            raise ValueError(f"Dataset '{dataset.dataset_id}' contains zero evaluation samples.")

        split = dataset.dataset_split

        # ---------------------------------------------------------------------
        # 2. Locked-Test Authorization Safeguards (PRD §13.1, §14 G1, Check 3)
        # ---------------------------------------------------------------------
        if split == DatasetSplit.LOCKED_TEST:
            if not auth_context:
                raise LockedSetAccessError(
                    message="Locked-test dataset execution requires a trusted AuthorizationContext.",
                    tenant_id=dataset.tenant_id,
                    dataset_id=dataset.dataset_id,
                )

            auth_decision = self.authorizer.authorize(
                context=auth_context,
                tenant_id=dataset.tenant_id,
                dataset_id=dataset.dataset_id,
                split=split,
            )
            if not auth_decision.is_authorized:
                raise LockedSetAccessError(
                    message=f"Locked-test evaluation authorization denied: {auth_decision.reason}",
                    principal_id=auth_context.principal_id,
                    tenant_id=dataset.tenant_id,
                    dataset_id=dataset.dataset_id,
                )

            # Gate-grade ground truth validation (>= 2 distinct annotators + adjudication)
            for sample in dataset.samples:
                sample.ground_truth.validate_for_gate()

        # ---------------------------------------------------------------------
        # 3. Bind Test Schema (SCH-007, US-DOCX-010)
        # ---------------------------------------------------------------------
        snapshot = self.lifecycle_manager.bind_test_schema(
            tenant_id=dataset.tenant_id,
            schema_id=dataset.schema_id,
            version=dataset.schema_version,
        )

        # ---------------------------------------------------------------------
        # 4. Execute Dynamic Extraction across Samples
        # ---------------------------------------------------------------------
        real_sample_list: List[EvaluationSample] = []
        synth_sample_list: List[EvaluationSample] = []
        real_predictions: Dict[str, SamplePrediction] = {}
        synth_predictions: Dict[str, SamplePrediction] = {}
        sample_observations: List[SampleObservation] = []

        total_samples = len(dataset.samples)
        real_samples = 0
        synthetic_samples = 0
        failed_samples = 0

        execution_modes_set: Set[str] = set()
        providers_set: Set[str] = set()
        models_set: Set[str] = set()
        prompt_versions_set: Set[str] = set()
        conformance_run_ids: List[str] = []
        seen_run_ids: Set[str] = set()

        gr005_attempted = 0
        gr005_exceptions = 0

        for sample in dataset.samples:
            # Track real vs synthetic sample classification (PRD §13.1, Check 2)
            is_synthetic = bool(
                sample.metadata.get("is_synthetic")
                or sample.stratum == "synthetic"
            )
            if is_synthetic:
                synthetic_samples += 1
                synth_sample_list.append(sample)
            else:
                real_samples += 1
                real_sample_list.append(sample)

            # Resolve Module B output payload & verify hash integrity
            t0 = time.perf_counter()
            payload = self.payload_resolver.resolve_payload(sample)

            # Verify tenant alignment
            if payload.tenant_id != dataset.tenant_id:
                raise DocumentIntegrityError(
                    f"Sample '{sample.sample_id}' payload tenant '{payload.tenant_id}' "
                    f"does not match dataset tenant '{dataset.tenant_id}'.",
                    sample_id=sample.sample_id,
                )

            # Execute extraction via DynamicExtractionAdapter
            gr005_attempted += 1
            envelope: Optional[CanonicalExtractionEnvelope] = None
            sample_error: Optional[str] = None

            try:
                envelope = self.adapter.extract_and_assemble(
                    tenant_id=dataset.tenant_id,
                    schema_id=dataset.schema_id,
                    version=dataset.schema_version,
                    payload=payload,
                )
                run_id = envelope.run.run_id if (envelope and envelope.run) else None
                if run_id and run_id.strip():
                    if run_id not in seen_run_ids:
                        seen_run_ids.add(run_id)
                        conformance_run_ids.append(run_id)
            except LLMExecutionError as exc:
                if "untrusted provenance mode" in str(exc).lower():
                    # Re-raise untrusted provenance / execution error (SCH-007)
                    raise
                gr005_exceptions += 1
                failed_samples += 1
                sample_error = str(exc)
            except Exception as exc:
                gr005_exceptions += 1
                failed_samples += 1
                sample_error = str(exc)

            latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

            if envelope is not None:
                prov = envelope.run.provenance
                if prov:
                    execution_modes_set.add(prov.mode.value)
                    if prov.provider:
                        providers_set.add(prov.provider)
                    if prov.model_name:
                        models_set.add(prov.model_name)
                    prompt_versions_set.add(prov.prompt_version)

                    # Rejection of heuristic/fallback execution for benchmark (SCH-007)
                    if prov.fallback_occurred or prov.mode in (
                        ExecutionMode.HEURISTIC,
                        ExecutionMode.HEURISTIC_FALLBACK,
                    ):
                        raise LLMExecutionError(
                            f"Sample '{sample.sample_id}' returned untrusted heuristic provenance "
                            f"mode '{prov.mode.value}' (fallback: {prov.fallback_occurred}).",
                            model_name=prov.model_name,
                        )

                # Map envelope to SamplePrediction with explicit missing field handling
                gt_paths = set(sample.ground_truth.fields.keys())
                sample_pred, extracted_paths, null_paths, omitted_paths = map_envelope_to_sample_prediction(
                    sample_id=sample.sample_id,
                    envelope=envelope,
                    expected_field_paths=gt_paths,
                )
                if is_synthetic:
                    synth_predictions[sample.sample_id] = sample_pred
                else:
                    real_predictions[sample.sample_id] = sample_pred

                # Check if this sample had key errors for M11 escape tracking
                has_key_err = False
                if sample_pred.document_status == "auto_approved":
                    for fpath, gt_field in sample.ground_truth.fields.items():
                        if gt_field.is_key_field:
                            fpred = sample_pred.predictions.get(fpath)
                            pred_val = fpred.predicted_value if fpred else None
                            if gt_field.is_null:
                                if pred_val is not None:
                                    has_key_err = True
                                    break
                            else:
                                if pred_val is None:
                                    has_key_err = True
                                    break

                sample_observations.append(
                    SampleObservation(
                        sample_id=sample.sample_id,
                        document_hash=sample.document_hash_sha256,
                        document_status=envelope.document_metadata.status,
                        is_synthetic=is_synthetic,
                        execution_time_ms=latency_ms,
                        provenance=prov,
                        extracted_fields=extracted_paths,
                        explicit_null_fields=null_paths,
                        omitted_fields=omitted_paths,
                        has_key_error=has_key_err,
                        error=None,
                    )
                )
            else:
                # Failed sample handling without silent exclusion
                # Create empty prediction map so all ground-truth fields count as omissions (FN)
                empty_preds: Dict[str, FieldPrediction] = {}
                for fpath in sample.ground_truth.fields.keys():
                    empty_preds[fpath] = FieldPrediction(
                        field_path=fpath,
                        predicted_value=None,
                        confidence=0.0,
                        is_null=True,
                    )
                failed_pred = SamplePrediction(
                    sample_id=sample.sample_id,
                    document_status="failed",
                    predictions=empty_preds,
                )
                if is_synthetic:
                    synth_predictions[sample.sample_id] = failed_pred
                else:
                    real_predictions[sample.sample_id] = failed_pred

                sample_observations.append(
                    SampleObservation(
                        sample_id=sample.sample_id,
                        document_hash=sample.document_hash_sha256,
                        document_status="failed",
                        is_synthetic=is_synthetic,
                        execution_time_ms=latency_ms,
                        provenance=None,
                        extracted_fields=[],
                        explicit_null_fields=[],
                        omitted_fields=list(sample.ground_truth.fields.keys()),
                        has_key_error=True,
                        error=sample_error,
                    )
                )

        # ---------------------------------------------------------------------
        # 5. Evaluate Metrics & Policy Decision (SCH-007, M1, M6, M11, Check 2)
        # ---------------------------------------------------------------------
        # PRD §13.1 Check 2: Release gate metrics MUST be computed exclusively
        # from real samples. Synthetic samples are evaluated separately as diagnostics.
        if real_samples > 0:
            real_dataset = dataset.model_copy(update={"samples": real_sample_list})
            field_metrics_map, aggregate_metrics, metrics_decision = evaluate_dataset(
                dataset=real_dataset,
                sample_predictions=real_predictions,
                policy=self.policy,
            )
        else:
            # Synthetic-only dataset: release gate metrics cannot be verified from real data
            field_metrics_map = {}
            aggregate_metrics = AggregateMetrics(
                primary_aggregation=self.policy.primary_aggregation,
                macro_f1_all_fields=0.0,
                macro_f1_key_fields=0.0,
                macro_f1_non_key_fields=0.0,
                macro_precision_all=0.0,
                macro_recall_all=0.0,
                micro_f1_all_fields=0.0,
                micro_f1_key_fields=0.0,
                micro_precision_all=0.0,
                micro_recall_all=0.0,
                mean_calibration_gap_points=0.0,
                calibration_sample_count=0,
                total_samples=0,
                total_field_types=0,
                key_field_count=0,
                total_predictions=0,
                total_tp=0,
                total_fp=0,
                total_fn=0,
                total_tn=0,
                has_insufficient_data=True,
                insufficient_data_reasons=["SYNTHETIC_ONLY_DATASET: Zero real samples available in dataset for release gate verification."],
            )
            metrics_decision = self.policy.evaluate(
                metrics=aggregate_metrics,
                dataset_split=split,
            )

        # Diagnostic synthetic metrics evaluation (completely separated from release gate)
        synthetic_field_metrics: Optional[Dict[str, FieldMetrics]] = None
        synthetic_aggregate_metrics: Optional[AggregateMetrics] = None
        if synth_sample_list:
            synth_dataset = dataset.model_copy(update={"samples": synth_sample_list})
            s_f_metrics, s_agg_metrics, _ = evaluate_dataset(
                dataset=synth_dataset,
                sample_predictions=synth_predictions,
                policy=self.policy,
            )
            synthetic_field_metrics = s_f_metrics
            synthetic_aggregate_metrics = s_agg_metrics

        # ---------------------------------------------------------------------
        # 6. GR-005 Conformance Accounting (PRD GR-005, §14 G1)
        # ---------------------------------------------------------------------
        unique_runs_count = len(conformance_run_ids)
        target_runs = max(self.gr005_target_runs, 1000)

        if gr005_exceptions > 0:
            gr005_status: Literal["INCOMPLETE", "VERIFIED", "FAILED"] = "FAILED"
            is_gr005_verified = False
            limitation_note = (
                f"PRD GR-005 FAILED: {gr005_exceptions} schema-validation exception(s) occurred "
                f"during {gr005_attempted} attempted runs. PRD GR-005 requires 1,000 runs with 0 exceptions (M3)."
            )
        elif unique_runs_count >= target_runs:
            gr005_status = "VERIFIED"
            is_gr005_verified = True
            limitation_note = (
                f"PRD GR-005 VERIFIED: Completed {unique_runs_count} unique qualifying schema-constrained "
                f"runs with 0 schema-validation exceptions."
            )
        else:
            gr005_status = "INCOMPLETE"
            is_gr005_verified = False
            limitation_note = (
                f"PRD GR-005 INCOMPLETE: Session executed {unique_runs_count} unique qualifying "
                f"schema-constrained runs. PRD GR-005 requires {target_runs} runs with 0 exceptions; "
                f"remaining {target_runs - unique_runs_count} runs must be confirmed via separate longevity suite."
            )

        gr005_accounting = GR005Accounting(
            status=gr005_status,
            total_attempted_runs=gr005_attempted,
            successful_validations=unique_runs_count,
            unique_qualifying_runs=unique_runs_count,
            validation_exceptions=gr005_exceptions,
            conformance_run_ids=conformance_run_ids,
            required_runs=target_runs,
            is_conformance_verified=is_gr005_verified,
            limitation_note=limitation_note,
        )

        # ---------------------------------------------------------------------
        # 7. Gate Eligibility Assessment (PRD §13.1, §14 G1, Check 2, Check 4)
        # ---------------------------------------------------------------------
        gate_eligible = True
        gate_reasons: List[str] = []

        is_mock_run = ExecutionMode.MOCK.value in execution_modes_set

        # Mock runs cannot qualify a release gate
        if is_mock_run:
            gate_eligible = False
            gate_reasons.append("MOCK_EXECUTION: Mock execution cannot qualify a production release gate (PRD SCH-007).")

        # Synthetic-only datasets cannot qualify a release gate
        if real_samples == 0:
            gate_eligible = False
            gate_reasons.append(
                "SYNTHETIC_ONLY_DATASET: A dataset consisting solely of synthetic samples "
                "cannot qualify a release gate independently (PRD §13.1)."
            )

        # Execution failures disqualify the gate
        if failed_samples > 0:
            gate_eligible = False
            gate_reasons.append(
                f"EXECUTION_FAILURES: {failed_samples} sample(s) failed extraction during benchmark execution."
            )

        # Metrics decision must pass
        if not metrics_decision.overall_pass:
            gate_eligible = False
            gate_reasons.append("METRICS_FAILED: Evaluation metrics did not meet acceptance thresholds.")

        # Insufficient data disqualifies the gate
        if metrics_decision.has_insufficient_data:
            gate_eligible = False
            gate_reasons.append("INSUFFICIENT_DATA: Evaluation data is insufficient to verify all gate metrics.")

        # Minimum sample size advisory on locked_test
        if split == DatasetSplit.LOCKED_TEST and real_samples < self.min_locked_samples:
            gate_reasons.append(
                f"SAMPLE_COUNT_ADVISORY: Real sample count ({real_samples}) is below recommended gate size "
                f"({self.min_locked_samples} documents per PRD §13.1 proposed)."
            )

        # PRD GR-005 Schema Conformance Gate (Requirement 8):
        # Incomplete or failed GR-005 must prevent final release-gate PASS
        if gr005_status == "FAILED":
            gate_eligible = False
            gate_reasons.append(
                f"GR005_FAILED: Schema-constrained decoding raised {gr005_exceptions} "
                f"validation exception(s) (PRD GR-005 requires 1,000 runs with 0 exceptions)."
            )
        elif gr005_status == "INCOMPLETE":
            gate_eligible = False
            gate_reasons.append(
                f"GR005_INCOMPLETE: Schema conformance requires {target_runs} unique runs with 0 exceptions; "
                f"only {unique_runs_count} qualifying run(s) completed (PRD GR-005)."
            )
        elif gr005_status == "VERIFIED":
            gate_reasons.append(
                f"GR005_VERIFIED: {unique_runs_count} schema-constrained runs completed with 0 exceptions (PRD GR-005)."
            )

        if gate_eligible and not gate_reasons:
            gate_reasons.append("GATE_QUALIFIED: All metrics passed and gate qualification requirements satisfied.")

        overall_pass = metrics_decision.overall_pass and (
            gate_eligible if split == DatasetSplit.LOCKED_TEST else True
        )

        # ---------------------------------------------------------------------
        # 8. Assemble Benchmark Report
        # ---------------------------------------------------------------------
        summary = (
            f"Benchmark {benchmark_id} on {dataset.dataset_id} ({split.value}): "
            f"Pass={overall_pass}, RealSamples={real_samples}, SynthSamples={synthetic_samples}, "
            f"GR005={gr005_status} ({unique_runs_count}/{target_runs}, exceptions={gr005_exceptions}), "
            f"M1_all={aggregate_metrics.macro_f1_all_fields:.2%}, "
            f"M1_key={aggregate_metrics.macro_f1_key_fields:.2%}, "
            f"CalGap={aggregate_metrics.dataset_calibration_gap_points or 0.0:.2f}pts, "
            f"Escapes={aggregate_metrics.escape_count or 0}, "
            f"GateEligible={gate_eligible}."
        )

        return BenchmarkReport(
            benchmark_id=benchmark_id,
            run_timestamp=run_timestamp,
            dataset_id=dataset.dataset_id,
            tenant_id=dataset.tenant_id,
            schema_id=dataset.schema_id,
            schema_version=dataset.schema_version,
            document_type=dataset.document_type,
            dataset_split=split,
            policy_version=self.policy.policy_version,
            primary_aggregation=self.policy.primary_aggregation,
            total_samples=total_samples,
            real_samples=real_samples,
            synthetic_samples=synthetic_samples,
            failed_samples=failed_samples,
            is_mock_run=is_mock_run,
            execution_modes=sorted(list(execution_modes_set)),
            providers=sorted(list(providers_set)),
            models=sorted(list(models_set)),
            prompt_versions=sorted(list(prompt_versions_set)),
            field_metrics=field_metrics_map,
            aggregate_metrics=aggregate_metrics,
            metrics_decision=metrics_decision,
            synthetic_field_metrics=synthetic_field_metrics,
            synthetic_aggregate_metrics=synthetic_aggregate_metrics,
            gate_eligible=gate_eligible,
            gate_eligibility_reasons=gate_reasons,
            gr005_accounting=gr005_accounting,
            sample_observations=sample_observations,
            overall_pass=overall_pass,
            summary=summary,
        )
