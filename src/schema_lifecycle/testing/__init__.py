"""
Schema Testing and Evaluation Package (SCH-007, US-DOCX-010).
Provides foundational contracts, pure deterministic metric functions,
and versioned evaluation policies for benchmarking schema accuracy against ground truth.
"""

from schema_lifecycle.testing.contracts import (
    AggregateMetrics,
    ConfidenceBinMetric,
    DatasetSplit,
    EvaluationDataset,
    EvaluationSample,
    FieldMetrics,
    FieldPrediction,
    GroundTruthDocument,
    GroundTruthField,
    MatchStrategy,
    SamplePrediction,
)
from schema_lifecycle.testing.metrics import (
    compute_10_bin_calibration,
    compute_aggregate_metrics,
    compute_field_metrics,
    compute_token_f1,
    evaluate_dataset,
    match_field_values,
    normalize_for_matching,
)
from schema_lifecycle.testing.policy import (
    DEFAULT_GR003_FALSE_FILL_THRESHOLD,
    EvaluationDecision,
    EvaluationPolicy,
)
from schema_lifecycle.testing.runner import (
    AuthorizationContext,
    AuthorizationDecision,
    BenchmarkReport,
    DefaultLockedSetAuthorizer,
    DocumentIntegrityError,
    GR005Accounting,
    GR005Status,
    InMemoryPayloadResolver,
    LocalPayloadResolver,
    LockedSetAccessError,
    LockedSetAuthorizer,
    PayloadResolver,
    SampleObservation,
    TestHarnessRunner,
    TrustedPrincipal,
    map_envelope_to_sample_prediction,
)

__all__ = [
    # Enums
    "DatasetSplit",
    "MatchStrategy",
    # Dataset Models
    "GroundTruthField",
    "GroundTruthDocument",
    "EvaluationSample",
    "EvaluationDataset",
    # Prediction Models
    "FieldPrediction",
    "SamplePrediction",
    # Metric Models
    "ConfidenceBinMetric",
    "FieldMetrics",
    "AggregateMetrics",
    # Policy Models
    "DEFAULT_GR003_FALSE_FILL_THRESHOLD",
    "EvaluationPolicy",
    "EvaluationDecision",
    # Evaluation Functions
    "normalize_for_matching",
    "compute_token_f1",
    "match_field_values",
    "compute_10_bin_calibration",
    "compute_field_metrics",
    "compute_aggregate_metrics",
    "evaluate_dataset",
    # Runner Models & Protocols
    "TestHarnessRunner",
    "BenchmarkReport",
    "SampleObservation",
    "GR005Accounting",
    "GR005Status",
    "TrustedPrincipal",
    "AuthorizationContext",
    "AuthorizationDecision",
    "LockedSetAuthorizer",
    "DefaultLockedSetAuthorizer",
    "PayloadResolver",
    "InMemoryPayloadResolver",
    "LocalPayloadResolver",
    "LockedSetAccessError",
    "DocumentIntegrityError",
    "map_envelope_to_sample_prediction",
]

