"""
Unit and Integration Tests for Schema Evaluation Contracts and Metrics (TASK-4.3 Stage 1).
Covers PRD Requirements:
- SCH-007: Test harness running on sample documents with ground truth reporting per-field precision, recall, F1, calibration.
- M1: Field F1 benchmarks (≥92% all fields, ≥95% key fields, macro vs micro aggregation).
- M6: 10-bin confidence calibration gap (≤3.0 points).
- M11 / GR-027: Release gate escape evaluation (strictly on locked_test split).
- GR-003: Honest null-if-absent handling and false-fill rate reporting.
- US-DOCX-010: Test a schema before approval.

NOTE: All fixtures in this test suite are synthetic and used strictly for unit testing.
They do not represent genuine production or benchmark evaluation results.
"""

import pytest

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
    normalize_date,
    normalize_for_matching,
)
from schema_lifecycle.testing.policy import (
    DEFAULT_GR003_FALSE_FILL_THRESHOLD,
    EvaluationDecision,
    EvaluationPolicy,
)


# ============================================================================
# Synthetic Test Fixtures (Non-production, Synthetic Data)
# ============================================================================

DUMMY_HASH_1 = "a" * 64
DUMMY_HASH_2 = "b" * 64
DUMMY_HASH_3 = "c" * 64


# ============================================================================
# 1. Dataset Contracts & Validation Tests
# ============================================================================

def test_ground_truth_field_null_consistency():
    """GR-003: If expected_value is None, is_null must be True."""
    f1 = GroundTruthField(field_path="applicant.pan", expected_value=None)
    assert f1.is_null is True
    assert f1.expected_value is None

    # Expected value present, is_null defaults to False
    f2 = GroundTruthField(field_path="applicant.pan", expected_value="ABCDE1234F", is_key_field=True)
    assert f2.is_null is False
    assert f2.is_key_field is True

    # Inconsistent: is_null=True but expected_value is non-None
    with pytest.raises(ValueError, match="is_null=True but expected_value is not None"):
        GroundTruthField(field_path="applicant.pan", expected_value="ABCDE1234F", is_null=True)


def test_ground_truth_document_annotator_validation():
    """PRD §13.1: Ground truth requires valid, distinct annotator IDs."""
    doc = GroundTruthDocument(
        sample_id="SAMPLE-001",
        document_hash_sha256=DUMMY_HASH_1,
        annotator_ids=["annotator_1", "annotator_2"],
        adjudicated=True,
    )
    assert doc.annotator_ids == ["annotator_1", "annotator_2"]
    assert doc.adjudicated is True

    # Duplicate annotator IDs must be rejected
    with pytest.raises(ValueError, match="annotator_ids must be distinct"):
        GroundTruthDocument(
            sample_id="SAMPLE-001",
            document_hash_sha256=DUMMY_HASH_1,
            annotator_ids=["annotator_1", "annotator_1"],
        )

    # Empty or blank annotator IDs rejected
    with pytest.raises(ValueError, match="annotator_ids cannot contain empty"):
        GroundTruthDocument(
            sample_id="SAMPLE-001",
            document_hash_sha256=DUMMY_HASH_1,
            annotator_ids=["annotator_1", "   "],
        )

    # Invalid hash format
    with pytest.raises(ValueError, match="64-character hexadecimal SHA-256"):
        GroundTruthDocument(
            sample_id="SAMPLE-001",
            document_hash_sha256="invalid_short_hash",
            annotator_ids=["annotator_1"],
        )


def test_ground_truth_gate_grade_validation():
    """PRD §13.1: Gate-grade ground truth requires >= 2 distinct annotators and adjudication."""
    single_annotator_doc = GroundTruthDocument(
        sample_id="SAMPLE-001",
        document_hash_sha256=DUMMY_HASH_1,
        annotator_ids=["annotator_1"],
        adjudicated=False,
    )
    # Fails gate-grade check
    with pytest.raises(ValueError, match="requires at least 2 distinct annotators"):
        single_annotator_doc.validate_for_gate()

    unadjudicated_doc = GroundTruthDocument(
        sample_id="SAMPLE-001",
        document_hash_sha256=DUMMY_HASH_1,
        annotator_ids=["annotator_1", "annotator_2"],
        adjudicated=False,
    )
    with pytest.raises(ValueError, match="must be marked as adjudicated"):
        unadjudicated_doc.validate_for_gate()

    # Valid gate-grade document
    valid_doc = GroundTruthDocument(
        sample_id="SAMPLE-001",
        document_hash_sha256=DUMMY_HASH_1,
        annotator_ids=["annotator_1", "annotator_2"],
        adjudicated=True,
        adjudicator_id="lead_reviewer",
    )
    valid_doc.validate_for_gate()  # Should not raise


def test_evaluation_sample_alignment():
    """EvaluationSample enforces hash and sample_id alignment with ground truth."""
    gt = GroundTruthDocument(
        sample_id="SAMPLE-001",
        document_hash_sha256=DUMMY_HASH_1,
        annotator_ids=["annotator_1"],
    )
    # Valid sample
    sample = EvaluationSample(
        sample_id="SAMPLE-001",
        tenant_id="tenant_alpha",
        document_type="DOC-01",
        document_ref="storage://eval/samples/sample_001.pdf",
        document_hash_sha256=DUMMY_HASH_1,
        stratum="clean_print",
        ground_truth=gt,
    )
    assert sample.document_ref == "storage://eval/samples/sample_001.pdf"

    # Mismatched sample_id
    with pytest.raises(ValueError, match="sample_id mismatch"):
        EvaluationSample(
            sample_id="SAMPLE-002",
            tenant_id="tenant_alpha",
            document_type="DOC-01",
            document_ref="storage://eval/samples/sample_001.pdf",
            document_hash_sha256=DUMMY_HASH_1,
            ground_truth=gt,
        )

    # Mismatched hash
    with pytest.raises(ValueError, match="document_hash_sha256 mismatch"):
        EvaluationSample(
            sample_id="SAMPLE-001",
            tenant_id="tenant_alpha",
            document_type="DOC-01",
            document_ref="storage://eval/samples/sample_001.pdf",
            document_hash_sha256=DUMMY_HASH_2,
            ground_truth=gt,
        )


def test_evaluation_dataset_tenant_and_gate_validation():
    """EvaluationDataset enforces tenant isolation, document type consistency, and gate rules."""
    gt1 = GroundTruthDocument(
        sample_id="S1",
        document_hash_sha256=DUMMY_HASH_1,
        annotator_ids=["a1"],
        adjudicated=False,
    )
    s1 = EvaluationSample(
        sample_id="S1",
        tenant_id="tenant_alpha",
        document_type="DOC-01",
        document_ref="storage://s1",
        document_hash_sha256=DUMMY_HASH_1,
        ground_truth=gt1,
    )

    # Valid for development split with single annotator
    dev_dataset = EvaluationDataset(
        dataset_id="DEV-SET-01",
        tenant_id="tenant_alpha",
        schema_id="casa_schema",
        schema_version=1,
        document_type="DOC-01",
        dataset_split=DatasetSplit.DEVELOPMENT,
        samples=[s1],
    )
    assert len(dev_dataset.samples) == 1

    # LOCKED_TEST split MUST enforce gate-grade validation (fails because s1 has only 1 annotator)
    with pytest.raises(ValueError, match="requires at least 2 distinct annotators"):
        EvaluationDataset(
            dataset_id="LOCKED-SET-01",
            tenant_id="tenant_alpha",
            schema_id="casa_schema",
            schema_version=1,
            document_type="DOC-01",
            dataset_split=DatasetSplit.LOCKED_TEST,
            samples=[s1],
        )

    # Cross-tenant sample rejection
    s2_wrong_tenant = EvaluationSample(
        sample_id="S2",
        tenant_id="tenant_beta",
        document_type="DOC-01",
        document_ref="storage://s2",
        document_hash_sha256=DUMMY_HASH_2,
        ground_truth=GroundTruthDocument(
            sample_id="S2",
            document_hash_sha256=DUMMY_HASH_2,
            annotator_ids=["a1"],
        ),
    )
    with pytest.raises(ValueError, match="does not match dataset tenant_id"):
        EvaluationDataset(
            dataset_id="DEV-SET-02",
            tenant_id="tenant_alpha",
            schema_id="casa_schema",
            schema_version=1,
            document_type="DOC-01",
            dataset_split=DatasetSplit.DEVELOPMENT,
            samples=[s1, s2_wrong_tenant],
        )


# ============================================================================
# 2. Pure Matching Rules (PRD §13.3)
# ============================================================================

def test_normalization_and_exact_match():
    """PRD §13.3: Exact match after normalization for identifiers, dates, enums."""
    assert normalize_for_matching("  ABCDE1234F  ") == "abcde1234f"
    assert normalize_for_matching("50,000.00", field_type="decimal") == "50000"
    assert normalize_for_matching(True) == "true"
    assert normalize_for_matching(False) == "false"

    # PAN exact match with casing differences
    match, score = match_field_values("ABCDE1234F", "abcde1234f", MatchStrategy.EXACT_NORMALIZED)
    assert match is True
    assert score == 1.0

    # Currency string with comma formatting vs clean integer with field_type="decimal"
    match, score = match_field_values("50,000.00", "50000", MatchStrategy.EXACT_NORMALIZED, field_type="decimal")
    assert match is True
    assert score == 1.0

    # Incorrect non-null prediction must fail
    match, score = match_field_values("ABCDE1234F", "XYZ9999A", MatchStrategy.EXACT_NORMALIZED)
    assert match is False
    assert score == 0.0


def test_type_aware_normalization_identifiers_preserve_leading_zeros():
    """
    CORRECTION 1 REGRESSION TEST:
    Account numbers and string identifiers must preserve leading zeros and identifier semantics.
    Numeric equivalence must not apply to string identifiers.
    """
    # Account number string preserves leading zeros
    assert normalize_for_matching("0012345", field_type="string") == "0012345"
    assert normalize_for_matching("12345", field_type="string") == "12345"

    # Mismatch between "0012345" and "12345"
    match, score = match_field_values("0012345", "12345", MatchStrategy.EXACT_NORMALIZED, field_type="string")
    assert match is False
    assert score == 0.0

    # String identifiers with commas must preserve commas (not treated as thousands separators)
    assert normalize_for_matching("BATCH,123", field_type="string") == "batch,123"

    # Decimal amount matches with field_type="decimal"
    assert normalize_for_matching("1,250.50", field_type="decimal") == "1250.5"
    match, score = match_field_values("1,250.50", "1250.50", MatchStrategy.EXACT_NORMALIZED, field_type="decimal")
    assert match is True

    # Date normalization preserves ISO format
    assert normalize_for_matching("  2026-10-29  ", field_type="date") == "2026-10-29"

    # Case-insensitive identifier (default) vs Case-sensitive identifier profile
    match_ci, _ = match_field_values("IFSC001", "ifsc001", MatchStrategy.EXACT_NORMALIZED)
    assert match_ci is True

    match_cs, _ = match_field_values(
        "IFSC001", "ifsc001", MatchStrategy.EXACT_NORMALIZED, normalization_profile="case_sensitive_identifier"
    )
    assert match_cs is False


def test_numeric_tolerance_matching():
    """Numeric tolerance matching for decimals and percentages."""
    # Within 0.01 tolerance
    match, score = match_field_values(100.004, 100.009, MatchStrategy.NUMERIC_TOLERANCE, tolerance=0.01)
    assert match is True
    assert score == 1.0

    # Exceeding tolerance
    match, score = match_field_values(100.0, 100.05, MatchStrategy.NUMERIC_TOLERANCE, tolerance=0.01)
    assert match is False
    assert score == 0.0


def test_token_f1_matching_free_text():
    """PRD §13.3: Token-level F1 for free text."""
    gt_text = "Acute appendicitis with localized peritonitis"
    pred_exact = "acute appendicitis with localized peritonitis"
    score_exact = compute_token_f1(gt_text, pred_exact)
    assert score_exact == 1.0

    # Partial overlap
    pred_partial = "Acute appendicitis"
    score_partial = compute_token_f1(gt_text, pred_partial)
    # overlap = 2 ("acute", "appendicitis"), len_gt = 5, len_pred = 2 -> P = 2/2 = 1.0, R = 2/5 = 0.4 -> F1 = 2*(1*0.4)/1.4 = 0.5714
    assert round(score_partial, 4) == 0.5714

    # Match thresholding
    match, score = match_field_values(
        gt_text, pred_partial, strategy=MatchStrategy.TOKEN_F1, token_f1_threshold=0.5
    )
    assert match is True
    assert round(score, 4) == 0.5714

    match, score = match_field_values(
        gt_text, pred_partial, strategy=MatchStrategy.TOKEN_F1, token_f1_threshold=0.8
    )
    assert match is False


# ============================================================================
# 3. Precision, Recall, F1, False-Fill, and FP/FN Accounting Tests
# ============================================================================

def test_fp_fn_accounting_on_incorrect_non_null_prediction():
    """
    CRITICAL REQUIREMENT: An incorrect non-null exact-match prediction must not be
    treated as a successful match. It must count as 1 FP (incorrect extraction)
    and 1 FN (missed true value).
    """
    gt_fields = [
        GroundTruthField(field_path="applicant.pan", expected_value="ABCDE1234F", is_key_field=True),
    ]
    # Model extracts wrong PAN
    preds = [
        FieldPrediction(field_path="applicant.pan", predicted_value="WRONG1234X", confidence=0.9),
    ]

    metrics = compute_field_metrics("applicant.pan", gt_fields, preds)

    # TP must be 0, FP must be 1, FN must be 1
    assert metrics.true_positives == 0
    assert metrics.false_positives == 1
    assert metrics.false_negatives == 1
    assert metrics.true_negatives == 0
    assert metrics.precision == 0.0
    assert metrics.recall == 0.0
    assert metrics.f1 == 0.0
    assert metrics.false_fill_rate is None  # Undefined when null ground-truth support is zero


def test_false_fill_rate_accounting_on_null_ground_truth():
    """
    PRD §13.3 & GR-003: Predicting a value where ground truth is null counts as a
    false positive and is reported separately as the false-fill rate.
    """
    gt_fields = [
        GroundTruthField(field_path="applicant.middle_name", expected_value=None),  # Null in document
        GroundTruthField(field_path="applicant.middle_name", expected_value=None),  # Null in document
    ]
    # Sample 1: correctly predicts null
    # Sample 2: hallucinates a middle name "KUMAR" (false-fill)
    preds = [
        FieldPrediction(field_path="applicant.middle_name", predicted_value=None, confidence=0.0),
        FieldPrediction(field_path="applicant.middle_name", predicted_value="KUMAR", confidence=0.85),
    ]

    metrics = compute_field_metrics("applicant.middle_name", gt_fields, preds)

    assert metrics.true_positives == 0
    assert metrics.false_positives == 1  # The hallucinated value is an FP
    assert metrics.false_negatives == 0
    assert metrics.true_negatives == 1   # Sample 1 was a correct null prediction
    assert metrics.null_support == 2
    # False-fill rate: 1 false fill out of 2 null opportunities = 0.5 (50%)
    assert metrics.false_fill_rate == 0.5


def test_omission_accounting():
    """Model omits a present field (predicts null when GT is non-null). Counts as FN."""
    gt_fields = [
        GroundTruthField(field_path="applicant.dob", expected_value="1990-01-01"),
    ]
    preds = [
        FieldPrediction(field_path="applicant.dob", predicted_value=None, confidence=0.0),
    ]

    metrics = compute_field_metrics("applicant.dob", gt_fields, preds)
    assert metrics.true_positives == 0
    assert metrics.false_positives == 0
    assert metrics.false_negatives == 1
    assert metrics.recall == 0.0


def test_perfect_null_handling_f1():
    """When a field is always absent and model always predicts null, F1 should be 1.0 (GR-003)."""
    gt_fields = [
        GroundTruthField(field_path="optional_flag", expected_value=None),
        GroundTruthField(field_path="optional_flag", expected_value=None),
    ]
    preds = [
        FieldPrediction(field_path="optional_flag", predicted_value=None, confidence=0.0),
        FieldPrediction(field_path="optional_flag", predicted_value=None, confidence=0.0),
    ]

    metrics = compute_field_metrics("optional_flag", gt_fields, preds)
    assert metrics.true_negatives == 2
    assert metrics.false_positives == 0
    assert metrics.false_negatives == 0
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1 == 1.0
    assert metrics.false_fill_rate == 0.0


# ============================================================================
# 4. 10-Bin Confidence Calibration Math (PRD M6 / §13.3)
# ============================================================================

def test_10_bin_calibration_perfect():
    """Perfect calibration: stated confidence matches accuracy."""
    # 10 predictions with confidence 0.95 (bin 9), all 10 are correct (100% acc)
    # Gap in bin 9: |0.95 - 1.0| = 0.05
    confs = [0.95] * 10
    correct = [True] * 10

    gap_points, bins = compute_10_bin_calibration(confs, correct)
    assert len(bins) == 10
    assert bins[9].sample_count == 10
    assert bins[9].mean_confidence == 0.95
    assert bins[9].accuracy == 1.0
    assert bins[9].calibration_gap == 0.05
    assert gap_points == 5.0  # 0.05 * 100 = 5.0 points


def test_10_bin_calibration_overconfident_uncalibrated():
    """Uncalibrated overconfidence: 1.0 confidence but only 50% accurate."""
    confs = [1.0] * 10
    correct = [True, False] * 5  # 50% accurate

    gap_points, bins = compute_10_bin_calibration(confs, correct)
    assert bins[9].sample_count == 10
    assert bins[9].mean_confidence == 1.0
    assert bins[9].accuracy == 0.5
    assert bins[9].calibration_gap == 0.5
    assert gap_points == 50.0  # 50 points gap! Fails M6 (≤3.0 pts)


# ============================================================================
# 5. Full Dataset Evaluation & Policy Decision Tests
# ============================================================================

def test_full_dataset_evaluation_dev_split_pass():
    """Synthetic dataset on development split meeting M1 and M6."""
    gt_s1 = GroundTruthDocument(
        sample_id="DEV-S1",
        document_hash_sha256=DUMMY_HASH_1,
        annotator_ids=["a1"],
        fields={
            "applicant.name": GroundTruthField(field_path="applicant.name", expected_value="JOHN DOE", is_key_field=True),
            "applicant.pan": GroundTruthField(field_path="applicant.pan", expected_value="ABCDE1234F", is_key_field=True),
            "applicant.dob": GroundTruthField(field_path="applicant.dob", expected_value="1990-01-01", is_key_field=False),
        },
    )
    s1 = EvaluationSample(
        sample_id="DEV-S1",
        tenant_id="t1",
        document_type="DOC-01",
        document_ref="ref://s1",
        document_hash_sha256=DUMMY_HASH_1,
        ground_truth=gt_s1,
    )
    dataset = EvaluationDataset(
        dataset_id="DATASET-DEV-01",
        tenant_id="t1",
        schema_id="casa_v1",
        schema_version=1,
        document_type="DOC-01",
        dataset_split=DatasetSplit.DEVELOPMENT,
        samples=[s1],
    )

    # Predictions match ground truth with 0.99 confidence
    predictions = {
        "DEV-S1": SamplePrediction(
            sample_id="DEV-S1",
            document_status="auto_approved",
            predictions={
                "applicant.name": FieldPrediction(field_path="applicant.name", predicted_value="JOHN DOE", confidence=0.99),
                "applicant.pan": FieldPrediction(field_path="applicant.pan", predicted_value="ABCDE1234F", confidence=0.99),
                "applicant.dob": FieldPrediction(field_path="applicant.dob", predicted_value="1990-01-01", confidence=0.99),
            },
        )
    }

    policy = EvaluationPolicy(
        min_f1_all_fields=0.92,
        min_f1_key_fields=0.95,
        max_calibration_gap_points=3.0,
    )

    field_metrics, aggregate, decision = evaluate_dataset(dataset, predictions, policy)

    assert aggregate.macro_f1_all_fields == 1.0
    assert aggregate.macro_f1_key_fields == 1.0
    assert aggregate.dataset_calibration_gap_points <= 3.0
    # Development split does not evaluate M11 release-gate escapes
    assert aggregate.escape_count is None
    assert decision.meets_m11_escapes is None
    assert decision.overall_pass is True
    assert "PASS" in decision.summary


def test_locked_test_split_escape_causes_failure():
    """
    PRD M11 / GR-027: On locked_test split, ANY wrong key-field value on an
    auto-approved document is counted as an escape and blocks gate pass.
    """
    gt_s1 = GroundTruthDocument(
        sample_id="LOCKED-S1",
        document_hash_sha256=DUMMY_HASH_1,
        annotator_ids=["ann_1", "ann_2"],
        adjudicated=True,
        fields={
            "applicant.pan": GroundTruthField(field_path="applicant.pan", expected_value="ABCDE1234F", is_key_field=True),
        },
    )
    s1 = EvaluationSample(
        sample_id="LOCKED-S1",
        tenant_id="t1",
        document_type="DOC-01",
        document_ref="ref://s1",
        document_hash_sha256=DUMMY_HASH_1,
        ground_truth=gt_s1,
    )
    dataset = EvaluationDataset(
        dataset_id="DATASET-LOCKED-01",
        tenant_id="t1",
        schema_id="casa_v1",
        schema_version=1,
        document_type="DOC-01",
        dataset_split=DatasetSplit.LOCKED_TEST,
        samples=[s1],
    )

    # Document auto-approved but PAN is incorrect (escape!)
    predictions = {
        "LOCKED-S1": SamplePrediction(
            sample_id="LOCKED-S1",
            document_status="auto_approved",
            predictions={
                "applicant.pan": FieldPrediction(field_path="applicant.pan", predicted_value="WRONG9999Z", confidence=0.95),
            },
        )
    }

    policy = EvaluationPolicy()
    _, aggregate, decision = evaluate_dataset(dataset, predictions, policy)

    assert aggregate.escape_count == 1
    assert decision.meets_m11_escapes is False
    assert decision.overall_pass is False
    assert any("M11 escape violation" in v for v in decision.violations)


def test_locked_test_needs_review_not_counted_as_escape():
    """
    PRD M11: If a document has status 'needs_review', incorrect key field is routed
    to human review and does NOT count as an escape.
    """
    gt_s1 = GroundTruthDocument(
        sample_id="LOCKED-S2",
        document_hash_sha256=DUMMY_HASH_2,
        annotator_ids=["ann_1", "ann_2"],
        adjudicated=True,
        fields={
            "applicant.pan": GroundTruthField(field_path="applicant.pan", expected_value="ABCDE1234F", is_key_field=True),
        },
    )
    s1 = EvaluationSample(
        sample_id="LOCKED-S2",
        tenant_id="t1",
        document_type="DOC-01",
        document_ref="ref://s2",
        document_hash_sha256=DUMMY_HASH_2,
        ground_truth=gt_s1,
    )
    dataset = EvaluationDataset(
        dataset_id="DATASET-LOCKED-02",
        tenant_id="t1",
        schema_id="casa_v1",
        schema_version=1,
        document_type="DOC-01",
        dataset_split=DatasetSplit.LOCKED_TEST,
        samples=[s1],
    )

    # Status is 'needs_review' -> not an escape
    predictions = {
        "LOCKED-S2": SamplePrediction(
            sample_id="LOCKED-S2",
            document_status="needs_review",
            predictions={
                "applicant.pan": FieldPrediction(field_path="applicant.pan", predicted_value="WRONG9999Z", confidence=0.4),
            },
        )
    }

    policy = EvaluationPolicy()
    _, aggregate, decision = evaluate_dataset(dataset, predictions, policy)

    assert aggregate.escape_count == 0
    assert decision.meets_m11_escapes is True
    # Still fails M1 because PAN was wrong, but escape target (0) is satisfied
    assert decision.meets_m1_key_fields is False


def test_macro_vs_micro_aggregation():
    """
    Demonstrates and verifies the mathematical difference between macro- and micro-aggregation.
    Field A has 1 sample (100% accurate).
    Field B has 9 samples (0% accurate).
    Macro F1 = (1.0 + 0.0) / 2 = 0.50.
    Micro F1 = 2 * 1 / (2 * 1 + 9) = 2 / 11 = 0.1818.
    """
    field_a = FieldMetrics(
        field_path="field_a",
        is_key_field=False,
        precision=1.0,
        recall=1.0,
        f1=1.0,
        false_fill_rate=None,
        true_positives=1,
        false_positives=0,
        false_negatives=0,
        true_negatives=0,
        support=1,
        null_support=0,
        calibration_gap_points=0.0,
    )
    field_b = FieldMetrics(
        field_path="field_b",
        is_key_field=True,
        precision=0.0,
        recall=0.0,
        f1=0.0,
        false_fill_rate=None,
        true_positives=0,
        false_positives=9,
        false_negatives=9,
        true_negatives=0,
        support=9,
        null_support=0,
        calibration_gap_points=0.0,
    )

    dataset = EvaluationDataset(
        dataset_id="DATASET-DUMMY",
        tenant_id="t1",
        schema_id="s1",
        schema_version=1,
        document_type="DOC-01",
        dataset_split=DatasetSplit.DEVELOPMENT,
        samples=[
            EvaluationSample(
                sample_id="S1",
                tenant_id="t1",
                document_type="DOC-01",
                document_ref="ref://1",
                document_hash_sha256=DUMMY_HASH_1,
                ground_truth=GroundTruthDocument(
                    sample_id="S1",
                    document_hash_sha256=DUMMY_HASH_1,
                    annotator_ids=["a1"],
                ),
            )
        ],
    )

    aggregate = compute_aggregate_metrics(
        field_metrics={"field_a": field_a, "field_b": field_b},
        dataset=dataset,
        sample_predictions={},
    )

    assert aggregate.macro_f1_all_fields == 0.50
    assert aggregate.micro_f1_all_fields == 0.10
    # Key field is field_b
    assert aggregate.macro_f1_key_fields == 0.0


def test_extraneous_and_missing_field_predictions():
    """
    Tests evaluation when a predicted field is not in the ground truth
    (extraneous field -> treated as null ground truth -> FP / false-fill)
    and when a ground truth field is missing from predictions (omission -> FN).
    """
    gt_s1 = GroundTruthDocument(
        sample_id="S1",
        document_hash_sha256=DUMMY_HASH_1,
        annotator_ids=["a1"],
        fields={
            "present_field": GroundTruthField(field_path="present_field", expected_value="EXPECTED"),
        },
    )
    sample = EvaluationSample(
        sample_id="S1",
        tenant_id="t1",
        document_type="DOC-01",
        document_ref="ref://1",
        document_hash_sha256=DUMMY_HASH_1,
        ground_truth=gt_s1,
    )
    dataset = EvaluationDataset(
        dataset_id="DATASET-DEV",
        tenant_id="t1",
        schema_id="s1",
        schema_version=1,
        document_type="DOC-01",
        dataset_split=DatasetSplit.DEVELOPMENT,
        samples=[sample],
    )

    # Predictions omit present_field and predict an extraneous field
    predictions = {
        "S1": SamplePrediction(
            sample_id="S1",
            predictions={
                "extraneous_field": FieldPrediction(field_path="extraneous_field", predicted_value="HALLUCINATED", confidence=0.8),
            },
        )
    }

    field_metrics, aggregate, _ = evaluate_dataset(dataset, predictions)

    # present_field was omitted -> FN = 1
    assert field_metrics["present_field"].false_negatives == 1
    assert field_metrics["present_field"].recall == 0.0

    # extraneous_field was hallucinated against null GT -> FP = 1, false_fill = 1.0
    assert field_metrics["extraneous_field"].false_positives == 1
    assert field_metrics["extraneous_field"].false_fill_rate == 1.0


def test_m11_escape_accounting_negative_cases():
    """
    CORRECTION 5 REGRESSION TEST:
    Verify that:
    1. Auto-approved locked-test documents with errors ONLY in NON-KEY fields do NOT count as escapes.
    2. Needs-review locked-test documents with wrong key fields do NOT count as escapes.
    3. Auto-approved development-split documents with wrong key fields do NOT count as escapes (escape_count is None).
    """
    gt_doc = GroundTruthDocument(
        sample_id="LOCKED-SAMPLE-01",
        document_hash_sha256=DUMMY_HASH_1,
        annotator_ids=["ann_1", "ann_2"],
        adjudicated=True,
        fields={
            "key_field": GroundTruthField(field_path="key_field", expected_value="KEY_VAL", is_key_field=True),
            "non_key_field": GroundTruthField(field_path="non_key_field", expected_value="NON_KEY_VAL", is_key_field=False),
        },
    )
    sample = EvaluationSample(
        sample_id="LOCKED-SAMPLE-01",
        tenant_id="t1",
        document_type="DOC-01",
        document_ref="ref://1",
        document_hash_sha256=DUMMY_HASH_1,
        ground_truth=gt_doc,
    )
    locked_dataset = EvaluationDataset(
        dataset_id="LOCKED-SET",
        tenant_id="t1",
        schema_id="s1",
        schema_version=1,
        document_type="DOC-01",
        dataset_split=DatasetSplit.LOCKED_TEST,
        samples=[sample],
    )

    # Negative Case 1: Auto-approved document with wrong NON-KEY field, but correct KEY field
    preds_wrong_non_key = {
        "LOCKED-SAMPLE-01": SamplePrediction(
            sample_id="LOCKED-SAMPLE-01",
            document_status="auto_approved",
            predictions={
                "key_field": FieldPrediction(field_path="key_field", predicted_value="KEY_VAL", confidence=0.99),
                "non_key_field": FieldPrediction(field_path="non_key_field", predicted_value="WRONG_NON_KEY", confidence=0.99),
            },
        )
    }
    policy = EvaluationPolicy()
    _, aggregate_1, decision_1 = evaluate_dataset(locked_dataset, preds_wrong_non_key, policy)
    assert aggregate_1.escape_count == 0  # NOT an escape!
    assert decision_1.meets_m11_escapes is True

    # Negative Case 2: Needs-review document with wrong KEY field (routed to human review)
    preds_needs_review = {
        "LOCKED-SAMPLE-01": SamplePrediction(
            sample_id="LOCKED-SAMPLE-01",
            document_status="needs_review",
            predictions={
                "key_field": FieldPrediction(field_path="key_field", predicted_value="WRONG_KEY", confidence=0.4),
                "non_key_field": FieldPrediction(field_path="non_key_field", predicted_value="NON_KEY_VAL", confidence=0.99),
            },
        )
    }
    _, aggregate_2, decision_2 = evaluate_dataset(locked_dataset, preds_needs_review, policy)
    assert aggregate_2.escape_count == 0  # NOT an escape!
    assert decision_2.meets_m11_escapes is True

    # Negative Case 3: Development split with wrong KEY field
    dev_gt = GroundTruthDocument(
        sample_id="DEV-SAMPLE-01",
        document_hash_sha256=DUMMY_HASH_1,
        annotator_ids=["ann_1"],
        fields={
            "key_field": GroundTruthField(field_path="key_field", expected_value="KEY_VAL", is_key_field=True),
        },
    )
    dev_dataset = EvaluationDataset(
        dataset_id="DEV-SET",
        tenant_id="t1",
        schema_id="s1",
        schema_version=1,
        document_type="DOC-01",
        dataset_split=DatasetSplit.DEVELOPMENT,
        samples=[
            EvaluationSample(
                sample_id="DEV-SAMPLE-01",
                tenant_id="t1",
                document_type="DOC-01",
                document_ref="ref://1",
                document_hash_sha256=DUMMY_HASH_1,
                ground_truth=dev_gt,
            )
        ],
    )
    preds_dev_wrong_key = {
        "DEV-SAMPLE-01": SamplePrediction(
            sample_id="DEV-SAMPLE-01",
            document_status="auto_approved",
            predictions={
                "key_field": FieldPrediction(field_path="key_field", predicted_value="WRONG_KEY", confidence=0.9),
            },
        )
    }
    _, aggregate_3, decision_3 = evaluate_dataset(dev_dataset, preds_dev_wrong_key, policy)
    # On development split, escapes are not evaluated
    assert aggregate_3.escape_count is None
    assert decision_3.meets_m11_escapes is None


def test_insufficient_data_handling():
    """
    CORRECTION 6 REGRESSION TEST:
    Missing eligible fields, empty calibration populations, and undefined denominators
    must never produce an automatic pass.
    """
    # Case 1: Dataset with NO key fields defined
    gt_no_key = GroundTruthDocument(
        sample_id="S1",
        document_hash_sha256=DUMMY_HASH_1,
        annotator_ids=["a1"],
        fields={
            "non_key_1": GroundTruthField(field_path="non_key_1", expected_value="V1", is_key_field=False),
        },
    )
    sample_no_key = EvaluationSample(
        sample_id="S1",
        tenant_id="t1",
        document_type="DOC-01",
        document_ref="ref://1",
        document_hash_sha256=DUMMY_HASH_1,
        ground_truth=gt_no_key,
    )
    dataset_no_key = EvaluationDataset(
        dataset_id="DATASET-NO-KEY",
        tenant_id="t1",
        schema_id="s1",
        schema_version=1,
        document_type="DOC-01",
        dataset_split=DatasetSplit.DEVELOPMENT,
        samples=[sample_no_key],
    )
    preds_perfect = {
        "S1": SamplePrediction(
            sample_id="S1",
            predictions={
                "non_key_1": FieldPrediction(field_path="non_key_1", predicted_value="V1", confidence=0.99),
            },
        )
    }
    _, aggregate_no_key, decision_no_key = evaluate_dataset(dataset_no_key, preds_perfect)

    # Missing key fields must fail M1 key-fields and block overall pass!
    assert aggregate_no_key.key_field_count == 0
    assert aggregate_no_key.has_insufficient_data is True
    assert decision_no_key.meets_m1_key_fields is False
    assert decision_no_key.overall_pass is False
    assert any("No key fields defined" in r for r in decision_no_key.violations)

    # Case 2: Empty calibration population (zero predictions with confidence)
    preds_no_confs = {
        "S1": SamplePrediction(
            sample_id="S1",
            predictions={},  # Empty predictions
        )
    }
    _, aggregate_no_cal, decision_no_cal = evaluate_dataset(dataset_no_key, preds_no_confs)
    assert aggregate_no_cal.calibration_sample_count == 0
    assert aggregate_no_cal.dataset_calibration_gap_points is None
    assert aggregate_no_cal.has_insufficient_data is True
    assert decision_no_cal.meets_m6_calibration is False
    assert decision_no_cal.overall_pass is False
    assert any("Empty calibration population" in r for r in decision_no_cal.violations)


def test_token_f1_threshold_separation():
    """
    CORRECTION 4 REGRESSION TEST:
    Clearly separate the raw token-F1 metric from the configurable acceptance threshold.
    """
    gt_field = GroundTruthField(
        field_path="diagnosis",
        expected_value="acute appendicitis with localized peritonitis",
        match_strategy=MatchStrategy.TOKEN_F1,
        token_f1_threshold=0.8,
    )
    # Prediction has 2 overlapping tokens out of 5 ("acute", "appendicitis") -> Token-F1 ~ 0.5714
    pred = FieldPrediction(
        field_path="diagnosis",
        predicted_value="acute appendicitis",
        confidence=0.95,
    )

    metrics_thresh_80 = compute_field_metrics(
        "diagnosis",
        ground_truth_fields=[gt_field],
        predictions=[pred],
        token_f1_threshold=0.8,
    )
    # Under 0.8 threshold: is_match is False -> TP=0, FP=1, FN=1, F1=0.0
    assert metrics_thresh_80.true_positives == 0
    assert metrics_thresh_80.f1 == 0.0
    # But raw token-F1 score is preserved!
    assert metrics_thresh_80.mean_token_f1 is not None
    assert round(metrics_thresh_80.mean_token_f1, 4) == 0.5714

    # Now evaluate under a lenient threshold (0.5)
    gt_field_lenient = GroundTruthField(
        field_path="diagnosis",
        expected_value="acute appendicitis with localized peritonitis",
        match_strategy=MatchStrategy.TOKEN_F1,
        token_f1_threshold=0.5,
    )
    metrics_thresh_50 = compute_field_metrics(
        "diagnosis",
        ground_truth_fields=[gt_field_lenient],
        predictions=[pred],
        token_f1_threshold=0.5,
    )
    # Under 0.5 threshold: is_match is True -> TP=1, FP=0, FN=0, F1=1.0
    assert metrics_thresh_50.true_positives == 1
    assert metrics_thresh_50.f1 == 1.0
    assert round(metrics_thresh_50.mean_token_f1, 4) == 0.5714


# ============================================================================
# 7. Final Metric Integrity Regression Tests (TASK-4.3 Stage 1 Audit)
# ============================================================================

def test_token_f1_omission_handling():
    """
    INTEGRITY CHECK 1:
    - A non-null ground-truth free-text field with a missing/null prediction must
      contribute token-F1 = 0.0.
    - Omissions must NOT be excluded from the mean token-F1 calculation (which would
      artificially inflate text extraction performance).
    - Preserves correct FP/FN accounting: omissions increment FN only.
    """
    gt_field = GroundTruthField(
        field_path="clinical_notes",
        expected_value="patient presented with fever and dry cough",
        match_strategy=MatchStrategy.TOKEN_F1,
        token_f1_threshold=0.8,
    )
    # 3 samples:
    # Sample 1: Perfect match (token-F1 = 1.0)
    # Sample 2: Model prediction is None (omitted) -> token-F1 = 0.0, FN += 1
    # Sample 3: Model predicts empty/null -> token-F1 = 0.0, FN += 1
    gt_fields = [gt_field, gt_field, gt_field]
    predictions = [
        FieldPrediction(
            field_path="clinical_notes",
            predicted_value="patient presented with fever and dry cough",
            confidence=0.95,
        ),
        None,
        FieldPrediction(
            field_path="clinical_notes",
            predicted_value=None,
            is_null=True,
            confidence=0.0,
        ),
    ]

    metrics = compute_field_metrics(
        "clinical_notes",
        ground_truth_fields=gt_fields,
        predictions=predictions,
    )

    # Accounting verification:
    # TP = 1 (sample 1)
    # FP = 0 (no incorrect extractions)
    # FN = 2 (two omissions)
    # TN = 0
    assert metrics.true_positives == 1
    assert metrics.false_positives == 0
    assert metrics.false_negatives == 2
    assert metrics.true_negatives == 0
    assert metrics.precision == 1.0
    assert metrics.recall == round(1 / 3, 4)
    # F1 = 2 * (1.0 * 1/3) / (1.0 + 1/3) = (2/3) / (4/3) = 0.50
    assert metrics.f1 == 0.5

    # Token-F1 verification:
    # 3 tokens scores: [1.0, 0.0, 0.0] -> Mean = 1.0 / 3 = 0.3333
    # If omissions were excluded, mean would have been 1.0 (artificially inflated)
    assert metrics.mean_token_f1 is not None
    assert metrics.mean_token_f1 == pytest.approx(round(1 / 3, 4), abs=1e-4)


def test_date_normalization_explicit_and_supported_formats():
    """
    INTEGRITY CHECK 2:
    - Normalize dates using an explicit declared date format or normalization profile.
    - Equivalent dates across supported formats (%Y-%m-%d, %d/%m/%Y, %d-%m-%Y, %d-%b-%Y, etc.)
      must match accurately.
    - Does NOT introduce ambiguous generic date parsing or fuzzy guessing.
    """
    # 1. Direct date normalizer tests
    assert normalize_date("2026-10-29") == "2026-10-29"
    assert normalize_date("29/10/2026") == "2026-10-29"
    assert normalize_date("29-10-2026") == "2026-10-29"
    assert normalize_date("29.10.2026") == "2026-10-29"
    assert normalize_date("29-Oct-2026") == "2026-10-29"
    assert normalize_date("29 October 2026") == "2026-10-29"

    # US format with explicit date_format declaration
    assert normalize_date("10/29/2026", date_format="%m/%d/%Y") == "2026-10-29"

    # 2. Field value matching with field_type="date"
    is_match, score = match_field_values(
        expected_value="2026-10-29",
        predicted_value="29/10/2026",
        strategy=MatchStrategy.EXACT_NORMALIZED,
        field_type="date",
    )
    assert is_match is True
    assert score == 1.0

    # 3. Matching with normalization_profile="date" and text month
    is_match, score = match_field_values(
        expected_value="2026-10-29",
        predicted_value="29-Oct-2026",
        strategy=MatchStrategy.EXACT_NORMALIZED,
        normalization_profile="date",
    )
    assert is_match is True
    assert score == 1.0

    # 4. Matching with declared date_format
    is_match, score = match_field_values(
        expected_value="2026-10-29",
        predicted_value="10/29/2026",
        strategy=MatchStrategy.EXACT_NORMALIZED,
        field_type="date",
        date_format="%m/%d/%Y",
    )
    assert is_match is True
    assert score == 1.0

    # 5. Non-matching dates
    is_match, score = match_field_values(
        expected_value="2026-10-29",
        predicted_value="2026-10-30",
        strategy=MatchStrategy.EXACT_NORMALIZED,
        field_type="date",
    )
    assert is_match is False
    assert score == 0.0

    # 6. Non-date strings fall back deterministically to exact match without exceptions
    is_match, score = match_field_values(
        expected_value="NOT_A_DATE",
        predicted_value="NOT_A_DATE",
        strategy=MatchStrategy.EXACT_NORMALIZED,
        field_type="date",
    )
    assert is_match is True


def test_false_fill_rate_undefined_when_zero_null_support():
    """
    INTEGRITY CHECK 3:
    - When null ground-truth support is zero, false-fill rate must be undefined (None),
      not automatically 0.0.
    - Evaluation decision must NOT claim the GR-003 target was demonstrated without
      eligible null cases; it must flag insufficient data.
    """
    gt_doc = GroundTruthDocument(
        sample_id="SAMPLE-POSITIVE-ONLY",
        document_hash_sha256=DUMMY_HASH_1,
        annotator_ids=["ann_1", "ann_2"],
        adjudicated=True,
        fields={
            "field_1": GroundTruthField(
                field_path="field_1",
                expected_value="VALUE_1",
                is_key_field=True,
                is_null=False,
            ),
        },
    )
    sample = EvaluationSample(
        sample_id="SAMPLE-POSITIVE-ONLY",
        tenant_id="tenant_x",
        document_type="INVOICE",
        document_ref="ref://1",
        document_hash_sha256=DUMMY_HASH_1,
        ground_truth=gt_doc,
    )
    dataset = EvaluationDataset(
        dataset_id="DATASET-NO-NULLS",
        tenant_id="tenant_x",
        schema_id="schema_1",
        schema_version=1,
        document_type="INVOICE",
        dataset_split=DatasetSplit.DEVELOPMENT,
        samples=[sample],
    )
    predictions = {
        "SAMPLE-POSITIVE-ONLY": SamplePrediction(
            sample_id="SAMPLE-POSITIVE-ONLY",
            predictions={
                "field_1": FieldPrediction(
                    field_path="field_1",
                    predicted_value="VALUE_1",
                    confidence=0.98,
                ),
            },
        )
    }

    # Field-level verification
    gt_field = gt_doc.fields["field_1"]
    fpred = predictions["SAMPLE-POSITIVE-ONLY"].predictions["field_1"]
    field_metrics = compute_field_metrics("field_1", [gt_field], [fpred])
    assert field_metrics.null_support == 0
    assert field_metrics.false_fill_rate is None  # Undefined, NOT 0.0

    # Aggregate and Policy evaluation requiring GR-003 compliance (max_false_fill_rate <= 0.005 / 0.5%)
    policy_with_gr003 = EvaluationPolicy(max_false_fill_rate=DEFAULT_GR003_FALSE_FILL_THRESHOLD)
    f_metrics, aggregate, decision = evaluate_dataset(dataset, predictions, policy_with_gr003)

    assert aggregate.total_null_support == 0
    assert aggregate.overall_false_fill_rate is None
    # Must NOT claim GR-003 was met!
    assert decision.meets_gr003_false_fill is False
    assert decision.overall_pass is False
    assert any("GR-003 false-fill target cannot be verified: zero eligible null ground-truth cases" in v
               for v in decision.violations)

    # In contrast, if policy does NOT mandate false-fill check (max_false_fill_rate is None):
    policy_no_gr003 = EvaluationPolicy(max_false_fill_rate=None)
    _, _, decision_lenient = evaluate_dataset(dataset, predictions, policy_no_gr003)
    assert decision_lenient.meets_gr003_false_fill is None
    assert decision_lenient.overall_pass is True


def test_gr003_false_fill_threshold_boundaries():
    """
    PRD GR-003 BOUNDARY TEST:
    Target: Blank-field benchmark false-fill rate <= 0.5% (0.005).
    Verifies exact boundary evaluation:
    - overall_false_fill_rate == 0.0050 (0.50%): meets_gr003 is True.
    - overall_false_fill_rate == 0.0049 (0.49%): meets_gr003 is True.
    - overall_false_fill_rate == 0.0051 (0.51%): meets_gr003 is False.
    """
    assert DEFAULT_GR003_FALSE_FILL_THRESHOLD == 0.005

    policy = EvaluationPolicy(max_false_fill_rate=DEFAULT_GR003_FALSE_FILL_THRESHOLD)

    def _make_dummy_aggregate(false_fill_rate: float) -> AggregateMetrics:
        return AggregateMetrics(
            macro_f1_all_fields=0.98,
            macro_f1_key_fields=0.98,
            macro_precision_all=0.98,
            macro_recall_all=0.98,
            micro_f1_all_fields=0.98,
            micro_f1_key_fields=0.98,
            micro_precision_all=0.98,
            micro_recall_all=0.98,
            mean_calibration_gap_points=1.0,
            dataset_calibration_gap_points=1.0,
            calibration_sample_count=100,
            total_null_support=1000,
            total_false_fills=int(false_fill_rate * 1000),
            overall_false_fill_rate=false_fill_rate,
            total_samples=10,
            total_field_types=2,
            key_field_count=1,
            total_predictions=100,
            total_tp=98,
            total_fp=2,
            total_fn=2,
            total_tn=98,
        )

    # Boundary 1: Exactly 0.5% (0.0050) -> PASS
    agg_exact = _make_dummy_aggregate(0.0050)
    decision_exact = policy.evaluate(agg_exact, dataset_split=DatasetSplit.DEVELOPMENT)
    assert decision_exact.meets_gr003_false_fill is True
    assert decision_exact.overall_pass is True

    # Boundary 2: 0.49% (0.0049) -> PASS
    agg_below = _make_dummy_aggregate(0.0049)
    decision_below = policy.evaluate(agg_below, dataset_split=DatasetSplit.DEVELOPMENT)
    assert decision_below.meets_gr003_false_fill is True
    assert decision_below.overall_pass is True

    # Boundary 3: 0.51% (0.0051) -> FAIL (exceeds 0.5% threshold)
    agg_above = _make_dummy_aggregate(0.0051)
    decision_above = policy.evaluate(agg_above, dataset_split=DatasetSplit.DEVELOPMENT)
    assert decision_above.meets_gr003_false_fill is False
    assert decision_above.overall_pass is False
    assert any("Overall false-fill rate (0.0051) exceeds threshold (0.0050)" in v
               for v in decision_above.violations)


