"""
Pure, Deterministic Evaluation Metrics Engine (SCH-007, M1, M6, M11, GR-003).
Strictly adheres to PRD v1.1 §13.3:
- Exact normalized matching for identifiers, dates, amounts, enums, booleans.
- Token-level F1 for free text.
- Rigorous FP/FN accounting: incorrect non-null exact-match prediction produces 1 FP and 1 FN.
- False-fill rate reported separately for predictions against null ground truth (GR-003).
- 10-bin Expected Calibration Error (ECE / M6 calibration gap).
- Strict separation of locked_test escape evaluation from dev/tuning runs.
"""

from __future__ import annotations

import collections
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from schema_lifecycle.testing.contracts import (
    AggregateMetrics,
    ConfidenceBinMetric,
    DatasetSplit,
    EvaluationDataset,
    EvaluationSample,
    FieldMetrics,
    FieldPrediction,
    GroundTruthField,
    MatchStrategy,
    SamplePrediction,
)
from schema_lifecycle.testing.policy import EvaluationDecision, EvaluationPolicy


# ============================================================================
# Pure Matching and Normalization Functions (PRD §13.3)
# ============================================================================

# Supported deterministic date formats for unambiguous ISO normalization (PRD FR-VAL-001)
SUPPORTED_DATE_FORMATS = [
    "%Y-%m-%d",      # Canonical ISO 8601 (YYYY-MM-DD)
    "%d/%m/%Y",      # Indian / UK standard (DD/MM/YYYY)
    "%d-%m-%Y",      # Indian / UK standard (DD-MM-YYYY)
    "%d.%m.%Y",      # Dot notation (DD.MM.YYYY)
    "%d-%b-%Y",      # 29-Oct-2026
    "%d %b %Y",      # 29 Oct 2026
    "%d-%B-%Y",      # 29-October-2026
    "%d %B %Y",      # 29 October 2026
    "%Y/%m/%d",      # YYYY/MM/DD
]


def normalize_date(
    val: Any,
    date_format: Optional[str] = None,
    normalization_profile: Optional[str] = None,
) -> str:
    """
    Normalizes dates using an explicit declared date format or normalization profile.
    Equivalent dates in supported formats match and standardize to ISO 8601 (YYYY-MM-DD).
    Does not introduce ambiguous generic or fuzzy date parsing.
    """
    if val is None:
        return ""
    clean = str(val).strip()
    if not clean:
        return ""

    from datetime import datetime

    # 1. Explicit declared date format takes highest precedence
    if date_format:
        try:
            dt = datetime.strptime(clean, date_format)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            pass

    # 2. Explicit profile with format (e.g. 'date:%d/%m/%Y' or 'date:%m/%d/%Y')
    if normalization_profile and normalization_profile.startswith("date:"):
        fmt = normalization_profile.split("date:", 1)[1].strip()
        try:
            dt = datetime.strptime(clean, fmt)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            pass

    # 3. Supported standard deterministic formats
    for fmt in SUPPORTED_DATE_FORMATS:
        try:
            dt = datetime.strptime(clean, fmt)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue

    return clean


def normalize_for_matching(
    val: Any,
    field_type: Optional[str] = None,
    normalization_profile: Optional[str] = None,
    date_format: Optional[str] = None,
) -> str:
    """
    Normalizes a value for exact matching per PRD §13.3 respecting data types:
    - string / enum / identifier: preserves leading zeros, preserves commas, casefolds (case-insensitive)
      unless normalization_profile is 'case_sensitive_identifier'.
    - numeric / decimal / integer: applies numeric parsing, canonical decimal formatting, comma stripping.
    - boolean: 'true' or 'false'.
    - date: normalizes to ISO 8601 (YYYY-MM-DD) via explicit declared format or supported profiles.
    """
    if val is None:
        return ""

    if isinstance(val, bool) or field_type == "boolean":
        return "true" if val else "false"

    # Date normalization via declared format or supported deterministic formats
    is_date = (
        field_type == "date"
        or date_format is not None
        or (normalization_profile and normalization_profile.startswith("date"))
    )
    if is_date:
        return normalize_date(val, date_format=date_format, normalization_profile=normalization_profile)

    # Numeric equivalence applies strictly to numeric/decimal types or explicit numeric profile
    is_numeric = (
        field_type in ["integer", "decimal", "numeric"]
        or normalization_profile == "numeric"
        or (
            field_type is None
            and normalization_profile is None
            and isinstance(val, (int, float))
            and not isinstance(val, bool)
        )
    )

    if is_numeric:
        raw_str = str(val).strip().replace(",", "")
        try:
            num = float(raw_str)
            if num.is_integer():
                return str(int(num))
            return f"{num:.6f}".rstrip("0").rstrip(".")
        except (ValueError, TypeError):
            pass

    raw_str = str(val).strip()

    # Case-sensitive string identifiers if explicitly profiled
    if normalization_profile == "case_sensitive_identifier":
        return re.sub(r"\s+", " ", raw_str)

    # Default string identifiers / account numbers / enums: preserve leading zeros, collapse whitespace, casefold
    return re.sub(r"\s+", " ", raw_str.casefold())


def compute_token_f1(expected: str, predicted: str) -> float:
    """
    Computes token-level F1 score for free text (PRD §13.3).
    Extracts word tokens (alphanumeric sequences) and computes multiset overlap.
    """
    exp_tokens = re.findall(r"\w+", expected.lower())
    pred_tokens = re.findall(r"\w+", predicted.lower())

    if not exp_tokens and not pred_tokens:
        return 1.0
    if not exp_tokens or not pred_tokens:
        return 0.0

    exp_counts = collections.Counter(exp_tokens)
    pred_counts = collections.Counter(pred_tokens)
    overlap = sum((exp_counts & pred_counts).values())

    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(exp_tokens)
    return (2.0 * precision * recall) / (precision + recall)


def match_field_values(
    expected_value: Any,
    predicted_value: Any,
    strategy: MatchStrategy = MatchStrategy.EXACT_NORMALIZED,
    tolerance: float = 0.01,
    token_f1_threshold: float = 0.8,
    field_type: Optional[str] = None,
    normalization_profile: Optional[str] = None,
    date_format: Optional[str] = None,
) -> Tuple[bool, float]:
    """
    Evaluates whether a predicted value matches expected ground truth according to strategy.
    Returns (is_match: bool, match_score: float).

    PRD §13.3 Rules:
    - Exact match after normalisation for identifiers, dates, amounts, enums, booleans.
    - Dates are normalized via explicit declared format or standard deterministic formats.
    - Numeric equivalence applies only to numeric/decimal fields; string identifiers (e.g. account numbers)
      preserve leading zeros and identifier semantics.
    - Token-level F1 for free text; acceptance is evaluated against token_f1_threshold while
      returning the exact token-level F1 score.
    - An incorrect non-null prediction returns (False, 0.0).
    """
    # Null checks
    if expected_value is None and predicted_value is None:
        return True, 1.0
    if expected_value is None or predicted_value is None:
        return False, 0.0

    # Strict boolean matching
    if strategy == MatchStrategy.BOOLEAN_EXACT or (
        isinstance(expected_value, bool) and isinstance(predicted_value, bool)
    ):
        is_match = (expected_value is predicted_value)
        return is_match, 1.0 if is_match else 0.0

    # Numeric tolerance matching
    is_numeric_type = (
        strategy == MatchStrategy.NUMERIC_TOLERANCE
        or field_type in ["integer", "decimal", "numeric"]
        or normalization_profile == "numeric"
        or (
            field_type is None
            and normalization_profile is None
            and isinstance(expected_value, (int, float))
            and isinstance(predicted_value, (int, float))
            and not isinstance(expected_value, bool)
            and not isinstance(predicted_value, bool)
        )
    )

    if strategy == MatchStrategy.NUMERIC_TOLERANCE or is_numeric_type:
        try:
            exp_num = float(str(expected_value).replace(",", "").strip())
            pred_num = float(str(predicted_value).replace(",", "").strip())
            is_match = abs(exp_num - pred_num) <= tolerance
            return is_match, 1.0 if is_match else 0.0
        except (ValueError, TypeError):
            # If not parseable as numeric, fall through to exact normalized
            pass

    # Token F1 matching for free text
    if strategy == MatchStrategy.TOKEN_F1:
        score = compute_token_f1(str(expected_value), str(predicted_value))
        is_match = (score >= token_f1_threshold)
        return is_match, score

    # Default: Exact normalized matching
    norm_exp = normalize_for_matching(
        expected_value,
        field_type=field_type,
        normalization_profile=normalization_profile,
        date_format=date_format,
    )
    norm_pred = normalize_for_matching(
        predicted_value,
        field_type=field_type,
        normalization_profile=normalization_profile,
        date_format=date_format,
    )
    is_match = (norm_exp == norm_pred)
    return is_match, 1.0 if is_match else 0.0


# ============================================================================
# Calibration Math (PRD M6 / §13.3)
# ============================================================================

def compute_10_bin_calibration(
    confidences: List[float],
    correct_flags: List[bool],
) -> Tuple[float, List[ConfidenceBinMetric]]:
    """
    Computes 10-bin confidence calibration per PRD §13.3:
    'confidence is binned in 10 equal bins; the gap is the absolute difference
    between mean confidence and accuracy, weighted by bin size.'

    Bins are [0.0, 0.1), [0.1, 0.2), ..., [0.9, 1.0].
    Returns (calibration_gap_points: float, bin_metrics: List[ConfidenceBinMetric]).
    """
    if len(confidences) != len(correct_flags):
        raise ValueError("confidences and correct_flags must have identical lengths")

    bin_samples: List[List[Tuple[float, bool]]] = [[] for _ in range(10)]

    for conf, is_corr in zip(confidences, correct_flags):
        c = max(0.0, min(1.0, float(conf)))
        if c == 1.0:
            bin_idx = 9
        else:
            bin_idx = min(int(c * 10), 9)
        bin_samples[bin_idx].append((c, is_corr))

    total_predictions = len(confidences)
    bin_metrics: List[ConfidenceBinMetric] = []
    weighted_gap_sum = 0.0

    for idx in range(10):
        lower = idx / 10.0
        upper = (idx + 1) / 10.0
        samples_in_bin = bin_samples[idx]
        count = len(samples_in_bin)

        if count > 0:
            mean_conf = sum(s[0] for s in samples_in_bin) / count
            acc = sum(1.0 for s in samples_in_bin if s[1]) / count
            gap = abs(mean_conf - acc)
            if total_predictions > 0:
                weighted_gap_sum += (count / total_predictions) * gap
        else:
            mean_conf = (lower + upper) / 2.0
            acc = 0.0
            gap = 0.0

        bin_metrics.append(
            ConfidenceBinMetric(
                bin_index=idx,
                bin_lower=lower,
                bin_upper=upper,
                sample_count=count,
                mean_confidence=round(mean_conf, 4),
                accuracy=round(acc, 4),
                calibration_gap=round(gap, 4),
            )
        )

    # Convert ECE fraction to percentage points per PRD M6 (e.g. 0.025 -> 2.5 points)
    gap_points = round(weighted_gap_sum * 100.0, 4)
    return gap_points, bin_metrics


# ============================================================================
# Field and Aggregate Metric Evaluation
# ============================================================================

def compute_field_metrics(
    field_path: str,
    ground_truth_fields: List[GroundTruthField],
    predictions: List[Optional[FieldPrediction]],
    token_f1_threshold: float = 0.8,
) -> FieldMetrics:
    """
    Computes deterministic metrics for a single schema field across samples.

    Accounting Rules:
    - GT is null, Pred is null: TN += 1
    - GT is null, Pred is non-null: FP += 1, false_fill += 1
    - GT is non-null, Pred is null: FN += 1
    - GT is non-null, Pred is non-null:
        - match: TP += 1
        - non-match: FP += 1, FN += 1 (missed ground truth and produced wrong prediction)
    """
    if len(ground_truth_fields) != len(predictions):
        raise ValueError("ground_truth_fields and predictions must have identical lengths")

    is_key = any(gt.is_key_field for gt in ground_truth_fields)
    field_type = next((gt.field_type for gt in ground_truth_fields if gt.field_type), None)
    norm_profile = next((gt.normalization_profile for gt in ground_truth_fields if gt.normalization_profile), None)

    tp = 0
    fp = 0
    fn = 0
    tn = 0
    false_fills = 0
    support = 0
    null_support = 0

    confidences: List[float] = []
    correct_flags: List[bool] = []
    token_f1_scores: List[float] = []

    for gt, pred in zip(ground_truth_fields, predictions):
        eff_token_thresh = gt.token_f1_threshold if gt.token_f1_threshold is not None else token_f1_threshold
        curr_field_type = gt.field_type or field_type
        curr_norm_profile = gt.normalization_profile or norm_profile

        if gt.is_null or gt.expected_value is None:
            null_support += 1
            if pred is None or pred.is_null or pred.predicted_value is None:
                # Correctly predicted null
                tn += 1
            else:
                # Predicted value when ground truth is null (False-Fill per PRD §13.3)
                fp += 1
                false_fills += 1
                confidences.append(pred.confidence)
                correct_flags.append(False)
        else:
            support += 1
            if pred is None or pred.is_null or pred.predicted_value is None:
                # Missed present field
                fn += 1
                if gt.match_strategy == MatchStrategy.TOKEN_F1:
                    token_f1_scores.append(0.0)
            else:
                is_match, score = match_field_values(
                    expected_value=gt.expected_value,
                    predicted_value=pred.predicted_value,
                    strategy=gt.match_strategy,
                    tolerance=gt.tolerance or 0.01,
                    token_f1_threshold=eff_token_thresh,
                    field_type=curr_field_type,
                    normalization_profile=curr_norm_profile,
                    date_format=gt.date_format,
                )
                if gt.match_strategy == MatchStrategy.TOKEN_F1:
                    token_f1_scores.append(score)

                confidences.append(pred.confidence)
                correct_flags.append(is_match)

                if is_match:
                    tp += 1
                else:
                    # Incorrect non-null prediction: both missed true value and generated false positive
                    fp += 1
                    fn += 1

    # Precision, Recall, F1 calculations
    if (tp + fp) > 0:
        precision = tp / (tp + fp)
    else:
        # If no positive predictions were made:
        # Perfect precision if no ground truth was missed and all nulls were respected
        precision = 1.0 if (tn > 0 and fn == 0) else 0.0

    if (tp + fn) > 0:
        recall = tp / (tp + fn)
    else:
        # If no ground truth positives existed:
        # Perfect recall if no false positives were hallucinated
        recall = 1.0 if fp == 0 else 0.0

    if (precision + recall) > 0.0:
        f1 = (2.0 * precision * recall) / (precision + recall)
    else:
        f1 = 0.0

    # False-fill rate: false fills / count of null ground-truth occurrences
    # PRD §13.3: Undefined (None) when null ground-truth support is zero
    if null_support > 0:
        false_fill_rate: Optional[float] = round(false_fills / null_support, 4)
    else:
        false_fill_rate = None

    # Calibration measurement
    if confidences:
        cal_gap, bins = compute_10_bin_calibration(confidences, correct_flags)
    else:
        cal_gap = 0.0
        bins = []

    mean_tok_f1 = (sum(token_f1_scores) / len(token_f1_scores)) if token_f1_scores else None
    has_sufficient = (support > 0 or null_support > 0)

    return FieldMetrics(
        field_path=field_path,
        is_key_field=is_key,
        field_type=field_type,
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        false_fill_rate=round(false_fill_rate, 4) if false_fill_rate is not None else None,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        true_negatives=tn,
        support=support,
        null_support=null_support,
        mean_token_f1=round(mean_tok_f1, 4) if mean_tok_f1 is not None else None,
        has_sufficient_data=has_sufficient,
        calibration_gap_points=round(cal_gap, 4),
        bin_metrics=bins,
    )


def compute_aggregate_metrics(
    field_metrics: Dict[str, FieldMetrics],
    dataset: EvaluationDataset,
    sample_predictions: Dict[str, SamplePrediction],
    token_f1_threshold: float = 0.8,
    primary_aggregation: Literal["macro", "micro"] = "macro",
) -> AggregateMetrics:
    """
    Computes dataset-wide macro- and micro-aggregate metrics (M1, M6, M11, SCH-007).
    Preserves strict isolation of escape evaluation to locked_test split.
    Detects and reports insufficient evaluation data explicitly.
    """
    if not field_metrics:
        raise ValueError("field_metrics cannot be empty")

    all_fields = list(field_metrics.values())
    key_fields = [f for f in all_fields if f.is_key_field]
    non_key_fields = [f for f in all_fields if not f.is_key_field]

    insufficient_reasons: List[str] = []
    has_insufficient = False

    if len(dataset.samples) == 0:
        has_insufficient = True
        insufficient_reasons.append("INSUFFICIENT_DATA: Evaluation dataset contains zero samples")

    if len(key_fields) == 0:
        has_insufficient = True
        insufficient_reasons.append(
            "INSUFFICIENT_DATA: No key fields defined in evaluation dataset (M1 key-field target unmeasurable)"
        )

    # Macro averages (unweighted mean across fields per PRD M1)
    macro_f1_all = sum(f.f1 for f in all_fields) / len(all_fields) if all_fields else 0.0
    macro_prec_all = sum(f.precision for f in all_fields) / len(all_fields) if all_fields else 0.0
    macro_rec_all = sum(f.recall for f in all_fields) / len(all_fields) if all_fields else 0.0

    if key_fields:
        macro_f1_key = sum(f.f1 for f in key_fields) / len(key_fields)
    else:
        # Zero if no key fields defined; insufficient data prevents accidental pass
        macro_f1_key = 0.0

    macro_f1_non_key = (
        sum(f.f1 for f in non_key_fields) / len(non_key_fields)
        if non_key_fields
        else None
    )

    # Micro aggregations (pooled count-based metrics)
    total_tp = sum(f.true_positives for f in all_fields)
    total_fp = sum(f.false_positives for f in all_fields)
    total_fn = sum(f.false_negatives for f in all_fields)
    total_tn = sum(f.true_negatives for f in all_fields)

    micro_prec_all = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_rec_all = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1_all = (
        (2.0 * micro_prec_all * micro_rec_all) / (micro_prec_all + micro_rec_all)
        if (micro_prec_all + micro_rec_all) > 0
        else 0.0
    )

    key_tp = sum(f.true_positives for f in key_fields)
    key_fp = sum(f.false_positives for f in key_fields)
    key_fn = sum(f.false_negatives for f in key_fields)
    key_prec = key_tp / (key_tp + key_fp) if (key_tp + key_fp) > 0 else 0.0
    key_rec = key_tp / (key_tp + key_fn) if (key_tp + key_fn) > 0 else 0.0
    micro_f1_key = (
        (2.0 * key_prec * key_rec) / (key_prec + key_rec)
        if (key_prec + key_rec) > 0
        else 0.0
    )

    # Calibration pooling across all predictions in the dataset
    all_confs: List[float] = []
    all_correct: List[bool] = []

    for sample in dataset.samples:
        spred = sample_predictions.get(sample.sample_id)
        gt_doc = sample.ground_truth
        for field_path, gt_field in gt_doc.fields.items():
            fpred = spred.predictions.get(field_path) if spred else None
            if fpred and not fpred.is_null and fpred.predicted_value is not None:
                if gt_field.is_null or gt_field.expected_value is None:
                    all_confs.append(fpred.confidence)
                    all_correct.append(False)
                else:
                    eff_tok_thresh = gt_field.token_f1_threshold or token_f1_threshold
                    is_match, _ = match_field_values(
                        expected_value=gt_field.expected_value,
                        predicted_value=fpred.predicted_value,
                        strategy=gt_field.match_strategy,
                        tolerance=gt_field.tolerance or 0.01,
                        token_f1_threshold=eff_tok_thresh,
                        field_type=gt_field.field_type,
                        normalization_profile=gt_field.normalization_profile,
                    )
                    all_confs.append(fpred.confidence)
                    all_correct.append(is_match)

    if all_confs:
        dataset_cal_gap, _ = compute_10_bin_calibration(all_confs, all_correct)
    else:
        dataset_cal_gap = None
        has_insufficient = True
        insufficient_reasons.append(
            "INSUFFICIENT_DATA: Empty calibration population (zero predictions with confidence scores; M6 unmeasurable)"
        )

    mean_cal_gap = (
        sum(f.calibration_gap_points for f in all_fields) / len(all_fields)
        if all_fields
        else 0.0
    )

    # Overall false fill rate across all null ground truth instances
    total_null_support = sum(f.null_support for f in all_fields)
    total_false_fills = sum(
        round(f.false_fill_rate * f.null_support)
        for f in all_fields
        if f.false_fill_rate is not None
    )
    if total_null_support > 0:
        overall_false_fill: Optional[float] = round(total_false_fills / total_null_support, 4)
    else:
        # PRD §13.3: Undefined/insufficient when total null ground-truth support is zero
        overall_false_fill = None

    # Escape evaluation strictly for locked_test (PRD M11 / GR-027)
    escape_count: Optional[int] = None
    total_auto_approved: Optional[int] = None

    if dataset.dataset_split == DatasetSplit.LOCKED_TEST:
        escape_count = 0
        total_auto_approved = 0

        for sample in dataset.samples:
            spred = sample_predictions.get(sample.sample_id)
            if not spred:
                continue

            # Check if auto-approved
            if spred.document_status == "auto_approved":
                total_auto_approved += 1
                has_key_error = False

                for field_path, gt_field in sample.ground_truth.fields.items():
                    # Only wrong KEY-FIELD predictions count as escapes (PRD M11)
                    if not gt_field.is_key_field:
                        continue

                    fpred = spred.predictions.get(field_path)
                    pred_val = fpred.predicted_value if fpred else None

                    # If ground truth is null
                    if gt_field.is_null or gt_field.expected_value is None:
                        if pred_val is not None:
                            has_key_error = True
                            break
                    else:
                        # Ground truth is present
                        if pred_val is None:
                            has_key_error = True
                            break
                        eff_tok_thresh = gt_field.token_f1_threshold or token_f1_threshold
                        is_match, _ = match_field_values(
                            expected_value=gt_field.expected_value,
                            predicted_value=pred_val,
                            strategy=gt_field.match_strategy,
                            tolerance=gt_field.tolerance or 0.01,
                            token_f1_threshold=eff_tok_thresh,
                            field_type=gt_field.field_type,
                            normalization_profile=gt_field.normalization_profile,
                            date_format=gt_field.date_format,
                        )
                        if not is_match:
                            has_key_error = True
                            break

                if has_key_error:
                    escape_count += 1

    return AggregateMetrics(
        primary_aggregation=primary_aggregation,
        macro_f1_all_fields=round(macro_f1_all, 4),
        macro_f1_key_fields=round(macro_f1_key, 4),
        macro_f1_non_key_fields=round(macro_f1_non_key, 4) if macro_f1_non_key is not None else None,
        macro_precision_all=round(macro_prec_all, 4),
        macro_recall_all=round(macro_rec_all, 4),
        micro_f1_all_fields=round(micro_f1_all, 4),
        micro_f1_key_fields=round(micro_f1_key, 4),
        micro_precision_all=round(micro_prec_all, 4),
        micro_recall_all=round(micro_rec_all, 4),
        mean_calibration_gap_points=round(mean_cal_gap, 4),
        dataset_calibration_gap_points=round(dataset_cal_gap, 4) if dataset_cal_gap is not None else None,
        calibration_sample_count=len(all_confs),
        total_null_support=total_null_support,
        total_false_fills=total_false_fills,
        overall_false_fill_rate=overall_false_fill,
        has_insufficient_data=has_insufficient,
        insufficient_data_reasons=insufficient_reasons,
        escape_count=escape_count,
        total_auto_approved=total_auto_approved,
        total_samples=len(dataset.samples),
        total_field_types=len(all_fields),
        key_field_count=len(key_fields),
        total_predictions=len(all_confs),
        total_tp=total_tp,
        total_fp=total_fp,
        total_fn=total_fn,
        total_tn=total_tn,
    )


def evaluate_dataset(
    dataset: EvaluationDataset,
    sample_predictions: Dict[str, SamplePrediction],
    policy: Optional[EvaluationPolicy] = None,
) -> Tuple[Dict[str, FieldMetrics], AggregateMetrics, EvaluationDecision]:
    """
    Evaluates an EvaluationDataset against predictions, producing per-field metrics,
    dataset aggregate metrics, and policy decision.
    """
    eval_policy = policy or EvaluationPolicy()

    # Collect all unique field paths across all samples
    all_field_paths: Set[str] = set()
    for sample in dataset.samples:
        for fpath in sample.ground_truth.fields.keys():
            all_field_paths.add(fpath)

    # Also collect any predicted fields that were not in ground truth
    for spred in sample_predictions.values():
        for fpath in spred.predictions.keys():
            all_field_paths.add(fpath)

    sorted_fields = sorted(all_field_paths)
    field_metrics_map: Dict[str, FieldMetrics] = {}

    for field_path in sorted_fields:
        gt_list: List[GroundTruthField] = []
        pred_list: List[Optional[FieldPrediction]] = []

        for sample in dataset.samples:
            gt_field = sample.ground_truth.fields.get(field_path)
            if gt_field is None:
                # Field absent from ground truth for this sample -> treat as null ground truth
                gt_field = GroundTruthField(
                    field_path=field_path,
                    expected_value=None,
                    is_null=True,
                    is_key_field=False,
                )
            gt_list.append(gt_field)

            spred = sample_predictions.get(sample.sample_id)
            fpred = spred.predictions.get(field_path) if spred else None
            pred_list.append(fpred)

        metrics = compute_field_metrics(
            field_path=field_path,
            ground_truth_fields=gt_list,
            predictions=pred_list,
            token_f1_threshold=eval_policy.token_f1_match_threshold,
        )
        field_metrics_map[field_path] = metrics

    aggregate_metrics = compute_aggregate_metrics(
        field_metrics=field_metrics_map,
        dataset=dataset,
        sample_predictions=sample_predictions,
        token_f1_threshold=eval_policy.token_f1_match_threshold,
        primary_aggregation=eval_policy.primary_aggregation,
    )

    decision = eval_policy.evaluate(
        metrics=aggregate_metrics,
        dataset_split=dataset.dataset_split,
    )

    return field_metrics_map, aggregate_metrics, decision
