"""
Evaluation Policy and Benchmark Decision Engine (M1, M6, M11, GR-003).
Strictly adheres to PRD v1.1 §2.4, §13.3, §14 G1, and SCH-007:
- Explicit, versioned evaluation policy configuration.
- Configurable macro vs. micro aggregation with documented recommendation.
- Strict isolation of locked-test-only escape evaluation (M11) from dev/tuning runs.
- No invented approval-eligibility transitions (reserved for TASK-4.2).
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, ConfigDict, Field, field_validator

from schema_lifecycle.testing.contracts import AggregateMetrics, DatasetSplit


class EvaluationDecision(BaseModel):
    """
    Structured outcome of evaluating AggregateMetrics against an EvaluationPolicy.
    Records compliance per PRD metric without invoking schema lifecycle transitions.
    """
    model_config = ConfigDict(frozen=True)

    policy_version: str = Field(..., description="Version of the evaluation policy applied")
    dataset_split: DatasetSplit = Field(..., description="Dataset split evaluated")
    primary_aggregation: Literal["macro", "micro"] = Field(..., description="Aggregation method applied for M1")
    meets_m1_all_fields: bool = Field(..., description="True if all-fields F1 meets M1 target (≥92%)")
    meets_m1_key_fields: bool = Field(..., description="True if key-fields F1 meets M1 target (≥95%)")
    meets_m6_calibration: bool = Field(..., description="True if calibration gap meets M6 target (≤3.0 points)")
    meets_m11_escapes: Optional[bool] = Field(
        default=None,
        description="True if escape count == 0 on locked set; strictly None for dev/tuning (PRD M11 / GR-027)",
    )
    meets_gr003_false_fill: Optional[bool] = Field(
        default=None,
        description="True if false-fill rate meets ceiling; False if exceeded or unmeasurable; None if reporting only",
    )
    has_insufficient_data: bool = Field(
        default=False,
        description="True if dataset lacks key fields, calibration population, or samples",
    )
    insufficient_data_reasons: List[str] = Field(
        default_factory=list,
        description="Explanations of insufficient evaluation data",
    )
    overall_pass: bool = Field(..., description="True if all applicable policy criteria are satisfied")
    violations: List[str] = Field(default_factory=list, description="List of unmet threshold descriptions")
    summary: str = Field(..., description="Human-readable summary of evaluation decision")


DEFAULT_GR003_FALSE_FILL_THRESHOLD: float = 0.005  # PRD GR-003: Blank-field benchmark false-fill rate <= 0.5% (0.005)


class EvaluationPolicy(BaseModel):
    """
    Versioned benchmark evaluation policy representing PRD M1, M6, M11, and GR-003 thresholds.

    ### Aggregation Policy & Recommendation:
    PRD M1 specifies:
      'Harmonic mean of precision and recall on key-value pairs, structured and semi-structured types:
       ≥ 92% all fields; ≥ 95% key fields'

    **Provisional Default: Macro-averaged F1 (`primary_aggregation='macro'`)**
    NOTE: Macro-aggregation is a provisional configurable default, not an established PRD decision.
    PRD v1.1 §2.4 specifies M1 targets but does not prescribe macro vs. micro aggregation.
    Rationale:
    In complex multi-field forms, low-entropy or frequent fields (e.g. currency, simple booleans)
    can dominate instance counts. Under micro-averaging, high accuracy on frequent fields can
    drown out severe degradation on critical but low-frequency fields (e.g. PAN, DOB, nominee share).
    Macro-averaging weights every schema field equally, ensuring that every field meets quality
    expectations per SCH-007 ('every field lists precision, recall, F1 and calibration').
    Both macro and micro metrics are always computed and reported; the policy explicitly records
    the selected aggregation method in the evaluation decision.
    """
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    policy_version: str = Field(default="1.0.0", description="Policy semantic version")
    min_f1_all_fields: float = Field(
        default=0.92,
        ge=0.0,
        le=1.0,
        description="PRD M1 threshold: Minimum F1 across all fields (default 0.92 = 92%)",
    )
    min_f1_key_fields: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="PRD M1 threshold: Minimum F1 across designated key fields (default 0.95 = 95%)",
    )
    max_calibration_gap_points: float = Field(
        default=3.0,
        ge=0.0,
        description="PRD M6 threshold: Maximum confidence calibration gap in percentage points (default 3.0 points)",
    )
    max_false_fill_rate: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="PRD GR-003 false-fill rate ceiling (canonical PRD target <= 0.005 = 0.5% of blank fields); None indicates reporting-only",
    )
    primary_aggregation: Literal["macro", "micro"] = Field(
        default="macro",
        description="Primary aggregation method used to evaluate M1: 'macro' (provisional default) or 'micro'",
    )
    enforce_escape_zero_tolerance: bool = Field(
        default=True,
        description="PRD M11 / GR-027: Enforce escape_count == 0 strictly on locked_test split",
    )
    token_f1_match_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Default field-match acceptance threshold for token-level F1 free-text matching (PRD §13.3)",
    )

    @field_validator("policy_version")
    @classmethod
    def validate_policy_version(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("policy_version must be a non-empty string")
        return v.strip()

    def evaluate(
        self,
        metrics: AggregateMetrics,
        dataset_split: Union[DatasetSplit, str],
    ) -> EvaluationDecision:
        """
        Evaluates AggregateMetrics against this policy.
        Preserves strict separation between locked_test escape evaluation and dev/tuning runs.
        Ensures missing eligible fields, empty calibration populations, and undefined
        denominators cannot produce an automatic pass.
        """
        split = DatasetSplit(dataset_split) if isinstance(dataset_split, str) else dataset_split
        violations: List[str] = []
        insufficient_reasons: List[str] = list(metrics.insufficient_data_reasons)
        has_insufficient = metrics.has_insufficient_data or (metrics.total_samples == 0)

        if metrics.total_samples == 0:
            insufficient_reasons.append("INSUFFICIENT_DATA: Evaluation dataset contains zero samples")

        # 1. Evaluate M1 All Fields
        f1_all = (
            metrics.macro_f1_all_fields
            if self.primary_aggregation == "macro"
            else metrics.micro_f1_all_fields
        )
        if metrics.total_field_types == 0:
            meets_m1_all = False
            has_insufficient = True
            insufficient_reasons.append("INSUFFICIENT_DATA: Zero field types evaluated")
            violations.append("M1 all-fields evaluation failed: Zero field types evaluated")
        else:
            meets_m1_all = f1_all >= self.min_f1_all_fields
            if not meets_m1_all:
                violations.append(
                    f"M1 all-fields {self.primary_aggregation}-F1 ({f1_all:.4f}) is below threshold ({self.min_f1_all_fields:.4f})"
                )

        # 2. Evaluate M1 Key Fields (Explicit key field check)
        if metrics.key_field_count == 0:
            meets_m1_key = False
            has_insufficient = True
            insufficient_reasons.append(
                "INSUFFICIENT_DATA: No key fields defined in dataset; cannot verify M1 key-field target (≥95%)"
            )
            violations.append(
                "M1 key-fields evaluation failed: No key fields defined in dataset (target: ≥95%)"
            )
        else:
            f1_key = (
                metrics.macro_f1_key_fields
                if self.primary_aggregation == "macro"
                else metrics.micro_f1_key_fields
            )
            meets_m1_key = f1_key >= self.min_f1_key_fields
            if not meets_m1_key:
                violations.append(
                    f"M1 key-fields {self.primary_aggregation}-F1 ({f1_key:.4f}) is below threshold ({self.min_f1_key_fields:.4f})"
                )

        # 3. Evaluate M6 Calibration Gap
        cal_gap = metrics.dataset_calibration_gap_points
        if cal_gap is None or metrics.calibration_sample_count == 0:
            meets_m6 = False
            has_insufficient = True
            insufficient_reasons.append(
                "INSUFFICIENT_DATA: Empty calibration population (zero predictions with confidence scores; M6 unmeasurable)"
            )
            violations.append("M6 calibration evaluation failed: Empty calibration population (zero predictions with confidence)")
        else:
            meets_m6 = cal_gap <= self.max_calibration_gap_points
            if not meets_m6:
                violations.append(
                    f"M6 dataset calibration gap ({cal_gap:.2f} pts) exceeds maximum ({self.max_calibration_gap_points:.2f} pts)"
                )

        # 4. Optional False-Fill Rate Check (GR-003)
        meets_gr003: Optional[bool] = None
        if self.max_false_fill_rate is not None:
            if metrics.overall_false_fill_rate is None or metrics.total_null_support == 0:
                meets_gr003 = False
                has_insufficient = True
                insufficient_reasons.append(
                    "INSUFFICIENT_DATA: Zero eligible null ground-truth cases (null support == 0); GR-003 false-fill target cannot be verified"
                )
                violations.append(
                    "GR-003 false-fill target cannot be verified: zero eligible null ground-truth cases in dataset (insufficient null support)"
                )
            elif metrics.overall_false_fill_rate > self.max_false_fill_rate:
                meets_gr003 = False
                violations.append(
                    f"Overall false-fill rate ({metrics.overall_false_fill_rate:.4f}) exceeds threshold ({self.max_false_fill_rate:.4f})"
                )
            else:
                meets_gr003 = True

        # 5. Evaluate M11 Escapes strictly for locked_test split (PRD M11, GR-027, Gate G1)
        meets_m11: Optional[bool] = None
        if split == DatasetSplit.LOCKED_TEST:
            if self.enforce_escape_zero_tolerance:
                escape_cnt = metrics.escape_count if metrics.escape_count is not None else 0
                meets_m11 = (escape_cnt == 0)
                if not meets_m11:
                    violations.append(
                        f"M11 escape violation: {escape_cnt} escape(s) detected on locked_test set (target: 0)"
                    )
            else:
                meets_m11 = True
        else:
            # For development or tuning, escapes are not a release gate criterion
            meets_m11 = None

        if has_insufficient:
            violations.extend([r for r in insufficient_reasons if r not in violations])

        overall_pass = len(violations) == 0 and not has_insufficient

        if overall_pass:
            cal_str = f"{cal_gap:.2f}pts" if cal_gap is not None else "N/A"
            summary = (
                f"PASS: Meets M1 (all={f1_all:.2%}, key={f1_key:.2%}) and M6 (cal={cal_str}) on {split.value} split "
                f"using {self.primary_aggregation}-aggregation."
            )
            if meets_m11 is not None:
                summary += f" Escapes={metrics.escape_count or 0}."
        else:
            reason_str = "; ".join(violations)
            summary = f"FAIL: {len(violations)} issue(s) on {split.value} split: {reason_str}."

        return EvaluationDecision(
            policy_version=self.policy_version,
            dataset_split=split,
            primary_aggregation=self.primary_aggregation,
            meets_m1_all_fields=meets_m1_all,
            meets_m1_key_fields=meets_m1_key,
            meets_m6_calibration=meets_m6,
            meets_m11_escapes=meets_m11,
            meets_gr003_false_fill=meets_gr003,
            has_insufficient_data=has_insufficient,
            insufficient_data_reasons=insufficient_reasons,
            overall_pass=overall_pass,
            violations=violations,
            summary=summary,
        )

