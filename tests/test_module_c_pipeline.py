"""
Comprehensive Unit and Integration Tests for Module C.
Covers:
- Schemas (DOC-01, DOC-02, DOC-03)
- Validation rules (PAN, Aadhaar Verhoeff, Dates, Phones, Chronology, Arithmetic)
- Evidence linking and extractor
- Module C Pipeline and Canonical Envelope assembly
"""

import pytest
from datetime import datetime, timezone

from doc_extractor.contracts_b import (
    BoundingBox,
    ClassificationOutput,
    ConfidenceProvenance,
    LayoutBlock,
    LayoutElementType,
    ModuleBOutputPayload,
    MVPClassificationDomain,
    MVPDocumentType,
    PageLayoutOutput,
)
from doc_extractor.contracts_c import (
    CanonicalExtractionEnvelope,
    CanonicalOutputEnvelope,
    EvidenceSnippet,
    ExtractedField,
    FlagItem,
    DocumentMetadata,
    RunMetadata,
)
from doc_extractor.schemas import CASASchema, InsuranceApplicationSchema, DischargeSummarySchema
from doc_extractor.validation_rules import (
    validate_pan,
    validate_aadhaar_checksum,
    validate_aadhaar,
    generate_valid_aadhaar,
    validate_ifsc,
    normalize_date_to_iso,
    normalize_phone_number,
    normalize_currency_amount,
    validate_chronology,
    validate_arithmetic,
    calculate_calibrated_confidence,
)
from doc_extractor.extractor import SchemaGuidedExtractor, find_evidence_for_value
from doc_extractor.pipeline_c import ModuleCPipeline


def test_validation_pan():
    valid, err = validate_pan("ABCDE1234F")
    assert valid == "ABCDE1234F"
    assert err is None

    invalid, err = validate_pan("12345ABCDE")
    assert err == "INVALID_PAN_FORMAT"

    none_val, err = validate_pan(None)
    assert none_val is None
    assert err is None


def test_validation_aadhaar():
    # Valid Aadhaar numbers using Verhoeff
    assert validate_aadhaar_checksum("234567890126") is False  # random invalid
    # Test valid Verhoeff sequence
    assert validate_aadhaar_checksum("invalid") is False
    assert validate_aadhaar_checksum("123") is False


def test_validation_dates_and_phone():
    norm_date, err = normalize_date_to_iso("22/06/1985")
    assert norm_date == "1985-06-22"
    assert err is None

    norm_date2, err2 = normalize_date_to_iso("2026-10-29")
    assert norm_date2 == "2026-10-29"
    assert err2 is None

    bad_date, err_bad = normalize_date_to_iso("not-a-date")
    assert err_bad == "INVALID_DATE_FORMAT"

    norm_phone, p_err = normalize_phone_number("9876543210")
    assert norm_phone == "+919876543210"
    assert p_err is None


def test_chronology_and_arithmetic():
    # Normal chronology
    assert validate_chronology("2026-01-01", "2026-01-10") is None

    # Mismatch chronology
    flag = validate_chronology("2026-01-10", "2026-01-01")
    assert flag is not None
    assert flag["code"] == "CHRONOLOGY_MISMATCH"

    # Arithmetic check
    assert validate_arithmetic([100.0, 200.0, 50.0], 350.0) is None
    mismatch = validate_arithmetic([100.0, 200.0], 500.0)
    assert mismatch is not None
    assert mismatch["code"] == "ARITHMETIC_MISMATCH"


def test_pipeline_c_casa_end_to_end():
    # Construct a synthetic ModuleBOutputPayload for DOC-01 CASA
    block_name = LayoutBlock(
        block_id="blk_1",
        element_type=LayoutElementType.PARAGRAPH,
        text="Applicant Name: SURESH KUMAR RAMAN",
        ocr_confidence=0.98,
        bbox_preprocessed=BoundingBox(x0=10, y0=20, x1=200, y1=40),
        bbox_original=BoundingBox(x0=10, y0=20, x1=200, y1=40),
        reading_order_index=0,
    )
    block_pan = LayoutBlock(
        block_id="blk_2",
        element_type=LayoutElementType.PARAGRAPH,
        text="PAN: ABCDE1234F",
        ocr_confidence=0.99,
        bbox_preprocessed=BoundingBox(x0=10, y0=50, x1=150, y1=70),
        bbox_original=BoundingBox(x0=10, y0=50, x1=150, y1=70),
        reading_order_index=1,
    )

    page = PageLayoutOutput(
        page_number=1,
        markdown_content="Applicant Name: SURESH KUMAR RAMAN\nPAN: ABCDE1234F\nCheque Book: [x]",
        blocks=[block_name, block_pan],
        mean_ocr_confidence=0.98,
    )

    payload = ModuleBOutputPayload(
        run_id="run_test_001",
        document_id="doc_test_001",
        tenant_id="tenant_default",
        status="processing",
        classification=ClassificationOutput(
            domain=MVPClassificationDomain.BFSI,
            document_type=MVPDocumentType.DOC_01_CASA,
            classification_confidence=0.98,
            confidence_provenance=ConfidenceProvenance.RULE_HEURISTIC,
            schema_id="DOC-01",
            schema_version="1.0.0",
        ),
        pages=[page],
        combined_markdown="Applicant Name: SURESH KUMAR RAMAN\nPAN: ABCDE1234F\nCheque Book: [x]",
        parser_model_name="mock-layout-parser",
        classifier_model_name="mock-classifier",
        processing_time_ms=120.0,
        created_at=datetime.now(timezone.utc).isoformat(),
        completed_at=datetime.now(timezone.utc).isoformat(),
    )

    pipeline = ModuleCPipeline()
    envelope = pipeline.process(payload)

    assert isinstance(envelope, CanonicalOutputEnvelope)
    assert envelope.document_id == "doc_test_001"
    assert envelope.document_metadata.document_type == "DOC-01"
    assert "applicant.full_name" in envelope.fields
    assert envelope.fields["applicant.full_name"].value == "SURESH KUMAR RAMAN"
    assert envelope.fields["applicant.full_name"].evidence is not None
    assert envelope.fields["applicant.full_name"].evidence.page == 1
    assert envelope.fields["applicant.pan_card_number"].value == "ABCDE1234F"
    assert envelope.fields["applicant.pan_card_number"].validation == "passed"


def test_pipeline_c_discharge_summary_chronology_flag():
    # Construct a synthetic ModuleBOutputPayload for DOC-03 with inverted dates
    page = PageLayoutOutput(
        page_number=1,
        markdown_content="Patient: RAJESH SHARMA\nAdmission Date: 2026-05-15\nDischarge Date: 2026-05-10",
        blocks=[
            LayoutBlock(
                block_id="b1",
                element_type=LayoutElementType.PARAGRAPH,
                text="Patient: RAJESH SHARMA",
                ocr_confidence=0.95,
                bbox_preprocessed=BoundingBox(x0=10, y0=10, x1=100, y1=30),
                bbox_original=BoundingBox(x0=10, y0=10, x1=100, y1=30),
                reading_order_index=0,
            )
        ],
        mean_ocr_confidence=0.95,
    )

    payload = ModuleBOutputPayload(
        run_id="run_test_002",
        document_id="doc_test_002",
        tenant_id="tenant_default",
        status="processing",
        classification=ClassificationOutput(
            domain=MVPClassificationDomain.HEALTHCARE,
            document_type=MVPDocumentType.DOC_03_DISCHARGE_SUMMARY,
            classification_confidence=0.96,
            confidence_provenance=ConfidenceProvenance.RULE_HEURISTIC,
            schema_id="DOC-03",
            schema_version="1.0.0",
        ),
        pages=[page],
        combined_markdown=page.markdown_content,
        parser_model_name="mock-layout-parser",
        classifier_model_name="mock-classifier",
        processing_time_ms=140.0,
        created_at=datetime.now(timezone.utc).isoformat(),
        completed_at=datetime.now(timezone.utc).isoformat(),
    )

    pipeline = ModuleCPipeline()
    envelope = pipeline.process(payload)

    # Inverted dates should trigger chronology mismatch flag and route to needs_review
    assert envelope.document_metadata.status == "needs_review"
    assert any(flag.code == "CHRONOLOGY_MISMATCH" for flag in envelope.flags)
    assert envelope.fields["admission_date"].review_required is True
    assert envelope.fields["discharge_date"].review_required is True


def test_validation_ifsc():
    valid, err = validate_ifsc("HDFC0001234")
    assert valid == "HDFC0001234"
    assert err is None

    valid2, err2 = validate_ifsc("SBIN0000300")
    assert valid2 == "SBIN0000300"
    assert err2 is None

    invalid, err_inv = validate_ifsc("HDFC1001234")
    assert err_inv == "INVALID_IFSC_FORMAT"

    none_val, err_none = validate_ifsc(None)
    assert none_val is None
    assert err_none is None


def test_validation_currency_amounts():
    amt, curr, err = normalize_currency_amount("10,000 INR")
    assert amt == 10000.0
    assert curr == "INR"
    assert err is None

    amt2, curr2, err2 = normalize_currency_amount("Rs. 45,250.75")
    assert amt2 == 45250.75
    assert curr2 == "INR"
    assert err2 is None

    amt3, curr3, err3 = normalize_currency_amount("$1,200.00")
    assert amt3 == 1200.0
    assert curr3 == "USD"
    assert err3 is None

    amt4, curr4, err4 = normalize_currency_amount(5000)
    assert amt4 == 5000.0
    assert curr4 == "INR"

    # Negative currency normalization
    neg_amt1, neg_curr1, neg_err1 = normalize_currency_amount("-500.00")
    assert neg_amt1 == -500.0
    assert neg_curr1 == "INR"
    assert neg_err1 is None

    neg_amt2, neg_curr2, neg_err2 = normalize_currency_amount("Rs. -1200")
    assert neg_amt2 == -1200.0
    assert neg_curr2 == "INR"
    assert neg_err2 is None

    neg_amt3, neg_curr3, neg_err3 = normalize_currency_amount("-$1,200.00")
    assert neg_amt3 == -1200.0
    assert neg_curr3 == "USD"
    assert neg_err3 is None


def test_verhoeff_aadhaar_generation_and_validation():
    # Generate a mathematically valid Aadhaar number with Verhoeff check digit
    valid_aadhaar = generate_valid_aadhaar("23456789012")
    assert len(valid_aadhaar) == 12
    assert validate_aadhaar_checksum(valid_aadhaar) is True

    cleaned, err = validate_aadhaar(f"{valid_aadhaar[:4]} {valid_aadhaar[4:8]} {valid_aadhaar[8:]}")
    assert cleaned == valid_aadhaar
    assert err is None

    # Corrupt last digit to ensure check digit detects single digit errors
    corrupted_last = valid_aadhaar[:-1] + str((int(valid_aadhaar[-1]) + 1) % 10)
    assert validate_aadhaar_checksum(corrupted_last) is False
    _, bad_err = validate_aadhaar(corrupted_last)
    assert bad_err == "INVALID_AADHAAR_CHECKSUM"


def test_calibrated_confidence_calculation():
    # Normal optical confidence with passing validation
    conf = calculate_calibrated_confidence(ocr_confidence=0.98, model_confidence=1.0, validation_passed=True)
    assert conf == 0.98

    # Validation failed penalty
    conf_failed = calculate_calibrated_confidence(ocr_confidence=0.98, model_confidence=1.0, validation_passed=False)
    assert conf_failed == 0.49

    # Handwriting penalty
    conf_hw = calculate_calibrated_confidence(ocr_confidence=0.90, model_confidence=1.0, validation_passed=True, is_handwritten=True)
    assert conf_hw < 0.90

    # No evidence score
    conf_no_ev = calculate_calibrated_confidence(ocr_confidence=0.90, has_evidence=False)
    assert conf_no_ev == 0.0


def test_evidence_missing_enforcement():
    # Create payload where markdown claims a PAN, but NO block or grid contains it
    page = PageLayoutOutput(
        page_number=1,
        markdown_content="Applicant Name: SURESH KUMAR RAMAN\nPAN: ABCDE1234F",
        blocks=[
            LayoutBlock(
                block_id="b_name",
                element_type=LayoutElementType.PARAGRAPH,
                text="Applicant Name: SURESH KUMAR RAMAN",
                ocr_confidence=0.95,
                bbox_preprocessed=BoundingBox(x0=10, y0=10, x1=200, y1=30),
                bbox_original=BoundingBox(x0=10, y0=10, x1=200, y1=30),
                reading_order_index=0,
            )
            # Notice: block with PAN is completely missing from blocks/grids
        ],
        mean_ocr_confidence=0.95,
    )

    payload = ModuleBOutputPayload(
        run_id="run_test_missing_ev",
        document_id="doc_test_missing_ev",
        tenant_id="tenant_default",
        status="processing",
        classification=ClassificationOutput(
            domain=MVPClassificationDomain.BFSI,
            document_type=MVPDocumentType.DOC_01_CASA,
            classification_confidence=0.98,
            schema_id="DOC-01",
            schema_version="1.0.0",
        ),
        pages=[page],
        combined_markdown=page.markdown_content,
        parser_model_name="mock-parser",
        classifier_model_name="mock-classifier",
        processing_time_ms=100.0,
        created_at=datetime.now(timezone.utc).isoformat(),
        completed_at=datetime.now(timezone.utc).isoformat(),
    )

    pipeline = ModuleCPipeline()
    envelope = pipeline.process(payload)

    # PAN without evidence must be forced to null, null_reason='absent', review_required=True
    pan_field = envelope.fields["applicant.pan_card_number"]
    assert pan_field.value is None
    assert pan_field.null_reason == "absent"
    assert pan_field.review_required is True
    assert pan_field.validation == "failed"
    assert pan_field.evidence is None
    assert any(flag.code == "EVIDENCE_MISSING" for flag in envelope.flags)
    assert envelope.document_metadata.status == "needs_review"


def test_pipeline_c_doc_02_insurance_repeating_groups():
    # Construct DOC-02 Insurance payload with nominees and medical declarations
    page = PageLayoutOutput(
        page_number=1,
        markdown_content=(
            "Proposer Name: ANITA VERMA\n"
            "Sum Insured: 500,000 INR\n"
            "Policy Term: 15 years\n"
            "Nominee: ROHIT VERMA (Spouse), 100%\n"
            "Diabetes: [ ]\n"
            "Hypertension: [x]"
        ),
        blocks=[
            LayoutBlock(
                block_id="b1",
                element_type=LayoutElementType.PARAGRAPH,
                text="Proposer Name: ANITA VERMA",
                ocr_confidence=0.96,
                bbox_preprocessed=BoundingBox(x0=10, y0=10, x1=200, y1=30),
                bbox_original=BoundingBox(x0=10, y0=10, x1=200, y1=30),
                reading_order_index=0,
            ),
            LayoutBlock(
                block_id="b2",
                element_type=LayoutElementType.PARAGRAPH,
                text="Sum Insured: 500,000 INR",
                ocr_confidence=0.95,
                bbox_preprocessed=BoundingBox(x0=10, y0=40, x1=200, y1=60),
                bbox_original=BoundingBox(x0=10, y0=40, x1=200, y1=60),
                reading_order_index=1,
            ),
            LayoutBlock(
                block_id="b3",
                element_type=LayoutElementType.PARAGRAPH,
                text="Nominee: ROHIT VERMA (Spouse), 100%",
                ocr_confidence=0.97,
                bbox_preprocessed=BoundingBox(x0=10, y0=70, x1=300, y1=90),
                bbox_original=BoundingBox(x0=10, y0=70, x1=300, y1=90),
                reading_order_index=2,
            ),
        ],
        mean_ocr_confidence=0.96,
    )

    payload = ModuleBOutputPayload(
        run_id="run_test_ins",
        document_id="doc_test_ins",
        tenant_id="tenant_default",
        status="processing",
        classification=ClassificationOutput(
            domain=MVPClassificationDomain.BFSI,
            document_type=MVPDocumentType.DOC_02_INSURANCE_APP,
            classification_confidence=0.99,
            schema_id="DOC-02",
            schema_version="1.0.0",
        ),
        pages=[page],
        combined_markdown=page.markdown_content,
        parser_model_name="mock-parser",
        classifier_model_name="mock-classifier",
        processing_time_ms=110.0,
        created_at=datetime.now(timezone.utc).isoformat(),
        completed_at=datetime.now(timezone.utc).isoformat(),
    )

    pipeline = ModuleCPipeline()
    envelope = pipeline.process(payload)

    assert envelope.fields["proposer_name"].value == "ANITA VERMA"
    assert envelope.fields["sum_insured"].value == 500000.0
    assert envelope.fields["nominees.0.name"].value == "ROHIT VERMA"
    assert envelope.fields["nominees.0.relationship"].value == "Spouse"
    assert envelope.fields["nominees.0.share_percentage"].value == 100.0
    assert envelope.fields["medical_declaration.has_diabetes"].value is False
    assert envelope.fields["medical_declaration.has_hypertension"].value is True


def test_canonical_envelope_json_serialization():
    # Verify serialization adheres to PRD §6.3 JSON Schema
    envelope = CanonicalExtractionEnvelope(
        document_id="d_01HZX123",
        document_metadata=DocumentMetadata(
            document_type="CASA_ACCOUNT_OPENING",
            schema_id="casa_account_opening",
            schema_version=3,
            classification_confidence=0.96,
            status="auto_approved",
            page_count=1,
        ),
        fields={
            "applicant.full_name": ExtractedField(
                value="SURESH KUMAR RAMAN",
                confidence=0.97,
                null_reason=None,
                review_required=False,
                validation="passed",
                evidence=EvidenceSnippet(page=1, bbox=[112.0, 240.0, 610.0, 281.0], text="S U R E S H ..."),
            ),
            "applicant.pan_card_number": ExtractedField(
                value=None,
                confidence=0.0,
                null_reason="absent",
                review_required=False,
                validation="not_applicable",
                evidence=None,
            ),
        },
        flags=[FlagItem(code="TEST_FLAG", fields=["applicant.full_name"])],
        run=RunMetadata(
            run_id="r_01HZX",
            model_versions={"parser": "mock-parser", "extractor": "Qwen2.5-7B-Instruct"},
            prompt_version="1.0.0",
        ),
    )

    json_dict = envelope.model_dump()
    assert json_dict["document_id"] == "d_01HZX123"
    assert json_dict["document_metadata"]["status"] == "auto_approved"
    assert json_dict["fields"]["applicant.full_name"]["value"] == "SURESH KUMAR RAMAN"
    assert json_dict["fields"]["applicant.pan_card_number"]["null_reason"] == "absent"
    assert json_dict["flags"][0]["code"] == "TEST_FLAG"


def test_evidence_snippet_bbox_strict_validation():
    # Valid bounding box tuple
    valid_snippet = EvidenceSnippet(
        page=1,
        bbox=(10.0, 20.0, 100.0, 50.0),
        text="Valid text",
    )
    assert valid_snippet.bbox == (10.0, 20.0, 100.0, 50.0)

    # Valid bounding box passed as list (coerced to tuple and validated)
    list_snippet = EvidenceSnippet(
        page=1,
        bbox=[15.0, 25.0, 115.0, 55.0],
        text="List bbox",
    )
    assert list_snippet.bbox == (15.0, 25.0, 115.0, 55.0)

    # Invalid: x1 < x0
    with pytest.raises(ValueError, match="x1 .* must be >= x0"):
        EvidenceSnippet(
            page=1,
            bbox=(100.0, 20.0, 50.0, 50.0),
            text="Invalid x bounds",
        )

    # Invalid: y1 < y0
    with pytest.raises(ValueError, match="y1 .* must be >= y0"):
        EvidenceSnippet(
            page=1,
            bbox=(10.0, 80.0, 100.0, 40.0),
            text="Invalid y bounds",
        )

    # Invalid coordinate count
    with pytest.raises(ValueError, match="must contain exactly 4 coordinates"):
        EvidenceSnippet(
            page=1,
            bbox=(10.0, 20.0, 30.0),  # only 3 coordinates
            text="Incomplete bbox",
        )


def test_evidence_matching_indian_currency_comma_system():
    # Page with text in Indian numbering format: "5,00,000 INR" and "10,00,000 INR"
    page = PageLayoutOutput(
        page_number=1,
        markdown_content="Initial Deposit: 5,00,000 INR\nSum Assured: 10,00,000 INR",
        blocks=[
            LayoutBlock(
                block_id="blk_dep",
                element_type=LayoutElementType.PARAGRAPH,
                text="Initial Deposit: 5,00,000 INR",
                ocr_confidence=0.97,
                bbox_preprocessed=BoundingBox(x0=10, y0=20, x1=250, y1=45),
                bbox_original=BoundingBox(x0=10, y0=20, x1=250, y1=45),
                reading_order_index=0,
            ),
            LayoutBlock(
                block_id="blk_sum",
                element_type=LayoutElementType.PARAGRAPH,
                text="Sum Assured: 10,00,000 INR",
                ocr_confidence=0.98,
                bbox_preprocessed=BoundingBox(x0=10, y0=50, x1=300, y1=75),
                bbox_original=BoundingBox(x0=10, y0=50, x1=300, y1=75),
                reading_order_index=1,
            ),
        ],
        mean_ocr_confidence=0.975,
    )

    # Test float value matching against Indian numbering string "5,00,000"
    ev1 = find_evidence_for_value(500000.0, [page])
    assert ev1 is not None
    assert ev1["page"] == 1
    assert ev1["bbox"] == (10.0, 20.0, 250.0, 45.0)

    # Test integer value matching against Indian numbering string "10,00,000"
    ev2 = find_evidence_for_value(1000000, [page])
    assert ev2 is not None
    assert ev2["page"] == 1
    assert ev2["bbox"] == (10.0, 50.0, 300.0, 75.0)


def test_document_metadata_schema_version_typing():
    # Test that DocumentMetadata accepts both int and str schema_version per PRD SCH-011
    meta_int = DocumentMetadata(
        document_type="DOC-01",
        schema_id="casa_account_opening",
        schema_version=3,
        classification_confidence=0.98,
        status="auto_approved",
        page_count=2,
    )
    assert meta_int.schema_version == 3

    meta_str = DocumentMetadata(
        document_type="DOC-01",
        schema_id="casa_account_opening",
        schema_version="1.0.0",
        classification_confidence=0.98,
        status="auto_approved",
        page_count=2,
    )
    assert meta_str.schema_version == "1.0.0"
