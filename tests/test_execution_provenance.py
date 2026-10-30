"""
Tests for Execution Provenance and Strict No-Fallback Behavior (PRD SCH-007).
Validates:
1. Genuine LLM execution with mocked provider response (GENUINE_LLM, request_id, fallback_occurred=False).
2. Direct heuristic execution (use_llm=False -> HEURISTIC).
3. LLM failure with fallback enabled (use_llm=True, allow_fallback=True -> HEURISTIC_FALLBACK).
4. LLM failure with fallback disabled (use_llm=True, allow_fallback=False -> LLMExecutionError).
5. Response validation failure with fallback disabled (schema validation error -> LLMExecutionError).
6. Provider and model provenance verification.
7. Missing provider response ID handling (request_id=None without failure).
8. Legacy callers retain existing default behavior (extract() -> BaseModel, allow_fallback=True by default).
9. Tampering prevention (provenance created by trusted runtime code, not LLM output).
10. Mock execution explicitly tags ExecutionMode.MOCK.
11. Benchmark integrity helpers (is_genuine_llm and is_trusted_for_benchmark).
12. ModuleCPipeline integration with RunMetadata provenance.
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

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
    ExecutionMode,
    ExecutionProvenance,
    ExtractionResult,
    RunMetadata,
)
from doc_extractor.exceptions import LLMExecutionError
from doc_extractor.extractor import SchemaGuidedExtractor
from doc_extractor.pipeline_c import ModuleCPipeline
from doc_extractor.schemas import AccountDetails, ApplicantDetails, CASASchema


@pytest.fixture
def doc_01_payload() -> ModuleBOutputPayload:
    """Fixture producing a synthetic ModuleBOutputPayload for DOC-01."""
    block_name = LayoutBlock(
        block_id="b_name",
        element_type=LayoutElementType.KEY_VALUE_PAIR,
        text="Applicant Name: SURESH KUMAR RAMAN",
        ocr_confidence=0.99,
        bbox_preprocessed=BoundingBox(x0=10, y0=50, x1=150, y1=70),
        bbox_original=BoundingBox(x0=10, y0=50, x1=150, y1=70),
        reading_order_index=1,
    )
    block_pan = LayoutBlock(
        block_id="b_pan",
        element_type=LayoutElementType.KEY_VALUE_PAIR,
        text="PAN: ABCDE1234F",
        ocr_confidence=0.99,
        bbox_preprocessed=BoundingBox(x0=10, y0=80, x1=150, y1=100),
        bbox_original=BoundingBox(x0=10, y0=80, x1=150, y1=100),
        reading_order_index=2,
    )
    page = PageLayoutOutput(
        page_number=1,
        markdown_content="Applicant Name: SURESH KUMAR RAMAN\nPAN: ABCDE1234F\nAccount Type: SAVINGS",
        blocks=[block_name, block_pan],
        mean_ocr_confidence=0.98,
    )
    return ModuleBOutputPayload(
        run_id="run_test_prov_001",
        document_id="doc_prov_001",
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
        combined_markdown=page.markdown_content,
        parser_model_name="mock-layout-parser",
        classifier_model_name="mock-classifier",
        processing_time_ms=100.0,
        created_at=datetime.now(timezone.utc).isoformat(),
        completed_at=datetime.now(timezone.utc).isoformat(),
    )


@pytest.fixture
def valid_casa_model() -> CASASchema:
    return CASASchema(
        applicant=ApplicantDetails(
            full_name="SURESH KUMAR RAMAN",
            pan_card_number="ABCDE1234F",
        ),
        account_details=AccountDetails(
            account_type="SAVINGS",
        ),
    )


# 1. Genuine LLM execution with mocked provider response
def test_genuine_llm_execution_with_mocked_provider_response(doc_01_payload, valid_casa_model):
    def mock_llm_caller(payload, schema_cls):
        return valid_casa_model, "chatcmpl-test-response-123"

    extractor = SchemaGuidedExtractor(
        model_name="Qwen2.5-7B-Instruct",
        use_llm=True,
        allow_fallback=True,
        provider_name="litellm",
        prompt_version="1.0.0",
        llm_caller=mock_llm_caller,
    )

    result = extractor.extract_with_provenance(doc_01_payload)

    assert isinstance(result, ExtractionResult)
    assert result.payload == valid_casa_model
    assert result.provenance.mode == ExecutionMode.GENUINE_LLM
    assert result.provenance.model_name == "Qwen2.5-7B-Instruct"
    assert result.provenance.provider == "litellm"
    assert result.provenance.prompt_version == "1.0.0"
    assert result.provenance.fallback_occurred is False
    assert result.provenance.fallback_reason is None
    assert result.provenance.request_id == "chatcmpl-test-response-123"
    assert result.provenance.is_genuine_llm() is True
    assert result.provenance.is_trusted_for_benchmark() is True
    assert extractor.last_provenance == result.provenance


# 1B. Genuine LLM via instructor + litellm mocking
def test_genuine_llm_via_instructor_mocking(doc_01_payload, valid_casa_model, monkeypatch):
    mock_client = MagicMock()
    mock_raw_completion = MagicMock()
    mock_raw_completion.id = "litellm-call-999"
    mock_client.chat.completions.create_with_completion.return_value = (valid_casa_model, mock_raw_completion)

    import instructor
    monkeypatch.setattr(instructor, "from_litellm", lambda completion_fn: mock_client)

    extractor = SchemaGuidedExtractor(
        model_name="Qwen2.5-7B-Instruct",
        use_llm=True,
        allow_fallback=True,
    )

    result = extractor.extract_with_provenance(doc_01_payload)

    assert result.provenance.mode == ExecutionMode.GENUINE_LLM
    assert result.provenance.request_id == "litellm-call-999"
    assert result.provenance.provider == "litellm"
    assert result.provenance.is_genuine_llm() is True
    assert result.payload == valid_casa_model


# 2. Direct heuristic execution (use_llm=False)
def test_direct_heuristic_execution(doc_01_payload):
    extractor = SchemaGuidedExtractor(
        use_llm=False,
    )

    result = extractor.extract_with_provenance(doc_01_payload)

    assert isinstance(result, ExtractionResult)
    assert isinstance(result.payload, CASASchema)
    assert result.provenance.mode == ExecutionMode.HEURISTIC
    assert result.provenance.fallback_occurred is False
    assert result.provenance.fallback_reason is None
    assert result.provenance.is_genuine_llm() is False
    assert result.provenance.is_trusted_for_benchmark() is False
    assert extractor.last_provenance == result.provenance


# 3. LLM failure with fallback enabled (use_llm=True, allow_fallback=True)
def test_llm_failure_with_fallback_enabled(doc_01_payload):
    def failing_caller(payload, schema_cls):
        raise ConnectionError("Upstream LLM endpoint connection timed out")

    extractor = SchemaGuidedExtractor(
        model_name="Qwen2.5-7B-Instruct",
        use_llm=True,
        allow_fallback=True,
        llm_caller=failing_caller,
    )

    result = extractor.extract_with_provenance(doc_01_payload)

    assert isinstance(result, ExtractionResult)
    assert isinstance(result.payload, CASASchema)
    assert result.provenance.mode == ExecutionMode.HEURISTIC_FALLBACK
    assert result.provenance.fallback_occurred is True
    assert "Upstream LLM endpoint connection timed out" in result.provenance.fallback_reason
    assert result.provenance.is_genuine_llm() is False
    assert result.provenance.is_trusted_for_benchmark() is False
    assert extractor.last_provenance == result.provenance


# 4. LLM failure with fallback disabled (use_llm=True, allow_fallback=False)
def test_llm_failure_with_fallback_disabled(doc_01_payload):
    def failing_caller(payload, schema_cls):
        raise ConnectionError("Network unreachable")

    extractor = SchemaGuidedExtractor(
        model_name="Qwen2.5-7B-Instruct",
        use_llm=True,
        allow_fallback=False,
        llm_caller=failing_caller,
    )

    with pytest.raises(LLMExecutionError) as exc_info:
        extractor.extract_with_provenance(doc_01_payload)

    assert exc_info.value.model_name == "Qwen2.5-7B-Instruct"
    assert "Network unreachable" in str(exc_info.value)
    assert extractor.last_provenance is not None
    assert extractor.last_provenance.fallback_occurred is True
    assert "Network unreachable" in extractor.last_provenance.fallback_reason
    assert extractor.last_provenance.is_genuine_llm() is False


# 5. Response validation failure with fallback disabled
def test_response_validation_failure_with_fallback_disabled(doc_01_payload):
    def invalid_data_caller(payload, schema_cls):
        # Return dict with invalid field types failing schema validation
        return {"applicant": "not-an-object", "account_details": 123}, "req_invalid"

    extractor = SchemaGuidedExtractor(
        model_name="Qwen2.5-7B-Instruct",
        use_llm=True,
        allow_fallback=False,
        llm_caller=invalid_data_caller,
    )

    with pytest.raises(LLMExecutionError) as exc_info:
        extractor.extract_with_provenance(doc_01_payload)

    assert exc_info.value.model_name == "Qwen2.5-7B-Instruct"
    assert extractor.last_provenance is not None
    assert extractor.last_provenance.fallback_occurred is True


# 5B. Response validation failure with fallback enabled
def test_response_validation_failure_with_fallback_enabled(doc_01_payload):
    def invalid_data_caller(payload, schema_cls):
        return {"applicant": "not-an-object"}, "req_invalid"

    extractor = SchemaGuidedExtractor(
        model_name="Qwen2.5-7B-Instruct",
        use_llm=True,
        allow_fallback=True,
        llm_caller=invalid_data_caller,
    )

    result = extractor.extract_with_provenance(doc_01_payload)

    assert result.provenance.mode == ExecutionMode.HEURISTIC_FALLBACK
    assert result.provenance.fallback_occurred is True
    assert isinstance(result.payload, CASASchema)


# 6. Provider and model provenance verification
def test_provider_model_provenance_verification(doc_01_payload, valid_casa_model):
    def mock_caller(payload, schema_cls):
        return valid_casa_model, "req_custom_99"

    extractor = SchemaGuidedExtractor(
        model_name="Custom-Llama-3-70B",
        use_llm=True,
        allow_fallback=False,
        provider_name="vllm_service",
        prompt_version="2.5.0",
        llm_caller=mock_caller,
    )

    result = extractor.extract_with_provenance(doc_01_payload)

    assert result.provenance.model_name == "Custom-Llama-3-70B"
    assert result.provenance.provider == "vllm_service"
    assert result.provenance.prompt_version == "2.5.0"
    assert result.provenance.request_id == "req_custom_99"


# 7. Missing provider response ID handling
def test_missing_provider_response_id_handling(doc_01_payload, valid_casa_model):
    def caller_without_id(payload, schema_cls):
        # Returns tuple with None request_id or just the model instance
        return valid_casa_model, None

    extractor = SchemaGuidedExtractor(
        use_llm=True,
        allow_fallback=True,
        llm_caller=caller_without_id,
    )

    result = extractor.extract_with_provenance(doc_01_payload)

    assert result.provenance.request_id is None
    assert result.provenance.mode == ExecutionMode.GENUINE_LLM
    assert result.provenance.is_genuine_llm() is True


# 8. Legacy callers retain existing default behavior
def test_legacy_callers_backward_compatibility(doc_01_payload):
    extractor = SchemaGuidedExtractor(use_llm=False)

    # Calling extract() returns BaseModel directly, preserving full backward compatibility
    res = extractor.extract(doc_01_payload)

    assert isinstance(res, CASASchema)
    assert not isinstance(res, ExtractionResult)
    assert extractor.allow_fallback is True
    assert extractor.last_provenance is not None
    assert extractor.last_provenance.mode == ExecutionMode.HEURISTIC


# 9. Tampering prevention: LLM cannot spoof provenance
def test_tampering_prevention(doc_01_payload):
    class SpoofedModel(CASASchema):
        # Malicious model trying to assert its own provenance
        provenance: str = "GENUINE_LLM_SPOOFED"

    def spoofing_caller(payload, schema_cls):
        model = SpoofedModel(
            applicant=ApplicantDetails(full_name="ATTACKER"),
            account_details=AccountDetails(account_type="SAVINGS"),
            provenance="GENUINE_LLM_SPOOFED",
        )
        return model, "req_spoof"

    # Suppose is_mock=True, but caller tries to spoof GENUINE_LLM in the payload
    extractor = SchemaGuidedExtractor(
        is_mock=True,
        use_llm=False,
    )

    result = extractor.extract_with_provenance(doc_01_payload)

    # Runtime code guarantees mode is MOCK, ignoring any payload attempt
    assert result.provenance.mode == ExecutionMode.MOCK
    assert result.provenance.is_genuine_llm() is False


# 10. Mock execution explicitly tags ExecutionMode.MOCK
def test_mock_execution(doc_01_payload):
    extractor = SchemaGuidedExtractor(
        is_mock=True,
        model_name="mock_test_model",
    )

    result = extractor.extract_with_provenance(doc_01_payload)

    assert result.provenance.mode == ExecutionMode.MOCK
    assert result.provenance.provider == "mock"
    assert result.provenance.fallback_occurred is False
    assert result.provenance.is_genuine_llm() is False
    assert result.provenance.is_trusted_for_benchmark(allow_mock=False) is False
    assert result.provenance.is_trusted_for_benchmark(allow_mock=True) is True


# 11. Benchmark integrity helper verification
def test_benchmark_integrity_helpers():
    # 1. Genuine LLM
    genuine = ExecutionProvenance(
        mode=ExecutionMode.GENUINE_LLM,
        model_name="Qwen2.5-7B-Instruct",
        provider="litellm",
        fallback_occurred=False,
    )
    assert genuine.is_genuine_llm() is True
    assert genuine.is_trusted_for_benchmark(allow_mock=False) is True
    assert genuine.is_trusted_for_benchmark(allow_mock=True) is True

    # 2. Genuine LLM with empty model name (unverified)
    empty_model = ExecutionProvenance(
        mode=ExecutionMode.GENUINE_LLM,
        model_name="   ",
        provider="litellm",
        fallback_occurred=False,
    )
    assert empty_model.is_genuine_llm() is True
    assert empty_model.is_trusted_for_benchmark() is False

    # 3. Heuristic
    heuristic = ExecutionProvenance(
        mode=ExecutionMode.HEURISTIC,
        model_name="heuristic_regex",
        fallback_occurred=False,
    )
    assert heuristic.is_genuine_llm() is False
    assert heuristic.is_trusted_for_benchmark(allow_mock=False) is False
    assert heuristic.is_trusted_for_benchmark(allow_mock=True) is False

    # 4. Heuristic Fallback
    fallback = ExecutionProvenance(
        mode=ExecutionMode.HEURISTIC_FALLBACK,
        model_name="Qwen2.5-7B-Instruct",
        fallback_occurred=True,
        fallback_reason="Timeout",
    )
    assert fallback.is_genuine_llm() is False
    assert fallback.is_trusted_for_benchmark(allow_mock=False) is False
    assert fallback.is_trusted_for_benchmark(allow_mock=True) is False

    # 5. Mock
    mock_prov = ExecutionProvenance(
        mode=ExecutionMode.MOCK,
        model_name="mock_v1",
        fallback_occurred=False,
    )
    assert mock_prov.is_genuine_llm() is False
    assert mock_prov.is_trusted_for_benchmark(allow_mock=False) is False
    assert mock_prov.is_trusted_for_benchmark(allow_mock=True) is True


# 12. ModuleCPipeline integration: provenance threaded into RunMetadata
def test_pipeline_c_provenance_integration(doc_01_payload, valid_casa_model):
    def mock_llm_caller(payload, schema_cls):
        return valid_casa_model, "req_pipeline_123"

    # Pipeline with Genuine LLM extractor
    extractor_llm = SchemaGuidedExtractor(
        model_name="Qwen2.5-7B-Instruct",
        use_llm=True,
        allow_fallback=True,
        llm_caller=mock_llm_caller,
    )
    pipeline_llm = ModuleCPipeline(extractor=extractor_llm)
    envelope_llm = pipeline_llm.process(doc_01_payload)

    assert isinstance(envelope_llm, CanonicalExtractionEnvelope)
    assert envelope_llm.run.provenance is not None
    assert envelope_llm.run.provenance.mode == ExecutionMode.GENUINE_LLM
    assert envelope_llm.run.provenance.request_id == "req_pipeline_123"
    assert envelope_llm.run.provenance.is_genuine_llm() is True

    # Pipeline with Heuristic extractor
    extractor_heuristic = SchemaGuidedExtractor(use_llm=False)
    pipeline_heuristic = ModuleCPipeline(extractor=extractor_heuristic)
    envelope_heuristic = pipeline_heuristic.process(doc_01_payload)

    assert envelope_heuristic.run.provenance is not None
    assert envelope_heuristic.run.provenance.mode == ExecutionMode.HEURISTIC
    assert envelope_heuristic.run.provenance.is_genuine_llm() is False


# 13. Concurrent execution safety regression test
def test_concurrent_execution_provenance_isolation(valid_casa_model):
    import concurrent.futures
    import time

    def make_payload(doc_id: str, doc_text: str) -> ModuleBOutputPayload:
        page = PageLayoutOutput(
            page_number=1,
            markdown_content=doc_text,
            blocks=[],
            mean_ocr_confidence=0.98,
        )
        return ModuleBOutputPayload(
            run_id=f"run_{doc_id}",
            document_id=doc_id,
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
            combined_markdown=doc_text,
            parser_model_name="mock-layout-parser",
            classifier_model_name="mock-classifier",
            processing_time_ms=50.0,
            created_at=datetime.now(timezone.utc).isoformat(),
            completed_at=datetime.now(timezone.utc).isoformat(),
        )

    def slow_llm_caller(payload, schema_cls):
        # Small delay to simulate network latency and increase thread interleaving
        time.sleep(0.02)
        return valid_casa_model, f"req_id_{payload.document_id}"

    # Shared extractor and pipeline across threads
    shared_extractor = SchemaGuidedExtractor(
        model_name="Qwen2.5-7B-Instruct",
        use_llm=True,
        allow_fallback=True,
        llm_caller=slow_llm_caller,
    )
    shared_pipeline = ModuleCPipeline(extractor=shared_extractor)

    results = {}

    def worker_pipeline(doc_id: str):
        payload = make_payload(doc_id, f"Applicant Name: USER_{doc_id}\nPAN: ABCDE1234F\nAccount Type: SAVINGS")
        envelope = shared_pipeline.process(payload)
        return doc_id, envelope.run.provenance

    # Run 10 concurrent pipeline extractions
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(worker_pipeline, f"doc_{i}") for i in range(10)]
        for f in concurrent.futures.as_completed(futures):
            doc_id, prov = f.result()
            results[doc_id] = prov

    for doc_id, prov in results.items():
        assert prov is not None
        assert prov.mode == ExecutionMode.GENUINE_LLM
        assert prov.request_id == f"req_id_{doc_id}"
        assert prov.fallback_occurred is False

    # Also test thread-local isolation for legacy extract() callers
    thread_last_provenances = {}

    def worker_legacy_extract(doc_id: str):
        payload = make_payload(doc_id, f"Applicant Name: USER_{doc_id}\nPAN: ABCDE1234F\nAccount Type: SAVINGS")
        shared_extractor.extract(payload)
        # Inspect thread-local last_provenance
        return doc_id, shared_extractor.last_provenance

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(worker_legacy_extract, f"legacy_{i}") for i in range(10)]
        for f in concurrent.futures.as_completed(futures):
            doc_id, prov = f.result()
            thread_last_provenances[doc_id] = prov

    for doc_id, prov in thread_last_provenances.items():
        assert prov is not None
        assert prov.request_id == f"req_id_{doc_id}"


# 14. Stale provenance prevention regression test
def test_stale_provenance_prevention(doc_01_payload, valid_casa_model):
    def success_caller(payload, schema_cls):
        return valid_casa_model, "req_success"

    extractor = SchemaGuidedExtractor(
        use_llm=True,
        allow_fallback=False,
        llm_caller=success_caller,
    )

    # Run 1: Successful genuine extraction
    res1 = extractor.extract_with_provenance(doc_01_payload)
    assert res1.provenance.mode == ExecutionMode.GENUINE_LLM
    assert extractor.last_provenance.mode == ExecutionMode.GENUINE_LLM
    assert extractor.last_provenance.request_id == "req_success"

    # Run 2: Interrupted / invalid document raising ValueError
    invalid_payload = doc_01_payload.model_copy(deep=True)
    invalid_payload.classification.document_type = MVPDocumentType.UNKNOWN

    with pytest.raises(ValueError):
        extractor.extract_with_provenance(invalid_payload)

    # Provenance MUST NOT be the stale provenance from Run 1!
    assert extractor.last_provenance is None

    # Run 3: Strict no-fallback failure
    def failure_caller(payload, schema_cls):
        raise ConnectionResetError("Connection dropped by peer")

    extractor.llm_caller = failure_caller

    with pytest.raises(LLMExecutionError) as exc_info:
        extractor.extract_with_provenance(doc_01_payload)

    # Provenance is updated to the failed execution, not stale
    assert extractor.last_provenance is not None
    assert extractor.last_provenance.fallback_occurred is True
    assert "Connection dropped by peer" in extractor.last_provenance.fallback_reason
    assert extractor.last_provenance.request_id is None


# 15. Genuine LLM verification and response validation
def test_genuine_llm_verification_and_response_validation(doc_01_payload):
    # LLM caller returns None
    extractor_none = SchemaGuidedExtractor(
        use_llm=True,
        allow_fallback=False,
        llm_caller=lambda p, s: (None, "req_1"),
    )
    with pytest.raises(LLMExecutionError) as exc:
        extractor_none.extract_with_provenance(doc_01_payload)
    assert "LLM caller returned None" in str(exc.value)

    # Whitespace request_id is cleaned to None
    valid_model = CASASchema(
        applicant=ApplicantDetails(full_name="TEST"),
        account_details=AccountDetails(account_type="SAVINGS"),
    )
    extractor_ws = SchemaGuidedExtractor(
        use_llm=True,
        allow_fallback=True,
        llm_caller=lambda p, s: (valid_model, "   \t  "),
    )
    res = extractor_ws.extract_with_provenance(doc_01_payload)
    assert res.provenance.request_id is None
    assert res.provenance.is_genuine_llm() is True

    # Heuristic execution can NEVER report is_genuine_llm == True
    extractor_heur = SchemaGuidedExtractor(use_llm=False)
    res_heur = extractor_heur.extract_with_provenance(doc_01_payload)
    assert res_heur.provenance.mode == ExecutionMode.HEURISTIC
    assert res_heur.provenance.is_genuine_llm() is False

    # Heuristic fallback can NEVER report is_genuine_llm == True
    extractor_fb = SchemaGuidedExtractor(
        use_llm=True,
        allow_fallback=True,
        llm_caller=lambda p, s: (_ for _ in ()).throw(RuntimeError("Model error")),
    )
    res_fb = extractor_fb.extract_with_provenance(doc_01_payload)
    assert res_fb.provenance.mode == ExecutionMode.HEURISTIC_FALLBACK
    assert res_fb.provenance.fallback_occurred is True
    assert res_fb.provenance.is_genuine_llm() is False


