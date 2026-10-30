"""
Document Extraction Engine - Module A: Document Ingestion & Preprocessing.

Module A is responsible for:
- Secure document ingestion and magic-byte input validation (PDF, JPG, PNG, TIFF)
- Malware scanning pre-check integration and decompression bomb guards
- High-fidelity PDF rendering with pypdfium2 at 200 DPI
- OpenCV-based image preprocessing (de-skew, rotation correction, contrast normalisation)
- Secure tenant-isolated document & page storage with audit tracking
- Private, customer-controlled processing with zero document-data egress (ARC-001, SEC-011)
"""

__version__ = "0.1.0"

from doc_extractor.config import Settings, load_settings
from doc_extractor.exceptions import (
    ConfigurationError,
    ConflictError,
    DocumentExtractorError,
    DocumentProcessingError,
    MalwareDetectedError,
    ScannerUnavailableError,
    SecurityCheckFailedError,
    StorageError,
    UnsupportedFileError,
)
from doc_extractor.idempotency import IdempotencyStore, LocalIdempotencyStore
from doc_extractor.ingestion import IngestionResult, IngestionService, ProcessingStatus
from doc_extractor.models import (
    VALID_STATUS_TRANSITIONS,
    DocumentRecord,
    DocumentStatus,
    IdempotencyRecord,
    IngestionState,
    compute_file_sha256,
    compute_request_fingerprint,
    compute_stream_sha256,
    map_internal_state_to_external_status,
)
from doc_extractor.malware import (
    EICAR_SIGNATURE,
    MalwareScanResult,
    MalwareScanner,
    MockMalwareScanner,
    ScanStatus,
    UnconfiguredScanner,
)
from doc_extractor.handoff import (
    MODULE_A_CONTRACT_VERSION,
    CoordinateTransform,
    LocalMemoryQueueAdapter,
    ModuleAHandoffPayload,
    PageHandoffRecord,
    QueueAdapter,
)
from doc_extractor.pipeline import ModuleAPipeline
from doc_extractor.preprocessing import (
    PagePreprocessor,
    PageTransformationMetadata,
    PreprocessingResult,
)
from doc_extractor.renderer import DocumentRenderResult, PdfRenderer, RenderedPage
from doc_extractor.storage import LocalDiskStorage, StorageBackend, StoredArtifact
from doc_extractor.validation import (
    DocumentValidator,
    SupportedFormat,
    ValidationResult,
    detect_magic_format,
)
from doc_extractor.contracts_c import (
    CanonicalExtractionEnvelope,
    CanonicalOutputEnvelope,
    DocumentMetadata,
    EvidenceSnippet,
    ExtractedField,
    FlagItem,
    RunMetadata,
)
from doc_extractor.extractor import SchemaGuidedExtractor
from doc_extractor.pipeline_c import ModuleCPipeline

__all__ = [
    "__version__",
    "Settings",
    "load_settings",
    "DocumentExtractorError",
    "ConfigurationError",
    "UnsupportedFileError",
    "SecurityCheckFailedError",
    "MalwareDetectedError",
    "ScannerUnavailableError",
    "ConflictError",
    "DocumentProcessingError",
    "StorageError",
    "IngestionResult",
    "IngestionService",
    "ProcessingStatus",
    "DocumentStatus",
    "IngestionState",
    "DocumentRecord",
    "IdempotencyRecord",
    "IdempotencyStore",
    "LocalIdempotencyStore",
    "compute_stream_sha256",
    "compute_file_sha256",
    "compute_request_fingerprint",
    "map_internal_state_to_external_status",
    "VALID_STATUS_TRANSITIONS",
    "MalwareScanResult",
    "MalwareScanner",
    "MockMalwareScanner",
    "ScanStatus",
    "UnconfiguredScanner",
    "EICAR_SIGNATURE",
    "DocumentValidator",
    "SupportedFormat",
    "ValidationResult",
    "detect_magic_format",
    "StorageBackend",
    "LocalDiskStorage",
    "StoredArtifact",
    "PdfRenderer",
    "RenderedPage",
    "DocumentRenderResult",
    "PagePreprocessor",
    "PageTransformationMetadata",
    "PreprocessingResult",
    "MODULE_A_CONTRACT_VERSION",
    "ModuleAHandoffPayload",
    "PageHandoffRecord",
    "CoordinateTransform",
    "QueueAdapter",
    "LocalMemoryQueueAdapter",
    "ModuleAPipeline",
    "CanonicalExtractionEnvelope",
    "CanonicalOutputEnvelope",
    "DocumentMetadata",
    "EvidenceSnippet",
    "ExtractedField",
    "FlagItem",
    "RunMetadata",
    "SchemaGuidedExtractor",
    "ModuleCPipeline",
]

