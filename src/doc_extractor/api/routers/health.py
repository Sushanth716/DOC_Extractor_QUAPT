"""
Health and readiness inspection endpoints for Document Extraction Engine (API-009).

Integrates existing module health checking functionality from doc_extractor.main.
"""

from typing import Any, Dict, Optional
from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, ConfigDict, Field

from doc_extractor.api.config import ApiSettings
from doc_extractor.api.dependencies import get_settings
from doc_extractor.main import get_module_health


class HealthResponse(BaseModel):
    """PRD API-009 health inspection response schema."""
    model_config = ConfigDict(extra="allow")

    status: str = Field("healthy", description="Overall service status (healthy, degraded, unhealthy)")
    service: str = Field(..., description="Service identifier")
    version: str = Field(..., description="Package version")
    api_version: str = Field(..., description="REST API semantic version")
    environment: str = Field(..., description="Deployment environment (development, staging, production)")
    egress_isolated: bool = Field(True, description="Strict private boundary confirmation (ARC-001/SEC-011)")
    target_render_dpi: int = Field(200, description="Pipeline rendering target DPI")
    malware_scan_enabled: bool = Field(True, description="Pre-processing scan status")
    malware_scanner_type: str = Field(..., description="Configured scanner backend")
    malware_scanner_ready: bool = Field(False, description="Whether scanner is active and verified")
    malware_scanner_engine: str = Field(..., description="Class name of active scanner")


router = APIRouter(tags=["Health & Status"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Check service operational status and security readiness",
    description="Returns service health, version, VPC egress boundary state, and scanner readiness.",
)
async def check_health(
    request: Request,
    settings: ApiSettings = Depends(get_settings),
) -> Dict[str, Any]:
    """Inspects engine health status, reusing existing get_module_health."""
    scanner = getattr(request.app.state, "malware_scanner", None)
    health_data = get_module_health(settings, scanner=scanner)
    health_data["api_version"] = settings.api_version
    return health_data
