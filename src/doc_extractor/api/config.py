"""
Centralized configuration for the Document Extraction Engine REST API (Module M5).

Adheres to:
- PRD §10.1 ARC-001 / SEC-011: Strict zero-egress VPC/on-prem boundary.
- Production safety: Disallow debug mode in production environments.
"""

from functools import lru_cache
from typing import Any, Dict, List, Optional
from pydantic import Field, field_validator, model_validator
from doc_extractor.config import Settings
from doc_extractor.exceptions import ConfigurationError


class ApiSettings(Settings):
    """Configuration settings for the REST API application."""

    service_name: str = Field(
        default="doc-extractor-api",
        description="Name of the REST API service",
    )
    api_prefix: str = Field(
        default="/api/v1",
        description="Global route prefix for REST API endpoints (PRD §7.1)",
    )
    api_title: str = Field(
        default="Document Extraction Engine REST API",
        description="OpenAPI specification title",
    )
    api_version: str = Field(
        default="1.0.0",
        description="Semantic version of the REST API",
    )
    api_description: str = Field(
        default="On-premise schema-guided document extraction engine REST API.",
        description="OpenAPI specification description",
    )
    docs_url: Optional[str] = Field(
        default="/docs",
        description="Swagger UI documentation route (set None to disable in hardened environments)",
    )
    redoc_url: Optional[str] = Field(
        default="/redoc",
        description="ReDoc documentation route (set None to disable)",
    )
    openapi_url: Optional[str] = Field(
        default="/openapi.json",
        description="OpenAPI JSON specification route",
    )
    request_id_header: str = Field(
        default="X-Request-ID",
        description="HTTP header name used for request correlation (API-010)",
    )
    allow_client_request_id: bool = Field(
        default=True,
        description="Whether strictly validated client-provided request IDs are accepted",
    )
    cors_origins: List[str] = Field(
        default_factory=list,
        description="Allowed origins for CORS (default empty for private intranet)",
    )
    cors_allow_credentials: bool = Field(
        default=True,
        description="Whether to support credentials for CORS requests",
    )

    @model_validator(mode="after")
    def validate_production_guards(self) -> "ApiSettings":
        """Enforces security hardening in production/staging environments and CORS constraints."""
        if self.app_env in ("production", "staging") and self.debug:
            raise ValueError("Debug mode must be disabled in production and staging environments.")
        if "*" in self.cors_origins and self.cors_allow_credentials:
            raise ValueError(
                "Wildcard CORS origin ('*') cannot be combined with credentials (allow_credentials=True). "
                "Specify explicit origins to prevent cross-origin credential reflection."
            )
        if self.app_env in ("production", "staging") and "*" in self.cors_origins:
            raise ValueError("Wildcard CORS origin ('*') is strictly prohibited in production and staging environments.")
        return self


def load_api_settings(env_overrides: Optional[Dict[str, Any]] = None) -> ApiSettings:
    """
    Load and validate API settings with safe error translation.

    Raises:
        ConfigurationError: When supplied settings fail validation.
    """
    try:
        data = env_overrides or {}
        return ApiSettings(**data)
    except Exception as exc:
        raise ConfigurationError(
            message=f"Failed to load or validate API configuration: {str(exc)}",
            details={"original_error": str(exc)},
        ) from exc


@lru_cache(maxsize=1)
def get_api_settings() -> ApiSettings:
    """Dependency injection provider for API settings."""
    return load_api_settings()
