"""
FastAPI application factory and lifecycle management (Module M5).

Adheres to:
- PRD §7.2: Canonical REST API architecture.
- Centralized configuration and decoupled application initialization.
- Startup/shutdown lifecycle via modern async lifespan context manager.
- Request ID middleware and API-010 error handling registration.
- Health endpoint integration reusing get_module_health.
- Schema-compliant OpenAPI generation.
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi

from doc_extractor.api.config import ApiSettings, load_api_settings
from doc_extractor.api.exceptions import register_exception_handlers
from doc_extractor.api.middleware import RequestIdMiddleware
from doc_extractor.api.routers.health import router as health_router

logger = logging.getLogger("doc_extractor.api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifecycle context manager.
    Coordinates startup validation and clean resource shutdown.
    """
    settings: ApiSettings = app.state.settings
    logger.info(
        "Initializing Document Extraction Engine API (env=%s, service=%s, egress_isolated=%s)",
        settings.app_env,
        settings.service_name,
        not settings.allow_external_egress,
    )

    # Verification: Private VPC / Zero-egress boundary (ARC-001 / SEC-011)
    if settings.allow_external_egress:
        raise RuntimeError("FATAL: External egress is enabled. System must run in isolated VPC/on-prem.")

    yield

    logger.info("Shutting down Document Extraction Engine API. Releasing resources.")


def create_app(settings: Optional[ApiSettings] = None) -> FastAPI:
    """
    Application factory building the FastAPI HTTP application.

    Args:
        settings: Optional pre-loaded ApiSettings. If omitted, loaded from environment.

    Returns:
        Configured FastAPI instance with middleware, routes, and exception handlers.
    """
    resolved_settings = settings or load_api_settings()

    app = FastAPI(
        title=resolved_settings.api_title,
        version=resolved_settings.api_version,
        description=resolved_settings.api_description,
        docs_url=resolved_settings.docs_url,
        redoc_url=resolved_settings.redoc_url,
        openapi_url=resolved_settings.openapi_url,
        lifespan=lifespan,
    )

    # Attach centralized configuration to application state
    app.state.settings = resolved_settings

    # Initialize hexagonal adapter slots on state (None until concrete adapters configured)
    app.state.auth_provider = None
    app.state.document_metadata_repo = None
    app.state.review_task_repo = None
    app.state.api_key_repo = None
    app.state.webhook_subscription_repo = None
    app.state.webhook_delivery_repo = None
    app.state.audit_repo = None
    app.state.queue_producer = None
    app.state.queue_consumer = None
    app.state.audit_logger = None
    app.state.malware_scanner = None

    # Register CORS if origins configured
    if resolved_settings.cors_origins:
        if "*" in resolved_settings.cors_origins and resolved_settings.cors_allow_credentials:
            raise ValueError(
                "Wildcard origin '*' cannot be combined with allow_credentials=True. "
                "Specify explicit origins to prevent cross-origin credential reflection."
            )
        app.add_middleware(
            CORSMiddleware,
            allow_origins=resolved_settings.cors_origins,
            allow_credentials=resolved_settings.cors_allow_credentials,
            allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
            allow_headers=["*"],
            expose_headers=[resolved_settings.request_id_header],
        )

    # Register Request ID middleware
    app.add_middleware(
        RequestIdMiddleware,
        header_name=resolved_settings.request_id_header,
        allow_client_request_id=resolved_settings.allow_client_request_id,
    )

    # Register canonical API-010 exception handlers
    register_exception_handlers(app)

    # Register routers: root /health and versioned /v1/health
    app.include_router(health_router, prefix="", tags=["Health & Status"])
    if resolved_settings.api_prefix:
        app.include_router(health_router, prefix=resolved_settings.api_prefix, tags=["Health & Status"])

    # Custom OpenAPI generator ensuring metadata and clean schemas
    def custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema
        openapi_schema = get_openapi(
            title=resolved_settings.api_title,
            version=resolved_settings.api_version,
            description=resolved_settings.api_description,
            routes=app.routes,
        )
        app.openapi_schema = openapi_schema
        return app.openapi_schema

    app.openapi = custom_openapi

    return app
