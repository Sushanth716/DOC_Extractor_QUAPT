"""
Document Extraction Engine - REST API Foundation (Module M5).

Provides HTTP application factory, request middleware, centralized configuration,
canonical API-010 error handling, authentication/tenant interfaces, repository and
durable queue protocols.
"""

from doc_extractor.api.app import create_app
from doc_extractor.api.config import ApiSettings, load_api_settings

__all__ = ["create_app", "ApiSettings", "load_api_settings"]
