"""
Pytest configuration and fixtures for unit tests.
Mocks all GCP-related services to prevent actual API calls during testing.
"""

import os
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

# ============================================================================
# ENVIRONMENT SETUP - before app import (config reads PROJECT_ID at import time)
# ============================================================================

os.environ.setdefault("ENV", "test")
os.environ.setdefault("PROJECT_ID", "test-project")
os.environ.setdefault("GCP_PROJECT_ID", "test-project")
os.environ.setdefault("GCP_LOCATION", "us-central1")
os.environ.setdefault("MODEL_VERSION", "gemini-2.5-flash")
# Skip temporal vocab load at app.main import time (GCS and local file)
os.environ.setdefault("TEMPORAL_GCS_BUCKET", "")
os.environ.setdefault("TEMPORAL_GCS_PATH", "")
os.environ.setdefault("TEMPORAL_LOCAL_PATH", "")
# Skip the record-type GCS fetch at app.main import time; the bundled local
# file (RECORD_TYPE_LOCAL_PATH default) is allowed to load.
os.environ.setdefault("RECORD_TYPE_GCS_BUCKET", "")
os.environ.setdefault("RECORD_TYPE_GCS_PATH", "")
# Avoid Cloud Trace export noise during local tests
os.environ.setdefault("ENABLE_TRACING", "false")

if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") == "":
    os.environ.pop("GOOGLE_APPLICATION_CREDENTIALS", None)

# Stub aie_logging if the package is not installed in the local env
if "aie_logging" not in sys.modules:
    _aie = types.ModuleType("aie_logging")
    _aie.GCPLogger = MagicMock
    _aie.Severity = MagicMock()
    sys.modules["aie_logging"] = _aie

mock_logger = MagicMock()
mock_logger.log_struct = MagicMock()

mock_credentials = MagicMock()
mock_project_id = "test-project"

# Small vocab so TemporalVocab.from_gcs never hits the network if called
_mock_vocab = MagicMock()
_mock_vocab._names = []
_mock_vocab.transactions.return_value = []
_mock_vocab.name_for_id.return_value = None
_mock_vocab.name_from_text.return_value = None
_mock_vocab.primary_for_name.return_value = (None, None)

patchers = [
    patch("aie_logging.GCPLogger", return_value=mock_logger),
    patch(
        "app.mod_intent_extraction.google.auth.default",
        return_value=(mock_credentials, mock_project_id),
    ),
    patch("app.mod_intent_extraction.genai.Client", return_value=MagicMock()),
    patch(
        "app.utils.temporal_vocab.TemporalVocab.from_gcs",
        return_value=_mock_vocab,
    ),
    patch(
        "app.utils.temporal_vocab.TemporalVocab.from_file",
        return_value=_mock_vocab,
    ),
]

for patcher in patchers:
    patcher.start()

if "GOOGLE_APPLICATION_CREDENTIALS" in os.environ:
    os.environ.pop("GOOGLE_APPLICATION_CREDENTIALS", None)

from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI application."""
    return TestClient(app)


@pytest.fixture(autouse=True)
def _reset_document_catalog():
    """The document catalog is cached process-wide; tests that fake BigQuery
    must not see another test's catalog."""
    from app.utils import context_lonic_document_cluster as _dc

    _dc.reset_catalog_cache()
    yield
    _dc.reset_catalog_cache()


@pytest.fixture(autouse=True)
def mock_gcp_credentials():
    """Automatically mock GCP credentials for all tests."""
    with patch.dict(
        os.environ,
        {
            "GOOGLE_APPLICATION_CREDENTIALS": "/tmp/fake-credentials.json",
        },
    ):
        yield


@pytest.fixture
def mock_pipeline():
    """Mock the ContextualIntentPipeline to avoid actual LLM calls."""
    with patch("app.main.pipeline") as mock_pipe:
        v1_response = {
            "original_query": "Patient with DM and SOB",
            "expanded_query": "Patient with Diabetes Mellitus and Shortness of Breath",
            "abbreviations_expanded": ["DM", "SOB"],
            "is_clinical": True,
            "intents": [
                {
                    "intent_title": "Diabetes Mellitus",
                    "description": "Patient has diabetes mellitus",
                    "nature": "Clinical History / Chronic Condition",
                    "sub_natures": [
                        {
                            "category_path": "Condition >> Type >> Classification",
                            "atomic_concepts": ["diabetes mellitus", "type 2"],
                        }
                    ],
                    "final_queries": ["diabetes mellitus", "type 2 diabetes"],
                }
            ],
            "timestamp": "2025-01-01T10:00:00Z",
            "processing_time_seconds": 1.5,
            "usage_metadata": {
                "query_expansion": {
                    "prompt_token_count": 10,
                    "candidates_token_count": 5,
                    "total_token_count": 15,
                    "thinking_token_count": 0,
                },
                "intent_extraction": {
                    "prompt_token_count": 20,
                    "candidates_token_count": 10,
                    "total_token_count": 30,
                    "thinking_token_count": 0,
                },
            },
        }
        mock_pipe.run.return_value = dict(v1_response)

        v2_response = {
            "original_query": "Patient with DM and SOB",
            "expanded_query": "Patient with Diabetes Mellitus and Shortness of Breath",
            "abbreviations_expanded": ["DM", "SOB"],
            "representative_terms": ["diabetes mellitus"],
            "total_intents_detected": 1,
            "intents": [
                {
                    "intent_title": "Diabetes Mellitus",
                    "description": "Patient has diabetes mellitus",
                    "nature": "Clinical History / Chronic Condition",
                    "sub_natures": [
                        {
                            "category_path": "Condition >> Type >> Classification",
                            "atomic_concepts": ["diabetes mellitus", "type 2"],
                        }
                    ],
                    "final_queries": ["diabetes mellitus", "type 2 diabetes"],
                }
            ],
            "timestamp": "2025-01-01T10:00:00Z",
            "processing_time_seconds": 1.5,
            "usage_metadata": {
                "query_expansion": {
                    "prompt_token_count": 10,
                    "candidates_token_count": 5,
                    "total_token_count": 15,
                    "thinking_token_count": 0,
                },
                "intent_extraction": {
                    "prompt_token_count": 20,
                    "candidates_token_count": 10,
                    "total_token_count": 30,
                    "thinking_token_count": 0,
                },
                "representative_terms": {
                    "prompt_token_count": 5,
                    "candidates_token_count": 2,
                    "total_token_count": 7,
                    "thinking_token_count": 0,
                },
            },
        }
        mock_pipe.run_v2.return_value = dict(v2_response)
        yield mock_pipe


@pytest.fixture
def mock_build_context():
    """Mock pipeline.build_context for endpoint / unit tests that need step 4."""
    with patch("app.main.pipeline.build_context") as mock_ctx:
        mock_ctx.return_value = {
            "context": None,
            "temporal": None,
            "usage_metadata": {
                "prompt_token_count": 0,
                "candidates_token_count": 0,
                "total_token_count": 0,
                "thinking_token_count": 0,
            },
        }
        yield mock_ctx
