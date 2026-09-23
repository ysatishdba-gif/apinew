"""Unit tests for request model validation."""

import pytest
from pydantic import ValidationError

from app import config
from app.mod_requests import IntentRequest, RetrievalSignalsRequest


class TestModelNameValidation:
    def test_intent_request_accepts_allowed_model(self):
        req = IntentRequest(texts=["Patient with DM"], model_name="gemini-3.7-flash")
        assert req.model_name == "gemini-3.7-flash"

    def test_intent_request_accepts_omitted_model_when_default_allowed(self):
        req = IntentRequest(texts=["Patient with DM"])
        assert req.model_name is None
        assert config.MODEL_VERSION in config.ALLOWED_GEN_MODELS

    def test_intent_request_rejects_unknown_model(self):
        with pytest.raises(ValidationError, match="Invalid 'model_name'"):
            IntentRequest(texts=["Patient with DM"], model_name="gemini-99")

    def test_retrieval_signals_request_accepts_allowed_model(self):
        req = RetrievalSignalsRequest(
            queries=[{"text": ["chest x-ray"]}],
            model_name="gemini-3.5-flash",
        )
        assert req.model_name == "gemini-3.5-flash"

    def test_retrieval_signals_request_rejects_unknown_model(self):
        with pytest.raises(ValidationError, match="Invalid 'model_name'"):
            RetrievalSignalsRequest(
                queries=[{"text": ["chest x-ray"]}],
                model_name="not-a-real-model",
            )

    def test_allowed_models_include_env_default(self):
        assert config.MODEL_VERSION in config.ALLOWED_GEN_MODELS
