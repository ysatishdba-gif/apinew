"""
Unit tests for JSON parsing functionality in ContextualIntentPipeline.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest

from app.mod_intent_extraction import ContextualIntentPipeline


class TestJsonParsing:
    """Test cases for JSON parsing and error handling."""

    @pytest.fixture
    def mock_usage_metadata(self):
        """Standard usage metadata for mocking."""
        return {
            "prompt_token_count": 100,
            "candidates_token_count": 50,
            "total_token_count": 150,
            "thinking_token_count": 0,
        }

    @pytest.fixture
    def pipeline(self):
        """Create pipeline instance with mocked GCP dependencies."""
        with (
            patch("app.mod_intent_extraction.google.auth") as mock_auth,
            patch("app.mod_intent_extraction.genai") as mock_genai,
        ):
            mock_credentials = Mock()
            mock_auth.default.return_value = (mock_credentials, "test-project")
            mock_genai.Client.return_value = MagicMock()

            pipeline = ContextualIntentPipeline(
                project="test-project",
                location="us-central1",
                model="test-model",
                logger=None,
                tracer=None,
            )

            yield pipeline

    def test_truncated_response_detection(self, pipeline):
        """Test that _safe_json detects truncated responses"""
        # Response that doesn't end with }
        truncated_response = '{"is_clinical": true, "intents": ['

        result = pipeline._safe_json(truncated_response)

        # Should return error dict with truncation flag
        assert result.get("_parsing_error") is True
        assert result.get("_is_truncated") is True
        assert "_error_message" in result

    def test_parsing_error_returns_error_dict(self, pipeline):
        """Test that _safe_json returns error dict with proper fields"""
        invalid_json = "This is definitely not JSON at all"

        result = pipeline._safe_json(invalid_json)

        # Should return error dict
        assert result.get("_parsing_error") is True
        assert "_error_message" in result
        assert "_response_length" in result
        assert "_response_snippet" in result

    def test_parsing_error_in_intent_extraction(self, pipeline, mock_usage_metadata):
        """Test that extract_intents handles _parsing_error and returns error_details"""

        def mock_call_model(prompt, **kwargs):
            return ("not valid json {{{", mock_usage_metadata)

        with patch.object(pipeline, "_call_model", side_effect=mock_call_model):
            result = pipeline.extract_intents("test query", "test expanded")

        # Should return error response
        assert "error" in result
        assert result["error"] == "LLM response parsing failed"
        assert "error_details" in result
        assert result["intents"] == []
        assert result["total_intents_detected"] == 0

    def test_parsing_error_in_expand_query(self, pipeline, mock_usage_metadata):
        """Test that expand_query handles _parsing_error with fallback"""

        def mock_call_model(prompt, **kwargs):
            return ("malformed json response", mock_usage_metadata)

        with patch.object(pipeline, "_call_model", side_effect=mock_call_model):
            result = pipeline.expand_query("Patient with DM")

        # Should fall back to original query
        assert result["expanded_query"] == "Patient with DM"
        assert result["expansion_failed"] is True
        assert "expansion_error" in result
