"""
Unit tests for ContextualIntentPipeline core logic.
"""

import json
from unittest.mock import MagicMock, Mock, patch

import pytest
from google.api_core.exceptions import FailedPrecondition, InvalidArgument, NotFound

from app.exceptions import (
    LLMError,
    LLMInvalidRequestError,
    LLMLocationError,
    LLMRateLimitError,
    LLMServiceError,
    LLMTimeoutError,
)
from app.mod_intent_extraction import ContextualIntentPipeline


class TestContextualIntentPipeline:
    """Test cases for ContextualIntentPipeline."""

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
        """
        Create pipeline instance with mocked GCP dependencies.

        Note: mock_gcp_credentials is automatically applied via conftest.py (autouse=True).
        Logger is optional and not tested, so we pass None.
        """
        # Mock GCP/Vertex AI dependencies to avoid authentication
        with (
            patch("app.mod_intent_extraction.google.auth") as mock_auth,
            patch("app.mod_intent_extraction.genai") as mock_genai,
        ):
            # Mock credentials
            mock_credentials = Mock()
            mock_auth.default.return_value = (mock_credentials, "test-project")

            # Mock genai Client
            mock_client = MagicMock()
            mock_genai.Client.return_value = mock_client

            # Create pipeline instance (logger is optional, pass None)
            pipeline = ContextualIntentPipeline(
                project="test-project",
                location="us-central1",
                model="test-model",
                logger=None,
                tracer=None,
            )

            yield pipeline

    # =========================================================================
    # Core Pipeline Tests
    # =========================================================================

    def test_successful_clinical_intent_extraction(self, pipeline, mock_usage_metadata):
        """Test Case 1: Successful clinical intent extraction"""
        # Mock query expansion response
        mock_expansion_response = json.dumps(
            {
                "expanded_query": "Patient with Diabetes Mellitus and Shortness of Breath",
                "abbreviations_expanded": ["DM", "SOB"],
            }
        )

        # Mock intent extraction response
        mock_intent_response = json.dumps(
            {
                "is_clinical": True,
                "reason": "",
                "original_query": "Patient with DM and SOB",
                "expanded_query": "Patient with Diabetes Mellitus and Shortness of Breath",
                "total_intents_detected": 2,
                "intents": [
                    {
                        "intent_title": "Diabetes Mellitus",
                        "description": "Patient has diabetes mellitus",
                        "nature": "Clinical History / Chronic Condition",
                        "sub_natures": [
                            {
                                "category_path": "Condition >> Endocrine >> Diabetes",
                                "atomic_concepts": ["Diabetes Mellitus"],
                            }
                        ],
                        "final_queries": ["Diabetes Mellitus"],
                    },
                    {
                        "intent_title": "Shortness of Breath",
                        "description": "Patient experiencing shortness of breath",
                        "nature": "Symptoms & Findings / Symptom",
                        "sub_natures": [
                            {
                                "category_path": "Symptom >> Respiratory >> Dyspnea",
                                "atomic_concepts": ["Shortness of Breath"],
                            }
                        ],
                        "final_queries": ["Shortness of Breath"],
                    },
                ],
            }
        )

        # Mock _call_model to return tuple (text, usage_metadata)
        call_count = [0]

        def mock_call_model(prompt, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:  # First call is for expansion
                return (mock_expansion_response, mock_usage_metadata)
            else:  # Second call is for intent extraction
                return (mock_intent_response, mock_usage_metadata)

        with patch.object(pipeline, "_call_model", side_effect=mock_call_model):
            result = pipeline.run("Patient with DM and SOB")

        # Assertions
        assert result["is_clinical"] is True
        assert result["original_query"] == "Patient with DM and SOB"
        assert "expanded_query" in result
        assert "abbreviations_expanded" in result
        assert len(result["intents"]) == 2
        assert result["intents"][0]["intent_title"] == "Diabetes Mellitus"
        assert result["intents"][1]["intent_title"] == "Shortness of Breath"
        assert "timestamp" in result
        assert "processing_time_seconds" in result
        # Verify usage_metadata is present
        assert "usage_metadata" in result
        assert "query_expansion" in result["usage_metadata"]
        assert "intent_extraction" in result["usage_metadata"]

    def test_non_clinical_query_rejection(self, pipeline, mock_usage_metadata):
        """Test Case 2: Non-clinical query rejection"""
        # Mock query expansion response
        mock_expansion_response = json.dumps(
            {
                "expanded_query": "What is the weather today?",
                "abbreviations_expanded": [],
            }
        )

        # Mock non-clinical intent response
        mock_non_clinical_response = json.dumps(
            {
                "is_clinical": False,
                "reason": "Query is not clinical in nature",
                "original_query": "What is the weather today?",
                "expanded_query": "What is the weather today?",
                "total_intents_detected": 0,
                "intents": [],
            }
        )

        # Mock _call_model to return tuple
        call_count = [0]

        def mock_call_model(prompt, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return (mock_expansion_response, mock_usage_metadata)
            else:
                return (mock_non_clinical_response, mock_usage_metadata)

        with patch.object(pipeline, "_call_model", side_effect=mock_call_model):
            result = pipeline.run("What is the weather today?")

        # Assertions
        assert result["is_clinical"] is False
        assert "rejected_reason" in result
        assert result["rejected_reason"] == "Query is not clinical in nature"
        assert len(result["intents"]) == 0
        assert "timestamp" in result
        assert "processing_time_seconds" in result
        # Verify usage_metadata is present
        assert "usage_metadata" in result

    def test_pydantic_validation_failure(self, pipeline, mock_usage_metadata):
        """Test Case 3: Pydantic validation failure handling"""
        # Mock invalid intent response (missing total_intents_detected)
        mock_invalid_response = json.dumps(
            {
                "is_clinical": True,
                "reason": "",
                "original_query": "Patient with diabetes",
                "expanded_query": "Patient with diabetes",
                # Missing: total_intents_detected
                "intents": [
                    {
                        "intent_title": "Diabetes",
                        "description": "Patient has diabetes",
                        "nature": "Condition",
                        "sub_natures": [
                            {
                                "category_path": "Condition >> Endocrine",
                                "atomic_concepts": ["Diabetes"],
                            }
                        ],
                        "final_queries": ["Diabetes"],
                    }
                ],
            }
        )

        # Mock _call_model to return the invalid response directly
        # (extract_intents only calls _call_model once)
        def mock_call_model(prompt, **kwargs):
            return (mock_invalid_response, mock_usage_metadata)

        with patch.object(pipeline, "_call_model", side_effect=mock_call_model):
            result = pipeline.extract_intents(
                "Patient with diabetes", "Patient with diabetes"
            )

        # Assertions - error should be present (either parsing or validation)
        assert "error" in result
        assert "failed" in result["error"].lower()
        assert result["is_clinical"] is True
        assert len(result["intents"]) == 0
        assert result["total_intents_detected"] == 0
        # Verify usage_metadata is present
        assert "usage_metadata" in result

    def test_empty_invalid_json_handling(self, pipeline, mock_usage_metadata):
        """Test Case 4: Empty/invalid JSON response handling"""
        # Mock empty/malformed response that results in parsing error
        mock_empty_response = "This is not valid JSON"

        # Mock _call_model to return the invalid response directly
        # (extract_intents only calls _call_model once)
        def mock_call_model(prompt, **kwargs):
            return (mock_empty_response, mock_usage_metadata)

        with patch.object(pipeline, "_call_model", side_effect=mock_call_model):
            result = pipeline.extract_intents(
                "Patient with diabetes", "Patient with diabetes"
            )

        # Assertions - now returns parsing error instead of validation error
        assert "error" in result
        assert result["is_clinical"] is True
        assert len(result["intents"]) == 0
        assert result["total_intents_detected"] == 0
        # Verify usage_metadata is present
        assert "usage_metadata" in result

    # =========================================================================
    # LLM Exception Handling Tests
    # =========================================================================

    def test_expand_query_llm_timeout_fallback(self, pipeline):
        """Test that LLMTimeoutError in expand_query triggers graceful fallback"""

        def mock_call_model_timeout(prompt, **kwargs):
            raise LLMTimeoutError("LLM call timed out after 300 seconds")

        with patch.object(pipeline, "_call_model", side_effect=mock_call_model_timeout):
            result = pipeline.expand_query("Patient with DM")

        # Assertions - should fall back to original query
        assert result["expanded_query"] == "Patient with DM"
        assert result["abbreviations_expanded"] == []
        assert result["expansion_failed"] is True
        assert "expansion_error" in result
        assert result["expansion_error_type"] == "llm_timeout"

    def test_expand_query_llm_rate_limit_fallback(self, pipeline):
        """Test that LLMRateLimitError in expand_query triggers graceful fallback"""

        def mock_call_model_rate_limit(prompt, **kwargs):
            raise LLMRateLimitError("Rate limit exceeded")

        with patch.object(
            pipeline, "_call_model", side_effect=mock_call_model_rate_limit
        ):
            result = pipeline.expand_query("Patient with SOB")

        # Assertions - should fall back to original query
        assert result["expanded_query"] == "Patient with SOB"
        assert result["expansion_failed"] is True
        assert result["expansion_error_type"] == "rate_limit_exceeded"

    def test_expand_query_llm_service_error_fallback(self, pipeline):
        """Test that LLMServiceError in expand_query triggers graceful fallback"""

        def mock_call_model_service_error(prompt, **kwargs):
            raise LLMServiceError("LLM service unavailable")

        with patch.object(
            pipeline, "_call_model", side_effect=mock_call_model_service_error
        ):
            result = pipeline.expand_query("Patient with HTN")

        # Assertions - should fall back to original query
        assert result["expanded_query"] == "Patient with HTN"
        assert result["expansion_failed"] is True
        assert result["expansion_error_type"] == "service_unavailable"

    def test_extract_intents_llm_error_propagates(self, pipeline):
        """Test that LLMError in extract_intents propagates (not caught)"""

        def mock_call_model_error(prompt, **kwargs):
            raise LLMError("Generic LLM error")

        with (
            patch.object(pipeline, "_call_model", side_effect=mock_call_model_error),
            pytest.raises(LLMError) as exc_info,
        ):
            pipeline.extract_intents("test query", "test expanded query")

        assert "Generic LLM error" in str(exc_info.value)

    # =========================================================================
    # Model Configuration Override Tests
    # =========================================================================

    def test_model_name_passed_to_call_model(self, pipeline, mock_usage_metadata):
        """Test that model_name parameter reaches _call_model"""
        mock_response = json.dumps(
            {"expanded_query": "test", "abbreviations_expanded": []}
        )

        captured_kwargs = []

        def mock_call_model(prompt, **kwargs):
            captured_kwargs.append(kwargs)
            return (mock_response, mock_usage_metadata)

        with patch.object(pipeline, "_call_model", side_effect=mock_call_model):
            pipeline.expand_query("test", model_name="custom-model")

        # Verify model_name was passed
        assert len(captured_kwargs) == 1
        assert captured_kwargs[0].get("model_name") == "custom-model"

    def test_generation_config_passed_to_call_model(
        self, pipeline, mock_usage_metadata
    ):
        """Test that generation_config parameter reaches _call_model"""
        mock_response = json.dumps(
            {"expanded_query": "test", "abbreviations_expanded": []}
        )

        captured_kwargs = []

        def mock_call_model(prompt, **kwargs):
            captured_kwargs.append(kwargs)
            return (mock_response, mock_usage_metadata)

        custom_config = {"temperature": 0.5, "max_output_tokens": 1000}

        with patch.object(pipeline, "_call_model", side_effect=mock_call_model):
            pipeline.expand_query("test", generation_config=custom_config)

        # Verify generation_config was passed
        assert len(captured_kwargs) == 1
        assert captured_kwargs[0].get("generation_config") == custom_config

    # =========================================================================
    # Usage Metadata Aggregation Tests
    # =========================================================================

    def test_usage_metadata_in_run_response(self, pipeline):
        """Test that run() response contains usage_metadata from both steps"""
        expansion_metadata = {
            "prompt_token_count": 50,
            "candidates_token_count": 25,
            "total_token_count": 75,
            "thinking_token_count": 0,
        }
        intent_metadata = {
            "prompt_token_count": 200,
            "candidates_token_count": 100,
            "total_token_count": 300,
            "thinking_token_count": 10,
        }

        mock_expansion = json.dumps(
            {"expanded_query": "expanded test", "abbreviations_expanded": []}
        )
        mock_intent = json.dumps(
            {
                "is_clinical": True,
                "reason": "",
                "original_query": "test",
                "expanded_query": "expanded test",
                "total_intents_detected": 0,
                "intents": [],
            }
        )

        call_count = [0]

        def mock_call_model(prompt, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return (mock_expansion, expansion_metadata)
            else:
                return (mock_intent, intent_metadata)

        with patch.object(pipeline, "_call_model", side_effect=mock_call_model):
            result = pipeline.run("test")

        # Verify usage_metadata structure
        assert "usage_metadata" in result
        assert "query_expansion" in result["usage_metadata"]
        assert "intent_extraction" in result["usage_metadata"]
        assert result["usage_metadata"]["query_expansion"] == expansion_metadata
        assert result["usage_metadata"]["intent_extraction"] == intent_metadata

    def test_usage_metadata_structure(self, pipeline, mock_usage_metadata):
        """Test that usage_metadata has correct structure"""
        mock_response = json.dumps(
            {"expanded_query": "test", "abbreviations_expanded": []}
        )

        def mock_call_model(prompt, **kwargs):
            return (mock_response, mock_usage_metadata)

        with patch.object(pipeline, "_call_model", side_effect=mock_call_model):
            result = pipeline.expand_query("test")

        # Verify usage_metadata structure
        assert "usage_metadata" in result
        metadata = result["usage_metadata"]
        assert "prompt_token_count" in metadata
        assert "candidates_token_count" in metadata
        assert "total_token_count" in metadata
        assert "thinking_token_count" in metadata


class TestCallModelLocationErrorClassification:
    """Exercises the real _call_model exception handling (not mocked away) to
    verify downstream google API errors are classified into LLMLocationError
    (404) vs. the existing generic error types."""

    @pytest.fixture
    def pipeline(self):
        with (
            patch("app.mod_intent_extraction.google.auth") as mock_auth,
            patch("app.mod_intent_extraction.genai") as mock_genai,
        ):
            mock_auth.default.return_value = (Mock(), "test-project")
            mock_client = MagicMock()
            mock_genai.Client.return_value = mock_client
            pipeline = ContextualIntentPipeline(
                project="test-project",
                location="us-central1",
                model="test-model",
                logger=None,
                tracer=None,
            )
            yield pipeline

    def test_failed_precondition_always_becomes_location_error(self, pipeline):
        pipeline.client.models.generate_content.side_effect = FailedPrecondition(
            "model not supported in this configuration"
        )
        with pytest.raises(LLMLocationError) as exc_info:
            pipeline._call_model("prompt")
        assert exc_info.value.status_code == 404

    def test_not_found_with_location_marker_becomes_location_error(self, pipeline):
        pipeline.client.models.generate_content.side_effect = NotFound(
            "Publisher Model was not found in location us-east1"
        )
        with pytest.raises(LLMLocationError):
            pipeline._call_model("prompt")

    def test_not_found_without_marker_stays_invalid_request(self, pipeline):
        pipeline.client.models.generate_content.side_effect = NotFound(
            "Resource xyz missing"
        )
        with pytest.raises(LLMInvalidRequestError) as exc_info:
            pipeline._call_model("prompt")
        assert exc_info.value.status_code == 400

    def test_invalid_argument_with_region_marker_becomes_location_error(self, pipeline):
        pipeline.client.models.generate_content.side_effect = InvalidArgument(
            "Invalid region specified for this request"
        )
        with pytest.raises(LLMLocationError):
            pipeline._call_model("prompt")

    def test_invalid_argument_without_marker_stays_invalid_request(self, pipeline):
        pipeline.client.models.generate_content.side_effect = InvalidArgument(
            "Missing required field 'prompt'"
        )
        with pytest.raises(LLMInvalidRequestError) as exc_info:
            pipeline._call_model("prompt")
        assert exc_info.value.status_code == 400

    def test_generic_exception_with_marker_becomes_location_error(self, pipeline):
        pipeline.client.models.generate_content.side_effect = RuntimeError(
            "requested model was not found for this project"
        )
        with pytest.raises(LLMLocationError):
            pipeline._call_model("prompt")

    def test_generic_exception_without_marker_stays_generic(self, pipeline):
        pipeline.client.models.generate_content.side_effect = RuntimeError(
            "unexpected failure"
        )
        with pytest.raises(LLMError) as exc_info:
            pipeline._call_model("prompt")
        assert exc_info.value.status_code == 502
