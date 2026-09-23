"""
Unit tests for tracer integration in ContextualIntentPipeline.
"""

from unittest.mock import MagicMock, Mock, patch

from app.mod_intent_extraction import ContextualIntentPipeline


class TestTracerIntegration:
    """Test cases for tracer integration."""

    def test_pipeline_accepts_tracer_parameter(self):
        """Test that pipeline can be instantiated with tracer parameter"""
        mock_tracer = MagicMock()

        with (
            patch("app.mod_intent_extraction.google.auth") as mock_auth,
            patch("app.mod_intent_extraction.genai") as mock_genai,
        ):
            mock_credentials = Mock()
            mock_auth.default.return_value = (mock_credentials, "test-project")
            mock_genai.Client.return_value = MagicMock()

            # Should not raise any errors
            pipeline = ContextualIntentPipeline(
                project="test-project",
                location="us-central1",
                model="test-model",
                logger=None,
                tracer=mock_tracer,
            )

            assert pipeline.tracer == mock_tracer
