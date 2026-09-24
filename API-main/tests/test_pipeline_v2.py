"""Pipeline and endpoint tests for v2 nature-breakdown (mocked LLM)."""

import json
from unittest.mock import MagicMock, Mock, patch

import pytest

from app import config
from app.mod_intent_extraction import ContextualIntentPipeline
from app.utils.temporal_vocab import TemporalVocab

V1_NB = f"{config.API_PREFIX}/v1/nature-breakdown"
V2_NB = f"{config.API_PREFIX}/v2/nature-breakdown"

USAGE = {
    "prompt_token_count": 10,
    "candidates_token_count": 5,
    "total_token_count": 15,
    "thinking_token_count": 0,
}

EXPANSION_JSON = json.dumps(
    {
        "expanded_query": "Elevated prostate-specific antigen within the last three years",
        "abbreviations_expanded": ["PSA"],
    }
)

INTENT_JSON = json.dumps(
    {
        "total_intents_detected": 1,
        "intents": [
            {
                "intent_title": "Elevated PSA",
                "description": "PSA finding over time",
                "nature": "[Clinical Finding] / [Diagnostic]",
                "sub_natures": [
                    {
                        "category_path": "Lab >> PSA",
                        "atomic_concepts": [
                            "elevated PSA levels",
                            "prostate-specific antigen",
                        ],
                    }
                ],
                "final_queries": [
                    "elevated PSA levels",
                    "prostate-specific antigen",
                ],
            }
        ],
    }
)

REP_TERMS_JSON = json.dumps({"representative_terms": ["PSA", "prostate cancer"]})

CONTEXT_JSON = json.dumps(
    {
        "concepts_with_context": [
            {
                "atomic_concept": "elevated PSA levels",
                "intent_title": "Elevated PSA",
                "record_types": ["laboratory_report"],
                "author_roles": ["pathologist"],
                "longitudinal_scope": ["screening"],
                "content_signals": ["measurement_value"],
                "clinical_settings": ["outpatient"],
            },
            {
                "atomic_concept": "prostate-specific antigen",
                "intent_title": "Elevated PSA",
                "record_types": ["laboratory_report", "order_entry"],
                "author_roles": ["urologist"],
                "longitudinal_scope": ["follow_up"],
                "content_signals": ["loinc_lab_code"],
                "clinical_settings": ["specialty_clinic"],
            },
        ],
        "temporal_by_intent": [
            {
                "intent_title": "Elevated PSA",
                "candidates": [
                    {
                        "candidate": "elevated PSA levels",
                        "temporal_signal": [
                            {
                                "signal": "last three years",
                                "selected_id": "id_1",
                                "selected_name": "In past 3 years",
                            }
                        ],
                    },
                    {
                        "candidate": "prostate-specific antigen",
                        "temporal_signal": [
                            {
                                "signal": "recent",
                                "selected_id": "id_2",
                                "selected_name": "Recent",
                            }
                        ],
                    },
                ],
            }
        ],
    }
)


@pytest.fixture
def vocab() -> TemporalVocab:
    return TemporalVocab(
        {
            "In past 3 years": [
                {"cui": "C3843792", "formula": "REF_POINT - 3Y"},
            ],
            "Recent": [{"cui": "C0332185", "formula": None}],
        }
    )


@pytest.fixture
def pipeline(vocab):
    with (
        patch("app.mod_intent_extraction.google.auth") as mock_auth,
        patch("app.mod_intent_extraction.genai") as mock_genai,
    ):
        mock_auth.default.return_value = (Mock(), "test-project")
        mock_genai.Client.return_value = MagicMock()
        p = ContextualIntentPipeline(
            project="test-project",
            location="us-central1",
            model="test-model",
            logger=None,
            tracer=None,
            temporal_vocab=vocab,
        )
        yield p


def _mock_call_by_step(calls_out=None):
    """Return a _call_model side_effect keyed by step_name."""

    def _call(prompt, model_name=None, generation_config=None, **kwargs):
        step = kwargs.get("step_name", "llm_call")
        if calls_out is not None:
            calls_out.append(step)
        if step == "query_expansion_v2":
            return EXPANSION_JSON, USAGE
        if step == "intent_extraction_v2":
            return INTENT_JSON, USAGE
        if step == "representative_terms":
            return REP_TERMS_JSON, USAGE
        if step == "contextual_environment":
            return CONTEXT_JSON, USAGE
        if step == "query_expansion":
            return EXPANSION_JSON, USAGE
        if step == "intent_extraction":
            # v1 clinical schema
            return json.dumps(
                {
                    "is_clinical": True,
                    "reason": "",
                    "original_query": "elevated psa",
                    "expanded_query": "Elevated PSA",
                    "total_intents_detected": 1,
                    "intents": json.loads(INTENT_JSON)["intents"],
                }
            ), USAGE
        raise AssertionError(f"unexpected step_name: {step}")

    return _call


def _as_async_mock(sync_fn):
    async def _async_call(*args, **kwargs):
        return sync_fn(*args, **kwargs)

    return _async_call


class TestRunV2Pipeline:
    def test_v2_signals_off_no_step4(self, pipeline):
        steps = []
        with patch.object(
            pipeline,
            "_call_model_async",
            side_effect=_as_async_mock(_mock_call_by_step(steps)),
        ):
            result = pipeline.run_v2(
                "elevated psa in last 3 years",
                enable_retrieval_signals=False,
            )

        assert "contextual_environment" not in steps
        assert set(steps) == {
            "query_expansion_v2",
            "intent_extraction_v2",
            "representative_terms",
        }
        assert result["representative_terms"] == ["PSA", "prostate cancer"]
        assert len(result["intents"]) == 1
        assert "final_candidates" not in result["intents"][0]
        assert "retrieval_signals" not in result["intents"][0]
        assert "final_queries" in result["intents"][0]
        assert "is_clinical" not in result
        assert set(result["usage_metadata"].keys()) == {
            "query_expansion",
            "intent_extraction",
            "representative_terms",
        }

    def test_v2_signals_on_full_flow(self, pipeline):
        steps = []
        with patch.object(
            pipeline,
            "_call_model_async",
            side_effect=_as_async_mock(_mock_call_by_step(steps)),
        ):
            result = pipeline.run_v2(
                "elevated psa in last 3 years",
                enable_retrieval_signals=True,
            )

        assert "contextual_environment" in steps
        assert len(steps) == 4
        intent = result["intents"][0]
        assert "final_candidates" in intent
        assert len(intent["final_candidates"]) == 2
        assert intent["final_candidates"][0]["candidate_id"] == "fc_001"
        fc_signals = intent["final_candidates"][0]["retrieval_signals"]
        assert "authors" in fc_signals
        assert "author_roles" not in fc_signals
        assert fc_signals["temporal"][0]["codes"] == "C3843792"
        assert "retrieval_signals" in intent
        assert intent["final_queries"]
        assert "contextual_environment" in result["usage_metadata"]

    def test_v2_always_v2_prompts(self, pipeline):
        """Even with signals off, v2 prompt constants / step names are used."""
        prompts_by_step = {}

        def capture(prompt, model_name=None, generation_config=None, **kwargs):
            step = kwargs.get("step_name", "llm_call")
            prompts_by_step[step] = prompt
            return _mock_call_by_step()(
                prompt,
                model_name=model_name,
                generation_config=generation_config,
                **kwargs,
            )

        with (
            patch(
                "app.mod_intent_extraction.QUERY_EXPANSION_PROMPT_V2",
                "V2_EXPANSION_MARKER {query}",
            ),
            patch(
                "app.mod_intent_extraction.QUERY_EXPANSION_PROMPT",
                "V1_EXPANSION_MARKER {query}",
            ),
            patch(
                "app.mod_intent_extraction.INTENT_EXTRACTION_PROMPT_V2",
                "V2_INTENT_MARKER {expanded_query} {timestamp}",
            ),
            patch(
                "app.mod_intent_extraction.INTENT_EXTRACTION_PROMPT",
                "V1_INTENT_MARKER {original_query} {expanded_query} {timestamp}",
            ),
            patch(
                "app.mod_intent_extraction.REPRESENTATIVE_TERMS_PROMPT_V2",
                "V2_REP_MARKER {expanded_query}",
            ),
            patch.object(
                pipeline, "_call_model_async", side_effect=_as_async_mock(capture)
            ),
        ):
            pipeline.run_v2("elevated psa", enable_retrieval_signals=False)

        assert "query_expansion_v2" in prompts_by_step
        assert "intent_extraction_v2" in prompts_by_step
        assert "representative_terms" in prompts_by_step
        assert "V2_EXPANSION_MARKER" in prompts_by_step["query_expansion_v2"]
        assert "V1_EXPANSION_MARKER" not in prompts_by_step["query_expansion_v2"]
        assert "V2_INTENT_MARKER" in prompts_by_step["intent_extraction_v2"]
        assert "V1_INTENT_MARKER" not in prompts_by_step["intent_extraction_v2"]
        assert "V2_REP_MARKER" in prompts_by_step["representative_terms"]


class TestV1EndpointRegression:
    def test_v1_ignores_signals_flag(self, client, mock_pipeline):
        """v1 handler must call run(), not run_v2, even if flag is set."""
        resp = client.post(
            V1_NB,
            json={
                "texts": ["Patient with DM"],
                "enable_retrieval_signals": True,
            },
        )
        assert resp.status_code == 200
        mock_pipeline.run.assert_called_once()
        mock_pipeline.run_v2.assert_not_called()

    def test_v1_regression_unchanged(self, client, mock_pipeline):
        resp = client.post(
            V1_NB,
            json={"texts": ["Patient with DM"]},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == 1
        results = body["output"]["results"]
        assert len(results) == 1
        intent = results[0]["intents"][0]
        assert "final_queries" in intent
        assert "final_candidates" not in intent
        assert "retrieval_signals" not in intent
        assert "representative_terms" not in results[0]


class TestV2Endpoint:
    def test_v2_endpoint_signals_off(self, client, mock_pipeline):
        resp = client.post(
            V2_NB,
            json={
                "texts": ["Patient with DM"],
                "enable_retrieval_signals": False,
            },
        )
        assert resp.status_code == 200
        mock_pipeline.run_v2.assert_called_once()
        kwargs = mock_pipeline.run_v2.call_args.kwargs
        assert kwargs.get("enable_retrieval_signals") is False
        body = resp.json()
        assert body["output"]["results"][0]["representative_terms"] == [
            "diabetes mellitus"
        ]
        details_usage = body["details"]["usage_metadata"]
        assert "representative_terms" in details_usage
        assert "contextual_environment" not in details_usage

    def test_v2_endpoint_signals_on_passes_flag(self, client, mock_pipeline):
        mock_pipeline.run_v2.return_value = {
            "original_query": "Patient with DM",
            "expanded_query": "Patient with Diabetes Mellitus",
            "abbreviations_expanded": ["DM"],
            "representative_terms": ["diabetes mellitus"],
            "total_intents_detected": 1,
            "intents": [
                {
                    "intent_title": "Diabetes Mellitus",
                    "description": "desc",
                    "nature": "nature",
                    "sub_natures": [
                        {
                            "category_path": "A >> B",
                            "atomic_concepts": ["diabetes mellitus"],
                        }
                    ],
                    "final_queries": ["diabetes mellitus"],
                    "final_candidates": [
                        {
                            "candidate_id": "fc_001",
                            "intent_title": "Diabetes Mellitus",
                            "nature": "nature",
                            "sub_nature": "A >> B",
                            "candidate": "diabetes mellitus",
                            "retrieval_signals": {
                                "record_types": ["progress_note"],
                                "temporal": [
                                    {
                                        "time_window": "Recent",
                                        "codes": "C0332185",
                                        "formula": None,
                                    }
                                ],
                                "authors": ["primary_care_physician"],
                                "longitudinal_scope": ["follow_up"],
                                "content_signals": [],
                                "clinical_setting": ["outpatient"],
                            },
                        }
                    ],
                    "retrieval_signals": {
                        "record_types": ["progress_note"],
                        "temporal": [
                            {
                                "time_window": "Recent",
                                "codes": "C0332185",
                                "formula": None,
                            }
                        ],
                        "authors": ["primary_care_physician"],
                        "longitudinal_scope": ["follow_up"],
                        "content_signals": [],
                        "clinical_setting": ["outpatient"],
                    },
                }
            ],
            "timestamp": "2025-01-01T10:00:00Z",
            "processing_time_seconds": 2.0,
            "usage_metadata": {
                "query_expansion": USAGE,
                "intent_extraction": USAGE,
                "representative_terms": USAGE,
                "contextual_environment": USAGE,
            },
        }
        resp = client.post(
            V2_NB,
            json={
                "texts": ["Patient with DM"],
                "enable_retrieval_signals": True,
            },
        )
        assert resp.status_code == 200
        kwargs = mock_pipeline.run_v2.call_args.kwargs
        assert kwargs.get("enable_retrieval_signals") is True
        result = resp.json()["output"]["results"][0]
        assert "final_candidates" in result["intents"][0]
        assert (
            resp.json()["details"]["usage_metadata"].get("contextual_environment")
            is not None
        )


class TestLocationParameter:
    """Request-level `location` validation and resolution into pipeline calls."""

    def test_rejects_invalid_location(self, client, mock_pipeline):
        resp = client.post(
            V2_NB,
            json={"texts": ["Patient with DM"], "location": "asia-south1"},
        )
        assert resp.status_code == 422

    def test_rejects_invalid_model_name(self, client, mock_pipeline):
        resp = client.post(
            V2_NB,
            json={"texts": ["Patient with DM"], "model_name": "gemini-99"},
        )
        assert resp.status_code == 422
        assert "Invalid 'model_name'" in str(resp.json()["detail"])

    def test_defaults_to_configured_location_when_omitted(self, client, mock_pipeline):
        resp = client.post(
            V2_NB,
            json={"texts": ["Patient with DM"]},
        )
        assert resp.status_code == 200
        kwargs = mock_pipeline.run_v2.call_args.kwargs
        assert kwargs.get("location") == "us-central1"
        assert resp.json()["details"]["location"] == "us-central1"

    def test_empty_string_location_treated_as_not_provided(self, client, mock_pipeline):
        resp = client.post(
            V2_NB,
            json={"texts": ["Patient with DM"], "location": ""},
        )
        assert resp.status_code == 200
        assert resp.json()["details"]["location"] == "us-central1"

    def test_null_location_treated_as_not_provided(self, client, mock_pipeline):
        resp = client.post(
            V2_NB,
            json={"texts": ["Patient with DM"], "location": None},
        )
        assert resp.status_code == 200
        assert resp.json()["details"]["location"] == "us-central1"

    def test_v1_endpoint_also_resolves_location(self, client, mock_pipeline):
        resp = client.post(
            V1_NB,
            json={"texts": ["Patient with DM"], "location": "us"},
        )
        assert resp.status_code == 200
        kwargs = mock_pipeline.run.call_args.kwargs
        assert kwargs.get("location") == "us"


class TestResolveModelLocation:
    """Unit tests for the resolve_model_location precedence chain."""

    def test_model_family_map_wins_over_config_location(self):
        from app.utils.location import resolve_model_location

        _, loc = resolve_model_location(
            model="gemini-3-pro",
            location=None,
            config_map={"gemini-3": "us"},
            config_location="us-central1",
        )
        assert loc == "us"

    def test_config_location_used_when_no_map_match(self):
        from app.utils.location import resolve_model_location

        _, loc = resolve_model_location(
            model="gemini-2.5-flash",
            location=None,
            config_map={"gemini-3": "us"},
            config_location="us-central1",
        )
        assert loc == "us-central1"

    def test_fallback_used_when_nothing_else_set(self):
        from app.utils.location import DEFAULT_LOCATION, resolve_model_location

        _, loc = resolve_model_location(
            model="gemini-2.5-flash",
            location=None,
            config_map={},
            config_location=None,
        )
        assert loc == DEFAULT_LOCATION


class TestPipelineClientCaching:
    """The pipeline should build/cache one genai.Client per distinct location."""

    def test_get_client_caches_per_location(self, pipeline):
        client_a = pipeline._get_client("us-central1")
        pipeline._get_client("us")
        client_a_again = pipeline._get_client("us-central1")
        assert client_a is client_a_again
        assert pipeline._clients.keys() == {"us-central1", "us"}

    def test_get_client_defaults_to_pipeline_default_location(self, pipeline):
        default_client = pipeline._get_client(None)
        assert default_client is pipeline.client
