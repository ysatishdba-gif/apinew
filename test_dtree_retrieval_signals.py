"""Unit tests for POST /v1/retrieval-signals (no LLM calls).

Relies on tests/conftest.py mocking (aie_logging, vertexai, genai, vocab
loaders) exactly like the existing endpoint tests.
"""

from unittest.mock import patch

import pytest
from pydantic import ValidationError

from app import config
from app.mod_requests import RetrievalSignalsRequest
from app.utils.concept_vocab import ConceptVocab, normalize_facet_name
from app.utils.dtree_signals import (
    merge_hints,
    project_query_signals,
    project_record_types,
    project_tags,
    project_temporal,
)

DTREE_RS = f"{config.API_PREFIX}/v1/retrieval-signals"

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rt_vocab() -> ConceptVocab:
    return ConceptVocab(
        {
            "_comment": "ignored",
            "Radiology Report": [{"cui": "C0034571"}],
            "Clinical Note": [{"cui": "C1961136"}],
            "Consultation note": [{"cui": "C1144643"}],
            "Consultation note &#x7C; Hospital &#x7C; Document ontology": [
                {"cui": "C4319915"}
            ],
            "Consultation note &#x7C; Outpatient &#x7C; Document ontology": [
                {"cui": "C4319916"}
            ],
        }
    )


@pytest.fixture
def v2_result():
    """Shape of pipeline.run_v2(..., enable_retrieval_signals=True) after
    processing_time/usage_metadata are popped by the handler."""
    return {
        "original_query": "chest x-ray reports from the last year",
        "expanded_query": "chest radiograph reports from the last one year",
        "representative_terms": ["Chest X-ray"],
        "total_intents_detected": 1,
        "record_type_matches": [
            {
                "record_type": "radiology_report",
                "selected_names": ["Radiology Report"],
            },
            {
                "record_type": "clinical_note",
                "selected_names": ["Clinical Note"],
            },
            {
                "record_type": "unknown_doc",
                "selected_names": [],
            },
        ],
        "intents": [
            {
                "intent_title": "Chest X-ray reports",
                "description": "Radiology reports for chest x-rays",
                "nature": "[Diagnostic] / [Imaging]",
                "sub_natures": [
                    {
                        "category_path": "Imaging >> Radiology >> Chest",
                        "atomic_concepts": ["chest x-ray"],
                    }
                ],
                "final_queries": ["chest x-ray"],
                "final_candidates": [],
                "retrieval_signals": {
                    "record_types": [
                        "radiology_report",
                        "clinical_note",
                        "unknown_doc",
                    ],
                    "temporal": [
                        {
                            "time_window": "Last One Year",
                            "codes": "C4722677",
                            "formula": ["REF_POINT", "REF_POINT - 1Y"],
                        }
                    ],
                    "authors": ["radiologist"],
                    "longitudinal_scope": [],
                    "content_signals": [],
                    "clinical_setting": ["outpatient"],
                },
            }
        ],
    }


# ---------------------------------------------------------------------------
# ConceptVocab
# ---------------------------------------------------------------------------


class TestConceptVocab:
    def test_exact_and_folded_lookup(self, rt_vocab):
        name, codes = rt_vocab.lookup("Radiology Report")
        assert name == "Radiology Report" and codes[0].cui == "C0034571"
        name, codes = rt_vocab.lookup("radiology_report")
        assert name == "Radiology Report" and codes[0].cui == "C0034571"

    def test_miss_returns_none_and_empty(self, rt_vocab):
        assert rt_vocab.lookup("operative_report") == (None, [])

    def test_metadata_keys_ignored(self, rt_vocab):
        assert rt_vocab.lookup("_comment") == (None, [])
        assert len(rt_vocab) == 5

    def test_normalize_facet_name(self):
        assert normalize_facet_name("radiology_report") == "Radiology Report"
        assert normalize_facet_name("  clinical note ") == "Clinical Note"


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------


class TestProjection:
    def test_record_types_coded_and_ordered(self, v2_result, rt_vocab):
        rts = project_record_types(v2_result, rt_vocab)
        assert [r["name"] for r in rts] == [
            "Radiology Report",
            "Clinical Note",
            "Unknown Doc",  # vocab miss -> normalized name, empty coding
        ]
        assert rts[0]["coding"] == [{"code": "C0034571"}]
        assert rts[2]["coding"] == []

    def test_record_type_match_accepts_multiple_vocab_cuis_under_emitted_name(
        self, rt_vocab
    ):
        result = {
            "record_type_matches": [
                {
                    "record_type": "consultation_note",
                    "selected_names": [
                        "Consultation note",
                        "Consultation note &#x7C; Hospital &#x7C; Document ontology",
                        "Consultation note &#x7C; Outpatient &#x7C; Document ontology",
                    ],
                }
            ],
            "intents": [
                {
                    "retrieval_signals": {
                        "record_types": ["consultation_note"],
                    }
                }
            ],
        }

        rts = project_record_types(result, rt_vocab)

        assert rts == [
            {
                "name": "Consultation Note",
                "coding": [
                    {"code": "C1144643"},
                    {"code": "C4319915"},
                    {"code": "C4319916"},
                ],
            }
        ]

    def test_record_type_llm_match_collects_all_selected_codes(self):
        vocab = ConceptVocab(
            {
                "Discharge summary": [{"cui": "C0743221"}],
                "Discharge summary note | Hospital": [{"cui": "C5213341"}],
                "Discharge summary note | Outpatient": [{"cui": "C5213345"}],
                "Visit note": [{"cui": "C5453862"}],
            }
        )
        result = {
            "record_type_matches": [
                {
                    "record_type": "discharge_summary",
                    "selected_names": [
                        "Discharge summary",
                        "Discharge summary note | Hospital",
                        "Discharge summary note | Outpatient",
                    ],
                }
            ],
            "intents": [{"retrieval_signals": {"record_types": ["discharge_summary"]}}],
        }

        assert project_record_types(result, vocab) == [
            {
                "name": "Discharge Summary",
                "coding": [
                    {"code": "C0743221"},
                    {"code": "C5213341"},
                    {"code": "C5213345"},
                ],
            }
        ]

    def test_record_type_without_llm_matches_has_no_vocab_inferred_codes(self):
        vocab = ConceptVocab(
            {
                "After visit summary": [{"cui": "C_AFTER_VISIT"}],
                "Discharge summary": [{"cui": "C0743221"}],
                "Summary (document)": [{"cui": "C1706244"}],
            }
        )
        result = {
            "intents": [
                {"retrieval_signals": {"record_types": ["after_visit_summary"]}}
            ]
        }

        assert project_record_types(result, vocab) == [
            {
                "name": "After Visit Summary",
                "coding": [],
            }
        ]

    def test_temporal_first_resolved_entry(self, v2_result):
        t = project_temporal(v2_result)
        assert t == {
            "name": "Last One Year",
            "formula": ["REF_POINT", "REF_POINT - 1Y"],
            "coding": [{"system": "UMLS", "code": "C4722677"}],
        }

    def test_temporal_falls_back_when_no_signals(self):
        assert project_temporal({"intents": []}) == {
            "name": "Recent",
            "formula": ["REF_POINT", "REF_POINT"],
            "coding": [{"system": "UMLS", "code": "C0332185"}],
        }

    def test_tags_from_representative_terms(self, v2_result):
        tags = project_tags(v2_result, tag_vocab=None)
        assert tags[0]["name"] == "Chest X-ray"
        assert tags[0]["coding"] == []
        assert "imaging" in tags[0]["topics"]
        assert "radiology" in tags[0]["topics"]

    def test_full_projection_with_hints(self, v2_result, rt_vocab):
        block = project_query_signals(
            query_id="q1",
            text="chest x-ray reports from the last year",
            v2_result=v2_result,
            record_type_vocab=rt_vocab,
            hint_record_types=[
                {"name": "Radiology Report", "coding": [{"code": "CLIENT"}]},
                {"name": "Nursing Note", "coding": []},
            ],
            hint_temporal={
                "name": "Last 6 Months",
                "formula": "REF_POINT - 6M",
                "coding": [],
            },
        )
        assert block["id"] == "q1"
        # Hint order wins and dedupes derived duplicate by name; client coding preserved.
        names = [r["name"] for r in block["record_types"]]
        assert names[:2] == ["Radiology Report", "Nursing Note"]
        assert names.count("Radiology Report") == 1
        assert block["record_types"][0]["coding"][0]["code"] == "CLIENT"
        # Explicit temporal hint overrides derived window.
        assert block["temporal"]["name"] == "Last 6 Months"

    def test_merge_hints_noop_without_hints(self, v2_result, rt_vocab):
        derived = project_query_signals("q1", "t", v2_result, rt_vocab)
        merged = merge_hints(dict(derived), None, None, None)
        assert merged == derived


# ---------------------------------------------------------------------------
# Request model validation
# ---------------------------------------------------------------------------


class TestRequestModel:
    def test_valid_minimal(self):
        req = RetrievalSignalsRequest(queries=[{"text": ["chest x-ray"]}])
        assert req.queries[0].text == ["chest x-ray"]

    def test_rejects_scalar_text(self):
        with pytest.raises(ValidationError):
            RetrievalSignalsRequest(queries=[{"text": "chest x-ray"}])

    def test_rejects_empty_queries(self):
        with pytest.raises(ValidationError):
            RetrievalSignalsRequest(queries=[])

    def test_rejects_missing_text(self):
        with pytest.raises(ValidationError):
            RetrievalSignalsRequest(queries=[{}])

    def test_accepts_structured_hints(self):
        req = RetrievalSignalsRequest(
            queries=[
                {
                    "text": ["chest x-ray reports"],
                    "record_types": [
                        {"name": "Radiology Report", "coding": [{"code": "C0034571"}]}
                    ],
                    "temporal": {"name": "Last One Year", "formula": "REF_POINT - 1Y"},
                    "tags": [{"name": "Chest X-ray", "topics": ["imaging"]}],
                }
            ]
        )
        assert req.queries[0].record_types[0].coding == [{"code": "C0034571"}]
        assert req.queries[0].temporal.formula == ["REF_POINT - 1Y"]


# ---------------------------------------------------------------------------
# Endpoint (FastAPI TestClient; pipeline mocked)
# ---------------------------------------------------------------------------


async def _mock_v2(text: str, **kwargs):
    return {
        "original_query": text,
        "expanded_query": text,
        "representative_terms": ["Chest X-ray"],
        "total_intents_detected": 1,
        "intents": [
            {
                "intent_title": "Chest X-ray reports",
                "description": "d",
                "nature": "[Diagnostic] / [Imaging]",
                "sub_natures": [
                    {
                        "category_path": "Imaging >> Chest",
                        "atomic_concepts": ["chest x-ray"],
                    }
                ],
                "final_queries": ["chest x-ray"],
                "retrieval_signals": {
                    "record_types": ["radiology_report"],
                    "temporal": [
                        {
                            "time_window": "Last One Year",
                            "codes": "C4722677",
                            "formula": "REF_POINT - 1Y",
                        }
                    ],
                    "authors": [],
                    "longitudinal_scope": [],
                    "content_signals": [],
                    "clinical_setting": [],
                },
            }
        ],
        "timestamp": "2025-01-01T10:00:00Z",
        "processing_time_seconds": 1.0,
        "usage_metadata": {
            "query_expansion": {
                "prompt_token_count": 1,
                "candidates_token_count": 1,
                "total_token_count": 2,
                "thinking_token_count": 0,
            },
            "intent_extraction": {
                "prompt_token_count": 1,
                "candidates_token_count": 1,
                "total_token_count": 2,
                "thinking_token_count": 0,
            },
            "representative_terms": {
                "prompt_token_count": 1,
                "candidates_token_count": 1,
                "total_token_count": 2,
                "thinking_token_count": 0,
            },
            "contextual_environment": {
                "prompt_token_count": 1,
                "candidates_token_count": 1,
                "total_token_count": 2,
                "thinking_token_count": 0,
            },
        },
    }


class TestEndpoint:
    def test_200_single_query(self, client):
        with patch("app.main.pipeline.run_v2_async", side_effect=_mock_v2) as m:
            resp = client.post(
                DTREE_RS,
                json={
                    "queries": [
                        {"id": "q1", "text": ["chest x-ray reports from the last year"]}
                    ]
                },
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == 1
        q = body["output"]["queries"][0]
        assert q["id"] == "q1"
        assert q["temporal"]["coding"] == [{"system": "UMLS", "code": "C4722677"}]
        assert q["tags"][0]["name"] == "Chest X-ray"
        # enable_retrieval_signals must be forced on
        assert m.call_args.kwargs["enable_retrieval_signals"] is True

    def test_200_multiple_queries_processed_independently(self, client):
        with patch("app.main.pipeline.run_v2_async", side_effect=_mock_v2):
            resp = client.post(
                DTREE_RS,
                json={
                    "queries": [
                        {"id": "q1", "text": ["chest x-ray reports"]},
                        {"id": "q2", "text": ["diabetes labs last 3 months"]},
                    ]
                },
            )
        assert resp.status_code == 200
        ids = [q["id"] for q in resp.json()["output"]["queries"]]
        assert ids == ["q1", "q2"]
        assert resp.json()["output"]["total_queries"] == 2

    def test_llm_error_maps_like_other_endpoints(self, client):
        """LLMError -> handle_llm_exception -> HTTPException with the existing
        detail shape {error, message, details}, same as /v1 and /v2."""
        from app.exceptions import LLMRateLimitError, LLMServiceError, LLMTimeoutError

        def _make_raise_llm(error):
            async def _raise_llm(*_args, **_kwargs):
                raise error

            return _raise_llm

        for exc, code, err_type in [
            (LLMServiceError("boom"), 503, "service_unavailable"),
            (LLMTimeoutError("slow"), 504, "llm_timeout"),
            (LLMRateLimitError("busy"), 429, "rate_limit_exceeded"),
        ]:
            with patch(
                "app.main.pipeline.run_v2_async", side_effect=_make_raise_llm(exc)
            ):
                resp = client.post(
                    DTREE_RS,
                    json={"queries": [{"id": "q1", "text": ["chest x-ray"]}]},
                )
            assert resp.status_code == code
            detail = resp.json()["detail"]
            assert detail["error"] == err_type
            assert "message" in detail and "details" in detail

    def test_422_empty_queries(self, client):
        resp = client.post(DTREE_RS, json={"queries": []})
        assert resp.status_code == 422  # Pydantic min_length on queries

    def test_200_ids_are_generated(self, client):
        with patch("app.main.pipeline.run_v2_async", side_effect=_mock_v2):
            resp = client.post(
                DTREE_RS,
                json={"queries": [{"text": ["a"]}, {"text": ["b"]}]},
            )
        assert resp.status_code == 200
        assert [q["id"] for q in resp.json()["output"]["queries"]] == ["q1", "q2"]

    def test_422_whitespace_text_existing_detail_shape(self, client):
        resp = client.post(
            DTREE_RS,
            json={"queries": [{"id": "q1", "text": ["   "]}]},
        )
        assert resp.status_code == 422
        detail = resp.json()["detail"]
        assert detail["error"] == "validation_error"
        assert detail["query_index"] == 0

    def test_422_scalar_text_is_rejected(self, client):
        with patch("app.main.pipeline.run_v2_async", side_effect=_mock_v2):
            resp = client.post(
                DTREE_RS,
                json={"queries": [{"text": "chest x-ray reports, diabetes labs"}]},
            )
        assert resp.status_code == 422

    def test_422_invalid_model_name(self, client):
        resp = client.post(
            DTREE_RS,
            json={
                "queries": [{"text": ["chest x-ray"]}],
                "model_name": "gemini-99",
            },
        )
        assert resp.status_code == 422
        assert "Invalid 'model_name'" in str(resp.json()["detail"])

    def test_200_list_text_preserves_commas_within_each_query(self, client):
        with patch("app.main.pipeline.run_v2_async", side_effect=_mock_v2):
            resp = client.post(
                DTREE_RS,
                json={
                    "queries": [
                        {
                            "text": [
                                "chest x-ray reports from the last year, including follow-up",
                                "tell me a joke",
                            ]
                        }
                    ]
                },
            )
        assert resp.status_code == 200
        queries = resp.json()["output"]["queries"]
        assert [q["id"] for q in queries] == ["q1", "q2"]
        assert [q["text"] for q in queries] == [
            "chest x-ray reports from the last year, including follow-up",
            "tell me a joke",
        ]

    def test_hint_temporal_overrides(self, client):
        with patch("app.main.pipeline.run_v2_async", side_effect=_mock_v2):
            resp = client.post(
                DTREE_RS,
                json={
                    "queries": [
                        {
                            "id": "q1",
                            "text": ["chest x-ray reports"],
                            "temporal": {
                                "name": "Last 6 Months",
                                "formula": "REF_POINT - 6M",
                            },
                        }
                    ]
                },
            )
        assert (
            resp.json()["output"]["queries"][0]["temporal"]["name"] == "Last 6 Months"
        )
