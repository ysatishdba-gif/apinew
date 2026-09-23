"""Unit tests for POST /v2/retrieval-signals (no LLM, no network).

Covers the two /v2 additions on top of /v1:
  1. Action Finder gate (app.action_finder + app.utils.action_gate)
  2. Document-cluster record-type coding (app.utils.context_lonic_document_cluster
     + dtree_signals.project_record_types_cluster)

Relies on tests/conftest.py mocking (aie_logging, vertexai, genai, vocab
loaders) exactly like the /v1 endpoint tests.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from app import config
from app.action_finder import ActionFinder, ActionFinderError, normalize_output
from app.exceptions import ClusterServiceError
from app.mod_requests import RetrievalSignalsRequest, RetrievalSignalsRequestV2
from app.utils import context_lonic_document_cluster as cluster
from app.utils.action_gate import (
    ACTION_TYPES,
    GateDecision,
    build_skip_block,
    decide,
    evaluate_gate,
    evaluate_gate_async,
    invalid_actions,
    normalize_action,
    normalize_actions,
)
from app.utils.dtree_signals import (
    collect_record_type_names,
    project_record_types_cluster,
)

RS_V1 = f"{config.API_PREFIX}/v1/retrieval-signals"
RS_V2 = f"{config.API_PREFIX}/v2/retrieval-signals"
RS_V2_ACTIONS = f"{config.API_PREFIX}/v2/retrieval-signals/action-types"


# ---------------------------------------------------------------------------
# Shared fakes
# ---------------------------------------------------------------------------


def _usage(n: int = 1) -> dict:
    return {
        "prompt_token_count": n,
        "candidates_token_count": n,
        "total_token_count": 2 * n,
        "thinking_token_count": 0,
    }


async def _mock_v2(text: str, **kwargs):
    """pipeline.run_v2_async(..., enable_retrieval_signals=True) shape."""
    return {
        "original_query": text,
        "expanded_query": text,
        "representative_terms": ["Chest X-ray"],
        "total_intents_detected": 1,
        "record_type_matches": [],
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
                    "record_types": ["radiology_report", "clinical_note"],
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
            "query_expansion": _usage(),
            "intent_extraction": _usage(),
            "representative_terms": _usage(),
            "contextual_environment": _usage(),
        },
    }


def _fake_resolver(name: str, context: str | None):
    """Stand-in for cluster.resolve_document_type_cuis."""
    table = {
        "Radiology Report": [
            {"name": "Radiology studies (set)", "coding": [{"code": "C0034571"}]},
            {"name": "Diagnostic imaging study", "coding": [{"code": "C0011923"}]},
        ],
        "Clinical Note": [{"name": "Clinical note", "coding": [{"code": "C1961136"}]}],
    }
    return table.get(name, [])


def _finder(actions, is_processable=None):
    """Async Action Finder stub returning a fixed classification."""

    async def _fn(text, **kwargs):
        return {
            "actions": list(actions),
            "is_processable": (
                is_processable
                if is_processable is not None
                else "information_retrieval" in actions
            ),
            "usage_metadata": _usage(5),
        }

    return AsyncMock(side_effect=_fn)


@pytest.fixture
def cluster_configured(monkeypatch):
    """Point the cluster module at a configured (fake) backend."""
    monkeypatch.setattr(
        config, "CLUSTER_SELECTION_URL", "https://cluster.example/select"
    )
    monkeypatch.setattr(config, "CLUSTER_BQ_DATASET", "test_dataset")
    cluster.reset_auth_cache()
    yield
    cluster.reset_auth_cache()


@pytest.fixture
def cluster_unconfigured(monkeypatch):
    monkeypatch.setattr(config, "CLUSTER_SELECTION_URL", "")
    monkeypatch.setattr(config, "CLUSTER_BQ_DATASET", "")


@pytest.fixture
def resolver_patch():
    """Route record-type coding through the fake resolver (no HTTP / BigQuery)."""
    original = cluster.resolve_record_type_cuis
    with patch(
        "app.utils.context_lonic_document_cluster.resolve_record_type_cuis",
        side_effect=lambda rts, tags, resolver=None: original(
            rts, tags, _fake_resolver
        ),
    ) as m:
        yield m


# ---------------------------------------------------------------------------
# action_gate
# ---------------------------------------------------------------------------


class TestActionGate:
    def test_normalize_action_variants(self):
        assert normalize_action("Information Retrieval") == "information_retrieval"
        assert normalize_action("information-retrieval") == "information_retrieval"
        assert normalize_action("  ADMIN_RESPONSE ") == "admin_response"
        assert normalize_action(None) == ""
        assert normalize_action(42) == ""

    def test_normalize_actions_dedupes_and_drops_blanks(self):
        assert normalize_actions(["a", "A", "", None, "b"]) == ["a", "b"]
        assert normalize_actions(None) == []

    def test_invalid_actions_reports_as_sent(self):
        assert invalid_actions(
            ["information_retrieval", "informational_retrieval"]
        ) == ["informational_retrieval"]
        assert invalid_actions(["Information Retrieval"]) == []

    def test_decide_overlap(self):
        assert decide(
            ["information_retrieval"], ["admin_response", "Information Retrieval"]
        )
        assert not decide(["information_retrieval"], ["admin_response"])

    def test_evaluate_gate_matches_only_with_information_retrieval(self):
        d = evaluate_gate(
            "q",
            ["information_retrieval"],
            lambda t: {"actions": ["information_retrieval"], "is_processable": True},
        )
        assert d.matched and d.skip_reason is None
        assert d.identified_actions == ["information_retrieval"]

        d = evaluate_gate(
            "q",
            ["admin_response"],
            lambda t: {"actions": ["admin_response"], "is_processable": False},
        )
        # Overlaps the expected list but is not retrieval-serviceable.
        assert not d.matched and d.skip_reason == "action_mismatch"

    def test_evaluate_gate_fails_open_on_finder_error(self):
        def _boom(t):
            raise RuntimeError("finder down")

        d = evaluate_gate("q", ["information_retrieval"], _boom)
        assert d.matched is True
        assert d.failed_open is True
        assert d.error == "RuntimeError: finder down"
        assert d.identified_actions == []

    def test_evaluate_gate_async_carries_usage_metadata(self):
        async def _fn(t):
            return {
                "actions": ["information_retrieval"],
                "is_processable": True,
                "usage_metadata": _usage(3),
            }

        d = asyncio.run(evaluate_gate_async("q", ["information_retrieval"], _fn))
        assert d.matched
        assert d.usage_metadata == _usage(3)

    def test_evaluate_gate_async_fails_open(self):
        async def _fn(t):
            raise ActionFinderError("no actions")

        d = asyncio.run(evaluate_gate_async("q", ["information_retrieval"], _fn))
        assert d.matched and d.failed_open and "ActionFinderError" in d.error

    def test_skip_block_contract(self):
        decision = GateDecision(
            matched=False,
            expected_actions=["information_retrieval"],
            identified_actions=["admin_response", "conversational_response"],
            is_processable=False,
        )
        block = build_skip_block("q1", "book me an appointment", decision)
        assert block == {
            "id": "q1",
            "text": "book me an appointment",
            "skipped": True,
            "skip_reason": ["admin_response", "conversational_response"],
            "skip_reason_detail": (
                "Action Finder classified this query as "
                "admin_response, conversational_response"
            ),
            "record_types": [],
            # /v1 shape: temporal is a single object or null, never a list.
            "temporal": None,
            "tags": [],
        }


# ---------------------------------------------------------------------------
# action_finder
# ---------------------------------------------------------------------------


class TestActionFinder:
    def test_normalize_output_parses_json_and_enforces_invariants(self):
        out = normalize_output('{"actions": ["Information Retrieval", "bogus"]}')
        assert out == {"actions": ["information_retrieval"], "is_processable": True}

        out = normalize_output({"actions": ["admin_response", "out_of_scope"]})
        assert out == {"actions": ["out_of_scope"], "is_processable": False}

    def test_normalize_output_rejects_unusable(self):
        with pytest.raises(ActionFinderError):
            normalize_output('{"actions": []}')
        with pytest.raises(ActionFinderError):
            normalize_output("not json")
        with pytest.raises(ActionFinderError):
            normalize_output(["information_retrieval"])

    def test_find_actions_uses_pipeline_model_client(self, monkeypatch):
        pipeline = MagicMock()
        pipeline.model_name = "gemini-2.5-flash"
        pipeline._call_model_async = AsyncMock(
            return_value=(
                '{"actions": ["information_retrieval"], "is_processable": true}',
                _usage(7),
            )
        )
        monkeypatch.setattr(config, "ACTION_FINDER_MODEL", "")
        finder = ActionFinder(pipeline)

        out = asyncio.run(
            finder.find_actions_async(
                "  last A1C  ", model_name="gemini-3.5-flash", location="us"
            )
        )
        assert out == {
            "actions": ["information_retrieval"],
            "is_processable": True,
            "usage_metadata": _usage(7),
        }
        kwargs = pipeline._call_model_async.call_args.kwargs
        assert kwargs["model_name"] == "gemini-3.5-flash"
        assert kwargs["location"] == "us"
        assert kwargs["step_name"] == "action_finder"
        assert kwargs["generation_config"] is None
        assert kwargs["temperature"] == 0.0
        assert "last A1C" in pipeline._call_model_async.call_args.args[0]

    def test_find_actions_model_override(self, monkeypatch):
        pipeline = MagicMock()
        pipeline.model_name = "gemini-2.5-flash"
        pipeline._call_model_async = AsyncMock(
            return_value=('{"actions": ["admin_response"]}', _usage())
        )
        monkeypatch.setattr(config, "ACTION_FINDER_MODEL", "gemini-2.5-flash-lite")
        finder = ActionFinder(pipeline)
        out = asyncio.run(finder.find_actions_async("q", model_name="gemini-3.5-flash"))
        assert out["actions"] == ["admin_response"] and out["is_processable"] is False
        assert (
            pipeline._call_model_async.call_args.kwargs["model_name"]
            == "gemini-2.5-flash-lite"
        )

    def test_find_actions_rejects_empty_query(self):
        finder = ActionFinder(MagicMock())
        with pytest.raises(ActionFinderError):
            asyncio.run(finder.find_actions_async("   "))


# ---------------------------------------------------------------------------
# context_lonic_document_cluster
# ---------------------------------------------------------------------------


class TestDocumentCluster:
    def test_extract_member_names_recursive_sorted(self):
        payload = {
            "results": [
                {"member_name": "Radiology studies (set)"},
                {
                    "nested": {
                        "member_name": ["Chest X-ray report", " ", "Chest X-ray report"]
                    }
                },
            ],
            "member_name": "Diagnostic imaging study",
        }
        assert cluster.extract_member_names(payload) == [
            "Chest X-ray report",
            "Diagnostic imaging study",
            "Radiology studies (set)",
        ]

    def test_missing_configuration_raises_cluster_error(self, cluster_unconfigured):
        assert cluster.is_configured() is False
        assert cluster.missing_configuration() == [
            "CLUSTER_SELECTION_URL",
            "CLUSTER_BQ_DATASET",
        ]
        with pytest.raises(ClusterServiceError) as exc:
            cluster.call_cluster_selection("Radiology Report")
        assert exc.value.status_code == 503
        assert exc.value.error_type == "cluster_service_unavailable"

    def test_no_identity_token_returns_empty_without_calling_service(
        self, cluster_configured
    ):
        with (
            patch.object(cluster, "_fetch_identity_token", return_value=None),
            patch("app.utils.context_lonic_document_cluster.requests.post") as post,
        ):
            assert cluster.call_cluster_selection("Radiology Report") == {}
        post.assert_not_called()

    def test_cluster_selection_payload_and_context(
        self, cluster_configured, monkeypatch
    ):
        monkeypatch.setattr(config, "CLUSTER_SET", "loinc_document_v001")
        monkeypatch.setattr(config, "CLUSTER_TOP_K", 10)
        monkeypatch.setattr(config, "CLUSTER_TOP_P", 20)
        monkeypatch.setattr(config, "CLUSTER_COMBINED", True)
        response = MagicMock(status_code=200)
        response.json.return_value = {
            "results": [{"member_name": "Chest X-ray report"}]
        }
        with (
            patch.object(cluster, "_fetch_identity_token", return_value="tok"),
            patch(
                "app.utils.context_lonic_document_cluster.requests.post",
                return_value=response,
            ) as post,
        ):
            docs = cluster.get_probable_documents(
                "Radiology Report", "chest X-ray, imaging"
            )
        assert docs == ["Chest X-ray report"]
        kwargs = post.call_args.kwargs
        assert kwargs["json"] == {
            "cluster_set": "loinc_document_v001",
            "text_list": ["Radiology Report"],
            "context_list": ["chest X-ray, imaging"],
            "top_k": 10,
            "top_p": 20,
            "combined": True,
        }
        assert kwargs["headers"]["Authorization"] == "Bearer tok"

        # Without tag context the record-type name is its own context.
        with (
            patch.object(cluster, "_fetch_identity_token", return_value="tok"),
            patch(
                "app.utils.context_lonic_document_cluster.requests.post",
                return_value=response,
            ) as post,
        ):
            cluster.get_probable_documents("Radiology Report", "  ")
        assert post.call_args.kwargs["json"]["context_list"] == ["Radiology Report"]

    def test_rejected_token_is_refreshed_once_then_empty(self, cluster_configured):
        denied = MagicMock(status_code=401, text="denied")
        ok = MagicMock(status_code=200)
        ok.json.return_value = {"member_name": "Doc A"}
        with (
            patch.object(
                cluster, "_fetch_identity_token", side_effect=["old", "new"]
            ) as tok,
            patch(
                "app.utils.context_lonic_document_cluster.requests.post",
                side_effect=[denied, ok],
            ) as post,
        ):
            assert cluster.call_cluster_selection("x") == {"member_name": "Doc A"}
        assert tok.call_count == 2
        assert post.call_args_list[1].kwargs["headers"]["Authorization"] == "Bearer new"

        cluster.reset_auth_cache()
        with (
            patch.object(cluster, "_fetch_identity_token", return_value="tok"),
            patch(
                "app.utils.context_lonic_document_cluster.requests.post",
                return_value=denied,
            ),
        ):
            assert cluster.call_cluster_selection("x") == {}

    def test_service_error_raises_cluster_error(self, cluster_configured):
        bad = MagicMock(status_code=500, text="boom")
        with (
            patch.object(cluster, "_fetch_identity_token", return_value="tok"),
            patch(
                "app.utils.context_lonic_document_cluster.requests.post",
                return_value=bad,
            ),
            pytest.raises(ClusterServiceError) as exc,
        ):
            cluster.call_cluster_selection("x")
        assert exc.value.details["status_code"] == 500

        import requests as _requests

        with (
            patch.object(cluster, "_fetch_identity_token", return_value="tok"),
            patch(
                "app.utils.context_lonic_document_cluster.requests.post",
                side_effect=_requests.ConnectionError("down"),
            ),
            pytest.raises(ClusterServiceError),
        ):
            cluster.call_cluster_selection("x")

    def test_resolve_document_type_cuis_dedupes_by_cui(self, cluster_configured):
        rows = [
            {
                "document_class_cui": "C1",
                "document_class": "Class A",
                "possible_document_name": "d1",
            },
            {
                "document_class_cui": "C1",
                "document_class": "Class A",
                "possible_document_name": "d2",
            },
            {
                "document_class_cui": "C2",
                "document_class": "Class B",
                "possible_document_name": "d1",
            },
            {
                "document_class_cui": "",
                "document_class": "Missing",
                "possible_document_name": "d1",
            },
        ]
        with (
            patch.object(cluster, "get_probable_documents", return_value=["d1", "d2"]),
            patch.object(cluster, "get_document_classes", return_value=rows) as bq,
        ):
            out = cluster.resolve_document_type_cuis("Radiology Report", "ctx")
        assert out == [
            {"name": "Class A", "coding": [{"code": "C1"}]},
            {"name": "Class B", "coding": [{"code": "C2"}]},
        ]
        bq.assert_called_once_with(["d1", "d2"])

        with patch.object(cluster, "get_probable_documents", return_value=[]):
            assert cluster.resolve_document_type_cuis("Unknown") == []

    def test_resolve_record_type_cuis_uses_names_and_tag_context(self):
        calls = []

        def resolver(name, context):
            calls.append((name, context))
            return _fake_resolver(name, context)

        out = cluster.resolve_record_type_cuis(
            [
                {"name": "Radiology Report", "coding": []},
                "radiology report",  # duplicate (case-insensitive) dropped
                {"record_type": "Clinical Note"},
                {"name": "Unknown Doc"},
                42,
            ],
            tags=[
                {"name": "chest X-ray"},
                "imaging",
                {"name": "chest X-ray"},
                {"x": 1},
            ],
            resolver=resolver,
        )
        assert calls == [
            ("Radiology Report", "chest X-ray, imaging"),
            ("Clinical Note", "chest X-ray, imaging"),
            ("Unknown Doc", "chest X-ray, imaging"),
        ]
        assert out == [
            {
                "name": "Radiology Report",
                "coding": [{"code": "C0034571"}, {"code": "C0011923"}],
            },
            {"name": "Clinical Note", "coding": [{"code": "C1961136"}]},
            {"name": "Unknown Doc", "coding": []},
        ]
        assert cluster.resolve_record_type_cuis([], tags=["x"], resolver=resolver) == []

    def test_resolve_record_type_cuis_default_path_batches_bigquery(
        self, cluster_configured
    ):
        """Fan-out cluster selection per record type, one BigQuery lookup,
        per-record-type coding identical to resolving each name alone."""
        probable = {
            "Radiology Report": ["Chest X-ray report", "CT report"],
            "Clinical Note": ["Progress note"],
        }
        rows = [
            {
                "document_class_cui": "C10",
                "document_class": "Imaging",
                "possible_document_name": "Chest X-ray report",
            },
            {
                "document_class_cui": "C10",
                "document_class": "Imaging",
                "possible_document_name": "CT report",
            },
            {
                "document_class_cui": "C20",
                "document_class": "Note",
                "possible_document_name": "Progress note",
            },
            {
                "document_class_cui": "C30",
                "document_class": "CT",
                "possible_document_name": "CT report",
            },
        ]
        with (
            patch.object(
                cluster,
                "get_probable_documents",
                side_effect=lambda n, c: probable.get(n, []),
            ),
            patch.object(cluster, "get_document_classes", return_value=rows) as bq,
        ):
            out = cluster.resolve_record_type_cuis(
                ["Radiology Report", "Clinical Note", "Nothing"], tags=["ctx"]
            )
        assert bq.call_count == 1
        assert bq.call_args.args[0] == [
            "Chest X-ray report",
            "CT report",
            "Progress note",
        ]
        assert out == [
            {"name": "Radiology Report", "coding": [{"code": "C10"}, {"code": "C30"}]},
            {"name": "Clinical Note", "coding": [{"code": "C20"}]},
            {"name": "Nothing", "coding": []},
        ]


# ---------------------------------------------------------------------------
# dtree_signals cluster projection
# ---------------------------------------------------------------------------


class TestClusterProjection:
    def test_collect_record_type_names_display_and_dedupe(self):
        v2 = asyncio.run(_mock_v2("q"))
        v2["intents"][0]["retrieval_signals"]["record_types"] += [
            "Radiology Report",
            "  ",
        ]
        assert collect_record_type_names(v2) == ["Radiology Report", "Clinical Note"]
        assert collect_record_type_names({"intents": []}) == []

    def test_project_record_types_cluster_resolves_each_type_with_tag_context(self):
        v2 = asyncio.run(_mock_v2("q"))
        tags = [{"name": "Chest X-ray", "coding": [], "topics": []}]
        calls = []

        def resolver(name, context):
            calls.append((name, context))
            return _fake_resolver(name, context)

        out = project_record_types_cluster(v2, tags, resolver)
        assert calls == [
            ("Radiology Report", "Chest X-ray"),
            ("Clinical Note", "Chest X-ray"),
        ]
        assert out == [
            {
                "name": "Radiology Report",
                "coding": [{"code": "C0034571"}, {"code": "C0011923"}],
            },
            {"name": "Clinical Note", "coding": [{"code": "C1961136"}]},
        ]
        assert project_record_types_cluster({"intents": []}, tags, resolver) == []


# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------


class TestRequestModelV2:
    def test_v2_accepts_actions_and_defaults(self):
        req = RetrievalSignalsRequestV2(queries=[{"text": ["a"]}])
        assert req.actions is None and req.enable_action_finder is True

        req = RetrievalSignalsRequestV2(
            queries=[{"text": ["a"]}],
            actions=["information_retrieval"],
            enable_action_finder=False,
        )
        assert req.actions == ["information_retrieval"]
        assert req.enable_action_finder is False

    def test_v2_inherits_v1_validation(self):
        with pytest.raises(ValidationError):
            RetrievalSignalsRequestV2(queries=[{"text": ["a"]}], model_name="gemini-99")
        with pytest.raises(ValidationError):
            RetrievalSignalsRequestV2(
                queries=[{"text": ["a"]}], location="europe-west4"
            )
        with pytest.raises(ValidationError):
            RetrievalSignalsRequestV2(queries=[])

    def test_v2_schema_exposes_action_enum(self):
        schema = RetrievalSignalsRequestV2.model_json_schema()
        actions = schema["properties"]["actions"]
        array_variant = next(v for v in actions["anyOf"] if v.get("type") == "array")
        assert array_variant["items"]["enum"] == sorted(ACTION_TYPES)
        assert actions["example"] == ["information_retrieval"]

    def test_v1_model_unchanged(self):
        assert "actions" not in RetrievalSignalsRequest.model_fields
        assert "enable_action_finder" not in RetrievalSignalsRequest.model_fields


# ---------------------------------------------------------------------------
# Endpoint (FastAPI TestClient; pipeline, finder and cluster mocked)
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("cluster_configured", "resolver_patch")
class TestEndpointV2:
    def test_action_types_endpoint(self, client):
        resp = client.get(RS_V2_ACTIONS)
        assert resp.status_code == 200
        assert resp.json() == {
            "enable_action_finder": True,
            "actions": sorted(ACTION_TYPES),
        }

    def test_200_ungated_same_envelope_as_v1_with_cluster_record_types(self, client):
        with (
            patch("app.main.pipeline.run_v2_async", side_effect=_mock_v2) as run,
            patch("app.main.action_finder.find_actions_async") as finder,
        ):
            resp = client.post(
                RS_V2,
                json={"queries": [{"id": "q1", "text": ["chest x-ray reports"]}]},
            )
        assert resp.status_code == 200
        finder.assert_not_called()
        body = resp.json()
        assert body["status"] == 1
        assert body["service"] == config.SERVICE_ID
        assert set(body["details"]) == {
            "timing",
            "usage_metadata",
            "version",
            "timestamp",
            "source",
            "model",
            "location",
        }
        assert list(body["details"]["usage_metadata"]) == [
            "action_finder",
            "query_expansion",
            "intent_extraction",
            "representative_terms",
            "contextual_environment",
        ]
        assert (
            body["details"]["usage_metadata"]["action_finder"]["total_token_count"] == 0
        )

        q = body["output"]["queries"][0]
        assert q["id"] == "q1" and "skipped" not in q
        # Record types: names from the pipeline, codes from the document cluster.
        assert q["record_types"] == [
            {
                "name": "Radiology Report",
                "coding": [{"code": "C0034571"}, {"code": "C0011923"}],
            },
            {"name": "Clinical Note", "coding": [{"code": "C1961136"}]},
        ]
        # Temporal and tags exactly as /v1 (single temporal object).
        assert q["temporal"]["name"] == "Last One Year"
        assert q["temporal"]["coding"] == [{"system": "UMLS", "code": "C4722677"}]
        assert q["tags"][0]["name"] == "Chest X-ray"

        # The pipeline runs without the JSON record-type vocabulary block.
        kwargs = run.call_args.kwargs
        assert kwargs["enable_retrieval_signals"] is True
        assert kwargs["include_record_type_matching"] is False

    def test_matching_action_runs_pipeline_and_counts_finder_usage(self, client):
        finder = _finder(["information_retrieval"])
        with (
            patch("app.main.pipeline.run_v2_async", side_effect=_mock_v2) as run,
            patch("app.main.action_finder.find_actions_async", finder),
        ):
            resp = client.post(
                RS_V2,
                json={
                    "actions": ["Information Retrieval"],
                    "queries": [{"id": "q1", "text": ["last A1C for the patient"]}],
                    "model_name": "gemini-2.5-flash",
                },
            )
        assert resp.status_code == 200
        finder.assert_called_once()
        assert finder.call_args.args[0] == "last A1C for the patient"
        assert finder.call_args.kwargs["model_name"] == "gemini-2.5-flash"
        assert finder.call_args.kwargs["location"] == "us-central1"
        run.assert_called_once()
        body = resp.json()
        assert "skipped" not in body["output"]["queries"][0]
        assert body["details"]["usage_metadata"]["action_finder"] == _usage(5)

    def test_nonmatching_action_skips_pipeline(self, client):
        finder = _finder(["admin_response"])
        with (
            patch("app.main.pipeline.run_v2_async", side_effect=_mock_v2) as run,
            patch("app.main.action_finder.find_actions_async", finder),
        ):
            resp = client.post(
                RS_V2,
                json={
                    "actions": ["information_retrieval"],
                    "queries": [{"id": "q1", "text": ["book me an appointment"]}],
                },
            )
        assert resp.status_code == 200
        run.assert_not_called()
        body = resp.json()
        assert body["status"] == 1
        assert body["output"]["total_queries"] == 1
        q = body["output"]["queries"][0]
        assert q == {
            "id": "q1",
            "text": "book me an appointment",
            "skipped": True,
            "skip_reason": ["admin_response"],
            "skip_reason_detail": "Action Finder classified this query as admin_response",
            "record_types": [],
            "temporal": None,
            "tags": [],
        }
        # Nothing ran, so no pipeline timing.
        assert body["details"]["timing"]["total_llm_seconds"] == 0

    def test_matching_non_retrieval_action_still_skips(self, client):
        finder = _finder(["admin_response"])
        with (
            patch("app.main.pipeline.run_v2_async", side_effect=_mock_v2) as run,
            patch("app.main.action_finder.find_actions_async", finder),
        ):
            resp = client.post(
                RS_V2,
                json={
                    "actions": ["admin_response"],
                    "queries": [{"id": "q1", "text": ["book me an appointment"]}],
                },
            )
        assert resp.status_code == 200
        run.assert_not_called()
        assert resp.json()["output"]["queries"][0]["skipped"] is True

    def test_mixed_queries_keep_order_and_ids(self, client):
        async def _classify(text, **kwargs):
            actions = ["information_retrieval"] if "A1C" in text else ["admin_response"]
            return {
                "actions": actions,
                "is_processable": "information_retrieval" in actions,
                "usage_metadata": _usage(5),
            }

        with (
            patch("app.main.pipeline.run_v2_async", side_effect=_mock_v2) as run,
            patch("app.main.action_finder.find_actions_async", side_effect=_classify),
        ):
            resp = client.post(
                RS_V2,
                json={
                    "actions": ["information_retrieval"],
                    "queries": [
                        {"text": ["book me an appointment", "last A1C", "cancel it"]}
                    ],
                },
            )
        assert resp.status_code == 200
        queries = resp.json()["output"]["queries"]
        assert [q["id"] for q in queries] == ["q1", "q2", "q3"]
        assert [q.get("skipped", False) for q in queries] == [True, False, True]
        assert run.call_count == 1
        assert resp.json()["details"]["usage_metadata"]["action_finder"] == _usage(15)

    def test_disabled_action_finder_ignores_actions(self, client):
        with (
            patch("app.main.pipeline.run_v2_async", side_effect=_mock_v2) as run,
            patch("app.main.action_finder.find_actions_async") as finder,
        ):
            resp = client.post(
                RS_V2,
                json={
                    "actions": ["admin_response"],
                    "enable_action_finder": False,
                    "queries": [{"id": "q1", "text": ["chest x-ray reports"]}],
                },
            )
        assert resp.status_code == 200
        finder.assert_not_called()
        run.assert_called_once()

    def test_empty_actions_list_means_no_gating(self, client):
        with (
            patch("app.main.pipeline.run_v2_async", side_effect=_mock_v2) as run,
            patch("app.main.action_finder.find_actions_async") as finder,
        ):
            resp = client.post(
                RS_V2,
                json={"actions": [], "queries": [{"id": "q1", "text": ["x"]}]},
            )
        assert resp.status_code == 200
        finder.assert_not_called()
        run.assert_called_once()

    def test_422_unknown_action_type(self, client):
        with patch("app.main.pipeline.run_v2_async", side_effect=_mock_v2) as run:
            resp = client.post(
                RS_V2,
                json={
                    "actions": ["informational_retrieval"],
                    "queries": [{"id": "q1", "text": ["x"]}],
                },
            )
        assert resp.status_code == 422
        run.assert_not_called()
        detail = resp.json()["detail"]
        assert detail["error"] == "validation_error"
        assert detail["invalid_actions"] == ["informational_retrieval"]

    def test_finder_error_fails_open(self, client):
        async def _boom(text, **kwargs):
            raise ActionFinderError("model returned no recognizable actions")

        with (
            patch("app.main.pipeline.run_v2_async", side_effect=_mock_v2) as run,
            patch("app.main.action_finder.find_actions_async", side_effect=_boom),
        ):
            resp = client.post(
                RS_V2,
                json={
                    "actions": ["information_retrieval"],
                    "queries": [{"id": "q1", "text": ["chest x-ray reports"]}],
                },
            )
        assert resp.status_code == 200
        run.assert_called_once()
        assert "skipped" not in resp.json()["output"]["queries"][0]

    def test_503_when_cluster_not_configured_before_any_llm_call(
        self, client, cluster_unconfigured
    ):
        with (
            patch("app.main.pipeline.run_v2_async", side_effect=_mock_v2) as run,
            patch("app.main.action_finder.find_actions_async") as finder,
        ):
            resp = client.post(
                RS_V2,
                json={
                    "actions": ["information_retrieval"],
                    "queries": [{"id": "q1", "text": ["chest x-ray reports"]}],
                },
            )
        assert resp.status_code == 503
        run.assert_not_called()
        finder.assert_not_called()
        detail = resp.json()["detail"]
        assert detail["error"] == "cluster_service_unavailable"
        assert detail["details"]["missing_settings"] == [
            "CLUSTER_SELECTION_URL",
            "CLUSTER_BQ_DATASET",
        ]

    def test_503_when_cluster_resolution_fails(self, client):
        with (
            patch("app.main.pipeline.run_v2_async", side_effect=_mock_v2),
            patch(
                "app.utils.context_lonic_document_cluster.resolve_record_type_cuis",
                side_effect=ClusterServiceError("bq down", {"error_type": "Forbidden"}),
            ),
        ):
            resp = client.post(
                RS_V2, json={"queries": [{"id": "q1", "text": ["chest x-ray reports"]}]}
            )
        assert resp.status_code == 503
        detail = resp.json()["detail"]
        assert detail["error"] == "cluster_service_unavailable"
        assert detail["message"] == "bq down"

        # Any other failure inside the lookup is wrapped the same way.
        with (
            patch("app.main.pipeline.run_v2_async", side_effect=_mock_v2),
            patch(
                "app.utils.context_lonic_document_cluster.resolve_record_type_cuis",
                side_effect=KeyError("document_class_cui"),
            ),
        ):
            resp = client.post(
                RS_V2, json={"queries": [{"id": "q1", "text": ["chest x-ray reports"]}]}
            )
        assert resp.status_code == 503
        assert resp.json()["detail"]["details"]["error_type"] == "KeyError"

    def test_llm_error_maps_like_v1(self, client):
        from app.exceptions import LLMRateLimitError, LLMServiceError, LLMTimeoutError

        def _raiser(error):
            async def _raise(*_a, **_k):
                raise error

            return _raise

        for exc, code, err_type in [
            (LLMServiceError("boom"), 503, "service_unavailable"),
            (LLMTimeoutError("slow"), 504, "llm_timeout"),
            (LLMRateLimitError("busy"), 429, "rate_limit_exceeded"),
        ]:
            with patch("app.main.pipeline.run_v2_async", side_effect=_raiser(exc)):
                resp = client.post(
                    RS_V2, json={"queries": [{"id": "q1", "text": ["chest x-ray"]}]}
                )
            assert resp.status_code == code
            detail = resp.json()["detail"]
            assert detail["error"] == err_type
            assert "message" in detail and "details" in detail

    def test_hints_merge_like_v1(self, client):
        with patch("app.main.pipeline.run_v2_async", side_effect=_mock_v2):
            resp = client.post(
                RS_V2,
                json={
                    "queries": [
                        {
                            "id": "q1",
                            "text": ["chest x-ray reports"],
                            "record_types": [
                                {"name": "Discharge Summary", "coding": ["C0743221"]}
                            ],
                            "temporal": {
                                "name": "Last 6 Months",
                                "formula": "REF_POINT - 6M",
                            },
                            "tags": [{"name": "Imaging"}],
                        }
                    ]
                },
            )
        assert resp.status_code == 200
        q = resp.json()["output"]["queries"][0]
        assert q["temporal"]["name"] == "Last 6 Months"
        assert [rt["name"] for rt in q["record_types"]] == [
            "Discharge Summary",
            "Radiology Report",
            "Clinical Note",
        ]
        assert q["record_types"][0]["coding"] == [{"code": "C0743221"}]
        assert [t["name"] for t in q["tags"]] == ["Imaging", "Chest X-ray"]

    def test_422_validation_shared_with_v1(self, client):
        resp = client.post(RS_V2, json={"queries": [{"id": "q1", "text": ["   "]}]})
        assert resp.status_code == 422
        assert resp.json()["detail"]["error"] == "validation_error"

        resp = client.post(
            RS_V2, json={"queries": [{"text": ["x"]}], "model_name": "gemini-99"}
        )
        assert resp.status_code == 422
        assert "Invalid 'model_name'" in str(resp.json()["detail"])

        resp = client.post(RS_V2, json={"queries": [{"text": "scalar"}]})
        assert resp.status_code == 422

    def test_v1_endpoint_unaffected(self, client):
        """/v1 still codes through the JSON vocabulary and never calls the gate
        or the cluster, even when a client sends /v2-only fields."""
        with (
            patch("app.main.pipeline.run_v2_async", side_effect=_mock_v2) as run,
            patch("app.main.action_finder.find_actions_async") as finder,
            patch("app.main.project_record_types_cluster") as cluster_proj,
        ):
            resp = client.post(
                RS_V1,
                json={
                    "actions": ["admin_response"],
                    "queries": [{"id": "q1", "text": ["chest x-ray reports"]}],
                },
            )
        assert resp.status_code == 200
        finder.assert_not_called()
        cluster_proj.assert_not_called()
        assert "include_record_type_matching" not in run.call_args.kwargs
        assert "skipped" not in resp.json()["output"]["queries"][0]


@pytest.mark.usefixtures("cluster_configured")
def test_default_resolver_path_end_to_end(client):
    """Real resolver wiring: endpoint -> cluster-selection HTTP -> BigQuery rows
    -> coded record types (only the network clients are faked)."""
    selection = MagicMock(status_code=200)
    selection.json.side_effect = lambda: {
        "results": [{"member_name": "Chest X-ray report"}, {"member_name": "CT report"}]
    }
    rows = [
        {
            "document_class_cui": "C10",
            "document_class": "Imaging",
            "possible_document_name": "Chest X-ray report",
        },
        {
            "document_class_cui": "C30",
            "document_class": "CT",
            "possible_document_name": "CT report",
        },
    ]
    bq_client = MagicMock()
    bq_client.query.return_value.result.return_value = rows

    with (
        patch("app.main.pipeline.run_v2_async", side_effect=_mock_v2),
        patch.object(cluster, "_fetch_identity_token", return_value="tok"),
        patch(
            "app.utils.context_lonic_document_cluster.requests.post",
            return_value=selection,
        ) as post,
        patch.object(cluster, "get_bq_client", return_value=bq_client),
    ):
        resp = client.post(
            RS_V2, json={"queries": [{"id": "q1", "text": ["chest x-ray reports"]}]}
        )
    assert resp.status_code == 200
    q = resp.json()["output"]["queries"][0]
    # Two record types -> two cluster-selection calls (with tag context),
    # one BigQuery query, both types coded from the same rows.
    assert post.call_count == 2
    assert {c.kwargs["json"]["text_list"][0] for c in post.call_args_list} == {
        "Radiology Report",
        "Clinical Note",
    }
    assert all(
        c.kwargs["json"]["context_list"] == ["Chest X-ray"] for c in post.call_args_list
    )
    assert bq_client.query.call_count == 1
    assert q["record_types"] == [
        {"name": "Radiology Report", "coding": [{"code": "C10"}, {"code": "C30"}]},
        {"name": "Clinical Note", "coding": [{"code": "C10"}, {"code": "C30"}]},
    ]


# ---------------------------------------------------------------------------
# Pipeline flag: include_record_type_matching
# ---------------------------------------------------------------------------


class TestPipelineRecordTypeMatchingFlag:
    """Drive the real run_v2_async (model call faked per step) and inspect the
    contextual-environment prompt: /v2 must not inject the JSON vocabulary."""

    @pytest.fixture
    def pipeline(self):
        from unittest.mock import Mock

        from app.mod_intent_extraction import ContextualIntentPipeline
        from app.utils.concept_vocab import ConceptVocab

        with (
            patch("app.mod_intent_extraction.google.auth") as mock_auth,
            patch("app.mod_intent_extraction.genai") as mock_genai,
        ):
            mock_auth.default.return_value = (Mock(), "test-project")
            mock_genai.Client.return_value = MagicMock()
            yield ContextualIntentPipeline(
                project="test-project",
                location="us-central1",
                model="test-model",
                logger=None,
                tracer=None,
                temporal_vocab=None,
                record_type_vocab=ConceptVocab(
                    {"Radiology Report": [{"cui": "C0034571"}]}
                ),
            )

    @pytest.mark.parametrize("include", [True, False])
    def test_vocabulary_block_follows_flag(self, pipeline, include):
        from tests.test_pipeline_v2 import _as_async_mock, _mock_call_by_step

        prompts: dict[str, str] = {}
        inner = _mock_call_by_step()

        def capture(prompt, *args, **kwargs):
            prompts[kwargs.get("step_name")] = prompt
            return inner(prompt, *args, **kwargs)

        with patch.object(
            pipeline, "_call_model_async", side_effect=_as_async_mock(capture)
        ):
            result = asyncio.run(
                pipeline.run_v2_async(
                    "elevated psa in last 3 years",
                    enable_retrieval_signals=True,
                    include_record_type_matching=include,
                )
            )

        prompt = prompts["contextual_environment"]
        assert ("KNOWN RECORD TYPES" in prompt) is include
        assert ('"Radiology Report"' in prompt) is include
        # Generic normalisation guidance is present either way.
        assert "RECORD TYPE MATCHING" in prompt
        assert result["intents"], "pipeline still assembles intents"
        assert "record_type_matches" in result

    def test_default_is_backwards_compatible(self, pipeline):
        from tests.test_pipeline_v2 import _as_async_mock, _mock_call_by_step

        prompts: dict[str, str] = {}
        inner = _mock_call_by_step()

        def capture(prompt, *args, **kwargs):
            prompts[kwargs.get("step_name")] = prompt
            return inner(prompt, *args, **kwargs)

        with patch.object(
            pipeline, "_call_model_async", side_effect=_as_async_mock(capture)
        ):
            asyncio.run(
                pipeline.run_v2_async("elevated psa", enable_retrieval_signals=True)
            )
        assert "KNOWN RECORD TYPES" in prompts["contextual_environment"]
