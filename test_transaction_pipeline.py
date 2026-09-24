"""Record-type resolution through the transaction-selection pipeline
(app/utils/document_transaction_pipeline.py): the standalone link-code flow
made callable per record type — cluster selection -> activities -> Stage 1
-> related documents -> Stage 2 -> document classes -> Stage 3 -> CUIs."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app import config
from app.utils import context_lonic_document_cluster as dc
from app.utils import document_transaction_pipeline as tx

URL = "https://transactions.example/select"


@pytest.fixture
def pipeline_on(monkeypatch):
    monkeypatch.setattr(config, "TRANSACTION_SELECTION_URL", URL)
    monkeypatch.setattr(config, "RECORD_TYPE_RESOLUTION_MODE", "auto")
    monkeypatch.setattr(
        config, "CLUSTER_SELECTION_URL", "https://cluster.example/select"
    )
    monkeypatch.setattr(config, "CLUSTER_BQ_DATASET", "ds")
    monkeypatch.setattr(config, "TRANSACTION_BATCH_SIZE", 2)
    monkeypatch.setattr(config, "TRANSACTION_RETRY_DELAY_SECONDS", 0)
    yield


class _FakeTransactionService:
    """Answers Yes when the candidate (last element of a transaction) contains
    ``yes_marker``; records every payload; can fail the first call."""

    def __init__(self, yes_marker: str = "REL", fail_first: str | None = None):
        self.yes_marker = yes_marker
        self.payloads: list[dict] = []
        self.fail_first = fail_first

    def post(self, url, json=None, headers=None, timeout=None):
        assert url == URL
        self.payloads.append(json)
        resp = MagicMock()
        if self.fail_first:
            kind, self.fail_first = self.fail_first, None
            if kind == "count":
                resp.status_code = 400
                resp.text = "incorrect number of items"
                return resp
            if kind == "503":
                resp.status_code = 503
                resp.text = "unavailable"
                return resp
        resp.status_code = 200
        out = []
        for t in json["transactions"]:
            candidate = " ".join(str(x) for x in t[1:])
            out.append(
                {
                    "answer": "Yes" if self.yes_marker in candidate else "No",
                    "reasoning": "r",
                }
            )
        resp.json.return_value = {"output": out}
        resp.text = ""
        return resp


def _fake_bq(sql: str, name: str, values: list[str]) -> list[dict]:
    """Three tables, keyed by which SQL ran."""
    if "activity_details.activity_id AS activity_id" in sql:  # activities of documents
        rows = []
        for doc in values:
            if doc == "lab report":
                rows += [
                    {
                        "document_name": "Lab Report",
                        "activity_id": "A1",
                        "activity_name": "REL glucose test",
                        "activity_definition": "d1",
                    },
                    {
                        "document_name": "Lab Report",
                        "activity_id": "A2",
                        "activity_name": "billing",
                        "activity_definition": "d2",
                    },
                    {
                        "document_name": "Lab Report",
                        "activity_id": "A3",
                        "activity_name": "REL hba1c test",
                        "activity_definition": "d3",
                    },
                ]
        return rows
    if "SELECT DISTINCT document_name" in sql:  # documents of activities
        docs = {
            "A1": ["Lab Report", "REL Chemistry Panel"],
            "A3": ["REL HbA1c Result"],
            "A2": ["Invoice"],
        }
        return [{"document_name": d} for a in values for d in docs.get(a, [])]
    if "AS node_id" in sql:  # classes of documents
        classes = {
            "lab report": [("Laboratory", "C_LAB")],
            "rel chemistry panel": [
                ("REL Laboratory", "C_LAB"),
                ("Chemistry", "C_CHEM"),
            ],
            "rel hba1c result": [
                ("REL Laboratory", "C_LAB"),
                ("Diabetes REL Monitoring", "C_DM"),
            ],
        }
        return [
            {"document_name": d, "document_class": c, "node_id": n}
            for d in values
            for c, n in classes.get(d, [])
        ]
    raise AssertionError(sql)


class TestTransactionPipeline:
    def test_disabled_without_url_or_in_cluster_mode(self, monkeypatch):
        monkeypatch.setattr(config, "TRANSACTION_SELECTION_URL", "")
        assert not tx.is_enabled()
        monkeypatch.setattr(config, "TRANSACTION_SELECTION_URL", URL)
        assert tx.is_enabled()
        monkeypatch.setattr(config, "RECORD_TYPE_RESOLUTION_MODE", "cluster")
        assert not tx.is_enabled()

    def test_payload_shapes_with_and_without_context(self):
        acts = [{"activity_name": "a", "activity_definition": "d"}]
        p = tx.generate_activity_transaction_payload(acts, "Lab Report", "diabetes")
        assert p["transactions"] == [["Lab Report", "diabetes", "a", "d"]]
        assert (
            p["definitions"]["instructions"][0]
            == "Each transaction contains exactly 4 elements."
        )
        p = tx.generate_activity_transaction_payload(acts, "Lab Report", None)
        assert p["transactions"] == [["Lab Report", "a", "d"]]
        p = tx.generate_document_transaction_payload(["X"], "Lab Report", "")
        assert p["transactions"] == [["Lab Report", "X"]]
        p = tx.generate_class_transaction_payload(
            [{"document_name": "X", "document_class": "K"}], "Lab Report", "ctx"
        )
        assert p["transactions"] == [["Lab Report", "ctx", "X", "K"]]
        assert set(p) == {
            "model_name",
            "transactions",
            "objective",
            "definitions",
            "analysis_guidelines",
            "output_spec",
        }

    def test_seven_steps_select_classes_and_return_node_ids(self, pipeline_on):
        service = _FakeTransactionService()
        with (
            patch.object(dc, "get_probable_documents", return_value=["LAB REPORT"]),
            patch.object(tx, "_query", side_effect=_fake_bq),
            patch.object(tx.requests, "post", side_effect=service.post),
            patch.object(dc, "get_auth_headers_for", return_value={}),
            patch.object(dc, "_log"),
        ):
            hits = tx.resolve_via_transactions("Lab Report", "diabetes")
        # Stage 1 kept A1 and A3 (REL), dropped billing; Stage 2 kept the REL
        # documents; Stage 3 kept the REL classes -> distinct node ids
        assert hits == [
            {"name": "REL Laboratory", "coding": [{"code": "C_LAB"}]},
            {"name": "Diabetes REL Monitoring", "coding": [{"code": "C_DM"}]},
        ]
        stages = [p["definitions"]["instructions"][0] for p in service.payloads]
        # batch size 2: activities 3 -> 2 batches (4 elements); related documents
        # 3 -> 2 batches (3 elements); classes 4 -> 2 batches (4 elements)
        assert stages.count("Each transaction contains exactly 4 elements.") == 4
        assert stages.count("Each transaction contains exactly 3 elements.") == 2
        assert all(
            p["model_name"] == config.TRANSACTION_SELECTION_MODEL
            for p in service.payloads
        )

    def test_count_mismatch_and_transient_errors_are_retried(self, pipeline_on):
        service = _FakeTransactionService(fail_first="count")
        with (
            patch.object(tx.requests, "post", side_effect=service.post),
            patch.object(dc, "get_auth_headers_for", return_value={}),
        ):
            out = tx.send_transaction_selection(
                tx.generate_document_transaction_payload(["REL x"], "s")
            )
        assert (
            out == [{"answer": "Yes", "reasoning": "r"}] and len(service.payloads) == 2
        )
        service = _FakeTransactionService(fail_first="503")
        with (
            patch.object(tx.requests, "post", side_effect=service.post),
            patch.object(dc, "get_auth_headers_for", return_value={}),
        ):
            assert tx.send_transaction_selection(
                tx.generate_document_transaction_payload(["x"], "s")
            ) == [{"answer": "No", "reasoning": "r"}]

    def test_stage_with_nothing_selected_yields_no_hits(self, pipeline_on):
        service = _FakeTransactionService(yes_marker="NEVER")
        with (
            patch.object(dc, "get_probable_documents", return_value=["Lab Report"]),
            patch.object(tx, "_query", side_effect=_fake_bq),
            patch.object(tx.requests, "post", side_effect=service.post),
            patch.object(dc, "get_auth_headers_for", return_value={}),
            patch.object(dc, "_log") as log,
        ):
            assert tx.resolve_via_transactions("Lab Report", "diabetes") == []
        assert log.call_args.args[1]["outcome"] == "no activity selected"

    def test_bigquery_comparisons_are_normalised(self, pipeline_on):
        captured = {}

        def q(sql, name, values):
            captured[name] = (sql, values)
            return []

        with patch.object(tx, "_query", side_effect=q):
            tx.get_activities_for_documents([" Lab Report", "LAB REPORT"])
            tx.get_document_classes_from_results(["Chest X-Ray "])
        assert captured["document_names"][1] == ["chest x-ray"]
        assert "LOWER(TRIM(" in captured["document_names"][0]

    def test_record_types_use_the_pipeline_first_then_fallbacks(self, pipeline_on):
        service = _FakeTransactionService()
        cat = dc.DocumentCatalog(
            [
                {
                    "document_class_cui": "C_NOTE",
                    "document_class": "Note",
                    "possible_document_name": "Progress Note",
                }
            ]
        )
        with (
            patch.object(
                dc,
                "get_probable_documents",
                side_effect=lambda n, c: ["Lab Report"] if "lab" in n.lower() else [],
            ),
            patch.object(tx, "_query", side_effect=_fake_bq),
            patch.object(tx.requests, "post", side_effect=service.post),
            patch.object(dc, "get_auth_headers_for", return_value={}),
            patch.object(dc, "get_document_catalog", return_value=cat),
            patch.object(dc, "_log"),
        ):
            dc._catalog_state["catalog"] = cat
            out = dc.resolve_record_type_cuis(
                ["Laboratory Report", "Progress Note"],
                tags=["diabetes"],
                aliases={"laboratory report": ["Lab Report"]},
            )
        # Laboratory Report: its own label found no probable documents, its
        # alias "Lab Report" ran the pipeline -> node ids. Progress Note: the
        # pipeline found nothing -> the catalog stages coded it.
        assert out == [
            {
                "name": "Laboratory Report",
                "coding": [{"code": "C_LAB"}, {"code": "C_DM"}],
            },
            {"name": "Progress Note", "coding": [{"code": "C_NOTE"}]},
        ]

    def test_pipeline_failure_falls_through_to_other_stages(self, pipeline_on):
        cat = dc.DocumentCatalog(
            [
                {
                    "document_class_cui": "C_LAB",
                    "document_class": "Laboratory",
                    "possible_document_name": "Lab Report",
                }
            ]
        )
        with (
            patch.object(dc, "get_probable_documents", return_value=["Lab Report"]),
            patch.object(tx, "_query", side_effect=RuntimeError("bq down")),
            patch.object(dc, "get_document_catalog", return_value=cat),
            patch.object(dc, "_log") as log,
        ):
            dc._catalog_state["catalog"] = cat
            out = dc.resolve_record_type_cuis(["Lab Report"], tags=[])
        assert out == [{"name": "Lab Report", "coding": [{"code": "C_LAB"}]}]
        assert any(
            c.args[0] == "Record-type transaction pipeline failed"
            for c in log.call_args_list
        )

    def test_transaction_service_gets_its_own_audience_token(self, monkeypatch):
        monkeypatch.setattr(
            config, "CLUSTER_SELECTION_URL", "https://cluster.example/select"
        )
        dc.reset_auth_cache()
        with patch.object(
            dc,
            "_fetch_identity_token",
            side_effect=lambda aud=None: (
                f"tok-for-{aud or config.CLUSTER_SELECTION_URL}"
            ),
        ) as fetch:
            h1 = dc.get_auth_headers_for(URL)
            h2 = dc.get_auth_headers_for(URL)
            h3 = dc.get_auth_headers()
        assert h1["Authorization"] == f"Bearer tok-for-{URL}" and h1 == h2
        assert h3["Authorization"] == "Bearer tok-for-https://cluster.example/select"
        assert fetch.call_count == 2  # one per audience, cached
        dc.reset_auth_cache()
