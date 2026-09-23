"""DEEP integration test for /v1/retrieval-signals.

Unlike tests/test_dtree_retrieval_signals.py (which mocks pipeline.run_v2),
this mocks ONLY the LLM boundary (ContextualIntentPipeline._call_model) and
lets everything real run:

  expand_query_v2 -> extract_intents_v2 || extract_representative_terms
  -> build_context (real TemporalVocab, folded matching, name_for_id)
  -> assemble_v2_intents (real SignalsAssembler, CUI resolution, defaults)
  -> project_query_signals (real ConceptVocab from the shipped seed file)
  -> endpoint envelope

This is the strongest offline validation of shape-compatibility between the
new projection layer and the actual pipeline internals.
"""

import json
import os
import sys
import tempfile
import types
from unittest.mock import MagicMock, patch

# ---- env BEFORE app import ----
_tmp = tempfile.mkdtemp()
_temporal_path = os.path.join(_tmp, "temporal.json")
with open(_temporal_path, "w") as f:
    json.dump(
        {
            # id_1 (order matters: TemporalVocab assigns id_N by key order)
            "Less Than One Year": [{"cui": "C4722677", "formula": "REF_POINT - 1Y"}],
            # id_2
            "2 Days": [{"cui": "C1442455", "formula": "REF_POINT - 2D"}],
            # id_3 — exercised by the DEFAULT_TEMPORAL_TERMS ("recent") path
            "Recent": [{"cui": "C0332185", "formula": None}],
        },
        f,
    )

os.environ.update(
    {
        "ENV": "test",
        "PROJECT_ID": "test-project",
        "GCP_LOCATION": "us-central1",
        "MODEL_VERSION": "gemini-2.5-flash",
        "ENABLE_TRACING": "false",
        "TEMPORAL_GCS_BUCKET": "",
        "TEMPORAL_GCS_PATH": "",
        "TEMPORAL_LOCAL_PATH": _temporal_path,
        "RECORD_TYPE_GCS_BUCKET": "",
        "RECORD_TYPE_GCS_PATH": "",
        # the ACTUAL shipped seed file — validates it parses and _comment is ignored
        "RECORD_TYPE_LOCAL_PATH": os.path.abspath("app/record_type_name_to_cui.json"),
    }
)


# ---- stub heavy GCP deps (same spirit as tests/conftest.py) ----
def _stub(name, attrs=None):
    m = sys.modules.get(name) or types.ModuleType(name)
    for a, v in (attrs or {}).items():
        setattr(m, a, v)
    sys.modules[name] = m
    return m


_stub("aie_logging", {"GCPLogger": MagicMock, "Severity": MagicMock()})
_stub("vertexai", {"init": MagicMock()})
g = _stub("google")
genai = _stub("google.genai", {"Client": MagicMock()})
_stub("google.genai.types", {"GenerateContentConfig": MagicMock()})
auth = _stub(
    "google.auth", {"default": MagicMock(return_value=(MagicMock(), "test-project"))}
)
_stub("google.api_core")
_stub(
    "google.api_core.exceptions",
    {
        k: type(k, (Exception,), {})
        for k in [
            "DeadlineExceeded",
            "ResourceExhausted",
            "ServiceUnavailable",
            "InvalidArgument",
            "GoogleAPIError",
        ]
    },
)
g.genai, g.auth = genai, auth
_stub("google.cloud")
_stub("google.cloud.storage", {"Client": MagicMock()})
for m in [
    "opentelemetry",
    "opentelemetry.trace",
    "opentelemetry.sdk",
    "opentelemetry.sdk.trace",
    "opentelemetry.sdk.trace.export",
    "opentelemetry.sdk.trace.sampling",
    "opentelemetry.sdk.resources",
    "opentelemetry.exporter.cloud_trace",
    "opentelemetry.instrumentation.fastapi",
    "opentelemetry.instrumentation.requests",
]:
    _stub(m)
sys.modules["opentelemetry"].trace = sys.modules["opentelemetry.trace"]
for n, attrs in [
    ("opentelemetry.trace", ["Span"]),
    ("opentelemetry.sdk.trace", ["TracerProvider", "SpanProcessor"]),
    ("opentelemetry.sdk.trace.export", ["BatchSpanProcessor"]),
    ("opentelemetry.sdk.trace.sampling", ["ParentBased", "ALWAYS_ON"]),
    ("opentelemetry.sdk.resources", ["Resource"]),
    ("opentelemetry.exporter.cloud_trace", ["CloudTraceSpanExporter"]),
    ("opentelemetry.instrumentation.fastapi", ["FastAPIInstrumentor"]),
    ("opentelemetry.instrumentation.requests", ["RequestsInstrumentor"]),
]:
    for a in attrs:
        setattr(sys.modules[n], a, MagicMock())

from app.exceptions import LLMTimeoutError
from fastapi.testclient import TestClient

from app import config
from app.main import app, pipeline, record_type_vocab, temporal_vocab

DTREE_RS = f"{config.API_PREFIX}/v1/retrieval-signals"
V2_NB = f"{config.API_PREFIX}/v2/nature-breakdown"


assert temporal_vocab is not None, "temporal vocab must load from local file"
assert record_type_vocab is not None, "record-type seed vocab must load"
assert len(record_type_vocab) == 14, "_comment key must be ignored (14 real entries)"

USAGE = {
    "prompt_token_count": 10,
    "candidates_token_count": 5,
    "total_token_count": 15,
    "thinking_token_count": 0,
}

# ---- canned LLM JSON per real step_name (exact schemas the pipeline validates) ----

INTENTS_JSON = {
    "total_intents_detected": 1,
    "intents": [
        {
            "intent_title": "Chest X-ray Reports",
            "description": "Radiology reports documenting chest x-ray imaging",
            "nature": "[Diagnostic Procedure] / [Imaging]",
            "sub_natures": [
                {
                    "category_path": "Imaging >> Radiology >> Chest",
                    "atomic_concepts": ["chest x-ray"],
                }
            ],
            "final_queries": ["chest x-ray"],
        }
    ],
}

CONTEXT_JSON = {
    "concepts_with_context": [
        {
            "atomic_concept": "chest x-ray",
            "intent_title": "Chest X-ray Reports",
            "record_types": ["radiology_report", "clinical_note", "some_new_doc"],
            "author_roles": ["radiologist"],
            "longitudinal_scope": ["diagnostic_workup"],
            "content_signals": ["structured_finding"],
            "clinical_settings": ["outpatient"],
        }
    ],
    "temporal_by_intent": [
        {
            "intent_title": "Chest X-ray Reports",
            "candidates": [
                {
                    "candidate": "chest x-ray",
                    "temporal_signal": [
                        {
                            "signal": "last year",
                            "selected_id": "id_1",  # -> "Less Than One Year"
                            "selected_name": "Less Than One Year",
                            "selected_reasoning": "explicit one-year window",
                        }
                    ],
                }
            ],
        }
    ],
}

# variant with NO temporal signal -> must hit the vocab "Recent" default path
CONTEXT_JSON_NO_TEMPORAL = {
    "concepts_with_context": CONTEXT_JSON["concepts_with_context"],
    "temporal_by_intent": [
        {
            "intent_title": "Chest X-ray Reports",
            "candidates": [{"candidate": "chest x-ray", "temporal_signal": []}],
        }
    ],
}

call_log = []


def make_call_model(context_payload):
    def fake_call_model(
        prompt,
        model_name=None,
        generation_config=None,
        temperature=0.0,
        max_tokens=65536,
        timeout=300,
        step_name="llm_call",
        response_schema=None,
    ):
        call_log.append(step_name)
        if step_name == "query_expansion_v2":
            return json.dumps(
                {
                    "expanded_query": "chest radiograph (chest x-ray) reports from the last one year",
                    "abbreviations_expanded": [],
                }
            ), dict(USAGE)
        if step_name == "intent_extraction_v2":
            return json.dumps(INTENTS_JSON), dict(USAGE)
        if step_name == "representative_terms":
            return json.dumps({"representative_terms": ["Chest X-ray"]}), dict(USAGE)
        if step_name == "contextual_environment":
            # sanity: the folded-matching block must be injected when vocab is set
            assert "TEMPORAL CONCEPTS (id, name):" in prompt
            assert "Less Than One Year" in prompt
            return json.dumps(context_payload), dict(USAGE)
        raise AssertionError(f"unexpected step: {step_name}")

    return fake_call_model


client = TestClient(app)

# =========================================================================
# TEST 1 — full pipeline, temporal resolved through the REAL vocab
# =========================================================================
with patch.object(pipeline, "_call_model", side_effect=make_call_model(CONTEXT_JSON)):
    resp = client.post(
        DTREE_RS,
        json={
            "queries": [{"id": "q1", "text": "chest x-ray reports from the last year"}]
        },
    )

assert resp.status_code == 200, resp.text
body = resp.json()
assert body["status"] == 1
assert body["service"].startswith("intent-nature-breakdown-api")
assert sorted(set(call_log)) == sorted(
    {
        "query_expansion_v2",
        "intent_extraction_v2",
        "representative_terms",
        "contextual_environment",
    }
), call_log

q = body["output"]["queries"][0]
assert q["id"] == "q1"
assert q["text"] == "chest x-ray reports from the last year"

# record types: real SignalsAssembler union -> real ConceptVocab (seed file) codes
assert q["record_types"][0] == {
    "name": "Radiology Report",
    "coding": [{"code": "C0034571"}],
}
assert q["record_types"][1]["name"] == "Clinical Note"
assert q["record_types"][1]["coding"] == [{"code": "C1961136"}]
# vocab miss -> normalized name, empty coding (endpoint still functions)
assert q["record_types"][2] == {"name": "Some New Doc", "coding": []}

# temporal: LLM picked id_1 -> vocab resolved name/cui/formula end-to-end
assert q["temporal"] == {
    "name": "Less Than One Year",
    "formula": "REF_POINT - 1Y",
    "coding": [{"system": "UMLS", "code": "C4722677"}],
}

# tags: real representative_terms step -> projection
assert q["tags"][0]["name"] == "Chest X-ray"
assert q["tags"][0]["coding"] == []  # no tag vocab provisioned (documented)
assert "imaging" in q["tags"][0]["topics"] and "radiology" in q["tags"][0]["topics"]

# envelope details: usage aggregated across all four v2 steps
um = body["details"]["usage_metadata"]
for step in [
    "query_expansion",
    "intent_extraction",
    "representative_terms",
    "contextual_environment",
]:
    assert um[step]["total_token_count"] == 15, (step, um[step])

print("TEST 1 PASS — full pipeline, vocab-resolved temporal + coded record types")

# =========================================================================
# TEST 2 — no temporal signal from LLM -> SignalsAssembler default via vocab
# ('recent' in DEFAULT_TEMPORAL_TERMS resolves against the loaded vocab)
# =========================================================================
call_log.clear()
with patch.object(
    pipeline, "_call_model", side_effect=make_call_model(CONTEXT_JSON_NO_TEMPORAL)
):
    resp = client.post(
        DTREE_RS,
        json={"queries": [{"id": "q2", "text": "chest x-ray reports"}]},
    )
q = resp.json()["output"]["queries"][0]
assert q["temporal"] == {
    "name": "Recent",
    "formula": None,
    "coding": [{"system": "UMLS", "code": "C0332185"}],
}, q["temporal"]
print("TEST 2 PASS — temporal default path resolves 'Recent'/C0332185 via vocab")

# =========================================================================
# TEST 3 — multi-query independence + hint override, through the real pipeline
# =========================================================================
with patch.object(pipeline, "_call_model", side_effect=make_call_model(CONTEXT_JSON)):
    resp = client.post(
        DTREE_RS,
        json={
            "queries": [
                {"id": "a", "text": "chest x-ray reports from the last year"},
                {
                    "id": "b",
                    "text": "chest x-ray reports",
                    "temporal": {
                        "name": "Last 6 Months",
                        "formula": "REF_POINT - 6M",
                        "coding": [{"code": "C0332177"}],
                    },
                    "record_types": [
                        {
                            "name": "Radiology Report",
                            "coding": [{"code": "CLIENT-CODE"}],
                        }
                    ],
                },
            ]
        },
    )
body = resp.json()
qa, qb = body["output"]["queries"]
assert (qa["id"], qb["id"]) == ("a", "b")
assert body["output"]["total_queries"] == 2
# derived temporal on a, hint override on b
assert qa["temporal"]["name"] == "Less Than One Year"
assert qb["temporal"]["name"] == "Last 6 Months"
assert qb["temporal"]["coding"] == [
    {"system": "UMLS", "code": "C0332177"}
]  # Coding default system applied
# hint record type wins the dedupe and keeps client coding
assert qb["record_types"][0]["coding"] == [{"code": "CLIENT-CODE"}]
assert [r["name"] for r in qb["record_types"]].count("Radiology Report") == 1
print("TEST 3 PASS — multi-query independence + hint override/merge end-to-end")

# =========================================================================
# TEST 4 — LLM failure surfaces exactly like the existing endpoints
# =========================================================================


def timeout_call(*a, **k):
    raise LLMTimeoutError("LLM call timed out", details={"step": k.get("step_name")})


with patch.object(pipeline, "_call_model", side_effect=timeout_call):
    resp = client.post(
        DTREE_RS,
        json={"queries": [{"id": "q1", "text": "chest x-ray"}]},
    )
assert resp.status_code == 504, resp.text
d = resp.json()["detail"]
assert d["error"] == "llm_timeout" and "message" in d and "details" in d
print(
    "TEST 4 PASS — LLMTimeoutError -> 504 {error, message, details} (same as /v1,/v2)"
)

# =========================================================================
# TEST 5 — existing endpoints still work in the SAME app instance
# =========================================================================
with patch.object(pipeline, "_call_model", side_effect=make_call_model(CONTEXT_JSON)):
    r1 = client.post(
        V2_NB,
        json={"texts": "chest x-ray reports", "enable_retrieval_signals": True},
    )
assert r1.status_code == 200 and r1.json()["status"] == 1
sig = r1.json()["output"]["results"][0]["intents"][0]["retrieval_signals"]
assert sig["temporal"][0]["codes"] == "C4722677"  # v2 keeps its own shape untouched
print("TEST 5 PASS — /v2/nature-breakdown unaffected, same app, same pipeline")

print("\nALL 5 DEEP INTEGRATION TESTS PASS")
