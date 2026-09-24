"""Unit tests for /v3/retrieval-signals — canonical temporal mode (no LLM,
no network).

Covers the temporal index built from the vocabulary file, the canonical
window resolver, the cadence memory, the prompt block, the pipeline wiring
(prompt content, schema, size, logging, shadow) and the endpoint. /v1 and /v2
behaviour is asserted unchanged.
"""

import asyncio
import json
import os
import sys
from unittest.mock import MagicMock, Mock, patch

import pytest

from app import config
from app.mod_intent_extraction import (
    TEMPORAL_MODE_CANONICAL,
    TEMPORAL_MODE_VOCAB_LIST,
    ContextualIntentPipeline,
)
from app.prompts.v3.contextual_environment import (
    CONTEXTUAL_ENVIRONMENT_PROMPT_V3,
    build_contextual_environment_prompt_v3,
    render_examples,
    render_menu,
    render_shortlist,
)
from app.utils import cadence_memory as cm
from app.utils.dtree_signals import project_temporal
from app.utils.retrieval_signals import (
    ContextualEnvironmentOutput,
    ContextualEnvironmentOutputCanonical,
    assemble_v2_intents,
)
from app.utils.temporal_canonical import (
    CanonicalTemporalResolver,
    CanonicalWindow,
    window_span_seconds,
)
from app.utils.temporal_index import (
    LexicalSimilarity,
    TemporalIndex,
    parse_formula,
    parse_number_unit,
)
from app.utils.temporal_vocab import TemporalVocab

RS_V2 = f"{config.API_PREFIX}/v2/retrieval-signals"
RS_V3 = f"{config.API_PREFIX}/v3/retrieval-signals"
VOCAB_FILE = os.path.join(
    os.path.dirname(__file__), "..", "app", "temporal_name_to_cui.json"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def vocab() -> TemporalVocab:
    """Small vocabulary in the production file's shape; formulas drive the index."""
    return TemporalVocab(
        {
            "_default_temporal_name": "Recent",
            "Recent": [{"cui": "C0332185", "formula": ["REF_POINT", "REF_POINT"]}],
            "Past 6 Months": [
                {"cui": "C_6M_A", "formula": ["REF_POINT", "REF_POINT - 6M"]}
            ],
            "6 months": [{"cui": "C_6M_B", "formula": ["REF_POINT", "REF_POINT - 6M"]}],
            "In the last 6Mo, did you skip meals": [
                {"cui": "C_6M_Q", "formula": ["REF_POINT", "REF_POINT - 6M"]}
            ],
            "In past 3 years": [
                {"cui": "C_3Y", "formula": ["REF_POINT", "REF_POINT - 3Y"]}
            ],
            "Past Year": [{"cui": "C_1Y", "formula": ["REF_POINT", "REF_POINT - 1Y"]}],
            "Within 30 days": [
                {"cui": "C_30D", "formula": ["REF_POINT", "REF_POINT - 30D"]}
            ],
            "Past 2 Weeks": [
                {"cui": "C_2W", "formula": ["REF_POINT", "REF_POINT - 2W"]}
            ],
            "Between 6 months and 1 year ago": [
                {"cui": "C_6M_1Y", "formula": ["REF_POINT - 6M", "REF_POINT - 1Y"]}
            ],
            "24 hours post dose": [
                {"cui": "C_24H", "formula": ["REF_POINT", "REF_POINT - 24H"]}
            ],
            "Every five minutes": [
                {"cui": "C_5MIN", "formula": ["REF_POINT", "REF_POINT - 5MIN"]}
            ],
            "Unparseable": [{"cui": "C_X", "formula": ["SOMETHING", "ELSE"]}],
        }
    )


@pytest.fixture(scope="module")
def index(vocab) -> TemporalIndex:
    return TemporalIndex.from_vocab(vocab)


@pytest.fixture(scope="module")
def resolver(index) -> CanonicalTemporalResolver:
    return CanonicalTemporalResolver(index, max_codes=2)


@pytest.fixture
def memory() -> cm.CadenceMemory:
    return cm.CadenceMemory(
        {
            "_meta": {"version": "2026-09-23T00:00:00Z"},
            "concepts": {
                "hba1c": {
                    "concept": "HbA1c",
                    "total": 100,
                    "windows": [
                        {"label": "last 6 months", "count": 80, "share": 0.8},
                        {"label": "last 3 months", "count": 20, "share": 0.2},
                    ],
                },
                "metformin": {
                    "concept": "metformin",
                    "total": 10,
                    "windows": [{"label": "last 12 months", "count": 10, "share": 1.0}],
                },
            },
        }
    )


def _pipeline(vocab, index=None, memory=None, settings=None, logger=None):
    with (
        patch("app.mod_intent_extraction.google.auth") as mock_auth,
        patch("app.mod_intent_extraction.genai") as mock_genai,
    ):
        mock_auth.default.return_value = (Mock(), "test-project")
        mock_genai.Client.return_value = MagicMock()
        return ContextualIntentPipeline(
            project="test-project",
            location="us-central1",
            model="test-model",
            logger=logger,
            tracer=None,
            temporal_vocab=vocab,
            temporal_index=index,
            cadence_memory=memory,
            temporal_settings=settings,
        )


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------


class TestTemporalIndex:
    def test_parse_formula_shapes(self):
        assert parse_formula(["REF_POINT", "REF_POINT - 3Y"]) == (
            "single",
            3.0,
            "Y",
            None,
            None,
        )
        assert parse_formula(["REF_POINT - 4W", "REF_POINT - 6M"]) == (
            "range",
            4.0,
            "W",
            6.0,
            "M",
        )
        assert parse_formula(["REF_POINT", "REF_POINT"]) is None
        assert parse_formula(["x"]) is None
        assert parse_formula(None) is None

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("last 3 years", [(3.0, "year")]),
            ("previous 36 months", [(36.0, "month")]),
            ("within thirty days", [(30.0, "day")]),
            ("twenty four hours", [(24.0, "hour")]),
            ("last year", [(1.0, "year")]),
            ("a week", [(1.0, "week")]),
            ("6 to 12 months ago", [(6.0, "month"), (12.0, "month")]),
            ("6-12 months", [(6.0, "month"), (12.0, "month")]),
            ("patient with diabetes", []),
            ("1.5 hours", [(1.5, "hour")]),
        ],
    )
    def test_parse_number_unit(self, text, expected):
        assert parse_number_unit(text) == expected

    def test_index_learns_unit_codes_from_the_file(self, index):
        cov = index.coverage()
        assert cov["entries"] == 12
        assert (
            cov["with_canonical_key"] == 10
        )  # Recent (no span) and Unparseable are not keyed
        # Learned from the file (formula code paired with the unit word in the
        # name); "second" never occurs in this vocabulary and gets the
        # conventional code so uncodeable windows still carry a formula.
        assert index.unit_codes == {
            "month": "M",
            "year": "Y",
            "day": "D",
            "week": "W",
            "hour": "H",
            "minute": "MIN",
            "second": "S",
        }
        assert index.coverage()["keyed_by_formula"] == cov["with_canonical_key"]
        assert index.similarity_provider == "lexical"

    def test_tolerant_formula_syntax_and_name_fallback(self):
        # A vocabulary whose formulas use another reference token, a single
        # string, no spaces, unit words, or nothing parseable at all.
        tv = TemporalVocab(
            {
                "Past 6 Months": [{"cui": "C1", "formula": "T0 minus 6 months"}],
                "Six to twelve months ago": [{"cui": "C2", "formula": ["x"]}],
                "Item A": [{"cui": "C3", "formula": ["NOW", "NOW-2Y"]}],
                "Item B": [{"cui": "C4", "formula": "ref_date | ref_date - 30 min"}],
                "Item C": [
                    {"cui": "C5", "formula": {"start": "NOW", "end": "NOW - 3M"}}
                ],
                "Something else": [{"cui": "C6", "formula": ["NOW", "NOW"]}],
            }
        )
        idx = TemporalIndex.from_vocab(tv)
        cov = idx.coverage()
        assert cov["keyed_by_formula"] == 3 and cov["keyed_by_name"] == 2
        assert cov["unkeyed"] == 1 and cov["unkeyed_samples"] == [
            {"name": "Something else", "formula": ["NOW", "NOW"]}
        ]
        by = {e.name: e for e in idx.entries}
        assert by["Past 6 Months"].key == idx.key_for("single", 6, "month")
        assert by["Past 6 Months"].key_source == "name"
        assert by["Past 6 Months"].formula == ["T0 minus 6 months"]  # kept verbatim
        assert by["Six to twelve months ago"].key == idx.key_for(
            "range", 6, "month", 12, "month"
        )
        assert by["Item A"].key == idx.key_for("single", 2, "year")
        assert by["Item B"].key == idx.key_for("single", 30, "minute")
        assert by["Item C"].key == idx.key_for("single", 3, "month")
        # No unit word ever appeared in a name: codes derived from the code text.
        assert idx.code_units["MIN"] == "minute" and idx.code_units["M"] == "month"

    def test_rank_for_key_prefers_wording_then_brevity(self, index):
        key = index.key_for("single", 6, "month")
        names = [
            m.entry.name for m in index.rank_for_key(key, "past 6 months", max_codes=3)
        ]
        # Wording match first, then the short plain name, the questionnaire sentence last.
        assert names == [
            "Past 6 Months",
            "6 months",
            "In the last 6Mo, did you skip meals",
        ]
        assert [
            m.entry.cui for m in index.rank_for_key(key, "recent", max_codes=2)
        ] == [
            "C_6M_B",
            "C_6M_A",
        ] or len(index.rank_for_key(key, "recent", max_codes=2)) == 2

    def test_rank_for_key_range_and_missing(self, index):
        key = index.key_for("range", 6, "month", 1, "year")
        assert [
            m.entry.cui for m in index.rank_for_key(key, "6 months to 1 year", 2)
        ] == ["C_6M_1Y"]
        assert (
            index.rank_for_key(index.key_for("single", 7, "year"), "7 years", 2) == []
        )
        assert index.can_encode(index.key_for("single", 7, "year"))
        # No "second" entry in the vocabulary: the window has no CUI but still
        # encodes to a well-formed formula (conventional code).
        assert index.entries_for_key(index.key_for("single", 7, "second")) == []
        assert index.key_for("single", 7, "second").formula(index.unit_codes) == [
            "REF_POINT",
            "REF_POINT - 7S",
        ]

    def test_menu_and_nearest_and_windows_from_text(self, index):
        menu = index.menu(12, units=["day", "week", "month", "year"])
        assert menu == {"day": [30], "week": [2], "month": [6], "year": [1, 3]}
        top = index.nearest("chest x-ray reports from the past year", k=3)
        assert top and top[0].entry.name == "Past Year"
        keys = index.windows_from_text("labs from 6 to 12 months ago")
        assert keys[0].kind == "range" and keys[0].value == 6 and keys[0].to_value == 12
        assert index.windows_from_text("patient with diabetes") == []

    def test_production_vocabulary_indexes_fully(self):
        if not os.path.exists(VOCAB_FILE) or os.path.getsize(VOCAB_FILE) == 0:
            pytest.skip("production vocabulary file not bundled")
        with open(VOCAB_FILE, encoding="utf-8") as f:
            idx = TemporalIndex.from_vocab(
                TemporalVocab(json.load(f))
            )  # from_file is mocked in conftest
        cov = idx.coverage()
        assert cov["entries"] > 1000
        assert cov["with_canonical_key"] / cov["entries"] > 0.99
        assert {"day", "week", "month", "year", "hour", "minute"} <= set(idx.unit_codes)
        assert (
            idx.rank_for_key(idx.key_for("single", 6, "month"), "past 6 months", 1)[
                0
            ].entry.name
            == "Past 6 Months"
        )

    def test_lexical_similarity_is_deterministic(self):
        sim = LexicalSimilarity(["Past Year", "Past 6 Months"])
        assert sim.scores("past year") == sim.scores("past year")
        assert sim.scores("past year")[0] > sim.scores("past year")[1]

    def test_vertex_provider_falls_back_to_lexical_on_failure(self, vocab):
        with patch("google.genai.Client", side_effect=RuntimeError("no creds")):
            idx = TemporalIndex.from_vocab(
                vocab,
                similarity_provider="vertex",
                embedding_model="text-embedding-x",
                project="p",
                location="us",
            )
        assert idx.similarity_provider == "lexical"


# ---------------------------------------------------------------------------
# Canonical resolver
# ---------------------------------------------------------------------------


class TestCanonicalResolver:
    def test_window_normalisation(self, index):
        w = CanonicalWindow(relation="Last", value=6, unit="Months")
        assert (w.relation, w.unit) == ("last", "month")
        key = CanonicalWindow(
            relation="range", value=1, unit="year", to_value=6, to_unit="month"
        ).key(index)
        assert (key.value, key.unit, key.to_value, key.to_unit) == (
            6,
            "month",
            1,
            "year",
        )  # ordered by span
        assert (
            CanonicalWindow(relation="range", value=1, unit="year").key(index) is None
        )

    def test_resolve_window_exact_broader_widest_and_bad(self, resolver):
        out = resolver.resolve_window(
            {"relation": "last", "value": 6, "unit": "month"},
            wording="past 6 months",
            basis="explicit",
        )
        assert [m.codes for m in out] == ["C_6M_A", "C_6M_B"]
        assert out[0].resolution == "vocabulary" and out[0].basis == "explicit"
        assert out[0].window == {"relation": "last", "value": 6, "unit": "month"}

        # Not in the vocabulary -> the narrowest vocabulary window that still
        # contains it (the list prompt's "nearest broader, never narrower").
        out = resolver.resolve_window(
            {"relation": "last", "value": 11, "unit": "day"}, basis="inferred"
        )
        assert [(m.time_window, m.codes, m.resolution) for m in out] == [
            ("Past 2 Weeks", "C_2W", "broader")  # 14 days: narrowest that contains 11
        ]
        assert out[0].formula == ["REF_POINT", "REF_POINT - 2W"]  # the entry's own
        assert out[0].window == {"relation": "last", "value": 11, "unit": "day"}
        out = resolver.resolve_window({"relation": "last", "value": 20, "unit": "day"})
        assert (out[0].time_window, out[0].resolution) == ("Within 30 days", "broader")
        out = resolver.resolve_window(
            {"relation": "range", "value": 2, "unit": "month", "to_value": 4}
        )
        assert out[0].time_window in ("Past 6 Months", "6 months")
        assert out[0].resolution == "broader"
        # Wider than anything the vocabulary has -> its widest window.
        out = resolver.resolve_window(
            {"relation": "last", "value": 7, "unit": "year"},
            basis="inferred",
            rationale="r",
        )
        assert [(m.time_window, m.codes, m.resolution) for m in out] == [
            ("In past 3 years", "C_3Y", "widest")
        ]
        # Every resolution is a vocabulary entry: name, CUI and formula as /v1.
        assert all(m.codes and m.time_window and m.formula for m in out)

        assert (
            resolver.resolve_window({"relation": "last", "value": 0, "unit": "year"})
            == []
        )
        # an unknown relation word still means "back from the reference point"
        out = resolver.resolve_window({"relation": "since", "value": 2, "unit": "year"})
        assert out and out[0].time_window == "In past 3 years"  # 2y -> nearest broader
        assert (
            resolver.resolve_window(
                {"relation": "last", "value": 2, "unit": "fortnight"}
            )
            == []
        )
        assert resolver.resolve_window(None) == []

    def test_resolve_text_fallback(self, resolver):
        out = resolver.resolve_text("last 3 years", basis="explicit")
        assert [m.codes for m in out] == ["C_3Y"]
        assert resolver.resolve_text("recent") == []

    def test_window_span_seconds(self):
        assert (
            window_span_seconds({"relation": "last", "value": 1, "unit": "year"})
            == 365 * 86400
        )
        assert (
            window_span_seconds(
                {
                    "relation": "range",
                    "value": 6,
                    "unit": "month",
                    "to_value": 1,
                    "to_unit": "year",
                }
            )
            == 365 * 86400
        )
        assert (
            window_span_seconds({"relation": "last", "value": "x", "unit": "year"})
            is None
        )
        assert window_span_seconds(None) is None


# ---------------------------------------------------------------------------
# Assembler in canonical mode
# ---------------------------------------------------------------------------

_INTENTS = [
    {
        "intent_title": "Diabetes",
        "nature": "[Chronic]",
        "sub_natures": [
            {
                "category_path": "x",
                "atomic_concepts": ["HbA1c", "metformin", "foot exam"],
            }
        ],
    }
]


def _context(with_windows: bool = True):
    def entry(signal, window, basis, sb="type_default"):
        e = {"signal": signal, "signal_basis": sb}
        if with_windows:
            e.update({"window": window, "basis": basis, "rationale": "because"})
        return e

    return {
        "concepts_with_context": [
            {
                "atomic_concept": c,
                "intent_title": "Diabetes",
                "record_types": ["progress_note"],
            }
            for c in ("HbA1c", "metformin", "foot exam")
        ],
        "record_type_matches": [],
        "temporal_by_intent": [
            {
                "intent_title": "Diabetes",
                "candidates": [
                    {
                        "candidate": "HbA1c",
                        "temporal_signal": [
                            entry(
                                "recent",
                                {"relation": "last", "value": 6, "unit": "month"},
                                "inferred",
                            )
                        ],
                    },
                    {
                        "candidate": "metformin",
                        "temporal_signal": [
                            entry(
                                "current",
                                {"relation": "last", "value": 7, "unit": "year"},
                                "inferred",
                            )
                        ],
                    },
                    {
                        "candidate": "foot exam",
                        "temporal_signal": [
                            entry("last 3 years", None, "explicit", "explicit_window")
                        ],
                    },
                ],
            }
        ],
    }


class TestCanonicalAssembly:
    def test_canonical_models_keep_window_fields(self):
        ctx = ContextualEnvironmentOutputCanonical.model_validate(_context())
        sig = ctx.temporal_by_intent[0].candidates[0].temporal_signal[0]
        assert (
            sig.window.value == 6
            and sig.basis == "inferred"
            and sig.rationale == "because"
        )
        # Legacy model ignores them (schema unchanged).
        legacy = ContextualEnvironmentOutput.model_validate(_context())
        assert not hasattr(
            legacy.temporal_by_intent[0].candidates[0].temporal_signal[0], "window"
        )

    def test_assembler_resolves_codes_broader_and_text_fallback(self, vocab, resolver):
        ctx = ContextualEnvironmentOutputCanonical.model_validate(_context())
        out = assemble_v2_intents(
            _INTENTS, context=ctx, vocab=vocab, canonical_resolver=resolver
        )
        by = {
            fc["candidate"]: fc["retrieval_signals"]["temporal"]
            for fc in out[0]["final_candidates"]
        }
        # Wording ("recent") is not close to any 6-month name: one vocabulary
        # entry, the plain shortest name, as the list prompt would have picked.
        assert [t["codes"] for t in by["HbA1c"]] == ["C_6M_B"]
        assert by["HbA1c"][0]["time_window"] == "6 months"
        assert by["metformin"] == [
            {
                "time_window": "In past 3 years",
                "codes": "C_3Y",
                "formula": ["REF_POINT", "REF_POINT - 3Y"],
                "basis": "inferred",
                "rationale": "because",
                "window": {"relation": "last", "value": 7, "unit": "year"},
                "resolution": "widest",
            }
        ]
        # No window from the model -> parsed from the wording.
        assert (
            by["foot exam"][0]["codes"] == "C_3Y"
            and by["foot exam"][0]["basis"] == "explicit"
        )

    def test_assembler_default_comes_from_vocabulary_not_a_constant(
        self, vocab, resolver
    ):
        ctx = _context()
        # No candidate carries a window or a parseable span -> nothing resolves.
        for cand in ctx["temporal_by_intent"][0]["candidates"]:
            cand["temporal_signal"] = [
                {"signal": "current", "signal_basis": "type_default"}
            ]
        out = assemble_v2_intents(
            _INTENTS,
            context=ContextualEnvironmentOutputCanonical.model_validate(ctx),
            vocab=vocab,
            canonical_resolver=resolver,
        )
        hba1c = out[0]["final_candidates"][0]["retrieval_signals"]["temporal"]
        assert (
            hba1c[0]["time_window"] == "Recent"
            and hba1c[0]["resolution"] == "default"
            and hba1c[0]["basis"] == "default"
        )

    def test_legacy_assembly_shape_unchanged(self, vocab):
        out = assemble_v2_intents(
            _INTENTS,
            context=ContextualEnvironmentOutput.model_validate(_context()),
            vocab=vocab,
        )
        t = out[0]["final_candidates"][0]["retrieval_signals"]["temporal"][0]
        assert set(t) == {"time_window", "codes", "formula"}


# ---------------------------------------------------------------------------
# Cadence memory
# ---------------------------------------------------------------------------


class TestCadenceMemory:
    def test_window_label(self):
        assert (
            cm.window_label({"relation": "last", "value": 6.0, "unit": "month"})
            == "last 6 months"
        )
        assert (
            cm.window_label({"relation": "last", "value": 1, "unit": "year"})
            == "last 1 year"
        )
        assert (
            cm.window_label(
                {
                    "relation": "range",
                    "value": 6,
                    "unit": "month",
                    "to_value": 12,
                    "to_unit": "month",
                }
            )
            == "between 6 and 12 months"
        )
        assert (
            cm.window_label(
                {
                    "relation": "range",
                    "value": 6,
                    "unit": "month",
                    "to_value": 1,
                    "to_unit": "year",
                }
            )
            == "between 6 months and 1 year"
        )
        assert cm.window_label({}) is None

    def test_examples_for_uses_similarity_and_limits(self, memory):
        ex = memory.examples_for(
            ["Hemoglobin A1c", "HbA1c", "metformin", "unknown thing"],
            per_concept=1,
            max_concepts=10,
            min_similarity=0.3,
        )
        assert ex == ["HbA1c: last 6 months (80x)", "metformin: last 12 months (10x)"]
        assert memory.examples_for(
            ["HbA1c"], per_concept=5, max_concepts=1, min_similarity=0.9
        ) == ["HbA1c: last 6 months (80x), last 3 months (20x)"]
        assert memory.examples_for(["zzz"], 1, 1, 0.9) == []
        assert cm.CadenceMemory.empty().examples_for(["HbA1c"], 1, 1, 0.1) == []

    def test_aggregate_and_drift(self):
        events = [
            {
                "basis": "inferred",
                "resolution": "vocabulary",
                "candidate": "HbA1c",
                "window": {"relation": "last", "value": 6, "unit": "month"},
                "timestamp": "2026-09-20T10:00:00Z",
            },
            {
                "basis": "inferred",
                "resolution": "vocabulary",
                "candidate": "hba1c",
                "window": {"relation": "last", "value": 6, "unit": "month"},
                "timestamp": "2026-09-21T10:00:00Z",
            },
            {
                "basis": "inferred",
                "resolution": "broader",
                "candidate": "HbA1c",
                "window": {"relation": "last", "value": 3, "unit": "month"},
            },
            {
                "basis": "explicit",
                "resolution": "vocabulary",
                "candidate": "HbA1c",
                "window": {"relation": "last", "value": 1, "unit": "year"},
            },
            {
                "basis": "inferred",
                "resolution": "default",
                "candidate": "HbA1c",
                "window": None,
            },
            {
                "basis": "inferred",
                "resolution": "vocabulary",
                "candidate": "",
                "window": {"relation": "last", "value": 1, "unit": "year"},
            },
        ]
        mem = cm.aggregate_events(events, version="v1")
        assert mem["_meta"]["events"] == 3 and mem["_meta"]["concepts"] == 1
        hb = mem["concepts"]["hba1c"]
        assert hb["total"] == 3 and hb["windows"][0] == {
            "label": "last 6 months",
            "relation": "last",
            "value": 6,
            "unit": "month",
            "count": 2,
            "last_seen": "2026-09-21",
            "share": 0.6667,
        }
        later = cm.aggregate_events(
            [
                {
                    "basis": "inferred",
                    "resolution": "vocabulary",
                    "candidate": "HbA1c",
                    "window": {"relation": "last", "value": 3, "unit": "month"},
                }
            ]
            * 3,
            version="v2",
        )
        report = cm.drift_report(mem, later)
        assert (
            report[0]["top_changed"] is True
            and report[0]["current_top"] == "last 3 months"
        )
        assert 0.6 < report[0]["distribution_shift"] <= 0.67
        assert cm.drift_report(None, later) == []

    def test_from_file(self, tmp_path):
        path = tmp_path / "m.json"
        path.write_text(
            json.dumps(
                {
                    "_meta": {"version": "x"},
                    "concepts": {
                        "a": {
                            "concept": "A",
                            "windows": [
                                {"label": "last 1 year", "count": 1, "share": 1}
                            ],
                        }
                    },
                }
            )
        )
        mem = cm.CadenceMemory.from_file(str(path))
        assert mem.version == "x" and len(mem) == 1


# ---------------------------------------------------------------------------
# Prompt block and pipeline wiring
# ---------------------------------------------------------------------------


class TestPromptBlockAndPipeline:
    def test_v3_template_is_standalone(self):
        # Frozen v3 copy: the vocabulary-list slot of v2 does not exist, the
        # canonical section does, and the data slots are the only variables.
        assert "{temporal_matching_block}" not in CONTEXTUAL_ENVIRONMENT_PROMPT_V3
        assert "selected_id" not in CONTEXTUAL_ENVIRONMENT_PROMPT_V3
        for slot in (
            "{concept_count}",
            "{record_type_matching_block}",
            "{temporal_units}",
            "{temporal_menu}",
            "{temporal_shortlist}",
            "{temporal_examples}",
            "{original_query}",
            "{expanded_query}",
            "{intents_json}",
            "{concepts_json}",
        ):
            assert slot in CONTEXTUAL_ENVIRONMENT_PROMPT_V3, slot

    def test_renderers_render_only_what_exists(self):
        assert "day: 1, 30" in render_menu({"day": [1, 30], "year": [1]})
        assert render_menu({}) == "  (none available)"
        assert render_shortlist([]) == "" and render_examples([]) == ""
        assert "- Past Year" in render_shortlist(["Past Year"])
        assert "HbA1c: last 6 months" in render_examples(["HbA1c: last 6 months (80x)"])

    def test_v3_prompt_builder_fills_every_slot(self):
        prompt = build_contextual_environment_prompt_v3(
            original_query="q",
            expanded_query="eq",
            intents_json="[]",
            concepts_json="[]",
            concept_count=0,
            units=["day", "year"],
            menu={"day": [1, 30], "year": [1]},
            shortlist=[],
            examples=[],
        )
        assert "day: 1, 30" in prompt and "year: 1" in prompt
        assert "TERMINOLOGY ENTRIES" not in prompt and "INFERRED BEFORE" not in prompt
        assert 'signal "<its Section B qualifier>"' in prompt
        prompt = build_contextual_environment_prompt_v3(
            original_query="q",
            expanded_query="eq",
            intents_json="[]",
            concepts_json="[]",
            concept_count=0,
            units=["day"],
            menu={},
            shortlist=["Past Year"],
            examples=["HbA1c: last 6 months (80x)"],
        )
        assert (
            "(none available)" in prompt
            and "- Past Year" in prompt
            and "HbA1c: last 6 months" in prompt
        )

    def _run(
        self,
        pipeline,
        mode,
        query="elevated psa in last 3 years",
        context_json=None,
        shadow=False,
    ):
        from tests.test_pipeline_v2 import (
            USAGE,
            _as_async_mock,
            _mock_call_by_step,
        )

        prompts: dict[str, tuple[str, str | None]] = {}
        inner = _mock_call_by_step()

        def cap(prompt, *a, **k):
            step = k.get("step_name")
            prompts.setdefault(step, []).append(
                (
                    prompt,
                    k.get("response_schema").__name__
                    if k.get("response_schema")
                    else None,
                )
            )
            if step == "contextual_environment" and context_json is not None:
                return context_json, USAGE
            return inner(prompt, *a, **k)

        with patch.object(
            pipeline, "_call_model_async", side_effect=_as_async_mock(cap)
        ):
            result = asyncio.run(
                pipeline.run_v2_async(
                    query,
                    enable_retrieval_signals=True,
                    temporal_mode=mode,
                    temporal_shadow=shadow,
                )
            )
        return result, prompts

    def test_canonical_prompt_drops_vocabulary_and_is_much_smaller(
        self, vocab, index, memory
    ):
        lg = MagicMock()
        p = _pipeline(vocab, index, memory, logger=lg)
        _, legacy = self._run(p, TEMPORAL_MODE_VOCAB_LIST)
        _, canon = self._run(p, TEMPORAL_MODE_CANONICAL)
        legacy_prompt, legacy_schema = legacy["contextual_environment"][0]
        canon_prompt, canon_schema = canon["contextual_environment"][0]
        assert (
            legacy_schema == "ContextualEnvironmentOutput"
            and canon_schema == "ContextualEnvironmentOutputCanonicalSchema"
        )
        assert (
            "TEMPORAL CONCEPTS (id, name)" in legacy_prompt
            and "TEMPORAL CONCEPTS (id, name)" not in canon_prompt
        )
        assert "CANONICAL WINDOW" in canon_prompt and "CODEABLE WINDOWS" in canon_prompt
        # The shortlist appears because the query states a span; no example (no known concept).
        assert "TERMINOLOGY ENTRIES CLOSE TO THIS QUERY" in canon_prompt
        assert "INFERRED BEFORE" not in canon_prompt
        # No vocabulary name that only the list would carry.
        assert "Every five minutes" not in canon_prompt
        assert len(canon_prompt) < len(legacy_prompt)

    def test_canonical_prompt_size_on_production_vocabulary(self):
        if not os.path.exists(VOCAB_FILE) or os.path.getsize(VOCAB_FILE) == 0:
            pytest.skip("production vocabulary file not bundled")
        with open(VOCAB_FILE, encoding="utf-8") as f:
            tv = TemporalVocab(json.load(f))  # from_file is mocked in conftest
        p = _pipeline(tv, TemporalIndex.from_vocab(tv))
        _, legacy = self._run(p, TEMPORAL_MODE_VOCAB_LIST)
        _, canon = self._run(p, TEMPORAL_MODE_CANONICAL)
        legacy_len = len(legacy["contextual_environment"][0][0])
        canon_len = len(canon["contextual_environment"][0][0])
        assert legacy_len > 60_000
        assert canon_len < 0.3 * legacy_len

    def test_canonical_run_resolves_and_logs_inferences(self, vocab, index, memory):
        from tests.test_pipeline_v2 import CONTEXT_JSON

        ctx = json.loads(CONTEXT_JSON)
        for it in ctx["temporal_by_intent"]:
            for c in it["candidates"]:
                for e in c["temporal_signal"]:
                    e.pop("selected_id", None)
                    e.pop("selected_name", None)
                    if "three years" in e["signal"]:
                        e.update(
                            {
                                "window": {
                                    "relation": "last",
                                    "value": 3,
                                    "unit": "year",
                                },
                                "basis": "explicit",
                                "rationale": "stated in query",
                            }
                        )
                    else:
                        e.update(
                            {
                                "window": {
                                    "relation": "last",
                                    "value": 6,
                                    "unit": "month",
                                },
                                "basis": "inferred",
                                "rationale": "lab cadence",
                            }
                        )
        lg = MagicMock()
        p = _pipeline(vocab, index, memory, logger=lg)
        result, _prompts = self._run(
            p, TEMPORAL_MODE_CANONICAL, context_json=json.dumps(ctx)
        )
        by = {
            fc["candidate"]: fc["retrieval_signals"]["temporal"]
            for fc in result["intents"][0]["final_candidates"]
        }
        assert (
            by["elevated PSA levels"][0]["codes"] == "C_3Y"
            and by["elevated PSA levels"][0]["basis"] == "explicit"
        )
        assert {t["codes"] for t in by["prostate-specific antigen"]} == {"C_6M_B"}
        events = [
            c.kwargs["structured_data"]
            for c in lg.log_struct.call_args_list
            if c.kwargs.get("message") == "Temporal inference"
        ]
        # one event per resolved window: PSA (one 6-month entry) + the second candidate
        assert len(events) == 2
        assert events[0]["temporal_mode"] == TEMPORAL_MODE_CANONICAL
        assert (
            events[0]["window_label"] == "last 3 years"
            and events[0]["index_version"] == index.vocab_hash
        )
        assert events[0]["memory_version"] == "2026-09-23T00:00:00Z"
        assert "shadow_contextual_environment" not in result["usage_metadata"]

    def test_shadow_mode_runs_legacy_and_logs_agreement(self, vocab, index):
        lg = MagicMock()
        p = _pipeline(vocab, index, logger=lg)
        result, prompts = self._run(p, TEMPORAL_MODE_CANONICAL, shadow=True)
        assert len(prompts["contextual_environment"]) == 2
        schemas = {s for _, s in prompts["contextual_environment"]}
        assert schemas == {
            "ContextualEnvironmentOutputCanonicalSchema",
            "ContextualEnvironmentOutput",
        }
        comparisons = [
            c.kwargs["structured_data"]
            for c in lg.log_struct.call_args_list
            if c.kwargs.get("message") == "Temporal shadow comparison"
        ]
        assert comparisons and {"agreement", "canonical", "legacy", "candidate"} <= set(
            comparisons[0]
        )
        assert "shadow_contextual_environment" not in result["usage_metadata"]
        assert any(
            c.kwargs.get("message") == "Temporal shadow usage"
            for c in lg.log_struct.call_args_list
        )

    def test_canonical_mode_without_index_is_an_llm_error(self, vocab):
        from app.exceptions import LLMError

        p = _pipeline(vocab, index=None)
        with pytest.raises(LLMError):
            asyncio.run(
                p.run_v2_async(
                    "q",
                    enable_retrieval_signals=True,
                    temporal_mode=TEMPORAL_MODE_CANONICAL,
                )
            )

    def test_memory_examples_reach_the_prompt(self, vocab, index, memory):
        from tests.test_pipeline_v2 import INTENT_JSON  # noqa: F401

        p = _pipeline(vocab, index, memory)
        intents = [
            {
                "intent_title": "Diabetes",
                "sub_natures": [{"atomic_concepts": ["HbA1c"]}],
            }
        ]
        slots = p._canonical_temporal_slots(
            "patient with diabetes", "diabetes mellitus", intents
        )
        assert len(slots["examples"]) == 1
        assert slots["examples"][0].startswith("HbA1c: last 6 months (80x)")
        assert slots["shortlist"] == []  # no span in the query -> no shortlist
        assert slots["units"] and slots["menu"]


# ---------------------------------------------------------------------------
# dtree projections
# ---------------------------------------------------------------------------


def _v2_result_canonical():
    def t(name, cui, formula, basis, window, rationale="r"):
        return {
            "time_window": name,
            "codes": cui,
            "formula": formula,
            "basis": basis,
            "rationale": rationale,
            "window": window,
            "resolution": "vocabulary",
        }

    hba1c = [
        t(
            "Past 6 Months",
            "C_6M_A",
            ["REF_POINT", "REF_POINT - 6M"],
            "inferred",
            {"relation": "last", "value": 6, "unit": "month"},
        ),
        t(
            "6 months",
            "C_6M_B",
            ["REF_POINT", "REF_POINT - 6M"],
            "inferred",
            {"relation": "last", "value": 6, "unit": "month"},
        ),
    ]
    dx = [
        t(
            "In past 3 years",
            "C_3Y",
            ["REF_POINT", "REF_POINT - 3Y"],
            "inferred",
            {"relation": "last", "value": 20, "unit": "year"},
        )
    ]
    return {
        "intents": [
            {
                "intent_title": "Diabetes",
                "retrieval_signals": {
                    "record_types": ["progress_note"],
                    "temporal": hba1c + dx,
                },
                "final_candidates": [
                    {"candidate": "HbA1c", "retrieval_signals": {"temporal": hba1c}},
                    {
                        "candidate": "diabetes diagnosis",
                        "retrieval_signals": {"temporal": dx},
                    },
                ],
            }
        ]
    }


class TestProjections:
    def test_v1_projection_applies_unchanged_to_canonical_results(self, vocab):
        # The response is /v1's: first non-default window in pipeline order,
        # codes expanded by vocabulary name; the canonical audit fields never
        # reach the projected block.
        res = _v2_result_canonical()
        primary = project_temporal(res, vocab)
        assert primary == {
            "name": "Past 6 Months",
            "formula": ["REF_POINT", "REF_POINT - 6M"],
            "coding": [{"system": "UMLS", "code": "C_6M_A"}],
        }
        assert set(primary) == {"name", "formula", "coding"}


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


async def _mock_v3(text: str, **kwargs):
    res = _v2_result_canonical()
    res["intents"][0].update(
        {
            "description": "d",
            "nature": "[Chronic]",
            "sub_natures": [{"category_path": "x", "atomic_concepts": ["HbA1c"]}],
            "final_queries": ["HbA1c"],
        }
    )
    res.update(
        {
            "original_query": text,
            "expanded_query": text,
            "representative_terms": ["Diabetes Mellitus"],
            "total_intents_detected": 1,
            "record_type_matches": [],
            "timestamp": "2025-01-01T10:00:00Z",
            "processing_time_seconds": 1.0,
            "usage_metadata": {
                k: {
                    "prompt_token_count": 1,
                    "candidates_token_count": 1,
                    "total_token_count": 2,
                    "thinking_token_count": 0,
                }
                for k in (
                    "query_expansion",
                    "intent_extraction",
                    "representative_terms",
                    "contextual_environment",
                )
            },
        }
    )
    return res


@pytest.fixture
def cluster_ready(monkeypatch, index):
    """Cluster configured (fake resolver) and a real temporal index on the app
    pipeline (the test app has no temporal vocabulary loaded)."""
    from app.utils import context_lonic_document_cluster as cluster

    monkeypatch.setattr(
        config, "CLUSTER_SELECTION_URL", "https://cluster.example/select"
    )
    monkeypatch.setattr(config, "CLUSTER_BQ_DATASET", "ds")
    original = cluster.resolve_record_type_cuis
    with (
        patch(
            "app.utils.context_lonic_document_cluster.resolve_record_type_cuis",
            side_effect=lambda rts, tags, resolver=None, **kw: original(
                rts, tags, lambda n, c: []
            ),
        ),
        patch("app.main.pipeline._temporal_index", index),
        patch(
            "app.main.pipeline._canonical_resolver", CanonicalTemporalResolver(index)
        ),
    ):
        yield


@pytest.mark.usefixtures("cluster_ready")
class TestEndpointV3:
    def test_200_canonical_projection_and_details(self, client):
        with patch("app.main.pipeline.run_v2_async", side_effect=_mock_v3) as run:
            resp = client.post(
                RS_V3,
                json={"queries": [{"id": "q1", "text": ["patient with diabetes"]}]},
            )
        assert resp.status_code == 200
        body = resp.json()
        q = body["output"]["queries"][0]
        # /v1 per-query block, nothing added: same keys, same temporal choice.
        assert set(q) == {"id", "text", "record_types", "temporal", "tags"}
        assert q["temporal"] == {
            "name": "Past 6 Months",
            "formula": ["REF_POINT", "REF_POINT - 6M"],
            "coding": [{"system": "UMLS", "code": "C_6M_A"}],
        }
        assert q["record_types"] == [{"name": "Progress Note", "coding": []}]
        # /v1 envelope: no temporal_mode / temporal_shadow / extra timing keys.
        assert set(body["details"]) == {
            "timing",
            "usage_metadata",
            "version",
            "timestamp",
            "source",
            "model",
            "location",
        }
        assert set(body["details"]["timing"]) == {
            "total_llm_seconds",
            "max_llm_seconds",
            "max_llm_seconds_input_index",
        }
        assert "shadow_contextual_environment" not in body["details"]["usage_metadata"]
        kwargs = run.call_args.kwargs
        assert kwargs["temporal_mode"] == TEMPORAL_MODE_CANONICAL
        # The /v1 prompt (KNOWN RECORD TYPES block included): record-type
        # names are generated exactly as /v1; only their coding differs.
        assert kwargs["include_record_type_matching"] is True

    def test_record_type_json_fallback_when_cluster_has_no_code(self, client):
        from app.utils.concept_vocab import ConceptVocab

        async def _mock(text: str, **kwargs):
            res = await _mock_v3(text, **kwargs)
            res["record_type_matches"] = [
                {"record_type": "progress_note", "selected_names": ["Progress Note"]}
            ]
            return res

        rt_vocab = ConceptVocab({"Progress Note": [{"cui": "C0747978"}]})
        with (
            patch("app.main.pipeline.run_v2_async", side_effect=_mock),
            patch("app.main.record_type_vocab", rt_vocab),
        ):
            resp = client.post(RS_V3, json={"queries": [{"text": ["x"]}]})
            q = resp.json()["output"]["queries"][0]
            # The cluster (fake resolver) coded nothing -> the /v1 vocabulary
            # match for that name is kept, so the record type is never blank.
            assert q["record_types"] == [
                {"name": "Progress Note", "coding": [{"code": "C0747978"}]}
            ]
            with patch("app.main.config.RECORD_TYPE_JSON_FALLBACK", False):
                resp = client.post(RS_V3, json={"queries": [{"text": ["x"]}]})
            assert resp.json()["output"]["queries"][0]["record_types"] == [
                {"name": "Progress Note", "coding": []}
            ]

    def test_v2_response_unchanged(self, client):
        with patch("app.main.pipeline.run_v2_async", side_effect=_mock_v3) as run:
            resp = client.post(
                RS_V2,
                json={"queries": [{"id": "q1", "text": ["patient with diabetes"]}]},
            )
        assert resp.status_code == 200
        body = resp.json()
        q = body["output"]["queries"][0]
        assert "temporal_by_candidate" not in q
        assert (
            "temporal_mode" not in body["details"]
            and "temporal_shadow" not in body["details"]
        )
        # /v2 keeps the legacy first-entry projection.
        assert q["temporal"]["name"] == "Past 6 Months"
        assert run.call_args.kwargs["temporal_mode"] == TEMPORAL_MODE_VOCAB_LIST

    def test_503_when_no_temporal_index(self, client):
        with (
            patch("app.main.pipeline._canonical_resolver", None),
            patch("app.main.pipeline.run_v2_async", side_effect=_mock_v3) as run,
        ):
            resp = client.post(RS_V3, json={"queries": [{"id": "q1", "text": ["x"]}]})
        assert resp.status_code == 503  # before any LLM spend
        run.assert_not_called()
        assert resp.json()["detail"]["error"] == "temporal_index_unavailable"

    def test_action_gate_applies_to_v3(self, client):
        async def _finder(text, **kwargs):
            return {
                "actions": ["admin_response"],
                "is_processable": False,
                "usage_metadata": {},
            }

        with (
            patch("app.main.pipeline.run_v2_async", side_effect=_mock_v3) as run,
            patch("app.main.action_finder.find_actions_async", side_effect=_finder),
        ):
            resp = client.post(
                RS_V3,
                json={
                    "actions": ["information_retrieval"],
                    "queries": [{"text": ["book me"]}],
                },
            )
        assert resp.status_code == 200
        run.assert_not_called()
        assert resp.json()["output"]["queries"][0]["skipped"] is True

    def test_shadow_flag_comes_from_config(self, client, monkeypatch):
        monkeypatch.setattr(config, "TEMPORAL_SHADOW_MODE", True)
        with patch("app.main.pipeline.run_v2_async", side_effect=_mock_v3) as run:
            resp = client.post(RS_V3, json={"queries": [{"text": ["x"]}]})
        assert run.call_args.kwargs["temporal_shadow"] is True
        # Shadow is operational: nothing about it in the response.
        assert "temporal_shadow" not in resp.json()["details"]
        assert (
            "shadow_contextual_environment"
            not in resp.json()["details"]["usage_metadata"]
        )


# ---------------------------------------------------------------------------
# Scripts
# ---------------------------------------------------------------------------


class TestScripts:
    def test_build_cadence_memory_end_to_end(self, tmp_path):
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
        import build_cadence_memory as job

        events = tmp_path / "events.jsonl"
        rows = [
            {
                "jsonPayload": {
                    "message": "Temporal inference",
                    "basis": "inferred",
                    "resolution": "vocabulary",
                    "candidate": "HbA1c",
                    "window": {"relation": "last", "value": 6, "unit": "month"},
                },
                "timestamp": "2026-09-22T01:00:00Z",
            },
            {
                "message": "Temporal inference",
                "basis": "inferred",
                "resolution": "vocabulary",
                "candidate": "HbA1c",
                "window": {"relation": "last", "value": 6, "unit": "month"},
            },
            {"message": "Something else", "basis": "inferred"},
            "not json",
        ]
        events.write_text(
            "\n".join(json.dumps(r) if not isinstance(r, str) else r for r in rows)
        )
        out = tmp_path / "memory.json"
        assert job.main(["--input", str(events), "--output", str(out)]) == 0
        mem = json.loads(out.read_text())
        assert (
            mem["_meta"]["events"] == 2
            and mem["concepts"]["hba1c"]["windows"][0]["count"] == 2
        )
        # Drift gate: a second build where the top window moved beyond the threshold exits 3.
        events2 = tmp_path / "events2.jsonl"
        events2.write_text(
            json.dumps(
                {
                    "message": "Temporal inference",
                    "basis": "inferred",
                    "resolution": "vocabulary",
                    "candidate": "HbA1c",
                    "window": {"relation": "last", "value": 3, "unit": "month"},
                }
            )
        )
        assert (
            job.main(
                [
                    "--input",
                    str(events2),
                    "--output",
                    str(tmp_path / "m2.json"),
                    "--previous",
                    str(out),
                    "--max-shift",
                    "0.5",
                ]
            )
            == 3
        )

    def test_evaluate_temporal_agreement(self, tmp_path):
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
        import evaluate_temporal as ev

        replay = tmp_path / "replay.jsonl"
        rows = [
            {
                "query": "a",
                "version": "v2",
                "temporal": {
                    "coding": [{"code": "C1"}],
                    "formula": ["REF_POINT", "REF_POINT - 1Y"],
                },
                "elapsed_seconds": 2.0,
                "usage_metadata": {"x": {"prompt_token_count": 100}},
            },
            {
                "query": "a",
                "version": "v3",
                "temporal": {
                    "coding": [{"code": "C1"}],
                    "formula": ["REF_POINT", "REF_POINT - 1Y"],
                },
                "elapsed_seconds": 1.0,
                "usage_metadata": {"x": {"prompt_token_count": 20}},
            },
            {
                "query": "b",
                "version": "v2",
                "temporal": {
                    "coding": [{"code": "C2"}],
                    "formula": ["REF_POINT", "REF_POINT - 6M"],
                },
                "elapsed_seconds": 2.0,
                "usage_metadata": {},
            },
            {
                "query": "b",
                "version": "v3",
                "temporal": {
                    "coding": [{"code": "C9"}],
                    "formula": ["REF_POINT", "REF_POINT - 6M"],
                },
                "elapsed_seconds": 1.0,
                "usage_metadata": {},
            },
            {
                "query": "c",
                "version": "v2",
                "temporal": {
                    "coding": [{"code": "C3"}],
                    "formula": ["REF_POINT", "REF_POINT - 1Y"],
                },
                "elapsed_seconds": 2.0,
                "usage_metadata": {},
            },
            {
                "query": "c",
                "version": "v3",
                "temporal": {"coding": [], "formula": ["REF_POINT", "REF_POINT - 5Y"]},
                "elapsed_seconds": 1.0,
                "usage_metadata": {},
            },
        ]
        replay.write_text("\n".join(json.dumps(r) for r in rows))
        report = tmp_path / "report.json"
        code = ev.main(
            ["--replay", str(replay), "--report", str(report), "--min-agreement", "0.5"]
        )
        assert code == 0
        rep = json.loads(report.read_text())
        assert rep["agreement"] == {"cui": 1, "formula": 1, "none": 1}
        assert rep["agreement_rate"] == 0.6667 and len(rep["disagreements"]) == 1
        assert rep["mean_prompt_tokens"] == {"v2": 33.333, "v3": 6.667}
        assert ev.main(["--replay", str(replay), "--min-agreement", "0.9"]) == 2


# ---------------------------------------------------------------------------
# Robustness against the live findings: malformed windows, cluster response
# shapes, case/whitespace differences between the service and BigQuery.
# ---------------------------------------------------------------------------
class TestLiveRobustness:
    def test_invalid_window_costs_the_entry_only(self):
        from app.utils.retrieval_signals import CandidateTemporalCanonical

        c = CandidateTemporalCanonical.model_validate(
            {
                "candidate": "HbA1c",
                "temporal_signal": [
                    {
                        "signal": "recent",
                        "signal_basis": "type_default",
                        "window": {"relation": "last", "value": 0, "unit": "month"},
                        "basis": "Inferred",
                        "rationale": "x",
                    },
                    {
                        "signal": "last 3 months",
                        "signal_basis": "explicit_window",
                        "window": {"relation": "last", "value": 3, "unit": "months"},
                        "basis": "explicit",
                        "rationale": "stated in query",
                    },
                    {"signal": "current", "window": "not an object", "basis": "bogus"},
                ],
            }
        )
        assert [t.signal for t in c.temporal_signal] == [
            "recent",
            "last 3 months",
            "current",
        ]
        assert c.temporal_signal[0].window is None  # value 0 rejected
        assert c.temporal_signal[0].basis == "inferred"
        assert c.temporal_signal[1].window.unit == "month"  # plural normalised
        assert (
            c.temporal_signal[2].window is None and c.temporal_signal[2].basis is None
        )

    def test_canonical_schema_converts_for_gemini_structured_output(self):
        # Contract with the installed google-genai: the schema handed to the
        # model must survive its Schema conversion (numeric constraints such as
        # exclusiveMinimum are rejected there and would fail every /v3 call).
        from google import genai
        from google.genai import _transformers as transformers

        from app.utils.retrieval_signals import (
            ContextualEnvironmentOutput,
            ContextualEnvironmentOutputCanonical,
            ContextualEnvironmentOutputCanonicalSchema,
        )

        client = genai.Client(
            vertexai=True, project="p", location="us-central1", credentials=MagicMock()
        )
        from app.action_finder import ActionFinderOutput
        from app.mod_requests import (
            IntentExtractionResponseV2,
            QueryExpansionOutput,
            RepresentativeTermsOutput,
        )
        from app.utils.record_type_fallback import RecordTypeFallbackOutput

        # Every model the service hands to Gemini as a response schema.
        for model in (
            ContextualEnvironmentOutput,
            ContextualEnvironmentOutputCanonical,
            ContextualEnvironmentOutputCanonicalSchema,
            ActionFinderOutput,
            IntentExtractionResponseV2,
            QueryExpansionOutput,
            RepresentativeTermsOutput,
            RecordTypeFallbackOutput,
        ):
            converted = transformers.t_schema(client, model).model_dump(
                exclude_none=True
            )
            assert converted["properties"], model.__name__
        strict = transformers.t_schema(
            client, ContextualEnvironmentOutputCanonicalSchema
        ).model_dump(exclude_none=True)
        entry = strict["properties"]["temporal_by_intent"]["items"]["properties"][
            "candidates"
        ]["items"]["properties"]["temporal_signal"]["items"]
        assert set(entry["required"]) == {
            "signal",
            "signal_basis",
            "window",
            "basis",
            "rationale",
        }

    def test_canonical_schema_requires_window_basis_rationale(self):
        from app.utils.retrieval_signals import (
            ContextualEnvironmentOutputCanonicalSchema,
        )

        sch = ContextualEnvironmentOutputCanonicalSchema.model_json_schema()
        entry = sch["$defs"]["TemporalMatchCanonicalSchema"]
        assert set(entry["required"]) == {
            "signal",
            "signal_basis",
            "window",
            "basis",
            "rationale",
        }
        assert entry["properties"]["basis"]["enum"] == ["explicit", "inferred"]
        assert sch["$defs"]["CanonicalWindow"]["required"] == [
            "relation",
            "value",
            "unit",
        ]

    def test_member_names_key_is_configurable(self):
        from app.utils import context_lonic_document_cluster as dc

        resp = {"results": [{"doc_name": "Progress Note", "member_name": ""}]}
        assert dc.extract_member_names(resp) == []
        assert dc.extract_member_names(resp, keys=("doc_name",)) == ["Progress Note"]
        with patch.object(dc.config, "CLUSTER_MEMBER_NAME_KEYS", "doc_name, name"):
            assert dc.extract_member_names(resp) == ["Progress Note"]

    def test_bigquery_match_is_case_and_whitespace_insensitive(self):
        from app.utils import context_lonic_document_cluster as dc

        rows = [
            {
                "document_class_cui": "C0747978",
                "document_class": "Progress note",
                "possible_document_name": "PROGRESS  NOTE ",
            }
        ]
        with (
            patch.object(dc, "get_probable_documents", return_value=["progress note"]),
            patch.object(dc, "get_document_classes", return_value=rows) as bq,
        ):
            out = dc.resolve_record_type_cuis([{"name": "Progress Note", "coding": []}])
        assert out == [{"name": "Progress Note", "coding": [{"code": "C0747978"}]}]
        # the member document the service named, then the label itself
        assert bq.call_args.args[0] == ["progress note", "Progress Note"]

    def test_bigquery_query_normalises_both_sides(self):
        from app.utils import context_lonic_document_cluster as dc

        captured: dict = {}

        class _Job:
            def result(self):
                return []

        class _Client:
            def query(self, query, job_config=None):
                captured["query"] = query
                captured["params"] = job_config.query_parameters
                return _Job()

        with (
            patch.object(dc, "get_bq_client", return_value=_Client()),
            patch.object(dc, "_require_configured"),
            patch.object(dc, "_log") as log,
            patch.object(dc.config, "CLUSTER_DOCUMENT_CATALOG_TTL_SECONDS", 0),
        ):
            assert (
                dc.get_document_classes([" Progress Note", "PROGRESS NOTE", "H&P"])
                == []
            )
        assert (
            "LOWER(TRIM(m.MEMBER_NAME)) IN UNNEST(@document_names)" in captured["query"]
        )
        assert captured["params"][0].values == ["h&p", "progress note"]
        # nothing matched -> the unmatched names are logged for diagnosis
        assert log.call_args.args[0].startswith("Document-cluster BigQuery lookup")
        assert log.call_args.args[1]["unmatched_sample"] == [
            "Progress Note",
            "PROGRESS NOTE",
            "H&P",
        ]

    def test_empty_cluster_response_is_logged_with_its_shape(self):
        from app.utils import context_lonic_document_cluster as dc

        with (
            patch.object(
                dc, "call_cluster_selection", return_value={"clusters": [{"id": 1}]}
            ),
            patch.object(dc, "_log") as log,
        ):
            assert dc.get_probable_documents("Progress Note") == []
        assert log.call_args.args[0] == "Cluster selection returned no member names"
        assert log.call_args.args[1]["response_top_level_keys"] == ["clusters"]

    def test_document_catalog_matches_exact_folded_and_similar(self):
        from app.utils import context_lonic_document_cluster as dc

        def row(cui, cls, name):
            return {
                "document_class_cui": cui,
                "document_class": cls,
                "possible_document_name": name,
            }

        cat = dc.DocumentCatalog(
            [
                row("C10", "Imaging", "Chest X-ray report"),
                row("C20", "Discharge", "Discharge Summary"),
                row("C30", "Consult", "Consultation Note | Outpatient"),
            ]
        )
        assert len(cat) == 3
        # exact (case / whitespace), folded (punctuation), similar (alias / order)
        assert cat.match("  chest x-ray REPORT ", 0.75)[1] == "exact"
        assert cat.match("Consultation Note - Outpatient", 0.75)[1] == "folded"
        # reordered qualifier: every token matches -> the token stage
        idxs, how, score = cat.match("Outpatient consultation note", 0.75)
        assert how == "tokens" and score == 1.0 and idxs == [2]
        # abbreviation / acronym / plural: the token stage, no data authored
        assert cat.match("Consult. note outpt", 0.75)[1] == "tokens"
        assert cat.match("chest xray reports", 0.75)[1] == "similar"  # x-ray vs xray
        assert cat.match("H&P", 0.75)[1] == "none"  # no such document in this catalog
        assert cat.match("Dischg summ", 0.75)[1] == "tokens"
        assert cat.match("DS", 0.75)[1] == "tokens"  # acronym of "Discharge Summary"
        # a lone generic token never matches a longer name
        assert cat.match("note", 0.75)[1] == "none"
        assert cat.match("summary", 0.75)[1] == "none"
        assert cat.match("Pathology report", 0.75)[1] == "none"
        # spelling variant: the n-gram stage
        idxs, how, score = cat.match("Discharge Sumary", 0.75, min_token_score=0)
        assert how == "similar" and score >= 0.75 and idxs == [1]
        assert cat.match("Discharge Sumary", 0.0, min_token_score=0)[1] == "none"
        with patch.object(dc, "_log") as log:
            out = cat.lookup(
                [
                    "discharge summary",
                    "Outpatient consultation note",
                    "Pathology report",
                ],
                0.75,
            )
        # rows carry the REQUESTED name so callers group by what they sent
        assert [
            (r["possible_document_name"], r["document_class_cui"]) for r in out
        ] == [
            ("discharge summary", "C20"),
            ("Outpatient consultation note", "C30"),
        ]
        messages = [c.args[0] for c in log.call_args_list]
        assert "Document-cluster member names matched by similarity" in messages
        assert (
            "Document-cluster member names not found in the document catalog"
            in messages
        )

    def test_document_catalog_is_loaded_once_and_used_for_lookups(self):
        from app.utils import context_lonic_document_cluster as dc

        client = MagicMock()
        client.query.return_value.result.return_value = [
            {
                "document_class_cui": "C10",
                "document_class": "Imaging",
                "possible_document_name": "Chest X-ray report",
            }
        ]
        with (
            patch.object(dc, "get_bq_client", return_value=client),
            patch.object(dc, "_require_configured"),
            patch.object(dc, "_log"),
        ):
            assert dc.get_document_classes(["CHEST X-RAY REPORT"]) == [
                {
                    "document_class_cui": "C10",
                    "document_class": "Imaging",
                    "possible_document_name": "CHEST X-RAY REPORT",
                }
            ]
            assert (
                dc.get_document_classes(["chest xray report"])[0]["document_class_cui"]
                == "C10"
            )
        # one catalog load served both lookups; no per-request query
        assert client.query.call_count == 1
        assert "IN UNNEST" not in client.query.call_args.args[0]

    def test_document_catalog_failure_falls_back_to_per_request_query(self):
        from app.utils import context_lonic_document_cluster as dc

        calls: list[str] = []

        class _Job:
            def __init__(self, rows):
                self._rows = rows

            def result(self):
                return self._rows

        class _Client:
            def query(self, query, job_config=None):
                calls.append(query)
                if "IN UNNEST" not in query:
                    raise RuntimeError("catalog too large")
                return _Job(
                    [
                        {
                            "document_class_cui": "C10",
                            "document_class": "Imaging",
                            "possible_document_name": "Chest X-ray report",
                        }
                    ]
                )

        with (
            patch.object(dc, "get_bq_client", return_value=_Client()),
            patch.object(dc, "_require_configured"),
            patch.object(dc, "_log") as log,
        ):
            out = dc.get_document_classes(["chest x-ray report"])
        assert out[0]["document_class_cui"] == "C10"
        assert len(calls) == 2 and "IN UNNEST" in calls[1]
        assert log.call_args_list[0].args[0].startswith("Document catalog load failed")

    def test_record_type_codes_through_its_own_label_and_aliases(self):
        """'PRES. record' style labels: the service is asked for the label AND
        the vocabulary labels the model matched it to; and the label itself is
        matched in the catalog (abbreviations), so a record type codes even
        when the service returns nothing usable."""
        from app.utils import context_lonic_document_cluster as dc

        cat = dc.DocumentCatalog(
            [
                {
                    "document_class_cui": "C_RX",
                    "document_class": "Prescription",
                    "possible_document_name": "Prescription Record",
                }
            ]
        )
        asked: list[str] = []

        def probable(text, context):
            asked.append(text)
            return []  # the service found nothing for any spelling

        with (
            patch.object(dc, "get_probable_documents", side_effect=probable),
            patch.object(dc, "get_document_catalog", return_value=cat),
            patch.object(dc, "_log"),
        ):
            out = dc.resolve_record_type_cuis(
                [{"name": "Pres. Record", "coding": []}],
                tags=[],
                aliases={"pres. record": ["Prescription Record", "Rx Record"]},
            )
        assert asked == ["Pres. Record", "Prescription Record", "Rx Record"]
        assert out == [{"name": "Pres. Record", "coding": [{"code": "C_RX"}]}]

        # the cap on alias texts is a setting
        asked.clear()
        with (
            patch.object(dc, "get_probable_documents", side_effect=probable),
            patch.object(dc, "get_document_catalog", return_value=cat),
            patch.object(dc, "_log"),
            patch.object(dc.config, "CLUSTER_ALIAS_TEXTS_MAX", 0),
        ):
            dc.resolve_record_type_cuis(
                ["Pres. Record"], aliases={"pres. record": ["Prescription Record"]}
            )
        assert asked == ["Pres. Record"]

    def test_projection_passes_the_models_vocabulary_matches_as_aliases(self):
        from app.utils import context_lonic_document_cluster as dc
        from app.utils.dtree_signals import project_record_types_cluster

        v2_result = {
            "intents": [
                {"retrieval_signals": {"record_types": ["prescription_record"]}}
            ],
            "record_type_matches": [
                {
                    "record_type": "prescription_record",
                    "selected_names": ["Prescription Record", "Medication Order"],
                }
            ],
        }
        with patch.object(dc, "resolve_record_type_cuis", return_value=[]) as res:
            project_record_types_cluster(v2_result, tags=[])
        assert res.call_args.kwargs["aliases"] == {
            "prescription record": ["Prescription Record", "Medication Order"]
        }


class TestNoBlankRecordTypes:
    """A record type must never come back with an empty coding while any
    stage can code it: cluster members -> own label / aliases in the catalog
    -> document-class label match -> /v1 vocabulary -> model against the
    cluster's document classes."""

    @staticmethod
    def _catalog():
        from app.utils import context_lonic_document_cluster as dc

        return dc.DocumentCatalog(
            [
                {
                    "document_class_cui": "C_RX",
                    "document_class": "Medication Order",
                    "possible_document_name": "Prescription Record",
                },
                {
                    "document_class_cui": "C_IMG",
                    "document_class": "Imaging Report",
                    "possible_document_name": "Chest X-ray report",
                },
            ]
        )

    def test_class_label_stage_codes_a_label_that_names_a_class(self):
        from app.utils import context_lonic_document_cluster as dc

        cat = self._catalog()
        assert cat.class_labels() == ["Medication Order", "Imaging Report"]
        assert cat.cuis_for_class_label("medication order") == ["C_RX"]
        assert cat.match_class("imaging  report", 0.75)[:2] == (["C_IMG"], "class")
        assert cat.match_class("Imaging Reports", 0.75)[:2] == (
            ["C_IMG"],
            "class_tokens",
        )
        assert cat.match_class("Med. Order", 0.75)[:2] == (["C_RX"], "class_tokens")
        assert cat.match_class("Pathology Report", 0.75)[0] == []
        with (
            patch.object(dc, "get_probable_documents", return_value=[]),
            patch.object(dc, "get_document_catalog", return_value=cat),
            patch.object(dc, "_log"),
        ):
            dc._catalog_state["catalog"] = (
                cat  # what get_document_classes leaves behind
            )
            out = dc.resolve_record_type_cuis(["Imaging report"], tags=[])
        assert out == [{"name": "Imaging report", "coding": [{"code": "C_IMG"}]}]

    def test_model_fallback_resolves_labels_to_class_cuis(self):
        from app.utils.record_type_fallback import (
            RecordTypeFallbackOutput,
            resolve_with_model,
            shortlist_labels,
        )

        cat = self._catalog()
        seen: dict = {}

        async def call(prompt, **kwargs):
            seen["prompt"] = prompt
            seen["kwargs"] = kwargs
            return (
                json.dumps(
                    {
                        "matches": [
                            {
                                "record_type": "PRES. record",
                                "selected_labels": ["Medication Order"],
                                "reasoning": "prescription = medication order",
                            },
                            {"record_type": "Unknown thing", "selected_labels": []},
                            {
                                "record_type": "Made up",
                                "selected_labels": ["Not a class"],
                            },
                        ]
                    }
                ),
                {"prompt_token_count": 10, "total_token_count": 12},
            )

        coded, usage = asyncio.run(
            resolve_with_model(
                [
                    {"name": "PRES. record", "aliases": ["Prescription Record"]},
                    {"name": "Unknown thing", "aliases": []},
                    {"name": "Made up", "aliases": []},
                ],
                cat,
                "patient on metformin",
                call,
                model_name="m",
                location="us",
                max_labels=400,
            )
        )
        assert coded == {
            "pres. record": ["C_RX"]
        }  # labels resolve through the catalog only
        assert usage["total_token_count"] == 12
        assert seen["kwargs"]["step_name"] == "record_type_fallback"
        assert seen["kwargs"]["response_schema"] is RecordTypeFallbackOutput
        assert "- PRES. record (also: Prescription Record)" in seen["prompt"]
        assert '"Medication Order"' in seen["prompt"]
        # shortlist keeps the closest labels when the class list is too long
        labels = [f"Class {i}" for i in range(50)] + ["Medication Order"]
        short = shortlist_labels(["med order"], labels, 5)
        assert len(short) == 5 and short[0] == "Medication Order"

    @pytest.mark.usefixtures("cluster_ready")
    def test_endpoint_never_returns_blank_coding_when_a_stage_can_code(self, client):
        from app.utils import context_lonic_document_cluster as dc

        cat = self._catalog()

        async def _mock(text: str, **kwargs):
            res = await _mock_v3(text, **kwargs)
            res["intents"][0]["retrieval_signals"]["record_types"] = ["pres_record"]
            res["record_type_matches"] = []
            return res

        async def model(prompt, **kwargs):
            return (
                json.dumps(
                    {
                        "matches": [
                            {
                                "record_type": "Pres Record",
                                "selected_labels": ["Medication Order"],
                            }
                        ]
                    }
                ),
                {"prompt_token_count": 1, "total_token_count": 2},
            )

        with (
            patch("app.main.pipeline.run_v2_async", side_effect=_mock),
            patch("app.main.pipeline._call_model_async", side_effect=model),
            patch.object(dc, "current_catalog", return_value=cat),
            patch("app.main.document_cluster.current_catalog", return_value=cat),
        ):
            resp = client.post(RS_V3, json={"queries": [{"text": ["x"]}]})
            assert resp.status_code == 200
            q = resp.json()["output"]["queries"][0]
            assert q["record_types"] == [
                {"name": "Pres Record", "coding": [{"code": "C_RX"}]}
            ]
            # the envelope is still /v1's: the fallback call is logged, not returned
            assert set(resp.json()["details"]["usage_metadata"]) == {
                "action_finder",
                "query_expansion",
                "intent_extraction",
                "representative_terms",
                "contextual_environment",
            }
            with patch("app.main.config.RECORD_TYPE_LLM_FALLBACK", False):
                resp = client.post(RS_V3, json={"queries": [{"text": ["x"]}]})
            assert resp.json()["output"]["queries"][0]["record_types"] == [
                {"name": "Pres Record", "coding": []}
            ]


class TestPerCandidateWindows:
    def test_same_qualifier_resolves_per_candidate_not_first_wins(
        self, vocab, resolver
    ):
        """Two candidates both say "current"; one means 6 months, the other 3
        years. Each must keep ITS window; the intent level carries both."""
        ctx = _context()
        cands = ctx["temporal_by_intent"][0]["candidates"]
        by_name = {c["candidate"]: c for c in cands}
        by_name["HbA1c"]["temporal_signal"] = [
            {
                "signal": "current",
                "signal_basis": "type_default",
                "window": {"relation": "last", "value": 6, "unit": "month"},
                "basis": "inferred",
                "rationale": "lab cadence",
            }
        ]
        by_name["metformin"]["temporal_signal"] = [
            {
                "signal": "current",
                "signal_basis": "type_default",
                "window": {"relation": "last", "value": 3, "unit": "year"},
                "basis": "inferred",
                "rationale": "chronic medication",
            }
        ]
        out = assemble_v2_intents(
            _INTENTS,
            context=ContextualEnvironmentOutputCanonical.model_validate(ctx),
            vocab=vocab,
            canonical_resolver=resolver,
        )
        by = {
            fc["candidate"]: fc["retrieval_signals"]["temporal"]
            for fc in out[0]["final_candidates"]
        }
        assert [t["codes"] for t in by["HbA1c"]] == ["C_6M_B"]
        assert [t["codes"] for t in by["metformin"]] == ["C_3Y"]
        assert by["HbA1c"][0]["resolution"] == "vocabulary"
        assert by["metformin"][0]["window"] == {
            "relation": "last",
            "value": 3,
            "unit": "year",
        }
        # intent level: the union, in candidate order; no default anywhere
        intent_codes = [t["codes"] for t in out[0]["retrieval_signals"]["temporal"]]
        assert intent_codes == ["C_6M_B", "C_3Y"]
        assert all(
            t["resolution"] != "default"
            for t in out[0]["retrieval_signals"]["temporal"]
        )
        # and the response's primary temporal is the first candidate's window, not the default
        primary = project_temporal({"intents": out}, vocab)
        assert primary["name"] == "6 months" and primary["coding"] == [
            {"system": "UMLS", "code": "C_6M_B"}
        ]

    def test_reworded_intent_title_echo_does_not_fall_to_default(self, vocab, resolver):
        ctx = _context()
        # the model echoes the title with different punctuation / case
        ctx["temporal_by_intent"][0]["intent_title"] = (
            ctx["temporal_by_intent"][0]["intent_title"].upper() + "."
        )
        for c in ctx["temporal_by_intent"][0]["candidates"]:
            c["candidate"] = c["candidate"].upper()
        out = assemble_v2_intents(
            _INTENTS,
            context=ContextualEnvironmentOutputCanonical.model_validate(ctx),
            vocab=vocab,
            canonical_resolver=resolver,
        )
        temporal = out[0]["final_candidates"][0]["retrieval_signals"]["temporal"]
        assert temporal and all(t["resolution"] != "default" for t in temporal)
