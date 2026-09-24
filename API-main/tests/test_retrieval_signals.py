"""Unit tests for retrieval-signals post-processing (no LLM calls)."""

import pytest

from app.utils.retrieval_signals import (
    CandidateTemporal,
    SignalsAssembler,
    TemporalExtractionOutput,
    assemble_v2_intents,
)
from app.utils.temporal_vocab import (
    DEFAULT_TEMPORAL_FALLBACK,
    TemporalVocab,
)


@pytest.fixture
def vocab() -> TemporalVocab:
    """Small in-memory vocab for deterministic CUI resolution."""
    return TemporalVocab(
        {
            "In past 3 years": [
                {
                    "cui": "C3843792",
                    "formula": ["REF_POINT", "REF_POINT - 3Y"],
                },
            ],
            "Recent": [
                {"cui": "C0332185", "formula": ["REF_POINT", "REF_POINT - 3Y"]},
            ],
        }
    )


@pytest.fixture
def sample_intents():
    return [
        {
            "intent_title": "Elevated PSA",
            "description": "PSA finding",
            "nature": "[Clinical Finding] / [Diagnostic]",
            "sub_natures": [
                {
                    "category_path": "Lab >> PSA",
                    "atomic_concepts": [
                        "elevated PSA levels",
                        "prostate-specific antigen",
                        "elevated PSA levels",  # duplicate — should dedupe
                    ],
                }
            ],
            "final_queries": ["elevated PSA levels", "prostate-specific antigen"],
        }
    ]


@pytest.fixture
def sample_context():
    return {
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
                "author_roles": ["urologist", "pathologist"],
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
                        "temporal_signal": [],
                    },
                ],
            }
        ],
    }


class TestCandidateTemporalCoerce:
    def test_candidate_temporal_coerce_strips_prefix(self):
        ct = CandidateTemporal.model_validate(
            {
                "candidate": "psa",
                "temporal_signal": [
                    "STATE: recent",
                    {"signal": "RELATIVE: last 3 years", "selected_id": "id_1"},
                    {"signal": "  ", "selected_id": None},
                ],
            }
        )
        assert [m.signal for m in ct.temporal_signal] == [
            "recent",
            "last 3 years",
        ]
        assert ct.temporal_signal[1].selected_id == "id_1"


class TestStructuredSignals:
    def test_structured_signals_resolves_temporal(self, vocab, sample_context):
        assembler = SignalsAssembler(
            context=None,
            temporal=TemporalExtractionOutput.model_validate(
                {"intents": sample_context["temporal_by_intent"]}
            ),
            vocab=vocab,
        )
        raw = {
            "record_types": ["laboratory_report"],
            "author_roles": ["pathologist"],
            "longitudinal_scope": ["screening"],
            "temporal_signal": ["last three years"],
            "content_signals": ["measurement_value"],
            "clinical_setting": ["outpatient"],
        }
        out = assembler.structured_signals(raw)
        assert out["temporal"] == [
            {
                "time_window": "In past 3 years",
                "codes": "C3843792",
                "formula": ["REF_POINT", "REF_POINT - 3Y"],
            }
        ]
        assert out["authors"] == ["pathologist"]
        assert out["clinical_setting"] == ["outpatient"]

    def test_structured_signals_temporal_fallback(self, vocab):
        assembler = SignalsAssembler(
            context=None,
            temporal=TemporalExtractionOutput(intents=[]),
            vocab=vocab,
        )
        # Unmatched phrase → no codes → drop → fall through to DEFAULT_TEMPORAL_TERMS
        # ("recent" resolves to "Recent" in this vocab)
        out = assembler.structured_signals(
            {
                "record_types": [],
                "author_roles": [],
                "longitudinal_scope": [],
                "temporal_signal": ["totally unknown window"],
                "content_signals": [],
                "clinical_setting": [],
            }
        )
        assert len(out["temporal"]) == 1
        assert out["temporal"][0]["codes"] == "C0332185"
        assert out["temporal"][0]["time_window"] == "Recent"
        # assert out["temporal"][0]["formula"] == ["REF_POINT", "REF_POINT - 3Y"]
        assert out["temporal"][0]["formula"] == ["REF_POINT", "REF_POINT"]

    def test_structured_signals_hardcoded_fallback_when_vocab_empty(self):
        empty_vocab = TemporalVocab({})
        assembler = SignalsAssembler(
            context=None,
            temporal=None,
            vocab=empty_vocab,
        )
        out = assembler.structured_signals({"temporal_signal": []})
        assert out["temporal"] == [
            {
                "time_window": DEFAULT_TEMPORAL_FALLBACK["time_window"],
                "codes": DEFAULT_TEMPORAL_FALLBACK["codes"],
                "formula": DEFAULT_TEMPORAL_FALLBACK["formula"],
            }
        ]


class TestTemporalVocabFormula:
    def test_codes_for_name_accepts_list_formula_from_production_vocab(self):
        vocab = TemporalVocab(
            {
                "In past 3 years": [
                    {
                        "cui": "C3843792",
                        "formula": ["REF_POINT", "REF_POINT - 3Y"],
                    },
                ],
            }
        )
        entries = vocab.codes_for_name("In past 3 years")
        assert len(entries) == 1
        assert entries[0].formula == ["REF_POINT", "REF_POINT - 3Y"]

    def test_codes_for_name_coerces_legacy_string_formula(self):
        vocab = TemporalVocab(
            {
                "In past 3 years": [
                    {"cui": "C3843792", "formula": "REF_POINT - 3Y"},
                ],
            }
        )
        entries = vocab.codes_for_name("In past 3 years")
        assert entries[0].formula == ["REF_POINT - 3Y"]


class TestFieldRenames:
    def test_field_renames(self, vocab, sample_intents, sample_context):
        out = assemble_v2_intents(
            intents=sample_intents,
            context=sample_context,
            vocab=vocab,
        )
        fc = out[0]["final_candidates"][0]
        signals = fc["retrieval_signals"]
        assert "authors" in signals
        assert "author_roles" not in signals
        assert "clinical_setting" in signals
        assert "clinical_settings" not in signals
        assert signals["authors"] == ["pathologist"]
        assert signals["clinical_setting"] == ["outpatient"]


class TestBuildFinalCandidates:
    def test_build_final_candidates_ids(self, vocab, sample_intents, sample_context):
        out = assemble_v2_intents(
            intents=sample_intents,
            context=sample_context,
            vocab=vocab,
        )
        candidates = out[0]["final_candidates"]
        assert [c["candidate_id"] for c in candidates] == ["fc_001", "fc_002"]
        assert [c["candidate"] for c in candidates] == [
            "elevated PSA levels",
            "prostate-specific antigen",
        ]

    def test_build_final_candidates_global_dedupe_across_intents(self, vocab):
        intents = [
            {
                "intent_title": "Intent A",
                "description": "A",
                "nature": "N",
                "sub_natures": [
                    {
                        "category_path": "P",
                        "atomic_concepts": ["shared concept", "only a"],
                    }
                ],
                "final_queries": ["shared concept"],
            },
            {
                "intent_title": "Intent B",
                "description": "B",
                "nature": "N",
                "sub_natures": [
                    {
                        "category_path": "P",
                        "atomic_concepts": ["Shared Concept", "only b"],
                    }
                ],
                "final_queries": ["only b"],
            },
        ]
        out = assemble_v2_intents(intents=intents, context=None, vocab=vocab)
        all_ids = [
            c["candidate_id"] for intent in out for c in intent["final_candidates"]
        ]
        all_names = [
            c["candidate"].lower() for intent in out for c in intent["final_candidates"]
        ]
        assert all_ids == ["fc_001", "fc_002", "fc_003"]
        assert all_names.count("shared concept") == 1


class TestMergeSignals:
    def test_merge_signals_dedupes(self, vocab, sample_intents, sample_context):
        out = assemble_v2_intents(
            intents=sample_intents,
            context=sample_context,
            vocab=vocab,
        )
        intent_signals = out[0]["retrieval_signals"]
        # Union of both candidates' record_types, order-preserving, no dupes
        assert intent_signals["record_types"] == [
            "laboratory_report",
            "order_entry",
        ]
        assert intent_signals["authors"] == ["pathologist", "urologist"]
        assert intent_signals["authors"].count("pathologist") == 1


class TestTemporalForCandidate:
    def test_temporal_for_candidate_own_vs_union(self, vocab, sample_context):
        temporal = TemporalExtractionOutput.model_validate(
            {
                "intents": [
                    {
                        "intent_title": "Elevated PSA",
                        "candidates": [
                            {
                                "candidate": "elevated PSA levels",
                                "temporal_signal": [{"signal": "last three years"}],
                            },
                            {
                                "candidate": "prostate-specific antigen",
                                "temporal_signal": [{"signal": "recent"}],
                            },
                        ],
                    }
                ]
            }
        )
        assembler = SignalsAssembler(context=None, temporal=temporal, vocab=vocab)
        # Own temporal preferred
        own = assembler._temporal_for_candidate("Elevated PSA", "elevated PSA levels")
        assert own == ["last three years"]

        # Candidate with empty own list falls back to intent union
        empty_own_temporal = TemporalExtractionOutput.model_validate(
            {
                "intents": [
                    {
                        "intent_title": "Elevated PSA",
                        "candidates": [
                            {
                                "candidate": "elevated PSA levels",
                                "temporal_signal": [{"signal": "last three years"}],
                            },
                            {
                                "candidate": "prostate-specific antigen",
                                "temporal_signal": [],
                            },
                        ],
                    }
                ]
            }
        )
        assembler2 = SignalsAssembler(
            context=None, temporal=empty_own_temporal, vocab=vocab
        )
        fallback = assembler2._temporal_for_candidate(
            "Elevated PSA", "prostate-specific antigen"
        )
        assert fallback == ["last three years"]


class TestAssembleV2Intents:
    def test_preserves_final_queries(self, vocab, sample_intents, sample_context):
        out = assemble_v2_intents(
            intents=sample_intents,
            context=sample_context,
            vocab=vocab,
        )
        assert out[0]["final_queries"] == sample_intents[0]["final_queries"]
        assert "final_candidates" in out[0]
        assert "retrieval_signals" in out[0]
