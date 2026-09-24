"""Temporal vocabulary: load name→CUI mappings and resolve LLM temporal matches."""

from __future__ import annotations

import json
import re
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, field_validator


def _coerce_formula(value: Any) -> Any:
    """Vocab JSON uses list[str]; a legacy string, a {start, end} / {from, to}
    object or a list of non-strings are normalised to list[str] so one
    differently-shaped entry never fails the whole vocabulary load."""
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        lo = value.get("start", value.get("from"))
        hi = value.get("end", value.get("to"))
        return [str(t) for t in (lo, hi) if t is not None]
    if isinstance(value, (list, tuple)):
        return [str(t) for t in value if t is not None]
    return [str(value)]


class CodeEntry(BaseModel):
    cui: str
    formula: list[str] | None = None

    @field_validator("formula", mode="before")
    @classmethod
    def _normalize_formula(cls, value: Any) -> Any:
        return _coerce_formula(value)


class TemporalMapping(BaseModel):
    time_window: str | None = None
    codes: str | None = None
    formula: list[str] | None = None

    @field_validator("formula", mode="before")
    @classmethod
    def _normalize_formula(cls, value: Any) -> Any:
        return _coerce_formula(value)


@runtime_checkable
class TemporalResolver(Protocol):
    """Fallback when no vocabulary is loaded: map phrases to TemporalMapping."""

    def resolve(self, signals: list[str | int]) -> list[TemporalMapping]: ...


class NullTemporalResolver:
    """No vocabulary: return each phrase with no match and no codes."""

    def resolve(self, signals: list[str | int]) -> list[TemporalMapping]:
        return [
            TemporalMapping(time_window=None, codes=None, formula=None) for _ in signals
        ]


# Try these names in the vocab file when no explicit temporal signal resolves.
DEFAULT_TEMPORAL_TERMS = ("recent", "present")

# Last-resort default when nothing in the file resolves (temporal is mandatory).
DEFAULT_TEMPORAL_FALLBACK = {
    "time_window": "Recent",
    "codes": "C0332185",
    "formula": ["REF_POINT", "REF_POINT"],
}

# Optional vocabulary key naming the window to use when no signal resolves.
DEFAULT_TEMPORAL_NAME_KEY = "_default_temporal_name"


class TemporalVocab:
    """Temporal concept vocabulary for folded matching in build_context.

    Names are injected into the contextual environment prompt as [id, name]
    transactions; the model picks selected_id inline and codes are resolved here.

    File shape: { name: [ {"cui", "formula"}, ... ] } — first entry wins.
    """

    def __init__(self, name_to_codes: dict[str, list[dict[str, str]]]):
        source = name_to_codes or {}
        # Underscore-prefixed keys are metadata (_comment, _default_temporal_name),
        # not concepts: without this they get an id and reach the prompt as
        # candidate temporal concepts.
        self._map = {k: v for k, v in source.items() if not str(k).startswith("_")}
        self._names = list(self._map.keys())
        self._id_to_name = {f"id_{i + 1}": n for i, n in enumerate(self._names)}
        self._lower_to_name = {n.strip().lower(): n for n in self._names}
        self._folded_to_name = {self._fold(n): n for n in self._names}
        self._declared_default = source.get(DEFAULT_TEMPORAL_NAME_KEY)

    @staticmethod
    def _fold(text: Any) -> str:
        return re.sub(r"[^a-z0-9]+", " ", str(text).strip().lower()).strip()

    @classmethod
    def from_file(cls, path: str) -> TemporalVocab:
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f))

    @classmethod
    def from_gcs(cls, project: str, bucket: str, blob_path: str) -> TemporalVocab:
        from google.cloud import storage

        blob = storage.Client(project=project).bucket(bucket).blob(blob_path)
        return cls(json.loads(blob.download_as_text()))

    def transactions(self) -> list[list[str]]:
        return [[f"id_{i + 1}", n] for i, n in enumerate(self._names)]

    def name_for_id(self, sel_id: Any) -> str | None:
        if sel_id is None or str(sel_id).strip().lower() in ("none", "null", ""):
            return None
        return self._id_to_name.get(str(sel_id).strip())

    def name_from_text(self, text: Any) -> str | None:
        """Resolve echoed concept name to a vocab key (exact, case-, then format-insensitive)."""
        if not isinstance(text, str) or not text.strip():
            return None
        t = text.strip()
        if t in self._map:
            return t
        return self._lower_to_name.get(t.lower()) or self._folded_to_name.get(
            self._fold(t)
        )

    def resolve_name(self, text: Any) -> str | None:
        return self.name_from_text(text)

    def codes_for_name(self, name: str | None) -> list[CodeEntry]:
        if not name:
            return []
        return [CodeEntry.model_validate(e) for e in self._map.get(name, [])]

    def first_name(self) -> str | None:
        """Deterministic first vocabulary name, or None if vocabulary is empty."""
        return self._names[0] if self._names else None

    def default_name(self) -> str | None:
        """Window to use when no temporal signal resolves: the vocabulary's
        declared default, else the conventional DEFAULT_TEMPORAL_TERMS, else
        the first entry."""
        declared = self.name_from_text(self._declared_default)
        if declared:
            return declared
        for term in DEFAULT_TEMPORAL_TERMS:
            name = self.name_from_text(term)
            if name:
                return name
        return self.first_name()

    def primary_for_name(self, name: str | None) -> tuple[str | None, list[str] | None]:
        """First (cui, formula) for a matched name, or (None, None)."""
        entries = self.codes_for_name(name)
        if not entries:
            return None, None
        return entries[0].cui, entries[0].formula
