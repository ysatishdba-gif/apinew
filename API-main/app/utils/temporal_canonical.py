"""Canonical temporal windows: the model emits a structured window, code
resolves it.

In canonical temporal mode the contextual-environment call no longer receives
the vocabulary list. Every temporal_signal entry carries, besides the
``signal`` wording and ``signal_basis`` it already had, a ``window``:

    {"relation": "last", "value": 3, "unit": "year"}
    {"relation": "range", "value": 6, "unit": "month", "to_value": 12, "to_unit": "month"}

``CanonicalTemporalResolver`` turns that into the same ``TemporalMapping``
objects the assembler has always produced (``time_window`` / ``codes`` /
``formula``): the formula is computed by rule from the window, the CUI is
attached from the TemporalIndex when the vocabulary contains an entry with
that exact formula, and the window stays uncoded (but still carries its
formula) when it does not. Nothing here is written by hand: unit codes come
from the vocabulary file, entries from the index.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

from app.utils.temporal_index import (
    CANONICAL_UNITS,
    UNIT_SECONDS,
    TemporalIndex,
    WindowKey,
)
from app.utils.temporal_vocab import TemporalMapping

RELATIONS: tuple[str, ...] = ("last", "within", "range")
BASES: tuple[str, ...] = ("explicit", "inferred")


class CanonicalWindow(BaseModel):
    """Structured-output window. ``last`` and ``within`` both mean
    "from REF_POINT back ``value`` ``unit``"; ``range`` is
    "between ``value`` and ``to_value`` units back"."""

    # No numeric constraints in the JSON schema (`gt=0` becomes
    # `exclusiveMinimum`, which the Gemini structured-output schema rejects);
    # positivity is enforced by the validator below instead.
    relation: str = Field(..., json_schema_extra={"enum": list(RELATIONS)})
    value: float
    unit: str = Field(..., json_schema_extra={"enum": list(CANONICAL_UNITS)})
    to_value: float | None = None
    to_unit: str | None = Field(None, json_schema_extra={"enum": list(CANONICAL_UNITS)})

    @field_validator("value", "to_value")
    @classmethod
    def _positive(cls, v: float | None) -> float | None:
        if v is not None and v <= 0:
            raise ValueError("window value must be > 0")
        return v

    @field_validator("relation", mode="before")
    @classmethod
    def _norm_relation(cls, v: Any) -> Any:
        return str(v).strip().lower() if isinstance(v, str) else v

    @field_validator("unit", "to_unit", mode="before")
    @classmethod
    def _norm_unit(cls, v: Any) -> Any:
        if not isinstance(v, str):
            return v
        u = v.strip().lower()
        return u[:-1] if u.endswith("s") and u[:-1] in CANONICAL_UNITS else u

    def key(self, index: TemporalIndex) -> WindowKey | None:
        if self.unit not in CANONICAL_UNITS:
            return None
        relation = self.relation
        if relation not in RELATIONS:
            # Any other wording ("past", "previous", "prior", "since") still
            # means "back from the reference point": a range when a far edge
            # was given, else a single span. A relation word never loses a
            # window.
            relation = "range" if self.to_value is not None else "last"
        if relation == "range":
            if self.to_value is None:
                return None
            to_unit = self.to_unit if self.to_unit in CANONICAL_UNITS else self.unit
            lo, hi = sorted(
                [(self.value, self.unit), (self.to_value, to_unit)],
                key=lambda p: _seconds(p[0], p[1]),
            )
            return index.key_for("range", lo[0], lo[1], hi[0], hi[1])
        return index.key_for("single", self.value, self.unit)


_UNIT_SECONDS = UNIT_SECONDS


def _seconds(value: float, unit: str) -> float:
    return float(value) * _UNIT_SECONDS.get(unit, 0)


def window_span_seconds(window: dict[str, Any] | None) -> float | None:
    """Span of a canonical window in seconds (the far edge for a range), used
    to order windows by breadth. None when the window is missing or malformed."""
    if not isinstance(window, dict):
        return None
    try:
        unit = str(window.get("unit") or "").lower()
        value = float(window.get("value"))
    except (TypeError, ValueError):
        return None
    if unit not in _UNIT_SECONDS:
        return None
    span = _seconds(value, unit)
    if window.get("relation") == "range" and window.get("to_value") is not None:
        to_unit = str(window.get("to_unit") or unit).lower()
        try:
            span = max(span, _seconds(float(window["to_value"]), to_unit))
        except (TypeError, ValueError):
            pass
    return span


class CanonicalTemporalMapping(TemporalMapping):
    """TemporalMapping plus the audit fields canonical mode adds. Only produced
    in canonical mode, so the legacy per-candidate shape is untouched."""

    basis: str | None = None
    rationale: str | None = None
    window: dict[str, Any] | None = None
    # vocabulary (exact window) | broader (narrowest containing window) |
    # widest (nothing contains it) | default
    resolution: str | None = None


class CanonicalTemporalResolver:
    """Resolve canonical windows to vocabulary-coded TemporalMappings."""

    def __init__(
        self,
        index: TemporalIndex,
        max_codes: int = 2,
        min_similarity: float = 0.3,
    ):
        self._index = index
        self._max_codes = max(1, int(max_codes))
        self._min_similarity = float(min_similarity)

    @property
    def index(self) -> TemporalIndex:
        return self._index

    def resolve_window(
        self,
        window: CanonicalWindow | dict[str, Any] | None,
        wording: str = "",
        basis: str | None = None,
        rationale: str | None = None,
    ) -> list[CanonicalTemporalMapping]:
        """One canonical window -> zero or more mappings (one per distinct CUI,
        ranked by closeness of the entry name to the model's wording). The
        output is always a VOCABULARY entry — its own name, CUI and formula,
        exactly what the list-based prompt produced: the exact window when the
        vocabulary has it, else the narrowest vocabulary window that contains
        it, else the widest one. An unparseable window returns []."""
        if window is None:
            return []
        try:
            cw = (
                window
                if isinstance(window, CanonicalWindow)
                else CanonicalWindow.model_validate(window)
            )
        except Exception:  # noqa: BLE001 — malformed model output is "no window"
            return []
        key = cw.key(self._index)
        if key is None:
            return []

        payload = {
            k: (int(v) if isinstance(v, float) and v.is_integer() else v)
            for k, v in cw.model_dump(exclude_none=True).items()
        }
        resolution = "vocabulary"
        matches = self._index.rank_for_key(
            key, wording, self._max_codes, self._min_similarity
        )
        if not matches:
            broader = self._index.nearest_broader(key)
            if broader is None:
                return []
            key, resolution = broader
            matches = self._index.rank_for_key(
                key, wording, self._max_codes, self._min_similarity
            )
        return [
            CanonicalTemporalMapping(
                time_window=m.entry.name,
                codes=m.entry.cui,
                formula=list(m.entry.formula),
                basis=basis,
                rationale=rationale,
                window=payload,
                resolution=resolution,
            )
            for m in matches
        ]

    def resolve_text(
        self, text: str, basis: str | None = None
    ) -> list[CanonicalTemporalMapping]:
        """Fallback for an entry without a usable ``window``: windows literally
        present in the wording ("last 3 months")."""
        out: list[CanonicalTemporalMapping] = []
        for key in self._index.windows_from_text(text):
            window = {
                "relation": "range" if key.kind == "range" else "last",
                "value": key.value,
                "unit": key.unit,
            }
            if key.kind == "range":
                window["to_value"] = key.to_value
                window["to_unit"] = key.to_unit
            out.extend(self.resolve_window(window, wording=text, basis=basis))
        return out
