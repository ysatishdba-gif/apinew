"""Cadence memory: the service's own inferred temporal windows, learned from
traffic and fed back as a consistency prior.

Nothing in here is authored. At request time the pipeline logs one
``Temporal inference`` event per resolved window (see
``ContextualIntentPipeline._log_temporal_inferences``). An offline job
(``scripts/build_cadence_memory.py``) aggregates those events per concept into
a versioned JSON file; the service loads that file at startup (GCS first,
local fallback, like the vocabularies) and, for the concepts found in a query,
injects the most common previously inferred windows into the prompt as
examples. The model may still override them from the query and the concept's
clinical nature; the prior only keeps inferences consistent across calls and
model versions, and the job reports drift when they shift.

File shape (built by the job, never edited by hand)::

    {
      "_meta": {"version": "2026-09-23T12:00:00Z", "events": 18342, "generated_by": "..."},
      "concepts": {
        "hba1c": {
          "concept": "HbA1c",
          "total": 412,
          "windows": [
            {"label": "last 6 months", "relation": "last", "value": 6, "unit": "month",
             "count": 301, "share": 0.73, "last_seen": "2026-09-22"},
            ...
          ]
        }
      }
    }
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

from app.utils.temporal_index import LexicalSimilarity, fold


def _fmt(value: Any) -> str:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    return str(int(f)) if f.is_integer() else str(f)


def window_label(window: dict[str, Any] | None) -> str | None:
    """Stable human label for a canonical window: ``last 6 months``,
    ``between 6 and 12 months``."""
    if not isinstance(window, dict):
        return None
    relation = str(window.get("relation") or "").lower()
    unit = str(window.get("unit") or "").lower()
    value = window.get("value")
    if not relation or not unit or value is None:
        return None

    def _unit(v: Any, u: str) -> str:
        return f"{u}{'' if _fmt(v) == '1' else 's'}"

    if relation == "range" and window.get("to_value") is not None:
        to_unit = str(window.get("to_unit") or unit).lower()
        if to_unit == unit:
            return f"between {_fmt(value)} and {_fmt(window['to_value'])} {_unit(window['to_value'], unit)}"
        return (
            f"between {_fmt(value)} {_unit(value, unit)} and "
            f"{_fmt(window['to_value'])} {_unit(window['to_value'], to_unit)}"
        )
    return f"last {_fmt(value)} {_unit(value, unit)}"


class CadenceMemory:
    """Read-only view over the aggregated inference file."""

    def __init__(self, data: dict[str, Any] | None):
        data = data or {}
        self.meta: dict[str, Any] = dict(data.get("_meta") or {})
        concepts = data.get("concepts") or {}
        self._concepts: dict[str, dict[str, Any]] = {}
        for key, entry in concepts.items():
            if not isinstance(entry, dict):
                continue
            k = fold(key)
            if k:
                self._concepts[k] = entry
        self._keys = list(self._concepts.keys())
        self._similarity = LexicalSimilarity(self._keys) if self._keys else None

    # ---- loading ----------------------------------------------------------
    @classmethod
    def empty(cls) -> CadenceMemory:
        return cls({})

    @classmethod
    def from_file(cls, path: str) -> CadenceMemory:
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f))

    @classmethod
    def from_gcs(cls, project: str, bucket: str, blob_path: str) -> CadenceMemory:
        from google.cloud import storage

        blob = storage.Client(project=project).bucket(bucket).blob(blob_path)
        return cls(json.loads(blob.download_as_text()))

    # ---- queries ----------------------------------------------------------
    def __len__(self) -> int:
        return len(self._concepts)

    @property
    def version(self) -> str | None:
        return self.meta.get("version")

    def lookup(self, concept: str, min_similarity: float) -> dict[str, Any] | None:
        """Exact folded match first, else the closest known concept above the
        similarity threshold (so "hemoglobin a1c" finds "hba1c" only when the
        names are close enough)."""
        key = fold(concept)
        if not key or not self._keys:
            return None
        if key in self._concepts:
            return self._concepts[key]
        scores = self._similarity.scores(key)
        best = max(range(len(scores)), key=lambda i: scores[i])
        if scores[best] >= min_similarity:
            return self._concepts[self._keys[best]]
        return None

    def examples_for(
        self,
        concepts: Iterable[str],
        per_concept: int,
        max_concepts: int,
        min_similarity: float,
        min_share: float = 0.0,
    ) -> list[str]:
        """Prompt lines for the concepts of a query, most-used windows first."""
        out: list[str] = []
        seen: set[str] = set()
        for concept in concepts:
            if len(out) >= max_concepts:
                break
            entry = self.lookup(concept, min_similarity)
            if not entry:
                continue
            ident = fold(entry.get("concept") or concept)
            if ident in seen:
                continue
            seen.add(ident)
            windows = [
                w
                for w in (entry.get("windows") or [])
                if isinstance(w, dict)
                and w.get("label")
                and float(w.get("share") or 0) >= min_share
            ]
            windows = sorted(windows, key=lambda w: -int(w.get("count") or 0))[
                : max(1, per_concept)
            ]
            if not windows:
                continue
            parts = ", ".join(
                f"{w['label']} ({int(w.get('count') or 0)}x)" for w in windows
            )
            out.append(f"{entry.get('concept') or concept}: {parts}")
        return out


# ---------------------------------------------------------------------------
# Aggregation (used by scripts/build_cadence_memory.py; pure, testable)
# ---------------------------------------------------------------------------
def aggregate_events(
    events: Iterable[dict[str, Any]],
    min_count: int = 1,
    version: str | None = None,
    generated_by: str = "scripts/build_cadence_memory.py",
) -> dict[str, Any]:
    """Fold ``Temporal inference`` events into the memory file shape. Only
    inferred windows that resolved to a vocabulary entry count (exact,
    broader or widest); explicit spans are the query's, not the model's, and
    say nothing about the concept's cadence; defaults are not inferences."""
    per_concept: dict[str, dict[str, Any]] = {}
    used = 0
    for ev in events:
        if not isinstance(ev, dict):
            continue
        if ev.get("basis") != "inferred":
            continue
        if ev.get("resolution") not in ("vocabulary", "broader", "widest"):
            continue
        concept = str(ev.get("candidate") or ev.get("concept") or "").strip()
        label = ev.get("window_label") or window_label(ev.get("window"))
        if not concept or not label:
            continue
        used += 1
        key = fold(concept)
        entry = per_concept.setdefault(
            key, {"concept": concept, "total": 0, "windows": {}}
        )
        entry["total"] += 1
        w = entry["windows"].setdefault(
            label,
            {
                "label": label,
                "relation": (ev.get("window") or {}).get("relation"),
                "value": (ev.get("window") or {}).get("value"),
                "unit": (ev.get("window") or {}).get("unit"),
                "count": 0,
                "last_seen": None,
            },
        )
        w["count"] += 1
        seen = str(ev.get("timestamp") or "")[:10] or None
        if seen and (w["last_seen"] is None or seen > w["last_seen"]):
            w["last_seen"] = seen

    concepts: dict[str, Any] = {}
    for key, entry in per_concept.items():
        windows = [w for w in entry["windows"].values() if w["count"] >= min_count]
        if not windows:
            continue
        total = sum(w["count"] for w in windows)
        for w in windows:
            w["share"] = round(w["count"] / total, 4)
        concepts[key] = {
            "concept": entry["concept"],
            "total": total,
            "windows": sorted(windows, key=lambda w: -w["count"]),
        }
    return {
        "_meta": {
            "version": version,
            "events": used,
            "concepts": len(concepts),
            "generated_by": generated_by,
        },
        "concepts": concepts,
    }


def drift_report(
    previous: dict[str, Any] | None, current: dict[str, Any]
) -> list[dict[str, Any]]:
    """Per concept: did the most common window change, and by how much did
    the distribution move (total variation distance)? Consumed by the job's
    exit code / alerting; empty when there is no previous file."""
    if not previous:
        return []
    prev = previous.get("concepts") or {}
    cur = current.get("concepts") or {}
    report: list[dict[str, Any]] = []
    for key, entry in cur.items():
        old = prev.get(key)
        if not old:
            continue
        old_top = (old.get("windows") or [{}])[0].get("label")
        new_top = (entry.get("windows") or [{}])[0].get("label")
        old_dist = {
            w["label"]: float(w.get("share") or 0) for w in old.get("windows") or []
        }
        new_dist = {
            w["label"]: float(w.get("share") or 0) for w in entry.get("windows") or []
        }
        labels = set(old_dist) | set(new_dist)
        tvd = 0.5 * sum(
            abs(old_dist.get(l, 0.0) - new_dist.get(l, 0.0)) for l in labels
        )
        report.append(
            {
                "concept": entry.get("concept"),
                "previous_top": old_top,
                "current_top": new_top,
                "top_changed": old_top != new_top,
                "distribution_shift": round(tvd, 4),
                "events": entry.get("total"),
            }
        )
    return sorted(report, key=lambda r: (-r["top_changed"], -r["distribution_shift"]))
