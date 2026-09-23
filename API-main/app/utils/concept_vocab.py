from __future__ import annotations

import json
import re

from pydantic import BaseModel


class ConceptCode(BaseModel):
    cui: str
    formula: str | None = None


def normalize_facet_name(raw: str) -> str:
    if not isinstance(raw, str):
        return ""
    words = re.split(r"[_\s]+", raw.strip())
    return " ".join(w.capitalize() if not w.isupper() else w for w in words if w)


_RELATED_MIN_QUERY_TOKENS = 2
_RELATED_AMBIGUITY_MARGIN = 0.10


class ConceptVocab:
    def __init__(self, name_to_codes: dict[str, list[dict[str, str]]]):
        self._map: dict[str, list[dict[str, str]]] = {}
        self.skipped_names = 0  # names dropped: not a list, or no usable codes
        self.skipped_codes = 0  # individual code entries dropped from kept names
        for key, entries in (name_to_codes or {}).items():
            if key.startswith("_"):  # metadata entries, not vocabulary
                continue
            if not isinstance(entries, list):
                self.skipped_names += 1
                continue
            usable = [
                e
                for e in entries
                if isinstance(e, dict)
                and isinstance(e.get("cui"), str)
                and e["cui"].strip()
            ]
            if not usable:
                self.skipped_names += 1
                continue
            self.skipped_codes += len(entries) - len(usable)
            self._map[key] = usable

        self._folded_to_name = {self._fold(n): n for n in self._map}
        self._name_tokens = {n: set(self._fold(n).split()) for n in self._map}

    @staticmethod
    def _fold(text: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", text.strip().lower()).strip()

    @classmethod
    def from_file(cls, path: str) -> ConceptVocab:
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f))

    @classmethod
    def from_gcs(cls, project: str, bucket: str, blob_path: str) -> ConceptVocab:
        from google.cloud import storage

        blob = storage.Client(project=project).bucket(bucket).blob(blob_path)
        return cls(json.loads(blob.download_as_text()))

    def name_from_text(self, text: str | None) -> str | None:
        """Resolve to canonical vocab key: exact, then folded (case/underscore-insensitive)."""
        if not isinstance(text, str) or not text.strip():
            return None
        t = text.strip()
        if t in self._map:
            return t
        return self._folded_to_name.get(self._fold(t))

    def names(self) -> list[str]:
        """Canonical vocabulary labels, for injection into the LLM prompt."""
        return list(self._map.keys())

    def codes_for_name(self, name: str | None) -> list[ConceptCode]:
        if not name:
            return []
        return [ConceptCode.model_validate(e) for e in self._map.get(name, [])]

    def lookup(self, text: str | None) -> tuple[str | None, list[ConceptCode]]:
        """(canonical name or None, codes) for arbitrary input text."""
        name = self.name_from_text(text)
        return name, self.codes_for_name(name)

    def lookup_related(
        self,
        text: str | None,
        max_matches: int | None = 3,
        min_score: float = 0.60,
        reject_ambiguous: bool = True,
    ) -> list[tuple[str, list[ConceptCode]]]:
        if not isinstance(text, str) or not text.strip():
            return []

        q_fold = self._fold(text)
        q_tokens = set(q_fold.split())
        if len(q_tokens) < _RELATED_MIN_QUERY_TOKENS:
            return []

        scored: list[tuple[float, str]] = []
        for name, name_tokens in self._name_tokens.items():
            if not name_tokens:
                continue
            inter = len(q_tokens & name_tokens)
            if inter == 0:
                continue

            # Coverage rewards candidates that include most query intent tokens.
            coverage = inter / max(1, len(q_tokens))
            jaccard = inter / max(1, len(q_tokens | name_tokens))

            name_fold = self._fold(name)
            phrase_bonus = 0.15 if q_fold in name_fold or name_fold in q_fold else 0.0

            score = (0.70 * coverage) + (0.30 * jaccard) + phrase_bonus
            if score >= min_score:
                scored.append((score, name))

        scored.sort(key=lambda x: (-x[0], x[1].lower()))

        if (
            reject_ambiguous
            and len(scored) > 1
            and (scored[0][0] - scored[1][0]) < _RELATED_AMBIGUITY_MARGIN
        ):
            return []

        out: list[tuple[str, list[ConceptCode]]] = []
        matches = scored if max_matches is None else scored[:max_matches]
        for _, name in matches:
            codes = self.codes_for_name(name)
            if codes:
                out.append((name, codes))
        return out

    def __len__(self) -> int:
        return len(self._map)
