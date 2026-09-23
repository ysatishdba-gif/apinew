"""In-process index over the temporal vocabulary for canonical temporal mode.

Built entirely from the vocabulary file the service already loads (no
hand-written mappings). Two structures:

  * a FORMULA index: every entry's formula (``REF_POINT - 3Y`` /
    ``REF_POINT - 4W | REF_POINT - 6M``) is parsed into a canonical window key
    ``(kind, value, unit)`` or ``(range, from, from_unit, to, to_unit)`` so a
    canonical window emitted by the model resolves to the vocabulary entries
    that mean exactly that span, without the model ever seeing the list;
  * a SIMILARITY index over entry names (character n-gram TF-IDF by default,
    optional Vertex embeddings) used to (a) rank entries that share a formula
    by closeness to the model's own wording and (b) shortlist candidates for a
    query at request time.

Unit codes used in formulas (``H``, ``D``, ``M`` ...) are learned from the
file itself by pairing each entry's formula unit with the unit word in its
name; nothing about the vocabulary content is hard-coded.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from app.utils.temporal_vocab import TemporalVocab

# Canonical unit names the model may emit (schema enum). These are a format
# contract, like the response envelope, not vocabulary content.
CANONICAL_UNITS: tuple[str, ...] = (
    "second",
    "minute",
    "hour",
    "day",
    "week",
    "month",
    "year",
)

# Surface forms of each unit as they occur in vocabulary names; used only to
# learn the file's formula unit codes and to normalise free text.
_UNIT_WORDS: dict[str, tuple[str, ...]] = {
    "second": ("second", "seconds", "sec", "secs", "s"),
    "minute": ("minute", "minutes", "min", "mins"),
    "hour": ("hour", "hours", "hr", "hrs", "h"),
    "day": ("day", "days", "d"),
    "week": ("week", "weeks", "wk", "wks", "w"),
    "month": ("month", "months", "mo", "mos"),
    "year": ("year", "years", "yr", "yrs", "y"),
}

_NUMBER_WORDS: dict[str, int] = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
    "hundred": 100,
}

# Determiners that make a bare unit word mean "one unit" ("last year").
_IMPLICIT_ONE = frozenset(
    {
        "a",
        "an",
        "last",
        "past",
        "previous",
        "prior",
        "this",
        "the",
        "one",
        "next",
        "per",
        "every",
        "within",
    }
)

_FORMULA_TERM = re.compile(
    r"^\s*REF_POINT(?:\s*-\s*(\d+(?:\.\d+)?)\s*([A-Za-z]+))?\s*$"
)


def _num(value: float) -> float | int:
    return int(value) if float(value).is_integer() else float(value)


@dataclass(frozen=True)
class WindowKey:
    """Canonical span. ``kind`` is ``single`` (REF_POINT back ``value`` units)
    or ``range`` (from ``value`` .. ``to_value`` units back)."""

    kind: str
    value: float | int
    unit: str
    to_value: float | int | None = None
    to_unit: str | None = None

    def formula(self, unit_codes: dict[str, str]) -> list[str]:
        if self.kind == "range":
            return [
                f"REF_POINT - {_num(self.value)}{unit_codes[self.unit]}",
                f"REF_POINT - {_num(self.to_value)}{unit_codes[self.to_unit or self.unit]}",
            ]
        return ["REF_POINT", f"REF_POINT - {_num(self.value)}{unit_codes[self.unit]}"]

    def label(self) -> str:
        """Display name for a window the vocabulary does not contain."""

        def _u(v: float, unit: str) -> str:
            return f"{_num(v)} {unit}{'' if v == 1 else 's'}"

        if self.kind == "range":
            return f"Between {_u(self.value, self.unit)} and {_u(self.to_value, self.to_unit or self.unit)}"
        return f"Last {_u(self.value, self.unit)}"


@dataclass
class IndexEntry:
    name: str
    cui: str
    formula: list[str]
    key: WindowKey | None
    tokens: int = 0


@dataclass
class Match:
    entry: IndexEntry
    score: float


# ---------------------------------------------------------------------------
# Text normalisation and lexical similarity (dependency-free)
# ---------------------------------------------------------------------------
def fold(text: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).strip().lower()).strip()


def _ngrams(text: str, n: int = 3) -> list[str]:
    s = f" {fold(text)} "
    if len(s) < n:
        return [s]
    return [s[i : i + n] for i in range(len(s) - n + 1)]


class LexicalSimilarity:
    """Character trigram TF-IDF cosine similarity. Deterministic, in-process,
    built from the vocabulary names at load time."""

    provider = "lexical"

    def __init__(self, names: Sequence[str]):
        self._names = list(names)
        df: Counter = Counter()
        self._docs: list[dict[str, float]] = []
        for name in self._names:
            counts = Counter(_ngrams(name))
            self._docs.append(dict(counts))
            df.update(counts.keys())
        n_docs = max(1, len(self._names))
        self._idf = {g: math.log((1 + n_docs) / (1 + c)) + 1.0 for g, c in df.items()}
        self._vectors = [self._vectorise(d) for d in self._docs]

    def _vectorise(self, counts: dict[str, float]) -> dict[str, float]:
        vec = {g: c * self._idf.get(g, 1.0) for g, c in counts.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {g: v / norm for g, v in vec.items()}

    def scores(self, text: str) -> list[float]:
        q = self._vectorise(dict(Counter(_ngrams(text))))
        out: list[float] = []
        for vec in self._vectors:
            if len(q) < len(vec):
                s = sum(w * vec.get(g, 0.0) for g, w in q.items())
            else:
                s = sum(w * q.get(g, 0.0) for g, w in vec.items())
            out.append(s)
        return out


class VertexEmbeddingSimilarity:
    """Optional dense embeddings through the google-genai client. Vectors are
    cached on disk keyed by (model, vocabulary hash) so a process restart does
    not re-embed. Falls back to lexical scoring on any failure."""

    provider = "vertex"

    def __init__(
        self,
        names: Sequence[str],
        model: str,
        project: str,
        location: str,
        cache_dir: str | None,
        vocab_hash: str,
        batch_size: int = 100,
    ):
        import numpy as np

        self._names = list(names)
        self._lexical = LexicalSimilarity(names)
        self._model = model
        self._project = project
        self._location = location
        self._matrix = None
        cache_file = None
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            cache_file = os.path.join(cache_dir, f"temporal_{model}_{vocab_hash}.npy")
            if os.path.exists(cache_file):
                self._matrix = np.load(cache_file)
        if self._matrix is None:
            vectors = self._embed_all(batch_size)
            self._matrix = np.asarray(vectors, dtype="float32")
            norms = np.linalg.norm(self._matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self._matrix = self._matrix / norms
            if cache_file:
                np.save(cache_file, self._matrix)

    def _client(self):
        from google import genai

        return genai.Client(
            vertexai=True, project=self._project, location=self._location
        )

    def _embed_all(self, batch_size: int) -> list[list[float]]:
        client = self._client()
        out: list[list[float]] = []
        for i in range(0, len(self._names), batch_size):
            batch = self._names[i : i + batch_size]
            response = client.models.embed_content(model=self._model, contents=batch)
            out.extend([list(e.values) for e in response.embeddings])
        return out

    def scores(self, text: str) -> list[float]:
        try:
            import numpy as np

            response = self._client().models.embed_content(
                model=self._model, contents=[text]
            )
            q = np.asarray(response.embeddings[0].values, dtype="float32")
            q = q / (np.linalg.norm(q) or 1.0)
            return [float(x) for x in self._matrix @ q]
        except Exception:  # noqa: BLE001 — never let scoring take the request down
            return self._lexical.scores(text)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def parse_formula(
    formula: Sequence[str] | None,
) -> tuple[str, float, str, float | None, str | None] | None:
    """``["REF_POINT", "REF_POINT - 3Y"]`` -> ("single", 3, "Y", None, None);
    ``["REF_POINT - 4W", "REF_POINT - 6M"]`` -> ("range", 4, "W", 6, "M")."""
    if not formula or len(formula) != 2:
        return None
    a = _FORMULA_TERM.match(str(formula[0]))
    b = _FORMULA_TERM.match(str(formula[1]))
    if not a or not b:
        return None
    a_val, a_code = a.group(1), a.group(2)
    b_val, b_code = b.group(1), b.group(2)
    if a_val is None and b_val is not None:
        return ("single", float(b_val), b_code.upper(), None, None)
    if a_val is not None and b_val is not None:
        return ("range", float(a_val), a_code.upper(), float(b_val), b_code.upper())
    return None


def parse_number_unit(text: str) -> list[tuple[float, str]]:
    """All (value, canonical unit) pairs mentioned in free text, in order.
    Understands digits, number words and unit abbreviations; ``1.5 hours``,
    ``three months``, ``30D``."""
    # Keep decimals ("1.5 hours") while folding everything else like fold().
    lowered = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", str(text).strip().lower())
    words = re.sub(r"[^a-z0-9.]+", " ", lowered).split()
    out: list[tuple[float, str]] = []
    unit_lookup = {w: u for u, ws in _UNIT_WORDS.items() for w in ws}
    # "6 to 12 months" / "6-12 months": the first number borrows the unit that
    # follows the second one.
    borrowed: dict[int, str] = {}
    is_number = re.compile(r"^\d+(?:\.\d+)?$").match
    for i in range(len(words) - 2):
        if not is_number(words[i]):
            continue
        # "6 to 12 months" / "6 and 12 months"
        if (
            i + 3 < len(words)
            and words[i + 1] in ("to", "and", "or")
            and is_number(words[i + 2])
            and words[i + 3] in unit_lookup
        ):
            borrowed[i] = unit_lookup[words[i + 3]]
        # "6-12 months" (the hyphen folds to a space)
        elif is_number(words[i + 1]) and words[i + 2] in unit_lookup:
            borrowed[i] = unit_lookup[words[i + 2]]
    i = 0
    while i < len(words):
        w = words[i]
        value: float | None = None
        unit: str | None = None
        m = re.match(r"^(\d+(?:\.\d+)?)([a-z]+)?$", w)
        if m:
            value = float(m.group(1))
            if m.group(2) and m.group(2) in unit_lookup:
                unit = unit_lookup[m.group(2)]
        elif w in _NUMBER_WORDS:
            value = float(_NUMBER_WORDS[w])
            # "twenty four" style compounds
            if (
                i + 1 < len(words)
                and words[i + 1] in _NUMBER_WORDS
                and _NUMBER_WORDS[words[i + 1]] < 10
                and value >= 20
            ):
                value += _NUMBER_WORDS[words[i + 1]]
                i += 1
        elif (
            w in _IMPLICIT_ONE
            and i + 1 < len(words)
            and words[i + 1] in unit_lookup
            and not (i + 2 < len(words) and words[i + 2] in ("ago",) and w in ("the",))
        ):
            # "last year", "past month", "a week", "within the year" -> 1 unit
            value = 1.0
        if value is not None:
            if unit is None and i in borrowed:
                unit = borrowed[i]
            if unit is None:
                j = i + 1
                while j < len(words) and j <= i + 2:
                    if words[j] in unit_lookup:
                        unit = unit_lookup[words[j]]
                        break
                    if words[j] in ("to", "and", "or") or re.match(r"^\d", words[j]):
                        break
                    j += 1
            if unit is not None:
                out.append((value, unit))
        i += 1
    return out


# ---------------------------------------------------------------------------
# The index
# ---------------------------------------------------------------------------
@dataclass
class TemporalIndex:
    entries: list[IndexEntry]
    unit_codes: dict[str, str]  # canonical unit -> formula code, learned
    code_units: dict[str, str]  # formula code -> canonical unit
    vocab_hash: str
    similarity: Any = None
    _by_key: dict[WindowKey, list[int]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    # ---- construction -----------------------------------------------------
    @classmethod
    def from_vocab(
        cls,
        vocab: TemporalVocab,
        similarity_provider: str = "lexical",
        embedding_model: str = "",
        project: str = "",
        location: str = "",
        cache_dir: str | None = None,
    ) -> TemporalIndex:
        raw = {
            name: [e.model_dump() for e in vocab.codes_for_name(name)]
            for name in vocab._names
        }
        vocab_hash = hashlib.sha256(
            json.dumps(raw, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()[:16]

        # 1. Learn formula unit codes from (formula code, unit word in name) pairs.
        votes: dict[str, Counter] = defaultdict(Counter)
        parsed: list[tuple[str, str, list[str], tuple | None]] = []
        for name, codes in raw.items():
            if not codes:
                continue
            cui = codes[0].get("cui") or ""
            formula = list(codes[0].get("formula") or [])
            p = parse_formula(formula)
            parsed.append((name, cui, formula, p))
            if p is None:
                continue
            units_in_name = [u for _, u in parse_number_unit(name)]
            kind, _v, code, _tv, to_code = p
            if kind == "single" and len(units_in_name) >= 1:
                votes[code][units_in_name[-1]] += 1
            elif kind == "range" and len(units_in_name) >= 2:
                votes[code][units_in_name[0]] += 1
                votes[to_code][units_in_name[-1]] += 1
        code_units: dict[str, str] = {}
        for code, counter in votes.items():
            unit, _ = counter.most_common(1)[0]
            code_units[code] = unit
        unit_codes: dict[str, str] = {}
        for code, unit in code_units.items():
            # keep the shortest code per unit when several map to one unit
            if unit not in unit_codes or len(code) < len(unit_codes[unit]):
                unit_codes[unit] = code

        # 2. Build entries with canonical keys.
        entries: list[IndexEntry] = []
        for name, cui, formula, p in parsed:
            key = None
            if p is not None:
                kind, v, code, tv, to_code = p
                unit = code_units.get(code)
                to_unit = code_units.get(to_code) if to_code else None
                if unit and (kind == "single" or to_unit):
                    key = WindowKey(
                        kind,
                        _num(v),
                        unit,
                        _num(tv) if tv is not None else None,
                        to_unit,
                    )
            entries.append(
                IndexEntry(
                    name=name,
                    cui=cui,
                    formula=formula,
                    key=key,
                    tokens=len(fold(name).split()),
                )
            )

        index = cls(
            entries=entries,
            unit_codes=unit_codes,
            code_units=code_units,
            vocab_hash=vocab_hash,
        )
        for i, e in enumerate(entries):
            if e.key is not None:
                index._by_key.setdefault(e.key, []).append(i)

        # 3. Similarity index.
        names = [e.name for e in entries]
        if similarity_provider == "vertex" and embedding_model:
            try:
                index.similarity = VertexEmbeddingSimilarity(
                    names, embedding_model, project, location, cache_dir, vocab_hash
                )
            except Exception:  # noqa: BLE001 — embeddings are an optimisation only
                index.similarity = LexicalSimilarity(names)
        else:
            index.similarity = LexicalSimilarity(names)
        return index

    # ---- queries ----------------------------------------------------------
    def __len__(self) -> int:
        return len(self.entries)

    @property
    def similarity_provider(self) -> str:
        return getattr(self.similarity, "provider", "none")

    def coverage(self) -> dict[str, Any]:
        keyed = sum(1 for e in self.entries if e.key is not None)
        return {
            "entries": len(self.entries),
            "with_canonical_key": keyed,
            "distinct_windows": len(self._by_key),
            "unit_codes": dict(self.unit_codes),
            "similarity_provider": self.similarity_provider,
            "vocab_hash": self.vocab_hash,
        }

    def key_for(
        self,
        kind: str,
        value: float,
        unit: str,
        to_value: float | None = None,
        to_unit: str | None = None,
    ) -> WindowKey:
        return WindowKey(
            kind,
            _num(value),
            unit,
            _num(to_value) if to_value is not None else None,
            to_unit,
        )

    def entries_for_key(self, key: WindowKey) -> list[IndexEntry]:
        return [self.entries[i] for i in self._by_key.get(key, [])]

    def can_encode(self, key: WindowKey) -> bool:
        units = {key.unit} | ({key.to_unit} if key.to_unit else set())
        return all(u in self.unit_codes for u in units)

    def nearest(self, text: str, k: int = 10, min_score: float = 0.0) -> list[Match]:
        """Top-k vocabulary entries by similarity to free text."""
        if not text or not text.strip() or not self.entries:
            return []
        scores = self.similarity.scores(text)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        out: list[Match] = []
        for i in ranked[:k]:
            if scores[i] < min_score:
                break
            out.append(Match(self.entries[i], float(scores[i])))
        return out

    def rank_for_key(
        self, key: WindowKey, wording: str, max_codes: int, min_score: float = 0.0
    ) -> list[Match]:
        """Entries that mean exactly ``key``, ranked by closeness to ``wording``
        (then by brevity, so questionnaire sentences that merely mention a
        span rank below the span's own name). Returns at most ``max_codes``
        distinct CUIs."""
        candidates = self.entries_for_key(key)
        if not candidates:
            return []
        scores = (
            self.similarity.scores(wording) if wording else [0.0] * len(self.entries)
        )
        pos = {id(e): i for i, e in enumerate(self.entries)}
        ranked = sorted(
            candidates,
            key=lambda e: (-(scores[pos[id(e)]]), e.tokens, e.name.lower()),
        )
        out: list[Match] = []
        seen: set[str] = set()
        for e in ranked:
            s = float(scores[pos[id(e)]])
            if out and s < min_score:
                break
            if e.cui in seen:
                continue
            seen.add(e.cui)
            out.append(Match(e, s))
            if len(out) >= max_codes:
                break
        return out

    def windows_from_text(self, text: str) -> list[WindowKey]:
        """Canonical windows literally present in text (``last 3 months`` ->
        single 3 month; ``6 to 12 months`` -> range)."""
        pairs = parse_number_unit(text)
        if not pairs:
            return []
        folded = fold(text)
        out: list[WindowKey] = []
        if len(pairs) >= 2 and re.search(r"\b(to|and|between|-)\b", folded):
            (a, au), (b, bu) = pairs[0], pairs[1]
            out.append(self.key_for("range", a, au, b, bu))
        for v, u in pairs:
            out.append(self.key_for("single", v, u))
        return list(dict.fromkeys(out))

    def menu(
        self, max_values_per_unit: int, units: Iterable[str] | None = None
    ) -> dict[str, list[float | int]]:
        """Codeable single-window values per unit, most-represented values
        first then ascending — a compact, data-derived list of the windows the
        vocabulary can code, for the prompt."""
        per_unit: dict[str, Counter] = defaultdict(Counter)
        for key, idxs in self._by_key.items():
            if key.kind == "single" and (units is None or key.unit in units):
                per_unit[key.unit][key.value] += len(idxs)
        out: dict[str, list[float | int]] = {}
        for unit in CANONICAL_UNITS:
            if unit not in per_unit:
                continue
            top = [v for v, _ in per_unit[unit].most_common(max_values_per_unit)]
            out[unit] = sorted(top)
        return out
