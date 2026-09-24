#!/usr/bin/env python
"""Temporal evaluation over a replay file: agreement between two endpoint
versions, unresolved rates, token and latency deltas, and an optional
LLM-as-judge pass over inferred windows.

    python scripts/evaluate_temporal.py --replay replay.jsonl \\
        --baseline v2 --candidate v3 \\
        [--judge-model gemini-2.5-flash --project my-project --location us-central1 --judge-sample 200] \\
        [--report report.json] [--min-agreement 0.995]

Agreement is measured on queries where the baseline emitted a coded window:
"cui" when the candidate's primary window shares a CUI, "formula" when it
shares the formula but not the CUI (a different vocabulary name for the same
span), "none" otherwise. The exit code is 2 when agreement (cui or formula)
falls below ``--min-agreement`` so CI can gate on it. The judge, when enabled,
scores each inferred window of the candidate version against the query and
the query's concepts (record types and tags stand in for them, since the
response carries no per-candidate window), and only the rejections are printed
for a human to sample; nothing is labelled by hand up front.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

JUDGE_PROMPT = """You are reviewing a clinical document-retrieval system.
For the query and the candidate concept below, the system inferred a retrieval time window
(no explicit time was given in the query). Decide whether the window is clinically reasonable
for retrieving that concept's documentation: too narrow loses documents, too wide only costs
ranking effort, so prefer accepting a wider-than-ideal window and rejecting a too-narrow one.

Query: {query}
Concepts / documents retrieved for it: {candidate}
Retrieval window: {window}

Return ONLY JSON: {{"verdict": "accept" | "reject", "reason": "<one sentence>"}}
"""


def load(path: str) -> dict[str, dict[str, dict]]:
    by_version: dict[str, dict[str, dict]] = defaultdict(dict)
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            by_version[row["version"]][row["query"]] = row
    return by_version


def codes(temporal: dict | None) -> set[str]:
    if not temporal:
        return set()
    return {c.get("code") for c in temporal.get("coding") or [] if c.get("code")}


def formula(temporal: dict | None) -> tuple:
    if not temporal:
        return ()
    return tuple(temporal.get("formula") or [])


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 3) if values else None


def _total_tokens(row: dict) -> int:
    usage = row.get("usage_metadata") or {}
    return sum(int((v or {}).get("prompt_token_count") or 0) for v in usage.values())


def judge(
    rows: list[dict],
    model: str,
    project: str,
    location: str,
    sample: int,
    seed: int = 7,
) -> list[dict]:
    from google import genai

    client = genai.Client(vertexai=True, project=project, location=location)
    items: list[dict] = []
    for row in rows:
        t = row.get("temporal") or {}
        if not t:
            continue
        concepts = [r.get("name") for r in row.get("record_types") or []] + [
            g.get("name") for g in row.get("tags") or []
        ]
        items.append(
            {
                "query": row["query"],
                "candidate": ", ".join(c for c in concepts if c) or "(none)",
                "window": t.get("name") or t.get("formula"),
            }
        )
    random.Random(seed).shuffle(items)
    out: list[dict] = []
    for item in items[:sample]:
        resp = client.models.generate_content(
            model=model,
            contents=JUDGE_PROMPT.format(**item),
            config={"response_mime_type": "application/json", "temperature": 0},
        )
        try:
            verdict = json.loads(resp.text)
        except (ValueError, TypeError):
            verdict = {"verdict": "unparseable", "reason": (resp.text or "")[:200]}
        out.append({**item, **verdict})
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--replay", required=True)
    ap.add_argument("--baseline", default="v2")
    ap.add_argument("--candidate", default="v3")
    ap.add_argument("--min-agreement", type=float, default=0.995)
    ap.add_argument("--judge-model")
    ap.add_argument("--project")
    ap.add_argument("--location", default="us-central1")
    ap.add_argument("--judge-sample", type=int, default=200)
    ap.add_argument("--report")
    args = ap.parse_args(argv)

    data = load(args.replay)
    base, cand = data.get(args.baseline, {}), data.get(args.candidate, {})
    common = [
        q
        for q in base
        if q in cand and not base[q].get("error") and not cand[q].get("error")
    ]

    agreement: Counter = Counter()
    disagreements: list[dict] = []
    for q in common:
        b, c = base[q]["temporal"], cand[q]["temporal"]
        if not codes(b):
            agreement["baseline_uncoded"] += 1
            continue
        if codes(b) & codes(c):
            agreement["cui"] += 1
        elif formula(b) and formula(b) == formula(c):
            agreement["formula"] += 1
        else:
            agreement["none"] += 1
            disagreements.append({"query": q, "baseline": b, "candidate": c})
    scored = agreement["cui"] + agreement["formula"] + agreement["none"]
    rate = (agreement["cui"] + agreement["formula"]) / scored if scored else None

    report = {
        "queries": len(common),
        "agreement": dict(agreement),
        "agreement_rate": round(rate, 4) if rate is not None else None,
        "candidate_unresolved_rate": round(
            sum(1 for q in common if not codes(cand[q]["temporal"])) / len(common), 4
        )
        if common
        else None,
        "baseline_unresolved_rate": round(
            sum(1 for q in common if not codes(base[q]["temporal"])) / len(common), 4
        )
        if common
        else None,
        "mean_prompt_tokens": {
            args.baseline: _mean([_total_tokens(base[q]) for q in common]),
            args.candidate: _mean([_total_tokens(cand[q]) for q in common]),
        },
        "mean_elapsed_seconds": {
            args.baseline: _mean([base[q]["elapsed_seconds"] for q in common]),
            args.candidate: _mean([cand[q]["elapsed_seconds"] for q in common]),
        },
        "disagreements": disagreements[:200],
    }

    if args.judge_model:
        if not args.project:
            ap.error("--project is required with --judge-model")
        verdicts = judge(
            [cand[q] for q in common],
            args.judge_model,
            args.project,
            args.location,
            args.judge_sample,
        )
        accepted = sum(1 for v in verdicts if v.get("verdict") == "accept")
        report["judge"] = {
            "model": args.judge_model,
            "sampled": len(verdicts),
            "accept_rate": round(accepted / len(verdicts), 4) if verdicts else None,
            "rejected": [v for v in verdicts if v.get("verdict") != "accept"],
        }

    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.report:
        Path(args.report).write_text(text, encoding="utf-8")
    print(text)
    if rate is not None and rate < args.min_agreement:
        print(
            f"agreement {rate:.4f} below --min-agreement {args.min_agreement}",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
