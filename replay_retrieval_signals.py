#!/usr/bin/env python
"""Replay queries against a running service and record what each endpoint
version returns, for the temporal evaluation (gold set + agreement).

    python scripts/replay_retrieval_signals.py \\
        --base-url http://localhost:8000/intent-nature-breakdown \\
        --queries queries.txt            # one query per line, or JSONL with {"text": ...}
        --versions v2 v3 \\
        --output replay.jsonl [--concurrency 2] [--limit 500]

One output line per (query, version) with the returned ``temporal``,
``record_types``, ``tags``, timing and usage metadata (the response shape is
the same for every version; only the matching behind it differs).
Feed two versions to scripts/evaluate_temporal.py for the agreement report.
The gold set is whatever the current production version returns; nothing is
labelled by hand.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests


def load_queries(path: str, limit: int | None) -> list[str]:
    out: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("{"):
                try:
                    line = str(json.loads(line).get("text") or "").strip()
                except json.JSONDecodeError:
                    continue
            if line:
                out.append(line)
            if limit and len(out) >= limit:
                break
    return out


def call(base_url: str, version: str, query: str, timeout: int) -> dict:
    started = time.time()
    resp = requests.post(
        f"{base_url.rstrip('/')}/{version}/retrieval-signals",
        json={"queries": [{"text": [query]}]},
        timeout=timeout,
    )
    elapsed = time.time() - started
    row = {
        "query": query,
        "version": version,
        "status_code": resp.status_code,
        "elapsed_seconds": round(elapsed, 3),
    }
    try:
        body = resp.json()
    except ValueError:
        row["error"] = resp.text[:500]
        return row
    if resp.status_code != 200:
        row["error"] = body.get("detail")
        return row
    q = (body.get("output") or {}).get("queries") or [{}]
    q = q[0] if q else {}
    row.update(
        {
            "temporal": q.get("temporal"),
            "record_types": q.get("record_types"),
            "tags": q.get("tags"),
            "skipped": q.get("skipped", False),
            "usage_metadata": (body.get("details") or {}).get("usage_metadata"),
            "timing": (body.get("details") or {}).get("timing"),
            "service": body.get("service"),
        }
    )
    return row


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--versions", nargs="+", default=["v2", "v3"])
    ap.add_argument("--output", required=True)
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args(argv)

    queries = load_queries(args.queries, args.limit)
    jobs = [(v, q) for q in queries for v in args.versions]
    print(
        f"{len(queries)} queries x {len(args.versions)} versions -> {len(jobs)} calls",
        file=sys.stderr,
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with (
        open(args.output, "w", encoding="utf-8") as out,
        ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool,
    ):
        for row in pool.map(
            lambda j: call(args.base_url, j[0], j[1], args.timeout), jobs
        ):
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
    print(f"wrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
