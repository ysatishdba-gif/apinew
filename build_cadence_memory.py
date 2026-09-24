#!/usr/bin/env python
"""Build the cadence-memory file from the service's own ``Temporal inference``
log events.

The service (in canonical temporal mode, /v3) logs one structured event per
resolved candidate window. Export those events (Cloud Logging sink, BigQuery
log table, or `gcloud logging read ... --format=json`) to JSON Lines and run:

    python scripts/build_cadence_memory.py \\
        --input events.jsonl [--input more.jsonl ...] \\
        --output app/cadence_memory.json \\
        [--previous app/cadence_memory.json] [--min-count 2] \\
        [--max-shift 0.3] [--upload gs://bucket/Nature_breakdown/cadence_memory.json]

Each input line may be the raw structured_data dict, a Cloud Logging entry
(``jsonPayload``), or a BigQuery log-table row (``jsonpayload`` / ``json_payload``).
Only events with ``message == "Temporal inference"`` and ``basis == "inferred"``
count: explicit spans belong to the query, not to the concept's cadence.

Drift: with ``--previous`` the script prints, per concept, whether the most
common window changed and how much the distribution moved; when any concept
moved more than ``--max-shift`` the exit code is 3 so a scheduler can hold the
upload for review. Nothing in the output is written by hand.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.utils.cadence_memory import aggregate_events, drift_report


def _payload(record: dict) -> dict | None:
    for key in ("jsonPayload", "jsonpayload", "json_payload", "structured_data"):
        if isinstance(record.get(key), dict):
            return record[key]
    return record


def iter_events(paths: Iterable[str]) -> Iterator[dict]:
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(record, list):
                    records = record
                else:
                    records = [record]
                for r in records:
                    if not isinstance(r, dict):
                        continue
                    payload = _payload(r)
                    if not isinstance(payload, dict):
                        continue
                    if (
                        payload.get("message", "Temporal inference")
                        != "Temporal inference"
                    ):
                        continue
                    if "timestamp" not in payload and r.get("timestamp"):
                        payload = {**payload, "timestamp": r["timestamp"]}
                    yield payload


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--input",
        action="append",
        required=True,
        help="JSONL export of log events (repeatable)",
    )
    ap.add_argument("--output", required=True, help="cadence_memory.json to write")
    ap.add_argument(
        "--previous", help="previous cadence_memory.json for the drift report"
    )
    ap.add_argument(
        "--min-count",
        type=int,
        default=1,
        help="drop windows seen fewer times than this",
    )
    ap.add_argument(
        "--max-shift",
        type=float,
        default=1.0,
        help="exit 3 when any concept's distribution shift exceeds this",
    )
    ap.add_argument(
        "--upload", help="gs://bucket/path to upload the file to after writing"
    )
    args = ap.parse_args(argv)

    version = datetime.now(UTC).replace(microsecond=0).isoformat()
    memory = aggregate_events(
        iter_events(args.input), min_count=args.min_count, version=version
    )

    previous = None
    if args.previous and Path(args.previous).exists():
        with open(args.previous, encoding="utf-8") as f:
            previous = json.load(f)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(memory, f, indent=2, ensure_ascii=False)
    meta = memory["_meta"]
    print(
        f"wrote {args.output}: {meta['concepts']} concepts from {meta['events']} inferred windows (version {version})"
    )

    report = drift_report(previous, memory)
    worst = 0.0
    for row in report[:50]:
        flag = "TOP CHANGED " if row["top_changed"] else ""
        print(
            f"  {flag}{row['concept']}: {row['previous_top']} -> {row['current_top']} (shift {row['distribution_shift']}, n={row['events']})"
        )
        worst = max(worst, row["distribution_shift"])

    if args.upload:
        from google.cloud import storage

        bucket, _, blob = args.upload.removeprefix("gs://").partition("/")
        storage.Client().bucket(bucket).blob(blob).upload_from_filename(args.output)
        print(f"uploaded to {args.upload}")

    if report and worst > args.max_shift:
        print(
            f"drift {worst} exceeds --max-shift {args.max_shift}; review before publishing",
            file=sys.stderr,
        )
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
