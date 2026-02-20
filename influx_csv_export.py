#!/usr/bin/env python3
"""Export InfluxDB time-series data to a CSV file."""

from __future__ import annotations

import argparse
import csv
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from influxdb_client import InfluxDBClient
from influxdb_client.client.exceptions import InfluxDBError

from formula_base import escape_identifier, flux_escape
from nas_config import influx_cli_defaults, resolve_defaults


def _parse_timestamp(label: str, raw: str | None) -> datetime:
    if not raw:
        raise SystemExit(f"{label} timestamp is required.")
    candidate = raw.strip()
    if candidate.endswith("Z"):
        candidate = candidate[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise SystemExit(
            f"{label} must follow the ISO-8601 format (e.g. 2026-02-15T14:30:00Z). Received {raw!r}."
        ) from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _format_timestamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _parse_tags(tag_args: List[str]) -> Dict[str, str]:
    tags: Dict[str, str] = {}
    for entry in tag_args:
        if "=" not in entry:
            raise SystemExit(f"Tag filters must use the key=value format (received {entry!r}).")
        key, value = entry.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise SystemExit(f"Tag filter {entry!r} is missing a key before '='.")
        tags[key] = value
    return tags


def _measurement_filter_clause(measurements: Sequence[str]) -> str:
    cleaned = [m for m in (item.strip() for item in measurements) if item]
    if not cleaned:
        raise SystemExit("At least one measurement must be specified.")
    if len(cleaned) == 1:
        measurement = flux_escape(cleaned[0])
        return f'  |> filter(fn: (r) => r._measurement == "{measurement}")'
    expression = " or ".join(f'r._measurement == "{flux_escape(m)}"' for m in cleaned)
    return f"  |> filter(fn: (r) => ({expression}))"


def _build_flux_query(
    bucket: str,
    measurements: Sequence[str],
    field: str | None,
    tags: Dict[str, str],
    start: datetime,
    stop: datetime,
) -> str:
    escaped_bucket = flux_escape(bucket)
    clauses = [
        f'from(bucket: "{escaped_bucket}")',
        f"  |> range(start: {_format_timestamp(start)}, stop: {_format_timestamp(stop)})",
        _measurement_filter_clause(measurements),
    ]
    if field:
        clauses.append(f'  |> filter(fn: (r) => r._field == "{flux_escape(field)}")')
    for tag_key, tag_value in tags.items():
        escaped_key = escape_identifier(tag_key)
        clauses.append(f'  |> filter(fn: (r) => r["{escaped_key}"] == "{flux_escape(tag_value)}")')
    clauses.append('  |> sort(columns: ["_time"])')
    return "\n".join(clauses)


def _ordered_columns(columns: Iterable[str]) -> List[str]:
    priority = [
        "_time",
        "_value",
        "_measurement",
        "_field",
        "_start",
        "_stop",
        "result",
        "table",
    ]
    prioritized = [col for col in priority if col in columns]
    remaining = sorted(col for col in columns if col not in priority)
    ordered = prioritized + remaining
    return ordered or priority[:4]


def _collect_rows(query_api, org: str, flux: str) -> Tuple[List[Dict[str, object]], List[str]]:
    rows: List[Dict[str, object]] = []
    columns: set[str] = set()
    try:
        for record in query_api.query_stream(org=org, query=flux):
            values = dict(record.values)
            for key, value in list(values.items()):
                if isinstance(value, datetime):
                    values[key] = value.astimezone(timezone.utc).isoformat()
            rows.append(values)
            columns.update(values.keys())
    except InfluxDBError as exc:
        raise SystemExit(f"Flux query failed: {exc}") from exc
    return rows, _ordered_columns(columns)


def _default_output_name(bucket: str, measurements: Sequence[str], start: datetime, stop: datetime) -> str:
    def _stamp(value: datetime) -> str:
        return value.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    if not measurements:
        meas_slug = "all"
    elif len(measurements) == 1:
        meas_slug = measurements[0]
    else:
        meas_slug = f"{measurements[0]}_plus{len(measurements) - 1}"
    safe_slug = meas_slug.replace("/", "-")
    return f"influx_export_{bucket}_{safe_slug}_{_stamp(start)}_{_stamp(stop)}.csv"


def parse_args() -> argparse.Namespace:
    defaults = resolve_defaults()
    cli_defaults = influx_cli_defaults(defaults)
    parser = argparse.ArgumentParser(
        description="Fetch a time-bounded slice from InfluxDB and write it to CSV."
    )
    parser.add_argument("--bucket", required=True, help="InfluxDB bucket containing the measurement.")
    parser.add_argument(
        "--measurement",
        dest="measurements",
        action="append",
        required=True,
        help="Measurement to export (repeat for multiple measurements).",
    )
    parser.add_argument(
        "--field",
        default=os.getenv("H2_EXPORT_FIELD", "value"),
        help="Field within the measurement to export (default: value).",
    )
    parser.add_argument("--start", required=True, help="ISO-8601 timestamp (inclusive) for the export window.")
    parser.add_argument("--stop", required=True, help="ISO-8601 timestamp (exclusive) for the export window.")
    parser.add_argument(
        "--tag",
        action="append",
        default=[],
        help="Optional tag filter expressed as key=value (repeat for multiple tags).",
    )
    parser.add_argument("--output", help="Destination CSV file. Defaults to a timestamped filename.")
    parser.add_argument(
        "--url",
        default=cli_defaults["url"],
        help=f"InfluxDB URL (default: {cli_defaults['url']}).",
    )
    parser.add_argument(
        "--token",
        default=cli_defaults["token"],
        help="InfluxDB API token (defaults to INFLUX_TOKEN or NAS fallback).",
    )
    parser.add_argument(
        "--org",
        default=cli_defaults["org"],
        help="InfluxDB organization (defaults to INFLUX_ORG or NAS fallback).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = _parse_timestamp("Start", args.start)
    stop = _parse_timestamp("Stop", args.stop)
    if stop <= start:
        raise SystemExit("Stop timestamp must be later than the start timestamp.")
    tags = _parse_tags(args.tag)
    field = args.field.strip() if isinstance(args.field, str) else None
    measurements = [item.strip() for item in (args.measurements or []) if item and item.strip()]
    if not measurements:
        raise SystemExit("Specify at least one --measurement.")
    flux = _build_flux_query(args.bucket, measurements, field or None, tags, start, stop)

    output_name = args.output or _default_output_name(args.bucket, measurements, start, stop)
    output_path = Path(output_name).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with InfluxDBClient(url=args.url, token=args.token, org=args.org, timeout=60000) as client:
        rows, columns = _collect_rows(client.query_api(), args.org, flux)

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    count = len(rows)
    print(f"Wrote {count} record{'s' if count != 1 else ''} to {output_path}")


if __name__ == "__main__":
    main()
