"""Shared base utilities for hydrogen formula calculators."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from string import Template
from typing import List, Optional, Tuple

from influxdb_client import InfluxDBClient
from influxdb_client.client.exceptions import InfluxDBError
from influxdb_client.client.write_api import SYNCHRONOUS

from faraday_monitor import SignalSelection


FILTERED_LAST_TEMPLATE = Template(
    """
from(bucket: "$bucket")
  |> range(start: -$window)
  |> filter(fn: (r) => r._measurement == "$measurement")
  |> filter(fn: (r) => r._field == "$field")
  |> filter(fn: (r) => r["$tag_key"] == "$tag_value")
  |> last()
"""
)


@dataclass(frozen=True)
class BaseInfluxExperienceConfig:
    """Connection and filtering settings shared by all calculators."""

    url: str
    token: str
    org: str
    tag_key: str
    experience: str
    range_window: str


@dataclass(frozen=True)
class ScalarSource:
    """Describes a scalar input that can be fetched or provided directly."""

    name: str
    signal: Optional[SignalSelection] = None
    fixed_value: Optional[float] = None


class InfluxCalculatorBase:
    """Shared InfluxDB client and query helpers."""

    def __init__(self, config: BaseInfluxExperienceConfig) -> None:
        self.config = config
        self.client = InfluxDBClient(
            url=config.url,
            token=config.token,
            org=config.org,
            timeout=60000,
        )
        self.query_api = self.client.query_api()
        self.write_api = self.client.write_api(write_options=SYNCHRONOUS)

    def close(self) -> None:
        self.client.close()

    def _fetch_latest(self, signal: SignalSelection) -> Tuple[Optional[float], Optional[datetime]]:
        flux = self._build_query(signal)
        try:
            tables = self.query_api.query(org=self.config.org, query=flux)
        except InfluxDBError as exc:
            logging.error(
                "Flux query failed for bucket=%s measurement=%s field=%s: %s",
                signal.bucket,
                signal.measurement,
                signal.field,
                exc,
            )
            return None, None

        for table in tables:
            for record in table.records:
                return record.get_value(), record.get_time()
        return None, None

    def _build_query(self, signal: SignalSelection) -> str:
        return FILTERED_LAST_TEMPLATE.substitute(
            bucket=signal.bucket,
            window=self.config.range_window,
            measurement=signal.measurement,
            field=signal.field,
            tag_key=escape_identifier(self.config.tag_key),
            tag_value=flux_escape(self.config.experience),
        )

    def _fetch_series(
        self,
        signal: SignalSelection,
        start: datetime,
        stop: datetime,
    ) -> List[Tuple[datetime, float]]:
        flux = _build_range_query(
            bucket=signal.bucket,
            measurement=signal.measurement,
            field=signal.field,
            tag_key=escape_identifier(self.config.tag_key),
            tag_value=flux_escape(self.config.experience),
            start=_format_timestamp(start),
            stop=_format_timestamp(stop),
        )
        tables = self.query_api.query(org=self.config.org, query=flux)
        series: List[Tuple[datetime, float]] = []
        for table in tables:
            for record in table.records:
                value = record.get_value()
                if value is None:
                    continue
                series.append((record.get_time(), value))
        return series


def flux_escape(value: str) -> str:
    """Escape user-provided strings so they remain valid Flux string literals."""

    return value.replace("\\", "\\\\").replace('"', '\\"')


def escape_identifier(value: str) -> str:
    if '"' in value:
        raise SystemExit("Tag keys containing double quotes are not supported.")
    return value


def pick_timestamp(*candidates: Optional[datetime]) -> datetime:
    timestamps = [dt for dt in candidates if dt is not None]
    return max(timestamps) if timestamps else datetime.now(timezone.utc)


def _format_timestamp(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def _build_range_query(
    bucket: str,
    measurement: str,
    field: str,
    tag_key: str,
    tag_value: str,
    start: str,
    stop: str,
) -> str:
    return f"""
from(bucket: "{bucket}")
  |> range(start: {start}, stop: {stop})
  |> filter(fn: (r) => r._measurement == "{measurement}")
  |> filter(fn: (r) => r._field == "{field}")
  |> filter(fn: (r) => r["{tag_key}"] == "{tag_value}")
  |> sort(columns: ["_time"])
"""


__all__ = [
    "BaseInfluxExperienceConfig",
    "ScalarSource",
    "InfluxCalculatorBase",
    "flux_escape",
    "escape_identifier",
    "pick_timestamp",
]
