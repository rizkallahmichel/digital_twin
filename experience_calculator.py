"""Core Faraday experience data models and calculations."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from string import Template
from typing import Optional, Tuple

from influxdb_client import InfluxDBClient
from influxdb_client.client.exceptions import InfluxDBError

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
class ExperienceConfig:
    """Fully validated configuration for a single experience request."""

    url: str
    token: str
    org: str
    tag_key: str
    experience: str
    range_window: str
    current_signal: SignalSelection
    efficiency_signal: SignalSelection
    efficiency_is_percent: bool
    faraday_constant: float
    electrons_per_molecule: float
    molar_volume: float


@dataclass(frozen=True)
class ExperienceResult:
    """Snapshot of the derived Faraday outputs for a particular experience."""

    experience: str
    timestamp: datetime
    current: float
    efficiency_ratio: float
    molar_rate: float
    volumetric_rate: float


class ExperienceCalculator:
    """Binds together the InfluxDB access and Faraday-law math for an experience."""

    def __init__(self, config: ExperienceConfig) -> None:
        self.config = config
        self.client = InfluxDBClient(
            url=config.url,
            token=config.token,
            org=config.org,
            timeout=60000,
        )
        self.query_api = self.client.query_api()
        self._denominator = config.electrons_per_molecule * config.faraday_constant

    def close(self) -> None:
        self.client.close()

    def compute(self) -> ExperienceResult:
        current_value, current_time = self._fetch_latest(self.config.current_signal)
        if current_value is None:
            raise RuntimeError(
                f"No current reading found for experience '{self.config.experience}' in "
                f"bucket '{self.config.current_signal.bucket}'."
            )

        efficiency_value, efficiency_time = self._fetch_latest(self.config.efficiency_signal)
        if efficiency_value is None:
            raise RuntimeError(
                f"No efficiency reading found for experience '{self.config.experience}' in "
                f"bucket '{self.config.efficiency_signal.bucket}'."
            )

        efficiency_ratio = (
            efficiency_value / 100.0 if self.config.efficiency_is_percent else efficiency_value
        )
        molar_rate = (efficiency_ratio * current_value) / self._denominator
        volumetric_rate = molar_rate * self.config.molar_volume
        timestamp = _pick_timestamp(current_time, efficiency_time)

        return ExperienceResult(
            experience=self.config.experience,
            timestamp=timestamp,
            current=current_value,
            efficiency_ratio=efficiency_ratio,
            molar_rate=molar_rate,
            volumetric_rate=volumetric_rate,
        )

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
            tag_key=_escape_identifier(self.config.tag_key),
            tag_value=_flux_escape(self.config.experience),
        )


def _flux_escape(value: str) -> str:
    """Escape user-provided strings so they remain valid Flux string literals."""

    return value.replace("\\", "\\\\").replace('"', '\\"')


def _escape_identifier(value: str) -> str:
    if '"' in value:
        raise SystemExit("Tag keys containing double quotes are not supported.")
    return value


def _pick_timestamp(*candidates: Optional[datetime]) -> datetime:
    timestamps = [dt for dt in candidates if dt is not None]
    return max(timestamps) if timestamps else datetime.now(timezone.utc)
