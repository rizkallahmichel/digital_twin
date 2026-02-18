"""Fuel-cell efficiency calculations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable, List, Optional, Tuple

from influxdb_client import Point, WritePrecision

from formula_base import BaseInfluxExperienceConfig, InfluxCalculatorBase, ScalarSource, pick_timestamp


@dataclass(frozen=True)
class FuelCellConfig(BaseInfluxExperienceConfig):
    """Configuration inputs for the fuel-cell calculator."""

    hydrogen_consumption: ScalarSource  # kg/s
    voltage: ScalarSource  # V
    current: ScalarSource  # A
    hydrogen_lhv_kj_per_mol: float = 241.8
    hydrogen_molar_mass_kg: float = 0.002016


@dataclass(frozen=True)
class FuelCellResult:
    """Outputs for the fuel-cell calculation."""

    experience: str
    timestamp: datetime
    voltage_v: float
    current_a: float
    power_w: float
    hydrogen_consumption_kg_s: float
    hydrogen_power_w: float
    efficiency_ratio: float


@dataclass(frozen=True)
class FuelCellWriteTarget:
    """Measurement metadata for fuel-cell outputs."""

    bucket: str
    measurement: str
    field: str = "efficiency_ratio"


class FuelCellCalculator(InfluxCalculatorBase):
    """Calculator for fuel-cell efficiency."""

    def __init__(self, config: FuelCellConfig) -> None:
        super().__init__(config)
        self.config = config

    def compute(self) -> FuelCellResult:
        h2_rate, rate_time = self._resolve_scalar(self.config.hydrogen_consumption)
        voltage, v_time = self._resolve_scalar(self.config.voltage)
        current, i_time = self._resolve_scalar(self.config.current)
        timestamp = pick_timestamp(rate_time, v_time, i_time)

        return self._build_result_from_values(timestamp, h2_rate, voltage, current)

    def write_result(self, result: FuelCellResult, target: FuelCellWriteTarget) -> None:
        """Persist efficiency and supporting metrics."""

        point = (
            Point(target.measurement)
            .tag(self.config.tag_key, self.config.experience)
            .field(target.field, result.efficiency_ratio)
            .field("power_w", result.power_w)
            .field("hydrogen_power_w", result.hydrogen_power_w)
            .field("hydrogen_consumption_kg_s", result.hydrogen_consumption_kg_s)
            .time(result.timestamp, WritePrecision.NS)
        )
        self.write_api.write(bucket=target.bucket, org=self.config.org, record=[point])

    def iter_experience(
        self,
        start: datetime,
        stop: datetime,
        step_seconds: float,
    ) -> Tuple[Iterable[FuelCellResult], Optional[datetime], Optional[datetime]]:
        if step_seconds <= 0:
            raise RuntimeError("Sampling interval must be greater than zero seconds.")

        h2_rate = self._timeline_for_scalar(self.config.hydrogen_consumption, start, stop)
        voltage = self._timeline_for_scalar(self.config.voltage, start, stop)
        current = self._timeline_for_scalar(self.config.current, start, stop)

        start_candidates = [ts for ts in (h2_rate.first_timestamp, voltage.first_timestamp, current.first_timestamp) if ts]
        end_candidates = [ts for ts in (h2_rate.last_timestamp, voltage.last_timestamp, current.last_timestamp) if ts]
        actual_start = max([start] + start_candidates) if start_candidates else start
        actual_end = min([stop] + end_candidates) if end_candidates else stop
        if actual_end < actual_start:
            return iter(()), actual_start, actual_end

        step = timedelta(seconds=step_seconds)

        def generator() -> Iterable[FuelCellResult]:
            ts = actual_start
            while ts <= actual_end:
                yield self._build_result_from_values(
                    timestamp=ts,
                    hydrogen_rate=h2_rate.value_at(ts),
                    voltage=voltage.value_at(ts),
                    current=current.value_at(ts),
                )
                ts += step

        return generator(), actual_start, actual_end

    def _resolve_scalar(self, source: ScalarSource) -> Tuple[float, Optional[datetime]]:
        if source.fixed_value is not None:
            return source.fixed_value, None
        if source.signal is None:
            raise RuntimeError(f"Scalar '{source.name}' is missing both a fixed value and a signal selection.")

        value, timestamp = self._fetch_latest(source.signal)
        if value is None:
            raise RuntimeError(
                f"No reading found for '{source.name}' in bucket '{source.signal.bucket}' "
                f"({source.signal.measurement}/{source.signal.field})."
            )
        return value, timestamp

    def _timeline_for_scalar(self, source: ScalarSource, start: datetime, stop: datetime) -> "_ScalarTimeline":
        if source.fixed_value is not None:
            return _ScalarTimeline(constant=source.fixed_value)
        if source.signal is None:
            raise RuntimeError(f"Scalar '{source.name}' is missing both a fixed value and a signal selection.")

        series = self._fetch_series(source.signal, start, stop)
        if not series:
            raise RuntimeError(
                f"No readings found for '{source.name}' in bucket '{source.signal.bucket}' "
                f"({source.signal.measurement}/{source.signal.field}) between {start} and {stop}."
            )
        return _ScalarTimeline(series=series)

    def _build_result_from_values(
        self,
        timestamp: datetime,
        hydrogen_rate: float,
        voltage: float,
        current: float,
    ) -> FuelCellResult:
        power_w = voltage * current
        if self.config.hydrogen_molar_mass_kg <= 0:
            raise RuntimeError("Hydrogen molar mass must be positive.")
        molar_rate = hydrogen_rate / self.config.hydrogen_molar_mass_kg
        hydrogen_power_w = molar_rate * self.config.hydrogen_lhv_kj_per_mol * 1000.0
        efficiency_ratio = power_w / hydrogen_power_w if hydrogen_power_w > 0 else 0.0

        return FuelCellResult(
            experience=self.config.experience,
            timestamp=timestamp,
            voltage_v=voltage,
            current_a=current,
            power_w=power_w,
            hydrogen_consumption_kg_s=hydrogen_rate,
            hydrogen_power_w=hydrogen_power_w,
            efficiency_ratio=efficiency_ratio,
        )


class _ScalarTimeline:
    def __init__(self, series: Optional[List[Tuple[datetime, float]]] = None, constant: Optional[float] = None) -> None:
        self.series = series or []
        self.constant = constant
        self._index = 0

    @property
    def first_timestamp(self) -> Optional[datetime]:
        if self.constant is not None or not self.series:
            return None
        return self.series[0][0]

    @property
    def last_timestamp(self) -> Optional[datetime]:
        if self.constant is not None or not self.series:
            return None
        return self.series[-1][0]

    def value_at(self, ts: datetime) -> float:
        if self.constant is not None:
            return self.constant
        while self._index + 1 < len(self.series) and self.series[self._index + 1][0] <= ts:
            self._index += 1
        return self.series[self._index][1]


__all__ = [
    "FuelCellCalculator",
    "FuelCellConfig",
    "FuelCellResult",
    "FuelCellWriteTarget",
]
