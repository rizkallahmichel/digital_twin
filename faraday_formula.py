"""Faraday-law based hydrogen production calculations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable, List, Optional, Tuple

from influxdb_client import Point, WritePrecision

from faraday_monitor import SignalSelection
from formula_base import BaseInfluxExperienceConfig, InfluxCalculatorBase, pick_timestamp


@dataclass(frozen=True)
class FaradayConfig(BaseInfluxExperienceConfig):
    """Configuration for Faraday-law computations."""

    current_signal: Optional[SignalSelection]
    efficiency_signal: Optional[SignalSelection]
    efficiency_is_percent: bool
    efficiency_fixed_ratio: Optional[float]
    faraday_constant: float
    electrons_per_molecule: float
    molar_volume: float
    current_value: Optional[float] = None
    sample_timestamp: Optional[datetime] = None


@dataclass(frozen=True)
class FaradayResult:
    """Snapshot of the derived Faraday outputs for a particular experience."""

    experience: str
    timestamp: datetime
    current: float
    efficiency_ratio: float
    molar_rate: float
    volumetric_rate: float


@dataclass(frozen=True)
class FaradayWriteTarget:
    """Describes where the derived rates should be stored in InfluxDB."""

    bucket: str
    molar_measurement: str
    volumetric_measurement: str
    field: str = "value"


class FaradayCalculator(InfluxCalculatorBase):
    """Binds together the InfluxDB access and Faraday-law math for an experience."""

    def __init__(self, config: FaradayConfig) -> None:
        super().__init__(config)
        self.config = config
        self._denominator = config.electrons_per_molecule * config.faraday_constant

    # Pull latest readings, combine them with Faraday math, and return the derived snapshot.
    def compute(self) -> FaradayResult:
        if self.config.current_value is not None:
            current_value = self.config.current_value
            current_time = self.config.sample_timestamp or datetime.now(timezone.utc)
        else:
            if self.config.current_signal is None:
                raise RuntimeError("Current signal is not configured for Faraday calculation.")
            current_value, current_time = self._fetch_latest(self.config.current_signal)
            if current_value is None:
                raise RuntimeError(
                    f"No current reading found for experience '{self.config.experience}' in "
                    f"bucket '{self.config.current_signal.bucket}'."
                )

        efficiency_value: Optional[float] = None
        efficiency_time: Optional[datetime] = None

        if self.config.efficiency_fixed_ratio is not None:
            efficiency_ratio = self.config.efficiency_fixed_ratio
        else:
            if self.config.efficiency_signal is None:
                raise RuntimeError("Efficiency signal is not configured and no fixed ratio provided.")

            efficiency_value, efficiency_time = self._fetch_latest(self.config.efficiency_signal)
            if efficiency_value is None:
                raise RuntimeError(
                    f"No efficiency reading found for experience '{self.config.experience}' in "
                    f"bucket '{self.config.efficiency_signal.bucket}'."
                )

            efficiency_ratio = (
                efficiency_value / 100.0 if self.config.efficiency_is_percent else efficiency_value
            )
        timestamp = pick_timestamp(current_time, efficiency_time)
        return self._build_result(timestamp, current_value, efficiency_ratio)

    def iter_experience(
        self,
        start: datetime,
        stop: datetime,
        step_seconds: float,
    ) -> Tuple[Iterable[FaradayResult], Optional[datetime], Optional[datetime]]:
        """Yield Faraday results across the requested window at fixed intervals."""

        if step_seconds <= 0:
            raise RuntimeError("Sampling interval must be greater than zero seconds.")
        if self.config.current_signal is None:
            raise RuntimeError("Current signal must be configured when replaying an experience.")

        current_series = self._fetch_series(self.config.current_signal, start, stop)
        if not current_series:
            return iter(()), None, None

        actual_start = max(start, current_series[0][0])
        actual_end = min(stop, current_series[-1][0])
        if actual_end < actual_start:
            return iter(()), actual_start, actual_end

        efficiency_series: Optional[List[Tuple[datetime, float]]] = None
        if self.config.efficiency_fixed_ratio is None:
            if self.config.efficiency_signal is None:
                raise RuntimeError(
                    "Efficiency signal must be configured when no fixed ratio is provided."
                )
            efficiency_series = self._fetch_series(self.config.efficiency_signal, start, stop)
            if not efficiency_series:
                return iter(()), actual_start, actual_end

        def _advance_index(series: List[Tuple[datetime, float]], idx: int, ts: datetime) -> int:
            while idx + 1 < len(series) and series[idx + 1][0] <= ts:
                idx += 1
            return idx

        def _generator() -> Iterable[FaradayResult]:
            current_idx = 0
            efficiency_idx = 0
            step = timedelta(seconds=step_seconds)
            ts = actual_start
            while ts <= actual_end:
                current_idx = _advance_index(current_series, current_idx, ts)
                current_value = current_series[current_idx][1]

                if efficiency_series is not None:
                    efficiency_idx = _advance_index(efficiency_series, efficiency_idx, ts)
                    efficiency_value = efficiency_series[efficiency_idx][1]
                    efficiency_ratio = (
                        efficiency_value / 100.0 if self.config.efficiency_is_percent else efficiency_value
                    )
                else:
                    if self.config.efficiency_fixed_ratio is None:
                        raise RuntimeError("Efficiency ratio is undefined.")
                    efficiency_ratio = self.config.efficiency_fixed_ratio

                yield self._build_result(ts, current_value, efficiency_ratio)
                ts += step

        return _generator(), actual_start, actual_end

    # Push the molar and volumetric rate points into the configured bucket.
    def write_result(self, result: FaradayResult, target: FaradayWriteTarget) -> None:
        """Persist the derived molar/volumetric rates back to InfluxDB."""

        points = self._result_points(result, target)
        self.write_api.write(bucket=target.bucket, org=self.config.org, record=points)

    def _build_result(self, timestamp: datetime, current_value: float, efficiency_ratio: float) -> FaradayResult:
        molar_rate = (efficiency_ratio * current_value) / self._denominator
        volumetric_rate = molar_rate * self.config.molar_volume
        return FaradayResult(
            experience=self.config.experience,
            timestamp=timestamp,
            current=current_value,
            efficiency_ratio=efficiency_ratio,
            molar_rate=molar_rate,
            volumetric_rate=volumetric_rate,
        )

    def _result_points(self, result: FaradayResult, target: FaradayWriteTarget):
        return [
            Point(target.molar_measurement)
            .tag(self.config.tag_key, self.config.experience)
            .field(target.field, result.molar_rate)
            .time(result.timestamp, WritePrecision.NS),
            Point(target.volumetric_measurement)
            .tag(self.config.tag_key, self.config.experience)
            .field(target.field, result.volumetric_rate)
            .time(result.timestamp, WritePrecision.NS),
        ]


__all__ = [
    "FaradayCalculator",
    "FaradayConfig",
    "FaradayResult",
    "FaradayWriteTarget",
]
