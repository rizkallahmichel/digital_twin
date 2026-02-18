"""Alpha loss (heat loss coefficient) calculations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable, List, Optional, Tuple

from influxdb_client import Point, WritePrecision

from formula_base import BaseInfluxExperienceConfig, InfluxCalculatorBase, ScalarSource, pick_timestamp


@dataclass(frozen=True)
class AlphaLossConfig(BaseInfluxExperienceConfig):
    """Configuration inputs required to compute Q_loss."""

    alpha_loss: ScalarSource  # kJ/(s*K)
    reactor_temp: ScalarSource  # °C (or same unit as ambient)
    ambient_temp: ScalarSource  # °C


@dataclass(frozen=True)
class AlphaLossResult:
    """Outputs for the alpha loss calculation."""

    experience: str
    timestamp: datetime
    reactor_temp: float
    ambient_temp: float
    alpha_loss: float
    q_loss: float


@dataclass(frozen=True)
class AlphaLossWriteTarget:
    """Measurement metadata for alpha loss results."""

    bucket: str
    measurement: str
    field: str = "q_loss_kj_s"


class AlphaLossCalculator(InfluxCalculatorBase):
    """Calculator producing Q_loss = alpha * (Tr - Tamb)."""

    def __init__(self, config: AlphaLossConfig) -> None:
        super().__init__(config)
        self.config = config

    def compute(self) -> AlphaLossResult:
        alpha_value, alpha_time = self._resolve_scalar(self.config.alpha_loss)
        reactor_temp, tr_time = self._resolve_scalar(self.config.reactor_temp)
        ambient_temp, ta_time = self._resolve_scalar(self.config.ambient_temp)
        timestamp = pick_timestamp(alpha_time, tr_time, ta_time)
        return self._build_result(timestamp, alpha_value, reactor_temp, ambient_temp)

    def write_result(self, result: AlphaLossResult, target: AlphaLossWriteTarget) -> None:
        point = (
            Point(target.measurement)
            .tag(self.config.tag_key, self.config.experience)
            .field(target.field, result.q_loss)
            .field("alpha_loss_kj_s_k", result.alpha_loss)
            .field("reactor_temp_c", result.reactor_temp)
            .field("ambient_temp_c", result.ambient_temp)
            .time(result.timestamp, WritePrecision.NS)
        )
        self.write_api.write(bucket=target.bucket, org=self.config.org, record=[point])

    def iter_experience(
        self,
        start: datetime,
        stop: datetime,
        step_seconds: float,
    ) -> Tuple[Iterable[AlphaLossResult], Optional[datetime], Optional[datetime]]:
        if step_seconds <= 0:
            raise RuntimeError("Sampling interval must be greater than zero seconds.")

        alpha_timeline = self._timeline_for_scalar(self.config.alpha_loss, start, stop)
        reactor_timeline = self._timeline_for_scalar(self.config.reactor_temp, start, stop)
        ambient_timeline = self._timeline_for_scalar(self.config.ambient_temp, start, stop)

        start_candidates = [
            ts for ts in (alpha_timeline.first_timestamp, reactor_timeline.first_timestamp, ambient_timeline.first_timestamp) if ts
        ]
        end_candidates = [
            ts for ts in (alpha_timeline.last_timestamp, reactor_timeline.last_timestamp, ambient_timeline.last_timestamp) if ts
        ]
        actual_start = max([start] + start_candidates) if start_candidates else start
        actual_end = min([stop] + end_candidates) if end_candidates else stop
        if actual_end < actual_start:
            return iter(()), actual_start, actual_end

        step = timedelta(seconds=step_seconds)

        def generator() -> Iterable[AlphaLossResult]:
            ts = actual_start
            while ts <= actual_end:
                yield self._build_result(
                    timestamp=ts,
                    alpha_value=alpha_timeline.value_at(ts),
                    reactor_temp=reactor_timeline.value_at(ts),
                    ambient_temp=ambient_timeline.value_at(ts),
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

    def _build_result(self, timestamp: datetime, alpha_value: float, reactor_temp: float, ambient_temp: float) -> AlphaLossResult:
        q_loss = alpha_value * (reactor_temp - ambient_temp)
        return AlphaLossResult(
            experience=self.config.experience,
            timestamp=timestamp,
            reactor_temp=reactor_temp,
            ambient_temp=ambient_temp,
            alpha_loss=alpha_value,
            q_loss=q_loss,
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
    "AlphaLossCalculator",
    "AlphaLossConfig",
    "AlphaLossResult",
    "AlphaLossWriteTarget",
]
