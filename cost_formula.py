"""Operating cost aggregation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable, List, Optional, Tuple

from influxdb_client import Point, WritePrecision

from formula_base import BaseInfluxExperienceConfig, InfluxCalculatorBase, ScalarSource, pick_timestamp


@dataclass(frozen=True)
class CostConfig(BaseInfluxExperienceConfig):
    """Configuration inputs for cost calculations."""

    energy_used_kwh: ScalarSource
    energy_price_per_kwh: ScalarSource
    lohc_degradation_rate: Optional[ScalarSource] = None  # fraction of LOHC replaced per period
    lohc_inventory_mass: Optional[ScalarSource] = None  # kg
    lohc_price_per_kg: Optional[ScalarSource] = None
    catalyst_deactivation_rate: Optional[ScalarSource] = None  # fraction per period
    catalyst_inventory_mass: Optional[ScalarSource] = None  # kg
    catalyst_price_per_kg: Optional[ScalarSource] = None


@dataclass(frozen=True)
class CostResult:
    """Outputs for the cost calculation."""

    experience: str
    timestamp: datetime
    energy_cost: float
    lohc_cost: float
    catalyst_cost: float
    total_cost: float


@dataclass(frozen=True)
class CostWriteTarget:
    """Measurement metadata for costs."""

    bucket: str
    measurement: str
    field: str = "total_cost"


class CostCalculator(InfluxCalculatorBase):
    """Calculator for OPEX-style cost reporting."""

    def __init__(self, config: CostConfig) -> None:
        super().__init__(config)
        self.config = config

    def compute(self) -> CostResult:
        energy_used, e_time = self._resolve_scalar(self.config.energy_used_kwh)
        energy_price, price_time = self._resolve_scalar(self.config.energy_price_per_kwh)
        lohc_args = self._optional_cost_terms(
            self.config.lohc_degradation_rate,
            self.config.lohc_inventory_mass,
            self.config.lohc_price_per_kg,
        )
        catalyst_args = self._optional_cost_terms(
            self.config.catalyst_deactivation_rate,
            self.config.catalyst_inventory_mass,
            self.config.catalyst_price_per_kg,
        )
        timestamp = pick_timestamp(e_time, price_time, lohc_args.timestamp, catalyst_args.timestamp)
        return self._build_result_from_values(
            timestamp=timestamp,
            energy_used=energy_used,
            energy_price=energy_price,
            lohc_tuple=lohc_args,
            catalyst_tuple=catalyst_args,
        )

    def write_result(self, result: CostResult, target: CostWriteTarget) -> None:
        """Persist cost metrics."""
        point = (
            Point(target.measurement)
            .tag(self.config.tag_key, self.config.experience)
            .field(target.field, result.total_cost)
            .field("energy_cost", result.energy_cost)
            .field("lohc_cost", result.lohc_cost)
            .field("catalyst_cost", result.catalyst_cost)
            .time(result.timestamp, WritePrecision.NS)
        )
        self.write_api.write(bucket=target.bucket, org=self.config.org, record=[point])

    def iter_experience(
        self,
        start: datetime,
        stop: datetime,
        step_seconds: float,
    ) -> Tuple[Iterable[CostResult], Optional[datetime], Optional[datetime]]:
        if step_seconds <= 0:
            raise RuntimeError("Sampling interval must be greater than zero seconds.")

        energy_used = self._timeline_for_scalar(self.config.energy_used_kwh, start, stop)
        energy_price = self._timeline_for_scalar(self.config.energy_price_per_kwh, start, stop)
        lohc_timeline = self._optional_timeline(
            self.config.lohc_degradation_rate,
            self.config.lohc_inventory_mass,
            self.config.lohc_price_per_kg,
            start,
            stop,
        )
        catalyst_timeline = self._optional_timeline(
            self.config.catalyst_deactivation_rate,
            self.config.catalyst_inventory_mass,
            self.config.catalyst_price_per_kg,
            start,
            stop,
        )

        start_candidates = [
            ts
            for ts in (
                energy_used.first_timestamp,
                energy_price.first_timestamp,
                lohc_timeline.first_timestamp if lohc_timeline else None,
                catalyst_timeline.first_timestamp if catalyst_timeline else None,
            )
            if ts is not None
        ]
        end_candidates = [
            ts
            for ts in (
                energy_used.last_timestamp,
                energy_price.last_timestamp,
                lohc_timeline.last_timestamp if lohc_timeline else None,
                catalyst_timeline.last_timestamp if catalyst_timeline else None,
            )
            if ts is not None
        ]
        actual_start = max([start] + start_candidates) if start_candidates else start
        actual_end = min([stop] + end_candidates) if end_candidates else stop
        if actual_end < actual_start:
            return iter(()), actual_start, actual_end

        step = timedelta(seconds=step_seconds)

        def generator() -> Iterable[CostResult]:
            ts = actual_start
            while ts <= actual_end:
                lohc_tuple = (
                    lohc_timeline.value_at(ts)
                    if lohc_timeline
                    else OptionalCostTerms(0.0, None)
                )
                catalyst_tuple = (
                    catalyst_timeline.value_at(ts)
                    if catalyst_timeline
                    else OptionalCostTerms(0.0, None)
                )
                yield self._build_result_from_values(
                    timestamp=ts,
                    energy_used=energy_used.value_at(ts),
                    energy_price=energy_price.value_at(ts),
                    lohc_tuple=lohc_tuple,
                    catalyst_tuple=catalyst_tuple,
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

    def _optional_cost_terms(
        self,
        rate_source: Optional[ScalarSource],
        mass_source: Optional[ScalarSource],
        price_source: Optional[ScalarSource],
    ) -> "OptionalCostTerms":
        if rate_source is None or mass_source is None or price_source is None:
            return OptionalCostTerms(value=0.0, timestamp=None)

        rate, r_time = self._resolve_scalar(rate_source)
        mass, m_time = self._resolve_scalar(mass_source)
        price, p_time = self._resolve_scalar(price_source)
        timestamp = pick_timestamp(r_time, m_time, p_time)
        value = rate * mass * price
        return OptionalCostTerms(value=value, timestamp=timestamp)

    def _optional_timeline(
        self,
        rate_source: Optional[ScalarSource],
        mass_source: Optional[ScalarSource],
        price_source: Optional[ScalarSource],
        start: datetime,
        stop: datetime,
    ) -> Optional["_CostTimeline"]:
        if rate_source is None or mass_source is None or price_source is None:
            return None
        rate_tl = self._timeline_for_scalar(rate_source, start, stop)
        mass_tl = self._timeline_for_scalar(mass_source, start, stop)
        price_tl = self._timeline_for_scalar(price_source, start, stop)
        return _CostTimeline(rate_tl, mass_tl, price_tl)

    def _build_result_from_values(
        self,
        timestamp: datetime,
        energy_used: float,
        energy_price: float,
        lohc_tuple: "OptionalCostTerms",
        catalyst_tuple: "OptionalCostTerms",
    ) -> CostResult:
        energy_cost = energy_used * energy_price
        lohc_cost = lohc_tuple.value
        catalyst_cost = catalyst_tuple.value
        total_cost = energy_cost + lohc_cost + catalyst_cost
        return CostResult(
            experience=self.config.experience,
            timestamp=timestamp,
            energy_cost=energy_cost,
            lohc_cost=lohc_cost,
            catalyst_cost=catalyst_cost,
            total_cost=total_cost,
        )


@dataclass(frozen=True)
class OptionalCostTerms:
    value: float
    timestamp: Optional[datetime]


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


class _CostTimeline:
    """Helper for optional cost components."""

    def __init__(self, rate: _ScalarTimeline, mass: _ScalarTimeline, price: _ScalarTimeline) -> None:
        self.rate = rate
        self.mass = mass
        self.price = price

    @property
    def first_timestamp(self) -> Optional[datetime]:
        candidates = [self.rate.first_timestamp, self.mass.first_timestamp, self.price.first_timestamp]
        return max([c for c in candidates if c is not None], default=None)

    @property
    def last_timestamp(self) -> Optional[datetime]:
        candidates = [self.rate.last_timestamp, self.mass.last_timestamp, self.price.last_timestamp]
        return min([c for c in candidates if c is not None], default=None)

    def value_at(self, ts: datetime) -> OptionalCostTerms:
        return OptionalCostTerms(
            value=self.rate.value_at(ts) * self.mass.value_at(ts) * self.price.value_at(ts),
            timestamp=ts,
        )


__all__ = [
    "CostCalculator",
    "CostConfig",
    "CostResult",
    "CostWriteTarget",
]
