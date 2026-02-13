"""LOHC hydrogenation and hydrogen storage rate calculations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable, List, Optional, Tuple

from influxdb_client import Point, WritePrecision

from formula_base import (
    BaseInfluxExperienceConfig,
    InfluxCalculatorBase,
    ScalarSource,
    pick_timestamp,
)


@dataclass(frozen=True)
class LohcRateConfig(BaseInfluxExperienceConfig):
    """Configuration for LOHC hydrogenation/storage rate calculations."""

    doh_ratio: ScalarSource
    lohc_density: ScalarSource
    lohc_molar_mass: float
    partial_pressure: ScalarSource
    henry_constant: ScalarSource
    volumetric_fraction: ScalarSource
    temperature: ScalarSource
    gas_constant: float
    kinetic_constant: Optional[ScalarSource]
    pre_exponential: Optional[float]
    activation_energy: Optional[float]


@dataclass(frozen=True)
class LohcRateResult:
    """Result snapshot for the LOHC hydrogenation rate."""

    experience: str
    timestamp: datetime
    doh_ratio: float
    nmih_concentration: float
    hydrogenated_concentration: float
    kinetic_constant: float
    hydrogenation_rate: float
    hydrogen_storage_rate: float


@dataclass(frozen=True)
class LohcRateWriteTarget:
    """Describes where LOHC hydrogenation/storage rates are stored in InfluxDB."""

    bucket: str
    hydrogenation_measurement: str
    storage_measurement: str
    field: str = "value"


class LohcRateCalculator(InfluxCalculatorBase):
    """Calculator for LOHC hydrogenation and hydrogen storage rates."""

    def __init__(self, config: LohcRateConfig) -> None:
        super().__init__(config)
        self.config = config

    def compute(self) -> LohcRateResult:
        timestamps: List[Optional[datetime]] = []

        doh_ratio, doh_time = self._resolve_scalar(self.config.doh_ratio)
        timestamps.append(doh_time)

        density, density_time = self._resolve_scalar(self.config.lohc_density)
        timestamps.append(density_time)

        partial_pressure, pp_time = self._resolve_scalar(self.config.partial_pressure)
        timestamps.append(pp_time)

        henry_constant, henry_time = self._resolve_scalar(self.config.henry_constant)
        timestamps.append(henry_time)

        volumetric_fraction, vf_time = self._resolve_scalar(self.config.volumetric_fraction)
        timestamps.append(vf_time)

        temperature_k, temp_time = self._resolve_scalar(self.config.temperature)
        timestamps.append(temp_time)

        kinetic_constant, k_time = self._resolve_kinetic_constant(temperature_k)
        timestamps.append(k_time)

        timestamp = pick_timestamp(*timestamps)

        return self._build_result_from_values(
            timestamp=timestamp,
            doh_ratio=doh_ratio,
            density=density,
            partial_pressure=partial_pressure,
            henry_constant=henry_constant,
            volumetric_fraction=volumetric_fraction,
            temperature_k=temperature_k,
            kinetic_constant=kinetic_constant,
        )

    def write_result(self, result: LohcRateResult, target: LohcRateWriteTarget) -> None:
        """Persist LOHC hydrogenation and storage rates to InfluxDB."""

        points = [
            Point(target.hydrogenation_measurement)
            .tag(self.config.tag_key, self.config.experience)
            .field(target.field, result.hydrogenation_rate)
            .time(result.timestamp, WritePrecision.NS),
            Point(target.storage_measurement)
            .tag(self.config.tag_key, self.config.experience)
            .field(target.field, result.hydrogen_storage_rate)
            .time(result.timestamp, WritePrecision.NS),
        ]
        self.write_api.write(bucket=target.bucket, org=self.config.org, record=points)

    def _resolve_kinetic_constant(self, temperature_k: float) -> Tuple[float, Optional[datetime]]:
        if self.config.kinetic_constant is not None:
            return self._resolve_scalar(self.config.kinetic_constant)

        kinetic_constant = self._arrhenius_constant(temperature_k)
        return kinetic_constant, None

    def _arrhenius_constant(self, temperature_k: float) -> float:
        if self.config.pre_exponential is None or self.config.activation_energy is None:
            raise RuntimeError(
                "Provide either --lohc-k inputs or both --lohc-k0 and --lohc-activation-energy for Arrhenius calculation."
            )
        return self.config.pre_exponential * math.exp(
            -self.config.activation_energy / (self.config.gas_constant * temperature_k)
        )

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

    def iter_experience(
        self,
        start: datetime,
        stop: datetime,
        step_seconds: float,
    ) -> Tuple[Iterable[LohcRateResult], Optional[datetime], Optional[datetime]]:
        if step_seconds <= 0:
            raise RuntimeError("Sampling interval must be greater than zero seconds.")

        doh_ratio = self._timeline_for_scalar(self.config.doh_ratio, start, stop, required=True, allow_constant=True)
        density = self._timeline_for_scalar(self.config.lohc_density, start, stop, required=True, allow_constant=True)
        partial_pressure = self._timeline_for_scalar(
            self.config.partial_pressure, start, stop, required=True, allow_constant=True
        )
        henry_constant = self._timeline_for_scalar(
            self.config.henry_constant, start, stop, required=True, allow_constant=True
        )
        volumetric_fraction = self._timeline_for_scalar(
            self.config.volumetric_fraction, start, stop, required=True, allow_constant=True
        )
        temperature = self._timeline_for_scalar(
            self.config.temperature, start, stop, required=True, allow_constant=True
        )

        kinetic_timeline: Optional[_ScalarTimeline] = None
        if self.config.kinetic_constant is not None:
            kinetic_timeline = self._timeline_for_scalar(
                self.config.kinetic_constant, start, stop, required=True, allow_constant=True
            )

        timelines = [doh_ratio, density, partial_pressure, henry_constant, volumetric_fraction, temperature]
        if kinetic_timeline is not None:
            timelines.append(kinetic_timeline)

        start_candidates = [ts for ts in (_t.first_timestamp for _t in timelines) if ts is not None]
        end_candidates = [ts for ts in (_t.last_timestamp for _t in timelines) if ts is not None]
        actual_start = max([start] + start_candidates) if start_candidates else None
        actual_end = min([stop] + end_candidates) if end_candidates else None
        if actual_start is None or actual_end is None or actual_end < actual_start:
            return iter(()), actual_start, actual_end

        step = timedelta(seconds=step_seconds)

        def generator() -> Iterable[LohcRateResult]:
            ts = actual_start
            while ts <= actual_end:
                doh_value = doh_ratio.value_at(ts)
                density_value = density.value_at(ts)
                partial_pressure_value = partial_pressure.value_at(ts)
                henry_value = henry_constant.value_at(ts)
                volumetric_fraction_value = volumetric_fraction.value_at(ts)
                temperature_value = temperature.value_at(ts)

                if kinetic_timeline is not None:
                    kinetic_value = kinetic_timeline.value_at(ts)
                else:
                    kinetic_value = self._arrhenius_constant(temperature_value)

                yield self._build_result_from_values(
                    timestamp=ts,
                    doh_ratio=doh_value,
                    density=density_value,
                    partial_pressure=partial_pressure_value,
                    henry_constant=henry_value,
                    volumetric_fraction=volumetric_fraction_value,
                    temperature_k=temperature_value,
                    kinetic_constant=kinetic_value,
                )
                ts += step

        return generator(), actual_start, actual_end

    def _build_result_from_values(
        self,
        timestamp: datetime,
        doh_ratio: float,
        density: float,
        partial_pressure: float,
        henry_constant: float,
        volumetric_fraction: float,
        temperature_k: float,
        kinetic_constant: float,
    ) -> LohcRateResult:
        if doh_ratio < 0:
            raise RuntimeError("DoH ratio must be non-negative.")
        if density <= 0:
            raise RuntimeError("LOHC density must be positive.")
        if self.config.lohc_molar_mass <= 0:
            raise RuntimeError("LOHC molar mass must be positive.")
        if temperature_k <= 0:
            raise RuntimeError("Temperature must be above 0 K.")
        if henry_constant == 0:
            raise RuntimeError("Henry constant must be non-zero.")

        nmih_concentration = density / self.config.lohc_molar_mass
        hydrogenated_concentration = doh_ratio * nmih_concentration
        hydrogenation_rate = (
            volumetric_fraction * kinetic_constant * hydrogenated_concentration * partial_pressure / henry_constant
        )
        hydrogen_storage_rate = 4.0 * hydrogenation_rate

        return LohcRateResult(
            experience=self.config.experience,
            timestamp=timestamp,
            doh_ratio=doh_ratio,
            nmih_concentration=nmih_concentration,
            hydrogenated_concentration=hydrogenated_concentration,
            kinetic_constant=kinetic_constant,
            hydrogenation_rate=hydrogenation_rate,
            hydrogen_storage_rate=hydrogen_storage_rate,
        )

    def _timeline_for_scalar(
        self,
        source: ScalarSource,
        start: datetime,
        stop: datetime,
        required: bool,
        allow_constant: bool,
    ) -> "_ScalarTimeline":
        if source.fixed_value is not None:
            if not allow_constant:
                raise RuntimeError(
                    f"Scalar '{source.name}' must reference an Influx signal when replaying a completed experience."
                )
            return _ScalarTimeline(label=source.name, constant=source.fixed_value)

        if source.signal is None:
            if required:
                raise RuntimeError(f"Scalar '{source.name}' is missing both a fixed value and a signal selection.")
            return _ScalarTimeline(label=source.name, constant=None)

        series = self._fetch_series(source.signal, start, stop)
        if not series:
            raise RuntimeError(
                f"No readings found for '{source.name}' in bucket '{source.signal.bucket}' "
                f"({source.signal.measurement}/{source.signal.field}) between {start} and {stop}."
            )
        return _ScalarTimeline(label=source.name, series=series)


class _ScalarTimeline:
    def __init__(
        self,
        label: str,
        series: Optional[List[Tuple[datetime, float]]] = None,
        constant: Optional[float] = None,
    ) -> None:
        self.label = label
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
    "LohcRateCalculator",
    "LohcRateConfig",
    "LohcRateResult",
    "LohcRateWriteTarget",
]
