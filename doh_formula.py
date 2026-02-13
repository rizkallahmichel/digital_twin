"""Degree of Hydrogenation (DoH) calculations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable, List, Optional, Tuple

from influxdb_client import Point, WritePrecision

from faraday_monitor import SignalSelection
from formula_base import BaseInfluxExperienceConfig, InfluxCalculatorBase, ScalarSource, pick_timestamp


@dataclass(frozen=True)
class DoHConfig(BaseInfluxExperienceConfig):
    """Configuration for Degree of Hydrogenation (DoH) computations."""

    hydrogen_volume: ScalarSource
    pressure: ScalarSource
    temperature: ScalarSource
    lohc_mass: ScalarSource
    lohc_molar_mass: float
    lohc_volume: Optional[ScalarSource]
    lohc_density: Optional[ScalarSource]
    gas_constant: float
    reactor_volume_liters: float
    volume_to_m3_factor: float


@dataclass(frozen=True)
class DoHResult:
    """Snapshot of the derived Degree of Hydrogenation outputs."""

    experience: str
    timestamp: datetime
    doh_ratio: float
    stored_h2_moles: float
    max_h2_moles: float
    hydrogen_volume_liters: float
    net_hydrogen_volume_liters: float


@dataclass(frozen=True)
class DoHWriteTarget:
    """Describes where the DoH ratio should be stored in InfluxDB."""

    bucket: str
    measurement: str
    field: str = "value"


class DoHCalculator(InfluxCalculatorBase):
    """Binds Influx access and DoH-specific calculations."""

    def __init__(self, config: DoHConfig) -> None:
        super().__init__(config)
        self.config = config

    def compute(self) -> DoHResult:
        timestamps: List[Optional[datetime]] = []

        hydrogen_volume_liters, hv_time = self._resolve_scalar(self.config.hydrogen_volume)
        timestamps.append(hv_time)

        pressure_pa, pressure_time = self._resolve_scalar(self.config.pressure)
        timestamps.append(pressure_time)

        temperature_k, temp_time = self._resolve_scalar(self.config.temperature)
        timestamps.append(temp_time)
        if temperature_k <= 0:
            raise RuntimeError("Temperature must be above 0 K for DoH calculation.")

        lohc_mass, mass_time = self._resolve_scalar(self.config.lohc_mass)
        timestamps.append(mass_time)

        lohc_volume_liters: float
        if self.config.lohc_volume is not None:
            lohc_volume_liters, volume_time = self._resolve_scalar(self.config.lohc_volume)
            timestamps.append(volume_time)
        else:
            if self.config.lohc_density is None:
                raise RuntimeError("Provide LOHC volume or density so it can be derived from mass.")
            lohc_density, density_time = self._resolve_scalar(self.config.lohc_density)
            timestamps.append(density_time)
            if lohc_density <= 0:
                raise RuntimeError("LOHC density must be positive.")
            lohc_volume_liters = lohc_mass / lohc_density

        timestamp = pick_timestamp(*timestamps)
        return self._build_result_from_values(
            timestamp=timestamp,
            hydrogen_volume_liters=hydrogen_volume_liters,
            pressure_pa=pressure_pa,
            temperature_k=temperature_k,
            lohc_mass=lohc_mass,
            lohc_volume_liters=lohc_volume_liters,
        )

    def write_result(self, result: DoHResult, target: DoHWriteTarget) -> None:
        """Persist the DoH ratio back to InfluxDB."""

        point = (
            Point(target.measurement)
            .tag(self.config.tag_key, self.config.experience)
            .field(target.field, result.doh_ratio)
            .time(result.timestamp, WritePrecision.NS)
        )
        self.write_api.write(bucket=target.bucket, org=self.config.org, record=[point])

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
    ) -> Tuple[Iterable[DoHResult], Optional[datetime], Optional[datetime]]:
        if step_seconds <= 0:
            raise RuntimeError("Sampling interval must be greater than zero seconds.")

        hydrogen_volume = self._timeline_for_scalar(
            self.config.hydrogen_volume, start, stop, required=True, allow_constant=True
        )
        pressure = self._timeline_for_scalar(self.config.pressure, start, stop, required=True, allow_constant=True)
        temperature = self._timeline_for_scalar(
            self.config.temperature, start, stop, required=True, allow_constant=True
        )
        lohc_mass = self._timeline_for_scalar(self.config.lohc_mass, start, stop, required=True, allow_constant=True)

        lohc_volume_timeline: Optional[_ScalarTimeline] = None
        lohc_density_timeline: Optional[_ScalarTimeline] = None
        if self.config.lohc_volume is not None:
            lohc_volume_timeline = self._timeline_for_scalar(
                self.config.lohc_volume, start, stop, required=True, allow_constant=True
            )
        else:
            if self.config.lohc_density is None:
                raise RuntimeError("Provide LOHC volume or density so it can be derived from mass.")
            lohc_density_timeline = self._timeline_for_scalar(
                self.config.lohc_density, start, stop, required=True, allow_constant=True
            )

        timelines = [hydrogen_volume, pressure, temperature, lohc_mass]
        if lohc_volume_timeline is not None:
            timelines.append(lohc_volume_timeline)
        elif lohc_density_timeline is not None:
            timelines.append(lohc_density_timeline)

        start_candidates = [ts for ts in (_t.first_timestamp for _t in timelines) if ts is not None]
        end_candidates = [ts for ts in (_t.last_timestamp for _t in timelines) if ts is not None]
        actual_start = max([start] + start_candidates) if start_candidates else None
        actual_end = min([stop] + end_candidates) if end_candidates else None
        if actual_start is None or actual_end is None or actual_end < actual_start:
            return iter(()), actual_start, actual_end

        step = timedelta(seconds=step_seconds)

        def generator() -> Iterable[DoHResult]:
            ts = actual_start
            while ts <= actual_end:
                hydrogen_volume_liters = hydrogen_volume.value_at(ts)
                pressure_pa = pressure.value_at(ts)
                temperature_k = temperature.value_at(ts)
                if temperature_k <= 0:
                    raise RuntimeError("Temperature must be above 0 K for DoH calculation.")
                lohc_mass_value = lohc_mass.value_at(ts)
                if lohc_volume_timeline is not None:
                    lohc_volume_liters = lohc_volume_timeline.value_at(ts)
                else:
                    density_value = lohc_density_timeline.value_at(ts) if lohc_density_timeline else None
                    if density_value is None or density_value <= 0:
                        raise RuntimeError("LOHC density must be positive.")
                    lohc_volume_liters = lohc_mass_value / density_value

                yield self._build_result_from_values(
                    timestamp=ts,
                    hydrogen_volume_liters=hydrogen_volume_liters,
                    pressure_pa=pressure_pa,
                    temperature_k=temperature_k,
                    lohc_mass=lohc_mass_value,
                    lohc_volume_liters=lohc_volume_liters,
                )
                ts += step

        return generator(), actual_start, actual_end

    def _timeline_for_scalar(
        self,
        source: ScalarSource,
        start: datetime,
        stop: datetime,
        required: bool,
        allow_constant: bool,
    ) -> _ScalarTimeline:
        if source.fixed_value is not None:
            if not allow_constant:
                raise RuntimeError(
                    f"Scalar '{source.name}' must reference an Influx signal when replaying a completed experience."
                )
            return _ScalarTimeline(label=source.name, constant=source.fixed_value)

        if source.signal is None:
            raise RuntimeError(f"Scalar '{source.name}' is missing both a fixed value and a signal selection.")

        series = self._fetch_series(source.signal, start, stop)
        if not series:
            raise RuntimeError(
                f"No readings found for '{source.name}' in bucket '{source.signal.bucket}' "
                f"({source.signal.measurement}/{source.signal.field}) between {start} and {stop}."
            )
        return _ScalarTimeline(label=source.name, series=series)

    def _build_result_from_values(
        self,
        timestamp: datetime,
        hydrogen_volume_liters: float,
        pressure_pa: float,
        temperature_k: float,
        lohc_mass: float,
        lohc_volume_liters: float,
    ) -> DoHResult:
        headspace_liters = self.config.reactor_volume_liters - lohc_volume_liters
        if headspace_liters < 0:
            headspace_liters = 0.0

        net_hydrogen_volume_liters = hydrogen_volume_liters - headspace_liters
        if net_hydrogen_volume_liters < 0:
            net_hydrogen_volume_liters = 0.0

        stored_h2_moles = (
            pressure_pa
            * (net_hydrogen_volume_liters * self.config.volume_to_m3_factor)
            / (self.config.gas_constant * temperature_k)
        )

        if self.config.lohc_molar_mass <= 0:
            raise RuntimeError("LOHC molar mass must be positive.")
        n_nmid = lohc_mass / self.config.lohc_molar_mass
        max_h2_moles = 4.0 * n_nmid
        if max_h2_moles <= 0:
            raise RuntimeError("Maximum hydrogen moles computed as zero; check LOHC inputs.")

        doh_ratio = stored_h2_moles / max_h2_moles

        return DoHResult(
            experience=self.config.experience,
            timestamp=timestamp,
            doh_ratio=doh_ratio,
            stored_h2_moles=stored_h2_moles,
            max_h2_moles=max_h2_moles,
            hydrogen_volume_liters=hydrogen_volume_liters,
            net_hydrogen_volume_liters=net_hydrogen_volume_liters,
        )


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
    "DoHCalculator",
    "DoHConfig",
    "DoHResult",
    "DoHWriteTarget",
    "ScalarSource",
]
