"""Degree of Hydrogenation (DoH) calculations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple

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
        timestamp = pick_timestamp(*timestamps)

        return DoHResult(
            experience=self.config.experience,
            timestamp=timestamp,
            doh_ratio=doh_ratio,
            stored_h2_moles=stored_h2_moles,
            max_h2_moles=max_h2_moles,
            hydrogen_volume_liters=hydrogen_volume_liters,
            net_hydrogen_volume_liters=net_hydrogen_volume_liters,
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


__all__ = [
    "DoHCalculator",
    "DoHConfig",
    "DoHResult",
    "DoHWriteTarget",
    "ScalarSource",
]
