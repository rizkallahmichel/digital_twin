"""LOHC hydrogenation and hydrogen storage rate calculations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple

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
        if doh_ratio < 0:
            raise RuntimeError("DoH ratio must be non-negative.")

        density, density_time = self._resolve_scalar(self.config.lohc_density)
        timestamps.append(density_time)
        if density <= 0:
            raise RuntimeError("LOHC density must be positive.")

        if self.config.lohc_molar_mass <= 0:
            raise RuntimeError("LOHC molar mass must be positive.")

        nmih_concentration = density / self.config.lohc_molar_mass

        hydrogenated_concentration = doh_ratio * nmih_concentration

        partial_pressure, pp_time = self._resolve_scalar(self.config.partial_pressure)
        timestamps.append(pp_time)

        henry_constant, henry_time = self._resolve_scalar(self.config.henry_constant)
        timestamps.append(henry_time)
        if henry_constant == 0:
            raise RuntimeError("Henry constant must be non-zero.")

        volumetric_fraction, vf_time = self._resolve_scalar(self.config.volumetric_fraction)
        timestamps.append(vf_time)

        temperature_k, temp_time = self._resolve_scalar(self.config.temperature)
        timestamps.append(temp_time)
        if temperature_k <= 0:
            raise RuntimeError("Temperature must be above 0 K.")

        kinetic_constant, k_time = self._resolve_kinetic_constant(temperature_k)
        timestamps.append(k_time)

        hydrogenation_rate = (
            volumetric_fraction * kinetic_constant * hydrogenated_concentration * partial_pressure / henry_constant
        )
        hydrogen_storage_rate = 4.0 * hydrogenation_rate

        timestamp = pick_timestamp(*timestamps)

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

        if self.config.pre_exponential is None or self.config.activation_energy is None:
            raise RuntimeError(
                "Provide either --lohc-k inputs or both --lohc-k0 and --lohc-activation-energy for Arrhenius calculation."
            )

        kinetic_constant = self.config.pre_exponential * math.exp(
            -self.config.activation_energy / (self.config.gas_constant * temperature_k)
        )
        return kinetic_constant, None

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
    "LohcRateCalculator",
    "LohcRateConfig",
    "LohcRateResult",
    "LohcRateWriteTarget",
]
