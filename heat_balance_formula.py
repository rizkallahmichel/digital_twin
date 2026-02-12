"""Heat balance, energy efficiency, and space time yield calculations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple

from influxdb_client import Point, WritePrecision

from formula_base import (
    BaseInfluxExperienceConfig,
    InfluxCalculatorBase,
    ScalarSource,
    pick_timestamp,
)


@dataclass(frozen=True)
class HeatBalanceConfig(BaseInfluxExperienceConfig):
    """Configuration inputs for the heat balance calculator."""

    hydrogenation_rate: ScalarSource  # mol/s of 8HNMID
    storage_rate_multiplier: float
    reaction_enthalpy_kj_per_mol: float
    mixture_mass: ScalarSource  # kg
    mixture_heat_capacity: ScalarSource  # kJ/(kg*K)
    reactor_temp: ScalarSource  # current reactor temperature (C)
    reactor_temp_prev: ScalarSource  # previous reactor temperature (C)
    accumulation_interval_seconds: float  # seconds between Tr1 and Tr2
    jacket_temp: ScalarSource  # thermostat/jacket temperature (C)
    ambient_temp: ScalarSource  # ambient temperature (C)
    ua_coefficient: ScalarSource  # kJ/(s*K)
    alpha_loss: ScalarSource  # kJ/(s*K)
    agitator_power: ScalarSource  # kJ/s
    hydrogen_heat_capacity: Optional[ScalarSource] = None  # kJ/(kg*K)
    hydrogen_mass_dosed: Optional[ScalarSource] = None  # kg
    thermostat_power_limit: Optional[float] = None  # kJ/s
    lower_heating_value: float = 241.8  # kJ/mol
    molar_mass_h2_kg: float = 0.002016  # kg/mol
    reactor_volume_m3: float = 0.01  # m^3


@dataclass(frozen=True)
class HeatBalanceResult:
    """Outputs for the heat balance calculation."""

    experience: str
    timestamp: datetime
    hydrogenation_rate: float
    hydrogen_storage_rate: float
    q_flow: float
    q_accu: float
    q_loss: float
    q_dos: float
    q_net_measured: float
    q_net_theoretical: float
    thermostat_limit: Optional[float]
    q_net_minus_limit: Optional[float]
    efficiency: float
    mass_rate_h2: float
    space_time_yield: float


@dataclass(frozen=True)
class HeatBalanceWriteTarget:
    """Where the heat balance related metrics should be written."""

    bucket: str
    heat_measurement: str
    energy_measurement: str
    sty_measurement: str
    field: str = "value"


class HeatBalanceCalculator(InfluxCalculatorBase):
    """Calculator for reactor heat balance, energy efficiency, and STY."""

    def __init__(self, config: HeatBalanceConfig) -> None:
        super().__init__(config)
        self.config = config

    def compute(self) -> HeatBalanceResult:
        (
            hydrogenation_rate,
            hydrogenation_time,
        ) = self._resolve_scalar(self.config.hydrogenation_rate)
        storage_rate = hydrogenation_rate * self.config.storage_rate_multiplier

        mass, mass_time = self._resolve_scalar(self.config.mixture_mass)
        cp_mixture, cp_time = self._resolve_scalar(self.config.mixture_heat_capacity)
        reactor_temp, tr_time = self._resolve_scalar(self.config.reactor_temp)
        reactor_temp_prev, tr_prev_time = self._resolve_scalar(self.config.reactor_temp_prev)
        jacket_temp, jacket_time = self._resolve_scalar(self.config.jacket_temp)
        ambient_temp, ambient_time = self._resolve_scalar(self.config.ambient_temp)
        ua_coeff, ua_time = self._resolve_scalar(self.config.ua_coefficient)
        alpha_loss, alpha_time = self._resolve_scalar(self.config.alpha_loss)
        agitator_power, agitator_time = self._resolve_scalar(self.config.agitator_power)

        q_flow = ua_coeff * (reactor_temp - jacket_temp)
        delta_temp = reactor_temp - reactor_temp_prev
        if self.config.accumulation_interval_seconds <= 0:
            raise RuntimeError("Heat accumulation interval must be positive.")
        q_accu = (mass * cp_mixture * delta_temp) / self.config.accumulation_interval_seconds
        q_loss = alpha_loss * (reactor_temp - ambient_temp)

        q_dos = 0.0
        if self.config.hydrogen_mass_dosed and self.config.hydrogen_heat_capacity:
            mass_h2, mass_h2_time = self._resolve_scalar(self.config.hydrogen_mass_dosed)
            cp_h2, cp_h2_time = self._resolve_scalar(self.config.hydrogen_heat_capacity)
            q_dos = mass_h2 * cp_h2 * (reactor_temp - ambient_temp)
            timestamps = [mass_h2_time, cp_h2_time]
        else:
            timestamps = []

        q_net_measured = q_accu + q_flow + q_loss + q_dos
        q_net_theoretical = self.config.reaction_enthalpy_kj_per_mol * storage_rate

        denominator = abs(q_net_measured) + max(agitator_power, 0.0)
        efficiency = (storage_rate * self.config.lower_heating_value / denominator) if denominator else 0.0

        mass_rate_h2 = storage_rate * self.config.molar_mass_h2_kg
        if self.config.reactor_volume_m3 <= 0:
            raise RuntimeError("Reactor volume must be positive.")
        space_time_yield = (mass_rate_h2 * 3600.0) / self.config.reactor_volume_m3

        q_net_minus_limit: Optional[float] = None
        if self.config.thermostat_power_limit is not None:
            q_net_minus_limit = q_net_measured - self.config.thermostat_power_limit

        timestamp = pick_timestamp(
            hydrogenation_time,
            mass_time,
            cp_time,
            tr_time,
            tr_prev_time,
            jacket_time,
            ambient_time,
            ua_time,
            alpha_time,
            agitator_time,
            *timestamps,
        )

        return HeatBalanceResult(
            experience=self.config.experience,
            timestamp=timestamp,
            hydrogenation_rate=hydrogenation_rate,
            hydrogen_storage_rate=storage_rate,
            q_flow=q_flow,
            q_accu=q_accu,
            q_loss=q_loss,
            q_dos=q_dos,
            q_net_measured=q_net_measured,
            q_net_theoretical=q_net_theoretical,
            thermostat_limit=self.config.thermostat_power_limit,
            q_net_minus_limit=q_net_minus_limit,
            efficiency=efficiency,
            mass_rate_h2=mass_rate_h2,
            space_time_yield=space_time_yield,
        )

    def write_result(self, result: HeatBalanceResult, target: HeatBalanceWriteTarget) -> None:
        """Persist the derived heat, energy, and STY metrics."""

        heat_point = Point(target.heat_measurement).tag(self.config.tag_key, self.config.experience)
        heat_point.field("hydrogenation_rate_mol_s", result.hydrogenation_rate)
        heat_point.field("hydrogen_storage_rate_mol_s", result.hydrogen_storage_rate)
        heat_point.field("q_flow_kj_s", result.q_flow)
        heat_point.field("q_accu_kj_s", result.q_accu)
        heat_point.field("q_loss_kj_s", result.q_loss)
        heat_point.field("q_dos_kj_s", result.q_dos)
        heat_point.field("q_net_measured_kj_s", result.q_net_measured)
        heat_point.field("q_net_theoretical_kj_s", result.q_net_theoretical)
        if result.thermostat_limit is not None:
            heat_point.field("thermostat_limit_kj_s", result.thermostat_limit)
        if result.q_net_minus_limit is not None:
            heat_point.field("q_net_minus_limit_kj_s", result.q_net_minus_limit)
        heat_point.time(result.timestamp, WritePrecision.NS)

        energy_point = (
            Point(target.energy_measurement)
            .tag(self.config.tag_key, self.config.experience)
            .field("efficiency_ratio", result.efficiency)
            .field("mass_rate_h2_kg_s", result.mass_rate_h2)
            .time(result.timestamp, WritePrecision.NS)
        )

        sty_point = (
            Point(target.sty_measurement)
            .tag(self.config.tag_key, self.config.experience)
            .field("sty_kg_m3_h", result.space_time_yield)
            .time(result.timestamp, WritePrecision.NS)
        )

        self.write_api.write(
            bucket=target.bucket,
            org=self.config.org,
            record=[heat_point, energy_point, sty_point],
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


__all__ = [
    "HeatBalanceCalculator",
    "HeatBalanceConfig",
    "HeatBalanceResult",
    "HeatBalanceWriteTarget",
]
