"""Heat balance, energy efficiency, and space time yield calculations."""

from __future__ import annotations

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

        mass, mass_time = self._resolve_scalar(self.config.mixture_mass)
        cp_mixture, cp_time = self._resolve_scalar(self.config.mixture_heat_capacity)
        reactor_temp, tr_time = self._resolve_scalar(self.config.reactor_temp)
        reactor_temp_prev, tr_prev_time = self._resolve_scalar(self.config.reactor_temp_prev)
        jacket_temp, jacket_time = self._resolve_scalar(self.config.jacket_temp)
        ambient_temp, ambient_time = self._resolve_scalar(self.config.ambient_temp)
        ua_coeff, ua_time = self._resolve_scalar(self.config.ua_coefficient)
        alpha_loss, alpha_time = self._resolve_scalar(self.config.alpha_loss)
        agitator_power, agitator_time = self._resolve_scalar(self.config.agitator_power)

        mass_h2 = None
        cp_h2 = None
        timestamps: List[Optional[datetime]] = []
        if self.config.hydrogen_mass_dosed and self.config.hydrogen_heat_capacity:
            mass_h2, mass_h2_time = self._resolve_scalar(self.config.hydrogen_mass_dosed)
            cp_h2, cp_h2_time = self._resolve_scalar(self.config.hydrogen_heat_capacity)
            timestamps.extend([mass_h2_time, cp_h2_time])

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

        return self._build_result_from_values(
            timestamp=timestamp,
            hydrogenation_rate=hydrogenation_rate,
            mass=mass,
            cp_mixture=cp_mixture,
            reactor_temp=reactor_temp,
            reactor_temp_prev=reactor_temp_prev,
            jacket_temp=jacket_temp,
            ambient_temp=ambient_temp,
            ua_coeff=ua_coeff,
            alpha_loss=alpha_loss,
            agitator_power=agitator_power,
            mass_h2=mass_h2,
            cp_h2=cp_h2,
        )

    def write_result(self, result: HeatBalanceResult, target: HeatBalanceWriteTarget) -> None:
        """Persist the derived heat, energy, and STY metrics."""

        points = self._result_points(result, target)
        self.write_api.write(
            bucket=target.bucket,
            org=self.config.org,
            record=points,
        )

    def iter_experience(
        self,
        start: datetime,
        stop: datetime,
        step_seconds: float,
    ) -> Tuple[Iterable[HeatBalanceResult], Optional[datetime], Optional[datetime]]:
        if step_seconds <= 0:
            raise RuntimeError("Sampling interval must be greater than zero seconds.")

        required_timelines = {
            "hydrogenation": self._timeline_for_scalar(self.config.hydrogenation_rate, start, stop, required=True),
            "mass": self._timeline_for_scalar(self.config.mixture_mass, start, stop, required=True),
            "cp_mixture": self._timeline_for_scalar(self.config.mixture_heat_capacity, start, stop, required=True),
            "reactor_temp": self._timeline_for_scalar(self.config.reactor_temp, start, stop, required=True),
            "reactor_temp_prev": self._timeline_for_scalar(
                self.config.reactor_temp_prev, start, stop, required=True
            ),
            "jacket_temp": self._timeline_for_scalar(self.config.jacket_temp, start, stop, required=True),
            "ambient_temp": self._timeline_for_scalar(self.config.ambient_temp, start, stop, required=True),
            "ua_coeff": self._timeline_for_scalar(self.config.ua_coefficient, start, stop, required=True),
            "alpha_loss": self._timeline_for_scalar(self.config.alpha_loss, start, stop, required=True),
            "agitator_power": self._timeline_for_scalar(self.config.agitator_power, start, stop, required=True),
        }

        optional_timelines = {}
        if self.config.hydrogen_mass_dosed and self.config.hydrogen_heat_capacity:
            optional_timelines["mass_h2"] = self._timeline_for_scalar(
                self.config.hydrogen_mass_dosed, start, stop, required=False
            )
            optional_timelines["cp_h2"] = self._timeline_for_scalar(
                self.config.hydrogen_heat_capacity, start, stop, required=False
            )

        all_timelines = list(required_timelines.values()) + [
            tl for tl in optional_timelines.values() if tl is not None
        ]
        start_candidates = [
            ts for ts in (_timeline.first_timestamp for _timeline in all_timelines) if ts is not None
        ]
        end_candidates = [ts for ts in (_timeline.last_timestamp for _timeline in all_timelines) if ts is not None]
        actual_start = max([start] + start_candidates) if start_candidates else start
        actual_end = min([stop] + end_candidates) if end_candidates else stop
        if actual_end < actual_start:
            return iter(()), actual_start, actual_end

        step = timedelta(seconds=step_seconds)

        def generator() -> Iterable[HeatBalanceResult]:
            ts = actual_start
            while ts <= actual_end:
                hydrogenation_rate = required_timelines["hydrogenation"].value_at(ts)
                mass = required_timelines["mass"].value_at(ts)
                cp_mixture = required_timelines["cp_mixture"].value_at(ts)
                reactor_temp = required_timelines["reactor_temp"].value_at(ts)
                reactor_temp_prev = required_timelines["reactor_temp_prev"].value_at(ts)
                jacket_temp = required_timelines["jacket_temp"].value_at(ts)
                ambient_temp = required_timelines["ambient_temp"].value_at(ts)
                ua_coeff = required_timelines["ua_coeff"].value_at(ts)
                alpha_loss = required_timelines["alpha_loss"].value_at(ts)
                agitator_power = required_timelines["agitator_power"].value_at(ts)

                mass_h2 = optional_timelines.get("mass_h2").value_at(ts) if optional_timelines.get("mass_h2") else None
                cp_h2 = optional_timelines.get("cp_h2").value_at(ts) if optional_timelines.get("cp_h2") else None

                yield self._build_result_from_values(
                    timestamp=ts,
                    hydrogenation_rate=hydrogenation_rate,
                    mass=mass,
                    cp_mixture=cp_mixture,
                    reactor_temp=reactor_temp,
                    reactor_temp_prev=reactor_temp_prev,
                    jacket_temp=jacket_temp,
                    ambient_temp=ambient_temp,
                    ua_coeff=ua_coeff,
                    alpha_loss=alpha_loss,
                    agitator_power=agitator_power,
                    mass_h2=mass_h2,
                    cp_h2=cp_h2,
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

    def _timeline_for_scalar(
        self,
        source: ScalarSource,
        start: datetime,
        stop: datetime,
        required: bool,
    ) -> "_ScalarTimeline":
        if source.fixed_value is not None:
            return _ScalarTimeline(label=source.name, constant=source.fixed_value)
        if source.signal is None:
            if required:
                raise RuntimeError(f"Scalar '{source.name}' is missing both a fixed value and a signal selection.")
            return _ScalarTimeline(label=source.name, constant=None)

        series = self._fetch_series(source.signal, start, stop)
        if not series:
            if required:
                raise RuntimeError(
                    f"No readings found for '{source.name}' in bucket '{source.signal.bucket}' "
                    f"({source.signal.measurement}/{source.signal.field}) between {start} and {stop}."
                )
            return _ScalarTimeline(label=source.name, constant=None)
        return _ScalarTimeline(label=source.name, series=series)

    def _build_result_from_values(
        self,
        timestamp: datetime,
        hydrogenation_rate: float,
        mass: float,
        cp_mixture: float,
        reactor_temp: float,
        reactor_temp_prev: float,
        jacket_temp: float,
        ambient_temp: float,
        ua_coeff: float,
        alpha_loss: float,
        agitator_power: float,
        mass_h2: Optional[float],
        cp_h2: Optional[float],
    ) -> HeatBalanceResult:
        storage_rate = hydrogenation_rate * self.config.storage_rate_multiplier
        q_flow = ua_coeff * (reactor_temp - jacket_temp)
        delta_temp = reactor_temp - reactor_temp_prev
        if self.config.accumulation_interval_seconds <= 0:
            raise RuntimeError("Heat accumulation interval must be positive.")
        q_accu = (mass * cp_mixture * delta_temp) / self.config.accumulation_interval_seconds
        q_loss = alpha_loss * (reactor_temp - ambient_temp)

        q_dos = 0.0
        if mass_h2 is not None and cp_h2 is not None:
            q_dos = mass_h2 * cp_h2 * (reactor_temp - ambient_temp)

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

    def _result_points(self, result: HeatBalanceResult, target: HeatBalanceWriteTarget) -> List[Point]:
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

        return [heat_point, energy_point, sty_point]


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
    "HeatBalanceCalculator",
    "HeatBalanceConfig",
    "HeatBalanceResult",
    "HeatBalanceWriteTarget",
]
