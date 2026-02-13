#!/usr/bin/env python3
"""CLI entry point for hydrogen formula computations filtered by an experience tag."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

from faraday_monitor import (
    DEFAULT_RANGE_WINDOW,
    ELECTRONS_PER_H2,
    FARADAY_CONSTANT,
    MOLAR_VOLUME_NL,
    SignalSelection,
)
from formula_base import ScalarSource
from doh_formula import DoHCalculator, DoHConfig, DoHResult, DoHWriteTarget
from faraday_formula import FaradayCalculator, FaradayConfig, FaradayResult, FaradayWriteTarget
from lohc_rate_formula import (
    LohcRateCalculator,
    LohcRateConfig,
    LohcRateResult,
    LohcRateWriteTarget,
)
from heat_balance_formula import (
    HeatBalanceCalculator,
    HeatBalanceConfig,
    HeatBalanceResult,
    HeatBalanceWriteTarget,
)
from nas_config import NasInfluxDefaults, influx_cli_defaults, resolve_defaults


WINDOW_PATTERN = re.compile(r"^\s*(\d+)\s*([smhdw])\s*$", re.IGNORECASE)


def _env_float(name: str, default: Optional[float] = None) -> Optional[float]:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise SystemExit(f"Environment variable {name} must be a float; received {value!r}.") from exc


def _parse_timestamp_arg(label: str, value: Optional[str]) -> Optional[datetime]:
    if value is None:
        return None
    normalized = value.strip()
    if not normalized:
        return None
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise SystemExit(
            f"{label} must be an ISO-8601 timestamp such as 2026-02-12T14:52:00Z (received {value!r})."
        ) from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _parse_range_window(label: str, window: str) -> timedelta:
    match = WINDOW_PATTERN.match(window or "")
    if not match:
        raise SystemExit(f"{label} must follow the '<value><unit>' format (e.g. 30d, 12h, 15m).")
    value = int(match.group(1))
    unit = match.group(2).lower()
    if value <= 0:
        raise SystemExit(f"{label} must be greater than zero.")
    multiplier = {
        "s": 1,
        "m": 60,
        "h": 3600,
        "d": 86400,
        "w": 604800,
    }[unit]
    return timedelta(seconds=value * multiplier)


def parse_args(defaults: NasInfluxDefaults) -> argparse.Namespace:
    """Set up CLI flags and return parsed arguments."""
    cli_defaults = influx_cli_defaults(defaults)
    parser = argparse.ArgumentParser(
        description="Fetch the latest readings for an experience tag and compute hydrogen formulas.",
    )
    parser.add_argument("experience", help="Experience/sensor tag value to filter on (e.g. test2).")
    parser.add_argument(
        "--formula",
        choices=("faraday", "doh", "lohc_rate", "heat_balance"),
        default=os.getenv("H2_FORMULA", "faraday"),
        help="Select which formula to run (faraday, doh, lohc_rate, heat_balance). Default: faraday.",
    )
    parser.add_argument(
        "--tag-key",
        default=os.getenv("H2_EXPERIENCE_TAG_KEY", "sensor"),
        help="Tag key that stores the experience name (default: sensor).",
    )
    parser.add_argument(
        "--url",
        default=cli_defaults["url"],
        help=f"InfluxDB URL (default: {cli_defaults['url']}).",
    )
    parser.add_argument(
        "--token",
        default=cli_defaults["token"],
        help="InfluxDB API token (defaults to INFLUX_TOKEN or NAS credential fallback).",
    )
    parser.add_argument(
        "--org",
        default=cli_defaults["org"],
        help="InfluxDB organization (defaults to INFLUX_ORG or NAS credential fallback).",
    )

    parser.add_argument("--current-bucket", default=os.getenv("H2_CURRENT_BUCKET"), help="Bucket for current data.")
    parser.add_argument(
        "--current-measurement",
        default=os.getenv("H2_CURRENT_MEASUREMENT", "Actual_Current"),
        help="Measurement name storing current readings.",
    )
    parser.add_argument(
        "--current-field",
        default=os.getenv("H2_CURRENT_FIELD", "value"),
        help="Field that stores the current values.",
    )

    parser.add_argument(
        "--efficiency-bucket",
        default=os.getenv("H2_EFFICIENCY_BUCKET"),
        help="Bucket for efficiency data (defaults to H2_EFFICIENCY_BUCKET env).",
    )
    parser.add_argument(
        "--efficiency-measurement",
        default=os.getenv("H2_EFFICIENCY_MEASUREMENT"),
        help="Measurement for efficiency readings.",
    )
    parser.add_argument(
        "--efficiency-field",
        default=os.getenv("H2_EFFICIENCY_FIELD", "value"),
        help="Field for efficiency values.",
    )
    parser.add_argument(
        "--efficiency-fixed-percent",
        type=float,
        default=float(os.getenv("H2_EFFICIENCY_FIXED_PERCENT", 60.0)),
        help="Use this fixed efficiency percentage instead of querying (default: 60).",
    )
    parser.add_argument(
        "--efficiency-use-signal",
        action="store_true",
        help="Ignore the fixed efficiency value and query Influx for efficiency readings.",
    )

    parser.add_argument(
        "--range-window",
        default=os.getenv("H2_RANGE_WINDOW", DEFAULT_RANGE_WINDOW),
        help="Flux range window when searching for the latest point (e.g. 5m).",
    )
    parser.add_argument(
        "--molar-volume",
        type=float,
        default=float(os.getenv("H2_MOLAR_VOLUME", MOLAR_VOLUME_NL)),
        help="Molar volume used for volumetric conversion (NL/mol).",
    )
    parser.add_argument(
        "--electrons-per-molecule",
        type=float,
        default=float(os.getenv("H2_ELECTRONS_PER_MOLECULE", ELECTRONS_PER_H2)),
        help="Number of electrons required for one H2 molecule.",
    )
    parser.add_argument(
        "--faraday-constant",
        type=float,
        default=float(os.getenv("H2_FARADAY_CONSTANT", FARADAY_CONSTANT)),
        help="Faraday constant to use (C/mol).",
    )

    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument(
        "--efficiency-percent",
        dest="efficiency_percent",
        action="store_true",
        help="Treat efficiency readings as percentages (default).",
    )
    output_group.add_argument(
        "--efficiency-decimal",
        dest="efficiency_percent",
        action="store_false",
        help="Treat efficiency readings as ratios in [0, 1].",
    )
    parser.set_defaults(efficiency_percent=True)

    parser.add_argument(
        "--write-results",
        action="store_true",
        help="Persist the computed molar/volumetric rates back into InfluxDB.",
    )
    parser.add_argument(
        "--result-bucket",
        default=os.getenv("H2_RESULT_BUCKET"),
        help="Bucket used when writing computed results (defaults to --current-bucket).",
    )
    parser.add_argument(
        "--result-molar-measurement",
        default=os.getenv("H2_RESULT_MOLAR_MEASUREMENT", "th_molar_rate"),
        help="Measurement name for the theoretical molar rate.",
    )
    parser.add_argument(
        "--result-volumetric-measurement",
        default=os.getenv("H2_RESULT_VOL_MEASUREMENT", "th_volumetric_rate"),
        help="Measurement name for the theoretical volumetric rate.",
    )
    parser.add_argument(
        "--result-field",
        default=os.getenv("H2_RESULT_FIELD", "value"),
        help="Field name that stores computed rate values.",
    )

    parser.add_argument(
        "--log-level",
        default=os.getenv("H2_EXPERIENCE_LOG_LEVEL", "INFO"),
        help="Python logging level (DEBUG, INFO, ...).",
    )
    parser.add_argument(
        "--output",
        choices=("text", "json"),
        default=os.getenv("H2_EXPERIENCE_OUTPUT", "text"),
        help="Output format for the computed result.",
    )
    parser.add_argument(
        "--sample-interval",
        type=float,
        default=float(os.getenv("H2_SAMPLE_INTERVAL", "0.5")),
        help="Seconds between samples when streaming continuously (default: 0.5).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=int(os.getenv("H2_MAX_SAMPLES", "0")),
        help="Stop after this many samples when > 0; otherwise run until interrupted.",
    )
    parser.add_argument(
        "--single-shot",
        action="store_true",
        help="Compute a single sample and exit (legacy behavior).",
    )
    parser.add_argument(
        "--experience-finished",
        action="store_true",
        help="Replay stored data for a completed experience instead of live streaming.",
    )
    parser.add_argument(
        "--experience-start",
        help="ISO-8601 timestamp marking the desired start of the experience window (UTC assumed when omitted).",
    )
    parser.add_argument(
        "--experience-end",
        help="ISO-8601 timestamp marking the desired end of the experience window (UTC assumed when omitted).",
    )

    doh_group = parser.add_argument_group(
        "Degree of Hydrogenation inputs",
        "Arguments used when --formula doh is selected.",
    )
    doh_group.add_argument(
        "--doh-h2-volume-bucket",
        default=os.getenv("DOH_H2_VOLUME_BUCKET"),
        help="Bucket with cumulative hydrogen volume (liters).",
    )
    doh_group.add_argument(
        "--doh-h2-volume-measurement",
        default=os.getenv("DOH_H2_VOLUME_MEASUREMENT"),
        help="Measurement for hydrogen volume readings.",
    )
    doh_group.add_argument(
        "--doh-h2-volume-field",
        default=os.getenv("DOH_H2_VOLUME_FIELD", "value"),
        help="Field storing hydrogen volume readings.",
    )
    doh_group.add_argument(
        "--doh-h2-volume-value",
        type=float,
        default=_env_float("DOH_H2_VOLUME_VALUE"),
        help="Fixed hydrogen volume in liters (skips querying Influx).",
    )

    doh_group.add_argument(
        "--doh-pressure-bucket",
        default=os.getenv("DOH_PRESSURE_BUCKET"),
        help="Bucket storing reactor pressure data (Pa).",
    )
    doh_group.add_argument(
        "--doh-pressure-measurement",
        default=os.getenv("DOH_PRESSURE_MEASUREMENT"),
        help="Measurement storing reactor pressure data.",
    )
    doh_group.add_argument(
        "--doh-pressure-field",
        default=os.getenv("DOH_PRESSURE_FIELD", "value"),
        help="Field storing reactor pressure values.",
    )
    doh_group.add_argument(
        "--doh-pressure-value",
        type=float,
        default=_env_float("DOH_PRESSURE_VALUE"),
        help="Fixed reactor pressure in Pascals.",
    )

    doh_group.add_argument(
        "--doh-temperature-bucket",
        default=os.getenv("DOH_TEMPERATURE_BUCKET"),
        help="Bucket storing reactor temperature data (Kelvin).",
    )
    doh_group.add_argument(
        "--doh-temperature-measurement",
        default=os.getenv("DOH_TEMPERATURE_MEASUREMENT"),
        help="Measurement storing temperature readings.",
    )
    doh_group.add_argument(
        "--doh-temperature-field",
        default=os.getenv("DOH_TEMPERATURE_FIELD", "value"),
        help="Field storing temperature readings.",
    )
    doh_group.add_argument(
        "--doh-temperature-value",
        type=float,
        default=_env_float("DOH_TEMPERATURE_VALUE"),
        help="Fixed temperature in Kelvin.",
    )

    doh_group.add_argument(
        "--doh-lohc-mass-bucket",
        default=os.getenv("DOH_LOHC_MASS_BUCKET"),
        help="Bucket storing LOHC mass readings.",
    )
    doh_group.add_argument(
        "--doh-lohc-mass-measurement",
        default=os.getenv("DOH_LOHC_MASS_MEASUREMENT"),
        help="Measurement storing LOHC mass readings.",
    )
    doh_group.add_argument(
        "--doh-lohc-mass-field",
        default=os.getenv("DOH_LOHC_MASS_FIELD", "value"),
        help="Field storing LOHC mass values.",
    )
    doh_group.add_argument(
        "--doh-lohc-mass-value",
        type=float,
        default=_env_float("DOH_LOHC_MASS_VALUE"),
        help="Fixed LOHC mass (consistent units with molar mass and density).",
    )

    doh_group.add_argument(
        "--doh-lohc-volume-bucket",
        default=os.getenv("DOH_LOHC_VOLUME_BUCKET"),
        help="Bucket storing LOHC volume readings (liters).",
    )
    doh_group.add_argument(
        "--doh-lohc-volume-measurement",
        default=os.getenv("DOH_LOHC_VOLUME_MEASUREMENT"),
        help="Measurement storing LOHC volume readings.",
    )
    doh_group.add_argument(
        "--doh-lohc-volume-field",
        default=os.getenv("DOH_LOHC_VOLUME_FIELD", "value"),
        help="Field storing LOHC volume readings.",
    )
    doh_group.add_argument(
        "--doh-lohc-volume-value",
        type=float,
        default=_env_float("DOH_LOHC_VOLUME_VALUE"),
        help="Fixed LOHC volume in liters (overrides density-based calculation).",
    )

    doh_group.add_argument(
        "--doh-lohc-density-bucket",
        default=os.getenv("DOH_LOHC_DENSITY_BUCKET"),
        help="Bucket storing LOHC density readings (mass per liter).",
    )
    doh_group.add_argument(
        "--doh-lohc-density-measurement",
        default=os.getenv("DOH_LOHC_DENSITY_MEASUREMENT"),
        help="Measurement storing LOHC density readings.",
    )
    doh_group.add_argument(
        "--doh-lohc-density-field",
        default=os.getenv("DOH_LOHC_DENSITY_FIELD", "value"),
        help="Field storing LOHC density values.",
    )
    doh_group.add_argument(
        "--doh-lohc-density-value",
        type=float,
        default=_env_float("DOH_LOHC_DENSITY_VALUE"),
        help="Fixed LOHC density (mass per liter).",
    )

    doh_group.add_argument(
        "--doh-lohc-molar-mass",
        type=float,
        default=_env_float("DOH_LOHC_MOLAR_MASS"),
        help="LOHC molar mass in units matching the LOHC mass input (e.g., kg/mol).",
    )
    doh_group.add_argument(
        "--doh-gas-constant",
        type=float,
        default=float(os.getenv("DOH_GAS_CONSTANT", "8.314462618")),
        help="Gas constant R in J/(mol*K). Default: 8.314462618.",
    )
    doh_group.add_argument(
        "--doh-reactor-volume-liters",
        type=float,
        default=float(os.getenv("DOH_REACTOR_VOLUME_LITERS", "10.0")),
        help="Total reactor volume in liters (default: 10.0).",
    )
    doh_group.add_argument(
        "--doh-volume-to-m3-factor",
        type=float,
        default=float(os.getenv("DOH_VOLUME_TO_M3_FACTOR", "0.001")),
        help="Conversion factor from liters to cubic meters (default: 0.001).",
    )
    doh_group.add_argument(
        "--doh-result-measurement",
        default=os.getenv("DOH_RESULT_MEASUREMENT", "doh_ratio"),
        help="Measurement name used when storing DoH ratios.",
    )

    lohc_group = parser.add_argument_group(
        "LOHC hydrogenation rate inputs",
        "Arguments used when --formula lohc_rate is selected.",
    )
    lohc_group.add_argument(
        "--lohc-doh-bucket",
        default=os.getenv("LOHC_DOH_BUCKET"),
        help="Bucket containing stored DoH ratios (0-1).",
    )
    lohc_group.add_argument(
        "--lohc-doh-measurement",
        default=os.getenv("LOHC_DOH_MEASUREMENT"),
        help="Measurement storing DoH ratios.",
    )
    lohc_group.add_argument(
        "--lohc-doh-field",
        default=os.getenv("LOHC_DOH_FIELD", "value"),
        help="Field storing DoH ratios.",
    )
    lohc_group.add_argument(
        "--lohc-doh-value",
        type=float,
        default=_env_float("LOHC_DOH_VALUE"),
        help="Fixed DoH ratio (0-1) instead of querying Influx.",
    )
    lohc_group.add_argument(
        "--lohc-density-bucket",
        default=os.getenv("LOHC_DENSITY_BUCKET"),
        help="Bucket storing LOHC density readings (mass per liter).",
    )
    lohc_group.add_argument(
        "--lohc-density-measurement",
        default=os.getenv("LOHC_DENSITY_MEASUREMENT"),
        help="Measurement storing LOHC density readings.",
    )
    lohc_group.add_argument(
        "--lohc-density-field",
        default=os.getenv("LOHC_DENSITY_FIELD", "value"),
        help="Field storing LOHC density readings.",
    )
    lohc_group.add_argument(
        "--lohc-density-value",
        type=float,
        default=_env_float("LOHC_DENSITY_VALUE"),
        help="Fixed LOHC density (mass per liter).",
    )
    lohc_group.add_argument(
        "--lohc-molar-mass",
        type=float,
        default=_env_float("LOHC_MOLAR_MASS"),
        help="LOHC molar mass (mass per mol).",
    )
    lohc_group.add_argument(
        "--lohc-ph2-bucket",
        default=os.getenv("LOHC_PH2_BUCKET"),
        help="Bucket storing partial pressure of H2 readings.",
    )
    lohc_group.add_argument(
        "--lohc-ph2-measurement",
        default=os.getenv("LOHC_PH2_MEASUREMENT"),
        help="Measurement storing partial pressure of H2 data.",
    )
    lohc_group.add_argument(
        "--lohc-ph2-field",
        default=os.getenv("LOHC_PH2_FIELD", "value"),
        help="Field storing partial pressure of H2.",
    )
    lohc_group.add_argument(
        "--lohc-ph2-value",
        type=float,
        default=_env_float("LOHC_PH2_VALUE"),
        help="Fixed partial pressure of H2 (matching Henry constant units).",
    )
    lohc_group.add_argument(
        "--lohc-henry-bucket",
        default=os.getenv("LOHC_HENRY_BUCKET"),
        help="Bucket storing Henry's law constant readings.",
    )
    lohc_group.add_argument(
        "--lohc-henry-measurement",
        default=os.getenv("LOHC_HENRY_MEASUREMENT"),
        help="Measurement storing Henry's law constant readings.",
    )
    lohc_group.add_argument(
        "--lohc-henry-field",
        default=os.getenv("LOHC_HENRY_FIELD", "value"),
        help="Field storing Henry's law constant readings.",
    )
    lohc_group.add_argument(
        "--lohc-henry-value",
        type=float,
        default=_env_float("LOHC_HENRY_VALUE"),
        help="Fixed Henry's law constant (matching pressure units).",
    )
    lohc_group.add_argument(
        "--lohc-vol-frac-bucket",
        default=os.getenv("LOHC_VOL_FRAC_BUCKET"),
        help="Bucket storing volumetric fraction of catalyst readings.",
    )
    lohc_group.add_argument(
        "--lohc-vol-frac-measurement",
        default=os.getenv("LOHC_VOL_FRAC_MEASUREMENT"),
        help="Measurement storing volumetric fraction readings.",
    )
    lohc_group.add_argument(
        "--lohc-vol-frac-field",
        default=os.getenv("LOHC_VOL_FRAC_FIELD", "value"),
        help="Field storing volumetric fraction readings.",
    )
    lohc_group.add_argument(
        "--lohc-vol-frac-value",
        type=float,
        default=_env_float("LOHC_VOL_FRAC_VALUE"),
        help="Fixed volumetric fraction of catalyst.",
    )
    lohc_group.add_argument(
        "--lohc-temperature-bucket",
        default=os.getenv("LOHC_TEMPERATURE_BUCKET"),
        help="Bucket storing reactor temperature (K).",
    )
    lohc_group.add_argument(
        "--lohc-temperature-measurement",
        default=os.getenv("LOHC_TEMPERATURE_MEASUREMENT"),
        help="Measurement storing temperature readings.",
    )
    lohc_group.add_argument(
        "--lohc-temperature-field",
        default=os.getenv("LOHC_TEMPERATURE_FIELD", "value"),
        help="Field storing temperature readings.",
    )
    lohc_group.add_argument(
        "--lohc-temperature-value",
        type=float,
        default=_env_float("LOHC_TEMPERATURE_VALUE"),
        help="Fixed reactor temperature in Kelvin.",
    )
    lohc_group.add_argument(
        "--lohc-k-bucket",
        default=os.getenv("LOHC_K_BUCKET"),
        help="Bucket storing kinetic constant values.",
    )
    lohc_group.add_argument(
        "--lohc-k-measurement",
        default=os.getenv("LOHC_K_MEASUREMENT"),
        help="Measurement storing kinetic constant values.",
    )
    lohc_group.add_argument(
        "--lohc-k-field",
        default=os.getenv("LOHC_K_FIELD", "value"),
        help="Field storing kinetic constant values.",
    )
    lohc_group.add_argument(
        "--lohc-k-value",
        type=float,
        default=_env_float("LOHC_K_VALUE"),
        help="Fixed kinetic constant (overrides Arrhenius inputs).",
    )
    lohc_group.add_argument(
        "--lohc-k0",
        type=float,
        default=_env_float("LOHC_K0"),
        help="Arrhenius pre-exponential factor K0 (used when --lohc-k-value not provided).",
    )
    lohc_group.add_argument(
        "--lohc-activation-energy",
        type=float,
        default=_env_float("LOHC_ACTIVATION_ENERGY"),
        help="Activation energy Ea in J/mol (used with --lohc-k0).",
    )
    lohc_group.add_argument(
        "--lohc-gas-constant",
        type=float,
        default=float(os.getenv("LOHC_GAS_CONSTANT", "8.314462618")),
        help="Gas constant for Arrhenius calculations (default: 8.314462618).",
    )
    lohc_group.add_argument(
        "--lohc-rate-measurement",
        default=os.getenv("LOHC_RATE_MEASUREMENT", "lohc_hydrogenation_rate"),
        help="Measurement name for LOHC hydrogenation rate outputs.",
    )
    lohc_group.add_argument(
        "--lohc-h2-rate-measurement",
        default=os.getenv("LOHC_H2_RATE_MEASUREMENT", "lohc_h2_storage_rate"),
        help="Measurement name for LOHC hydrogen storage rate outputs.",
    )

    heat_group = parser.add_argument_group(
        "Heat balance inputs",
        "Arguments used when --formula heat_balance is selected.",
    )
    heat_group.add_argument(
        "--heat-hydrogenation-rate-bucket",
        default=os.getenv("HEAT_HYDROGENATION_RATE_BUCKET"),
        help="Bucket storing LOHC hydrogenation rate (mol/s).",
    )
    heat_group.add_argument(
        "--heat-hydrogenation-rate-measurement",
        default=os.getenv("HEAT_HYDROGENATION_RATE_MEASUREMENT"),
        help="Measurement storing hydrogenation rate readings.",
    )
    heat_group.add_argument(
        "--heat-hydrogenation-rate-field",
        default=os.getenv("HEAT_HYDROGENATION_RATE_FIELD", "value"),
        help="Field storing hydrogenation rate readings.",
    )
    heat_group.add_argument(
        "--heat-hydrogenation-rate-value",
        type=float,
        default=_env_float("HEAT_HYDROGENATION_RATE_VALUE"),
        help="Fixed hydrogenation rate (mol/s).",
    )
    heat_group.add_argument(
        "--heat-storage-multiplier",
        type=float,
        default=float(os.getenv("HEAT_STORAGE_MULTIPLIER", "4.0")),
        help="Multiplier applied to hydrogenation rate to obtain hydrogen storage rate (default 4.0).",
    )
    heat_group.add_argument(
        "--heat-reaction-enthalpy",
        type=float,
        default=float(os.getenv("HEAT_REACTION_ENTHALPY", "-56.97")),
        help="Reaction enthalpy ΔH_hydro (kJ/mol H2). Default: -56.97.",
    )
    heat_group.add_argument(
        "--heat-mixture-mass-value",
        type=float,
        default=_env_float("HEAT_MIXTURE_MASS_VALUE"),
        help="Fixed reaction mixture mass (kg).",
    )
    heat_group.add_argument(
        "--heat-mixture-mass-bucket",
        default=os.getenv("HEAT_MIXTURE_MASS_BUCKET"),
        help="Bucket storing reaction mixture mass readings.",
    )
    heat_group.add_argument(
        "--heat-mixture-mass-measurement",
        default=os.getenv("HEAT_MIXTURE_MASS_MEASUREMENT"),
        help="Measurement storing reaction mixture mass readings.",
    )
    heat_group.add_argument(
        "--heat-mixture-mass-field",
        default=os.getenv("HEAT_MIXTURE_MASS_FIELD", "value"),
        help="Field storing reaction mixture mass readings.",
    )
    heat_group.add_argument(
        "--heat-mixture-cp-value",
        type=float,
        default=_env_float("HEAT_MIXTURE_CP_VALUE"),
        help="Fixed reaction mixture heat capacity (kJ/kg*K).",
    )
    heat_group.add_argument(
        "--heat-mixture-cp-bucket",
        default=os.getenv("HEAT_MIXTURE_CP_BUCKET"),
        help="Bucket storing reaction mixture heat capacity readings.",
    )
    heat_group.add_argument(
        "--heat-mixture-cp-measurement",
        default=os.getenv("HEAT_MIXTURE_CP_MEASUREMENT"),
        help="Measurement storing reaction mixture heat capacity readings.",
    )
    heat_group.add_argument(
        "--heat-mixture-cp-field",
        default=os.getenv("HEAT_MIXTURE_CP_FIELD", "value"),
        help="Field storing reaction mixture heat capacity readings.",
    )
    heat_group.add_argument(
        "--heat-reactor-temp-bucket",
        default=os.getenv("HEAT_REACTOR_TEMP_BUCKET"),
        help="Bucket storing current reactor temperature (°C).",
    )
    heat_group.add_argument(
        "--heat-reactor-temp-measurement",
        default=os.getenv("HEAT_REACTOR_TEMP_MEASUREMENT"),
        help="Measurement storing current reactor temperature.",
    )
    heat_group.add_argument(
        "--heat-reactor-temp-field",
        default=os.getenv("HEAT_REACTOR_TEMP_FIELD", "value"),
        help="Field storing current reactor temperature.",
    )
    heat_group.add_argument(
        "--heat-reactor-temp-value",
        type=float,
        default=_env_float("HEAT_REACTOR_TEMP_VALUE"),
        help="Fixed current reactor temperature (°C).",
    )
    heat_group.add_argument(
        "--heat-reactor-prev-temp-bucket",
        default=os.getenv("HEAT_REACTOR_PREV_TEMP_BUCKET"),
        help="Bucket storing previous reactor temperature (°C).",
    )
    heat_group.add_argument(
        "--heat-reactor-prev-temp-measurement",
        default=os.getenv("HEAT_REACTOR_PREV_TEMP_MEASUREMENT"),
        help="Measurement storing previous reactor temperature.",
    )
    heat_group.add_argument(
        "--heat-reactor-prev-temp-field",
        default=os.getenv("HEAT_REACTOR_PREV_TEMP_FIELD", "value"),
        help="Field storing previous reactor temperature.",
    )
    heat_group.add_argument(
        "--heat-reactor-prev-temp-value",
        type=float,
        default=_env_float("HEAT_REACTOR_PREV_TEMP_VALUE"),
        help="Fixed previous reactor temperature (°C).",
    )
    heat_group.add_argument(
        "--heat-accu-interval-seconds",
        type=float,
        default=float(os.getenv("HEAT_ACCU_INTERVAL_SECONDS", "60.0")),
        help="Time between Tr1 and Tr2 measurements in seconds (default: 60).",
    )
    heat_group.add_argument(
        "--heat-jacket-temp-bucket",
        default=os.getenv("HEAT_JACKET_TEMP_BUCKET"),
        help="Bucket storing thermostat/jacket temperature (°C).",
    )
    heat_group.add_argument(
        "--heat-jacket-temp-measurement",
        default=os.getenv("HEAT_JACKET_TEMP_MEASUREMENT"),
        help="Measurement storing thermostat temperature.",
    )
    heat_group.add_argument(
        "--heat-jacket-temp-field",
        default=os.getenv("HEAT_JACKET_TEMP_FIELD", "value"),
        help="Field storing thermostat temperature.",
    )
    heat_group.add_argument(
        "--heat-jacket-temp-value",
        type=float,
        default=_env_float("HEAT_JACKET_TEMP_VALUE"),
        help="Fixed thermostat temperature (°C).",
    )
    heat_group.add_argument(
        "--heat-ambient-temp-bucket",
        default=os.getenv("HEAT_AMBIENT_TEMP_BUCKET"),
        help="Bucket storing ambient temperature (°C).",
    )
    heat_group.add_argument(
        "--heat-ambient-temp-measurement",
        default=os.getenv("HEAT_AMBIENT_TEMP_MEASUREMENT"),
        help="Measurement storing ambient temperature.",
    )
    heat_group.add_argument(
        "--heat-ambient-temp-field",
        default=os.getenv("HEAT_AMBIENT_TEMP_FIELD", "value"),
        help="Field storing ambient temperature.",
    )
    heat_group.add_argument(
        "--heat-ambient-temp-value",
        type=float,
        default=_env_float("HEAT_AMBIENT_TEMP_VALUE"),
        help="Fixed ambient temperature (°C).",
    )
    heat_group.add_argument(
        "--heat-ua-value",
        type=float,
        default=_env_float("HEAT_UA_VALUE"),
        help="Fixed UA coefficient (kJ/s/K).",
    )
    heat_group.add_argument(
        "--heat-ua-bucket",
        default=os.getenv("HEAT_UA_BUCKET"),
        help="Bucket storing UA coefficient readings.",
    )
    heat_group.add_argument(
        "--heat-ua-measurement",
        default=os.getenv("HEAT_UA_MEASUREMENT"),
        help="Measurement storing UA coefficient readings.",
    )
    heat_group.add_argument(
        "--heat-ua-field",
        default=os.getenv("HEAT_UA_FIELD", "value"),
        help="Field storing UA coefficient readings.",
    )
    heat_group.add_argument(
        "--heat-alpha-loss-value",
        type=float,
        default=_env_float("HEAT_ALPHA_LOSS_VALUE"),
        help="Fixed α_loss coefficient (kJ/s/K).",
    )
    heat_group.add_argument(
        "--heat-alpha-loss-bucket",
        default=os.getenv("HEAT_ALPHA_LOSS_BUCKET"),
        help="Bucket storing α_loss readings.",
    )
    heat_group.add_argument(
        "--heat-alpha-loss-measurement",
        default=os.getenv("HEAT_ALPHA_LOSS_MEASUREMENT"),
        help="Measurement storing α_loss readings.",
    )
    heat_group.add_argument(
        "--heat-alpha-loss-field",
        default=os.getenv("HEAT_ALPHA_LOSS_FIELD", "value"),
        help="Field storing α_loss readings.",
    )
    heat_group.add_argument(
        "--heat-agitator-power-value",
        type=float,
        default=_env_float("HEAT_AGITATOR_POWER_VALUE"),
        help="Fixed agitator power (kJ/s).",
    )
    heat_group.add_argument(
        "--heat-agitator-power-bucket",
        default=os.getenv("HEAT_AGITATOR_POWER_BUCKET"),
        help="Bucket storing agitator power readings.",
    )
    heat_group.add_argument(
        "--heat-agitator-power-measurement",
        default=os.getenv("HEAT_AGITATOR_POWER_MEASUREMENT"),
        help="Measurement storing agitator power readings.",
    )
    heat_group.add_argument(
        "--heat-agitator-power-field",
        default=os.getenv("HEAT_AGITATOR_POWER_FIELD", "value"),
        help="Field storing agitator power readings.",
    )
    heat_group.add_argument(
        "--heat-h2-mass-value",
        type=float,
        default=_env_float("HEAT_H2_MASS_VALUE"),
        help="Fixed hydrogen mass for dosing term (kg).",
    )
    heat_group.add_argument(
        "--heat-h2-mass-bucket",
        default=os.getenv("HEAT_H2_MASS_BUCKET"),
        help="Bucket storing hydrogen mass readings.",
    )
    heat_group.add_argument(
        "--heat-h2-mass-measurement",
        default=os.getenv("HEAT_H2_MASS_MEASUREMENT"),
        help="Measurement storing hydrogen mass readings.",
    )
    heat_group.add_argument(
        "--heat-h2-mass-field",
        default=os.getenv("HEAT_H2_MASS_FIELD", "value"),
        help="Field storing hydrogen mass readings.",
    )
    heat_group.add_argument(
        "--heat-h2-cp-value",
        type=float,
        default=_env_float("HEAT_H2_CP_VALUE"),
        help="Fixed hydrogen heat capacity (kJ/kg*K).",
    )
    heat_group.add_argument(
        "--heat-h2-cp-bucket",
        default=os.getenv("HEAT_H2_CP_BUCKET"),
        help="Bucket storing hydrogen heat capacity readings.",
    )
    heat_group.add_argument(
        "--heat-h2-cp-measurement",
        default=os.getenv("HEAT_H2_CP_MEASUREMENT"),
        help="Measurement storing hydrogen heat capacity readings.",
    )
    heat_group.add_argument(
        "--heat-h2-cp-field",
        default=os.getenv("HEAT_H2_CP_FIELD", "value"),
        help="Field storing hydrogen heat capacity readings.",
    )
    heat_group.add_argument(
        "--heat-thermostat-power-limit",
        type=float,
        default=_env_float("HEAT_THERMOSTAT_POWER_LIMIT"),
        help="Thermostat maximum cooling/heating power (kJ/s).",
    )
    heat_group.add_argument(
        "--heat-lhv",
        type=float,
        default=float(os.getenv("HEAT_LHV", "241.8")),
        help="Lower heating value of hydrogen (kJ/mol).",
    )
    heat_group.add_argument(
        "--heat-molar-mass-h2",
        type=float,
        default=float(os.getenv("HEAT_MOLAR_MASS_H2", "0.002016")),
        help="Hydrogen molar mass (kg/mol).",
    )
    heat_group.add_argument(
        "--heat-reactor-volume",
        type=float,
        default=float(os.getenv("HEAT_REACTOR_VOLUME", "0.01")),
        help="Reactor volume (m^3).",
    )
    heat_group.add_argument(
        "--heat-rate-measurement",
        default=os.getenv("HEAT_RATE_MEASUREMENT", "heat_balance"),
        help="Measurement name for heat balance outputs.",
    )
    heat_group.add_argument(
        "--heat-energy-measurement",
        default=os.getenv("HEAT_ENERGY_MEASUREMENT", "energy_efficiency"),
        help="Measurement name for efficiency outputs.",
    )
    heat_group.add_argument(
        "--heat-sty-measurement",
        default=os.getenv("HEAT_STY_MEASUREMENT", "space_time_yield"),
        help="Measurement name for space time yield outputs.",
    )

    return parser.parse_args()


def configure_logging(level: str) -> None:
    """Initialize a simple logger for CLI runs."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )


def build_faraday_config(args: argparse.Namespace) -> FaradayConfig:
    """Translate CLI args into the calculator config."""
    if not args.token:
        raise SystemExit("Provide an InfluxDB API token via --token, INFLUX_TOKEN, or NAS defaults.")
    if not args.org:
        raise SystemExit("Provide an InfluxDB organization via --org, INFLUX_ORG, or NAS defaults.")

    current_signal = _build_signal(
        "current",
        args.current_bucket,
        args.current_measurement,
        args.current_field,
    )
    efficiency_signal: Optional[SignalSelection] = None
    efficiency_fixed_ratio: Optional[float] = None

    if args.efficiency_use_signal:
        efficiency_signal = _build_signal(
            "efficiency",
            args.efficiency_bucket,
            args.efficiency_measurement,
            args.efficiency_field,
        )
    else:
        if args.efficiency_fixed_percent is None:
            raise SystemExit(
                "Provide --efficiency-fixed-percent or enable --efficiency-use-signal to fetch efficiency."
            )
        efficiency_fixed_ratio = args.efficiency_fixed_percent / 100.0

    return FaradayConfig(
        url=args.url,
        token=args.token,
        org=args.org,
        tag_key=args.tag_key,
        experience=args.experience,
        range_window=args.range_window,
        current_signal=current_signal,
        efficiency_signal=efficiency_signal,
        efficiency_is_percent=args.efficiency_percent,
        efficiency_fixed_ratio=efficiency_fixed_ratio,
        faraday_constant=args.faraday_constant,
        electrons_per_molecule=args.electrons_per_molecule,
        molar_volume=args.molar_volume,
    )


def build_doh_config(args: argparse.Namespace) -> DoHConfig:
    """Translate CLI args into the DoH calculator config."""
    if not args.token:
        raise SystemExit("Provide an InfluxDB API token via --token, INFLUX_TOKEN, or NAS defaults.")
    if not args.org:
        raise SystemExit("Provide an InfluxDB organization via --org, INFLUX_ORG, or NAS defaults.")

    hydrogen_volume = _build_scalar_source(
        "doh hydrogen volume",
        args.doh_h2_volume_bucket,
        args.doh_h2_volume_measurement,
        args.doh_h2_volume_field,
        args.doh_h2_volume_value,
        required_message=(
            "Provide --doh-h2-volume-bucket/measurement/field or --doh-h2-volume-value for DoH calculations."
        ),
    )
    pressure = _build_scalar_source(
        "doh pressure",
        args.doh_pressure_bucket,
        args.doh_pressure_measurement,
        args.doh_pressure_field,
        args.doh_pressure_value,
        required_message=(
            "Provide --doh-pressure-bucket/measurement/field or --doh-pressure-value for DoH calculations."
        ),
    )
    temperature = _build_scalar_source(
        "doh temperature",
        args.doh_temperature_bucket,
        args.doh_temperature_measurement,
        args.doh_temperature_field,
        args.doh_temperature_value,
        required_message=(
            "Provide --doh-temperature-bucket/measurement/field or --doh-temperature-value for DoH calculations."
        ),
    )
    lohc_mass = _build_scalar_source(
        "doh lohc mass",
        args.doh_lohc_mass_bucket,
        args.doh_lohc_mass_measurement,
        args.doh_lohc_mass_field,
        args.doh_lohc_mass_value,
        required_message=(
            "Provide --doh-lohc-mass-bucket/measurement/field or --doh-lohc-mass-value for DoH calculations."
        ),
    )
    lohc_volume = _build_optional_scalar_source(
        "doh lohc volume",
        args.doh_lohc_volume_bucket,
        args.doh_lohc_volume_measurement,
        args.doh_lohc_volume_field,
        args.doh_lohc_volume_value,
    )
    lohc_density = _build_optional_scalar_source(
        "doh lohc density",
        args.doh_lohc_density_bucket,
        args.doh_lohc_density_measurement,
        args.doh_lohc_density_field,
        args.doh_lohc_density_value,
    )
    if lohc_volume is None and lohc_density is None:
        raise SystemExit("Provide either LOHC volume inputs or LOHC density inputs for DoH calculations.")

    molar_mass = args.doh_lohc_molar_mass
    if molar_mass is None:
        raise SystemExit("Provide --doh-lohc-molar-mass (or DOH_LOHC_MOLAR_MASS env) for DoH calculations.")

    return DoHConfig(
        url=args.url,
        token=args.token,
        org=args.org,
        tag_key=args.tag_key,
        experience=args.experience,
        range_window=args.range_window,
        hydrogen_volume=hydrogen_volume,
        pressure=pressure,
        temperature=temperature,
        lohc_mass=lohc_mass,
        lohc_molar_mass=molar_mass,
        lohc_volume=lohc_volume,
        lohc_density=lohc_density,
        gas_constant=args.doh_gas_constant,
        reactor_volume_liters=args.doh_reactor_volume_liters,
        volume_to_m3_factor=args.doh_volume_to_m3_factor,
    )


def build_lohc_rate_config(args: argparse.Namespace) -> LohcRateConfig:
    """Translate CLI args into the LOHC rate calculator config."""
    if not args.token:
        raise SystemExit("Provide an InfluxDB API token via --token, INFLUX_TOKEN, or NAS defaults.")
    if not args.org:
        raise SystemExit("Provide an InfluxDB organization via --org, INFLUX_ORG, or NAS defaults.")

    doh_ratio = _build_scalar_source(
        "lohc doh ratio",
        args.lohc_doh_bucket,
        args.lohc_doh_measurement,
        args.lohc_doh_field,
        args.lohc_doh_value,
        required_message=(
            "Provide --lohc-doh-bucket/measurement/field or --lohc-doh-value for LOHC rate calculations."
        ),
    )
    density = _build_scalar_source(
        "lohc density",
        args.lohc_density_bucket,
        args.lohc_density_measurement,
        args.lohc_density_field,
        args.lohc_density_value,
        required_message=(
            "Provide --lohc-density-bucket/measurement/field or --lohc-density-value for LOHC rate calculations."
        ),
    )
    molar_mass = args.lohc_molar_mass
    if molar_mass is None:
        raise SystemExit("Provide --lohc-molar-mass (or LOHC_MOLAR_MASS env) for LOHC rate calculations.")

    partial_pressure = _build_scalar_source(
        "lohc partial pressure",
        args.lohc_ph2_bucket,
        args.lohc_ph2_measurement,
        args.lohc_ph2_field,
        args.lohc_ph2_value,
        required_message=(
            "Provide --lohc-ph2-bucket/measurement/field or --lohc-ph2-value for LOHC rate calculations."
        ),
    )
    henry_constant = _build_scalar_source(
        "lohc Henry constant",
        args.lohc_henry_bucket,
        args.lohc_henry_measurement,
        args.lohc_henry_field,
        args.lohc_henry_value,
        required_message=(
            "Provide --lohc-henry-bucket/measurement/field or --lohc-henry-value for LOHC rate calculations."
        ),
    )
    volumetric_fraction = _build_scalar_source(
        "lohc volumetric fraction",
        args.lohc_vol_frac_bucket,
        args.lohc_vol_frac_measurement,
        args.lohc_vol_frac_field,
        args.lohc_vol_frac_value,
        required_message=(
            "Provide --lohc-vol-frac-bucket/measurement/field or --lohc-vol-frac-value for LOHC rate calculations."
        ),
    )
    temperature = _build_scalar_source(
        "lohc temperature",
        args.lohc_temperature_bucket,
        args.lohc_temperature_measurement,
        args.lohc_temperature_field,
        args.lohc_temperature_value,
        required_message=(
            "Provide --lohc-temperature-bucket/measurement/field or --lohc-temperature-value for LOHC rate calculations."
        ),
    )

    kinetic_constant = _build_optional_scalar_source(
        "lohc kinetic constant",
        args.lohc_k_bucket,
        args.lohc_k_measurement,
        args.lohc_k_field,
        args.lohc_k_value,
    )

    if kinetic_constant is None and (args.lohc_k0 is None or args.lohc_activation_energy is None):
        raise SystemExit(
            "Provide --lohc-k inputs or both --lohc-k0 and --lohc-activation-energy for LOHC rate calculations."
        )

    return LohcRateConfig(
        url=args.url,
        token=args.token,
        org=args.org,
        tag_key=args.tag_key,
        experience=args.experience,
        range_window=args.range_window,
        doh_ratio=doh_ratio,
        lohc_density=density,
        lohc_molar_mass=molar_mass,
        partial_pressure=partial_pressure,
        henry_constant=henry_constant,
        volumetric_fraction=volumetric_fraction,
        temperature=temperature,
        gas_constant=args.lohc_gas_constant,
        kinetic_constant=kinetic_constant,
        pre_exponential=args.lohc_k0,
        activation_energy=args.lohc_activation_energy,
    )


def build_heat_balance_config(args: argparse.Namespace) -> HeatBalanceConfig:
    """Translate CLI args into the heat balance calculator config."""
    if not args.token:
        raise SystemExit("Provide an InfluxDB API token via --token, INFLUX_TOKEN, or NAS defaults.")
    if not args.org:
        raise SystemExit("Provide an InfluxDB organization via --org, INFLUX_ORG, or NAS defaults.")

    hydrogenation_rate = _build_scalar_source(
        "heat hydrogenation rate",
        args.heat_hydrogenation_rate_bucket,
        args.heat_hydrogenation_rate_measurement,
        args.heat_hydrogenation_rate_field,
        args.heat_hydrogenation_rate_value,
        required_message=(
            "Provide --heat-hydrogenation-rate-bucket/measurement/field or --heat-hydrogenation-rate-value."
        ),
    )
    mixture_mass = _build_scalar_source(
        "heat mixture mass",
        args.heat_mixture_mass_bucket,
        args.heat_mixture_mass_measurement,
        args.heat_mixture_mass_field,
        args.heat_mixture_mass_value,
        required_message="Provide mixture mass inputs for heat balance calculations.",
    )
    mixture_cp = _build_scalar_source(
        "heat mixture cp",
        args.heat_mixture_cp_bucket,
        args.heat_mixture_cp_measurement,
        args.heat_mixture_cp_field,
        args.heat_mixture_cp_value,
        required_message="Provide mixture heat capacity inputs for heat balance calculations.",
    )
    reactor_temp = _build_scalar_source(
        "heat reactor temp",
        args.heat_reactor_temp_bucket,
        args.heat_reactor_temp_measurement,
        args.heat_reactor_temp_field,
        args.heat_reactor_temp_value,
        required_message="Provide current reactor temperature inputs for heat balance calculations.",
    )
    reactor_temp_prev = _build_scalar_source(
        "heat reactor previous temp",
        args.heat_reactor_prev_temp_bucket,
        args.heat_reactor_prev_temp_measurement,
        args.heat_reactor_prev_temp_field,
        args.heat_reactor_prev_temp_value,
        required_message="Provide previous reactor temperature inputs for heat balance calculations.",
    )
    jacket_temp = _build_scalar_source(
        "heat jacket temp",
        args.heat_jacket_temp_bucket,
        args.heat_jacket_temp_measurement,
        args.heat_jacket_temp_field,
        args.heat_jacket_temp_value,
        required_message="Provide thermostat temperature inputs for heat balance calculations.",
    )
    ambient_temp = _build_scalar_source(
        "heat ambient temp",
        args.heat_ambient_temp_bucket,
        args.heat_ambient_temp_measurement,
        args.heat_ambient_temp_field,
        args.heat_ambient_temp_value,
        required_message="Provide ambient temperature inputs for heat balance calculations.",
    )
    ua_coefficient = _build_scalar_source(
        "heat UA coefficient",
        args.heat_ua_bucket,
        args.heat_ua_measurement,
        args.heat_ua_field,
        args.heat_ua_value,
        required_message="Provide UA coefficient inputs for heat balance calculations.",
    )
    alpha_loss = _build_scalar_source(
        "heat alpha loss",
        args.heat_alpha_loss_bucket,
        args.heat_alpha_loss_measurement,
        args.heat_alpha_loss_field,
        args.heat_alpha_loss_value,
        required_message="Provide alpha loss inputs for heat balance calculations.",
    )
    agitator_power = _build_scalar_source(
        "heat agitator power",
        args.heat_agitator_power_bucket,
        args.heat_agitator_power_measurement,
        args.heat_agitator_power_field,
        args.heat_agitator_power_value,
        required_message="Provide agitator power inputs for heat balance calculations.",
    )

    hydrogen_mass = _build_optional_scalar_source(
        "heat hydrogen mass",
        args.heat_h2_mass_bucket,
        args.heat_h2_mass_measurement,
        args.heat_h2_mass_field,
        args.heat_h2_mass_value,
    )
    hydrogen_cp = _build_optional_scalar_source(
        "heat hydrogen cp",
        args.heat_h2_cp_bucket,
        args.heat_h2_cp_measurement,
        args.heat_h2_cp_field,
        args.heat_h2_cp_value,
    )

    return HeatBalanceConfig(
        url=args.url,
        token=args.token,
        org=args.org,
        tag_key=args.tag_key,
        experience=args.experience,
        range_window=args.range_window,
        hydrogenation_rate=hydrogenation_rate,
        storage_rate_multiplier=args.heat_storage_multiplier,
        reaction_enthalpy_kj_per_mol=args.heat_reaction_enthalpy,
        mixture_mass=mixture_mass,
        mixture_heat_capacity=mixture_cp,
        reactor_temp=reactor_temp,
        reactor_temp_prev=reactor_temp_prev,
        accumulation_interval_seconds=args.heat_accu_interval_seconds,
        jacket_temp=jacket_temp,
        ambient_temp=ambient_temp,
        ua_coefficient=ua_coefficient,
        alpha_loss=alpha_loss,
        agitator_power=agitator_power,
        hydrogen_heat_capacity=hydrogen_cp,
        hydrogen_mass_dosed=hydrogen_mass,
        thermostat_power_limit=args.heat_thermostat_power_limit,
        lower_heating_value=args.heat_lhv,
        molar_mass_h2_kg=args.heat_molar_mass_h2,
        reactor_volume_m3=args.heat_reactor_volume,
    )
def build_faraday_result_target(args: argparse.Namespace) -> Optional[FaradayWriteTarget]:
    """Create the optional Influx destination for computed values."""
    if not args.write_results:
        return None

    bucket = args.result_bucket or args.current_bucket
    if not bucket:
        raise SystemExit(
            "Result bucket is not set. Provide --result-bucket or set --current-bucket so it can be reused."
        )

    return FaradayWriteTarget(
        bucket=bucket,
        molar_measurement=args.result_molar_measurement,
        volumetric_measurement=args.result_volumetric_measurement,
        field=args.result_field,
    )


def build_doh_write_target(args: argparse.Namespace) -> Optional[DoHWriteTarget]:
    """Create the optional Influx destination for DoH values."""
    if not args.write_results:
        return None

    bucket = args.result_bucket or args.doh_h2_volume_bucket or args.current_bucket
    if not bucket:
        raise SystemExit(
            "Result bucket is not set. Provide --result-bucket or set --doh-h2-volume-bucket/--current-bucket."
        )
    measurement = args.doh_result_measurement
    if not measurement:
        raise SystemExit("Provide --doh-result-measurement when --write-results is enabled for DoH.")

    return DoHWriteTarget(bucket=bucket, measurement=measurement, field=args.result_field)


def build_lohc_write_target(args: argparse.Namespace) -> Optional[LohcRateWriteTarget]:
    """Create the optional Influx destination for LOHC rate values."""
    if not args.write_results:
        return None

    bucket = args.result_bucket or args.lohc_doh_bucket or args.current_bucket
    if not bucket:
        raise SystemExit(
            "Result bucket is not set. Provide --result-bucket or configure --lohc-doh-bucket/--current-bucket."
        )

    return LohcRateWriteTarget(
        bucket=bucket,
        hydrogenation_measurement=args.lohc_rate_measurement,
        storage_measurement=args.lohc_h2_rate_measurement,
        field=args.result_field,
    )


def build_heat_write_target(args: argparse.Namespace) -> Optional[HeatBalanceWriteTarget]:
    """Create the optional Influx destination for heat balance values."""
    if not args.write_results:
        return None

    bucket = args.result_bucket or args.heat_hydrogenation_rate_bucket or args.current_bucket
    if not bucket:
        raise SystemExit(
            "Result bucket is not set. Provide --result-bucket or configure --heat-hydrogenation-rate-bucket/--current-bucket."
        )

    return HeatBalanceWriteTarget(
        bucket=bucket,
        heat_measurement=args.heat_rate_measurement,
        energy_measurement=args.heat_energy_measurement,
        sty_measurement=args.heat_sty_measurement,
        field=args.result_field,
    )


def _build_signal(name: str, bucket: Optional[str], measurement: Optional[str], field: Optional[str]) -> SignalSelection:
    """Validate bucket/measurement/field inputs and wrap them in a dataclass."""
    missing = [label for value, label in ((bucket, "bucket"), (measurement, "measurement"), (field, "field")) if not value]
    if missing:
        raise SystemExit(
            f"Missing {name} signal configuration ({', '.join(missing)}). "
            f"Provide --{name}-bucket/measurement/field or set the matching environment variables."
        )
    return SignalSelection(bucket=bucket, measurement=measurement, field=field)


def _build_scalar_source(
    name: str,
    bucket: Optional[str],
    measurement: Optional[str],
    field: Optional[str],
    value: Optional[float],
    required_message: Optional[str] = None,
) -> ScalarSource:
    """Create a scalar source from either a fixed value or an Influx signal."""
    if value is not None:
        return ScalarSource(name=name, fixed_value=value)

    signal = _maybe_build_signal(name, bucket, measurement, field)
    if signal is None:
        raise SystemExit(required_message or f"Provide inputs for {name} (value or bucket/measurement/field).")
    return ScalarSource(name=name, signal=signal)


def _build_optional_scalar_source(
    name: str,
    bucket: Optional[str],
    measurement: Optional[str],
    field: Optional[str],
    value: Optional[float],
) -> Optional[ScalarSource]:
    """Return a scalar source if any inputs exist; otherwise None."""
    if value is None and not any((bucket, measurement, field)):
        return None
    return _build_scalar_source(name, bucket, measurement, field, value)


def _maybe_build_signal(
    name: str,
    bucket: Optional[str],
    measurement: Optional[str],
    field: Optional[str],
) -> Optional[SignalSelection]:
    if not any((bucket, measurement, field)):
        return None
    return _build_signal(name, bucket, measurement, field)


def _stream_results(
    calculator,
    target,
    emit_fn,
    args: argparse.Namespace,
    formula_label: str,
) -> None:
    """Continuously compute and optionally write/emit results."""
    interval = args.sample_interval
    if args.single_shot:
        max_samples: Optional[int] = 1
    else:
        max_samples = args.max_samples if args.max_samples and args.max_samples > 0 else None
    streaming = not args.single_shot and (max_samples is None or max_samples > 1)

    if streaming and interval <= 0:
        raise SystemExit("--sample-interval must be positive when streaming continuously.")

    sample_count = 0
    try:
        while True:
            try:
                result = calculator.compute()
            except Exception as exc:  # pragma: no cover - CLI safeguard
                logging.exception("%s computation failed: %s", formula_label, exc)
                calculator.close()
                sys.exit(1)

            if streaming and hasattr(result, "timestamp"):
                try:
                    result.timestamp = datetime.now(timezone.utc)
                except Exception:
                    pass

            if target:
                try:
                    calculator.write_result(result, target)
                except Exception as exc:  # pragma: no cover - write safeguard
                    logging.exception("Writing %s result failed: %s", formula_label, exc)
                    calculator.close()
                    sys.exit(1)

            emit_fn(result, args.output, streaming=streaming)

            sample_count += 1
            if not streaming or (max_samples is not None and sample_count >= max_samples):
                break

            try:
                time.sleep(interval)
            except KeyboardInterrupt:
                raise
    except KeyboardInterrupt:
        logging.info("Stopping %s sampling loop due to user interrupt.", formula_label)
    finally:
        calculator.close()


def _replay_faraday_experience(
    calculator: FaradayCalculator,
    target: Optional[FaradayWriteTarget],
    emit_fn,
    args: argparse.Namespace,
) -> None:
    """Replay stored Faraday data across the experience window and exit."""

    start_override = _parse_timestamp_arg("--experience-start", args.experience_start)
    end_override = _parse_timestamp_arg("--experience-end", args.experience_end)
    if start_override and end_override and end_override < start_override:
        raise SystemExit("--experience-end must not be earlier than --experience-start.")
    if args.sample_interval <= 0:
        raise SystemExit("--sample-interval must be positive when replaying an experience.")

    window_delta = _parse_range_window("--range-window", args.range_window)
    now = datetime.now(timezone.utc)
    query_start = start_override or (now - window_delta)
    query_end = end_override or now

    try:
        generator, actual_start, actual_end = calculator.iter_experience(
            query_start,
            query_end,
            args.sample_interval,
        )
        sample_count = 0
        for result in generator:
            if target:
                calculator.write_result(result, target)
            emit_fn(result, args.output, streaming=True)
            sample_count += 1
        logging.info(
            "Replayed %s Faraday samples for experience '%s' between %s and %s.",
            sample_count,
            calculator.config.experience,
            actual_start.isoformat(),
            actual_end.isoformat(),
        )
    except Exception as exc:  # pragma: no cover - CLI safeguard
        logging.exception("Faraday replay failed: %s", exc)
        sys.exit(1)
    finally:
        calculator.close()


def _faraday_result_to_dict(result: FaradayResult) -> dict:
    """Convert the result object into a serializable dict."""
    return {
        "experience": result.experience,
        "timestamp": result.timestamp.isoformat(),
        "current_a": result.current,
        "efficiency_ratio": result.efficiency_ratio,
        "efficiency_percent": result.efficiency_ratio * 100.0,
        "molar_rate_mol_s": result.molar_rate,
        "volumetric_rate_nl_s": result.volumetric_rate,
    }


def _doh_result_to_dict(result: DoHResult) -> dict:
    """Convert the DoH result into a serializable dict."""
    return {
        "experience": result.experience,
        "timestamp": result.timestamp.isoformat(),
        "doh_ratio": result.doh_ratio,
        "doh_percent": result.doh_ratio * 100.0,
        "stored_h2_moles": result.stored_h2_moles,
        "max_h2_moles": result.max_h2_moles,
        "hydrogen_volume_liters": result.hydrogen_volume_liters,
        "net_hydrogen_volume_liters": result.net_hydrogen_volume_liters,
    }


def _emit_faraday_result(result: FaradayResult, output: str, streaming: bool = False) -> None:
    """Print CLI output in text or JSON form."""
    if output == "json":
        payload = (
            json.dumps(_faraday_result_to_dict(result), separators=(",", ":"))
            if streaming
            else json.dumps(_faraday_result_to_dict(result), indent=2)
        )
        print(payload, flush=True)
        return

    lines = [
        f"Experience: {result.experience}",
        f"Timestamp: {result.timestamp.isoformat()}",
        f"Current: {result.current:.3f} A",
        f"Efficiency: {result.efficiency_ratio * 100.0:.2f} %",
        f"Molar rate: {result.molar_rate:.6f} mol/s",
        f"Volumetric rate: {result.volumetric_rate:.6f} NL/s",
    ]
    print("\n".join(lines), flush=True)


def _emit_doh_result(result: DoHResult, output: str, streaming: bool = False) -> None:
    """Print DoH output in text or JSON form."""
    if output == "json":
        payload = (
            json.dumps(_doh_result_to_dict(result), separators=(",", ":"))
            if streaming
            else json.dumps(_doh_result_to_dict(result), indent=2)
        )
        print(payload, flush=True)
        return

    lines = [
        f"Experience: {result.experience}",
        f"Timestamp: {result.timestamp.isoformat()}",
        f"DoH: {result.doh_ratio * 100.0:.3f} %",
        f"Stored H2: {result.stored_h2_moles:.6f} mol",
        f"Max H2: {result.max_h2_moles:.6f} mol",
        f"Net H2 volume: {result.net_hydrogen_volume_liters:.3f} L",
    ]
    print("\n".join(lines), flush=True)


def _lohc_result_to_dict(result: LohcRateResult) -> dict:
    """Convert the LOHC rate result into a serializable dict."""
    return {
        "experience": result.experience,
        "timestamp": result.timestamp.isoformat(),
        "doh_ratio": result.doh_ratio,
        "nmih_concentration": result.nmih_concentration,
        "hydrogenated_concentration": result.hydrogenated_concentration,
        "kinetic_constant": result.kinetic_constant,
        "lohc_hydrogenation_rate": result.hydrogenation_rate,
        "h2_storage_rate": result.hydrogen_storage_rate,
    }


def _emit_lohc_result(result: LohcRateResult, output: str, streaming: bool = False) -> None:
    """Print LOHC hydrogenation output."""
    if output == "json":
        payload = (
            json.dumps(_lohc_result_to_dict(result), separators=(",", ":"))
            if streaming
            else json.dumps(_lohc_result_to_dict(result), indent=2)
        )
        print(payload, flush=True)
        return

    lines = [
        f"Experience: {result.experience}",
        f"Timestamp: {result.timestamp.isoformat()}",
        f"DoH: {result.doh_ratio * 100.0:.3f} %",
        f"[NMID]: {result.nmih_concentration:.6f} mol/L",
        f"[8HNMID]: {result.hydrogenated_concentration:.6f} mol/L",
        f"Kinetic constant: {result.kinetic_constant:.6e}",
        f"Hydrogenation rate: {result.hydrogenation_rate:.6f} mol/(L*s)",
        f"H2 storage rate: {result.hydrogen_storage_rate:.6f} mol/s",
    ]
    print("\n".join(lines), flush=True)


def _heat_result_to_dict(result: HeatBalanceResult) -> dict:
    """Convert the heat balance result into a serializable dict."""
    return {
        "experience": result.experience,
        "timestamp": result.timestamp.isoformat(),
        "hydrogenation_rate_mol_s": result.hydrogenation_rate,
        "hydrogen_storage_rate_mol_s": result.hydrogen_storage_rate,
        "q_flow_kj_s": result.q_flow,
        "q_accu_kj_s": result.q_accu,
        "q_loss_kj_s": result.q_loss,
        "q_dos_kj_s": result.q_dos,
        "q_net_measured_kj_s": result.q_net_measured,
        "q_net_theoretical_kj_s": result.q_net_theoretical,
        "thermostat_limit_kj_s": result.thermostat_limit,
        "q_net_minus_limit_kj_s": result.q_net_minus_limit,
        "efficiency_ratio": result.efficiency,
        "mass_rate_h2_kg_s": result.mass_rate_h2,
        "sty_kg_m3_h": result.space_time_yield,
    }


def _emit_heat_result(result: HeatBalanceResult, output: str, streaming: bool = False) -> None:
    """Print heat balance output."""
    if output == "json":
        payload = (
            json.dumps(_heat_result_to_dict(result), separators=(",", ":"))
            if streaming
            else json.dumps(_heat_result_to_dict(result), indent=2)
        )
        print(payload, flush=True)
        return

    lines = [
        f"Experience: {result.experience}",
        f"Timestamp: {result.timestamp.isoformat()}",
        f"Hydrogenation rate: {result.hydrogenation_rate:.6f} mol/s",
        f"Hydrogen storage rate: {result.hydrogen_storage_rate:.6f} mol/s",
        f"Q_flow: {result.q_flow:.3f} kJ/s",
        f"Q_accu: {result.q_accu:.3f} kJ/s",
        f"Q_loss: {result.q_loss:.3f} kJ/s",
        f"Q_dos: {result.q_dos:.3f} kJ/s",
        f"Q_net (measured): {result.q_net_measured:.3f} kJ/s",
        f"Q_net (theoretical): {result.q_net_theoretical:.3f} kJ/s",
    ]
    if result.thermostat_limit is not None:
        lines.append(f"Thermostat limit: {result.thermostat_limit:.3f} kJ/s")
    if result.q_net_minus_limit is not None:
        lines.append(f"Q_net - limit: {result.q_net_minus_limit:.3f} kJ/s")
    lines.extend([
        f"Efficiency: {result.efficiency * 100.0:.2f} %",
        f"Mass rate H2: {result.mass_rate_h2:.6f} kg/s",
        f"Space time yield: {result.space_time_yield:.3f} kg/m^3/h",
    ])
    print("\n".join(lines), flush=True)


def main() -> None:
    """Entrypoint that wires parsing, computation, optional writes, and output."""
    nas_defaults = resolve_defaults()
    args = parse_args(nas_defaults)
    configure_logging(args.log_level)
    if args.formula == "faraday":
        config = build_faraday_config(args)
        result_target = build_faraday_result_target(args)
        calculator = FaradayCalculator(config)
        if args.experience_finished:
            _replay_faraday_experience(calculator, result_target, _emit_faraday_result, args)
        else:
            _stream_results(calculator, result_target, _emit_faraday_result, args, "Faraday")
        return

    if args.formula == "doh":
        doh_config = build_doh_config(args)
        doh_target = build_doh_write_target(args)
        calculator = DoHCalculator(doh_config)
        _stream_results(calculator, doh_target, _emit_doh_result, args, "DoH")
        return

    if args.formula == "lohc_rate":
        lohc_config = build_lohc_rate_config(args)
        lohc_target = build_lohc_write_target(args)
        calculator = LohcRateCalculator(lohc_config)
        _stream_results(calculator, lohc_target, _emit_lohc_result, args, "LOHC rate")
        return

    heat_config = build_heat_balance_config(args)
    heat_target = build_heat_write_target(args)
    calculator = HeatBalanceCalculator(heat_config)
    _stream_results(calculator, heat_target, _emit_heat_result, args, "Heat balance")


if __name__ == "__main__":
    main()
