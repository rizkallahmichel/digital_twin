#!/usr/bin/env python3
"""CLI entry point for hydrogen formula computations filtered by an experience tag."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
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
from nas_config import NasInfluxDefaults, influx_cli_defaults, resolve_defaults


def _env_float(name: str, default: Optional[float] = None) -> Optional[float]:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise SystemExit(f"Environment variable {name} must be a float; received {value!r}.") from exc


def parse_args(defaults: NasInfluxDefaults) -> argparse.Namespace:
    """Set up CLI flags and return parsed arguments."""
    cli_defaults = influx_cli_defaults(defaults)
    parser = argparse.ArgumentParser(
        description="Fetch the latest readings for an experience tag and compute hydrogen formulas.",
    )
    parser.add_argument("experience", help="Experience/sensor tag value to filter on (e.g. test2).")
    parser.add_argument(
        "--formula",
        choices=("faraday", "doh", "lohc_rate"),
        default=os.getenv("H2_FORMULA", "faraday"),
        help="Select which formula to run (faraday, doh, lohc_rate). Default: faraday.",
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


def _emit_faraday_result(result: FaradayResult, output: str) -> None:
    """Print CLI output in text or JSON form."""
    if output == "json":
        print(json.dumps(_faraday_result_to_dict(result), indent=2))
        return

    print(f"Experience: {result.experience}")
    print(f"Timestamp: {result.timestamp.isoformat()}")
    print(f"Current: {result.current:.3f} A")
    print(f"Efficiency: {result.efficiency_ratio * 100.0:.2f} %")
    print(f"Molar rate: {result.molar_rate:.6f} mol/s")
    print(f"Volumetric rate: {result.volumetric_rate:.6f} NL/s")


def _emit_doh_result(result: DoHResult, output: str) -> None:
    """Print DoH output in text or JSON form."""
    if output == "json":
        print(json.dumps(_doh_result_to_dict(result), indent=2))
        return

    print(f"Experience: {result.experience}")
    print(f"Timestamp: {result.timestamp.isoformat()}")
    print(f"DoH: {result.doh_ratio * 100.0:.3f} %")
    print(f"Stored H2: {result.stored_h2_moles:.6f} mol")
    print(f"Max H2: {result.max_h2_moles:.6f} mol")
    print(f"Net H2 volume: {result.net_hydrogen_volume_liters:.3f} L")


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


def _emit_lohc_result(result: LohcRateResult, output: str) -> None:
    """Print LOHC hydrogenation output."""
    if output == "json":
        print(json.dumps(_lohc_result_to_dict(result), indent=2))
        return

    print(f"Experience: {result.experience}")
    print(f"Timestamp: {result.timestamp.isoformat()}")
    print(f"DoH: {result.doh_ratio * 100.0:.3f} %")
    print(f"[NMID]: {result.nmih_concentration:.6f} mol/L")
    print(f"[8HNMID]: {result.hydrogenated_concentration:.6f} mol/L")
    print(f"Kinetic constant: {result.kinetic_constant:.6e}")
    print(f"Hydrogenation rate: {result.hydrogenation_rate:.6f} mol/(L·s)")
    print(f"H2 storage rate: {result.hydrogen_storage_rate:.6f} mol/s")


def main() -> None:
    """Entrypoint that wires parsing, computation, optional writes, and output."""
    nas_defaults = resolve_defaults()
    args = parse_args(nas_defaults)
    configure_logging(args.log_level)
    if args.formula == "faraday":
        config = build_faraday_config(args)
        result_target = build_faraday_result_target(args)
        calculator = FaradayCalculator(config)
        try:
            result = calculator.compute()
        except Exception as exc:  # pragma: no cover - CLI safeguard
            logging.exception("Experience computation failed: %s", exc)
            calculator.close()
            sys.exit(1)

        if result_target:
            try:
                calculator.write_result(result, result_target)
            except Exception as exc:  # pragma: no cover - write safeguard
                logging.exception("Writing computed result failed: %s", exc)
                calculator.close()
                sys.exit(1)
        calculator.close()
        _emit_faraday_result(result, args.output)
        return

    if args.formula == "doh":
        doh_config = build_doh_config(args)
        doh_target = build_doh_write_target(args)
        calculator = DoHCalculator(doh_config)
        try:
            doh_result = calculator.compute()
        except Exception as exc:  # pragma: no cover - CLI safeguard
            logging.exception("DoH computation failed: %s", exc)
            calculator.close()
            sys.exit(1)

        if doh_target:
            try:
                calculator.write_result(doh_result, doh_target)
            except Exception as exc:  # pragma: no cover - write safeguard
                logging.exception("Writing DoH result failed: %s", exc)
                calculator.close()
                sys.exit(1)
        calculator.close()
        _emit_doh_result(doh_result, args.output)
        return

    lohc_config = build_lohc_rate_config(args)
    lohc_target = build_lohc_write_target(args)
    calculator = LohcRateCalculator(lohc_config)
    try:
        lohc_result = calculator.compute()
    except Exception as exc:  # pragma: no cover - CLI safeguard
        logging.exception("LOHC rate computation failed: %s", exc)
        calculator.close()
        sys.exit(1)

    if lohc_target:
        try:
            calculator.write_result(lohc_result, lohc_target)
        except Exception as exc:  # pragma: no cover - write safeguard
            logging.exception("Writing LOHC rate result failed: %s", exc)
            calculator.close()
            sys.exit(1)
    calculator.close()
    _emit_lohc_result(lohc_result, args.output)


if __name__ == "__main__":
    main()
