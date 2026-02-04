#!/usr/bin/env python3
"""CLI entry point for Faraday computations filtered by an experience tag."""

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

from experience_calculator import ExperienceCalculator, ExperienceConfig, ExperienceResult
from nas_config import NasInfluxDefaults, influx_cli_defaults, resolve_defaults


def parse_args(defaults: NasInfluxDefaults) -> argparse.Namespace:
    cli_defaults = influx_cli_defaults(defaults)
    parser = argparse.ArgumentParser(
        description="Fetch the latest readings for an experience tag and compute Faraday-based hydrogen output.",
    )
    parser.add_argument("experience", help="Experience/sensor tag value to filter on (e.g. test2).")
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
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )


def build_config(args: argparse.Namespace) -> ExperienceConfig:
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
    efficiency_signal = _build_signal(
        "efficiency",
        args.efficiency_bucket,
        args.efficiency_measurement,
        args.efficiency_field,
    )

    return ExperienceConfig(
        url=args.url,
        token=args.token,
        org=args.org,
        tag_key=args.tag_key,
        experience=args.experience,
        range_window=args.range_window,
        current_signal=current_signal,
        efficiency_signal=efficiency_signal,
        efficiency_is_percent=args.efficiency_percent,
        faraday_constant=args.faraday_constant,
        electrons_per_molecule=args.electrons_per_molecule,
        molar_volume=args.molar_volume,
    )


def _build_signal(name: str, bucket: Optional[str], measurement: Optional[str], field: Optional[str]) -> SignalSelection:
    missing = [label for value, label in ((bucket, "bucket"), (measurement, "measurement"), (field, "field")) if not value]
    if missing:
        raise SystemExit(
            f"Missing {name} signal configuration ({', '.join(missing)}). "
            f"Provide --{name}-bucket/measurement/field or set the matching environment variables."
        )
    return SignalSelection(bucket=bucket, measurement=measurement, field=field)


def _result_to_dict(result: ExperienceResult) -> dict:
    return {
        "experience": result.experience,
        "timestamp": result.timestamp.isoformat(),
        "current_a": result.current,
        "efficiency_ratio": result.efficiency_ratio,
        "efficiency_percent": result.efficiency_ratio * 100.0,
        "molar_rate_mol_s": result.molar_rate,
        "volumetric_rate_nl_s": result.volumetric_rate,
    }


def _emit_result(result: ExperienceResult, output: str) -> None:
    if output == "json":
        print(json.dumps(_result_to_dict(result), indent=2))
        return

    print(f"Experience: {result.experience}")
    print(f"Timestamp: {result.timestamp.isoformat()}")
    print(f"Current: {result.current:.3f} A")
    print(f"Efficiency: {result.efficiency_ratio * 100.0:.2f} %")
    print(f"Molar rate: {result.molar_rate:.6f} mol/s")
    print(f"Volumetric rate: {result.volumetric_rate:.6f} NL/s")


def main() -> None:
    nas_defaults = resolve_defaults()
    args = parse_args(nas_defaults)
    configure_logging(args.log_level)
    config = build_config(args)

    calculator = ExperienceCalculator(config)
    try:
        result = calculator.compute()
    except Exception as exc:  # pragma: no cover - CLI safeguard
        logging.exception("Experience computation failed: %s", exc)
        calculator.close()
        sys.exit(1)
    calculator.close()

    _emit_result(result, args.output)


if __name__ == "__main__":
    main()
