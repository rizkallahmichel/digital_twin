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

from experience_calculator import (
    ExperienceCalculator,
    ExperienceConfig,
    ExperienceResult,
    ResultWriteTarget,
)
from nas_config import NasInfluxDefaults, influx_cli_defaults, resolve_defaults


def parse_args(defaults: NasInfluxDefaults) -> argparse.Namespace:
    """Set up CLI flags and return parsed arguments."""
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
    return parser.parse_args()


def configure_logging(level: str) -> None:
    """Initialize a simple logger for CLI runs."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )


def build_config(args: argparse.Namespace) -> ExperienceConfig:
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
        efficiency_fixed_ratio=efficiency_fixed_ratio,
        faraday_constant=args.faraday_constant,
        electrons_per_molecule=args.electrons_per_molecule,
        molar_volume=args.molar_volume,
    )


def build_result_target(args: argparse.Namespace) -> Optional[ResultWriteTarget]:
    """Create the optional Influx destination for computed values."""
    if not args.write_results:
        return None

    bucket = args.result_bucket or args.current_bucket
    if not bucket:
        raise SystemExit(
            "Result bucket is not set. Provide --result-bucket or set --current-bucket so it can be reused."
        )

    return ResultWriteTarget(
        bucket=bucket,
        molar_measurement=args.result_molar_measurement,
        volumetric_measurement=args.result_volumetric_measurement,
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


def _result_to_dict(result: ExperienceResult) -> dict:
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


def _emit_result(result: ExperienceResult, output: str) -> None:
    """Print CLI output in text or JSON form."""
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
    """Entrypoint that wires parsing, computation, optional writes, and output."""
    nas_defaults = resolve_defaults()
    args = parse_args(nas_defaults)
    configure_logging(args.log_level)
    config = build_config(args)
    result_target = build_result_target(args)

    calculator = ExperienceCalculator(config)
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

    _emit_result(result, args.output)


if __name__ == "__main__":
    main()
