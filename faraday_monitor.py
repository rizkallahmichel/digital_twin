"""Shared utilities/constants for Faraday-based hydrogen monitoring."""

from __future__ import annotations

from dataclasses import dataclass

# Default Flux lookback window for "latest" queries, override via env if needed.
DEFAULT_RANGE_WINDOW = "5m"

# Fundamental constants pulled out for clarity and reuse.
FARADAY_CONSTANT = 96485.33212  # Coulombs per mole
ELECTRONS_PER_H2 = 2.0  # Two electrons to produce one H2 molecule
MOLAR_VOLUME_NL = 24.465  # Normal liters per mole at STP


@dataclass(frozen=True)
class SignalSelection:
    """Describes the bucket/measurement/field triple for a Flux query."""

    bucket: str
    measurement: str
    field: str
