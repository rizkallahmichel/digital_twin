"""Centralized NAS and InfluxDB configuration defaults for the Faraday CLI."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class NasInfluxDefaults:
    """Captures the NAS networking layout plus InfluxDB credentials."""

    host: str
    physical_port: int
    ui_port: int
    influx_port: int
    username: str
    password: str
    token: str
    org: str

    @property
    def api_url(self) -> str:
        return f"http://{self.host}:{self.influx_port}"


def resolve_defaults() -> NasInfluxDefaults:
    """Return defaults overridden by environment variables when provided."""

    base = NasInfluxDefaults(
        host="192.168.5.102",
        physical_port=2,
        ui_port=8080,
        influx_port=8086,
        username="upscalehub",
        password="upscalehub",
        token="P3Z92mD4U0oduLSpzf_HKoWYqVsck4-e_rKJqWHqjKRbhREOtOKQ-CS7x4TAw-vrXFQJNlJ27ylAWPtVZdv3gg==",
        org="upscalehub",
    )

    return NasInfluxDefaults(
        host=os.getenv("H2_NAS_HOST", base.host),
        physical_port=int(os.getenv("H2_NAS_PHYSICAL_PORT", base.physical_port)),
        ui_port=int(os.getenv("H2_NAS_UI_PORT", base.ui_port)),
        influx_port=int(os.getenv("H2_INFLUX_PORT", base.influx_port)),
        username=os.getenv("H2_INFLUX_USERNAME", base.username),
        password=os.getenv("H2_INFLUX_PASSWORD", base.password),
        token=os.getenv("H2_INFLUX_TOKEN_FALLBACK", base.token),
        org=os.getenv("H2_INFLUX_ORG_FALLBACK", base.org),
    )


def influx_cli_defaults(defaults: NasInfluxDefaults | None = None) -> dict[str, str]:
    """Expose URL/token/org defaults for argument parsing and help text."""

    defaults = defaults or resolve_defaults()
    return {
        "url": os.getenv("INFLUX_URL", defaults.api_url),
        "token": os.getenv("INFLUX_TOKEN", defaults.token),
        "org": os.getenv("INFLUX_ORG", defaults.org),
    }
