from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Settings:
    root: Path
    values: dict[str, Any]

    def section(self, name: str) -> dict[str, Any]:
        value = self.values.get(name, {})
        if not isinstance(value, dict):
            raise TypeError(f"Configuration section '{name}' must be a mapping.")
        return value

    def path(self, name: str) -> Path:
        raw = self.section("paths").get(name)
        if not raw:
            raise ValueError(f"Missing paths.{name} in configuration.")
        return (self.root / str(raw)).resolve()

    def ensure_directories(self) -> None:
        for name in ("raw", "processed", "features", "outputs", "models", "logs"):
            self.path(name).mkdir(parents=True, exist_ok=True)


def default_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_settings(config_path: str | Path | None = None) -> Settings:
    root = default_project_root()
    path = Path(config_path).resolve() if config_path else root / "config" / "settings.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Configuration not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        values = yaml.safe_load(handle) or {}
    settings = Settings(root=root, values=values)
    settings.ensure_directories()
    return settings
