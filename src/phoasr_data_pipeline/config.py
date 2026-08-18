# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Configuration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class PipelineConfig:
    """Provide dotted-key access to the loaded pipeline configuration."""

    raw: dict[str, Any]
    path: Path

    @classmethod
    def load(cls, config_path: str | Path) -> "PipelineConfig":
        """Load a YAML configuration file from disk."""
        path = Path(config_path).resolve()
        with path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        return cls(raw=raw, path=path)

    def get(self, dotted_key: str, default: Any = None) -> Any:
        """Return a nested config value addressed by a dotted key."""
        current: Any = self.raw
        for key in dotted_key.split("."):
            if not isinstance(current, dict) or key not in current:
                return default
            current = current[key]
        return current

    def is_stage_enabled(self, stage: str) -> bool:
        """Return whether a stage is enabled, defaulting to True when unspecified."""
        enabled_map = self.get("stages.enabled", {}) or {}
        return bool(enabled_map.get(stage, True))

    @property
    def repo_root(self) -> Path:
        """Return the repository root inferred from the config file location."""
        return self.path.parent.parent.resolve()
