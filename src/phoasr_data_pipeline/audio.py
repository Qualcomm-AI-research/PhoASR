# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Audio preprocessing helpers."""

from __future__ import annotations

import subprocess
from pathlib import Path

from .config import PipelineConfig
from .io import copy_audio


def prepare_audio_copy(
    audio_path: str | Path,
    destination: str | Path,
    config: PipelineConfig,
) -> Path:
    """Copy audio into the run tree, normalizing it with ffmpeg when enabled."""
    destination = Path(destination)
    normalize_audio = bool(config.get("audio.normalize.enabled", False))
    sample_rate = int(config.get("audio.sample_rate", 16000))
    if not normalize_audio:
        return copy_audio(audio_path, destination)

    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(audio_path),
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        str(destination),
    ]
    subprocess.run(command, check=True, capture_output=True)
    return destination


def get_audio_duration_seconds(audio_path: str | Path) -> float:
    """Return audio duration in seconds using ffprobe."""
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(audio_path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return float(result.stdout.strip())
