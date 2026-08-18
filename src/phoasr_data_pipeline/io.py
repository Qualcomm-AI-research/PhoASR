# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Manifest IO and validation."""

from __future__ import annotations

import shutil
import string
from pathlib import Path

import pandas as pd

from .constants import REQUIRED_INPUT_COLUMNS


def load_manifest(path: str | Path) -> pd.DataFrame:
    """Load a CSV manifest."""
    return pd.read_csv(path)


def ensure_object_columns(df: pd.DataFrame, columns: list[str] | tuple[str, ...]) -> pd.DataFrame:
    """Ensure manifest text columns exist and are object-compatible."""
    for column in columns:
        if column not in df.columns:
            df[column] = ""
        df[column] = df[column].astype("object")
    return df


def resolve_repo_relative_path(path: str | Path, repo_root: str | Path) -> Path:
    """Resolve a repo-relative or absolute artifact path against the package root."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate.resolve()
    return (Path(repo_root).resolve() / candidate).resolve()


def relativize_to_repo(path: str | Path, repo_root: str | Path) -> str:
    """Store an artifact path relative to the package root."""
    return str(Path(path).resolve().relative_to(Path(repo_root).resolve()))


def detect_transcript_availability(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize transcript cells and label them as provided or missing."""
    out = ensure_object_columns(df.copy(), ["status", "drop_reason"])
    transcript_series = out["transcript"] if "transcript" in out.columns else ""
    if isinstance(transcript_series, str):
        cleaned = pd.Series([""] * len(out), index=out.index, dtype="object")
    else:
        cleaned = transcript_series.fillna("").astype(str).map(str.strip)
    out["transcript"] = cleaned
    out["transcript_source"] = cleaned.map(lambda text: "provided" if text else "missing")
    return out


def validate_input_manifest(df: pd.DataFrame, base_dir: str | Path | None = None) -> pd.DataFrame:
    """Validate required manifest columns and resolve audio paths."""
    missing = [column for column in REQUIRED_INPUT_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    validated = detect_transcript_availability(df)
    validated["sample_id"] = validated["sample_id"].astype(str)
    validated["audio_path"] = validated["audio_path"].astype(str)
    resolved_paths = []
    base_dir_path = Path(base_dir).resolve() if base_dir else None
    for audio_path in validated["audio_path"]:
        candidate = Path(audio_path)
        if not candidate.is_absolute() and base_dir_path is not None:
            candidate = base_dir_path / candidate
        candidate = candidate.resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        resolved_paths.append(str(candidate))
    validated["audio_path"] = resolved_paths
    return validated


def prepare_run_manifest(df: pd.DataFrame) -> pd.DataFrame:
    """Add the bookkeeping columns expected by downstream stages."""
    out = df.copy()
    out["status"] = "active"
    out["drop_reason"] = ""
    out["wer"] = pd.NA
    out = ensure_object_columns(
        out,
        [
            "final_transcript",
            "text_with_punc",
            "text_normalized_for_alignment",
            "timestamp_json_path",
            "verification_transcript",
            "alignment_input_text",
            "mapping_json_path",
        ],
    )
    return out


def filter_invalid_transcript_rows(
    df: pd.DataFrame,
    allowed_punctuations: list[str] | tuple[str, ...] | set[str],
) -> pd.DataFrame:
    """Drop provided transcripts containing characters outside the allowed text set."""
    out = df.copy()
    allowed_punct = set(allowed_punctuations)
    all_punct = set(string.punctuation)
    disallowed_punct = all_punct - allowed_punct

    for idx, row in out.iterrows():
        if row.get("transcript_source") != "provided":
            continue
        invalid_char = _find_invalid_text_character(row.get("transcript", ""), disallowed_punct)
        if invalid_char is None:
            continue
        out.at[idx, "status"] = "dropped"
        out.at[idx, "drop_reason"] = f"invalid_transcript_character:{invalid_char}"
    return out


def _find_invalid_text_character(text: str, disallowed_punct: set[str]) -> str | None:
    """Return the first invalid character in text, or None when it passes validation."""
    if not isinstance(text, str):
        return None
    for ch in text:
        if ch == " ":
            continue
        if ch.isalnum():
            continue
        if ch in disallowed_punct:
            return ch
        if ch in string.punctuation:
            continue
        return ch
    return None


def ensure_stage_dir(output_dir: str | Path, stage_dir_name: str) -> Path:
    """Create and return the directory used for one pipeline stage."""
    path = Path(output_dir) / stage_dir_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_manifest(df: pd.DataFrame, directory: str | Path, filename: str = "manifest.csv") -> Path:
    """Write a manifest CSV into a stage directory and return its path."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    df.to_csv(path, index=False)
    return path


def copy_audio(audio_path: str | Path, destination: str | Path) -> Path:
    """Copy one audio file into the run workspace."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(audio_path, destination)
    return destination
