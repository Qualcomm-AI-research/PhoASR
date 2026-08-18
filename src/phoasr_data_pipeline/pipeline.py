# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Pipeline orchestration."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .alignment import align_samples
from .audio import get_audio_duration_seconds, prepare_audio_copy
from .config import PipelineConfig
from .constants import STAGE_DIRS, STAGES
from .formatting import apply_punctuation_and_capitalization
from .io import (
    ensure_stage_dir,
    filter_invalid_transcript_rows,
    load_manifest,
    prepare_run_manifest,
    validate_input_manifest,
    write_manifest,
)
from .normalization import apply_num2word
from .postprocess import finalize_dataset
from .quality import apply_filter, score_wer
from .transcription import populate_missing_transcripts


def run_pipeline(
    config_path: str,
    input_manifest: str,
    output_dir: str,
    start_stage: str | None = None,
    end_stage: str | None = None,
) -> Path:
    """Run the configured stage range and return the final manifest path."""
    config = PipelineConfig.load(config_path)
    stage_list = _resolve_stage_range(start_stage, end_stage)
    output_root = Path(output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    manifest_path = _resolve_initial_manifest(output_root, input_manifest, stage_list[0])
    for stage in stage_list:
        if not config.is_stage_enabled(stage):
            continue
        stage_dir = ensure_stage_dir(output_root, STAGE_DIRS[stage])
        df = load_manifest(manifest_path)
        if stage == "validate":
            next_df = _run_validate_stage(
                df, stage_dir, config, base_dir=Path(manifest_path).parent
            )
            manifest_name = "manifest.csv"
        elif stage == "transcribe":
            next_df = populate_missing_transcripts(df, config)
            manifest_name = "manifest.csv"
        elif stage == "score_wer":
            next_df = score_wer(df, config)
            manifest_name = "manifest.csv"
        elif stage == "filter":
            next_df = apply_filter(df, config)
            manifest_name = "manifest.csv"
        elif stage == "punc_cap":
            next_df = apply_punctuation_and_capitalization(df, config)
            manifest_name = "manifest.csv"
        elif stage == "num2word":
            next_df = apply_num2word(df, config, stage_dir)
            manifest_name = "manifest.csv"
        elif stage == "align":
            next_df = align_samples(df, config, stage_dir)
            manifest_name = "manifest.csv"
        elif stage == "finalize":
            next_df = finalize_dataset(df, stage_dir, config.repo_root)
            manifest_name = "final_manifest.csv"
        else:
            raise ValueError(f"Unknown stage: {stage}")
        manifest_path = write_manifest(next_df, stage_dir, filename=manifest_name)
    return manifest_path


def _resolve_stage_range(start_stage: str | None, end_stage: str | None) -> list[str]:
    """Return the inclusive stage slice requested by the caller."""
    start_index = 0 if start_stage is None else _stage_index(start_stage)
    end_index = len(STAGES) - 1 if end_stage is None else _stage_index(end_stage)
    if start_index > end_index:
        raise ValueError("start_stage must be before or equal to end_stage")
    return STAGES[start_index : end_index + 1]


def _stage_index(stage: str) -> int:
    """Return the index of a known stage or raise for unknown names."""
    if stage not in STAGES:
        raise ValueError(f"Unknown stage: {stage}")
    return STAGES.index(stage)


def _resolve_initial_manifest(output_root: Path, input_manifest: str, stage: str) -> Path:
    """Resolve the manifest to read when starting from a given stage."""
    if stage == "validate":
        return Path(input_manifest).resolve()
    previous_stage = STAGES[STAGES.index(stage) - 1]
    previous_dir = output_root / STAGE_DIRS[previous_stage]
    candidates = [previous_dir / "final_manifest.csv", previous_dir / "manifest.csv"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not resume from stage `{stage}` because no upstream manifest was found in {previous_dir}"
    )


def _run_validate_stage(
    df: pd.DataFrame,
    stage_dir: Path,
    config: PipelineConfig,
    base_dir: Path,
) -> pd.DataFrame:
    """Validate inputs, materialize stage audio, and initialize bookkeeping columns."""
    validated = validate_input_manifest(df, base_dir=base_dir)
    prepared = prepare_run_manifest(validated)
    prepared = filter_invalid_transcript_rows(
        prepared,
        config.get(
            "text.allowed_punctuations",
            ["(", ")", ":", ";", ",", ".", "!", "?", "'", '"', "/"],
        ),
    )
    audio_dir = stage_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    max_duration = float(config.get("audio.max_duration_sec", 30))
    for idx, row in prepared.iterrows():
        suffix = Path(row["audio_path"]).suffix or ".wav"
        destination = audio_dir / f"{row['sample_id']}{suffix}"
        prepared_path = prepare_audio_copy(row["audio_path"], destination, config)
        prepared.at[idx, "audio_path"] = str(prepared_path.resolve().relative_to(config.repo_root))
        duration = get_audio_duration_seconds(prepared_path)
        prepared.at[idx, "duration"] = duration
        if duration > max_duration:
            prepared.at[idx, "status"] = "dropped"
            prepared.at[idx, "drop_reason"] = "duration_above_max"
    return prepared
