# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Transcript generation stage logic."""

from __future__ import annotations

import pandas as pd

from .backends.asr import (
    chunkformer_batch_transcribe,
    load_chunkformer_model,
    load_parakeet_model,
    parakeet_batch_transcribe,
    transcribe,
)
from .config import PipelineConfig


def populate_missing_transcripts(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """Transcribe rows whose transcripts are marked as missing."""
    out = df.copy()
    backend = config.get("asr.primary", {}) or {}

    missing_indices: list[int] = []
    missing_audio_paths: list[str] = []
    missing_sample_ids: list[str] = []

    for idx, row in out.iterrows():
        if row["transcript_source"] == "provided":
            out.at[idx, "transcript_source"] = "provided"
            continue
        missing_indices.append(idx)
        missing_audio_paths.append(str(row["audio_path"]))
        missing_sample_ids.append(str(row["sample_id"]))

    if not missing_indices:
        return out

    if backend.get("type") == "chunkformer":
        model = load_chunkformer_model(backend)
        predictions: list[str] = []
        batch_size = int(backend.get("batch_size", 8))
        for start in range(0, len(missing_audio_paths), batch_size):
            batch_audio_paths = missing_audio_paths[start : start + batch_size]
            predictions.extend(chunkformer_batch_transcribe(model, batch_audio_paths, backend))
    elif backend.get("type") == "parakeet_nemo":
        model = load_parakeet_model(backend)
        predictions = []
        batch_size = int(backend.get("batch_size", 8))
        for start in range(0, len(missing_audio_paths), batch_size):
            batch_audio_paths = missing_audio_paths[start : start + batch_size]
            predictions.extend(parakeet_batch_transcribe(model, batch_audio_paths, backend))
    else:
        predictions = [
            transcribe(audio_path, backend, sample_id=sample_id)
            for audio_path, sample_id in zip(missing_audio_paths, missing_sample_ids)
        ]

    for column in ["transcript", "transcript_source"]:
        if column not in out.columns:
            out[column] = ""
        out[column] = out[column].astype("object")

    for idx, predicted in zip(missing_indices, predictions):
        out.at[idx, "transcript"] = predicted
        out.at[idx, "transcript_source"] = "predicted"
    return out


def get_reference_transcript(row: pd.Series) -> str:
    """Return the stripped transcript used as the WER reference for a row."""
    return str(row.get("transcript", "")).strip()
