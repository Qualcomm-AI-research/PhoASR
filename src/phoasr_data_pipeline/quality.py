# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""WER scoring and filtering."""

from __future__ import annotations

import pandas as pd

from .backends.asr import transcribe_batch
from .backends.seq2seq import generate_text
from .config import PipelineConfig
from .io import ensure_object_columns
from .text import simple_wer
from .transcription import get_reference_transcript


def _reference_for_scoring(reference: str, config: PipelineConfig) -> str:
    """Return the reference in spoken form so WER matches the spoken-form verifier.

    The verifier ASR always emits spoken-form Vietnamese (e.g. "hai nghìn"),
    while provided transcripts may contain digits (e.g. "2000"). Comparing the
    two directly counts every number as an error. We normalize digit-bearing
    references through the num2word model *for scoring only* -- the stored
    transcript stays digit-form so the num2word stage can still build the
    digit->word mapping and finalize can project digits back into the output.
    """
    if not any(ch.isdigit() for ch in reference):
        return reference
    backend = config.get("num2word.model", {}) or {}
    if not backend or not backend.get("type"):
        return reference
    return generate_text(reference, backend)


def score_wer(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """Score WER using the verifier ASR against provided or predicted references."""
    out = ensure_object_columns(df.copy(), ["verification_transcript"])
    active_rows = out[out["status"] == "active"]
    if active_rows.empty:
        return out

    verifier = config.get("asr.verifier", {}) or {}
    if not verifier or not verifier.get("type"):
        return out

    audio_paths = active_rows["audio_path"].astype(str).tolist()
    sample_ids = active_rows["sample_id"].astype(str).tolist()
    verifications = transcribe_batch(audio_paths, verifier, sample_ids=sample_ids)

    for (idx, row), verification in zip(active_rows.iterrows(), verifications):
        out.at[idx, "verification_transcript"] = verification
        reference = _reference_for_scoring(get_reference_transcript(row), config)
        out.at[idx, "wer"] = simple_wer(reference, verification)
    return out


def apply_filter(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """Filter rows whose WER exceeds the configured threshold."""
    threshold = float(config.get("thresholds.wer", 0.05))
    out = ensure_object_columns(df.copy(), ["status", "drop_reason"])
    for idx, row in out.iterrows():
        if row["status"] != "active":
            continue
        wer_value = row["wer"]
        if pd.isna(wer_value):
            continue
        if float(wer_value) > threshold:
            out.at[idx, "status"] = "dropped"
            out.at[idx, "drop_reason"] = "wer_above_threshold"
    return out
